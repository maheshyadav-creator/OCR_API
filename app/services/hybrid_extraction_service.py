from __future__ import annotations

import gc
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pymupdf4llm
from PIL import Image

from app.services.ocr_service import (
    _extract_page_result,
    _ocr_numpy_image,
    get_ocr_engine,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Used ONLY when a completely scanned page must be sent
# to PaddleOCR.
PDF_OCR_DPI = int(
    os.getenv("OCR_PDF_DPI", "200")
)

MAX_RENDER_SIDE_PX = int(
    os.getenv(
        "OCR_MAX_RENDER_SIDE_PX",
        "2400",
    )
)

# Embedded images smaller than this are normally logos,
# icons, decorative objects, etc.
MIN_IMAGE_WIDTH = int(
    os.getenv(
        "OCR_MIXED_MIN_IMAGE_WIDTH",
        "150",
    )
)

MIN_IMAGE_HEIGHT = int(
    os.getenv(
        "OCR_MIXED_MIN_IMAGE_HEIGHT",
        "80",
    )
)

# An image occupying less than this percentage of the page
# is treated conservatively.
MIN_IMAGE_PAGE_AREA_RATIO = float(
    os.getenv(
        "OCR_MIXED_MIN_PAGE_AREA_RATIO",
        "0.01",
    )
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def _image_bytes_to_numpy(
    image_bytes: bytes,
) -> np.ndarray:

    with Image.open(
        io.BytesIO(image_bytes)
    ) as image:

        rgb_image = image.convert(
            "RGB"
        )

        return np.asarray(
            rgb_image
        ).copy()


def _extract_embedded_image(
    document: pymupdf.Document,
    xref: int,
) -> np.ndarray | None:

    try:

        image_info = document.extract_image(
            xref
        )

        image_bytes = image_info.get(
            "image"
        )

        if not image_bytes:
            return None

        width = int(
            image_info.get(
                "width",
                0,
            )
        )

        height = int(
            image_info.get(
                "height",
                0,
            )
        )

        if (
            width < MIN_IMAGE_WIDTH
            or height < MIN_IMAGE_HEIGHT
        ):
            return None

        return _image_bytes_to_numpy(
            image_bytes
        )

    except Exception as exc:

        print(
            f"Could not extract embedded image "
            f"xref={xref}: {exc}"
        )

        return None


# ============================================================
# IMAGE TEXT CHECK
# ============================================================

def _ocr_embedded_image(
    document: pymupdf.Document,
    xref: int,
    page_number: int,
) -> tuple[str, list[float]]:

    image = _extract_embedded_image(
        document,
        xref,
    )

    if image is None:
        return "", []

    try:

        ocr = get_ocr_engine()

        (
            _page_data,
            text,
            confidence_values,
        ) = _ocr_numpy_image(
            ocr,
            image,
            page_number,
        )

        return (
            text.strip(),
            confidence_values,
        )

    finally:

        image = None

        gc.collect()


# ============================================================
# SCANNED PAGE OCR
# ============================================================

def _render_scanned_page(
    page: pymupdf.Page,
) -> np.ndarray:

    rect = page.rect

    longest_side = max(
        rect.width,
        rect.height,
    )

    zoom = PDF_OCR_DPI / 72.0

    if longest_side > 0:

        zoom = min(
            zoom,
            MAX_RENDER_SIDE_PX / longest_side,
        )

    pixmap = None

    try:

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(
                zoom,
                zoom,
            ),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )

        return np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        ).copy()

    finally:

        pixmap = None

        gc.collect()


def _ocr_scanned_page(
    page: pymupdf.Page,
    page_number: int,
) -> tuple[str, list[float]]:

    image = None

    try:

        image = _render_scanned_page(
            page
        )

        ocr = get_ocr_engine()

        (
            _page_data,
            text,
            confidence_values,
        ) = _ocr_numpy_image(
            ocr,
            image,
            page_number,
        )

        return (
            text.strip(),
            confidence_values,
        )

    finally:

        image = None

        gc.collect()


# ============================================================
# PAGE NATIVE TEXT
# ============================================================

def _get_native_page_text(
    chunk: dict[str, Any],
) -> str:

    text = chunk.get(
        "text",
        "",
    )

    if text is None:
        return ""

    return str(text).strip()


# ============================================================
# EMBEDDED IMAGE ANALYSIS
# ============================================================

def _analyze_page_images(
    document: pymupdf.Document,
    page: pymupdf.Page,
    page_number: int,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[float],
]:

    images = page.get_images(
        full=True
    )

    image_details: list[
        dict[str, Any]
    ] = []

    image_texts: list[str] = []

    confidence_values: list[float] = []

    page_area = (
        abs(page.rect.width)
        * abs(page.rect.height)
    )

    for image_number, image in enumerate(
        images,
        start=1,
    ):

        xref = image[0]

        try:

            image_info = document.extract_image(
                xref
            )

            width = int(
                image_info.get(
                    "width",
                    0,
                )
            )

            height = int(
                image_info.get(
                    "height",
                    0,
                )
            )

            image_area = (
                width * height
            )

            area_ratio = (
                image_area / page_area
                if page_area > 0
                else 0.0
            )

            # --------------------------------------------
            # Conservative candidate filtering.
            #
            # Tiny images are skipped before OCR.
            # Larger images become candidates.
            # --------------------------------------------

            candidate = (
                width >= MIN_IMAGE_WIDTH
                and height >= MIN_IMAGE_HEIGHT
                and area_ratio
                >= MIN_IMAGE_PAGE_AREA_RATIO
            )

            detail = {
                "image": image_number,
                "xref": xref,
                "width": width,
                "height": height,
                "area_ratio": area_ratio,
                "candidate": candidate,
                "ocr_used": False,
                "text": "",
            }

            if not candidate:

                image_details.append(
                    detail
                )

                continue

            # --------------------------------------------
            # Candidate image.
            #
            # PaddleOCR determines whether this image
            # actually contains recognizable text.
            #
            # This means we don't blindly add every
            # image to the final result.
            # --------------------------------------------

            (
                image_text,
                image_confidences,
            ) = _ocr_embedded_image(
                document,
                xref,
                page_number,
            )

            if image_text:

                detail["ocr_used"] = True
                detail["text"] = image_text

                image_texts.append(
                    image_text
                )

                confidence_values.extend(
                    image_confidences
                )

            image_details.append(
                detail
            )

        except Exception as exc:

            image_details.append(
                {
                    "image": image_number,
                    "xref": xref,
                    "candidate": False,
                    "ocr_used": False,
                    "text": "",
                    "error": str(exc),
                }
            )

    return (
        image_details,
        image_texts,
        confidence_values,
    )


# ============================================================
# PDF HYBRID EXTRACTION
# ============================================================

def _extract_pdf(
    file_path: str,
) -> dict[str, Any]:

    pdf_path = Path(
        file_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # use_ocr=False means PyMuPDF4LLM performs ONLY native
    # extraction here.
    #
    # We supply PaddleOCR ourselves.
    # --------------------------------------------------------

    native_pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        use_ocr=False,
    )

    document = pymupdf.open(
        str(pdf_path)
    )

    pages: list[
        dict[str, Any]
    ] = []

    all_text: list[str] = []

    confidence_values: list[float] = []

    try:

        page_count = len(
            document
        )

        if len(native_pages) != page_count:

            raise RuntimeError(
                "PyMuPDF4LLM page count does not "
                "match the PDF page count."
            )

        for page_index in range(
            page_count
        ):

            page_number = (
                page_index + 1
            )

            page = document.load_page(
                page_index
            )

            chunk = native_pages[
                page_index
            ]

            native_text = (
                _get_native_page_text(
                    chunk
                )
            )

            # =================================================
            # CASE 1
            # Native text page
            # =================================================
            #
            # We do NOT render this page.
            #
            # We keep the native PDF text.
            #
            # Then we inspect embedded images separately
            # because this may be a mixed page.
            # =================================================

            if native_text:

                (
                    image_details,
                    image_texts,
                    image_confidences,
                ) = _analyze_page_images(
                    document,
                    page,
                    page_number,
                )

                confidence_values.extend(
                    image_confidences
                )

                merged_parts = [
                    native_text
                ]

                merged_parts.extend(
                    image_texts
                )

                merged_text = "\n".join(
                    part
                    for part in merged_parts
                    if part
                ).strip()

                pages.append(
                    {
                        "page": page_number,
                        "text": merged_text,
                        "native_text": native_text,
                        "ocr_text": "\n".join(
                            image_texts
                        ),
                        "source": (
                            "pymupdf4llm"
                            if not image_texts
                            else "mixed"
                        ),
                        "ocr_used": bool(
                            image_texts
                        ),
                        "images": image_details,
                    }
                )

                if merged_text:

                    all_text.append(
                        merged_text
                    )

                continue

            # =================================================
            # CASE 2
            # Scanned/image-only page
            # =================================================
            #
            # There is no useful native text.
            #
            # Now, and ONLY now, render the page for PaddleOCR.
            # =================================================

            (
                ocr_text,
                page_confidences,
            ) = _ocr_scanned_page(
                page,
                page_number,
            )

            confidence_values.extend(
                page_confidences
            )

            pages.append(
                {
                    "page": page_number,
                    "text": ocr_text,
                    "native_text": "",
                    "ocr_text": ocr_text,
                    "source": "paddleocr",
                    "ocr_used": True,
                    "images": [],
                }
            )

            if ocr_text:

                all_text.append(
                    ocr_text
                )

    finally:

        document.close()

        gc.collect()

    extracted_text = "\n\n".join(
        all_text
    )

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": average_confidence,
        "file_name": pdf_path.name,
        "extraction_method": "hybrid",
    }


# ============================================================
# IMAGE FILE
# ============================================================

def _extract_image_file(
    file_path: str,
) -> dict[str, Any]:

    path = Path(
        file_path
    )

    ocr = get_ocr_engine()

    results = []

    with Image.open(path) as image:

        rgb_image = image.convert(
            "RGB"
        )

        image_array = np.asarray(
            rgb_image
        ).copy()

    try:

        with_image_results = list(
            ocr.predict(
                image_array
            )
        )

        results = with_image_results

    finally:

        image_array = None

        gc.collect()

    pages: list[
        dict[str, Any]
    ] = []

    all_text: list[str] = []

    confidence_values: list[float] = []

    for page_number, result in enumerate(
        results,
        start=1,
    ):

        (
            page_data,
            page_text,
            page_confidences,
        ) = _extract_page_result(
            result,
            page_number,
        )

        pages.append(
            page_data
        )

        if page_text:

            all_text.append(
                page_text
            )

        confidence_values.extend(
            page_confidences
        )

    extracted_text = "\n\n".join(
        all_text
    )

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": average_confidence,
        "file_name": path.name,
        "extraction_method": "paddleocr",
    }


# ============================================================
# PUBLIC API
# ============================================================

def perform_hybrid_extraction(
    file_path: str,
) -> dict[str, Any]:

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Extraction file does not exist: "
            f"{file_path}"
        )

    if path.suffix.lower() == ".pdf":

        return _extract_pdf(
            file_path
        )

    return _extract_image_file(
        file_path
    )
PY