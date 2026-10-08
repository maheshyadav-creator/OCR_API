from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from pymupdf4llm.ocr.exec_ocr_interface import exec_ocr_full

from app.core.config import get_settings


# ============================================================
# CONFIGURATION
# ============================================================
#
# PaddleOCR is now responsible only for OCR inference.
#
# PDF rendering / PDF page decisions are NOT handled here.
#
# PDF processing will be handled by:
#
#     PyMuPDF4LLM
#            ↓
#     PyMuPDF4LLM OCR decision
#            ↓
#     PaddleOCR adapter
#            ↓
#     PaddleOCR 3.7.0
#
# This keeps PDF orchestration outside this service.
# ============================================================

OCR_CPU_THREADS = int(
    os.getenv("OCR_CPU_THREADS", "2")
)

OCR_DET_LIMIT_SIDE_LEN = int(
    os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1280")
)

OCR_REC_BATCH_SIZE = int(
    os.getenv("OCR_REC_BATCH_SIZE", "2")
)


# ============================================================
# SHARED PADDLEOCR ENGINE
# ============================================================
#
# Exactly ONE PaddleOCR engine per Python process.
#
# The engine is created lazily on the first request.
#
# _ENGINE_INIT_LOCK protects engine creation.
# OCR_ENGINE_LOCK protects inference on the shared engine.
# ============================================================

_ENGINE: PaddleOCR | None = None

_ENGINE_INIT_LOCK = threading.Lock()

OCR_ENGINE_LOCK = threading.Lock()


def _create_engine() -> PaddleOCR:
    """
    Create the single PaddleOCR 3.7.0 engine used by this
    Python process.
    """

    settings = get_settings()

    print("=" * 70)
    print("Loading PaddleOCR CPU engine (ONE per process)...")
    print(f"OCR language            : {settings.ocr_lang}")
    print(f"OCR device              : {settings.ocr_device}")
    print(f"OCR CPU threads         : {OCR_CPU_THREADS}")
    print(
        f"Detection side limit    : "
        f"{OCR_DET_LIMIT_SIDE_LEN}px"
    )
    print(
        f"Recognition batch size  : "
        f"{OCR_REC_BATCH_SIZE}"
    )
    print("Orientation/unwarping   : disabled")
    print("MKL-DNN                 : False")
    print("=" * 70)

    return PaddleOCR(
        lang=settings.ocr_lang,
        device=settings.ocr_device,

        # MKL-DNN is disabled for this project.
        enable_mkldnn=False,

        # CPU inference tuning.
        cpu_threads=OCR_CPU_THREADS,

        # Do not load unnecessary document models.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # Detection memory/speed control.
        text_det_limit_type="max",
        text_det_limit_side_len=OCR_DET_LIMIT_SIDE_LEN,

        # Recognition batch size.
        text_recognition_batch_size=OCR_REC_BATCH_SIZE,
    )


def get_ocr_engine() -> PaddleOCR:
    """
    Return the single shared PaddleOCR engine.

    The engine is created only once per Python process.

    Safe for multiple callers because engine initialization
    is protected by _ENGINE_INIT_LOCK.
    """

    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_INIT_LOCK:

        if _ENGINE is None:
            _ENGINE = _create_engine()

    return _ENGINE


# ============================================================
# RESULT EXTRACTION
# ============================================================

def _extract_page_result(
    result: Any,
    page_number: int,
) -> tuple[
    dict[str, Any],
    str,
    list[float],
]:
    """
    Convert one PaddleOCR 3.7.0 result into the application's
    simple page-level representation.
    """

    try:
        result_data = result.json
    except Exception:
        result_data = {}

    if callable(result_data):
        try:
            result_data = result_data()
        except Exception:
            result_data = {}

    if not isinstance(result_data, dict):
        result_data = {}

    data = result_data.get(
        "res",
        result_data,
    )

    if not isinstance(data, dict):
        data = {}

    texts = data.get(
        "rec_texts",
        [],
    )

    scores = data.get(
        "rec_scores",
        [],
    )

    page_text_parts: list[str] = []

    confidence_values: list[float] = []

    for text, score in zip(
        texts,
        scores,
    ):

        text = str(text).strip()

        if not text:
            continue

        try:
            confidence = float(score)
        except (
            TypeError,
            ValueError,
        ):
            continue

        page_text_parts.append(text)

        confidence_values.append(
            confidence
        )

    page_text = "\n".join(
        page_text_parts
    )

    page_data = {
        "page": page_number,
        "text": page_text,
    }

    return (
        page_data,
        page_text,
        confidence_values,
    )


# ============================================================
# OCR ONE NUMPY IMAGE
# ============================================================

def _ocr_numpy_image(
    ocr: PaddleOCR,
    image: np.ndarray,
    page_number: int,
) -> tuple[
    dict[str, Any],
    str,
    list[float],
]:
    """
    Run PaddleOCR on an already-rendered image.

    This helper is intentionally kept because the
    PyMuPDF4LLM adapter can use the same shared PaddleOCR
    engine.

    It does NOT know anything about PDFs or decide whether
    OCR is required.
    """

    with OCR_ENGINE_LOCK:

        results = list(
            ocr.predict(image)
        )

    for result in results:

        return _extract_page_result(
            result,
            page_number,
        )

    return (
        {
            "page": page_number,
            "text": "",
        },
        "",
        [],
    )


# ============================================================
# IMAGE OCR
# ============================================================

def _perform_image_ocr(
    file_path: str,
    ocr: PaddleOCR,
) -> dict[str, Any]:
    """
    Run PaddleOCR directly on an image file.

    This function is for standalone image uploads.

    PDF files DO NOT come through this function.
    """

    print(
        f"Running PaddleOCR on image: "
        f"{file_path}"
    )

    with OCR_ENGINE_LOCK:

        results = list(
            ocr.predict(file_path)
        )

    pages: list[
        dict[str, Any]
    ] = []

    all_text: list[str] = []

    confidence_values: list[
        float
    ] = []

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
        "file_name": Path(
            file_path
        ).name,
    }


# ============================================================
# PUBLIC OCR FUNCTION
# ============================================================

def perform_ocr(
    file_path: str,
) -> dict[str, Any]:
    """
    Public OCR entry point.

    IMPORTANT:

    PDF processing is intentionally NOT implemented here.

    PyMuPDF4LLM will become the PDF processing pipeline.

    Therefore:

        PDF
         ↓
        PyMuPDF4LLM
         ↓
        PyMuPDF4LLM decides whether OCR is needed
         ↓
        PaddleOCR adapter
         ↓
        shared PaddleOCR engine

    Standalone image files can still use PaddleOCR directly.
    """

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"OCR file does not exist: "
            f"{file_path}"
        )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------
    #
    # Do NOT silently fall back to the old PDF→pixels→OCR
    # implementation.
    #
    # This makes accidental use of the old architecture
    # impossible.
    # --------------------------------------------------------

    if path.suffix.lower() == ".pdf":

        raise RuntimeError(
            "Direct PDF OCR through ocr_service.py "
            "has been removed. "
            "PDFs must be processed through "
            "PyMuPDF4LLM with the PaddleOCR adapter."
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    ocr = get_ocr_engine()

    print(
        f"[{threading.current_thread().name}] "
        f"Running PaddleOCR on: "
        f"{path.name}"
    )

    try:

        return _perform_image_ocr(
            file_path,
            ocr,
        )

    finally:

        gc.collect()


# ============================================================
# PYMUPDF4LLM CUSTOM OCR CALLBACK
# ============================================================
#
# PyMuPDF4LLM remains responsible for:
#
#     - deciding whether OCR is required
#     - rendering the PDF page
#     - coordinate handling
#     - OCR text insertion
#     - continuing PDF extraction
#
# Our application is responsible only for:
#
#     - running PaddleOCR 3.7.0
#     - converting PaddleOCR results into the format
#       expected by PyMuPDF4LLM
#
# This does NOT use PyMuPDF4LLM's built-in RapidOCR backend.
# ============================================================


def _convert_paddle_box_to_polygon(
    box: Any,
) -> list[list[float]]:
    """
    Convert PaddleOCR 3.7.0 rec_boxes format:

        [x1, y1, x2, y2]

    into the polygon format expected by
    PyMuPDF4LLM:

        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
    """

    if not isinstance(
        box,
        (list, tuple),
    ):
        raise ValueError(
            f"Unexpected PaddleOCR box type: {type(box)}"
        )

    if len(box) != 4:
        raise ValueError(
            f"Expected [x1, y1, x2, y2], got: {box}"
        )

    x1, y1, x2, y2 = box

    return [
        [float(x1), float(y1)],
        [float(x2), float(y1)],
        [float(x2), float(y2)],
        [float(x1), float(y2)],
    ]


def paddleocr_full_ocr(
    image: np.ndarray,
) -> list[tuple[Any, str, float]]:
    """
    Low-level OCR callback used by PyMuPDF4LLM.

    PyMuPDF4LLM supplies the rendered image.

    Our PaddleOCR 3.7.0 engine performs the actual OCR.

    Returns:

        (polygon, text, confidence)

    for each detected text region.
    """

    ocr = get_ocr_engine()

    # Exactly one shared PaddleOCR engine exists per process.
    #
    # Serialize inference because multiple callers may reach
    # this callback while using the same PaddleOCR instance.
    with OCR_ENGINE_LOCK:

        results = list(
            ocr.predict(image)
        )

    output: list[
        tuple[Any, str, float]
    ] = []

    for result in results:

        try:
            result_data = result.json
        except Exception:
            continue

        if callable(result_data):

            try:
                result_data = result_data()
            except Exception:
                continue

        if not isinstance(
            result_data,
            dict,
        ):
            continue

        data = result_data.get(
            "res",
            result_data,
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        boxes = data.get(
            "rec_boxes",
            [],
        )

        texts = data.get(
            "rec_texts",
            [],
        )

        scores = data.get(
            "rec_scores",
            [],
        )

        if not isinstance(
            boxes,
            list,
        ):
            continue

        if not isinstance(
            texts,
            list,
        ):
            continue

        if not isinstance(
            scores,
            list,
        ):
            continue

        count = min(
            len(boxes),
            len(texts),
            len(scores),
        )

        for index in range(count):

            try:
                polygon = (
                    _convert_paddle_box_to_polygon(
                        boxes[index]
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            text = str(
                texts[index]
            ).strip()

            if not text:
                continue

            try:
                score = float(
                    scores[index]
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            output.append(
                (
                    polygon,
                    text,
                    score,
                )
            )

    return output


def paddleocr_ocr_function(
    page,
    dpi: int = 150,
    language: str | None = None,
    keep_ocr_text: bool = False,
) -> None:
    """
    PyMuPDF4LLM-compatible OCR callback.

    PyMuPDF4LLM calls this function only when its own OCR
    decision determines that OCR is required.

    PyMuPDF4LLM continues to handle the PDF orchestration.

    This function simply connects PyMuPDF4LLM's generic OCR
    interface to our custom PaddleOCR 3.7.0 implementation.
    """

    exec_ocr_full(
        page,
        paddleocr_full_ocr,
        dpi=dpi,
        language=language,
        keep_ocr_text=keep_ocr_text,
    )