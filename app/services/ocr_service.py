from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from paddleocr import PaddleOCR

from app.core.config import get_settings


# ============================================================
# CONFIGURATION
# ============================================================
#
# Every value can be changed with an environment variable
# (for example in docker-compose.yml) without editing code.
#
# Defaults are the settings that were measured on the 7.5 GiB
# laptop: ~20 s/page and ~2 GB peak memory per OCR process.
# ============================================================

# PDF page rendering resolution.
PDF_DPI = int(os.getenv("OCR_PDF_DPI", "200"))

# Never render a page with a longer side than this many pixels.
# A4 at 200 DPI is 2339 px, so normal pages are unchanged;
# only huge pages (posters, A0 drawings) are scaled down.
MAX_RENDER_SIDE_PX = int(os.getenv("OCR_MAX_RENDER_SIDE_PX", "2400"))

# CPU threads used by Paddle inside ONE inference.
# 2 was the fastest on this machine (3+ was slower).
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "2"))

# Text detection runs on an image whose longest side is at most
# this many pixels. This is the biggest memory/speed lever:
# 1280 -> ~2 GB peak, 960 -> ~1.5 GB peak and faster.
# Check extracted text length before going below 1280.
OCR_DET_LIMIT_SIDE_LEN = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1280"))

# Number of text lines recognised at once.
OCR_REC_BATCH_SIZE = int(os.getenv("OCR_REC_BATCH_SIZE", "2"))

# 0 = no limit. Set e.g. 100 to reject very long PDFs.
MAX_PDF_PAGES = int(os.getenv("OCR_MAX_PDF_PAGES", "0"))

# Only ONE inference may run at a time on the shared engine.
OCR_ENGINE_LOCK = threading.Lock()


# ============================================================
# PADDLEOCR ENGINE  (exactly one per Python process)
# ============================================================
#
# This replaces @lru_cache. lru_cache does NOT stop two threads
# from both building an engine when they call it at the same
# moment on the first job. The lock below does.
# ============================================================

_ENGINE: PaddleOCR | None = None
_ENGINE_INIT_LOCK = threading.Lock()


def _create_engine() -> PaddleOCR:
    settings = get_settings()

    print("=" * 70)
    print("Loading PaddleOCR CPU engine (ONE per process)...")
    print(f"OCR language            : {settings.ocr_lang}")
    print(f"OCR device              : {settings.ocr_device}")
    print(f"OCR CPU threads         : {OCR_CPU_THREADS}")
    print(f"Detection side limit    : {OCR_DET_LIMIT_SIDE_LEN}px")
    print(f"Recognition batch size  : {OCR_REC_BATCH_SIZE}")
    print(f"PDF DPI / max side      : {PDF_DPI} / {MAX_RENDER_SIDE_PX}px")
    print("Orientation/unwarping   : disabled")
    print("MKL-DNN                 : False")
    print("=" * 70)

    return PaddleOCR(
        lang=settings.ocr_lang,
        device=settings.ocr_device,
        # MKL-DNN crashes on this Paddle build.
        enable_mkldnn=False,
        cpu_threads=OCR_CPU_THREADS,
        # Optional models: not loaded.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # Memory/speed control for the detection model.
        text_det_limit_type="max",
        text_det_limit_side_len=OCR_DET_LIMIT_SIDE_LEN,
        text_recognition_batch_size=OCR_REC_BATCH_SIZE,
    )


def get_ocr_engine() -> PaddleOCR:
    """
    Return the single shared PaddleOCR engine, creating it
    on the first call. Safe to call from many threads.
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
) -> tuple[dict[str, Any], str, list[float]]:

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

    data = result_data.get("res", result_data)

    if not isinstance(data, dict):
        data = {}

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    page_text_parts: list[str] = []
    confidence_values: list[float] = []

    for text, score in zip(texts, scores):

        text = str(text).strip()

        if not text:
            continue

        try:
            confidence = float(score)
        except (TypeError, ValueError):
            continue

        page_text_parts.append(text)
        confidence_values.append(confidence)

    page_text = "\n".join(page_text_parts)

    page_data = {
        "page": page_number,
        "text": page_text,
    }

    return page_data, page_text, confidence_values


# ============================================================
# OCR ONE NUMPY IMAGE
# ============================================================

def _ocr_numpy_image(
    ocr: PaddleOCR,
    image: np.ndarray,
    page_number: int,
) -> tuple[dict[str, Any], str, list[float]]:

    # list(...) makes sure the whole inference finishes while
    # the lock is held, even if predict() returned a generator.
    with OCR_ENGINE_LOCK:
        results = list(ocr.predict(image))

    for result in results:
        return _extract_page_result(result, page_number)

    return {"page": page_number, "text": ""}, "", []


# ============================================================
# PDF OCR
# ============================================================

def _perform_pdf_ocr(
    file_path: str,
    ocr: PaddleOCR,
) -> dict[str, Any]:

    pdf_path = Path(file_path)
    thread_name = threading.current_thread().name

    pages: list[dict[str, Any]] = []
    all_text: list[str] = []
    confidence_values: list[float] = []

    document = fitz.open(str(pdf_path))

    try:

        page_count = len(document)

        if MAX_PDF_PAGES and page_count > MAX_PDF_PAGES:
            raise ValueError(
                f"PDF has {page_count} pages; "
                f"the limit is {MAX_PDF_PAGES}."
            )

        print(
            f"[{thread_name}] PDF opened: {pdf_path.name} "
            f"({page_count} pages, {PDF_DPI} DPI)"
        )

        for page_index in range(page_count):

            page_number = page_index + 1
            started = time.perf_counter()

            page = None
            pixmap = None
            image_array = None

            try:

                page = document.load_page(page_index)

                # Normal pages use PDF_DPI. Very large pages are
                # scaled down so the image stays <= MAX_RENDER_SIDE_PX.
                rect = page.rect
                longest_side = max(rect.width, rect.height)

                zoom = PDF_DPI / 72.0

                if longest_side > 0:
                    zoom = min(zoom, MAX_RENDER_SIDE_PX / longest_side)

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom),
                    colorspace=fitz.csRGB,
                    alpha=False,
                )

                image_array = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )

                (
                    page_data,
                    page_text,
                    page_confidences,
                ) = _ocr_numpy_image(
                    ocr,
                    image_array,
                    page_number,
                )

                pages.append(page_data)

                if page_text:
                    all_text.append(page_text)

                confidence_values.extend(page_confidences)

                print(
                    f"[{thread_name}] {pdf_path.name} "
                    f"page {page_number}/{page_count} "
                    f"({pixmap.width}x{pixmap.height}) "
                    f"done in {time.perf_counter() - started:.1f}s"
                )

            finally:

                image_array = None
                pixmap = None
                page = None

                gc.collect()

    finally:

        document.close()
        gc.collect()

    extracted_text = "\n\n".join(all_text)

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": average_confidence,
        "file_name": pdf_path.name,
    }


# ============================================================
# IMAGE OCR
# ============================================================

def _perform_image_ocr(
    file_path: str,
    ocr: PaddleOCR,
) -> dict[str, Any]:

    print(f"Running OCR on image: {file_path}")

    with OCR_ENGINE_LOCK:
        results = list(ocr.predict(file_path))

    pages: list[dict[str, Any]] = []
    all_text: list[str] = []
    confidence_values: list[float] = []

    for page_number, result in enumerate(results, start=1):

        (
            page_data,
            page_text,
            page_confidences,
        ) = _extract_page_result(result, page_number)

        pages.append(page_data)

        if page_text:
            all_text.append(page_text)

        confidence_values.extend(page_confidences)

    extracted_text = "\n\n".join(all_text)

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": average_confidence,
        "file_name": Path(file_path).name,
    }


# ============================================================
# PUBLIC OCR FUNCTION
# ============================================================

def perform_ocr(
    file_path: str,
) -> dict[str, Any]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"OCR file does not exist: {file_path}"
        )

    ocr = get_ocr_engine()

    print(
        f"[{threading.current_thread().name}] "
        f"Running OCR on: {path.name}"
    )

    if path.suffix.lower() == ".pdf":
        return _perform_pdf_ocr(file_path, ocr)

    return _perform_image_ocr(file_path, ocr)