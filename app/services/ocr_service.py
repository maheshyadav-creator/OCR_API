from functools import lru_cache
from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    """
    Create and cache one PaddleOCR engine.

    The engine is created only once per worker process.
    """

    settings = get_settings()

    print("Loading PaddleOCR 3.7 CPU engine...")

    return PaddleOCR(
        lang=settings.ocr_lang,
        device=settings.ocr_device,
        enable_mkldnn=False,
        cpu_threads=4,
    )


def perform_ocr(file_path: str) -> dict[str, Any]:
    """
    Run PaddleOCR on an image or PDF file.
    """

    ocr = get_ocr_engine()

    print(f"Running OCR on: {file_path}")

    results = ocr.predict(file_path)

    pages: list[dict[str, Any]] = []
    all_text: list[str] = []
    confidence_values: list[float] = []

    for page_number, result in enumerate(
        results,
        start=1,
    ):
        result_data = result.json

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

        for text, score in zip(texts, scores):

            text = str(text).strip()
            confidence = float(score)

            if not text:
                continue

            page_text_parts.append(text)
            confidence_values.append(confidence)

        page_text = "\n".join(
            page_text_parts
        )

        pages.append(
            {
                "page": page_number,
                "text": page_text,
                "raw": result_data,
            }
        )

        if page_text:
            all_text.append(page_text)

    extracted_text = "\n\n".join(
        all_text
    )

    if confidence_values:
        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        average_confidence = 0.0

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": average_confidence,
        "file_name": Path(file_path).name,
    }