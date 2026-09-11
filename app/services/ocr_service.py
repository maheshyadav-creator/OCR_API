from functools import lru_cache
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    settings = get_settings()

    print("Loading PaddleOCR...")

    return PaddleOCR(
        lang=settings.ocr_lang,
        device=settings.ocr_device,
        engine="paddle",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def make_json_serializable(value: Any) -> Any:

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def perform_ocr(image_path: str) -> dict[str, Any]:

    ocr = get_ocr_engine()

    results = ocr.predict(image_path)

    pages = []
    all_lines = []

    for result in results:

        result_data = make_json_serializable(
            result.json
        )

        if "res" in result_data:
            result_data = result_data["res"]

        texts = result_data.get(
            "rec_texts",
            [],
        )

        scores = result_data.get(
            "rec_scores",
            [],
        )

        lines = []

        for index, text in enumerate(texts):

            text = str(text).strip()

            if not text:
                continue

            try:
                score = float(scores[index])
            except (IndexError, TypeError, ValueError):
                score = 0.0

            line = {
                "text": text,
                "confidence": round(score, 6),
            }

            lines.append(line)
            all_lines.append(line)

        pages.append(
            {
                "text": "\n".join(
                    line["text"]
                    for line in lines
                ),
                "lines": lines,
                "raw": result_data,
            }
        )

    extracted_text = "\n".join(
        line["text"]
        for line in all_lines
    )

    confidence_values = [
        line["confidence"]
        for line in all_lines
    ]

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "text": extracted_text,
        "confidence": round(
            average_confidence,
            6,
        ),
        "pages": pages,
        "line_count": len(all_lines),
    }
