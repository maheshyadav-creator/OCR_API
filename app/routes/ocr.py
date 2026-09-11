import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.connection import get_db
from app.database.models import OCRResult
from app.schemas.ocr import OCRHistoryItem, OCRResponse
from app.services.ocr_service import perform_ocr


router = APIRouter(
    prefix="/api/v1/ocr",
    tags=["OCR"],
)

settings = get_settings()


ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}


@router.post(
    "",
    response_model=OCRResponse,
    status_code=status.HTTP_201_CREATED,
)
def process_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an image and run PaddleOCR.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    content_type = file.content_type or ""

    suffix = Path(file.filename).suffix.lower()

    if content_type in ALLOWED_CONTENT_TYPES:
        suffix = ALLOWED_CONTENT_TYPES[content_type]

    elif suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
    }:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image files are supported.",
        )

    max_size = settings.max_file_size_mb * 1024 * 1024

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        total_size = 0

        while True:
            chunk = file.file.read(1024 * 1024)

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > max_size:
                temp_path = temp_file.name
                temp_file.close()

                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File is too large. Maximum size is "
                        f"{settings.max_file_size_mb} MB."
                    ),
                )

            temp_file.write(chunk)

        temp_path = temp_file.name

    try:
        ocr_result = perform_ocr(temp_path)

        database_record = OCRResult(
            filename=file.filename,
            content_type=content_type or "application/octet-stream",
            extracted_text=ocr_result["text"],
            confidence=ocr_result["confidence"],
            result_json=ocr_result,
        )

        db.add(database_record)
        db.commit()
        db.refresh(database_record)

        return {
            "id": database_record.id,
            "filename": database_record.filename,
            "content_type": database_record.content_type,
            "extracted_text": database_record.extracted_text,
            "confidence": database_record.confidence,
            "result": database_record.result_json,
            "created_at": database_record.created_at,
        }

    except HTTPException:
        raise

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(exc)}",
        ) from exc

    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


@router.get(
    "/history",
    response_model=list[OCRHistoryItem],
)
def get_ocr_history(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    """
    Return previous OCR requests.
    """

    limit = min(max(limit, 1), 100)

    query = (
        select(OCRResult)
        .order_by(OCRResult.created_at.desc())
        .limit(limit)
    )

    records = db.scalars(query).all()

    return records


@router.get(
    "/{ocr_id}",
    response_model=OCRResponse,
)
def get_ocr_result(
    ocr_id: int,
    db: Session = Depends(get_db),
):
    """
    Get one OCR result by ID.
    """

    record = db.get(OCRResult, ocr_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR result not found.",
        )

    return {
        "id": record.id,
        "filename": record.filename,
        "content_type": record.content_type,
        "extracted_text": record.extracted_text,
        "confidence": record.confidence,
        "result": record.result_json,
        "created_at": record.created_at,
    }
