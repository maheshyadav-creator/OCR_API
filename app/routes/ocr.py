from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from psycopg2.extras import Json

from app.core.config import get_settings
from app.database.connection import get_db_connection
from app.schemas.ocr import OCRHistoryItem, OCRResponse
from app.services.ocr_service import perform_ocr


router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
)

settings = get_settings()


@router.post(
    "",
    response_model=OCRResponse,
    status_code=status.HTTP_201_CREATED,
)
async def process_ocr(
    file: UploadFile = File(...),
):
    """
    Upload an image and extract text using PaddleOCR.
    The OCR result is stored in PostgreSQL.
    """

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are supported.",
        )

    # Read uploaded file
    file_content = await file.read()

    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024

    if len(file_content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds the maximum allowed size "
                f"of {settings.max_file_size_mb} MB."
            ),
        )

    filename = file.filename or "unknown"
    content_type = file.content_type

    # Create temporary image file
    suffix = Path(filename).suffix or ".jpg"
    temp_path = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Perform OCR
        ocr_result = perform_ocr(temp_path)

        # Save OCR result to PostgreSQL
        insert_query = """
            INSERT INTO ocr_results (
                filename,
                content_type,
                extracted_text,
                confidence,
                result_json
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                id,
                filename,
                content_type,
                extracted_text,
                confidence,
                result_json,
                created_at;
        """

        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    insert_query,
                    (
                        filename,
                        content_type,
                        ocr_result["text"],
                        ocr_result["confidence"],
                        Json(ocr_result),
                    ),
                )

                row = cursor.fetchone()

            connection.commit()

        return {
            "id": row[0],
            "filename": row[1],
            "content_type": row[2],
            "extracted_text": row[3],
            "confidence": row[4],
            "result": row[5],
            "created_at": row[6],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(exc)}",
        ) from exc

    finally:
        # Always remove temporary image
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@router.get(
    "/history",
    response_model=list[OCRHistoryItem],
)
def get_ocr_history():
    """
    Return OCR processing history from PostgreSQL.
    """

    query = """
        SELECT
            id,
            filename,
            content_type,
            extracted_text,
            confidence,
            created_at
        FROM ocr_results
        ORDER BY created_at DESC;
    """

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "content_type": row[2],
                "extracted_text": row[3],
                "confidence": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve OCR history: {str(exc)}",
        ) from exc


@router.get(
    "/{ocr_id}",
    response_model=OCRResponse,
)
def get_ocr_result(
    ocr_id: int,
):
    """
    Return a single OCR result from PostgreSQL.
    """

    query = """
        SELECT
            id,
            filename,
            content_type,
            extracted_text,
            confidence,
            result_json,
            created_at
        FROM ocr_results
        WHERE id = %s;
    """

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (ocr_id,))
                row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="OCR result not found.",
            )

        return {
            "id": row[0],
            "filename": row[1],
            "content_type": row[2],
            "extracted_text": row[3],
            "confidence": row[4],
            "result": row[5],
            "created_at": row[6],
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve OCR result: {str(exc)}",
        ) from exc