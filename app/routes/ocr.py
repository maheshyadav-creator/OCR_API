from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.core.config import get_settings
from app.database.connection import get_db_connection
from app.queue.redis_queue import enqueue_ocr_job
from app.schemas.ocr import (
    OCRBatchResponse,
    OCRHistoryItem,
    OCRJobResponse,
    OCRJobStatusResponse,
    OCRResponse,
)
from app.services.file_service import validate_uploaded_file
from app.services.storage_service import save_uploaded_file


router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
)

settings = get_settings()


# ============================================================
# Create OCR jobs
# ============================================================

@router.post(
    "",
    response_model=OCRBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_ocr(
    files: list[UploadFile] = File(...),
):
    """
    Accept multiple images or PDFs and create asynchronous
    OCR jobs for each uploaded file.
    """

    # --------------------------------------------------------
    # 1. Make sure at least one file was uploaded
    # --------------------------------------------------------

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    # --------------------------------------------------------
    # 2. Validate ALL file types before creating jobs
    # --------------------------------------------------------

    for file in files:
        validate_uploaded_file(file)

    # --------------------------------------------------------
    # 3. Calculate maximum upload size
    # --------------------------------------------------------

    max_size_bytes = (
        settings.max_file_size_mb * 1024 * 1024
    )

    created_jobs: list[OCRJobResponse] = []

    # --------------------------------------------------------
    # 4. Process every uploaded file independently
    # --------------------------------------------------------

    for file in files:

        job_id = uuid4()

        filename = (
            file.filename
            or "unknown"
        )

        content_type = (
            file.content_type
            or "application/octet-stream"
        )

        file_path = None

        try:

            # ------------------------------------------------
            # 5. Save this file
            # ------------------------------------------------

            file_path, _ = await save_uploaded_file(
                job_id=job_id,
                filename=filename,
                file=file,
                max_size_bytes=max_size_bytes,
            )

            # ------------------------------------------------
            # 6. Create PostgreSQL job
            # ------------------------------------------------

            insert_query = """
                INSERT INTO ocr_jobs (
                    id,
                    filename,
                    content_type,
                    file_path,
                    status
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING
                    id,
                    filename,
                    content_type,
                    status,
                    created_at;
            """

            with get_db_connection() as connection:

                with connection.cursor() as cursor:

                    cursor.execute(
                        insert_query,
                        (
                            str(job_id),
                            filename,
                            content_type,
                            file_path,
                            "queued",
                        ),
                    )

                    row = cursor.fetchone()

                connection.commit()

            # ------------------------------------------------
            # 7. Add this job to Redis
            # ------------------------------------------------

            enqueue_ocr_job(
                job_id=str(job_id),
                file_path=file_path,
                filename=filename,
                content_type=content_type,
            )

            # ------------------------------------------------
            # 8. Add this job to the batch response
            # ------------------------------------------------

            created_jobs.append(
                OCRJobResponse(
                    job_id=str(row[0]),
                    filename=row[1],
                    content_type=row[2],
                    status=row[3],
                    created_at=row[4],
                )
            )

        except ValueError as exc:

            # ------------------------------------------------
            # File size error
            # ------------------------------------------------

            if str(exc) == "FILE_TOO_LARGE":

                if file_path:
                    Path(file_path).unlink(
                        missing_ok=True
                    )

                raise HTTPException(
                    status_code=(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                    ),
                    detail=(
                        f"File '{filename}' exceeds "
                        f"the maximum allowed size "
                        f"of {settings.max_file_size_mb} MB."
                    ),
                ) from exc

            # Unknown ValueError
            raise

        except Exception as exc:

            # ------------------------------------------------
            # Log unexpected error
            # ------------------------------------------------

            import traceback

            print(
                "========== OCR JOB ERROR =========="
            )
            print(
                f"FILE: {filename}"
            )
            print(
                f"ERROR TYPE: {type(exc).__name__}"
            )
            print(
                f"ERROR MESSAGE: {exc}"
            )

            traceback.print_exc()

            print(
                "==================================="
            )

            # ------------------------------------------------
            # Remove file if it was saved
            # ------------------------------------------------

            if file_path:
                Path(file_path).unlink(
                    missing_ok=True
                )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Failed to create OCR job "
                    f"for file '{filename}'."
                ),
            ) from exc

    # --------------------------------------------------------
    # 9. Return all created jobs
    # --------------------------------------------------------

    return {
        "jobs": created_jobs
    }


# ============================================================
# OCR history
# ============================================================

@router.get(
    "/history",
    response_model=list[OCRHistoryItem],
)
def get_ocr_history():
    """
    Return completed OCR results.
    """

    query = """
        SELECT
            id,
            job_id,
            filename,
            content_type,
            extracted_text,
            confidence,
            created_at
        FROM ocr_results
        ORDER BY created_at DESC;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(query)

            rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "job_id": (
                str(row[1])
                if row[1]
                else None
            ),
            "filename": row[2],
            "content_type": row[3],
            "extracted_text": row[4],
            "confidence": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


# ============================================================
# Get OCR job status
# ============================================================

@router.get(
    "/{job_id}",
    response_model=OCRJobStatusResponse,
)
def get_ocr_job_status(
    job_id: UUID,
):
    """
    Return the current status of an OCR job.
    """

    query = """
        SELECT
            id,
            filename,
            content_type,
            status,
            attempt_count,
            error_message,
            created_at,
            started_at,
            completed_at
        FROM ocr_jobs
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (str(job_id),),
            )

            row = cursor.fetchone()

    if row is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR job not found.",
        )

    return {
        "job_id": str(row[0]),
        "filename": row[1],
        "content_type": row[2],
        "status": row[3],
        "attempt_count": row[4],
        "error_message": row[5],
        "created_at": row[6],
        "started_at": row[7],
        "completed_at": row[8],
    }


# ============================================================
# Get OCR result
# ============================================================

@router.get(
    "/{job_id}/result",
    response_model=OCRResponse,
)
def get_ocr_result(
    job_id: UUID,
):
    """
    Return the OCR result for a completed job.
    """

    query = """
        SELECT
            id,
            job_id,
            filename,
            content_type,
            extracted_text,
            confidence,
            result_json,
            created_at
        FROM ocr_results
        WHERE job_id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (str(job_id),),
            )

            row = cursor.fetchone()

    if row is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "OCR result not found. "
                "The job may still be processing."
            ),
        )

    return {
        "id": row[0],
        "job_id": (
            str(row[1])
            if row[1]
            else None
        ),
        "filename": row[2],
        "content_type": row[3],
        "extracted_text": row[4],
        "confidence": row[5],
        "result": row[6],
        "created_at": row[7],
    }