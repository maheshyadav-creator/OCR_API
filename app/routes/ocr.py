from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from app.core.config import get_settings
from app.database.connection import get_db_connection
from app.schemas.ocr import (
    OCRBatchResponse,
    OCRHistoryItem,
    OCRJobResponse,
    OCRJobStatusResponse,
    OCRResponse,
)
from app.services.file_service import validate_uploaded_file
from app.services.storage_service import save_uploaded_file
from app.tasks.ocr_tasks import process_ocr_job


router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
)

settings = get_settings()


# ============================================================
# CREATE OCR JOBS
# ============================================================


@router.post(
    "",
    response_model=OCRBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ocr_jobs(
    files: list[UploadFile] = File(...),
    x_locust_run_id: str | None = Header(
        default=None,
        alias="X-Locust-Run-ID",
    ),
    x_locust_user_id: str | None = Header(
        default=None,
        alias="X-Locust-User-ID",
    ),
    x_locust_user_number: int | None = Header(
        default=None,
        alias="X-Locust-User-Number",
    ),
    x_locust_total_users: int | None = Header(
        default=None,
        alias="X-Locust-Total-Users",
    ),
    x_locust_pdf_number: int | None = Header(
        default=None,
        alias="X-Locust-PDF-Number",
    ),
    x_locust_pdfs_per_user: int | None = Header(
        default=None,
        alias="X-Locust-PDFs-Per-User",
    ),
) -> OCRBatchResponse:

    # --------------------------------------------------------
    # 1. Validate that at least one file was uploaded
    # --------------------------------------------------------

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    # --------------------------------------------------------
    # 2. Validate ALL files before creating any jobs
    # --------------------------------------------------------

    for file in files:
        try:
            validate_uploaded_file(file)

        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    # --------------------------------------------------------
    # 3. Calculate maximum upload size from application config
    # --------------------------------------------------------

    max_size_bytes = (
        settings.max_file_size_mb
        * 1024
        * 1024
    )

    jobs: list[OCRJobResponse] = []

    # --------------------------------------------------------
    # 4. Create one OCR job for each uploaded file
    # --------------------------------------------------------

    for file in files:

        # ----------------------------------------------------
        # Generate unique job ID FIRST
        # ----------------------------------------------------

        job_id = uuid4()

        filename = (
            file.filename
            or "unknown"
        )

        content_type = (
            file.content_type
            or "application/octet-stream"
        )

        file_path: str | None = None

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        try:
            file_path, _ = await save_uploaded_file(
                job_id=job_id,
                filename=filename,
                file=file,
                max_size_bytes=max_size_bytes,
            )

        except ValueError as exc:

            if str(exc) == "FILE_TOO_LARGE":
                raise HTTPException(
                    status_code=(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                    ),
                    detail=(
                        f"Uploaded file {filename} "
                        f"exceeds the maximum allowed size "
                        f"of {settings.max_file_size_mb} MB."
                    ),
                ) from exc

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Failed to save uploaded file "
                    f"{filename}: {exc}"
                ),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Failed to save uploaded file "
                    f"{filename}: {exc}"
                ),
            ) from exc

        # ----------------------------------------------------
        # Insert job into PostgreSQL
        # ----------------------------------------------------

        try:
            with get_db_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO ocr_jobs (
                            id,
                            filename,
                            content_type,
                            file_path,
                            status,
                            locust_run_id,
                            locust_user_id,
                            locust_user_number,
                            locust_total_users,
                            locust_pdf_number,
                            locust_pdfs_per_user
                        )
                        VALUES (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        RETURNING
                            id,
                            filename,
                            content_type,
                            status,
                            created_at
                        """,
                        (
                            job_id,
                            filename,
                            content_type,
                            file_path,
                            "queued",
                            x_locust_run_id,
                            x_locust_user_id,
                            x_locust_user_number,
                            x_locust_total_users,
                            x_locust_pdf_number,
                            x_locust_pdfs_per_user,
                        ),
                    )

                    row = cursor.fetchone()

                connection.commit()

        except Exception as exc:

            # ------------------------------------------------
            # PostgreSQL insert failed.
            # Remove the file because no valid job exists.
            # ------------------------------------------------

            if file_path is not None:
                try:
                    Path(file_path).unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Failed to create OCR job "
                    f"for {filename}: {exc}"
                ),
            ) from exc

        # ----------------------------------------------------
        # Locust logging
        # ----------------------------------------------------

        if x_locust_run_id:

            expected_jobs = None

            if (
                x_locust_total_users is not None
                and x_locust_pdfs_per_user is not None
            ):
                expected_jobs = (
                    x_locust_total_users
                    * x_locust_pdfs_per_user
                )

            print()
            print("=" * 70)
            print("LOCUST OCR JOB CREATED")

            print(
                f"RUN ID          : "
                f"{x_locust_run_id}"
            )

            print(
                f"USER            : "
                f"{x_locust_user_number or '?'}/"
                f"{x_locust_total_users or '?'}"
            )

            print(
                f"PDF             : "
                f"{x_locust_pdf_number or '?'}/"
                f"{x_locust_pdfs_per_user or '?'}"
            )

            print(
                f"EXPECTED JOBS   : "
                f"{expected_jobs or '?'}"
            )

            print(
                f"JOB ID          : "
                f"{job_id}"
            )

            print(
                f"FILENAME        : "
                f"{filename}"
            )

            print("=" * 70)
            print()

        # ----------------------------------------------------
        # Send ONLY job_id to Celery
        # ----------------------------------------------------

        try:
            process_ocr_job.delay(
                str(job_id)
            )

        except Exception as exc:

            # ------------------------------------------------
            # Celery enqueue failed.
            # Mark the PostgreSQL job as failed.
            # ------------------------------------------------

            try:
                with get_db_connection() as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            UPDATE ocr_jobs
                            SET
                                status = 'failed',
                                error_message = %s,
                                updated_at =
                                    CURRENT_TIMESTAMP
                            WHERE id = %s
                            """,
                            (
                                (
                                    "Failed to enqueue "
                                    f"Celery task: {exc}"
                                ),
                                job_id,
                            ),
                        )

                    connection.commit()

            except Exception:
                # The original Celery error is more useful
                # to the API caller than a secondary DB error.
                pass

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    f"Failed to enqueue OCR job "
                    f"{job_id}: {exc}"
                ),
            ) from exc

        # ----------------------------------------------------
        # Add job to response
        # ----------------------------------------------------

        jobs.append(
            OCRJobResponse(
                job_id=str(row[0]),
                filename=row[1],
                content_type=row[2],
                status=row[3],
                created_at=row[4],
            )
        )

    # --------------------------------------------------------
    # Return all created jobs
    # --------------------------------------------------------

    return OCRBatchResponse(
        jobs=jobs
    )


# ============================================================
# OCR HISTORY
# ============================================================


@router.get(
    "/history",
    response_model=list[OCRHistoryItem],
)
async def get_ocr_history() -> list[OCRHistoryItem]:

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    filename,
                    content_type,
                    extracted_text,
                    confidence,
                    result_json,
                    created_at
                FROM ocr_results
                ORDER BY created_at DESC
                """
            )

            rows = cursor.fetchall()

    return [
        OCRHistoryItem(
            id=row[0],
            filename=row[1],
            content_type=row[2],
            extracted_text=row[3],
            confidence=row[4],
            result_json=row[5],
            created_at=row[6],
        )
        for row in rows
    ]


# ============================================================
# OCR JOB STATUS
# ============================================================


@router.get(
    "/{job_id}",
    response_model=OCRJobStatusResponse,
)
async def get_ocr_job_status(
    job_id: UUID,
) -> OCRJobStatusResponse:

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
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
                WHERE id = %s
                """,
                (job_id,),
            )

            row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR job not found.",
        )

    return OCRJobStatusResponse(
        job_id=row[0],
        filename=row[1],
        content_type=row[2],
        status=row[3],
        attempt_count=row[4],
        error_message=row[5],
        created_at=row[6],
        started_at=row[7],
        completed_at=row[8],
    )


# ============================================================
# OCR RESULT
# ============================================================


@router.get(
    "/{job_id}/result",
    response_model=OCRResponse,
)
async def get_ocr_result(
    job_id: UUID,
) -> OCRResponse:

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    job_id,
                    filename,
                    content_type,
                    extracted_text,
                    confidence,
                    result_json,
                    created_at
                FROM ocr_results
                WHERE job_id = %s
                """,
                (job_id,),
            )

            row = cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR result not found.",
        )

    return OCRResponse(
        job_id=row[0],
        filename=row[1],
        content_type=row[2],
        extracted_text=row[3],
        confidence=row[4],
        result_json=row[5],
        created_at=row[6],
    )
