from __future__ import annotations

import socket
from pathlib import Path

from psycopg2.extras import Json

import pymupdf4llm

from app.celery_app import celery_app
from app.database.connection import (
    get_db_connection,
    initialize_connection_pool,
)

from app.services.ocr_service import paddleocr_ocr_function
from app.services.ocr_service import (
    paddleocr_ocr_function,
    perform_ocr,
)

MAX_RETRIES = 3

initialize_connection_pool()


# ============================================================
# DATABASE HELPERS
# ============================================================


def get_job(job_id: str):
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    filename,
                    content_type,
                    file_path,
                    status,
                    attempt_count,
                    locust_run_id,
                    locust_user_id,
                    locust_user_number,
                    locust_total_users,
                    locust_pdf_number,
                    locust_pdfs_per_user
                FROM ocr_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            return cursor.fetchone()


def mark_job_processing(
    job_id: str,
    worker_id: str,
) -> int | None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'processing',
                    attempt_count = attempt_count + 1,
                    worker_id = %s,
                    started_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE id = %s
                  AND status IN ('queued', 'retrying')
                  AND attempt_count < %s
                RETURNING attempt_count
                """,
                (
                    worker_id,
                    job_id,
                    MAX_RETRIES,
                ),
            )
            row = cursor.fetchone()

        connection.commit()

        if row is None:
            return None

        return int(row[0])


def mark_job_completed(job_id: str) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE id = %s
                """,
                (job_id,),
            )

        connection.commit()


def mark_job_retrying(
    job_id: str,
    error_message: str,
) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'retrying',
                    error_message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    error_message,
                    job_id,
                ),
            )

        connection.commit()


def mark_job_failed(
    job_id: str,
    error_message: str,
) -> None:
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'failed',
                    error_message = %s,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    error_message,
                    job_id,
                ),
            )

        connection.commit()


def save_ocr_result(
    job_id: str,
    filename: str,
    content_type: str,
    ocr_result: dict,
) -> None:
    extracted_text = ocr_result.get(
        "text",
        "",
    )

    confidence = float(
        ocr_result.get(
            "confidence",
            0.0,
        )
    )

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ocr_results (
                    filename,
                    content_type,
                    extracted_text,
                    confidence,
                    result_json,
                    job_id
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (job_id)
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    content_type = EXCLUDED.content_type,
                    extracted_text = EXCLUDED.extracted_text,
                    confidence = EXCLUDED.confidence,
                    result_json = EXCLUDED.result_json
                """,
                (
                    filename,
                    content_type,
                    extracted_text,
                    confidence,
                    Json(ocr_result),
                    job_id,
                ),
            )

        connection.commit()

    print(
        f"OCR result saved for {job_id}"
    )


# ============================================================
# FILE CLEANUP
# ============================================================


def delete_temporary_file(
    file_path: str,
) -> None:
    path = Path(file_path)

    try:
        path.unlink(
            missing_ok=True
        )

        print(
            f"Temporary file deleted: "
            f"{file_path}"
        )

    except Exception as exc:
        print(
            f"WARNING: Failed to delete "
            f"temporary file {file_path}: {exc}"
        )


# ============================================================
# PDF PROCESSING
# ============================================================


def _perform_pdf_ocr(
    file_path: str,
) -> dict:
    pages = pymupdf4llm.to_markdown(
        file_path,
        page_chunks=True,
        use_ocr=True,
        force_ocr=False,
        ocr_function=paddleocr_ocr_function,
        ocr_dpi=150,
        ocr_language="eng",
    )

    page_texts: list[str] = []

    for page in pages:
        text = page.get(
            "text",
            "",
        )

        if text:
            page_texts.append(
                text
            )

    extracted_text = "\n\n".join(
        page_texts
    )

    return {
        "text": extracted_text,
        "pages": pages,
        "page_count": len(pages),
        "confidence": 0.0,
        "file_name": Path(
            file_path
        ).name,
    }


# ============================================================
# CELERY TASK
# ============================================================


@celery_app.task(
    bind=True,
    name="ocr.process_job",
    max_retries=MAX_RETRIES - 1,
)
def process_ocr_job(
    self,
    job_id: str,
):
    worker_id = socket.gethostname()

    job = get_job(job_id)

    if job is None:
        print(
            f"CELERY OCR JOB NOT FOUND: {job_id}"
        )
        return

    (
        db_job_id,
        filename,
        content_type,
        file_path,
        status_value,
        existing_attempt_count,
        locust_run_id,
        locust_user_id,
        locust_user_number,
        locust_total_users,
        locust_pdf_number,
        locust_pdfs_per_user,
    ) = job

    expected_jobs = None

    if (
        locust_total_users is not None
        and locust_pdfs_per_user is not None
    ):
        expected_jobs = (
            locust_total_users
            * locust_pdfs_per_user
        )

    print()
    print("=" * 70)
    print("CELERY OCR JOB")

    print(
        f"RUN ID          : "
        f"{locust_run_id or 'NORMAL API REQUEST'}"
    )

    if locust_run_id:
        print(
            f"USER            : "
            f"{locust_user_number or '?'}/"
            f"{locust_total_users or '?'}"
        )

        print(
            f"PDF             : "
            f"{locust_pdf_number or '?'}/"
            f"{locust_pdfs_per_user or '?'}"
        )

        print(
            f"EXPECTED JOBS   : "
            f"{expected_jobs or '?'}"
        )

    print(
        f"JOB ID          : "
        f"{db_job_id}"
    )

    print(
        f"WORKER          : "
        f"{worker_id}"
    )

    print(
        f"FILE            : "
        f"{filename}"
    )

    print(
        f"CURRENT STATUS  : "
        f"{status_value}"
    )

    print(
        f"ATTEMPT         : "
        f"{existing_attempt_count + 1}/"
        f"{MAX_RETRIES}"
    )

    print("=" * 70)
    print()

    path = Path(file_path)

    if not path.exists():
        error_message = (
            f"Temporary file not found: "
            f"{file_path}"
        )

        print(
            f"ERROR: {error_message}"
        )

        mark_job_failed(
            job_id,
            error_message,
        )

        return

    attempt_number = mark_job_processing(
        job_id,
        worker_id,
    )

    if attempt_number is None:
        print(
            f"Job {job_id} was not claimed. "
            f"It may already be processing or completed."
        )

        return

    print(
        f"Job {job_id} claimed by worker "
        f"{worker_id}."
    )

    print(
        f"Attempt: "
        f"{attempt_number}/{MAX_RETRIES}"
    )

    try:
        print()
        print(
            f"Starting document processing "
            f"for {filename}"
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if path.suffix.lower() == ".pdf":
            print(
                f"PDF detected. Sending "
                f"{filename} through PyMuPDF4LLM."
            )

            if locust_run_id:
                print(
                    f"[LOCUST RUN {locust_run_id}] "
                    f"USER {locust_user_number}/"
                    f"{locust_total_users} "
                    f"PDF {locust_pdf_number}/"
                    f"{locust_pdfs_per_user} "
                    f"PyMuPDF4LLM STARTED"
                )

            ocr_result = _perform_pdf_ocr(
                file_path
            )

            if locust_run_id:
                print(
                    f"[LOCUST RUN {locust_run_id}] "
                    f"USER {locust_user_number}/"
                    f"{locust_total_users} "
                    f"PDF {locust_pdf_number}/"
                    f"{locust_pdfs_per_user} "
                    f"PyMuPDF4LLM COMPLETED"
                )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        else:
            print(
                f"Image detected. Sending "
                f"{filename} directly to PaddleOCR."
            )

            ocr_result = perform_ocr(
                file_path
            )

        print(
            f"Document processing completed "
            f"for {filename}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Save result FIRST
        # Then mark completed
        # Then delete temporary file
        # ----------------------------------------------------

        save_ocr_result(
            job_id=job_id,
            filename=filename,
            content_type=content_type,
            ocr_result=ocr_result,
        )

        mark_job_completed(
            job_id
        )

        print(
            f"Job {job_id} completed successfully."
        )

        if locust_run_id:
            print(
                f"[LOCUST RUN {locust_run_id}] "
                f"USER {locust_user_number}/"
                f"{locust_total_users} "
                f"PDF {locust_pdf_number}/"
                f"{locust_pdfs_per_user} "
                f"STATUS COMPLETED "
                f"JOB {job_id}"
            )

        delete_temporary_file(
            file_path
        )

    except Exception as exc:
        error_message = str(exc)

        print()
        print(
            f"ERROR processing job {job_id}: "
            f"{error_message}"
        )

        # ----------------------------------------------------
        # RETRY
        # ----------------------------------------------------

        if attempt_number < MAX_RETRIES:
            mark_job_retrying(
                job_id,
                error_message,
            )

            print(
                f"Job {job_id} marked for retry."
            )

            try:
                raise self.retry(
                    exc=exc,
                    countdown=5,
                )

            except self.MaxRetriesExceededError:
                mark_job_failed(
                    job_id,
                    error_message,
                )

                delete_temporary_file(
                    file_path
                )

                raise

        # ----------------------------------------------------
        # PERMANENT FAILURE
        # ----------------------------------------------------

        mark_job_failed(
            job_id,
            error_message,
        )

        print(
            f"Job {job_id} permanently failed "
            f"after {attempt_number} attempts."
        )

        if locust_run_id:
            print(
                f"[LOCUST RUN {locust_run_id}] "
                f"USER {locust_user_number}/"
                f"{locust_total_users} "
                f"PDF {locust_pdf_number}/"
                f"{locust_pdfs_per_user} "
                f"STATUS FAILED "
                f"JOB {job_id}"
            )

        delete_temporary_file(
            file_path
        )

        raise
