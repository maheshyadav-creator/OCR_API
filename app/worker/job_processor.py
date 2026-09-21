from psycopg2.extras import Json

from app.database.connection import get_db_connection
from app.services.ocr_service import perform_ocr

from app.worker.redis_consumer import (
    OCR_CONSUMER_GROUP,
    OCR_STREAM_NAME,
    WORKER_ID,
    redis_client,
)


MAX_RETRIES = 3

# A processing job is considered stale only after this amount
# of time. This protects long-running OCR jobs from being
# accidentally processed twice.
STALE_PROCESSING_MINUTES = 15


# ============================================================
# PostgreSQL job state
# ============================================================

def mark_job_processing(job_id: str) -> bool:
    """
    Mark a job as processing and increment its attempt count.

    A job can be processed when:

    1. It is newly queued.
    2. It was previously processing but became stale.

    Completed and permanently failed jobs are never processed
    again.
    """

    query = """
        UPDATE ocr_jobs
        SET
            status = 'processing',
            attempt_count = attempt_count + 1,
            worker_id = %s,
            started_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP,
            error_message = NULL
        WHERE id = %s
          AND attempt_count < %s
          AND (
                status = 'queued'
                OR (
                    status = 'processing'
                    AND started_at IS NOT NULL
                    AND started_at <
                        CURRENT_TIMESTAMP
                        - (%s * INTERVAL '1 minute')
                )
          )
        RETURNING attempt_count;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    WORKER_ID,
                    job_id,
                    MAX_RETRIES,
                    STALE_PROCESSING_MINUTES,
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    if row is None:
        return False

    print(
        f"Job {job_id} marked as processing."
    )

    print(
        f"Attempt: {row[0]}/{MAX_RETRIES}"
    )

    return True


def get_job_status(job_id: str) -> str | None:
    """
    Get the current status of a job.

    Returns None when the job does not exist.
    """

    query = """
        SELECT status
        FROM ocr_jobs
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (job_id,),
            )

            row = cursor.fetchone()

    if row is None:
        return None

    return str(row[0])


def mark_job_completed(job_id: str) -> None:
    """
    Mark the OCR job as successfully completed.
    """

    query = """
        UPDATE ocr_jobs
        SET
            status = 'completed',
            completed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (job_id,),
            )

        connection.commit()


def mark_job_failed(
    job_id: str,
    error_message: str,
) -> None:
    """
    Mark the job as permanently failed.
    """

    query = """
        UPDATE ocr_jobs
        SET
            status = 'failed',
            error_message = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    error_message,
                    job_id,
                ),
            )

        connection.commit()


def mark_job_for_retry(
    job_id: str,
    error_message: str,
) -> None:
    """
    Put a failed job back into queued state.
    """

    query = """
        UPDATE ocr_jobs
        SET
            status = 'queued',
            error_message = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (
                    error_message,
                    job_id,
                ),
            )

        connection.commit()


# ============================================================
# Get attempt count
# ============================================================

def get_attempt_count(job_id: str) -> int:
    """
    Get the number of attempts already made for this job.
    """

    query = """
        SELECT attempt_count
        FROM ocr_jobs
        WHERE id = %s;
    """

    with get_db_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                query,
                (job_id,),
            )

            row = cursor.fetchone()

    if row is None:
        raise ValueError(
            f"OCR job not found: {job_id}"
        )

    return int(row[0])


# ============================================================
# Save OCR result
# ============================================================

def save_ocr_result(
    job_id: str,
    filename: str,
    content_type: str,
    ocr_result: dict,
) -> None:
    """
    Save the OCR output into PostgreSQL.

    job_id is unique, so retrying the same job does not
    create duplicate OCR result rows.
    """

    query = """
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
            result_json = EXCLUDED.result_json,
            created_at = CURRENT_TIMESTAMP;
    """

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
                query,
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


# ============================================================
# Process one OCR message
# ============================================================

def process_message(
    message_id: str,
    data: dict,
) -> None:
    """
    Process one OCR job from Redis.
    """

    job_id = data["job_id"]
    file_path = data["file_path"]

    print()
    print("=" * 60)
    print(
        f"Processing OCR job: {job_id}"
    )
    print(
        f"Redis message ID: {message_id}"
    )
    print(
        f"File: {file_path}"
    )
    print(
        f"Worker: {WORKER_ID}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Check and mark job as processing
    # --------------------------------------------------------

    should_process = mark_job_processing(
        job_id
    )

    if not should_process:

        status = get_job_status(
            job_id
        )

        print(
            f"Job {job_id} will not be processed."
        )

        print(
            f"Current database status: {status}"
        )

        redis_client.xack(
            OCR_STREAM_NAME,
            OCR_CONSUMER_GROUP,
            message_id,
        )

        print(
            f"Stale Redis message acknowledged: "
            f"{message_id}"
        )

        return

    # --------------------------------------------------------
    # Step 2: Run OCR
    # --------------------------------------------------------

    print(
        f"Starting PaddleOCR for job {job_id}..."
    )

    ocr_result = perform_ocr(
        file_path
    )

    print(
        f"PaddleOCR completed for job {job_id}."
    )

    # --------------------------------------------------------
    # Step 3: Save result
    # --------------------------------------------------------

    filename = data.get(
        "filename",
        file_path,
    )

    content_type = data.get(
        "content_type",
        "application/octet-stream",
    )

    save_ocr_result(
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        ocr_result=ocr_result,
    )

    print(
        f"OCR result saved for job {job_id}."
    )

    # --------------------------------------------------------
    # Step 4: Mark completed
    # --------------------------------------------------------

    mark_job_completed(
        job_id
    )

    print(
        f"Job {job_id} marked as completed."
    )

    # --------------------------------------------------------
    # Step 5: Acknowledge Redis message
    # --------------------------------------------------------

    redis_client.xack(
        OCR_STREAM_NAME,
        OCR_CONSUMER_GROUP,
        message_id,
    )

    print(
        f"Redis message acknowledged: "
        f"{message_id}"
    )

    print(
        f"Job {job_id} completed successfully."
    )

    print("=" * 60)
    print()