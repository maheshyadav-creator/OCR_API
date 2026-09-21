from psycopg2 import OperationalError
from psycopg2.errors import (
    DeadlockDetected,
    SerializationFailure,
)

import redis

from app.worker.job_processor import (
    get_attempt_count,
    mark_job_failed,
    mark_job_for_retry,
)
from app.worker.redis_consumer import (
    OCR_CONSUMER_GROUP,
    OCR_STREAM_NAME,
    redis_client,
)


# Maximum number of total OCR processing attempts.
#
# Attempt 1 → initial processing
# Attempt 2 → first retry
# Attempt 3 → second retry
#
# After attempt 3, the job is permanently failed.
MAX_RETRIES = 3


def is_temporary_failure(exc: Exception) -> bool:
    """
    Determine whether an exception represents a temporary
    failure that may succeed if the job is attempted again.

    Only explicitly known temporary failures are retried.

    Unknown exceptions are treated as permanent failures
    so that programming errors or invalid input do not cause
    unnecessary repeated OCR attempts.
    """

    temporary_exceptions = (
        # Generic temporary connection/timeout failures.
        ConnectionError,
        TimeoutError,

        # PostgreSQL connection-level failures.
        OperationalError,

        # PostgreSQL concurrency failures that may succeed
        # when attempted again.
        DeadlockDetected,
        SerializationFailure,

        # Redis connectivity failures.
        redis.exceptions.ConnectionError,
        redis.exceptions.TimeoutError,
    )

    return isinstance(exc, temporary_exceptions)


def handle_failed_message(
    message_id: str,
    data: dict,
    exc: Exception,
) -> None:
    """
    Handle an OCR processing failure.

    The failure is first classified as temporary or permanent.

    Temporary failure:
        Retry the job until MAX_RETRIES is reached.

    Permanent failure:
        Mark the job as failed immediately.

    Unknown exceptions are treated as permanent failures.
    """

    job_id = data.get(
        "job_id",
        "unknown",
    )

    error_message = str(exc)

    print()
    print("=" * 60)
    print(
        f"OCR processing failed for job {job_id}"
    )
    print(
        f"Exception type: {type(exc).__name__}"
    )
    print(
        f"Error: {error_message}"
    )

    # --------------------------------------------------------
    # Step 1: Classify the failure
    # --------------------------------------------------------

    temporary_failure = is_temporary_failure(exc)

    if temporary_failure:
        print(
            f"Failure classification: TEMPORARY"
        )
    else:
        print(
            f"Failure classification: PERMANENT"
        )

    # --------------------------------------------------------
    # Step 2: Permanent failure
    # --------------------------------------------------------
    #
    # Permanent failures are NOT retried.
    #
    # Example:
    #
    # FileNotFoundError
    # invalid input
    # unsupported processing error
    #
    # These failures will not normally be fixed by running
    # PaddleOCR again on the exact same input.
    # --------------------------------------------------------

    if not temporary_failure:

        print(
            f"Job {job_id} will NOT be retried."
        )

        mark_job_failed(
            job_id,
            error_message,
        )

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
            f"Job {job_id} marked as permanently failed."
        )

        print("=" * 60)
        print()

        return

    # --------------------------------------------------------
    # Step 3: Get current attempt count
    # --------------------------------------------------------

    try:

        attempt_count = get_attempt_count(
            job_id
        )

    except Exception as attempt_error:

        # We cannot safely determine the attempt count.
        #
        # Do NOT assume the job has reached MAX_RETRIES,
        # because the database itself may simply be temporarily
        # unavailable.
        #
        # Leaving the original Redis message pending allows
        # recovery to handle it later.
        print(
            "Unable to determine attempt count."
        )

        print(
            f"Attempt-count error: "
            f"{attempt_error}"
        )

        print(
            "Leaving Redis message pending "
            "for later recovery."
        )

        print("=" * 60)
        print()

        return

    # --------------------------------------------------------
    # Step 4: Retry temporary failure
    # --------------------------------------------------------

    if attempt_count < MAX_RETRIES:

        print(
            f"Retrying job {job_id} "
            f"({attempt_count}/{MAX_RETRIES})..."
        )

        # ----------------------------------------------------
        # Step 4.1: Return job to queued state
        # ----------------------------------------------------

        mark_job_for_retry(
            job_id,
            error_message,
        )

        print(
            f"Job {job_id} returned to queued state."
        )

        # ----------------------------------------------------
        # Step 4.2: Create a new Redis message
        # ----------------------------------------------------

        retry_message_id = redis_client.xadd(
            OCR_STREAM_NAME,
            {
                "job_id": job_id,
                "file_path": data["file_path"],
                "filename": data.get(
                    "filename",
                    data["file_path"],
                ),
                "content_type": data.get(
                    "content_type",
                    "application/octet-stream",
                ),
            },
        )

        print(
            f"Retry message created: "
            f"{retry_message_id}"
        )

        # ----------------------------------------------------
        # Step 4.3: Acknowledge old Redis message
        # ----------------------------------------------------

        redis_client.xack(
            OCR_STREAM_NAME,
            OCR_CONSUMER_GROUP,
            message_id,
        )

        print(
            f"Old Redis message acknowledged: "
            f"{message_id}"
        )

        print(
            f"Job {job_id} added back to Redis "
            f"for retry."
        )

        print("=" * 60)
        print()

        return

    # --------------------------------------------------------
    # Step 5: Temporary failure but maximum attempts reached
    # --------------------------------------------------------

    print(
        f"Job {job_id} reached maximum attempts "
        f"({MAX_RETRIES})."
    )

    mark_job_failed(
        job_id,
        error_message,
    )

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
        f"Job {job_id} marked as permanently failed "
        f"after {MAX_RETRIES} attempts."
    )

    print("=" * 60)
    print()