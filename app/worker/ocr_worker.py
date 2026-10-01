from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
import time

from app.database.connection import (
    close_connection_pool,
    initialize_connection_pool,
)

from app.worker.job_processor import process_message
from app.worker.recovery import recover_pending_messages
from app.worker.redis_consumer import (
    OCR_CONSUMER_GROUP,
    OCR_STREAM_NAME,
    WORKER_ID,
    create_consumer_group,
    redis_client,
)
from app.worker.retry_handler import handle_failed_message


# ============================================================
# CONFIGURATION
# ============================================================

# How often each worker checks Redis for stale pending jobs.
RECOVERY_INTERVAL_SECONDS = 60

# Maximum number of OCR jobs that this ONE worker process
# can have active at the same time.
#
# IMPORTANT:
# This does NOT create two PaddleOCR engines.
# Both threads use the same cached PaddleOCR engine
# from app.services.ocr_service.
MAX_CONCURRENT_JOBS = 2


# ============================================================
# PROCESS ONE REDIS MESSAGE
# ============================================================

def _process_message_safely(
    message_id: str,
    data: dict,
) -> None:
    """
    Process one Redis OCR message.

    This function runs inside a ThreadPoolExecutor worker
    thread.

    process_message() contains the existing:
        - PostgreSQL job-state handling
        - OCR execution
        - result saving
        - Redis acknowledgement
        - retry-related behavior

    We keep that logic unchanged.
    """

    try:

        process_message(
            message_id=message_id,
            data=data,
        )

    except Exception as exc:

        handle_failed_message(
            message_id=message_id,
            data=data,
            exc=exc,
        )


# ============================================================
# WORKER MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 60)
    print(f"OCR worker started: {WORKER_ID}")
    print(f"Concurrent job slots: {MAX_CONCURRENT_JOBS}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Initialize PostgreSQL connection pool
    # ---------------------------------------------------------

    initialize_connection_pool()

    # ---------------------------------------------------------
    # Create executor
    # ---------------------------------------------------------

    executor = ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_JOBS,
        thread_name_prefix="ocr-slot",
    )

    # Active futures.
    #
    # Each future represents one OCR job currently being
    # processed by one thread.
    active_futures = {}

    try:

        # -----------------------------------------------------
        # Create Redis consumer group
        # -----------------------------------------------------

        create_consumer_group()

        # -----------------------------------------------------
        # Initial recovery
        # -----------------------------------------------------

        recover_pending_messages()

        print()
        print(
            "Worker is now waiting for new OCR jobs..."
        )

        print(
            f"Maximum active jobs: "
            f"{MAX_CONCURRENT_JOBS}"
        )

        print(
            "PaddleOCR engine: "
            "shared by the worker threads"
        )

        # The first periodic recovery should happen after the
        # configured interval, not immediately again.
        last_recovery_time = time.monotonic()

        # =====================================================
        # MAIN WORKER LOOP
        # =====================================================

        while True:

            # -------------------------------------------------
            # Periodic recovery
            # -------------------------------------------------

            current_time = time.monotonic()

            if (
                current_time - last_recovery_time
                >= RECOVERY_INTERVAL_SECONDS
            ):

                print()
                print(
                    f"Running periodic recovery "
                    f"for worker {WORKER_ID}..."
                )

                recover_pending_messages()

                last_recovery_time = time.monotonic()

            # -------------------------------------------------
            # Remove completed futures
            # -------------------------------------------------
            #
            # We first check whether any currently running
            # threads have finished.
            #
            # This frees slots for new Redis jobs.
            # -------------------------------------------------

            if active_futures:

                completed_futures = [
                    future
                    for future in active_futures
                    if future.done()
                ]

                for future in completed_futures:

                    message_id = active_futures.pop(
                        future
                    )

                    try:

                        future.result()

                    except Exception as exc:

                        # _process_message_safely() normally
                        # catches exceptions itself.
                        #
                        # This is an additional safety net so
                        # that an unexpected executor exception
                        # does not terminate the worker.
                        print(
                            f"Unexpected worker-thread "
                            f"error for Redis message "
                            f"{message_id}: {exc}"
                        )

            # -------------------------------------------------
            # If all slots are occupied, wait for at least
            # one job to finish before reading another job.
            # -------------------------------------------------

            if len(active_futures) >= MAX_CONCURRENT_JOBS:

                done, _ = wait(
                    active_futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:

                    message_id = active_futures.pop(
                        future
                    )

                    try:

                        future.result()

                    except Exception as exc:

                        print(
                            f"Unexpected worker-thread "
                            f"error for Redis message "
                            f"{message_id}: {exc}"
                        )

                continue

            # -------------------------------------------------
            # Calculate how many Redis messages we can accept.
            # -------------------------------------------------

            available_slots = (
                MAX_CONCURRENT_JOBS
                - len(active_futures)
            )

            if available_slots <= 0:
                continue

            # -------------------------------------------------
            # Read new Redis Stream messages
            # -------------------------------------------------
            #
            # We request only as many messages as we currently
            # have free processing slots.
            #
            # Example:
            #
            #   2 free slots → count=2
            #   1 free slot  → count=1
            #
            # This prevents the worker from pulling a large
            # number of jobs into memory.
            # -------------------------------------------------

            messages = redis_client.xreadgroup(
                groupname=OCR_CONSUMER_GROUP,
                consumername=WORKER_ID,
                streams={
                    OCR_STREAM_NAME: ">"
                },
                count=available_slots,
                block=5000,
            )

            if not messages:
                continue

            # -------------------------------------------------
            # Submit received messages to worker threads
            # -------------------------------------------------

            for stream_name, entries in messages:

                for message_id, data in entries:

                    # Safety check.
                    #
                    # Normally this cannot exceed
                    # MAX_CONCURRENT_JOBS because count is
                    # limited above.
                    if (
                        len(active_futures)
                        >= MAX_CONCURRENT_JOBS
                    ):
                        break

                    print()
                    print(
                        f"Submitting Redis message "
                        f"{message_id} "
                        f"to OCR thread."
                    )

                    future = executor.submit(
                        _process_message_safely,
                        message_id,
                        data,
                    )

                    active_futures[future] = (
                        message_id
                    )

                if (
                    len(active_futures)
                    >= MAX_CONCURRENT_JOBS
                ):
                    break

    except KeyboardInterrupt:

        print()
        print(
            "OCR worker shutting down..."
        )

    finally:

        # -----------------------------------------------------
        # Wait for currently running OCR jobs to finish.
        # -----------------------------------------------------
        #
        # We do not abruptly terminate active OCR operations.
        # This gives running jobs an opportunity to complete
        # their PostgreSQL save and Redis acknowledgement.
        # -----------------------------------------------------

        print()
        print(
            "Waiting for active OCR jobs to finish..."
        )

        executor.shutdown(
            wait=True
        )

        # -----------------------------------------------------
        # Close PostgreSQL connection pool
        # -----------------------------------------------------

        close_connection_pool()

        print(
            "OCR worker stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
