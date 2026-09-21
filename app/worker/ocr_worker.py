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


# How often each worker checks Redis for stale pending jobs.
RECOVERY_INTERVAL_SECONDS = 60


def main() -> None:
    print()
    print("=" * 60)
    print(f"OCR worker started: {WORKER_ID}")
    print("=" * 60)

    initialize_connection_pool()

    try:
        create_consumer_group()

        # ---------------------------------------------------------
        # Initial recovery
        # ---------------------------------------------------------
        recover_pending_messages()

        print("Worker is now waiting for new OCR jobs...")

        # The first periodic recovery should happen after the
        # configured interval, not immediately again.
        last_recovery_time = time.monotonic()

        while True:

            # -----------------------------------------------------
            # Periodic recovery
            # -----------------------------------------------------
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

            # -----------------------------------------------------
            # Read new Redis Stream messages
            # -----------------------------------------------------
            messages = redis_client.xreadgroup(
                groupname=OCR_CONSUMER_GROUP,
                consumername=WORKER_ID,
                streams={
                    OCR_STREAM_NAME: ">"
                },
                count=1,
                block=5000,
            )

            if not messages:
                continue

            # -----------------------------------------------------
            # Process received messages
            # -----------------------------------------------------
            for stream_name, entries in messages:

                for message_id, data in entries:

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

    except KeyboardInterrupt:
        print("OCR worker shutting down...")

    finally:
        close_connection_pool()
        print("OCR worker stopped.")


if __name__ == "__main__":
    main()