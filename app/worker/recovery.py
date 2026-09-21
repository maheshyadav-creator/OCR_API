from app.worker.job_processor import process_message
from app.worker.redis_consumer import (
    OCR_CONSUMER_GROUP,
    OCR_STREAM_NAME,
    WORKER_ID,
    redis_client,
)
from app.worker.retry_handler import handle_failed_message


# A Redis message is considered stale when it has been pending
# for at least 15 minutes.
PENDING_JOB_IDLE_MS = 15 * 60 * 1000

# Number of pending messages to inspect in one recovery scan.
RECOVERY_BATCH_SIZE = 10


def recover_pending_messages() -> int:
    """
    Recover Redis Stream messages that have been pending for
    at least PENDING_JOB_IDLE_MS.

    XAUTOCLAIM transfers ownership of stale pending messages
    from a failed/unresponsive worker to the current worker.

    Returns:
        Number of recovered messages.
    """

    print("Checking Redis for pending OCR jobs...")

    start_id = "0-0"
    recovered_count = 0

    while True:
        result = redis_client.xautoclaim(
            name=OCR_STREAM_NAME,
            groupname=OCR_CONSUMER_GROUP,
            consumername=WORKER_ID,
            min_idle_time=PENDING_JOB_IDLE_MS,
            start_id=start_id,
            count=RECOVERY_BATCH_SIZE,
        )

        next_start_id = result[0]
        messages = result[1]

        if not messages:
            break

        recovered_count += len(messages)

        print(
            f"Recovered {len(messages)} pending OCR job(s) "
            f"by worker {WORKER_ID}."
        )

        for message_id, data in messages:
            try:
                process_message(
                    message_id=message_id,
                    data=data,
                )

            except Exception as exc:
                print(
                    f"Recovered job failed: "
                    f"message_id={message_id}, error={exc}"
                )

                handle_failed_message(
                    message_id=message_id,
                    data=data,
                    exc=exc,
                )

        if next_start_id == "0-0":
            break

        start_id = next_start_id

    if recovered_count == 0:
        print("No stale pending OCR jobs found.")
    else:
        print(
            f"Pending OCR job recovery completed. "
            f"Recovered {recovered_count} job(s)."
        )

    return recovered_count