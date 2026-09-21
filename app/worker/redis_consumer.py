import socket

import redis

from app.core.config import get_settings


settings = get_settings()


# ============================================================
# Redis configuration
# ============================================================

OCR_STREAM_NAME = "ocr_jobs"
OCR_CONSUMER_GROUP = "ocr_workers"

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
)


# ============================================================
# Worker configuration
# ============================================================

WORKER_ID = socket.gethostname()


# ============================================================
# Redis Consumer Group
# ============================================================

def create_consumer_group() -> None:
    """
    Create the Redis consumer group if it does not already exist.
    """

    try:
        redis_client.xgroup_create(
            name=OCR_STREAM_NAME,
            groupname=OCR_CONSUMER_GROUP,
            id="0",
            mkstream=True,
        )

        print(
            f"Consumer group created: "
            f"{OCR_CONSUMER_GROUP}"
        )

    except redis.exceptions.ResponseError as exc:

        if "BUSYGROUP" in str(exc):

            print(
                f"Consumer group already exists: "
                f"{OCR_CONSUMER_GROUP}"
            )

        else:
            raise