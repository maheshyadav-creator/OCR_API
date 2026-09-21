import redis

from app.core.config import get_settings


settings = get_settings()

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
)


def enqueue_ocr_job(job_id: str, file_path: str) -> str:
    """
    Add an OCR job to the Redis Stream.
    """

    message_id = redis_client.xadd(
        "ocr_jobs",
        {
            "job_id": job_id,
            "file_path": file_path,
        },
    )

    return message_id