import redis

from app.core.config import get_settings


settings = get_settings()


redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
)


OCR_STREAM_NAME = "ocr_jobs"


def enqueue_ocr_job(
    job_id: str,
    file_path: str,
    filename: str,
    content_type: str,
) -> str:
    """
    Add an OCR job to the Redis Stream.

    Returns:
        Redis message ID.
    """

    message_id = redis_client.xadd(
        OCR_STREAM_NAME,
        {
            "job_id": job_id,
            "file_path": file_path,
            "filename": filename,
            "content_type": content_type,
        },
    )

    return message_id