import os

from celery import Celery


# ============================================================
# Celery configuration
# ============================================================


CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@rabbitmq:5672//",
)

celery_app = Celery(
    "ocr_api",
    broker=CELERY_BROKER_URL,
)


# ============================================================
# Celery configuration
# ============================================================

celery_app.conf.update(

    # --------------------------------------------------------
    # Memory / concurrency control
    # --------------------------------------------------------

    worker_prefetch_multiplier=1,

    worker_concurrency=1,

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    task_acks_late=True,

    task_reject_on_worker_lost=True,

    task_track_started=True,

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    task_serializer="json",

    accept_content=["json"],

    result_serializer="json",

    # --------------------------------------------------------
    # We use PostgreSQL as the source of truth.
    # Celery result backend is not required.
    # --------------------------------------------------------

    task_ignore_result=True,

    # --------------------------------------------------------
    # Timezone
    # --------------------------------------------------------

    timezone="Asia/Kolkata",

    enable_utc=True,
)


# ============================================================
# Register tasks
# ============================================================

celery_app.autodiscover_tasks(
    [
        "app.tasks",
    ]
)

celery_app.conf.imports = (
    "app.tasks.ocr_tasks",
)
