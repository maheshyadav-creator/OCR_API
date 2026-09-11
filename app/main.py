import time

from fastapi import FastAPI
from sqlalchemy import text

from app.core.config import get_settings
from app.database.connection import check_database_connection, engine
from app.database.models import Base
from app.routes.ocr import router as ocr_router


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "FastAPI OCR service using PaddleOCR and PostgreSQL."
    ),
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Create database tables after PostgreSQL becomes available.
    """

    max_attempts = 30

    for attempt in range(1, max_attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)

            print("PostgreSQL connection established.")
            print("Database tables initialized.")

            return

        except Exception as exc:
            print(
                f"Database connection attempt "
                f"{attempt}/{max_attempts} failed: {exc}"
            )

            if attempt == max_attempts:
                raise

            time.sleep(2)


@app.get("/")
def root():
    return {
        "message": "OCR API is running",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    database_ok = check_database_connection()

    return {
        "status": "healthy" if database_ok else "unhealthy",
        "database": "connected" if database_ok else "disconnected",
    }


app.include_router(ocr_router)
