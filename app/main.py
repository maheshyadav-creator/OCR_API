from fastapi import FastAPI

from app.core.config import get_settings
from app.database.connection import (
    check_database_connection,
    close_connection_pool,
    initialize_connection_pool,
    initialize_database,
)
from app.routes.ocr import router as ocr_router


settings = get_settings()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI OCR service using PaddleOCR and PostgreSQL.",
)


@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize the PostgreSQL connection pool and database schema.
    """

    max_attempts = 30

    for attempt in range(1, max_attempts + 1):
        try:
            initialize_connection_pool()

            if not check_database_connection():
                raise RuntimeError("PostgreSQL is not accepting connections.")

            initialize_database()

            print("PostgreSQL connection established.")
            print("Database tables initialized.")

            return

        except Exception as exc:
            print(
                f"Database connection attempt "
                f"{attempt}/{max_attempts} failed: {exc}"
            )

            close_connection_pool()

            if attempt == max_attempts:
                raise

    raise RuntimeError("Unable to initialize PostgreSQL database.")


@app.on_event("shutdown")
def shutdown_event() -> None:
    """
    Close all PostgreSQL connections when the application stops.
    """

    close_connection_pool()
    print("PostgreSQL connection pool closed.")


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