from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCR API"
    app_version: str = "1.0.0"
    debug: bool = False

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ocr_db"
    db_user: str = "ocr_user"
    db_password: str = "ocr_password"

    # OCR
    ocr_lang: str = "en"
    ocr_device: str = "cpu"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379

    # Upload limits
    max_file_size_mb: int = 25

    # Supported upload formats
    allowed_extensions: str = (
        ".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.pdf"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()