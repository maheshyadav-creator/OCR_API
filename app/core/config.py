from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCR API"
    app_version: str = "1.0.0"
    debug: bool = True

    database_url: str = "sqlite:///./ocr.db"

    ocr_lang: str = "en"
    ocr_device: str = "cpu"

    max_file_size_mb: int = 10

    paddle_pdx_model_source: str = "BOS"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
