from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class OCRResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    result: dict[str, Any]
    created_at: datetime


class OCRHistoryItem(BaseModel):
    id: int
    filename: str
    content_type: str
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    database: str