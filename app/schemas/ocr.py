from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OCRLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class OCRResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    extracted_text: str
    confidence: float
    result: dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OCRHistoryItem(BaseModel):
    id: int
    filename: str
    content_type: str
    extracted_text: str
    confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    database: str
