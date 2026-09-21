from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# OCR line
# ============================================================

class OCRLine(BaseModel):
    text: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# ============================================================
# Single OCR job response
# ============================================================

class OCRJobResponse(BaseModel):
    job_id: str
    filename: str
    content_type: str
    status: str
    created_at: datetime


# ============================================================
# Multiple OCR jobs response
# ============================================================

class OCRBatchResponse(BaseModel):
    jobs: list[OCRJobResponse]


# ============================================================
# OCR job status
# ============================================================

class OCRJobStatusResponse(BaseModel):
    job_id: str
    filename: str
    content_type: str
    status: str
    attempt_count: int
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ============================================================
# OCR result
# ============================================================

class OCRResponse(BaseModel):
    id: int
    job_id: str | None = None
    filename: str
    content_type: str
    extracted_text: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    result: dict[str, Any]
    created_at: datetime


# ============================================================
# OCR history item
# ============================================================

class OCRHistoryItem(BaseModel):
    id: int
    job_id: str | None = None
    filename: str
    content_type: str
    extracted_text: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    created_at: datetime


# ============================================================
# Health response
# ============================================================

class HealthResponse(BaseModel):
    status: str
    database: str