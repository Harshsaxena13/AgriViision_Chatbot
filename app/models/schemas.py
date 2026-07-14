from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=2000)
    conversation_id: Optional[str] = Field(default=None, max_length=80)
    farmer_name: Optional[str] = Field(default=None, max_length=100)
    crop: Optional[str] = Field(default=None, max_length=100)
    disease: Optional[str] = Field(default=None, max_length=150)
    location: Optional[str] = Field(default=None, max_length=150)
    language: str = Field(default="en", min_length=2, max_length=20)

    @field_validator(
        "message", "conversation_id", "farmer_name", "crop", "disease", "location", "language"
    )
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None


class AssistantResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[str] = None
    answer: str
    provider: str
    model: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    safety_notice: str = (
        "Use this guidance as decision support. Confirm severe crop disease, chemical dosage, "
        "and local regulations with a qualified agronomist or extension officer."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
    fields: Optional[list[dict[str, Any]]] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    crop: Optional[str] = None
    disease: Optional[str] = None
    location: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: str


class ChatSession(BaseModel):
    conversation_id: str
    messages: list[ChatMessage]


class DiseaseDetectionResponse(BaseModel):
    disease: str
    confidence: Optional[float] = None
    raw_response: dict[str, Any]
