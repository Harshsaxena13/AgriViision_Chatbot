from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=2000)
    conversation_id: Optional[str] = Field(default=None, max_length=80)
    farmer_name: Optional[str] = Field(default=None, max_length=100)
    crop: Optional[str] = Field(default=None, max_length=100)
    disease: Optional[str] = Field(default=None, max_length=150)
    location: Optional[str] = Field(default=None, max_length=150)
    language: str = Field(default="en", min_length=2, max_length=20)
    mode: Literal["text", "voice"] = Field(default="text")

    @field_validator(
        "message", "conversation_id", "farmer_name", "crop", "disease", "location", "language"
    )
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None


class AssistantChatForm(AssistantRequest):
    """Multipart form for /assistant/chat: the /assistant/ask fields plus an optional PDF."""

    file: Optional[UploadFile] = None

    @field_validator("file", mode="before")
    @classmethod
    def empty_file_to_none(cls, value: Any) -> Any:
        # Browsers send an empty string for a file input left blank.
        return None if value == "" else value


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


class SpeechToTextResponse(BaseModel):
    request_id: Optional[str] = None
    transcript: str
    language_code: Optional[str] = None
    language_probability: Optional[float] = None
    timestamps: Optional[dict[str, Any]] = None
    model: str = "saaras:v4"
    raw_response: dict[str, Any]


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2500)
    language_code: str = Field(default="en-IN", min_length=2, max_length=20)
    speaker: Optional[str] = Field(default="shubh", max_length=80)
    pace: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    speech_sample_rate: Optional[int] = Field(default=24000)
    output_audio_codec: Optional[str] = Field(default="wav", max_length=20)
    temperature: Optional[float] = Field(default=0.6, ge=0.01, le=2.0)

    @field_validator("text", "language_code", "speaker", "output_audio_codec")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip()
        return normalized or None


class VoiceChatResponse(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    answer_text: str
    audios: list[str]
    conversation_id: str


class TextToSpeechResponse(BaseModel):
    request_id: Optional[str] = None
    audios: list[str]
    model: str = "bulbul:v3"
    output_audio_codec: Optional[str] = "wav"
    speech_sample_rate: Optional[int] = 24000
    raw_response: dict[str, Any]
