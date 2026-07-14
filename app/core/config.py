import os
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv(override=True)


class Settings(BaseModel):
    app_name: str = Field(default="AI Agricultural Assistant")
    app_version: str = Field(default="1.0.0")
    api_prefix: str = Field(default="/api/v1")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    ai_provider: Literal["gemini", "ollama", "openrouter", "rule_based"] = Field(default="rule_based")
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="openai/gpt-4o-mini")
    ai_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    database_path: str = Field(default="data/agro_assistant.db")
    disease_api_url: str = Field(default="http://127.0.0.1:8000")
    disease_api_predict_path: str = Field(default="/predict")
    disease_api_file_field: str = Field(default="file")
    disease_api_timeout_seconds: float = Field(default=90.0, ge=1.0, le=180.0)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "AI Agricultural Assistant"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        debug=_get_bool("DEBUG", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        ai_provider=os.getenv("AI_PROVIDER", "rule_based"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        ai_timeout_seconds=float(os.getenv("AI_TIMEOUT_SECONDS", "30")),
        database_path=os.getenv("DATABASE_PATH", "data/agro_assistant.db"),
        disease_api_url=os.getenv("DISEASE_API_URL", "http://127.0.0.1:8000"),
        disease_api_predict_path=os.getenv("DISEASE_API_PREDICT_PATH", "/predict"),
        disease_api_file_field=os.getenv("DISEASE_API_FILE_FIELD", "file"),
        disease_api_timeout_seconds=float(os.getenv("DISEASE_API_TIMEOUT_SECONDS", "90")),
    )
