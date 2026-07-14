import asyncio
from typing import Optional

from app.models.schemas import AssistantRequest
from app.services.assistant import get_assistant_service


def ask_agro_ai(message: str, disease: Optional[str] = None) -> str:
    """Backward-compatible helper for older scripts."""
    request = AssistantRequest(message=message, disease=disease)
    response = asyncio.run(get_assistant_service().answer(request))
    return response.answer
