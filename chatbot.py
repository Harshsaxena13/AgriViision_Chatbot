import asyncio
from typing import Optional

from app.core.config import get_settings
from app.models.schemas import AssistantRequest, AssistantResponse
from app.services.assistant import AgriculturalAssistantService, create_assistant_http_client


def ask_agro_ai(message: str, disease: Optional[str] = None) -> str:
    """Backward-compatible helper for older scripts."""
    request = AssistantRequest(message=message, disease=disease)
    response = asyncio.run(_answer(request))
    return response.answer


async def _answer(request: AssistantRequest) -> AssistantResponse:
    settings = get_settings()
    async with create_assistant_http_client(settings) as client:
        return await AgriculturalAssistantService(settings, client).answer(request)
