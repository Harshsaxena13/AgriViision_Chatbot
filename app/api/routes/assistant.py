from fastapi import APIRouter, Depends, File, UploadFile, status

from app.models.schemas import (
    AssistantRequest,
    AssistantResponse,
    ChatSession,
    DiseaseDetectionResponse,
)
from app.services.chat_history import ChatHistoryService, get_chat_history_service
from app.services.disease_detection import (
    WheatDiseaseDetectionService,
    get_disease_detection_service,
)
from app.services.assistant import AgriculturalAssistantService, get_assistant_service


router = APIRouter()


@router.post(
    "/assistant/ask",
    response_model=AssistantResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_assistant(
    payload: AssistantRequest,
    service: AgriculturalAssistantService = Depends(get_assistant_service),
    history: ChatHistoryService = Depends(get_chat_history_service),
) -> AssistantResponse:
    conversation_id = await history.ensure_conversation(
        payload.conversation_id, payload.farmer_name
    )
    await history.add_message(
        conversation_id=conversation_id,
        role="user",
        content=payload.message,
        crop=payload.crop,
        disease=payload.disease,
        location=payload.location,
    )
    response = await service.answer(payload)
    response.conversation_id = conversation_id
    await history.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response.answer,
        crop=payload.crop,
        disease=payload.disease,
        location=payload.location,
        provider=response.provider,
        model=response.model,
    )
    return response


@router.get("/assistant/chats/{conversation_id}", response_model=ChatSession)
async def get_chat_history(
    conversation_id: str,
    history: ChatHistoryService = Depends(get_chat_history_service),
) -> ChatSession:
    return await history.get_session(conversation_id)


@router.post("/assistant/diagnose", response_model=DiseaseDetectionResponse)
async def diagnose_wheat_disease(
    image: UploadFile = File(...),
    service: WheatDiseaseDetectionService = Depends(get_disease_detection_service),
) -> DiseaseDetectionResponse:
    return await service.predict_upload(image)
