from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.models.schemas import (
    AssistantRequest,
    AssistantResponse,
    ChatSession,
    DiseaseDetectionResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from app.services.chat_history import ChatHistoryService, get_chat_history_service
from app.services.disease_detection import (
    WheatDiseaseDetectionService,
    get_disease_detection_service,
)
from app.services.assistant import AgriculturalAssistantService, get_assistant_service
from app.services.sarvam_voice import SarvamVoiceService, get_sarvam_voice_service


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


@router.post("/assistant/voice/transcribe", response_model=SpeechToTextResponse)
async def transcribe_voice(
    audio: UploadFile = File(...),
    language_code: Optional[str] = Form(default="unknown"),
    mode: Optional[str] = Form(default="transcribe"),
    with_timestamps: bool = Form(default=False),
    input_audio_codec: Optional[str] = Form(default=None),
    keyterms: Optional[list[str]] = Form(default=None),
    service: SarvamVoiceService = Depends(get_sarvam_voice_service),
) -> SpeechToTextResponse:
    return await service.transcribe_upload(
        audio=audio,
        language_code=language_code,
        mode=mode,
        with_timestamps=with_timestamps,
        input_audio_codec=input_audio_codec,
        keyterms=keyterms,
    )


@router.post("/assistant/voice/synthesize", response_model=TextToSpeechResponse)
async def synthesize_voice(
    payload: TextToSpeechRequest,
    service: SarvamVoiceService = Depends(get_sarvam_voice_service),
) -> TextToSpeechResponse:
    return await service.synthesize(payload)
