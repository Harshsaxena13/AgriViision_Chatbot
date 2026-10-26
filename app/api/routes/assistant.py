from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.models.schemas import (
    AssistantChatForm,
    AssistantRequest,
    AssistantResponse,
    ChatSession,
    DiseaseDetectionResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VoiceChatResponse,
)
from app.services.chat_history import ChatHistoryService, get_chat_history_service
from app.services.disease_detection import (
    WheatDiseaseDetectionService,
    get_disease_detection_service,
)
from app.services.assistant import AgriculturalAssistantService, get_assistant_service
from app.services.pdf_extraction import extract_pdf_upload
from app.services.sarvam_voice import SarvamVoiceService, get_sarvam_voice_service


router = APIRouter()


async def _answer_and_save(
    payload: AssistantRequest,
    service: AgriculturalAssistantService,
    history: ChatHistoryService,
) -> AssistantResponse:
    conversation_id = await history.ensure_conversation(
        payload.conversation_id, payload.farmer_name
    )
    # Load before saving the new message so it is not sent twice.
    previous_messages = await history.get_recent_messages(conversation_id, limit=10)
    await history.add_message(
        conversation_id=conversation_id,
        role="user",
        content=payload.message,
        crop=payload.crop,
        disease=payload.disease,
        location=payload.location,
    )
    response = await service.answer(payload, previous_messages)
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
    return await _answer_and_save(payload, service, history)


@router.post("/assistant/chat", response_model=AssistantResponse)
async def chat_with_attachment(
    form: Annotated[AssistantChatForm, Form()],
    service: AgriculturalAssistantService = Depends(get_assistant_service),
    history: ChatHistoryService = Depends(get_chat_history_service),
) -> AssistantResponse:
    """Multipart variant of /assistant/ask that accepts an optional PDF attachment."""
    payload = AssistantRequest(**form.model_dump(exclude={"file"}))
    file = form.file
    if file is not None:
        document_text = await extract_pdf_upload(file)
        # Added after validation so the document does not count toward the message length limit.
        payload = payload.model_copy(
            update={
                "message": (
                    f"{payload.message}\n\n"
                    f"Attached document: {file.filename or 'document.pdf'}\n"
                    f"{document_text}"
                )
            }
        )
    return await _answer_and_save(payload, service, history)


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


@router.post("/assistant/voice/chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: Optional[str] = Form(default=None),
    language_code: Optional[str] = Form(default=None),
    voice: SarvamVoiceService = Depends(get_sarvam_voice_service),
    service: AgriculturalAssistantService = Depends(get_assistant_service),
    history: ChatHistoryService = Depends(get_chat_history_service),
) -> VoiceChatResponse:
    speech = await voice.transcribe_upload(audio=audio, language_code=language_code or "unknown")
    detected_language = speech.language_code or (
        language_code if language_code and language_code != "unknown" else None
    )
    response = await _answer_and_save(
        AssistantRequest(
            message=speech.transcript,
            conversation_id=conversation_id,
            language=detected_language or "unknown",
            mode="voice",
        ),
        service,
        history,
    )
    audios = await voice.synthesize_speech(response.answer, detected_language)
    return VoiceChatResponse(
        transcript=speech.transcript,
        detected_language=detected_language,
        answer_text=response.answer,
        audios=audios,
        conversation_id=response.conversation_id,
    )


@router.post("/assistant/voice/synthesize", response_model=TextToSpeechResponse)
async def synthesize_voice(
    payload: TextToSpeechRequest,
    service: SarvamVoiceService = Depends(get_sarvam_voice_service),
) -> TextToSpeechResponse:
    return await service.synthesize(payload)
