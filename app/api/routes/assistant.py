import contextlib
import logging
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, WebSocket, status

from app.core.config import get_settings

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
from app.services.sarvam_voice import (
    SarvamVoiceService,
    get_sarvam_voice_service,
    load_keyterms,
)
from app.services.voice_stream import VoiceStreamSession


logger = logging.getLogger(__name__)

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
    started = time.perf_counter()
    speech = await voice.transcribe_upload(
        audio=audio,
        language_code=language_code or "unknown",
        keyterms=load_keyterms(),
    )
    stt_done = time.perf_counter()
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
    llm_done = time.perf_counter()
    audios = await voice.synthesize_speech(response.answer, detected_language)
    tts_done = time.perf_counter()
    # The LLM phase also covers the chat-history reads and writes around the model call.
    logger.info(
        "Voice chat timings conversation_id=%s language=%s stt_ms=%d llm_ms=%d tts_ms=%d "
        "total_ms=%d audio_chunks=%d",
        response.conversation_id,
        detected_language,
        (stt_done - started) * 1000,
        (llm_done - stt_done) * 1000,
        (tts_done - llm_done) * 1000,
        (tts_done - started) * 1000,
        len(audios),
    )
    return VoiceChatResponse(
        transcript=speech.transcript,
        detected_language=detected_language,
        answer_text=response.answer,
        audios=audios,
        conversation_id=response.conversation_id,
    )


@router.websocket("/assistant/voice/stream")
async def voice_stream(
    websocket: WebSocket,
    conversation_id: Optional[str] = Query(default=None, max_length=80),
    language_code: Optional[str] = Query(default=None, max_length=20),
    service: AgriculturalAssistantService = Depends(get_assistant_service),
    history: ChatHistoryService = Depends(get_chat_history_service),
) -> None:
    """Live voice conversation with barge-in.

    App -> server: binary frames of 16 kHz, 16-bit mono PCM (about 100 ms each works well).
    Server -> app: JSON text frames with a "type" of:
    - "ready": {"conversation_id"}; send audio after this.
    - "speech_start": {"interrupted"}; stop playback and drop queued audio (barge-in).
    - "speech_end", then "transcript": {"text", "language_code"}.
    - "answer_delta": {"text"} as the reply streams in.
    - "audio": {"audio" (base64), "content_type"}; play chunks in order.
    - "reply_end" when the reply's audio has all been sent.
    - "error": {"detail"}; reply errors keep the connection open, connection errors close it.
    """
    await websocket.accept()
    session = VoiceStreamSession(
        websocket, get_settings(), service, history, conversation_id, language_code
    )
    close_code = status.WS_1000_NORMAL_CLOSURE
    try:
        await session.run()
    except Exception as exc:
        logger.exception("Voice stream failed")
        await session.send_error(str(exc) or "The voice stream failed.")
        close_code = status.WS_1011_INTERNAL_ERROR
    # Closing fails harmlessly if the app has already disconnected.
    with contextlib.suppress(RuntimeError):
        await websocket.close(code=close_code)


@router.post("/assistant/voice/synthesize", response_model=TextToSpeechResponse)
async def synthesize_voice(
    payload: TextToSpeechRequest,
    service: SarvamVoiceService = Depends(get_sarvam_voice_service),
) -> TextToSpeechResponse:
    return await service.synthesize(payload)
