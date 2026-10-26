import asyncio
import contextlib
import logging
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import Settings
from app.models.schemas import AssistantRequest
from app.services.assistant import AgriculturalAssistantService
from app.services.chat_history import ChatHistoryService
from app.services.sarvam_streaming import SttStream, TtsStream, open_stt_stream, open_tts_stream
from app.services.sarvam_voice import load_keyterms, pop_sentences


logger = logging.getLogger(__name__)


async def _finish_despite_cancellation(task: "asyncio.Future[None]") -> None:
    """Wait for task to complete even if the caller is cancelled meanwhile, then re-raise."""
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise


class VoiceStreamSession:
    """One /voice/stream connection: app audio -> Sarvam STT -> streamed LLM reply -> Sarvam TTS -> app.

    A reply runs as its own task so that a new utterance (barge-in) can cancel it, which closes
    the OpenRouter stream and the TTS socket.
    """

    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        assistant: AgriculturalAssistantService,
        history: ChatHistoryService,
        conversation_id: Optional[str],
        language_code: Optional[str],
    ) -> None:
        self._websocket = websocket
        self._settings = settings
        self._assistant = assistant
        self._history = history
        self._conversation_id = conversation_id
        self._language_code = language_code
        self._send_lock = asyncio.Lock()
        self._reply: Optional[asyncio.Task[None]] = None
        self._speaking = False
        self._pending: list[str] = []
        self._pending_language: Optional[str] = None

    async def run(self) -> None:
        self._conversation_id = await self._history.ensure_conversation(self._conversation_id, None)
        async with open_stt_stream(self._settings, self._language_code, load_keyterms()) as stt:
            await self._send({"type": "ready", "conversation_id": self._conversation_id})
            uplink = asyncio.create_task(self._forward_audio(stt))
            downlink = asyncio.create_task(self._handle_stt(stt))
            try:
                done, _ = await asyncio.wait({uplink, downlink}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in (uplink, downlink):
                    task.cancel()
                await self._cancel_reply()

    async def send_error(self, detail: str) -> None:
        await self._send({"type": "error", "detail": detail})

    async def _forward_audio(self, stt: SttStream) -> None:
        while True:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes"):
                await stt.send_audio(message["bytes"])

    async def _handle_stt(self, stt: SttStream) -> None:
        async for event in stt.events():
            if event.kind == "speech_start":
                self._speaking = True
                interrupted = await self._cancel_reply()
                # The app should stop playback and drop queued audio whenever it gets this event.
                await self._send({"type": "speech_start", "interrupted": interrupted})
            elif event.kind == "speech_end":
                self._speaking = False
                await self._send({"type": "speech_end"})
                await self._start_reply_if_ready()
            elif event.kind == "transcript" and event.transcript:
                self._pending.append(event.transcript)
                self._pending_language = event.language_code or self._pending_language
                await self._send(
                    {
                        "type": "transcript",
                        "text": event.transcript,
                        "language_code": event.language_code,
                    }
                )
                await self._start_reply_if_ready()

    async def _start_reply_if_ready(self) -> None:
        # Reply once speech has ended and its transcript has arrived, in either order.
        if self._speaking or not self._pending:
            return
        transcript = " ".join(self._pending)
        language_code = self._pending_language or self._language_code
        self._pending.clear()
        self._pending_language = None
        await self._cancel_reply()
        self._reply = asyncio.create_task(self._respond(transcript, language_code))

    async def _cancel_reply(self) -> bool:
        """Cancel an in-progress reply and wait for it to save history. Returns True if one was running."""
        reply, self._reply = self._reply, None
        if reply is None or reply.done():
            return False
        reply.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reply
        return True

    async def _respond(self, transcript: str, language_code: Optional[str]) -> None:
        try:
            request = AssistantRequest(
                message=transcript,
                conversation_id=self._conversation_id,
                language=language_code or "unknown",
                mode="voice",
            )
        except ValidationError:
            await self.send_error("Could not understand that. Please say it again.")
            return

        previous_messages = await self._history.get_recent_messages(self._conversation_id, limit=10)
        answer_parts: list[str] = []
        completed = False
        try:
            async with open_tts_stream(self._settings, language_code) as tts:
                audio = asyncio.create_task(self._forward_tts_audio(tts))
                try:
                    unfinished = ""
                    async for delta in self._assistant.stream_answer(request, previous_messages):
                        answer_parts.append(delta)
                        await self._send({"type": "answer_delta", "text": delta})
                        sentences, unfinished = pop_sentences(unfinished + delta)
                        for sentence in sentences:
                            await tts.send_text(sentence)
                    if unfinished.strip():
                        await tts.send_text(unfinished.strip())
                    await tts.flush()
                    await asyncio.wait_for(audio, timeout=self._settings.sarvam_timeout_seconds)
                finally:
                    audio.cancel()
                    # Collect the task's outcome so a TTS error is not reported as unretrieved.
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await audio
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # An LLM or TTS failure ends this reply, not the session.
            logger.exception("Voice stream reply failed")
            await self.send_error(str(exc) or "The voice reply failed.")
        finally:
            # On barge-in this saves the partial answer generated before the interruption.
            # Shielded so a barge-in during the save cannot leave the turn half-written.
            await _finish_despite_cancellation(
                asyncio.ensure_future(self._save_turn(request, "".join(answer_parts).strip()))
            )
        if completed:
            await self._send({"type": "reply_end"})

    async def _forward_tts_audio(self, tts: TtsStream) -> None:
        async for chunk in tts.audio():
            await self._send(
                {"type": "audio", "audio": chunk.audio, "content_type": chunk.content_type}
            )

    async def _save_turn(self, request: AssistantRequest, answer: str) -> None:
        await self._history.add_message(
            conversation_id=self._conversation_id, role="user", content=request.message
        )
        if answer:
            await self._history.add_message(
                conversation_id=self._conversation_id,
                role="assistant",
                content=answer,
                provider=self._assistant.provider_name,
                model=self._assistant.model_name,
            )

    async def _send(self, message: dict[str, Any]) -> None:
        # Several tasks send on this socket; a disconnected app is not an error worth raising.
        async with self._send_lock:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await self._websocket.send_json(message)
