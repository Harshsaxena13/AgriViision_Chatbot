"""Clients for Sarvam's WebSocket streaming speech-to-text and text-to-speech APIs.

Message formats follow Sarvam's streaming API (as used by the official sarvamai SDK):
- STT (/speech-to-text/ws): send {"audio": {"data", "sample_rate", "encoding"}}; receive
  {"type": "events", "data": {"signal_type": "START_SPEECH" | "END_SPEECH"}} (with
  vad_signals=true), {"type": "data", "data": {"transcript", "language_code"}} and
  {"type": "error", "data": {"message"}}.
- TTS (/text-to-speech/ws): send a "config" message first, then "text" messages and a
  "flush"; receive {"type": "audio", "data": {"audio", "content_type"}} chunks and, with
  send_completion_event=true, {"type": "event", "data": {"event_type": "final"}}.
"""

import base64
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from app.core.config import Settings
from app.services.sarvam_voice import (
    KEYTERMS_MODEL,
    TTS_MAX_CHARS,
    SarvamVoiceError,
    tts_voice,
)


logger = logging.getLogger(__name__)

STT_SAMPLE_RATE = 16000


@dataclass
class SttEvent:
    kind: str  # "speech_start", "speech_end" or "transcript"
    transcript: str = ""
    language_code: Optional[str] = None


@dataclass
class TtsAudio:
    audio: str  # base64
    content_type: str


class SttStream:
    def __init__(self, websocket: ClientConnection) -> None:
        self._websocket = websocket

    async def send_audio(self, pcm: bytes) -> None:
        """Forward one chunk of 16 kHz, 16-bit mono PCM."""
        await self._websocket.send(
            json.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(pcm).decode("ascii"),
                        "sample_rate": STT_SAMPLE_RATE,
                        # The only accepted value; input_audio_codec=pcm_s16le marks the data as raw PCM.
                        "encoding": "audio/wav",
                    }
                }
            )
        )

    async def events(self) -> AsyncIterator[SttEvent]:
        async for raw in self._websocket:
            message = json.loads(raw)
            data = message.get("data") or {}
            if message.get("type") == "error":
                raise SarvamVoiceError(f"Sarvam streaming STT failed: {data.get('message')}")
            if message.get("type") == "events":
                signal = data.get("signal_type")
                if signal == "START_SPEECH":
                    yield SttEvent("speech_start")
                elif signal == "END_SPEECH":
                    yield SttEvent("speech_end")
            elif message.get("type") == "data":
                yield SttEvent(
                    "transcript",
                    transcript=str(data.get("transcript") or "").strip(),
                    language_code=data.get("language_code"),
                )


class TtsStream:
    def __init__(self, websocket: ClientConnection) -> None:
        self._websocket = websocket
        self._chars_sent = 0

    async def send_text(self, text: str) -> None:
        # Sarvam allows 2500 characters per TTS WebSocket session.
        if self._chars_sent + len(text) > TTS_MAX_CHARS:
            logger.warning("Skipping TTS text beyond the %d-character session limit", TTS_MAX_CHARS)
            return
        self._chars_sent += len(text)
        await self._websocket.send(json.dumps({"type": "text", "data": {"text": text}}))

    async def flush(self) -> None:
        await self._websocket.send(json.dumps({"type": "flush"}))

    async def audio(self) -> AsyncIterator[TtsAudio]:
        """Yield audio chunks until Sarvam signals the flushed text is fully synthesized."""
        async for raw in self._websocket:
            message = json.loads(raw)
            data = message.get("data") or {}
            if message.get("type") == "audio":
                yield TtsAudio(audio=data["audio"], content_type=data.get("content_type", "audio/mpeg"))
            elif message.get("type") == "event" and data.get("event_type") == "final":
                return
            elif message.get("type") == "error":
                raise SarvamVoiceError(f"Sarvam streaming TTS failed: {data.get('message')}")


def _headers(settings: Settings) -> dict[str, str]:
    if not settings.sarvam_api_key:
        raise SarvamVoiceError("SARVAM_API_KEY is required for voice API calls.")
    return {"api-subscription-key": settings.sarvam_api_key}


@asynccontextmanager
async def open_stt_stream(
    settings: Settings, language_code: Optional[str], keyterms: list[str]
) -> AsyncIterator[SttStream]:
    params = {
        "language_code": language_code or "unknown",
        "model": settings.sarvam_stt_model,
        "sample_rate": str(STT_SAMPLE_RATE),
        "input_audio_codec": "pcm_s16le",
        "vad_signals": "true",
    }
    if keyterms and settings.sarvam_stt_model == KEYTERMS_MODEL:
        params["keyterms"] = json.dumps(keyterms)
    async with connect(
        f"{settings.sarvam_stt_stream_url}?{urlencode(params)}",
        additional_headers=_headers(settings),
        open_timeout=settings.sarvam_timeout_seconds,
    ) as websocket:
        yield SttStream(websocket)


@asynccontextmanager
async def open_tts_stream(
    settings: Settings, language_code: Optional[str]
) -> AsyncIterator[TtsStream]:
    language_code, speaker = tts_voice(language_code)
    params = {"model": settings.sarvam_tts_model, "send_completion_event": "true"}
    async with connect(
        f"{settings.sarvam_tts_stream_url}?{urlencode(params)}",
        additional_headers=_headers(settings),
        open_timeout=settings.sarvam_timeout_seconds,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "config",
                    "data": {
                        "language_code": language_code,
                        "speaker": speaker,
                        "output_audio_codec": "mp3",
                    },
                }
            )
        )
        yield TtsStream(websocket)
