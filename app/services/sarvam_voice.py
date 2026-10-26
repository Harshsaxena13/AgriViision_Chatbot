import asyncio
import json
import logging
import re
import textwrap
from typing import Any, Optional

import httpx
from fastapi import UploadFile, status
from starlette.exceptions import HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import (
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)


logger = logging.getLogger(__name__)

TTS_MAX_CHARS = 2500
# Sarvam's recommended male bulbul:v3 voice for each TTS language. Its keys are the only
# languages Sarvam TTS supports; speaker names are only valid for bulbul:v3.
BULBUL_V3_SPEAKERS = {
    "bn-IN": "rehan",
    "en-IN": "ratan",
    "gu-IN": "ratan",
    "hi-IN": "shubh",
    "kn-IN": "shubh",
    "ml-IN": "shubh",
    "mr-IN": "ratan",
    "od-IN": "shubh",
    "pa-IN": "mani",
    "ta-IN": "ratan",
    "te-IN": "shubh",
}
# Used when STT detects no language or one that TTS cannot speak (e.g. ur-IN, as-IN).
FALLBACK_TTS_LANGUAGE = "hi-IN"
# Sentence ends in Latin, Devanagari/Bengali/Gurmukhi/Odia (danda) and Urdu punctuation.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥۔؟])\s+")


class SarvamVoiceError(RuntimeError):
    """Raised when Sarvam voice APIs cannot complete a request."""


def split_tts_text(text: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    """Pack whole sentences into chunks of at most max_chars, splitting overlong sentences on words."""
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
        for piece in textwrap.wrap(sentence, max_chars, break_on_hyphens=False):
            candidate = f"{current} {piece}" if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


class SarvamVoiceService:
    _allowed_audio_content_types = {
        "audio/aac",
        "audio/aiff",
        "audio/amr",
        "audio/flac",
        "audio/m4a",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/opus",
        "audio/wav",
        "audio/webm",
        "audio/x-m4a",
        "audio/x-wav",
        "video/mp4",
        "video/webm",
        "application/octet-stream",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def transcribe_upload(
        self,
        audio: UploadFile,
        language_code: Optional[str] = "unknown",
        mode: Optional[str] = "transcribe",
        with_timestamps: bool = False,
        input_audio_codec: Optional[str] = None,
        keyterms: Optional[list[str]] = None,
    ) -> SpeechToTextResponse:
        content_type = self._normalize_content_type(
            audio.content_type or "application/octet-stream"
        )
        if content_type not in self._allowed_audio_content_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Could not use this browser audio format for speech transcription.",
            )

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio is empty.",
            )

        return await self.transcribe(
            filename=audio.filename or "speech-audio",
            content_type=content_type,
            audio_bytes=audio_bytes,
            language_code=language_code,
            mode=mode,
            with_timestamps=with_timestamps,
            input_audio_codec=input_audio_codec,
            keyterms=keyterms,
        )

    async def transcribe(
        self,
        filename: str,
        content_type: str,
        audio_bytes: bytes,
        language_code: Optional[str] = "unknown",
        mode: Optional[str] = "transcribe",
        with_timestamps: bool = False,
        input_audio_codec: Optional[str] = None,
        keyterms: Optional[list[str]] = None,
    ) -> SpeechToTextResponse:
        self._require_api_key()
        data: dict[str, str] = {
            "model": self._settings.sarvam_stt_model,
            "with_timestamps": str(with_timestamps).lower(),
        }
        if language_code:
            data["language_code"] = language_code
        if mode:
            data["mode"] = mode
        if input_audio_codec:
            data["input_audio_codec"] = input_audio_codec
        if keyterms:
            data["keyterms"] = json.dumps(keyterms)

        logger.info(
            "Calling Sarvam STT endpoint=%s model=%s filename=%s language=%s",
            self._settings.sarvam_stt_url,
            self._settings.sarvam_stt_model,
            filename,
            language_code,
        )
        try:
            async with self._client() as client:
                response = await client.post(
                    self._settings.sarvam_stt_url,
                    data=data,
                    files={"file": (filename, audio_bytes, content_type)},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._response_detail(exc.response)
            raise SarvamVoiceError(f"Sarvam speech-to-text failed: {detail}") from exc
        except httpx.HTTPError as exc:
            raise SarvamVoiceError(f"Sarvam speech-to-text request failed: {exc}") from exc

        payload = self._json_payload(response, "Sarvam speech-to-text returned invalid JSON.")
        transcript = str(payload.get("transcript") or "").strip()
        if not transcript:
            raise SarvamVoiceError("Sarvam speech-to-text returned an empty transcript.")

        return SpeechToTextResponse(
            request_id=payload.get("request_id"),
            transcript=transcript,
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            timestamps=payload.get("timestamps"),
            model=self._settings.sarvam_stt_model,
            raw_response=payload,
        )

    async def synthesize(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        self._require_api_key()
        payload = request.model_dump(exclude_none=True)
        payload["model"] = self._settings.sarvam_tts_model

        logger.info(
            "Calling Sarvam TTS endpoint=%s model=%s language=%s speaker=%s",
            self._settings.sarvam_tts_url,
            self._settings.sarvam_tts_model,
            request.language_code,
            request.speaker,
        )
        try:
            async with self._client() as client:
                response = await client.post(self._settings.sarvam_tts_url, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._response_detail(exc.response)
            raise SarvamVoiceError(f"Sarvam text-to-speech failed: {detail}") from exc
        except httpx.HTTPError as exc:
            raise SarvamVoiceError(f"Sarvam text-to-speech request failed: {exc}") from exc

        body = self._json_payload(response, "Sarvam text-to-speech returned invalid JSON.")
        audios = body.get("audios")
        if not isinstance(audios, list) or not audios:
            raise SarvamVoiceError("Sarvam text-to-speech returned no audio.")

        return TextToSpeechResponse(
            request_id=body.get("request_id"),
            audios=[str(audio) for audio in audios],
            model=self._settings.sarvam_tts_model,
            output_audio_codec=request.output_audio_codec,
            speech_sample_rate=request.speech_sample_rate,
            raw_response=body,
        )

    async def synthesize_speech(self, text: str, language_code: Optional[str]) -> list[str]:
        """Synthesize text of any length, returning base64 WAV audios in reading order."""
        if language_code not in BULBUL_V3_SPEAKERS:
            logger.warning(
                "Sarvam TTS does not support language=%s; using %s",
                language_code,
                FALLBACK_TTS_LANGUAGE,
            )
            language_code = FALLBACK_TTS_LANGUAGE
        speaker = BULBUL_V3_SPEAKERS[language_code]
        responses = await asyncio.gather(
            *(
                self.synthesize(
                    TextToSpeechRequest(text=chunk, language_code=language_code, speaker=speaker)
                )
                for chunk in split_tts_text(text)
            )
        )
        return [audio for response in responses for audio in response.audios]

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._settings.sarvam_timeout_seconds,
            headers={"api-subscription-key": self._settings.sarvam_api_key or ""},
        )

    def _normalize_content_type(self, content_type: str) -> str:
        return content_type.split(";", 1)[0].strip().lower()

    def _require_api_key(self) -> None:
        if not self._settings.sarvam_api_key:
            raise SarvamVoiceError("SARVAM_API_KEY is required for voice API calls.")

    def _json_payload(self, response: httpx.Response, error_message: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SarvamVoiceError(error_message) from exc
        if not isinstance(payload, dict):
            raise SarvamVoiceError(error_message)
        return payload

    def _response_detail(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or response.reason_phrase
        return str(payload.get("detail") or payload.get("error") or payload)


def get_sarvam_voice_service() -> SarvamVoiceService:
    return SarvamVoiceService(get_settings())
