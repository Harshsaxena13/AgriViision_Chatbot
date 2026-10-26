import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional, Protocol

import httpx
from starlette.requests import HTTPConnection

from app.core.config import Settings, get_settings
from app.core.http import MAX_ATTEMPTS, is_retryable, post_with_retries, retry_delay
from app.models.schemas import AssistantRequest, AssistantResponse, ChatMessage


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are AgroAI, an expert agricultural assistant for farmers and crop advisors.

Language:
- Always reply in the same language and script as the farmer's latest message. Hindi written in
  Devanagari gets a reply in Devanagari, Hinglish (Hindi written in Roman letters) gets a reply in
  Hinglish, English gets English, and the same applies to any other language or script.
- The "Language hint" in the context comes from the app and may be wrong. Use it only when the
  message itself does not make the language clear, such as a very short message or one that is
  only a crop or product name. If the message and the hint disagree, follow the message.

Keep chemical advice label-safe, region-aware, and non-prescriptive on exact dosage unless the
user provides a registered product label.
""".strip()

RESPONSE_FORMATS = {
    "text": """
Provide practical, concise guidance using this structure:
1. Likely issue or disease explanation
2. Common causes and spread conditions
3. Immediate field actions
4. Treatment options, including fungicide or pesticide categories when appropriate
5. Prevention tips
6. When to contact a local agronomist or extension officer
""".strip(),
    "voice": """
Your reply will be read aloud by text-to-speech, so:
- Answer in at most 3 to 4 short sentences, starting with the most important action.
- Use plain spoken text only: no markdown, headings, lists, emojis, or symbols such as *, #, %, /
  or &.
- Write every number, unit, and measurement as words in the reply's language, for example
  "two litres per acre" instead of "2 L/acre".
- If the problem sounds serious, say in one sentence that they should contact a local agronomist.
""".strip(),
}


class AssistantServiceError(RuntimeError):
    """Raised when an AI provider cannot produce a response."""


class AssistantProvider(Protocol):
    provider_name: str
    model_name: Optional[str]

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a reply from OpenAI-style chat messages (role + content).

        Providers may also define `stream(messages)`, an async iterator of text deltas.
        """
        ...


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self._api_key = api_key

    async def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise AssistantServiceError(
                "google-generativeai is not installed. Install requirements or use another AI_PROVIDER."
            ) from exc

        system_instruction = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        # Gemini uses "model" instead of "assistant"; consecutive same-role turns are merged.
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "model" if message["role"] == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].append(message["content"])
            else:
                contents.append({"role": role, "parts": [message["content"]]})

        def _generate() -> str:
            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(
                self.model_name, system_instruction=system_instruction or None
            )
            response = model.generate_content(contents)
            return getattr(response, "text", "") or ""

        text = await asyncio.to_thread(_generate)
        if not text.strip():
            raise AssistantServiceError("Gemini returned an empty response.")
        return text.strip()


class OllamaProvider:
    provider_name = "ollama"

    def __init__(self, base_url: str, model_name: str, client: httpx.AsyncClient) -> None:
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def generate(self, messages: list[dict[str, str]]) -> str:
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={"model": self.model_name, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        payload = response.json()

        text = (payload.get("message") or {}).get("content", "")
        if not text.strip():
            raise AssistantServiceError("Ollama returned an empty response.")
        return text.strip()


class OpenRouterProvider:
    provider_name = "openrouter"

    def __init__(
        self, api_key: str, base_url: str, model_name: str, client: httpx.AsyncClient
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def generate(self, messages: list[dict[str, str]]) -> str:
        response = await post_with_retries(
            self._client,
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        payload = response.json()

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AssistantServiceError("OpenRouter returned an unexpected response.") from exc

        if not text.strip():
            raise AssistantServiceError("OpenRouter returned an empty response.")
        return text.strip()

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield reply text deltas as OpenRouter streams them (server-sent events).

        Retries 429/5xx before the stream starts. Cancelling the consumer closes the
        connection, which tells OpenRouter to stop generating.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.3,
                    "stream": True,
                },
            ) as response:
                if is_retryable(response.status_code) and attempt < MAX_ATTEMPTS:
                    delay = retry_delay(response, attempt)
                else:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        # Skips blank lines and ": OPENROUTER PROCESSING" keep-alive comments.
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        delta = self._parse_stream_chunk(data)
                        if delta:
                            yield delta
                    return
            logger.warning(
                "Retrying OpenRouter stream status=%d attempt=%d/%d delay=%.2fs",
                response.status_code,
                attempt,
                MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

    def _parse_stream_chunk(self, data: str) -> Optional[str]:
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise AssistantServiceError("OpenRouter sent an invalid stream chunk.") from exc
        # Errors after the stream has started arrive as a chunk with a top-level "error".
        if chunk.get("error"):
            message = (chunk["error"] or {}).get("message", "unknown error")
            raise AssistantServiceError(f"OpenRouter stream failed: {message}")
        choices = chunk.get("choices") or []
        if not choices:
            return None
        return (choices[0].get("delta") or {}).get("content")


class RuleBasedProvider:
    provider_name = "rule_based"
    model_name = "local-guidance-v1"

    async def generate(self, messages: list[dict[str, str]]) -> str:
        return (
            "Based on the details provided, inspect the crop closely, isolate heavily affected "
            "plants when practical, improve airflow and drainage, avoid overhead irrigation, "
            "and remove infected leaves or fruit. If symptoms are spreading, contact a local "
            "agronomist with photos before applying chemicals. Use only registered products for "
            "your crop and region, follow label dosage, wear protective equipment, and observe "
            "pre-harvest intervals."
        )


class AgriculturalAssistantService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._provider = self._build_provider(settings, client)

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def model_name(self) -> Optional[str]:
        return self._provider.model_name

    async def answer(
        self, request: AssistantRequest, history: Optional[list[ChatMessage]] = None
    ) -> AssistantResponse:
        messages = self._build_messages(request, history or [])
        self._log_generation(request, history)
        try:
            answer = await self._provider.generate(messages)
        except httpx.HTTPError as exc:
            raise AssistantServiceError(f"AI provider request failed: {exc}") from exc

        return AssistantResponse(
            answer=answer,
            provider=self._provider.provider_name,
            model=self._provider.model_name,
        )

    async def stream_answer(
        self, request: AssistantRequest, history: Optional[list[ChatMessage]] = None
    ) -> AsyncIterator[str]:
        """Yield the reply as text deltas; providers without streaming yield it in one piece."""
        messages = self._build_messages(request, history or [])
        self._log_generation(request, history)
        stream = getattr(self._provider, "stream", None)
        try:
            if stream is None:
                yield await self._provider.generate(messages)
                return
            async for delta in stream(messages):
                yield delta
        except httpx.HTTPError as exc:
            raise AssistantServiceError(f"AI provider request failed: {exc}") from exc

    def _log_generation(
        self, request: AssistantRequest, history: Optional[list[ChatMessage]]
    ) -> None:
        logger.info(
            "Generating agricultural guidance provider=%s mode=%s crop=%s disease=%s location=%s "
            "history=%d",
            self._provider.provider_name,
            request.mode,
            request.crop,
            request.disease,
            request.location,
            len(history or []),
        )

    def _build_provider(self, settings: Settings, client: httpx.AsyncClient) -> AssistantProvider:
        if settings.ai_provider == "gemini":
            if not settings.gemini_api_key:
                raise AssistantServiceError("GEMINI_API_KEY is required when AI_PROVIDER=gemini.")
            return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        if settings.ai_provider == "ollama":
            return OllamaProvider(settings.ollama_base_url, settings.ollama_model, client)
        if settings.ai_provider == "openrouter":
            if not settings.openrouter_api_key:
                raise AssistantServiceError("OPENROUTER_API_KEY is required when AI_PROVIDER=openrouter.")
            return OpenRouterProvider(
                settings.openrouter_api_key,
                settings.openrouter_base_url,
                settings.openrouter_model,
                client,
            )
        return RuleBasedProvider()

    def _build_messages(
        self, request: AssistantRequest, history: list[ChatMessage]
    ) -> list[dict[str, str]]:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{RESPONSE_FORMATS[request.mode]}"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": message.role, "content": message.content} for message in history)
        messages.append({"role": "user", "content": self._build_user_message(request)})
        return messages

    def _build_user_message(self, request: AssistantRequest) -> str:
        context = {
            "Crop": request.crop or "Not specified",
            "Suspected or detected disease": request.disease or "Not specified",
            "Location": request.location or "Not specified",
            "Language hint": request.language,
        }
        context_lines = "\n".join(f"- {key}: {value}" for key, value in context.items())
        return f"""
Context:
{context_lines}

Farmer question:
{request.message}
""".strip()


def create_assistant_http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=settings.ai_timeout_seconds)


def get_assistant_service(connection: HTTPConnection) -> AgriculturalAssistantService:
    # The HTTP client is shared across requests; it is opened and closed in the app lifespan.
    # HTTPConnection (not Request) so the dependency also works for WebSocket routes.
    return AgriculturalAssistantService(get_settings(), connection.app.state.assistant_http)
