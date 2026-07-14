# AI Agricultural Assistant

Production-ready FastAPI backend for agricultural disease guidance and farmer Q&A.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` when `DEBUG=true`.

The farmer chat UI is available at the assistant app URL, for example
`http://127.0.0.1:8001/` when your local model API is already using port `8000`.

## Endpoints

- `GET /api/v1/health`
- `POST /api/v1/assistant/ask`
- `POST /api/v1/assistant/diagnose`
- `GET /api/v1/assistant/chats/{conversation_id}`

Image diagnosis expects multipart form data:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/assistant/diagnose \
  -F "image=@/path/to/wheat-leaf.jpg"
```

Example request:

```json
{
  "message": "My tomato leaves have dark spots and yellow edges. What should I do?",
  "crop": "tomato",
  "disease": "early blight",
  "location": "Maharashtra",
  "language": "en"
}
```

Chat requests are saved to SQLite at `data/agro_assistant.db`.

The wheat disease image endpoint forwards uploaded images to your local model API by default:

`http://127.0.0.1:8000/predict`

Because your model API uses port `8000`, run this assistant app on a different port:

```bash
uvicorn main:app --host 127.0.0.1 --port 8001
```

## AI providers

Set `AI_PROVIDER` in `.env`:

- `rule_based`: local safe fallback, no external service required.
- `ollama`: calls a local Ollama server at `OLLAMA_BASE_URL`.
- `gemini`: calls Google Gemini and requires `GEMINI_API_KEY`.
- `openrouter`: calls OpenRouter and requires `OPENROUTER_API_KEY`.
