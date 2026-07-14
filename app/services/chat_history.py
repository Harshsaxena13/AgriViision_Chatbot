import asyncio
from typing import Optional
from uuid import uuid4

from app.db.sqlite import execute_write, fetch_all, fetch_one
from app.models.schemas import ChatMessage, ChatSession


class ChatHistoryService:
    async def ensure_conversation(
        self, conversation_id: Optional[str], farmer_name: Optional[str]
    ) -> str:
        if conversation_id:
            row = await asyncio.to_thread(
                fetch_one, "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            )
            if row:
                return conversation_id

        new_id = conversation_id or str(uuid4())
        await asyncio.to_thread(
            execute_write,
            """
            INSERT OR IGNORE INTO conversations (id, farmer_name)
            VALUES (?, ?)
            """,
            (new_id, farmer_name),
        )
        return new_id

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        crop: Optional[str] = None,
        disease: Optional[str] = None,
        location: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        await asyncio.to_thread(
            execute_write,
            """
            INSERT INTO messages (
                id, conversation_id, role, content, crop, disease, location, provider, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                conversation_id,
                role,
                content,
                crop,
                disease,
                location,
                provider,
                model,
            ),
        )
        await asyncio.to_thread(
            execute_write,
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )

    async def get_session(self, conversation_id: str) -> ChatSession:
        rows = await asyncio.to_thread(
            fetch_all,
            """
            SELECT role, content, crop, disease, location, provider, model, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        return ChatSession(
            conversation_id=conversation_id,
            messages=[ChatMessage(**dict(row)) for row in rows],
        )


def get_chat_history_service() -> ChatHistoryService:
    return ChatHistoryService()
