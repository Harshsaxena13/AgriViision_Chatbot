import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    database_path = Path(settings.database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                farmer_name TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                crop TEXT,
                disease TEXT,
                location TEXT,
                provider TEXT,
                model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
            ON messages(conversation_id, created_at)
            """
        )


async def initialize_database() -> None:
    await asyncio.to_thread(_initialize_database)


def execute_write(query: str, parameters: tuple[Any, ...]) -> None:
    with _connect() as connection:
        connection.execute(query, parameters)


def fetch_all(query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with _connect() as connection:
        return list(connection.execute(query, parameters).fetchall())


def fetch_one(query: str, parameters: tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    with _connect() as connection:
        return connection.execute(query, parameters).fetchone()
