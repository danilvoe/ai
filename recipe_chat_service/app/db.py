"""Слой хранения сервиса: SQLite-схема и CRUD (users / chats / messages).

Каждый метод открывает короткоживущее соединение — это безопасно при
многопоточной обработке запросов Flask без общего соединения.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
"""

DEFAULT_CHAT_TITLE = "Новый чат"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Тонкая обёртка над SQLite для пользователей, чатов и сообщений."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        """Создаёт таблицы и индексы, если их ещё нет."""
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    # ------------------------------------------------------------------ users

    def create_user(self, username: str, password_hash: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, _now()),
            )
            user_id = int(cursor.lastrowid)
        return {"id": user_id, "username": username, "created_at": _now()}

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ chats

    def create_chat(self, user_id: int, title: str = DEFAULT_CHAT_TITLE) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chats (user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            chat_id = int(cursor.lastrowid)
        return {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def list_chats(self, user_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
        return dict(row) if row else None

    def rename_chat(self, chat_id: int, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), chat_id),
            )

    def touch_chat(self, chat_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id)
            )

    def delete_chat(self, chat_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    # --------------------------------------------------------------- messages

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (chat_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content, meta_json, now),
            )
            message_id = int(cursor.lastrowid)
        return {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "meta": meta,
            "created_at": now,
        }

    def list_messages(self, chat_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["meta"] = json.loads(item["meta"]) if item["meta"] else None
            messages.append(item)
        return messages

    def clear_messages(self, chat_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id)
            )
