"""Оркестрация чата: связывает БД, RAG-движок и HTTP-слой.

:class:`ChatService` регистрирует пользователей, управляет чатами и
обрабатывает сообщения: сохраняет историю в SQLite, передаёт последние реплики
в :class:`app.engine.RagEngine` и возвращает ответ с источниками.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from core.rag import RagSource

from .config import ServiceConfig
from .db import DEFAULT_CHAT_TITLE, Database
from .engine import RagEngine
from .security import hash_password, validate_credentials, verify_password

MAX_TITLE_LENGTH = 60
MAX_MESSAGE_LENGTH = 4000


class ChatServiceError(Exception):
    """Ошибка уровня сервиса с HTTP-кодом для API."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def compact_sources(sources: list[RagSource]) -> list[dict[str, Any]]:
    """Компактное представление источников (без полного текста чанка)."""
    return [
        {
            "rank": source.rank,
            "title": source.title,
            "section": source.section,
            "step_number": source.step_number,
            "score": round(source.score, 4),
            "recipe_id": source.recipe_id,
            "url": source.url,
        }
        for source in sources
    ]


def _derive_title(message: str) -> str:
    text = " ".join(message.strip().split())
    if not text:
        return DEFAULT_CHAT_TITLE
    if len(text) <= MAX_TITLE_LENGTH:
        return text
    return text[: MAX_TITLE_LENGTH - 1].rstrip() + "…"


class ChatService:
    """Бизнес-логика сервиса рецептов поверх :class:`Database` и :class:`RagEngine`."""

    def __init__(self, db: Database, engine: RagEngine, config: ServiceConfig) -> None:
        self.db = db
        self.engine = engine
        self.config = config

    # -------------------------------------------------------------- auth

    def register(self, username: str, password: str) -> dict[str, Any]:
        username = (username or "").strip()
        error = validate_credentials(username, password or "")
        if error:
            raise ChatServiceError(error, 400)
        if self.db.get_user_by_username(username) is not None:
            raise ChatServiceError("Пользователь с таким именем уже существует.", 409)
        try:
            return self.db.create_user(username, hash_password(password))
        except sqlite3.IntegrityError as exc:
            # Гонка одновременных регистраций с одинаковым именем.
            raise ChatServiceError(
                "Пользователь с таким именем уже существует.", 409
            ) from exc

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        user = self.db.get_user_by_username((username or "").strip())
        if user is None or not verify_password(password or "", user["password_hash"]):
            return None
        return user

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        return self.db.get_user_by_id(user_id)

    # -------------------------------------------------------------- chats

    def list_chats(self, user_id: int) -> list[dict[str, Any]]:
        return self.db.list_chats(user_id)

    def create_chat(self, user_id: int, title: str | None = None) -> dict[str, Any]:
        clean = (title or "").strip() or DEFAULT_CHAT_TITLE
        return self.db.create_chat(user_id, clean[:MAX_TITLE_LENGTH])

    def get_chat(self, user_id: int, chat_id: int) -> dict[str, Any]:
        chat = self.db.get_chat(chat_id)
        if chat is None:
            raise ChatServiceError("Чат не найден.", 404)
        if int(chat["user_id"]) != int(user_id):
            raise ChatServiceError("Доступ к чужому чату запрещён.", 403)
        return chat

    def rename_chat(self, user_id: int, chat_id: int, title: str) -> dict[str, Any]:
        chat = self.get_chat(user_id, chat_id)
        clean = (title or "").strip()
        if not clean:
            raise ChatServiceError("Название чата не может быть пустым.", 400)
        self.db.rename_chat(chat_id, clean[:MAX_TITLE_LENGTH])
        chat["title"] = clean[:MAX_TITLE_LENGTH]
        return chat

    def delete_chat(self, user_id: int, chat_id: int) -> None:
        self.get_chat(user_id, chat_id)
        self.db.delete_chat(chat_id)

    def get_messages(self, user_id: int, chat_id: int) -> list[dict[str, Any]]:
        self.get_chat(user_id, chat_id)
        return self.db.list_messages(chat_id)

    def reset_chat(self, user_id: int, chat_id: int) -> None:
        self.get_chat(user_id, chat_id)
        self.db.clear_messages(chat_id)

    # -------------------------------------------------------------- messages

    def post_message(
        self,
        user_id: int,
        chat_id: int,
        message: str,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        chat = self.get_chat(user_id, chat_id)
        text = (message or "").strip()
        if not text:
            raise ChatServiceError("Сообщение не может быть пустым.", 400)
        if len(text) > MAX_MESSAGE_LENGTH:
            raise ChatServiceError("Сообщение слишком длинное.", 400)

        history = [
            {"role": item["role"], "content": item["content"], "meta": item.get("meta")}
            for item in self.db.list_messages(chat_id)
            if item["role"] in ("user", "assistant")
        ]

        self.db.add_message(chat_id, "user", text)
        if chat["title"] == DEFAULT_CHAT_TITLE:
            self.db.rename_chat(chat_id, _derive_title(text))

        answer = self.engine.ask(text, history=history, use_rag=use_rag)
        if answer.error:
            raise ChatServiceError(f"Ошибка генерации: {answer.error}", 502)

        mode = "rag" if use_rag else "plain"
        meta = {"mode": mode, "sources": compact_sources(answer.sources)}
        self.db.add_message(chat_id, "assistant", answer.content, meta=meta)
        self.db.touch_chat(chat_id)

        return {
            "reply": answer.content,
            "mode": mode,
            "sources": [source.to_dict() for source in answer.sources],
            "latency_ms": round(answer.latency_ms, 2),
        }

    # -------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        status = self.engine.status()
        status["allow_reindex"] = self.config.allow_reindex
        return status
