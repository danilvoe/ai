"""Day 27–28: веб-чат с локальной и облачной LLM.

Модуль поднимает Flask-приложение, которое отправляет запросы в выбранную
модель: **локальную** из Дня 26 (``llama-server``, OpenAI-совместимый endpoint
``/v1``) или **облачную** (любой OpenAI-совместимый API из конфигурации).
Пользователь выбирает модель (локальная/облачная) и режим ответа (RAG/без RAG)
прямо в интерфейсе чата. Облачная модель доступна, только если в конфигурации
задан ``api_key`` и ``model``; при работе ``--offline`` облачная опция отключена.

Компоненты:

* :class:`StubLLMClient` — детерминированный офлайн-клиент, совместимый с
  :class:`agent.agent.Agent` (нужен для проверок без сборки модели и сети);
* :class:`ChatChannel` — один канал генерации (агент + опциональная RAG-система);
* :class:`ChatSession` — сессия диалога с каналами «local»/«cloud»;
* :class:`WebChatBackend` — жизненный цикл моделей и реестр сессий;
* :func:`create_app` — Flask-приложение с JSON API и встроенным HTML-чатом.

Запуск (через точку входа)::

    python3 web_local.py            # живой локальный llama-server (+ облако)
    python3 web_local.py --no-cloud # только локальная модель
    python3 web_local.py --offline  # веб-чат на детерминированном стабе
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, request

from .agent import Agent
from .context import SlidingWindowStrategy
from .conversation import Conversation, SessionStore
from .indexing import HashingEmbedder
from .llm_client import Completion, LLMClient
from .local_llm import LocalLLMConfig, LocalLLMServer, StubLocalLLM
from .local_rag import (
    DEFAULT_CATEGORY_SUGGESTIONS,
    DEFAULT_GENERIC_MIN_SOURCES,
    DEFAULT_GENERIC_THRESHOLD,
    DEFAULT_HISTORY_MAX_CHARS,
    DEFAULT_RELEVANCE_THRESHOLD,
    LocalRagSystem,
)
from .rag import RagRetriever, RagSource, load_retriever
from .tokens import ContextOverflowError

DEFAULT_WINDOW_SIZE = 8
DEFAULT_MAX_SESSIONS = 64
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 6000


class RagUnavailableError(RuntimeError):
    """Режим «рецепты» выбран, но RAG-индекс недоступен."""


class ModelSourceUnavailableError(RuntimeError):
    """Выбран источник модели («local»/«cloud»), который недоступен."""


class StubLLMClient:
    """Офлайн-клиент локальной LLM, совместимый с :class:`agent.agent.Agent`.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` dict и
    ``complete(messages, ...)``), но отвечает детерминированно через
    :class:`agent.local_llm.StubLocalLLM` — без бинарников, модели и сети.
    """

    def __init__(self, model_label: str = "local-llm-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.7,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self._stub = StubLocalLLM(model_label)

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break
        content = self._stub.answer(prompt)
        return Completion(content=content, model=self.config["model"])


@dataclass
class ChatReply:
    """Нормализованный ответ ассистента (без RAG или с найденными источниками)."""

    content: str
    model: str
    sources: list[dict] = field(default_factory=list)
    model_source: str = "local"


def _compact_sources(sources: list[RagSource]) -> list[dict]:
    """Компактное представление источников для хранения в истории сессии.

    Без полного текста чанка (``text``) — иначе файл сессии быстро растёт.
    """
    return [
        {
            "rank": s.rank,
            "title": s.title,
            "section": s.section,
            "step_number": s.step_number,
            "score": round(s.score, 4),
            "recipe_id": s.recipe_id,
            "url": s.url,
        }
        for s in sources
    ]


@dataclass
class ChatChannel:
    """Один канал генерации: обычный агент и (опционально) RAG-система.

    ``label`` — человекочитаемое имя модели, ``agent`` — обычный канал без RAG,
    ``rag_system`` — канал «рецепты» (поиск по индексу + генерация).
    """

    agent: Agent
    rag_system: LocalRagSystem | None = None
    label: str = "local"


class ChatSession:
    """Одна сессия веб-чата: диалог с моделью и его история на диске.

    Содержит каналы генерации (обычно «local» и, если облачная модель
    настроена, «cloud»). Каждый канал — это :class:`ChatChannel` с обычным
    агентом и, если индекс рецептов доступен, RAG-системой. Режим ответа
    (RAG/без RAG) и источник модели (local/cloud) выбираются на каждое
    сообщение. Все каналы используют общую
    :class:`agent.conversation.Conversation`, поэтому история диалога
    сохраняется и учитывается (в RAG — как предыдущие реплики).
    """

    def __init__(
        self,
        conversation: Conversation,
        channels: dict[str, ChatChannel] | None = None,
        history_window: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._conversation = conversation
        self._channels = channels or {}
        self._history_window = max(2, int(history_window))
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._conversation.session_id

    @property
    def messages(self) -> list[dict]:
        """История сообщений сессии (user/assistant)."""
        return self._conversation.messages

    @property
    def rag_enabled(self) -> bool:
        """Доступен ли RAG хотя бы в одном канале."""
        return any(channel.rag_system is not None for channel in self._channels.values())

    @property
    def sources(self) -> list[str]:
        """Доступные источники модели (``local``/``cloud``)."""
        return list(self._channels)

    def ask(
        self, message: str, use_rag: bool = False, model_source: str = "local"
    ) -> ChatReply:
        """Отправляет сообщение в выбранную модель и возвращает ответ.

        ``model_source`` выбирает канал («local»/«cloud»), ``use_rag=True``
        включает поиск по базе рецептов; иначе вызывается обычный агент, и
        retrieval не задействуется вовсе. Оба режима пишут диалог в общую
        историю сессии.
        """
        with self._lock:
            channel = self._channels.get(model_source)
            if channel is None:
                raise ModelSourceUnavailableError(
                    f"Источник модели «{model_source}» недоступен"
                )
            if use_rag:
                if channel.rag_system is None:
                    raise RagUnavailableError("RAG (рецепты) недоступен")
                history = self._conversation.recent_messages[-(self._history_window):]
                answer = channel.rag_system.ask_with_rag(message, history=history)
                if answer.error:
                    raise RuntimeError(answer.error)
                # Фиксируем ход в истории, чтобы он сохранился и попал в
                # контекст следующих сообщений. Источники храним компактно.
                self._conversation.append("user", message)
                self._conversation.append(
                    "assistant",
                    answer.content,
                    meta={
                        "mode": "rag",
                        "model_source": model_source,
                        "sources": _compact_sources(answer.sources),
                    },
                )
                self._conversation.trim_to(self._history_window)
                return ChatReply(
                    content=answer.content,
                    model=channel.rag_system.model_label,
                    sources=[s.to_dict() for s in answer.sources],
                    model_source=model_source,
                )
            reply = channel.agent.ask(message)
            return ChatReply(
                content=reply.content, model=reply.model, model_source=model_source
            )

    def reset(self) -> None:
        """Очищает историю диалога сессии."""
        with self._lock:
            self._conversation.clear()


class WebChatBackend:
    """Бэкенд веб-чата: локальная и облачная модели + реестр сессий диалога.

    В живом режиме поднимает ``llama-server`` (День 26) и направляет на него
    агентов; если в конфигурации задан ``api_key`` и ``model``, рядом становится
    доступен и облачный канал (любой OpenAI-совместимый API). В офлайн-режиме
    использует :class:`StubLLMClient`, поэтому приложение и его проверки
    работают без сборки, модели и сети.

    RAG-возможность (база рецептов) включена по умолчанию: при старте бэкенд
    пытается загрузить локальный FAISS-индекс, а режим («рецепты» или обычный)
    выбирается на каждое сообщение флагом ``use_rag``. Источник модели
    («local»/«cloud») выбирается параметром ``model_source``. Если индекс
    недоступен, веб-чат всё равно работает в обычном режиме, а запрос
    «рецептов» возвращает ошибку :class:`RagUnavailableError`.
    """

    def __init__(
        self,
        config: dict,
        offline: bool = False,
        model_path: str | None = None,
        port: int | None = None,
        window_size: int | None = None,
        max_sessions: int | None = None,
        history_dir=None,
        rag: bool = False,
        retriever: RagRetriever | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        client=None,
        enable_rag: bool = True,
        cloud: bool = True,
        cloud_client=None,
    ) -> None:
        self.config = config
        self.offline = offline
        self.local = LocalLLMConfig.from_config(config)
        if model_path:
            self.local.model_path = model_path
        if port:
            self.local.port = port
        section = (config or {}).get("web_chat", {}) or {}
        window = window_size if window_size is not None else section.get(
            "window_size", DEFAULT_WINDOW_SIZE
        )
        sessions = max_sessions if max_sessions is not None else section.get(
            "max_sessions", DEFAULT_MAX_SESSIONS
        )
        rag_section = (config or {}).get("local_rag", {}) or {}
        self._window_size = max(1, int(window))
        self._max_sessions = max(1, int(sessions))
        self._top_k = max(1, int(top_k if top_k is not None else rag_section.get(
            "top_k", DEFAULT_TOP_K
        )))
        self._temperature = float(
            temperature if temperature is not None else rag_section.get(
                "temperature", DEFAULT_TEMPERATURE
            )
        )
        self._max_context_chars = int(
            rag_section.get("max_context_chars", DEFAULT_MAX_CONTEXT_CHARS)
        )
        self._relevance_threshold = float(
            rag_section.get("relevance_threshold", DEFAULT_RELEVANCE_THRESHOLD)
        )
        self._history_max_chars = int(
            rag_section.get("history_max_chars", DEFAULT_HISTORY_MAX_CHARS)
        )
        self._generic_threshold = float(
            rag_section.get("generic_threshold", DEFAULT_GENERIC_THRESHOLD)
        )
        self._generic_min_sources = max(
            1, int(rag_section.get("generic_min_sources", DEFAULT_GENERIC_MIN_SOURCES))
        )
        self._category_suggestions = max(
            1, int(rag_section.get("category_suggestions", DEFAULT_CATEGORY_SUGGESTIONS))
        )
        self._retriever = retriever
        # RAG-возможность: включена по умолчанию (`enable_rag`), флаг `rag`
        # оставлен для обратной совместимости, явный retriever тоже включает её.
        self._rag_capability = bool(enable_rag) or bool(rag) or retriever is not None
        self._rag_error: str | None = None
        self._store = (
            SessionStore(Path(history_dir)) if history_dir else SessionStore()
        )
        self._server: LocalLLMServer | None = None
        self._client = client
        self._injected_client = client is not None
        self._max_context_tokens: int | None = None
        # Облачный канал: включается флагом `cloud` и наличием api_key/model в
        # конфигурации. Явно переданный `cloud_client` включает его всегда
        # (используется в офлайн-проверках).
        self._cloud_capability = bool(cloud) or cloud_client is not None
        self._cloud_client = cloud_client
        self._injected_cloud = cloud_client is not None
        _cloud_ctx = (config or {}).get("max_context_tokens")
        self._cloud_max_context_tokens: int | None = (
            int(_cloud_ctx) if _cloud_ctx is not None else None
        )
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        """Режим работы: ``offline`` (стаб) или ``live`` (llama-server)."""
        return "offline" if self.offline else "live"

    @property
    def ready(self) -> bool:
        """Готов ли бэкенд принимать запросы."""
        return self._client is not None

    @property
    def rag_enabled(self) -> bool:
        """Доступен ли RAG-режим (индекс рецептов загружен)."""
        return self._retriever is not None

    @property
    def rag_error(self) -> str | None:
        """Причина недоступности RAG (если индекс не загрузился)."""
        return self._rag_error

    @property
    def model_label(self) -> str:
        return self.local.model_label

    @property
    def cloud_enabled(self) -> bool:
        """Доступен ли облачный канал (клиент настроен)."""
        return self._cloud_client is not None

    @property
    def cloud_model_label(self) -> str:
        """Метка облачной модели из конфигурации (пусто, если не задана)."""
        return str((self.config or {}).get("model", "") or "")

    @property
    def model_sources(self) -> list[str]:
        """Доступные источники модели в порядке отображения."""
        sources = ["local"]
        if self.cloud_enabled:
            sources.append("cloud")
        return sources

    @property
    def endpoint(self) -> str:
        """Адрес модели, к которому обращается веб-чат."""
        if self.offline:
            return "stub://local-llm"
        return f"http://{self.local.host}:{self.local.port}/v1"

    def _load_retriever(self) -> RagRetriever:
        """Загружает локальный FAISS-индекс рецептов из конфигурации."""
        rag_section = (self.config or {}).get("local_rag", {}) or {}
        rag_section = {**((self.config or {}).get("rag", {}) or {}), **rag_section}
        idx_section = (self.config or {}).get("indexing", {}) or {}
        embedder = HashingEmbedder(dim=int(idx_section.get("embedding_dim", 256)))
        return load_retriever(
            index_dir=idx_section.get("index_dir", "history/index"),
            dataset=rag_section.get("dataset", "recipt_all"),
            strategy=rag_section.get("strategy", "structural"),
            embedder=embedder,
            top_k=self._top_k,
        )

    def _build_cloud_client(self):
        """Создаёт облачный клиент, если он настроен (иначе None).

        В офлайн-режиме облако не поднимается: без сети облачный API недоступен.
        """
        if not self._cloud_capability or self.offline:
            return None
        if not (self.config or {}).get("api_key"):
            return None
        if not (self.config or {}).get("model"):
            return None
        return LLMClient(self.config)

    def start(self) -> WebChatBackend:
        """Готовит модели: поднимает llama-server, стаб и/или облачный клиент."""
        if self._injected_client:
            self._max_context_tokens = None
        elif self.offline:
            self._client = StubLLMClient(f"{self.local.model_label}-stub")
            self._max_context_tokens = None
        else:
            self._server = LocalLLMServer(self.local)
            self._server.start()
            self._client = self._server.build_client()
            self._max_context_tokens = self._server.max_context_tokens
        if not self._injected_cloud:
            self._cloud_client = self._build_cloud_client()
        if self._rag_capability and self._retriever is None:
            try:
                self._retriever = self._load_retriever()
                self._rag_error = None
            except Exception as exc:  # noqa: BLE001 — без RAG чат должен работать
                self._retriever = None
                self._rag_error = str(exc)
        return self

    def stop(self) -> None:
        """Останавливает локальный сервер, если он был запущен."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def status(self) -> dict:
        """Сводка состояния бэкенда для ``GET /api/status``."""
        result = {
            "ready": self.ready,
            "mode": self.mode,
            "model": self.model_label,
            "local_model": self.model_label,
            "cloud": self.cloud_enabled,
            "cloud_model": self.cloud_model_label if self.cloud_enabled else None,
            "sources": self.model_sources,
            "endpoint": self.endpoint,
            "sessions": len(self._sessions),
            "rag": self.rag_enabled,
        }
        if self.rag_enabled:
            result["dataset"] = self._retriever.index.metadata.get("source", "unknown")
            result["vectors"] = self._retriever.total_vectors
            result["top_k"] = self._top_k
        elif self._rag_error:
            result["rag_error"] = self._rag_error
        return result

    def _build_channel(
        self,
        conversation: Conversation,
        client,
        label: str,
        max_context_tokens: int | None,
    ) -> ChatChannel:
        """Собирает канал генерации (обычный агент + RAG, если индекс доступен)."""
        agent = Agent(
            client,
            conversation,
            max_context_tokens=max_context_tokens,
            context_strategy=SlidingWindowStrategy(size=self._window_size),
        )
        rag_system = None
        if self._retriever is not None:
            rag_system = LocalRagSystem(
                client=client,
                retriever=self._retriever,
                top_k=self._top_k,
                temperature=self._temperature,
                max_context_chars=self._max_context_chars,
                model_label=str(client.config.get("model", label)),
                relevance_threshold=self._relevance_threshold,
                history_max_messages=self._window_size,
                history_max_chars=self._history_max_chars,
                generic_threshold=self._generic_threshold,
                generic_min_sources=self._generic_min_sources,
                category_suggestions=self._category_suggestions,
            )
        return ChatChannel(agent=agent, rag_system=rag_system, label=label)

    def _build_session(self, session_id: str | None) -> ChatSession:
        conversation = (
            self._store.get(session_id) if session_id else self._store.create()
        )
        channels: dict[str, ChatChannel] = {}
        if self._client is not None:
            channels["local"] = self._build_channel(
                conversation, self._client, self.model_label, self._max_context_tokens
            )
        if self._cloud_client is not None:
            channels["cloud"] = self._build_channel(
                conversation,
                self._cloud_client,
                self.cloud_model_label or "cloud",
                self._cloud_max_context_tokens,
            )
        return ChatSession(
            conversation,
            channels=channels,
            history_window=self._window_size,
        )

    def _get_session(self, session_id: str | None) -> ChatSession:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            session = self._build_session(session_id)
            if (
                session.session_id not in self._sessions
                and len(self._sessions) >= self._max_sessions
            ):
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            self._sessions[session.session_id] = session
            return session

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        use_rag: bool = False,
        model_source: str = "local",
    ) -> dict:
        """Отправляет сообщение в выбранную модель и возвращает ответ.

        ``model_source`` выбирает источник модели («local»/«cloud»);
        ``use_rag=True`` включает поиск по базе рецептов (режим «рецепты»). Если
        источник или RAG недоступны, поднимается
        :class:`ModelSourceUnavailableError` / :class:`RagUnavailableError`.
        Возвращает словарь
        ``{session_id, reply, model, model_source, sources, used_rag, rag_available}``.
        """
        if not self.ready:
            raise RuntimeError("Бэкенд не запущен: вызовите start() перед запросами.")
        if model_source not in self.model_sources:
            raise ModelSourceUnavailableError(
                f"Источник модели «{model_source}» недоступен"
            )
        if use_rag and not self.rag_enabled:
            reason = self._rag_error or "индекс рецептов не загружен"
            raise RagUnavailableError(f"RAG (рецепты) недоступен: {reason}")
        session = self._get_session(session_id)
        reply = session.ask(message, use_rag=use_rag, model_source=model_source)
        return {
            "session_id": session.session_id,
            "reply": reply.content,
            "model": reply.model,
            "model_source": reply.model_source,
            "sources": reply.sources,
            "used_rag": bool(use_rag and self.rag_enabled),
            "rag_available": self.rag_enabled,
        }

    def reset(self, session_id: str | None) -> str:
        """Сбрасывает историю сессии и возвращает её id."""
        session = self._get_session(session_id)
        session.reset()
        return session.session_id

    def history(self, session_id: str | None) -> list[dict]:
        """История сообщений сессии."""
        return self._get_session(session_id).messages


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Локальный LLM-чат</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #0f1117; color: #e6e8ee;
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid #232734;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .status { font-size: 12px; color: #8b93a7; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #f0a020; margin-right: 6px; vertical-align: middle; }
  .dot.ok { background: #38b26a; }
  main { flex: 1; overflow-y: auto; padding: 18px; }
  .msg { max-width: 760px; margin: 0 auto 14px; display: flex; }
  .msg .bubble {
    padding: 10px 14px; border-radius: 12px; white-space: pre-wrap;
    word-wrap: break-word; max-width: 85%;
  }
  .msg.user { justify-content: flex-end; }
  .msg.user .bubble { background: #2b5cff; color: #fff; border-bottom-right-radius: 3px; }
  .msg.assistant .bubble { background: #1b1f2a; border: 1px solid #232734;
                           border-bottom-left-radius: 3px; }
  .msg.system .bubble { background: #2a2010; border: 1px solid #4a3a12; color: #f0c060; }
  .rbadge { display: inline-block; font-size: 11px; line-height: 1.6; padding: 0 7px;
            border-radius: 6px; margin-right: 6px; font-weight: 600; }
  .rbadge.rag { background: #243a2a; color: #7fd39a; border: 1px solid #2f5138; }
  .rbadge.plain { background: #2a2f3a; color: #9aa3b8; border: 1px solid #39404f; }
  .rbadge.src-local { background: #1e2a3a; color: #7fa8e0; border: 1px solid #2c3e56; }
  .rbadge.src-cloud { background: #2f2338; color: #cf9de0; border: 1px solid #4a2f56; }
  .sources { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #2c3140;
             font-size: 12px; color: #8b93a7; white-space: normal; }
  .sources .stitle { font-weight: 600; color: #aab2c8; margin-bottom: 4px; }
  .sources .sline { margin: 2px 0; overflow-wrap: anywhere; }
  .sources a { color: #6ea8ff; text-decoration: none; }
  .sources a:hover { text-decoration: underline; }
  footer { border-top: 1px solid #232734; padding: 12px 18px; }
  form { max-width: 760px; margin: 0 auto; display: flex; gap: 10px; }
  textarea {
    flex: 1; resize: none; height: 46px; padding: 12px; border-radius: 10px;
    border: 1px solid #2c3140; background: #151924; color: inherit; font: inherit;
  }
  textarea:focus { outline: none; border-color: #2b5cff; }
  select.mode {
    height: 46px; padding: 0 10px; border-radius: 10px;
    border: 1px solid #2c3140; background: #151924; color: inherit; font: inherit;
  }
  select.mode:focus { outline: none; border-color: #2b5cff; }
  select.mode:disabled { opacity: .5; }
  button {
    padding: 0 18px; border: 0; border-radius: 10px; cursor: pointer;
    background: #2b5cff; color: #fff; font: inherit; font-weight: 600;
  }
  button.ghost { background: #1b1f2a; border: 1px solid #2c3140; font-weight: 400; }
  button:disabled { opacity: .5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>LLM-чат: локальная и облачная модель</h1>
  <span class="status"><span id="dot" class="dot"></span><span id="status">подключение…</span></span>
  <span style="flex:1"></span>
  <button class="ghost" id="reset" type="button">Очистить</button>
</header>
<main id="log"></main>
<footer>
  <form id="form">
    <select id="source" class="mode" title="Модель">
      <option value="local" selected>локальная</option>
      <option value="cloud">облачная</option>
    </select>
    <select id="mode" class="mode" title="Режим ответа">
      <option value="off">не использовать RAG</option>
      <option value="rag" selected>рецепты</option>
    </select>
    <textarea id="input" placeholder="Спросите модель…" autofocus></textarea>
    <button id="send" type="submit">Отправить</button>
  </form>
</footer>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const resetBtn = document.getElementById('reset');
  const statusEl = document.getElementById('status');
  const dot = document.getElementById('dot');
  const modeEl = document.getElementById('mode');
  const sourceEl = document.getElementById('source');
  const HISTORY_KEY = 'llm_chat_session';
  const MODE_KEY = 'llm_chat_mode';
  const SOURCE_KEY = 'llm_chat_source';
  let sessionId = localStorage.getItem(HISTORY_KEY);

  function add(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  function makeBadge(cls, text) {
    const b = document.createElement('span');
    b.className = 'rbadge ' + cls;
    b.textContent = text;
    return b;
  }

  function setBadge(bubble, mode, source) {
    const frag = document.createDocumentFragment();
    const isCloud = source === 'cloud';
    frag.appendChild(makeBadge(isCloud ? 'src-cloud' : 'src-local',
                               isCloud ? 'облачная' : 'локальная'));
    frag.appendChild(makeBadge(mode === 'rag' ? 'rag' : 'plain',
                               mode === 'rag' ? 'рецепты' : 'без RAG'));
    bubble.prepend(frag);
  }

  function renderSources(bubble, sources) {
    if (!sources || !sources.length) return;
    const box = document.createElement('div');
    box.className = 'sources';
    const title = document.createElement('div');
    title.className = 'stitle';
    title.textContent = 'Источники (' + sources.length + '):';
    box.appendChild(title);
    sources.forEach((s) => {
      const line = document.createElement('div');
      line.className = 'sline';
      let label = '#' + s.rank + ' «' + (s.title || '—') + '»';
      const sec = s.section ? ' [' + s.section + (s.step_number ? ' #' + s.step_number : '') + ']' : '';
      label += sec + ' · сходство ' + s.score;
      if (s.url) {
        const a = document.createElement('a');
        a.href = s.url; a.target = '_blank'; a.rel = 'noopener';
        a.textContent = label;
        line.appendChild(a);
      } else {
        line.textContent = label;
      }
      box.appendChild(line);
    });
    bubble.appendChild(box);
  }

  async function loadHistory() {
    if (!sessionId) return;
    try {
      const r = await fetch('api/history?session_id=' + encodeURIComponent(sessionId));
      const data = await r.json();
      const messages = data.messages || [];
      if (!messages.length) return;
      add('system', 'История диалога восстановлена.');
      messages.forEach((m) => {
        const bubble = add(m.role === 'user' ? 'user' : 'assistant', m.content);
        if (m.role !== 'user') {
          const meta = m.meta || {};
          setBadge(bubble, meta.mode === 'rag' ? 'rag' : 'plain',
                   meta.model_source || 'local');
          renderSources(bubble, meta.sources || []);
        }
      });
    } catch (e) {
      // История необязательна: при сбое начинаем с чистого листа.
    }
  }

  async function loadStatus() {
    try {
      const r = await fetch('api/status');
      const s = await r.json();
      const ragNote = s.rag ? ' · RAG доступен' : ' · RAG недоступен';
      const cloudNote = s.cloud ? ' · облако доступно' : ' · облако недоступно';
      statusEl.textContent = s.mode + ragNote + cloudNote + ' · ' + s.model;
      dot.className = 'dot' + (s.ready ? ' ok' : '');
      // Облачная опция осмысленна только при настроенном облачном API.
      const cloudOpt = sourceEl.querySelector('option[value="cloud"]');
      if (cloudOpt) cloudOpt.disabled = !s.cloud;
      sourceEl.title = s.cloud
        ? 'Модель (облачная: ' + (s.cloud_model || '—') + ')'
        : 'Облачная модель недоступна (нет api_key/model)';
      // Источник по умолчанию: сохранённый в localStorage, иначе «локальная».
      let source = localStorage.getItem(SOURCE_KEY);
      if (!source) source = 'local';
      if (source === 'cloud' && !s.cloud) source = 'local';
      sourceEl.value = source;
      // «рецепты» осмысленны только при доступном индексе.
      modeEl.title = s.rag
        ? 'Режим ответа (RAG-база: ' + (s.dataset || 'рецепты') + ')'
        : 'RAG недоступен: ' + (s.rag_error || 'индекс не загружен');
      // Режим по умолчанию: сохранённый в localStorage, иначе «рецепты»
      // (если RAG доступен) — иначе «не использовать RAG».
      let mode = localStorage.getItem(MODE_KEY);
      if (!mode) mode = s.rag ? 'rag' : 'off';
      if (mode === 'rag' && !s.rag) mode = 'off';
      modeEl.value = mode;
    } catch (e) {
      statusEl.textContent = 'нет связи';
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    add('user', text);
    input.value = '';
    send.disabled = true;
    const pending = add('assistant', '…');
    try {
      const r = await fetch('api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          use_rag: modeEl.value === 'rag',
          model_source: sourceEl.value,
        }),
      });
      const data = await r.json();
      if (!r.ok) {
        pending.parentElement.className = 'msg system';
        pending.textContent = '⚠️ ' + (data.error || ('HTTP ' + r.status));
      } else {
        sessionId = data.session_id;
        localStorage.setItem(HISTORY_KEY, sessionId);
        pending.textContent = data.reply;
        setBadge(pending, data.used_rag ? 'rag' : 'plain',
                 data.model_source || sourceEl.value);
        renderSources(pending, data.sources);
      }
    } catch (err) {
      pending.parentElement.className = 'msg system';
      pending.textContent = '⚠️ Ошибка соединения: ' + err;
    } finally {
      send.disabled = false;
      input.focus();
    }
  });

  resetBtn.addEventListener('click', async () => {
    await fetch('api/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId}),
    });
    log.innerHTML = '';
    sessionId = null;
    localStorage.removeItem(HISTORY_KEY);
    add('system', 'История очищена.');
  });

  modeEl.addEventListener('change', () => {
    localStorage.setItem(MODE_KEY, modeEl.value);
  });

  sourceEl.addEventListener('change', () => {
    localStorage.setItem(SOURCE_KEY, sourceEl.value);
  });

  loadStatus();
  loadHistory();
</script>
</body>
</html>
"""


def create_app(backend: WebChatBackend) -> Flask:
    """Создаёт Flask-приложение поверх готового :class:`WebChatBackend`."""
    app = Flask(__name__)
    # Кириллица в JSON-ответах должна быть читаемой, а не \uXXXX.
    app.json.ensure_ascii = False

    @app.get("/")
    def index() -> Response:
        return Response(HTML_PAGE, mimetype="text/html; charset=utf-8")

    @app.get("/api/status")
    def status():
        return jsonify(backend.status())

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Пустое сообщение"}), 400
        session_id = data.get("session_id")
        use_rag = bool(data.get("use_rag", False))
        model_source = str(data.get("model_source", "local") or "local")
        if model_source not in backend.model_sources:
            return (
                jsonify(
                    {
                        "error": f"Источник модели «{model_source}» недоступен",
                        "session_id": session_id,
                    }
                ),
                400,
            )
        if use_rag and not backend.rag_enabled:
            reason = backend.rag_error or "индекс рецептов не загружен"
            return (
                jsonify(
                    {
                        "error": f"RAG (рецепты) недоступен: {reason}",
                        "session_id": session_id,
                    }
                ),
                400,
            )
        try:
            result = backend.chat(
                message, session_id, use_rag=use_rag, model_source=model_source
            )
        except ContextOverflowError:
            return (
                jsonify(
                    {
                        "error": "Контекст переполнен. Очистите историю.",
                        "session_id": session_id,
                    }
                ),
                409,
            )
        except (RagUnavailableError, ModelSourceUnavailableError) as exc:
            return jsonify({"error": str(exc), "session_id": session_id}), 400
        except Exception as exc:  # noqa: BLE001 — понятная ошибка модели
            return jsonify({"error": f"Ошибка модели: {exc}"}), 502
        return jsonify(result)

    @app.post("/api/reset")
    def reset():
        data = request.get_json(silent=True) or {}
        session_id = backend.reset(data.get("session_id"))
        return jsonify({"session_id": session_id, "ok": True})

    @app.get("/api/history")
    def history():
        session_id = request.args.get("session_id")
        return jsonify({"session_id": session_id, "messages": backend.history(session_id)})

    return app
