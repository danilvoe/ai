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
from .optimization import (
    GenerationParams,
    ModelVariant,
    PROMPT_TEMPLATES,
    find_template,
    find_variant,
    parse_presets,
    parse_variants,
)
from .rag import RagRetriever, RagSource, load_retriever
from .tokens import ContextOverflowError

DEFAULT_WINDOW_SIZE = 8
DEFAULT_MAX_SESSIONS = 64
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 6000


def _clamp(value: float, low: float, high: float) -> float:
    """Ограничивает значение диапазоном ``[low, high]``."""
    return max(low, min(high, float(value)))


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
        self,
        message: str,
        use_rag: bool = False,
        model_source: str = "local",
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        system_prompt: str | None = None,
    ) -> ChatReply:
        """Отправляет сообщение в выбранную модель и возвращает ответ.

        ``model_source`` выбирает канал («local»/«cloud»), ``use_rag=True``
        включает поиск по базе рецептов; иначе вызывается обычный агент, и
        retrieval не задействуется вовсе. ``temperature``/``max_tokens``/
        ``context_window`` (День 29) переопределяют параметры генерации на один
        запрос, ``system_prompt`` — prompt-шаблон под кейс. Оба режима пишут
        диалог в общую историю сессии.
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
                answer = channel.rag_system.ask_with_rag(
                    message,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_window=context_window,
                    system_prompt=system_prompt,
                )
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
            reply = channel.agent.ask(
                message,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window,
            )
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
        # День 29: параметры оптимизации — пресеты, варианты (квантование) и
        # prompt-шаблоны. Текущий выбор хранится как состояние бэкенда и
        # переопределяется на каждое сообщение из веб-интерфейса.
        opt_section = (config or {}).get("optimization", {}) or {}
        self._presets = parse_presets(config)
        self._variants = parse_variants(config)
        self._templates = dict(PROMPT_TEMPLATES)
        default_preset = str(opt_section.get("default_preset", "") or "")
        preset = self._presets.get(default_preset) if default_preset else None
        self._params = preset or GenerationParams(
            temperature=self._temperature,
            max_tokens=self.local.max_response_tokens,
            context_window=self._window_size,
        )
        self._preset_name = default_preset if preset else "custom"
        self._current_variant = self._match_variant(self.local.model_path) or find_variant(
            self._variants, None
        )
        self._template = find_template(
            opt_section.get("default_prompt_template", "baseline")
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
    def variants(self) -> list[ModelVariant]:
        """Доступные варианты локальной модели (квантования)."""
        return list(self._variants)

    @property
    def presets(self) -> dict[str, GenerationParams]:
        """Именованные пресеты параметров генерации."""
        return dict(self._presets)

    @property
    def prompt_templates(self) -> dict:
        """Доступные prompt-шаблоны (имя → :class:`PromptTemplate`)."""
        return dict(self._templates)

    @property
    def local_variant_label(self) -> str:
        """Метка выбранного варианта локальной модели (квантования)."""
        return self._current_variant.label if self._current_variant else ""

    @property
    def prompt_template_label(self) -> str:
        """Имя активного prompt-шаблона."""
        return self._template.name

    @property
    def generation_params(self) -> GenerationParams:
        """Текущие параметры генерации по умолчанию."""
        return self._params

    @property
    def ctx_size(self) -> int:
        """Размер контекста llama-server (``-c``)."""
        return int(self.local.ctx_size)

    def _match_variant(self, model_path: str | None) -> ModelVariant | None:
        """Находит вариант модели по пути (полному или по имени файла)."""
        if not model_path:
            return None
        norm = str(model_path).replace("\\", "/")
        for variant in self._variants:
            if variant.model_path.replace("\\", "/") == norm:
                return variant
        target_name = Path(norm).name
        for variant in self._variants:
            if variant.path.name == target_name:
                return variant
        return None

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
            # День 29: состояние оптимизации для веб-интерфейса.
            "ctx_size": self.ctx_size,
            "threads": int(self.local.threads),
            "local_variant": self.local_variant_label,
            "prompt_template": self.prompt_template_label,
            "generation": self._params.to_dict(),
            "preset": self._preset_name,
            "presets": {
                name: params.to_dict() for name, params in self._presets.items()
            },
            "variants": [v.to_dict() for v in self._variants],
            "prompt_templates": {
                name: template.to_dict() for name, template in self._templates.items()
            },
        }
        if self.rag_enabled:
            result["dataset"] = self._retriever.index.metadata.get("source", "unknown")
            result["vectors"] = self._retriever.total_vectors
            result["top_k"] = self._top_k
        elif self._rag_error:
            result["rag_error"] = self._rag_error
        return result

    def configure(
        self,
        ctx_size: int | None = None,
        variant: str | None = None,
    ) -> dict:
        """Применяет настройки уровня сервера (День 29).

        ``ctx_size`` меняет размер контекста llama-server (``-c``), ``variant`` —
        выбранное квантование (GGUF-модель). В живом режиме локальный сервер
        перезапускается с новыми параметрами, а сессии пересобираются на новом
        клиенте (история диалога при этом сохраняется на диске). В офлайне
        обновляются только метки.
        """
        changed = False
        if variant is not None:
            matched = find_variant(self._variants, variant)
            if matched is None:
                raise ValueError(f"Неизвестный вариант модели: {variant}")
            if not matched.available() and not self.offline and not self._injected_client:
                raise ValueError(
                    f"Файл модели варианта «{matched.label}» не найден: {matched.path}"
                )
            self._current_variant = matched
            if not self._injected_client and matched.model_path:
                self.local.model_path = matched.model_path
            if matched.ctx_size and ctx_size is None:
                self.local.ctx_size = int(matched.ctx_size)
            changed = True
        if ctx_size is not None:
            self.local.ctx_size = max(256, int(ctx_size))
            changed = True

        if changed and not self.offline and not self._injected_client:
            self._restart_local_server()
        elif changed and self.offline and not self._injected_client:
            self._client = StubLLMClient(f"{self.local.model_label}-stub")
        return self.status()

    def _restart_local_server(self) -> None:
        """Перезапускает llama-server с текущими настройками и пересобирает сессии."""
        self.stop()
        self._server = LocalLLMServer(self.local)
        self._server.start()
        self._client = self._server.build_client()
        self._max_context_tokens = self._server.max_context_tokens
        with self._lock:
            self._sessions.clear()

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
                # День 29: шаблон под кейс и лимит токенов по умолчанию.
                system_prompt=self._template.system_prompt,
                max_tokens=self._params.max_tokens,
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        preset: str | None = None,
        prompt_template: str | None = None,
    ) -> dict:
        """Отправляет сообщение в выбранную модель и возвращает ответ.

        ``model_source`` выбирает источник модели («local»/«cloud»);
        ``use_rag=True`` включает поиск по базе рецептов (режим «рецепты»). Если
        источник или RAG недоступны, поднимается
        :class:`ModelSourceUnavailableError` / :class:`RagUnavailableError`.

        День 29: ``preset`` выбирает именованный пресет параметров, а
        ``temperature``/``max_tokens``/``context_window`` переопределяют отдельные
        параметры поверх него; ``prompt_template`` выбирает prompt-шаблон под
        кейс. Возвращает словарь
        ``{session_id, reply, model, model_source, sources, used_rag,
        rag_available, generation, prompt_template}``.
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

        params = self._presets.get(preset) if preset else self._params
        if params is None:
            params = self._params
        if temperature is not None or max_tokens is not None or context_window is not None:
            params = GenerationParams(
                temperature=(
                    params.temperature if temperature is None else _clamp(temperature, 0.0, 2.0)
                ),
                max_tokens=(
                    params.max_tokens
                    if max_tokens is None
                    else max(1, int(max_tokens))
                ),
                context_window=(
                    params.context_window
                    if context_window is None
                    else max(1, int(context_window))
                ),
            )
        template = (
            self._templates.get(prompt_template, self._template)
            if prompt_template
            else self._template
        )

        session = self._get_session(session_id)
        reply = session.ask(
            message,
            use_rag=use_rag,
            model_source=model_source,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            context_window=params.context_window,
            system_prompt=template.system_prompt,
        )
        return {
            "session_id": session.session_id,
            "reply": reply.content,
            "model": reply.model,
            "model_source": reply.model_source,
            "sources": reply.sources,
            "used_rag": bool(use_rag and self.rag_enabled),
            "rag_available": self.rag_enabled,
            "generation": params.to_dict(),
            "prompt_template": template.name,
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
  .opts { border-bottom: 1px solid #232734; padding: 8px 18px;
          font-size: 13px; color: #aab2c8; }
  .opts summary { cursor: pointer; color: #8b93a7; font-size: 13px; }
  .opts .row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end;
               margin-top: 10px; }
  .opts label { display: flex; flex-direction: column; gap: 3px;
                font-size: 11px; color: #8b93a7; }
  .opts input, .opts select {
    height: 30px; padding: 0 8px; border-radius: 8px; border: 1px solid #2c3140;
    background: #151924; color: inherit; font: inherit;
  }
  .opts input:focus, .opts select:focus { outline: none; border-color: #2b5cff; }
  .opts button { height: 30px; padding: 0 12px; font-size: 12px; }
  .opts .hint { color: #6b7285; font-size: 11px; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h1>LLM-чат: локальная и облачная модель</h1>
  <span class="status"><span id="dot" class="dot"></span><span id="status">подключение…</span></span>
  <span style="flex:1"></span>
  <button class="ghost" id="reset" type="button">Очистить</button>
</header>
<details class="opts" id="opt-panel">
  <summary>Параметры оптимизации локальной LLM (День 29)</summary>
  <div class="row">
    <label>Модель (квантование)
      <select id="variant"></select>
    </label>
    <label>Пресет
      <select id="preset"><option value="">свой</option></select>
    </label>
    <label>Prompt-шаблон
      <select id="template">
        <option value="baseline">базовый</option>
        <option value="recipe_structured">структурированный рецепт</option>
      </select>
    </label>
    <label>Temperature
      <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.2">
    </label>
    <label>Max tokens
      <input id="max_tokens" type="number" min="1" step="64" value="512">
    </label>
    <label>Окно контекста (сообщений)
      <input id="context_window" type="number" min="1" step="1" value="8">
    </label>
    <label>Ctx сервера (-c)
      <input id="ctx_size" type="number" min="256" step="256" value="8192">
    </label>
    <button class="ghost" id="apply-settings" type="button">Применить модель/ctx</button>
  </div>
  <div class="hint">Модель и размер контекста применяются перезапуском
    llama-server; temperature, max tokens, окно и шаблон — на каждое сообщение.</div>
</details>
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
  const variantEl = document.getElementById('variant');
  const presetEl = document.getElementById('preset');
  const templateEl = document.getElementById('template');
  const temperatureEl = document.getElementById('temperature');
  const maxTokensEl = document.getElementById('max_tokens');
  const contextWindowEl = document.getElementById('context_window');
  const ctxSizeEl = document.getElementById('ctx_size');
  const applyBtn = document.getElementById('apply-settings');
  const HISTORY_KEY = 'llm_chat_session';
  const MODE_KEY = 'llm_chat_mode';
  const SOURCE_KEY = 'llm_chat_source';
  const VARIANT_KEY = 'llm_chat_variant';
  const PRESET_KEY = 'llm_chat_preset';
  const TEMPLATE_KEY = 'llm_chat_template';
  const TEMPERATURE_KEY = 'llm_chat_temperature';
  const MAXTOK_KEY = 'llm_chat_max_tokens';
  const CTXWIN_KEY = 'llm_chat_context_window';
  const CTXSIZE_KEY = 'llm_chat_ctx_size';
  let sessionId = localStorage.getItem(HISTORY_KEY);
  let PRESETS = {};

  function optParams() {
    return {
      temperature: parseFloat(temperatureEl.value),
      max_tokens: parseInt(maxTokensEl.value, 10) || undefined,
      context_window: parseInt(contextWindowEl.value, 10) || undefined,
      preset: presetEl.value || undefined,
      prompt_template: templateEl.value || undefined,
    };
  }

  function persistOpts() {
    localStorage.setItem(TEMPERATURE_KEY, temperatureEl.value);
    localStorage.setItem(MAXTOK_KEY, maxTokensEl.value);
    localStorage.setItem(CTXWIN_KEY, contextWindowEl.value);
    localStorage.setItem(CTXSIZE_KEY, ctxSizeEl.value);
    localStorage.setItem(VARIANT_KEY, variantEl.value);
    localStorage.setItem(PRESET_KEY, presetEl.value);
    localStorage.setItem(TEMPLATE_KEY, templateEl.value);
  }

  function restoreOpts() {
    [[temperatureEl, TEMPERATURE_KEY], [maxTokensEl, MAXTOK_KEY],
     [contextWindowEl, CTXWIN_KEY], [ctxSizeEl, CTXSIZE_KEY]].forEach(([el, key]) => {
      const v = localStorage.getItem(key);
      if (v !== null) el.value = v;
    });
    const tmpl = localStorage.getItem(TEMPLATE_KEY);
    if (tmpl) templateEl.value = tmpl;
  }

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

      // День 29: варианты модели (квантование), пресеты и prompt-шаблоны.
      PRESETS = s.presets || {};
      variantEl.innerHTML = '';
      (s.variants || []).forEach((v) => {
        const o = document.createElement('option');
        o.value = v.label;
        o.textContent = v.label + ' · ' + (v.size_mb ? v.size_mb + ' МБ' : '—')
          + (v.available ? '' : ' (нет файла)');
        variantEl.appendChild(o);
      });
      let savedVariant = localStorage.getItem(VARIANT_KEY);
      if (savedVariant && [...variantEl.options].some((o) => o.value === savedVariant)) {
        variantEl.value = savedVariant;
      } else if (s.local_variant) {
        variantEl.value = s.local_variant;
      }
      presetEl.innerHTML = '<option value="">свой</option>';
      Object.entries(PRESETS).forEach(([name, p]) => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = name + ' (t=' + p.temperature + ', ' + p.max_tokens + ' ток.)';
        presetEl.appendChild(o);
      });
      const savedPreset = localStorage.getItem(PRESET_KEY);
      if (savedPreset) presetEl.value = savedPreset;
      else if (s.preset && s.preset !== 'custom') presetEl.value = s.preset;

      restoreOpts();
      if (localStorage.getItem(TEMPERATURE_KEY) === null && s.generation) {
        temperatureEl.value = s.generation.temperature;
        maxTokensEl.value = s.generation.max_tokens;
        contextWindowEl.value = s.generation.context_window;
      }
      if (localStorage.getItem(CTXSIZE_KEY) === null && s.ctx_size) {
        ctxSizeEl.value = s.ctx_size;
      }
      if (localStorage.getItem(TEMPLATE_KEY) === null && s.prompt_template) {
        templateEl.value = s.prompt_template;
      }
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
        body: JSON.stringify(Object.assign({
          session_id: sessionId,
          message: text,
          use_rag: modeEl.value === 'rag',
          model_source: sourceEl.value,
        }, optParams())),
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

  [temperatureEl, maxTokensEl, contextWindowEl, ctxSizeEl, variantEl,
   templateEl].forEach((el) => {
    el.addEventListener('change', persistOpts);
  });

  presetEl.addEventListener('change', () => {
    const p = PRESETS[presetEl.value];
    if (p) {
      temperatureEl.value = p.temperature;
      maxTokensEl.value = p.max_tokens;
      contextWindowEl.value = p.context_window;
    }
    persistOpts();
  });

  applyBtn.addEventListener('click', async () => {
    applyBtn.disabled = true;
    try {
      const r = await fetch('api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          ctx_size: parseInt(ctxSizeEl.value, 10) || undefined,
          variant: variantEl.value || undefined,
        }),
      });
      const data = await r.json();
      if (!r.ok) {
        add('system', '⚠️ ' + (data.error || ('HTTP ' + r.status)));
      } else {
        add('system', 'Настройки применены: модель ' + (data.local_variant || '—')
          + ', ctx ' + data.ctx_size + ', потоки ' + data.threads + '.');
      }
      persistOpts();
    } catch (err) {
      add('system', '⚠️ Ошибка соединения: ' + err);
    } finally {
      applyBtn.disabled = false;
    }
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
                message,
                session_id,
                use_rag=use_rag,
                model_source=model_source,
                temperature=data.get("temperature"),
                max_tokens=data.get("max_tokens"),
                context_window=data.get("context_window"),
                preset=data.get("preset"),
                prompt_template=data.get("prompt_template"),
            )
        except (TypeError, ValueError) as exc:
            return (
                jsonify(
                    {"error": f"Некорректные параметры: {exc}", "session_id": session_id}
                ),
                400,
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

    @app.post("/api/settings")
    def settings():
        """Применяет настройки уровня сервера (День 29): ``ctx_size``/``variant``."""
        data = request.get_json(silent=True) or {}
        ctx_size = data.get("ctx_size")
        variant = data.get("variant")
        try:
            result = backend.configure(
                ctx_size=int(ctx_size) if ctx_size is not None else None,
                variant=str(variant) if variant else None,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001 — перезапуск сервера мог упасть
            return jsonify({"error": f"Ошибка применения настроек: {exc}"}), 502
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
