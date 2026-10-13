"""RAG-движок сервиса: один ретривер, одна локальная модель, семафор генерации.

:class:`RagEngine` загружает FAISS-индекс и держит единый
:class:`core.local_rag.LocalRagSystem`. Генерация сериализуется через
``threading.Semaphore(1)``, чтобы не перегружать слоты одного ``llama-server``.
В офлайн-режиме вместо сервера используется детерминированный
:class:`core.local_rag.LocalRagStubClient`.
"""

from __future__ import annotations

import threading
from typing import Any

from core.local_llm import LocalLLMServer
from core.local_rag import LocalRagStubClient, LocalRagSystem
from core.query_plan import CategoryTaxonomy
from core.rag import RagAnswer, RagRetriever, load_retriever

from .config import ServiceConfig


class RagEngine:
    """Общий RAG-движок на процесс: ретривер + клиент генерации + семафор."""

    def __init__(self, config: ServiceConfig, offline: bool = False) -> None:
        self.config = config
        self.offline = offline
        self._semaphore = threading.Semaphore(1)
        self._reload_lock = threading.Lock()

        self.retriever = self._load_retriever()
        self.taxonomy = self._build_taxonomy()
        self.server: LocalLLMServer | None = None
        self.client: Any = None
        self._start_client()

        self.rag = LocalRagSystem(
            client=self.client,
            retriever=self.retriever,
            top_k=config.top_k,
            temperature=config.temperature,
            max_context_chars=config.max_context_chars,
            model_label=self.model_label,
            relevance_threshold=config.relevance_threshold,
            history_max_messages=config.history_max_messages,
            history_max_chars=config.history_max_chars,
            generic_threshold=config.generic_threshold,
            generic_min_sources=config.generic_min_sources,
            category_suggestions=config.category_suggestions,
            list_max=config.list_max,
            taxonomy=self.taxonomy,
            planner_enabled=config.planner_enabled and not offline,
            planner_min_confidence=config.planner_min_confidence,
            category_menu_size=config.category_menu_size,
        )

    # ------------------------------------------------------------------ setup

    def _load_retriever(self) -> RagRetriever:
        return load_retriever(
            self.config.index_dir,
            dataset=self.config.dataset,
            strategy=self.config.strategy,
            top_k=self.config.top_k,
        )

    def _build_taxonomy(self) -> CategoryTaxonomy:
        """Таксономия категорий из метаданных чанков (синонимы + меню)."""
        raw_categories = [
            str((chunk.metadata or {}).get("category", ""))
            for chunk in self.retriever.index.chunks
        ]
        return CategoryTaxonomy(raw_categories)

    def _start_client(self) -> None:
        local_config = self.config.local_llm_config()
        self.model_label = local_config.model_label
        if self.offline:
            self.client = LocalRagStubClient(f"{local_config.model_label}-rag-stub")
            self.model_label = self.client.config["model"]
            return
        self.server = LocalLLMServer(local_config)
        self.server.start()
        self.client = self.server.build_client()
        self.model_label = local_config.model_label

    # ------------------------------------------------------------------- query

    def ask(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        use_rag: bool = True,
    ) -> RagAnswer:
        """Выполняет запрос (RAG или plain) под семафором генерации."""
        with self._semaphore:
            if use_rag:
                return self.rag.ask_with_rag(question, history=history)
            return self.rag.ask_without_rag(question)

    # ------------------------------------------------------------------ status

    @property
    def total_vectors(self) -> int:
        return self.retriever.total_vectors

    def status(self) -> dict[str, Any]:
        """Краткий статус движка для ``/api/status``."""
        return {
            "offline": self.offline,
            "model": self.model_label,
            "dataset": self.config.dataset,
            "strategy": self.config.strategy,
            "total_vectors": self.total_vectors,
            "top_k": self.config.top_k,
            "relevance_threshold": self.config.relevance_threshold,
        }

    # ----------------------------------------------------------------- reindex

    def reload(self) -> None:
        """Перечитывает индекс с диска (после пересборки)."""
        with self._reload_lock:
            retriever = self._load_retriever()
            self.retriever = retriever
            self.rag.retriever = retriever

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
