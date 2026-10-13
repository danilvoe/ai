"""Конфигурация сервиса рецептов (чтение ``config.json``).

Все пути разрешаются относительно корня сервиса (``recipe_chat_service/``),
чтобы сервис оставался переносимым. Отсутствующие секции заполняются
разумными значениями по умолчанию.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_NAME = "config.json"

DEFAULT_DATABASE_PATH = "data/service.db"
DEFAULT_INDEX_DIR = "index"
DEFAULT_DATASET = "recipt_all"
DEFAULT_STRATEGY = "structural"
DEFAULT_TOP_K = 5
DEFAULT_LIST_MAX = 10
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
DEFAULT_SECRET_PATH = "data/secret.key"


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass
class ServiceConfig:
    """Настройки веб-сервиса: БД, индекс, RAG и веб-сервер."""

    service_root: Path = SERVICE_ROOT
    database_path: Path = field(default_factory=lambda: SERVICE_ROOT / DEFAULT_DATABASE_PATH)
    index_dir: Path = field(default_factory=lambda: SERVICE_ROOT / DEFAULT_INDEX_DIR)
    dataset: str = DEFAULT_DATASET
    strategy: str = DEFAULT_STRATEGY
    top_k: int = DEFAULT_TOP_K
    temperature: float = 0.2
    max_context_chars: int = 6000
    relevance_threshold: float = 0.33
    generic_threshold: float = 0.25
    generic_min_sources: int = 3
    category_suggestions: int = 5
    list_max: int = DEFAULT_LIST_MAX
    history_max_messages: int = 8
    history_max_chars: int = 2000
    planner_enabled: bool = True
    planner_min_confidence: float = 0.5
    category_menu_size: int = 8
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    debug: bool = False
    allow_reindex: bool = False
    secret_key: str = ""
    secret_path: Path = field(default_factory=lambda: SERVICE_ROOT / DEFAULT_SECRET_PATH)
    local_llm: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        service_root: Path = SERVICE_ROOT,
    ) -> ServiceConfig:
        """Загружает конфигурацию из файла (по умолчанию ``config.json``)."""
        config_path = Path(path) if path else service_root / DEFAULT_CONFIG_NAME
        data: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8") as file:
                data = json.load(file)
        return cls.from_dict(data, service_root=service_root)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        service_root: Path = SERVICE_ROOT,
    ) -> ServiceConfig:
        data = data or {}
        database = data.get("database", {}) or {}
        index = data.get("index", {}) or {}
        rag = data.get("local_rag", {}) or {}
        web = data.get("web", {}) or {}
        security = data.get("security", {}) or {}

        db_value = database.get("path", DEFAULT_DATABASE_PATH)
        index_value = index.get("dir", DEFAULT_INDEX_DIR)
        secret_value = security.get("secret_path", DEFAULT_SECRET_PATH)

        return cls(
            service_root=service_root,
            database_path=_resolve(service_root, db_value),
            index_dir=_resolve(service_root, index_value),
            dataset=str(index.get("dataset", DEFAULT_DATASET)),
            strategy=str(index.get("strategy", DEFAULT_STRATEGY)),
            top_k=int(index.get("top_k", DEFAULT_TOP_K)),
            temperature=float(rag.get("temperature", 0.2)),
            max_context_chars=int(rag.get("max_context_chars", 6000)),
            relevance_threshold=float(rag.get("relevance_threshold", 0.33)),
            generic_threshold=float(rag.get("generic_threshold", 0.25)),
            generic_min_sources=int(rag.get("generic_min_sources", 3)),
            category_suggestions=int(rag.get("category_suggestions", 5)),
            list_max=int(rag.get("list_max", DEFAULT_LIST_MAX)),
            history_max_messages=int(rag.get("history_max_messages", 8)),
            history_max_chars=int(rag.get("history_max_chars", 2000)),
            planner_enabled=bool(rag.get("planner_enabled", True)),
            planner_min_confidence=float(rag.get("planner_min_confidence", 0.5)),
            category_menu_size=int(rag.get("category_menu_size", 8)),
            host=str(web.get("host", DEFAULT_HOST)),
            port=int(web.get("port", DEFAULT_PORT)),
            debug=bool(web.get("debug", False)),
            allow_reindex=bool(web.get("allow_reindex", False)),
            secret_key=str(security.get("secret_key", "") or ""),
            secret_path=_resolve(service_root, secret_value),
            local_llm=dict(data.get("local_llm", {}) or {}),
            raw=dict(data),
        )

    def local_llm_config(self) -> Any:
        """Возвращает :class:`core.local_llm.LocalLLMConfig` из секции конфига."""
        from core.local_llm import LocalLLMConfig

        return LocalLLMConfig.from_config({"local_llm": self.local_llm})

    def ensure_directories(self) -> None:
        """Создаёт каталоги для БД и секрета."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
