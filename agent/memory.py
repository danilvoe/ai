"""Модель памяти агента: три отдельных слоя с явным выбором, что куда сохранять.

Память агента разделена на три слоя, каждый хранится отдельно и наполняется
явно — агент (или вызывающий код) сам решает, что и в какой слой попадает:

- **Краткосрочная (short-term)** — текущий диалог: сообщения, которыми обменялись
  пользователь и агент в этой сессии. Это и есть ``conversation.messages``.
  Обновляется автоматически на каждом ходу.
- **Рабочая (working)** — данные текущей задачи: промежуточные результаты,
  введённые пользователем значения, ограничения конкретной задачи. Слой
  живёт в рамках сессии (одной задачи), его можно очистить при смене задачи.
  Наполняется через ``remember_working``.
- **Долговременная (long-term)** — профиль пользователя, принятые решения и
  накопленные знания. Слой сохраняется в отдельном файле и переживает сессии,
  поэтому агент помнит пользователя даже в новом диалоге. Наполняется через
  ``remember_long_term`` с указанием категории (profile / decision / knowledge).

Все три слоя собираются в один контекст :meth:`MemoryLayers.build_memory_context`,
который подставляется в запрос к модели (см. agent/agent.py). Слои разделены
физически: диалог — в JSON сессии, рабочая память — в том же файле сессии (блок
``working``), долговременная — в отдельном профильном файле ``profile.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .conversation import Conversation

# Категории долговременной памяти.
LONG_TERM_KINDS = ("profile", "decision", "knowledge")

LONG_TERM_LABELS = {
    "profile": "Профиль пользователя",
    "decision": "Принятые решения",
    "knowledge": "Накопленные знания",
}


class LongTermKind(str, Enum):
    """Категория записи долговременной памяти."""

    PROFILE = "profile"
    DECISION = "decision"
    KNOWLEDGE = "knowledge"


def default_profile_path(history_dir: Path | None = None) -> Path:
    """Путь к файлу долговременной памяти (по умолчанию рядом с историей)."""
    base = history_dir or (Path(__file__).resolve().parent.parent / "history")
    return base / "profile.json"


class WorkingMemory:
    """Слой рабочей памяти: данные текущей задачи (в рамках сессии).

    Хранится в JSON-файле сессии (блок ``working``), чтобы переживать
    перезапуск CLI, но обнуляется при смене задачи. Все операции записывают
    на диск немедленно.
    """

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation

    def all(self) -> dict[str, str]:
        """Все записи рабочей памяти текущей сессии."""
        return dict(self._conversation.working)

    def get(self, key: str) -> str | None:
        """Одна запись рабочей памяти."""
        return self._conversation.working.get(key)

    def remember(self, key: str, value: str) -> None:
        """Явно сохраняет запись в рабочую память."""
        self._conversation.set_working(key, value)

    def remember_many(self, items: dict[str, str]) -> None:
        """Сохраняет несколько записей в рабочую память."""
        self._conversation.update_working(items)

    def clear(self) -> None:
        """Очищает рабочую память (например, при смене текущей задачи)."""
        self._conversation.clear_working()

    def __bool__(self) -> bool:
        return bool(self._conversation.working)

    def __repr__(self) -> str:
        return f"WorkingMemory({self.all()!r})"


class LongTermMemory:
    """Слой долговременной памяти: профиль, решения, знания.

    Хранится в отдельном файле ``profile.json`` и переживает сессии — агент
    сохраняет память о пользователе между разными диалогами. Записи
    группируются по категориям (profile / decision / knowledge).
    """

    def __init__(self, path: Path = default_profile_path()) -> None:
        self._path = path
        self._data: dict[str, dict[str, str]] = {kind: {} for kind in LONG_TERM_KINDS}
        self._load()

    @property
    def path(self) -> Path:
        """Файл, в котором хранится долговременная память."""
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        for kind in LONG_TERM_KINDS:
            stored = data.get(kind, {}) if isinstance(data, dict) else {}
            if isinstance(stored, dict):
                self._data[kind] = {str(k): str(v) for k, v in stored.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)

    def _validate_kind(self, kind: str) -> str:
        kind = str(kind).strip().lower()
        if kind not in LONG_TERM_KINDS:
            raise ValueError(
                f"Неизвестная категория долговременной памяти: {kind!r}. "
                f"Допустимо: {', '.join(LONG_TERM_KINDS)}."
            )
        return kind

    def remember(self, kind: str, key: str, value: str) -> None:
        """Явно сохраняет запись в долговременную память нужной категории."""
        kind = self._validate_kind(kind)
        self._data[kind][str(key)] = str(value)
        self._save()

    def get(self, kind: str, key: str) -> str | None:
        """Одна запись долговременной памяти."""
        kind = self._validate_kind(kind)
        return self._data[kind].get(str(key))

    def all(self, kind: str | None = None) -> dict[str, dict[str, str]]:
        """Записи долговременной памяти (все категории или одна)."""
        if kind is not None:
            kind = self._validate_kind(kind)
            return dict(self._data[kind])
        return {k: dict(v) for k, v in self._data.items()}

    def forget(self, kind: str, key: str) -> bool:
        """Удаляет запись из долговременной памяти."""
        kind = self._validate_kind(kind)
        if key in self._data[kind]:
            del self._data[kind][key]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Полностью очищает долговременную память."""
        self._data = {kind: {} for kind in LONG_TERM_KINDS}
        self._save()

    def __bool__(self) -> bool:
        return any(self._data.values())

    def __repr__(self) -> str:
        return f"LongTermMemory({self.all()!r})"


@dataclass
class MemoryLayers:
    """Модель памяти агента: три слоя + сборка общего контекста для модели.

    Позволяет явно выбрать, что и куда сохраняется:
    ``remember_short_term`` (диалог), ``remember_working`` (текущая задача),
    ``remember_long_term`` (профиль / решения / знания). Слои хранятся отдельно,
    но :meth:`build_memory_context` собирает их в один набор system-сообщений,
    который подставляется в запрос к модели.
    """

    conversation: Conversation
    long_term_path: Path = field(default_factory=default_profile_path)
    _long_term: LongTermMemory = field(init=False)
    _working: WorkingMemory = field(init=False)

    def __post_init__(self) -> None:
        self._working = WorkingMemory(self.conversation)
        self._long_term = LongTermMemory(self.long_term_path)

    # ---- краткосрочная (текущий диалог) ------------------------------------

    @property
    def short_term(self) -> list[dict]:
        """Краткосрочная память: сообщения текущего диалога."""
        return self.conversation.messages

    def remember_short_term(self, role: str, content: str) -> None:
        """Добавляет сообщение в краткосрочную память (текущий диалог)."""
        self.conversation.append(role, content)

    # ---- рабочая (текущая задача) -----------------------------------------

    @property
    def working(self) -> WorkingMemory:
        """Рабочая память: данные текущей задачи."""
        return self._working

    def remember_working(self, key: str, value: str) -> None:
        """Явно сохраняет запись в рабочую память (текущая задача)."""
        self._working.remember(key, value)

    # ---- долговременная (профиль, решения, знания) -------------------------

    @property
    def long_term(self) -> LongTermMemory:
        """Долговременная память: профиль, решения, знания."""
        return self._long_term

    def remember_long_term(self, kind: str, key: str, value: str) -> None:
        """Явно сохраняет запись в долговременную память нужной категории."""
        self._long_term.remember(kind, key, value)

    # ---- сборка контекста ---------------------------------------------------

    def build_memory_context(self, include: Iterable[str] = LONG_TERM_KINDS) -> list[dict]:
        """Собирает блоки памяти в список system-сообщений для модели.

        :param include: какие категории долговременной памяти включать.
        Краткосрочная (диалог) и рабочая (текущая задача) память включаются
        всегда, категории долговременной — только перечисленные в ``include``.
        """
        blocks: list[str] = []
        if self._working:
            lines = [f"- {key}: {value}" for key, value in self._working.all().items()]
            blocks.append(
                "Рабочая память (данные текущей задачи, приоритетны):\n"
                + "\n".join(lines)
            )
        for kind in LONG_TERM_KINDS:
            if kind not in include:
                continue
            entries = self._long_term.all(kind)
            if not entries:
                continue
            lines = [f"- {key}: {value}" for key, value in entries.items()]
            blocks.append(f"{LONG_TERM_LABELS[kind]}:\n" + "\n".join(lines))
        return [{"role": "system", "content": block} for block in blocks]

    def summarize(self) -> str:
        """Краткая сводка всех слоёв памяти (для вывода пользователю/логов)."""
        lines = []
        lines.append(f"Краткосрочная (диалог): {len(self.short_term)} сообщений.")
        if self._working:
            lines.append(
                "Рабочая: " + ", ".join(f"{k}={v}" for k, v in self._working.all().items())
            )
        else:
            lines.append("Рабочая: пусто.")
        for kind in LONG_TERM_KINDS:
            entries = self._long_term.all(kind)
            if entries:
                lines.append(
                    f"{LONG_TERM_LABELS[kind]}: "
                    + ", ".join(f"{k}={v}" for k, v in entries.items())
                )
            else:
                lines.append(f"{LONG_TERM_LABELS[kind]}: пусто.")
        return "\n".join(lines)
