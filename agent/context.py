"""Стратегии управления контекстом диалога агента.

Вместо того чтобы всегда отправлять в модель полную историю (или сворачивать её
в summary, как в agent/compression.py), агент может применять одну из стратегий,
которая определяет, что реально уходит в запрос:

- **Sliding Window** — держим только последние N сообщений, всё остальное
  физически отбрасываем из истории.
- **Sticky Facts (Key-Value Memory)** — важные данные диалога (цель, ограничения,
  предпочтения, решения, договорённости) выносим в отдельный блок ``facts``
  (ключ-значение), который отправляется в запрос вместе с последними N
  сообщениями. Факты обновляются после каждого сообщения пользователя.
- **Branching** — сохраняем checkpoint, создаём от него независимые ветки
  диалога, продолжаем беседу в каждой ветке отдельно и переключаемся между ними.

Все стратегии реализуют общий интерфейс :class:`ContextStrategy` и подключаются
к агенту через параметр ``context_strategy`` (см. agent/agent.py).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .conversation import Conversation
from .tokens import estimate_messages_tokens

# Ключи, которые считаются "липкими" и извлекаются из строк вида "ключ: значение".
STICKY_KEYS = (
    "цель",
    "ограничение",
    "предпочтение",
    "решение",
    "договорённость",
    "договоренность",
    "задача",
    "дедлайн",
)

_FACT_ASSIGN_RE = re.compile(
    r"запомни\s*[:,\-]?\s*([\w ]+?)\s*=\s*(.+)", re.IGNORECASE | re.UNICODE
)
_FACT_COLON_RE = re.compile(r"([\w ]+?)\s*:\s*(.+)")


def extract_facts_from_text(text: str) -> dict[str, str]:
    """Эвристически извлекает пары ключ-значение из реплики пользователя.

    Распознаются два вида записи:

    - ``запомни: ключ = значение`` (или ``запомни ключ = значение``);
    - ``ключ: значение`` для известных липких ключей (цель, ограничение, ...).

    Это детерминированная локальная замена LLM-экстрактора: в офлайн-режиме она
    даёт предсказуемое поведение. В боевом режиме на её место можно поставить
    отдельный вызов LLM.
    """
    facts: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().strip(".!")
        if not line:
            continue
        match = _FACT_ASSIGN_RE.match(line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            facts[key] = value
            continue
        match = _FACT_COLON_RE.match(line)
        if match:
            key = match.group(1).strip().lower()
            if key in STICKY_KEYS:
                facts[key] = match.group(2).strip()
    return facts


class ContextStrategy(ABC):
    """Общий интерфейс стратегии управления контекстом."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Короткое имя стратегии (для логов и конфигурации)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Человекочитаемое название стратегии."""

    @abstractmethod
    def build_messages(self, conversation: Conversation) -> list[dict]:
        """Возвращает сообщения, которые будут отправлены в модель."""

    @abstractmethod
    def context_tokens(self, conversation: Conversation) -> int:
        """Оценка токенов контекста, отправляемого в модель."""

    def on_user_message(self, conversation: Conversation, user_request: str) -> None:
        """Хук после добавления сообщения пользователя (напр. извлечение фактов)."""

    def on_reply(self, conversation: Conversation, reply) -> None:
        """Хук после добавления ответа ассистента (напр. усечение окна)."""


@dataclass
class SlidingWindowStrategy(ContextStrategy):
    """Стратегия 1: храним только последние ``size`` сообщений."""

    size: int = 10

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def label(self) -> str:
        return "Sliding Window"

    def build_messages(self, conversation: Conversation) -> list[dict]:
        return conversation.messages[-self.size:]

    def context_tokens(self, conversation: Conversation) -> int:
        return estimate_messages_tokens(self.build_messages(conversation))

    def on_reply(self, conversation: Conversation, reply) -> None:
        # Отбрасываем всё, что вышло за окно (не только не отправляем).
        conversation.trim_to(self.size)


@dataclass
class StickyFactsStrategy(ContextStrategy):
    """Стратегия 2: липкие факты (ключ-значение) + последние ``window_size`` сообщений.

    После каждого сообщения пользователя факты извлекаются и складываются в
    ``conversation.facts``. В запрос уходит system-блок со всеми фактами и окно
    последних ``window_size`` сообщений.
    """

    window_size: int = 10
    max_facts: int = 30
    _facts: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "sticky_facts"

    @property
    def label(self) -> str:
        return "Sticky Facts"

    def build_messages(self, conversation: Conversation) -> list[dict]:
        messages: list[dict] = []
        facts = self._facts or conversation.facts
        if facts:
            lines = [f"- {key}: {value}" for key, value in facts.items()]
            messages.append(
                {
                    "role": "system",
                    "content": "Важные факты о диалоге (не противоречь им):\n"
                    + "\n".join(lines),
                }
            )
        messages.extend(conversation.messages[-self.window_size:])
        return messages

    def context_tokens(self, conversation: Conversation) -> int:
        return estimate_messages_tokens(self.build_messages(conversation))

    def on_user_message(self, conversation: Conversation, user_request: str) -> None:
        # Обновляем липкие факты после каждого сообщения пользователя.
        extracted = extract_facts_from_text(user_request)
        if extracted:
            if not self._facts:
                self._facts = conversation.facts
            for key, value in extracted.items():
                self._facts[key] = value
            if len(self._facts) > self.max_facts:
                self._facts = dict(list(self._facts.items())[-self.max_facts:])
            conversation.update_facts(self._facts)


@dataclass
class BranchingStrategy(ContextStrategy):
    """Стратегия 3: ветвление диалога от checkpoint.

    В общий контекст уходит общая часть (checkpoint) и сообщения активной ветки.
    Создание ветки/переключение выполняется через методы Conversation
    (``checkpoint``, ``create_branch``, ``switch_branch``).
    """

    window_size: int | None = None

    @property
    def name(self) -> str:
        return "branching"

    @property
    def label(self) -> str:
        return "Branching"

    def build_messages(self, conversation: Conversation) -> list[dict]:
        messages = conversation.messages
        if self.window_size is not None:
            messages = messages[-self.window_size:]
        return messages

    def context_tokens(self, conversation: Conversation) -> int:
        return estimate_messages_tokens(self.build_messages(conversation))


def strategy_from_config(config: dict) -> ContextStrategy | None:
    """Создаёт стратегию управления контекстом из секции ``context`` конфигурации.

    Возвращает None, если секция отсутствует или стратегия выключена — тогда
    агент работает по умолчанию (без стратегии, вся история как есть).
    """
    section = config.get("context", {}) or {}
    name = (section.get("strategy") or "").strip().lower()
    if not name:
        return None
    if name == "sliding_window":
        return SlidingWindowStrategy(size=int(section.get("size", 10)))
    if name == "sticky_facts":
        return StickyFactsStrategy(
            window_size=int(section.get("window_size", 10)),
            max_facts=int(section.get("max_facts", 30)),
        )
    if name == "branching":
        window_size = section.get("window_size")
        return BranchingStrategy(
            window_size=int(window_size) if window_size is not None else None
        )
    raise ValueError(f"Неизвестная стратегия управления контекстом: {name}")
