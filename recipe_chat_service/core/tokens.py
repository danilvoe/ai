"""Грубая оценка токенов (самодостаточная копия для сервиса).

Совпадает по поведению с ``agent/tokens.py`` в части, нужной индексатору:
аппроксимация «~4 символа на токен», одинаковая для кириллицы и латиницы.
"""

from __future__ import annotations

from math import ceil


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов в тексте (~4 символа на токен)."""
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Сумма токенов по всем сообщениям истории диалога."""
    return sum(estimate_tokens(message.get("content", "")) for message in messages)
