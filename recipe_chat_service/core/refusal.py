"""Проверка отказа и запроса уточнения в ответе модели.

Извлечено из ``agent/grounding.py`` (День 24) без зависимости на
``reranking.py``: сервису нужны только нормализация текста и правило
«не знаю + запрос уточнения».
"""

from __future__ import annotations

from typing import Any


def _normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру и заменяет букву ё."""
    return text.lower().replace("ё", "е")


def verify_refusal_and_clarification(text: str) -> dict[str, Any]:
    """Проверяет правило усиления: 'не знаю' + запрос уточнения."""
    norm = _normalize_text(text)

    refusal_markers = [
        "не знаю",
        "не содержит",
        "отсутствует",
        "нет данных",
        "нет информации",
        "не указано",
        "не указана",
        "не найдено",
        "не приводится",
        "не представлена",
        "не упоминается",
    ]
    is_refusal = any(m in norm for m in refusal_markers)

    clarification_markers = [
        "уточните",
        "уточнить",
        "пожалуйста, укажите",
        "пожалуйста, уточните",
        "задайте уточняющий",
        "какой именно рецепт",
        "какое конкретно блюдо",
        "сформулируйте запрос",
    ]
    clarification_requested = any(m in norm for m in clarification_markers)

    return {
        "is_refusal": is_refusal,
        "clarification_requested": clarification_requested,
        "rule_satisfied": is_refusal and clarification_requested,
    }
