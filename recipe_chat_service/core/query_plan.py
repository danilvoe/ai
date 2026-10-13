"""Планировщик запроса к базе рецептов (День 30, доработка).

Модуль решает класс ошибок, когда вопрос-подборка («что бы приготовить из
морепродуктов?») не распознаётся маршрутизатором и уходит в строгий ответ по
случайным чанкам, после чего модель отвечает отказом.

Состоит из двух слоёв:

* **Детерминированный** — :class:`CategoryTaxonomy` (таксономия категорий
  индекса + ручные алиасы-синонимы, проверяемые по фактически присутствующим
  категориям) и :func:`normalize_markers` (снятие вставных слов «бы/б»), а также
  эвристики из :mod:`core.local_rag`.
* **LLM-планировщик** — вызывается ТОЛЬКО при неопределённости и НЕ для
  офлайн-стаба: просит у модели строгий JSON ``{intent, query, category, dish}``
  с опорой на список допустимых категорий. Любой сбой парсинга откатывается к
  детерминированному плану.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

# Слова-заполнители, которые ломают сопоставление по подстрокам
# («что бы приготовить» ≠ «что приготовить»).
_FILLER_WORDS = frozenset({"бы", "б", "пожалуйста", "будь", "будьте"})

# Ручные группы алиасов: целевой стем категории → префиксы слов-синонимов.
# Целевой стем разрешается в реальную категорию индекса (см. CategoryTaxonomy).
_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "говядин": ("говядин",),
    "свинин": ("свинин",),
    "баранин": ("баранин", "барашк", "ягн"),
    "птиц": ("птиц", "куриц", "курин", "индейк", "утин", "гус"),
    "рыб": (
        "рыб",
        "лосос",
        "семг",
        "тунец",
        "скумбри",
        "треск",
        "дорадо",
        "форел",
        "сельд",
        "сардин",
        "карп",
        "окун",
        "камбал",
        "палтус",
        "сайд",
        "мойв",
    ),
    "дары моря": (
        "дары моря",
        "морепродукт",
        "креветк",
        "кальмар",
        "мидии",
        "миди",
        "осьминог",
        "краб",
        "лобстер",
        "гребешк",
        "устриц",
        "рапан",
        "лангустин",
    ),
    "соус": ("соус",),
    "маринад": ("маринад",),
    "гарнир": ("гарнир",),
    "десерт": ("десерт", "выпечк"),
    "бургер": ("бургер",),
    "ребр": ("ребр",),
    "закуск": ("закуск",),
    "овощ": ("овощ",),
    "фрукт": ("фрукт",),
}

# Допустимые намерения плана.
INTENTS = frozenset(
    {"specific_dish", "category", "browse", "followup", "chit_chat", "strict"}
)

DEFAULT_PLANNER_MIN_CONFIDENCE = 0.5
# Верхняя граница числа блюд, которое может запросить планировщик (защита от
# абсурдных значений в JSON слабой модели).
MAX_PLANNER_LIMIT = 50

# Промпт LLM-планировщика: строгий JSON, категория — из списка.
PLANNER_SYSTEM_PROMPT = (
    "Ты — маршрутизатор запросов к базе кулинарных рецептов гриля и BBQ.\n"
    "Определи намерение пользователя и верни СТРОГО JSON без пояснений и текста:\n"
    '{{"intent": "specific_dish|category|browse|followup|chit_chat", '
    '"query": "<поисковый запрос>", '
    '"category": "<одна из категорий или null>", '
    '"dish": "<название блюда или null>", '
    '"limit": <число блюд или null>}}\n'
    "Правила:\n"
    "- category: пользователь просит подборку по категории; значение — ровно одна "
    "из: {categories}.\n"
    "- specific_dish: названо конкретное блюдо; заполни dish его названием.\n"
    "- browse: общий запрос («что приготовить», «идеи», «варианты») или просьба "
    "дать список блюд.\n"
    "- followup: уточнение к предыдущему ответу («второй», «подробнее про него»).\n"
    "- chit_chat: приветствие или вопрос не про рецепты.\n"
    "- query: короткий поисковый запрос по базе (ключевые слова/название блюда).\n"
    "- limit: число, если просят список из N блюд («собери 10 блюд», «топ 5»); "
    "иначе null."
)


def _norm(text: str) -> str:
    """Нижний регистр, ё→е, сжатие пробелов."""
    return " ".join(text.lower().replace("ё", "е").split())


def _words(text: str) -> list[str]:
    """Слова текста в нижнем регистре (для сопоставления по началу слова)."""
    return re.findall(r"[a-zа-я0-9]+", _norm(text))


def normalize_markers(text: str) -> str:
    """Убирает слова-заполнители, чтобы маркеры подборок совпадали.

    Например, «что бы приготовить» → «что приготовить», после чего срабатывает
    маркер ``что приготовить``.
    """
    return " ".join(word for word in _words(text) if word not in _FILLER_WORDS)


@dataclass
class QueryPlan:
    """Структурированный план обработки запроса.

    ``intent`` — намерение (см. :data:`INTENTS`); ``search_query`` — запрос для
    поиска; ``category`` — каноническая категория индекса; ``dish`` — название
    блюда; ``limit`` — запрошенное число блюд для списка («собери 10 блюд»);
    ``source`` — откуда план (``heuristic``/``llm``); ``confidence`` —
    оценка уверенности детерминированного слоя.
    """

    intent: str
    search_query: str
    category: str | None = None
    dish: str | None = None
    source: str = "heuristic"
    confidence: float = 1.0
    limit: int | None = None


class CategoryTaxonomy:
    """Таксономия категорий индекса + синонимы, проверенные по данным.

    Строится по метаданным ``category`` чанков (мультизначным, «Говядина,
    Стейк»): собирает множество канонических категорий и карту алиасов, оставляя
    только те цели, что реально присутствуют в индексе.
    """

    def __init__(self, raw_categories: Iterable[str]) -> None:
        self._counts: dict[str, int] = {}
        for raw in raw_categories:
            for token in str(raw).split(","):
                name = _norm(token)
                if not name:
                    continue
                self._counts[name] = self._counts.get(name, 0) + 1
        self.categories: set[str] = set(self._counts)
        self.aliases: dict[str, str] = {}
        for target_stem, prefixes in _ALIAS_GROUPS.items():
            canonical = self.resolve(target_stem)
            if canonical is None:
                continue
            for prefix in prefixes:
                self.aliases[_norm(prefix)] = canonical

    def resolve(self, value: str | None) -> str | None:
        """Приводит значение к канонической категории индекса (или ``None``)."""
        if not value:
            return None
        needle = _norm(value)
        for category in self.categories:
            if category == needle or category.startswith(needle):
                return category
        return None

    def detect(self, text: str) -> str | None:
        """Определяет категорию по алиасам/каноническим именам в тексте."""
        norm = _norm(text)
        words = _words(text)
        for alias, target in self.aliases.items():
            if " " in alias:
                if alias in norm:
                    return target
            elif any(word.startswith(alias) for word in words):
                return target
        for category in self.categories:
            if " " in category and category in norm:
                return category
        return None

    def top_categories(self, limit: int = 8) -> list[str]:
        """Категории, отсортированные по числу чанков (для меню-подсказки)."""
        ordered = sorted(self._counts.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _count in ordered[: max(1, int(limit))]]


def _extract_json(text: str | None) -> dict[str, Any] | None:
    """Достаёт первый JSON-объект из ответа модели (терпимо к обвязке)."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_limit(value: Any) -> int | None:
    """Число блюд из поля ``limit`` (int/числовая строка); иначе ``None``.

    Отсекает нули, отрицательные значения и мусор; ограничивает сверху
    :data:`MAX_PLANNER_LIMIT`, чтобы слабая модель не запросила абсурд.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return min(number, MAX_PLANNER_LIMIT)


def _llm_plan(
    client: Any,
    question: str,
    history: list[dict[str, Any]],
    taxonomy: CategoryTaxonomy,
) -> QueryPlan | None:
    """Просит модель вернуть структурированный план; при сбое — ``None``."""
    categories = ", ".join(sorted(taxonomy.categories))
    transcript = "\n".join(
        f"{'Пользователь' if m.get('role') == 'user' else 'Ассистент'}: {m.get('content', '')}"
        for m in (history or [])[-6:]
    )
    user_content = (
        f"Категории: {categories or '—'}\n\n"
        f"Диалог:\n{transcript}\n\nВопрос: «{question}»"
        if transcript
        else f"Категории: {categories or '—'}\n\nВопрос: «{question}»"
    )
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(categories=categories or "—")},
        {"role": "user", "content": user_content},
    ]
    try:
        completion = client.complete(messages, temperature=0.0, max_tokens=120)
        data = _extract_json(getattr(completion, "content", ""))
    except Exception:  # noqa: BLE001 — при сбое используем детерминированный план
        return None
    if not data:
        return None

    intent = str(data.get("intent", "")).strip()
    if intent not in INTENTS:
        return None
    category = taxonomy.resolve(data.get("category"))
    query = str(data.get("query") or question).strip() or question
    dish = data.get("dish")
    dish = str(dish).strip() if dish else None
    if intent == "category" and category is None:
        # Модель назвала категорию вне индекса — понижаем до подборки.
        intent = "browse"
    limit = _parse_limit(data.get("limit"))
    if intent in ("specific_dish", "followup", "chit_chat"):
        limit = None
    return QueryPlan(
        intent=intent,
        search_query=query,
        category=category,
        dish=dish,
        source="llm",
        confidence=1.0,
        limit=limit,
    )


def _confidence(question: str, raw_sources: list[Any]) -> float:
    """Оценка уверенности прямого (строгого) ответа по выдаче ретривера.

    Учитывает пересечение значимых слов запроса с заголовком лучшего источника
    и его косинусное сходство. Низкая уверенность означает, что подмешивать
    выдачу в строгий ответ нельзя — нужен категорийный/подборочный маршрут.
    """
    from .local_rag import _share_prefix, _significant_tokens, _words

    if not raw_sources:
        return 0.0
    best = max(float(source.score) for source in raw_sources)
    significant = _significant_tokens(question)
    if significant:
        title_tokens = _words(raw_sources[0].title)
        overlap = sum(
            1
            for token in significant
            if any(_share_prefix(token, title_token) for title_token in title_tokens)
        )
        overlap_ratio = overlap / len(significant)
    else:
        overlap_ratio = 0.0
    score_ratio = min(best / 0.6, 1.0)
    return round(0.6 * overlap_ratio + 0.4 * score_ratio, 4)


def plan_query(
    client: Any,
    question: str,
    history: list[dict[str, Any]] | None,
    taxonomy: CategoryTaxonomy | None,
    raw_sources: list[Any],
    llm_enabled: bool = True,
    min_confidence: float = DEFAULT_PLANNER_MIN_CONFIDENCE,
) -> QueryPlan:
    """Строит план обработки запроса: детерминированно, при неясности — LLM.

    Сначала разрешает уточнения-отсылки и конкретные блюда, затем категорию и
    подборку. Только если уверенности не хватает и есть не-стаб клиент, вызывает
    LLM-планировщик; при любом сбое возвращает безопасный план-подборку.
    """
    from .local_rag import (
        LocalRagStubClient,
        _heuristic_resolve_reference,
        _matches_specific_dish,
        detect_category,
        extract_requested_count,
        is_browse_query,
        is_followup_query,
    )

    # 1) Уточнение-отсылка к предыдущему ответу.
    if history and is_followup_query(question):
        resolved = _heuristic_resolve_reference(question, history)
        if resolved:
            return QueryPlan(
                "followup", resolved, dish=resolved, source="heuristic", confidence=1.0
            )

    # 2) Названо конкретное блюдо — раскрываем весь его рецепт.
    if raw_sources and _matches_specific_dish(question, raw_sources):
        return QueryPlan(
            "specific_dish",
            question,
            dish=raw_sources[0].title,
            source="heuristic",
            confidence=1.0,
        )

    browse = is_browse_query(question)
    category = (
        taxonomy.detect(question) if taxonomy is not None else detect_category(question)
    )
    confidence = _confidence(question, raw_sources)
    limit = extract_requested_count(question)

    # 3) Категорийная подборка: категория распознана, а не конкретное блюдо.
    if category is not None and (browse or confidence < min_confidence):
        return QueryPlan(
            "category",
            question,
            category=category,
            source="heuristic",
            confidence=max(confidence, 0.6),
            limit=limit,
        )

    # 4) Общая подборка (в т.ч. запрос-список «собери N блюд»).
    if browse:
        return QueryPlan(
            "browse", question, source="heuristic", confidence=confidence, limit=limit
        )

    # 5) Уверенный прямой запрос — строгий маршрут.
    if confidence >= min_confidence:
        return QueryPlan("strict", question, source="heuristic", confidence=confidence)

    # 6) Неуверенность — пробуем LLM-планировщик (офлайн-стаб пропускаем).
    if (
        llm_enabled
        and taxonomy is not None
        and client is not None
        and not isinstance(client, LocalRagStubClient)
    ):
        planned = _llm_plan(client, question, history or [], taxonomy)
        if planned is not None:
            if planned.limit is None:
                planned.limit = limit
            return planned

    # 7) Безопасный фолбэк: считаем подборкой, чтобы не подмешивать мусор.
    return QueryPlan(
        "browse", question, category=category, source="heuristic", confidence=confidence, limit=limit
    )
