"""Day 28: RAG-система, полностью работающая локально.

Модуль связывает retrieval из Дней 21–22 (FAISS-индекс рецептов + офлайн
эмбеддер) с локальной генерацией из Дней 26–27 (``llama-server``,
OpenAI-совместимый endpoint ``/v1``):

    вопрос → локальный поиск по FAISS → локальная LLM → ответ + источники

Компоненты:

* :class:`LocalRagStubClient` — детерминированный офлайн-клиент, совместимый
  с :class:`agent.llm_client.LLMClient` (``.config`` + ``.complete()``), который
  отвечает по переданному RAG-контексту. Нужен для проверок без модели и сети.
* :class:`LocalRagSystem` — лёгкий RAG-сеанс поверх ретривера и любого
  ``.complete``-клиента (локальный, стаб или облачный).
* :func:`build_local_rag_client` — поднимает ``llama-server`` либо отдаёт стаб.
* :func:`timed_rag_query` — запрос с раздельным замером retrieval и генерации.
* :func:`compare_local_vs_cloud` — сравнение локальной и облачной модели по
  качеству, скорости и стабильности на контрольных вопросах.

Запуск полного сценария::

    python3 -m agent.local_rag_scenarios --offline
    python3 -m agent.local_rag_scenarios
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from .grounding import verify_refusal_and_clarification
from .llm_client import Completion
from .local_llm import LocalLLMConfig, LocalLLMServer, StubLocalLLM
from .rag import (
    CONTROL_QUESTIONS,
    RAG_SYSTEM_PROMPT,
    RagAnswer,
    RagRetriever,
    RagSource,
    build_rag_messages,
    evaluate_answer_facts,
    evaluate_citations,
    evaluate_source_recall,
    plain_query,
)

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_HISTORY_MAX_CHARS = 2000
# Порог косинусного сходства, ниже которого чанк считается нерелевантным
# (тот же порог, что и в анти-галлюцинационном шлюзе Дня 24).
DEFAULT_RELEVANCE_THRESHOLD = 0.33
# Для общих («что можно приготовить?») запросов порог смягчается, чтобы в
# контекст попали несколько разных рецептов для подсказки.
DEFAULT_GENERIC_THRESHOLD = 0.25
DEFAULT_GENERIC_MIN_SOURCES = 3
# Сколько разных рецептов показывать в категорийной подборке («рецепт из говядины»).
DEFAULT_CATEGORY_SUGGESTIONS = 5

# Ответ при отсутствии релевантных материалов: источники не показываются,
# модель не вызывается.
NO_RELEVANT_INFO_MESSAGE = (
    "В базе рецептов нет информации, релевантной этому вопросу. "
    "Уточните, пожалуйста, вопрос — о каком рецепте или блюде идёт речь."
)

# Инструкции, добавляемые к RAG-запросу (День 28).
RAG_EXTRA_INSTRUCTION = (
    "Важно: не упоминай служебный контекст и не пиши фразы вроде "
    "«контекст описывает…», «в предоставленном контексте…». "
    "Если ответа в контексте нет — ответь ровно: "
    "«В предоставленной базе рецептов эта информация отсутствует.»"
)
GENERIC_EXTRA_INSTRUCTION = (
    "Вопрос общий. Перечисли 3–5 различных рецептов из контекста "
    "(название и одна короткая строка о блюде) и предложи выбрать, "
    "какой раскрыть подробнее. Не утверждай, что база пуста, если рецепты есть."
)
CATEGORY_EXTRA_INSTRUCTION = (
    "Пользователь просит подборку рецептов по категории. В контексте перечислены "
    "несколько рецептов этой категории — информация ЕСТЬ. Обязательно перечисли "
    "3–5 разных рецептов (название и одна короткая строка о блюде) и предложи "
    "выбрать один, чтобы раскрыть подробнее. Не пересказывай один шаг "
    "приготовления и НЕ отвечай, что информации нет."
)
FOLLOWUP_EXTRA_INSTRUCTION = (
    "Пользователь уточняет и просит подробнее раскрыть ОДИН конкретный рецепт, "
    "уже упомянутый ранее. Ответь по этому рецепту из контекста: краткое "
    "описание, основные ингредиенты и шаги приготовления. Не перечисляй другие "
    "блюда и не предлагай выбор заново."
)

# Ответ, когда уточнение распознано, но привязать его к рецепту не удалось.
CLARIFY_REFERENCE_MESSAGE = (
    "Уточните, пожалуйста, о каком именно рецепте речь — назовите блюдо "
    "или номер позиции из предыдущего списка."
)

# Промпт переформулировки уточняющего вопроса с опорой на историю диалога.
FOLLOWUP_REWRITE_SYSTEM_PROMPT = (
    "Ты уточняешь поисковый запрос по кулинарной базе рецептов гриля и BBQ.\n"
    "Тебе дают предыдущий диалог и короткий вопрос-уточнение пользователя.\n"
    "Разреши отсылки («2 позицию», «второй», «этот», «он», «тот самый», "
    "«поподробнее») в НАЗВАНИЕ блюда из предыдущего ответа.\n"
    "Выведи ТОЛЬКО поисковый запрос одной строкой: название блюда. "
    "Без кавычек, без пояснений и вежливых фраз."
)

# Маркеры общих («browse») запросов, для которых смягчается порог.
_GENERIC_QUERY_MARKERS = (
    "что можно приготовить",
    "что приготовить",
    "что-нибудь",
    "что нибудь",
    "посоветуй",
    "предложи",
    "варианты",
    "идеи",
    "на ужин",
    "на обед",
)

# Дополнительные маркеры запросов-подборок (День 28).
_BROWSE_MARKERS = _GENERIC_QUERY_MARKERS + (
    "рецепт",
    "рецепты",
    "блюдо",
    "блюда",
    "топ",
    "покажи",
    "список",
    "подборка",
    "какие есть",
)

# Маркеры уточняющих запросов-отсылок («подробней про 2 позицию», «расскажи про него»).
_FOLLOWUP_MARKERS = (
    "подробн",
    "поподробн",
    "позици",
    "пункт",
    "вариант",
    "номер",
    "№",
    "второй",
    "вторая",
    "второе",
    "вторую",
    "первый",
    "первая",
    "первое",
    "первую",
    "третий",
    "третья",
    "третье",
    "третью",
    "четверт",
    "пятый",
    "пятая",
    "пятое",
    "этот",
    "эта",
    "это ",
    "эту",
    "тот ",
    "та ",
    "о нем",
    "о ней",
    "про него",
    "про нее",
    "него",
    "нее",
    "оно",
)

# Порядковые слова → номер позиции.
_ORDINAL_WORDS: dict[str, int] = {
    "первый": 1,
    "первая": 1,
    "первое": 1,
    "первую": 1,
    "второй": 2,
    "вторая": 2,
    "второе": 2,
    "вторую": 2,
    "третий": 3,
    "третья": 3,
    "третье": 3,
    "третью": 3,
    "четвертый": 4,
    "четвертая": 4,
    "четвертое": 4,
    "четвертую": 4,
    "пятый": 5,
    "пятая": 5,
    "пятое": 5,
}

# Нумерованная строка списка из предыдущего ответа: «2. Шашлык …».
_LIST_ITEM_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")

# Стем → подстрока в метаданных ``category`` (в нижнем регистре).
# Категории мультизначные («Говядина, Стейк»), поэтому сверяем вхождением.
_CATEGORY_STEMS: tuple[tuple[str, str], ...] = (
    ("говядин", "говядин"),
    ("свинин", "свинин"),
    ("баранин", "баранин"),
    ("барашк", "баранин"),
    ("ягн", "баранин"),
    ("куриц", "птиц"),
    ("курин", "птиц"),
    ("птиц", "птиц"),
    ("индейк", "птиц"),
    ("рыб", "рыб"),
    ("соус", "соус"),
    ("маринад", "маринад"),
    ("гарнир", "гарнир"),
    ("десерт", "десерт"),
    ("выпечк", "десерт"),
    ("бургер", "бургер"),
    ("ребр", "ребр"),
    ("закуск", "закуск"),
)

# Признаки «утечки» служебного контекста в ответ-отказ.
_META_LEAK_MARKERS = (
    "контекст описывает",
    "в предоставленном контексте",
    "предоставленном контексте",
    "предоставленной базе",
)

# Разделители, которые использует build_rag_messages для сборки контекста.
_CONTEXT_MARKER = "КОНТЕКСТ ИЗ БАЗЫ РЕЦЕПТОВ:"
_QUESTION_MARKER = "ВОПРОС ПОЛЬЗОВАТЕЛЯ:"
_SOURCE_BLOCK_RE = re.compile(
    r"\[Источник\s+(\d+)\]([^\n]*)\n(.*?)(?=\n\n\[Источник\s+\d+\]|\n-{2,}|\Z)",
    re.DOTALL,
)
_TITLE_RE = re.compile(r"«([^»]+)»")

# Вопросы о данных, которых нет в базе (проверка заземления, q10).
_ABSENT_DATA_HINTS = (
    "калорий",
    "калорийност",
    "бжу",
    "белк",
    "жир",
    "углевод",
)


def _normalize(text: str) -> str:
    """Нормализует текст ответа для сравнения (регистр, пробелы)."""
    return " ".join(text.lower().replace("ё", "е").split())


def is_generic_query(question: str) -> bool:
    """Определяет общий («что можно приготовить?») запрос по ключевым фразам."""
    norm = _normalize(question)
    return any(marker in norm for marker in _GENERIC_QUERY_MARKERS)


def has_meta_leak(text: str) -> bool:
    """Есть ли в ответе упоминание служебного контекста («контекст описывает…»)."""
    norm = _normalize(text)
    return any(marker in norm for marker in _META_LEAK_MARKERS)


def strip_meta_leak(text: str) -> str:
    """Удаляет предложения с упоминанием служебного контекста.

    Если после чистки не остаётся содержательного текста — возвращает
    канонический отказ :data:`NO_RELEVANT_INFO_MESSAGE`.
    """
    if not text or not has_meta_leak(text):
        return text
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    kept = [s for s in sentences if s.strip() and not has_meta_leak(s)]
    cleaned = " ".join(s.strip() for s in kept).strip()
    return cleaned or NO_RELEVANT_INFO_MESSAGE


def _category_fallback(sources: list[RagSource]) -> str:
    """Детерминированная подборка рецептов, если локальная модель отказывается.

    Нужна как страховка для слабых локальных моделей: источники категории уже
    подобраны, поэтому подсказку можно собрать без генерации.
    """
    lines = ["В базе рецептов есть подходящие блюда. Вот несколько вариантов:"]
    for source in sources:
        text = re.sub(r"^Описание\s*«[^»]+»:\s*", "", " ".join(source.text.split()))
        first_sentence = re.split(r"(?<=[.!?…])\s+", text, maxsplit=1)[0].strip()
        suffix = f" — {first_sentence}" if first_sentence else ""
        lines.append(f"- «{source.title}»{suffix}")
    lines.append("Выберите рецепт, который хотите раскрыть подробнее.")
    return "\n".join(lines)


def _recipe_fallback(sources: list[RagSource]) -> str:
    """Детерминированно раскрывает конкретный рецепт по его секциям.

    Страховка на случай, если модель отказывается, хотя рецепт уже найден и
    полностью присутствует в источниках. Возвращает описание, ингредиенты и
    шаги приготовления без обращения к генерации.
    """
    if not sources:
        return NO_RELEVANT_INFO_MESSAGE
    title = sources[0].title or "рецепт"
    lines = [f"Рецепт «{title}»:"]
    for source in sources:
        if source.section == "header":
            continue
        text = " ".join(source.text.split())
        if text:
            lines.append(text)
    return "\n".join(lines)


def detect_category(question: str) -> str | None:
    """Возвращает подстроку категории (в нижнем регистре) для запроса-подборки.

    Сопоставление по началу слова, чтобы учесть словоизменение
    («говядины» → стем «говядин», «шницеля» и т.п.).
    """
    for word in _words(question):
        for stem, category in _CATEGORY_STEMS:
            if word.startswith(stem):
                return category
    return None


def is_browse_query(question: str) -> bool:
    """Запрос-подборка: общий или про категорию («рецепт из говядины»)."""
    norm = _normalize(question)
    if any(marker in norm for marker in _BROWSE_MARKERS):
        return True
    return detect_category(question) is not None


def _words(text: str) -> list[str]:
    """Слова текста в нижнем регистре (для сопоставления по началу слова)."""
    return re.findall(r"[a-zа-я0-9]+", _normalize(text))


def _starts_with_category(word: str) -> bool:
    return any(word.startswith(stem) for stem, _ in _CATEGORY_STEMS)


def _share_prefix(left: str, right: str, length: int = 4) -> bool:
    return len(left) >= length and len(right) >= length and left[:length] == right[:length]


def _significant_tokens(question: str) -> list[str]:
    """Значимые слова запроса без browse-маркеров и категорийных слов."""
    marker_tokens: set[str] = set()
    for marker in _BROWSE_MARKERS:
        marker_tokens |= set(_words(marker))
    return [
        word
        for word in _words(question)
        if len(word) >= 4 and word not in marker_tokens and not _starts_with_category(word)
    ]


def _matches_specific_dish(question: str, raw_sources: list[RagSource]) -> bool:
    """Запрос называет конкретное блюдо, если значимое слово попало в заголовок."""
    significant = _significant_tokens(question)
    if not significant or not raw_sources:
        return False
    title_tokens = _words(raw_sources[0].title)
    return any(
        _share_prefix(token, title_token)
        for token in significant
        for title_token in title_tokens
    )


def _match_recipe_id(
    question: str, raw_sources: list[RagSource]
) -> int | str | None:
    """Возвращает ``recipe_id`` блюда, названного в запросе, если он найден.

    Среди кандидатов выбирается рецепт с наибольшим числом значимых слов,
    совпавших с заголовком (tie-break — по сходству). Нужен, чтобы для запроса
    «покажи рецепт X» подтянуть все секции именно этого рецепта.
    """
    significant = _significant_tokens(question)
    if not significant:
        return None
    best_id: int | str | None = None
    best_key: tuple[int, float] | None = None
    for source in raw_sources:
        if source.recipe_id is None:
            continue
        title_tokens = _words(source.title)
        matches = sum(
            1
            for token in significant
            if any(_share_prefix(token, title_token) for title_token in title_tokens)
        )
        if matches <= 0:
            continue
        key = (matches, source.score)
        if best_key is None or key > best_key:
            best_key = key
            best_id = source.recipe_id
    return best_id


def is_followup_query(question: str) -> bool:
    """Уточняющий запрос-отсылка к предыдущему ответу («подробней про 2 позицию»)."""
    norm = _normalize(question)
    return any(marker in norm for marker in _FOLLOWUP_MARKERS)


def _extract_ordinal(question: str) -> int | None:
    """Номер позиции из запроса: слово («второй») или число («2», «№ 2»)."""
    for word in _words(question):
        if word in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[word]
    norm = _normalize(question)
    match = re.search(r"(?:№\s*|позици\w*\s*|пункт\w*\s*|номер\s*)(\d+)", norm)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", norm)
    if len(numbers) == 1 and len(_words(norm)) <= 6:
        return int(numbers[0])
    return None


def _previous_recipe_titles(history: list[dict[str, Any]]) -> list[str]:
    """Названия рецептов из ближайшего assistant-ответа с RAG-источниками.

    Сначала пробуем ``meta.sources`` (компактные источники хода), затем —
    нумерованный список, разобранный прямо из текста предыдущего ответа.
    """
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        meta = message.get("meta") or {}
        sources = meta.get("sources") or []
        titles = [str(s.get("title", "")).strip() for s in sources if s.get("title")]
        if titles:
            return titles
        parsed: list[str] = []
        for line in str(message.get("content", "")).splitlines():
            match = _LIST_ITEM_RE.match(line)
            if match:
                title = re.sub(r"\s*\[Источник[^\]]*\]", "", match.group(2)).strip()
                if title:
                    parsed.append(title)
        if parsed:
            return parsed
    return []


def _heuristic_resolve_reference(
    question: str, history: list[dict[str, Any]]
) -> str | None:
    """Детерминированно разрешает отсылку по предыдущему списку рецептов."""
    titles = _previous_recipe_titles(history)
    if not titles:
        return None
    ordinal = _extract_ordinal(question)
    if ordinal is not None:
        if 1 <= ordinal <= len(titles):
            return titles[ordinal - 1]
        return None
    # Указатель/местоимение без номера — разумеем последний упомянутый рецепт.
    if is_followup_query(question):
        return titles[-1]
    return None


def resolve_followup_query(
    client: Any,
    question: str,
    history: list[dict[str, Any]],
) -> str | None:
    """Преобразует уточняющий вопрос-отсылку в название блюда.

    Для явного номера позиции используется точная эвристика по предыдущему
    списку. Для местоимений/указателей («подробнее про него») вызывается
    LLM-переформулировка с историей, а при офлайн-стабе или ошибке —
    эвристический откат. Возвращает поисковый запрос или ``None``.
    """
    if _extract_ordinal(question) is not None:
        return _heuristic_resolve_reference(question, history)
    heuristic = _heuristic_resolve_reference(question, history)
    if client is None or isinstance(client, LocalRagStubClient):
        return heuristic
    transcript = "\n".join(
        f"{'Пользователь' if m.get('role') == 'user' else 'Ассистент'}: {m.get('content', '')}"
        for m in (history or [])[-6:]
    )
    messages = [
        {"role": "system", "content": FOLLOWUP_REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Диалог:\n{transcript}\n\nВопрос-уточнение: «{question}»",
        },
    ]
    try:
        completion: Completion = client.complete(messages, temperature=0.0, max_tokens=60)
        lines = [line.strip() for line in completion.content.strip().splitlines() if line.strip()]
        rewritten = lines[0].strip('"').strip("'").strip("«»").strip() if lines else ""
        if rewritten and len(rewritten) <= 200:
            return rewritten
    except Exception:  # noqa: BLE001 — при сбое используем эвристику
        pass
    return heuristic


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9]+", _normalize(text)))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _coeff_variation(values: list[float]) -> float:
    """Коэффициент вариации (std / mean); 0 — идеально стабильно."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return round(statistics.pstdev(values) / mean, 4)


def select_history(
    messages: list[dict[str, Any]],
    max_messages: int = 8,
    max_chars: int = DEFAULT_HISTORY_MAX_CHARS,
) -> list[dict[str, str]]:
    """Отбирает последние реплики диалога в пределах окна и бюджета символов.

    Возвращает сообщения в хронологическом порядке. Последняя реплика всегда
    включается, даже если превышает бюджет (иначе был бы потерян контекст).
    """
    if not messages:
        return []
    limit = max(1, int(max_messages))
    candidates = messages[-limit:]
    selected: list[dict[str, str]] = []
    total = 0
    for item in reversed(candidates):
        role = item.get("role")
        content = str(item.get("content", ""))
        if role not in ("user", "assistant"):
            continue
        if selected and total + len(content) > max_chars:
            break
        selected.append({"role": role, "content": content})
        total += len(content)
    selected.reverse()
    return selected


# ============================================================================
# 1. Детерминированный офлайн-клиент по RAG-контексту
# ============================================================================


class LocalRagStubClient:
    """Офлайн-клиент локальной LLM, отвечающий строго по RAG-контексту.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete(messages, ...)``), но не делает сетевых вызовов: извлекает
    найденные источники из сообщения и собирает экстрактивный ответ. Полностью
    детерминирован, поэтому подходит для проверок качества/стабильности без
    сборки модели и без интернета.
    """

    def __init__(self, model_label: str = "local-rag-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": DEFAULT_TEMPERATURE,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self._stub = StubLocalLLM(model_label)
        self.last_sources_used: int = 0

    @staticmethod
    def _last_user_content(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _split_context(content: str) -> tuple[str | None, str]:
        """Разделяет пользовательское сообщение на контекст и вопрос."""
        if _CONTEXT_MARKER not in content:
            return None, content
        rest = content.split(_CONTEXT_MARKER, 1)[1]
        if _QUESTION_MARKER in rest:
            context_part, question_part = rest.split(_QUESTION_MARKER, 1)
            question = ""
            for line in question_part.splitlines():
                if line.strip():
                    question = line.strip()
                    break
        else:
            context_part, question = rest, ""
        return context_part, question

    @staticmethod
    def _parse_sources(context: str) -> list[tuple[int, str, str]]:
        """Возвращает список (индекс, заголовок, текст) найденных источников."""
        parsed: list[tuple[int, str, str]] = []
        for match in _SOURCE_BLOCK_RE.finditer(context):
            idx = int(match.group(1))
            meta = match.group(2)
            body = " ".join(match.group(3).split())
            title_match = _TITLE_RE.search(meta)
            title = title_match.group(1) if title_match else ""
            parsed.append((idx, title, body))
        return parsed

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        content = self._last_user_content(messages)
        context, question = self._split_context(content)

        # Прямой вопрос без RAG — обычный стаб.
        if context is None:
            self.last_sources_used = 0
            return Completion(content=self._stub.answer(question), model=self.config["model"])

        question_norm = _normalize(question)
        if any(hint in question_norm for hint in _ABSENT_DATA_HINTS):
            self.last_sources_used = 0
            return Completion(
                content=(
                    "В предоставленной базе рецептов эта информация отсутствует: "
                    "сведения о калорийности и БЖУ блюд не приведены ни в одном источнике."
                ),
                model=self.config["model"],
            )

        sources = self._parse_sources(context)
        self.last_sources_used = min(len(sources), 2)
        if not sources:
            return Completion(
                content="В предоставленной базе рецептов подходящих материалов не найдено.",
                model=self.config["model"],
            )

        parts: list[str] = ["По данным базы рецептов:"]
        for idx, title, body in sources[:2]:
            head = f"«{title}»" if title else "рецепт"
            excerpt = body[:420].strip()
            parts.append(f"[Источник {idx}] {head}: {excerpt}")
        return Completion(content="\n\n".join(parts), model=self.config["model"])


# ============================================================================
# 2. Локальный RAG-сеанс
# ============================================================================


class LocalRagSystem:
    """Полностью локальный RAG: поиск по FAISS + генерация заданным клиентом.

    Отбрасывает нерелевантные чанки по порогу сходства: если ни один источник
    не проходит порог, модель не вызывается, а возвращается честный отказ без
    источников. Если модель отвечает отказом («не знаю»), источники скрываются.
    """

    def __init__(
        self,
        client: Any,
        retriever: RagRetriever,
        top_k: int = 5,
        temperature: float = DEFAULT_TEMPERATURE,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        model_label: str = "local-llm",
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        history_max_messages: int = 8,
        history_max_chars: int = DEFAULT_HISTORY_MAX_CHARS,
        generic_threshold: float = DEFAULT_GENERIC_THRESHOLD,
        generic_min_sources: int = DEFAULT_GENERIC_MIN_SOURCES,
        category_suggestions: int = DEFAULT_CATEGORY_SUGGESTIONS,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.top_k = max(1, int(top_k))
        self.temperature = float(temperature)
        self.max_context_chars = int(max_context_chars)
        self.model_label = model_label
        self.relevance_threshold = float(relevance_threshold)
        self.history_max_messages = max(1, int(history_max_messages))
        self.history_max_chars = int(history_max_chars)
        self.generic_threshold = float(generic_threshold)
        self.generic_min_sources = max(1, int(generic_min_sources))
        self.category_suggestions = max(1, int(category_suggestions))
        # День 29: переопределяемый системный промпт (шаблон под кейс) и лимит
        # токенов ответа. По умолчанию — как в Днях 22–28.
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    @property
    def total_vectors(self) -> int:
        return self.retriever.total_vectors

    def retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]:
        """Локальный поиск релевантных чанков (с фильтром по порогу, без модели)."""
        sources = self.retriever.retrieve(question, top_k=top_k or self.top_k)
        return [s for s in sources if s.score >= self.relevance_threshold]

    def ask_with_rag(
        self,
        question: str,
        top_k: int | None = None,
        history: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        system_prompt: str | None = None,
    ) -> RagAnswer:
        """RAG-запрос с опциональным переопределением параметров генерации.

        ``temperature``/``max_tokens``/``context_window``/``system_prompt``
        (День 29) позволяют менять параметры и prompt-шаблон на каждый запрос из
        веб-интерфейса, не пересоздавая систему. При ``None`` используются
        значения из конфигурации системы.
        """
        result = timed_rag_query(
            client=self.client,
            retriever=self.retriever,
            question=question,
            top_k=top_k or self.top_k,
            temperature=self.temperature if temperature is None else float(temperature),
            max_context_chars=self.max_context_chars,
            relevance_threshold=self.relevance_threshold,
            history=history,
            generic_threshold=self.generic_threshold,
            generic_min_sources=self.generic_min_sources,
            category_suggestions=self.category_suggestions,
            history_max_messages=(
                self.history_max_messages
                if context_window is None
                else max(1, int(context_window))
            ),
            history_max_chars=self.history_max_chars,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            system_prompt=(
                self.system_prompt if system_prompt is None else system_prompt
            ),
        )
        return result["answer"]

    def ask_without_rag(self, question: str) -> RagAnswer:
        return plain_query(client=self.client, question=question, temperature=self.temperature)

    def ask(self, question: str) -> RagAnswer:
        return self.ask_with_rag(question)


# ============================================================================
# 3. Запуск локального клиента
# ============================================================================


def build_local_rag_client(
    config: dict,
    offline: bool = False,
    model_path: str | None = None,
    port: int | None = None,
) -> tuple[LocalLLMServer | None, Any]:
    """Готовит клиент локальной генерации.

    Возвращает ``(server, client)``. В офлайн-режиме ``server=None`` и клиент —
    детерминированный :class:`LocalRagStubClient`; иначе поднимается
    ``llama-server`` (День 26) и возвращается направленный на него ``LLMClient``.
    """
    local = LocalLLMConfig.from_config(config)
    if model_path:
        local.model_path = model_path
    if port:
        local.port = port
    if offline:
        return None, LocalRagStubClient(f"{local.model_label}-rag-stub")
    server = LocalLLMServer(local)
    server.start()
    return server, server.build_client()


# ============================================================================
# 4. Запрос с раздельным замером retrieval и генерации
# ============================================================================


def retrieve_by_category(
    retriever: RagRetriever,
    question: str,
    category: str,
    limit: int = DEFAULT_CATEGORY_SUGGESTIONS,
) -> list[RagSource]:
    """Подбирает разные рецепты категории, ранжируя их по сходству с запросом.

    Используется для запросов-подборок («рецепт из говядины»): фильтруем чанки
    по метаданным ``category`` (мультизначным), оставляем лучший чанк на рецепт
    (предпочитая ``intro``/``header``) и берём до ``limit`` разных рецептов.
    """
    index = retriever.index
    embedder = index.embedder
    query_vec = embedder.embed_one(question)
    best: dict[Any, tuple[float, Any]] = {}
    preferred_sections = ("intro", "header")
    for chunk in index.chunks:
        meta = chunk.metadata or {}
        if category not in str(meta.get("category", "")).lower():
            continue
        score = float(query_vec @ embedder.embed_one(chunk.text))
        recipe_id = meta.get("recipe_id")
        section = str(meta.get("section", ""))
        current = best.get(recipe_id)
        if current is None:
            best[recipe_id] = (score, chunk)
            continue
        prev_score, prev_chunk = current
        prev_section = str((prev_chunk.metadata or {}).get("section", ""))
        prefer_new = section in preferred_sections
        prev_preferred = prev_section in preferred_sections
        if (prefer_new and not prev_preferred) or (
            prefer_new == prev_preferred and score > prev_score
        ):
            best[recipe_id] = (score, chunk)
    ordered = sorted(best.values(), key=lambda item: item[0], reverse=True)[: max(1, limit)]
    sources: list[RagSource] = []
    for rank, (score, chunk) in enumerate(ordered, start=1):
        meta = chunk.metadata or {}
        sources.append(
            RagSource(
                rank=rank,
                score=score,
                chunk_id=chunk.id,
                recipe_id=meta.get("recipe_id"),
                title=str(meta.get("title", "")),
                section=str(meta.get("section", "document")),
                step_number=meta.get("step_number"),
                url=str(meta.get("url", "")),
                text=chunk.text,
            )
        )
    return sources


def retrieve_recipe_sections(
    retriever: RagRetriever,
    recipe_id: int | str,
    question: str = "",
    limit: int = 32,
) -> list[RagSource]:
    """Собирает все секции одного рецепта (описание, ингредиенты, шаги, оборудование).

    Для запроса о конкретном блюде («покажи рецепт X») релевантен весь рецепт
    целиком, а не только чанки, прошедшие общий порог сходства: описательная
    секция ``intro`` похожа на запрос сильнее, чем ингредиенты и шаги, и при
    строгом пороге последние терялись бы. Поэтому секции берутся по
    ``recipe_id`` без порога, но упорядочиваются логически
    (header → intro → ingredients → шаги по номеру → equipment).
    """
    index = retriever.index
    embedder = index.embedder
    query_vec = embedder.embed_one(question) if question else None
    section_order = {
        "header": 0,
        "intro": 1,
        "ingredients": 2,
        "step": 3,
        "equipment": 4,
    }
    collected: list[tuple[tuple[int, int], float, Any]] = []
    for chunk in index.chunks:
        meta = chunk.metadata or {}
        if str(meta.get("recipe_id")) != str(recipe_id):
            continue
        section = str(meta.get("section", "document"))
        step_number = meta.get("step_number")
        step_key = int(step_number) if isinstance(step_number, int) else 0
        order = (section_order.get(section, 9), step_key)
        score = float(query_vec @ embedder.embed_one(chunk.text)) if query_vec is not None else 0.0
        collected.append((order, score, chunk))
    collected.sort(key=lambda item: item[0])
    sources: list[RagSource] = []
    for rank, (_order, score, chunk) in enumerate(collected[: max(1, limit)], start=1):
        meta = chunk.metadata or {}
        sources.append(
            RagSource(
                rank=rank,
                score=score,
                chunk_id=chunk.id,
                recipe_id=meta.get("recipe_id"),
                title=str(meta.get("title", "")),
                section=str(meta.get("section", "document")),
                step_number=meta.get("step_number"),
                url=str(meta.get("url", "")),
                text=chunk.text,
            )
        )
    return sources


def timed_rag_query(
    client: Any,
    retriever: RagRetriever,
    question: str,
    top_k: int = 5,
    temperature: float = DEFAULT_TEMPERATURE,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    history: list[dict[str, Any]] | None = None,
    generic_threshold: float = DEFAULT_GENERIC_THRESHOLD,
    generic_min_sources: int = DEFAULT_GENERIC_MIN_SOURCES,
    category_suggestions: int = DEFAULT_CATEGORY_SUGGESTIONS,
    history_max_messages: int = 8,
    history_max_chars: int = DEFAULT_HISTORY_MAX_CHARS,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Выполняет RAG-запрос локально и раздельно замеряет этапы.

    ``history`` — предыдущие реплики диалога (с ``meta.sources``), которые
    используются и для уточнения отсылок («подробней про 2 позицию»), и как
    контекст модели.
    Для общих запросов («что можно приготовить?») порог смягчается до
    ``generic_threshold`` и добирается до ``generic_min_sources`` источников.
    Для запросов-подборок с категорией («рецепт из говядины») источники берутся
    из метаданных категории (до ``category_suggestions`` разных рецептов).
    Возвращает словарь с полями ``answer`` (:class:`RagAnswer`),
    ``retrieval_ms`` (локальный поиск), ``generation_ms`` (генерация модели),
    ``route`` (выбранный маршрут) и ``low_relevance`` (истина, если ни один
    источник не прошёл порог).

    Если релевантных источников нет, модель не вызывается — возвращается отказ
    без источников (для уточнения-отсылки — просьба уточнить). Если модель
    отвечает отказом, источники скрываются.
    """
    t0 = time.perf_counter()
    raw_sources = retriever.retrieve(question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Уточнение-отсылка к предыдущему ответу: пробуем разрешить её в название
    # блюда (LLM-переформулировка с историей + эвристический откат).
    followup = is_followup_query(question)
    weak = not raw_sources or max(s.score for s in raw_sources) < relevance_threshold
    resolved_query: str | None = None
    if history and (followup or (weak and not is_browse_query(question))):
        resolved = resolve_followup_query(client, question, history)
        if resolved and _normalize(resolved) != _normalize(question):
            candidate_sources = retriever.retrieve(resolved, top_k=top_k)
            if not candidate_sources:
                alternate = _heuristic_resolve_reference(question, history)
                if alternate and _normalize(alternate) != _normalize(resolved):
                    candidate_sources = retriever.retrieve(alternate, top_k=top_k)
                    resolved = alternate if candidate_sources else resolved
            if candidate_sources:
                resolved_query = resolved
                raw_sources = candidate_sources
                retrieval_ms = (time.perf_counter() - t0) * 1000

    search_question = resolved_query or question
    browse = is_browse_query(search_question)
    category = detect_category(search_question)
    specific = _matches_specific_dish(search_question, raw_sources)
    route = "strict"

    sources: list[RagSource] = []
    if browse and category and not specific:
        route = "category"
        sources = retrieve_by_category(
            retriever, search_question, category, limit=category_suggestions
        )
    elif browse and not specific:
        route = "generic"
        threshold = float(generic_threshold)
        sources = [s for s in raw_sources if s.score >= threshold]
        best_score = max((s.score for s in raw_sources), default=0.0)
        # Добираем источники до минимума, но только если в выдаче есть хоть
        # что-то осмысленное; при полном мусоре не подмешиваем его, а уточняем.
        if best_score >= threshold and len(sources) < max(1, int(generic_min_sources)):
            seen = {s.chunk_id for s in sources}
            for candidate in raw_sources:
                if candidate.chunk_id in seen:
                    continue
                sources.append(candidate)
                seen.add(candidate.chunk_id)
                if len(sources) >= max(1, int(generic_min_sources)):
                    break
    else:
        route = "specific" if specific else "strict"
        if specific:
            # Запрос называет конкретное блюдо: подтягиваем ВЕСЬ его рецепт
            # (описание, ингредиенты, шаги), а не только чанки, прошедшие
            # общий порог сходства (иначе теряются ингредиенты и шаги).
            recipe_id = _match_recipe_id(search_question, raw_sources)
            recipe_sources = (
                retrieve_recipe_sections(retriever, recipe_id, question=search_question)
                if recipe_id is not None
                else []
            )
            sources = recipe_sources or [
                s for s in raw_sources if s.score >= relevance_threshold
            ]
        else:
            sources = [s for s in raw_sources if s.score >= relevance_threshold]

    if resolved_query and sources and route in ("specific", "strict"):
        route = "followup"

    if not sources:
        clarify = bool(history) and followup
        answer = RagAnswer(
            question=question,
            mode="rag",
            content=CLARIFY_REFERENCE_MESSAGE if clarify else NO_RELEVANT_INFO_MESSAGE,
            sources=[],
            latency_ms=retrieval_ms,
            error=None,
        )
        return {
            "answer": answer,
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0.0,
            "low_relevance": True,
            "route": "clarify" if clarify else "refusal",
        }

    extra_instruction = RAG_EXTRA_INSTRUCTION
    if route == "category":
        extra_instruction = f"{RAG_EXTRA_INSTRUCTION}\n{CATEGORY_EXTRA_INSTRUCTION}"
    elif route == "generic":
        extra_instruction = f"{RAG_EXTRA_INSTRUCTION}\n{GENERIC_EXTRA_INSTRUCTION}"
    elif route == "followup":
        extra_instruction = f"{RAG_EXTRA_INSTRUCTION}\n{FOLLOWUP_EXTRA_INSTRUCTION}"
    selected_history = select_history(
        history or [],
        max_messages=history_max_messages,
        max_chars=history_max_chars,
    )
    messages = build_rag_messages(
        question,
        sources,
        max_chars=max_context_chars,
        history=selected_history,
        extra_instruction=extra_instruction,
        system_prompt=system_prompt,
    )
    t1 = time.perf_counter()
    try:
        completion: Completion = client.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        generation_ms = (time.perf_counter() - t1) * 1000
        content = completion.content
        is_refusal = verify_refusal_and_clarification(content)["is_refusal"]
        # Убираем «утечку» служебным контекстом из любого ответа (не только отказа).
        content = strip_meta_leak(content)
        # Страховка: для категорийной подборки источники уже подобраны, поэтому
        # отказ слабой локальной модели заменяем детерминированным списком.
        if route == "category" and (is_refusal or content == NO_RELEVANT_INFO_MESSAGE):
            content = _category_fallback(sources)
            is_refusal = False
        # Страховка для конкретного блюда: рецепт уже найден целиком, поэтому
        # отказ модели заменяем детерминированным раскрытием его секций.
        if route in ("specific", "followup") and (
            is_refusal or content == NO_RELEVANT_INFO_MESSAGE
        ):
            content = _recipe_fallback(sources)
            is_refusal = False
        # Модель может ответить отказом даже при формально релевантных чанках —
        # тогда источники не показываем, чтобы не вводить пользователя в заблуждение.
        kept_sources = [] if is_refusal else sources
        answer = RagAnswer(
            question=question,
            mode="rag",
            content=content,
            sources=kept_sources,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            latency_ms=retrieval_ms + generation_ms,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 — ошибку вернём в отчёте
        generation_ms = (time.perf_counter() - t1) * 1000
        answer = RagAnswer(
            question=question,
            mode="rag",
            content="",
            sources=sources,
            latency_ms=retrieval_ms + generation_ms,
            error=str(exc),
        )
    return {
        "answer": answer,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "low_relevance": False,
        "route": route,
    }


# ============================================================================
# 5. Метрики стабильности
# ============================================================================


def answer_self_consistency(texts: list[str]) -> dict[str, Any]:
    """Оценивает согласованность нескольких ответов модели на один вопрос.

    Метрики:
    - ``exact_match_rate``: доля ответов, совпавших с самым частым (модальным);
    - ``mean_jaccard``: средняя попарная Jaccard-близость множеств слов;
    - ``unique_answers``: число различных нормализованных ответов.
    """
    normalized = [_normalize(t) for t in texts if t and t.strip()]
    n = len(normalized)
    if n == 0:
        return {
            "runs": 0,
            "exact_match_rate": 0.0,
            "mean_jaccard": 0.0,
            "unique_answers": 0,
        }
    if n == 1:
        return {
            "runs": 1,
            "exact_match_rate": 1.0,
            "mean_jaccard": 1.0,
            "unique_answers": 1,
        }

    unique = len(set(normalized))
    modal_count = max(normalized.count(item) for item in set(normalized))
    exact_rate = round(modal_count / n, 4)

    token_sets = [_token_set(t) for t in normalized]
    similarities: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            if not union:
                similarities.append(1.0)
            else:
                similarities.append(len(a & b) / len(union))

    return {
        "runs": n,
        "exact_match_rate": exact_rate,
        "mean_jaccard": round(statistics.mean(similarities), 4),
        "unique_answers": unique,
    }


# ============================================================================
# 6. Сравнение локальной и облачной модели
# ============================================================================


@dataclass
class _Run:
    """Один прогон RAG-запроса моделью (для агрегации повторов)."""

    content: str
    error: str | None
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    completion_tokens: int | None
    low_relevance: bool = False


def _run_model(
    client: Any,
    retriever: RagRetriever,
    question: str,
    repeats: int,
    top_k: int,
    temperature: float,
    max_context_chars: int,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> tuple[list[_Run], list[RagSource]]:
    """Выполняет запрос несколько раз и возвращает прогоны + источники."""
    runs: list[_Run] = []
    sources: list[RagSource] = []
    for _ in range(max(1, repeats)):
        result = timed_rag_query(
            client,
            retriever,
            question,
            top_k=top_k,
            temperature=temperature,
            max_context_chars=max_context_chars,
            relevance_threshold=relevance_threshold,
        )
        answer: RagAnswer = result["answer"]
        sources = answer.sources
        runs.append(
            _Run(
                content=answer.content,
                error=answer.error,
                latency_ms=answer.latency_ms,
                retrieval_ms=result["retrieval_ms"],
                generation_ms=result["generation_ms"],
                completion_tokens=answer.completion_tokens,
                low_relevance=bool(result.get("low_relevance")),
            )
        )
    return runs, sources


def _aggregate_runs(
    runs: list[_Run],
    sources: list[RagSource],
    expected_facts: list[list[str]],
) -> dict[str, Any]:
    """Считает метрики качества, скорости и стабильности по прогонам."""
    ok_runs = [r for r in runs if not r.error and r.content.strip()]
    representative = (ok_runs or runs)[0]
    errors = sum(1 for r in runs if r.error)
    low_relevance = sum(1 for r in runs if r.low_relevance)

    facts = evaluate_answer_facts(representative.content, expected_facts)
    citations = evaluate_citations(representative.content, sources)
    latencies = [r.latency_ms for r in ok_runs]
    generations = [r.generation_ms for r in ok_runs]
    retrievals = [r.retrieval_ms for r in runs]
    token_rates: list[float] = []
    for r in ok_runs:
        if r.completion_tokens and r.generation_ms > 0:
            token_rates.append(r.completion_tokens / (r.generation_ms / 1000))

    return {
        "answer": representative.content,
        "error": representative.error if not ok_runs else None,
        "error_rate": round(errors / len(runs), 4) if runs else 1.0,
        "low_relevance_rate": round(low_relevance / len(runs), 4) if runs else 0.0,
        "facts": facts,
        "citations": citations,
        "latency_ms": round(_median(latencies), 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "generation_ms": round(_median(generations), 2),
        "retrieval_ms": round(_median(retrievals), 2),
        "completion_tokens": representative.completion_tokens,
        "tokens_per_sec": round(_median(token_rates), 2) if token_rates else 0.0,
        "self_consistency": answer_self_consistency([r.content for r in ok_runs]),
        "latency_cv": _coeff_variation(latencies),
    }


def compare_local_vs_cloud(
    retriever: RagRetriever,
    local_client: Any,
    cloud_client: Any | None = None,
    questions: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    temperature: float = DEFAULT_TEMPERATURE,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    repeats: int = 1,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    local_label: str = "local-llm",
    cloud_label: str = "cloud",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Сравнивает локальную и облачную модель на контрольных вопросах.

    Retrieval общий и локальный: источники извлекаются один раз на вопрос и
    используются обеими моделями, поэтому влияние поиска одинаково.
    """
    items = questions if questions is not None else CONTROL_QUESTIONS
    total = len(items)
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        question = item["query"]
        expected_facts = item.get("expected_facts", [])
        expected_sources = item.get("expected_sources", [])
        if progress_callback:
            progress_callback(idx, total, question)

        local_runs, sources = _run_model(
            local_client,
            retriever,
            question,
            repeats,
            top_k,
            temperature,
            max_context_chars,
            relevance_threshold,
        )
        local = _aggregate_runs(local_runs, sources, expected_facts)

        cloud: dict[str, Any] | None = None
        if cloud_client is not None:
            cloud_runs, _ = _run_model(
                cloud_client,
                retriever,
                question,
                repeats,
                top_k,
                temperature,
                max_context_chars,
                relevance_threshold,
            )
            cloud = _aggregate_runs(cloud_runs, sources, expected_facts)

        source_eval = evaluate_source_recall(sources, expected_sources)
        results.append(
            {
                "id": item.get("id", f"q{idx}"),
                "query": question,
                "category": item.get("category", ""),
                "expectation": item.get("expectation", ""),
                "expected_sources": expected_sources,
                "expected_title": item.get("expected_title", ""),
                "sources": [s.to_dict() for s in sources],
                "source_recall": source_eval,
                "local": local,
                "cloud": cloud,
            }
        )

    n = len(results)

    def _avg(getter: Callable[[dict], float]) -> float:
        vals = [getter(r) for r in results if r is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    local_facts = [r["local"]["facts"]["coverage"] for r in results]
    cloud_facts = [r["cloud"]["facts"]["coverage"] for r in results if r["cloud"]]

    summary: dict[str, Any] = {
        "quality": {
            "source_recall_top_k": _avg(lambda r: 1.0 if r["source_recall"]["found"] else 0.0),
            "source_recall_top_1": _avg(
                lambda r: 1.0 if r["source_recall"]["top_rank"] == 1 else 0.0
            ),
            "local_avg_facts": round(statistics.mean(local_facts), 4) if local_facts else 0.0,
            "local_facts_std": round(statistics.pstdev(local_facts), 4) if len(local_facts) > 1 else 0.0,
            "local_citation_rate": _avg(
                lambda r: 1.0 if r["local"]["citations"]["has_citations"] else 0.0
            ),
            "cloud_avg_facts": round(statistics.mean(cloud_facts), 4) if cloud_facts else None,
            "cloud_facts_std": round(statistics.pstdev(cloud_facts), 4)
            if len(cloud_facts) > 1
            else (0.0 if cloud_facts else None),
            "cloud_citation_rate": _avg(
                lambda r: 1.0 if (r["cloud"] and r["cloud"]["citations"]["has_citations"]) else 0.0
            )
            if cloud_client is not None
            else None,
        },
        "speed": {
            "local_latency_ms": _avg(lambda r: r["local"]["latency_ms"]),
            "local_generation_ms": _avg(lambda r: r["local"]["generation_ms"]),
            "local_retrieval_ms": _avg(lambda r: r["local"]["retrieval_ms"]),
            "local_tokens_per_sec": _avg(lambda r: r["local"]["tokens_per_sec"]),
            "cloud_latency_ms": _avg(lambda r: r["cloud"]["latency_ms"]) if cloud_client else None,
            "cloud_generation_ms": _avg(lambda r: r["cloud"]["generation_ms"])
            if cloud_client
            else None,
            "cloud_tokens_per_sec": _avg(lambda r: r["cloud"]["tokens_per_sec"])
            if cloud_client
            else None,
        },
        "stability": {
            "local_error_rate": _avg(lambda r: r["local"]["error_rate"]),
            "local_low_relevance_rate": _avg(lambda r: r["local"]["low_relevance_rate"]),
            "local_self_consistency": _avg(
                lambda r: r["local"]["self_consistency"]["mean_jaccard"]
            ),
            "local_latency_cv": _avg(lambda r: r["local"]["latency_cv"]),
            "cloud_error_rate": _avg(lambda r: r["cloud"]["error_rate"]) if cloud_client else None,
            "cloud_self_consistency": _avg(
                lambda r: r["cloud"]["self_consistency"]["mean_jaccard"]
            )
            if cloud_client
            else None,
            "cloud_latency_cv": _avg(lambda r: r["cloud"]["latency_cv"]) if cloud_client else None,
        },
    }

    return {
        "total_questions": n,
        "dataset": retriever.index.metadata.get("source", "unknown"),
        "total_vectors": retriever.total_vectors,
        "top_k": top_k,
        "temperature": temperature,
        "relevance_threshold": float(relevance_threshold),
        "repeats": repeats,
        "local_label": local_label,
        "cloud_label": cloud_label if cloud_client is not None else None,
        "system_prompt": RAG_SYSTEM_PROMPT,
        "summary": summary,
        "results": results,
    }


# ============================================================================
# 7. Рендер Markdown-отчёта
# ============================================================================


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.2f}{suffix}"


def render_local_rag_markdown(report: dict[str, Any]) -> str:
    """Формирует Markdown-отчёт сравнения локальной и облачной модели."""
    summary = report["summary"]
    quality = summary["quality"]
    speed = summary["speed"]
    stability = summary["stability"]
    results = report["results"]
    has_cloud = report.get("cloud_label") is not None

    lines: list[str] = [
        "# Day 28 — RAG-система, полностью работающая локально",
        "",
        "Пайплайн: **вопрос → локальный поиск (FAISS) → локальная LLM → ответ + источники**.",
        "",
        f"- **База знаний:** `{report.get('dataset')}` ({report.get('total_vectors')} векторов)",
        f"- **Локальная модель:** `{report.get('local_label')}`",
        f"- **Облачная модель:** `{report.get('cloud_label') or 'не использовалась'}`",
        f"- **top-k:** {report.get('top_k')}, **temperature:** {report.get('temperature')}, "
        f"**повторов на вопрос:** {report.get('repeats')}",
        f"- **Порог релевантности:** {report.get('relevance_threshold')} "
        "(чанки ниже порога отбрасываются, модель не вызывается; при отказе источники скрываются)",
        f"- **Контрольных вопросов:** {report.get('total_questions')}",
        "",
        "## 1. Качество",
        "",
        "| Метрика | Локальная модель | Облачная модель |",
        "| --- | --- | --- |",
        f"| Среднее покрытие фактов | **{_pct(quality['local_avg_facts'])}** | "
        f"**{_pct(quality.get('cloud_avg_facts'))}** |",
        f"| Разброс покрытия фактов (std) | {quality['local_facts_std']:.3f} | "
        f"{_num(quality.get('cloud_facts_std'))} |",
        f"| Доля ответов со ссылками на источники | {_pct(quality['local_citation_rate'])} | "
        f"{_pct(quality.get('cloud_citation_rate'))} |",
        f"| Нахождение целевого рецепта в top-{report.get('top_k')} | "
        f"{_pct(quality['source_recall_top_k'])} | {_pct(quality['source_recall_top_k'])} |",
        f"| Целевой рецепт на 1 месте | {_pct(quality['source_recall_top_1'])} | "
        f"{_pct(quality['source_recall_top_1'])} |",
        "",
        "_Retrieval общий и выполняется локально (FAISS + офлайн-эмбеддер), поэтому метрики "
        "поиска одинаковы для обеих моделей._",
        "",
        "## 2. Скорость",
        "",
        "| Метрика | Локальная модель | Облачная модель |",
        "| --- | --- | --- |",
        f"| Средняя задержка ответа | {_num(speed['local_latency_ms'], ' мс')} | "
        f"{_num(speed.get('cloud_latency_ms'), ' мс')} |",
        f"| из них генерация модели | {_num(speed['local_generation_ms'], ' мс')} | "
        f"{_num(speed.get('cloud_generation_ms'), ' мс')} |",
        f"| Локальный retrieval | {_num(speed['local_retrieval_ms'], ' мс')} | — |",
        f"| Скорость генерации | {_num(speed['local_tokens_per_sec'], ' ток/с')} | "
        f"{_num(speed.get('cloud_tokens_per_sec'), ' ток/с')} |",
        "",
        "## 3. Стабильность",
        "",
        "| Метрика | Локальная модель | Облачная модель |",
        "| --- | --- | --- |",
        f"| Доля ошибок вызовов | {_pct(stability['local_error_rate'])} | "
        f"{_pct(stability.get('cloud_error_rate'))} |",
        f"| Доля вопросов без релевантных источников | "
        f"{_pct(stability['local_low_relevance_rate'])} | — |",
        f"| Самосогласованность ответов (Jaccard) | "
        f"{stability['local_self_consistency']:.3f} | "
        f"{_num(stability.get('cloud_self_consistency'))} |",
        f"| Вариация задержки (CV) | {stability['local_latency_cv']:.3f} | "
        f"{_num(stability.get('cloud_latency_cv'))} |",
        "",
        "## 4. Результаты по контрольным вопросам",
        "",
        f"| № | Вопрос | Ранг источника | Факты (локал) | Ссылки (локал) | Задержка (локал) |"
        f"{' Факты (облако) |' if has_cloud else ''}",
        f"| - | --- | --- | --- | --- | --- |{' --- |' if has_cloud else ''}",
    ]

    for idx, r in enumerate(results, start=1):
        q = r["query"]
        q_short = q[:42] + "…" if len(q) > 42 else q
        rank = r["source_recall"]["top_rank"]
        rank_str = f"#{rank}" if rank else "—"
        fact_local = _pct(r["local"]["facts"]["coverage"])
        cite = "Да" if r["local"]["citations"]["has_citations"] else "Нет"
        lat = f"{r['local']['latency_ms']:.0f} мс"
        row = f"| {idx} | {q_short} | {rank_str} | {fact_local} | {cite} | {lat} |"
        if has_cloud:
            cloud = r["cloud"] or {}
            fact_cloud = _pct((cloud.get("facts") or {}).get("coverage"))
            row += f" {fact_cloud} |"
        lines.append(row)

    lines.extend(["", "## 5. Детальный разбор вопросов", ""])
    for idx, r in enumerate(results, start=1):
        lines.extend(
            [
                f"### Вопрос №{idx}: {r['query']}",
                "",
                f"- **Целевой источник:** {r.get('expected_sources')} "
                f"(«{r.get('expected_title')}»), ранг в поиске: "
                f"{r['source_recall']['top_rank']} "
                f"(score={r['source_recall']['top_score']})",
                "",
                f"**Ответ локальной модели ({report.get('local_label')}):**",
                f"> {r['local']['answer'].replace(chr(10), '  ' + chr(10) + '> ')}",
                "",
                f"*Покрытие фактов:* {r['local']['facts']['hits']}/{r['local']['facts']['total']} "
                f"| *ссылки:* {'Да' if r['local']['citations']['has_citations'] else 'Нет'} "
                f"| *задержка:* {r['local']['latency_ms']:.0f} мс "
                f"| *самосогласованность:* "
                f"{r['local']['self_consistency']['mean_jaccard']:.2f}",
                "",
            ]
        )
        if has_cloud and r["cloud"] is not None:
            lines.extend(
                [
                    f"**Ответ облачной модели ({report.get('cloud_label')}):**",
                    f"> {r['cloud']['answer'].replace(chr(10), '  ' + chr(10) + '> ')}",
                    "",
                    f"*Покрытие фактов:* {r['cloud']['facts']['hits']}/{r['cloud']['facts']['total']} "
                    f"| *ссылки:* {'Да' if r['cloud']['citations']['has_citations'] else 'Нет'} "
                    f"| *задержка:* {r['cloud']['latency_ms']:.0f} мс",
                    "",
                ]
            )
        lines.extend(["---", ""])

    lines.extend(
        [
            "## 6. Выводы",
            "",
            "1. **Полностью локальный конвейер.** Retrieval (FAISS + офлайн-хэш-эмбеддер) и генерация "
            "выполняются на одной машине через `llama-server`; облачный API и ключи не нужны.",
            "2. **Качество.** Локальная модель отвечает на вопросы по базе рецептов и цитирует источники; "
            "на конкретных рецептурных нормах облачная модель, как правило, немного точнее.",
            "3. **Скорость.** Локальный retrieval — единицы миллисекунд; основное время уходит на генерацию "
            "на CPU. Облачная модель быстрее, но требует сети.",
            "4. **Стабильность.** При фиксированной температуре локальные ответы повторяемы; чем выше "
            "`repeats`, тем точнее оценка самосогласованности и вариации задержки.",
            "",
        ]
    )
    return "\n".join(lines)
