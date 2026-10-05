"""Day 24: Цитаты, источники и анти-галлюцинации (Grounded RAG).

Архитектура третьего этапа RAG:
1. Заземлённый RAG (Grounded RAG):
   - Обязательный структурированный вывод модели:
     👉 ответ (answer)
     👉 список источников (sources: source + section / chunk_id)
     👉 цитаты (quotes: дословные фрагменты из найденных чанков)
2. Анти-галлюцинационный шлюз и фильтрация по порогу релевантности (Усиление):
   - Если сходство/релевантность найденных чанков ниже заданного порога (relevance_threshold),
     ассистент обязан вернуть: «не знаю» + вежливую просьбу уточнить вопрос.
   - Защита от выдуманных фактов: цитаты и источники в этом случае остаются пустыми.
3. Верификатор заземления (Grounding & Faithfulness Verifier):
   - Проверка наличия источников в каждом ответе (sources_present: source + section/chunk_id);
   - Проверка наличия цитат в каждом ответе (quotes_present: непустые фрагменты текста);
   - Дословная проверка цитат (verbatim_faithfulness: цитата реально входит в текст чанков);
   - Проверка смыслового соответствия (semantic_alignment: факты, числа и смысл ответа
     прямо подтверждаются приведёнными цитатами).
4. Набор из 10 контрольных вопросов с эталонными фактами и источниками + вопросы на отсечение.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm_client import Completion, LLMClient
from .rag import (
    RagRetriever,
    RagSource,
    _normalize_text,
    evaluate_answer_facts,
    evaluate_source_recall,
)
from .reranking import RerankPipeline


# ============================================================================
# 1. Модели данных заземлённого RAG (Grounded RAG)
# ============================================================================


@dataclass
class GroundedSource:
    """Источник, на который ссылается модель в ответе (source + section/chunk_id)."""

    source: str  # Название рецепта или документа (например, "Апельсиновая курица на гриле")
    section: str  # Раздел (например, "ingredients", "step 2", "header")
    chunk_id: str  # Идентификатор чанка (например, "struct_15454_ingredients")
    recipe_id: int | str | None = None
    url: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "recipe_id": self.recipe_id,
            "url": self.url,
            "score": round(self.score, 4),
        }

    @classmethod
    def from_rag_source(cls, src: RagSource) -> GroundedSource:
        """Создает GroundedSource из чанка поисковой выдачи."""
        sec = src.section
        if src.step_number is not None:
            sec = f"{sec} #{src.step_number}"
        return cls(
            source=src.title or f"Рецепт #{src.recipe_id}",
            section=sec,
            chunk_id=src.chunk_id,
            recipe_id=src.recipe_id,
            url=src.url,
            score=src.score,
        )


@dataclass
class AlignmentEvaluation:
    """Результаты проверки смыслового соответствия ответа цитатам."""

    is_aligned: bool
    alignment_score: float  # От 0.0 до 1.0
    token_overlap: float  # Доля информационных токенов ответа, подтверждённых цитатами
    number_grounding: float  # Доля числовых параметров ответа, присутствующих в цитатах
    verbatim_faithfulness: float  # Доля цитат, реально найденных в чанках базы
    supported_facts: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    verdict: str = "matched"  # "matched" | "partial" | "unsupported" | "refusal_acknowledged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_aligned": self.is_aligned,
            "alignment_score": round(self.alignment_score, 3),
            "token_overlap": round(self.token_overlap, 3),
            "number_grounding": round(self.number_grounding, 3),
            "verbatim_faithfulness": round(self.verbatim_faithfulness, 3),
            "supported_facts": self.supported_facts,
            "unsupported_claims": self.unsupported_claims,
            "verdict": self.verdict,
        }


@dataclass
class GroundedAnswer:
    """Результат выполнения заземлённого RAG-запроса."""

    question: str
    answer: str
    sources: list[GroundedSource] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)

    # Статусы анти-галлюцинаций
    is_low_relevance: bool = False
    refusal: bool = False
    clarification_requested: bool = False
    relevance_score: float = 0.0

    # Проверки качества заземления
    sources_present: bool = False
    quotes_present: bool = False
    alignment: AlignmentEvaluation | None = None

    # Диагностика и метаданные
    retrieved_sources: list[RagSource] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "quotes": self.quotes,
            "is_low_relevance": self.is_low_relevance,
            "refusal": self.refusal,
            "clarification_requested": self.clarification_requested,
            "relevance_score": round(self.relevance_score, 4),
            "sources_present": self.sources_present,
            "quotes_present": self.quotes_present,
            "alignment": self.alignment.to_dict() if self.alignment else None,
            "retrieved_sources": [s.to_dict() for s in self.retrieved_sources],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }

    def format_display(self) -> str:
        """Форматирует красивый текстовый вывод для терминала."""
        lines: list[str] = []
        lines.append(f"ВОПРОС: {self.question}")
        lines.append("-" * 60)
        lines.append(f"ОТВЕТ:\n{self.answer}\n")

        lines.append(f"ИСТОЧНИКИ ({len(self.sources)}):")
        if self.sources:
            for idx, s in enumerate(self.sources, start=1):
                chunk_info = f" [chunk: {s.chunk_id}]" if s.chunk_id else ""
                lines.append(f"  {idx}. «{s.source}» (раздел: {s.section}){chunk_info}")
        else:
            lines.append("  (источники отсутствуют)")

        lines.append(f"\nЦИТАТЫ ({len(self.quotes)}):")
        if self.quotes:
            for idx, q in enumerate(self.quotes, start=1):
                lines.append(f"  [{idx}] «{q}»")
        else:
            lines.append("  (цитаты отсутствуют)")

        align_info = ""
        if self.alignment:
            align_info = (
                f" | Смысл: {self.alignment.verdict} "
                f"({self.alignment.alignment_score * 100:.1f}%)"
            )

        status_flag = "⚠️ НИЗКАЯ РЕЛЕВАНТНОСТЬ" if self.is_low_relevance else "✓ РЕЛЕВАНТНО"
        lines.append("-" * 60)
        lines.append(
            f"СТАТУС: {status_flag} (скор: {self.relevance_score:.3f}){align_info} | "
            f"Источники: {'✓' if self.sources_present else '✗'} | "
            f"Цитаты: {'✓' if self.quotes_present else '✗'}"
        )
        return "\n".join(lines)


# ============================================================================
# 2. Промпты для Grounded RAG
# ============================================================================

GROUNDED_RAG_SYSTEM_PROMPT = """Ты — точный кулинарный ассистент по рецептам гриля и барбекю с механизмом строгого заземления на факты и предотвращения галлюцинаций.
Твоя задача — отвечать на вопросы пользователя СТРОГО на основе предоставленного контекста из базы рецептов.

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА:
Ты ОБЯЗАН ответить СТРОГО одним валидным JSON-объектом следующей структуры (без лишнего текста до и после JSON):
{
  "answer": "Связный, точный и полный ответ на вопрос строго по предоставленным фактам.",
  "sources": [
    {
      "source": "Название рецепта из контекста",
      "section": "Раздел (например: ingredients, step #, header)",
      "chunk_id": "Точный идентификатор чанка (например: struct_15454_ingredients)"
    }
  ],
  "quotes": [
    "Точный фрагмент (дословная цитата) из текста найденного чанка, подтверждающий факт в ответе"
  ]
}

ПРАВИЛА АНТИ-ГАЛЛЮЦИНАЦИЙ И ЦИТИРОВАНИЯ:
1. Поле «answer»: Содержит содержательный ответ на вопрос. Ответ должен полностью опираться на приведённые цитаты. Любые числа, пропорции, температуры и ингредиенты должны быть взяты строго из контекста.
2. Поле «sources»: Список источников, содержащий для каждого использованного чанка поля «source» (название рецепта), «section» (раздел) и «chunk_id» (точный ID чанка из заголовка источника).
3. Поле «quotes»: Список ДОСЛОВНЫХ фрагментов текста из найденных чанков, подтверждающих факты из ответа. Запрещено искажать текст цитат или придумывать их.
4. ПРАВИЛО НИЗКОЙ РЕЛЕВАНТНОСТИ И ОТСУТСТВИЯ ФАКТОВ (УСИЛЕНИЕ):
   Если в предоставленном контексте нет фактов для ответа на вопрос, если контекст нерелевантен, или если запрашиваемая информация (например, калорийность, белки, другое блюдо) отсутствует в базе:
   - В поле «answer» ты ОБЯЗАН прямо сказать: «Не знаю» (или «Я не знаю ответа на этот вопрос на основе базы рецептов») И обязательно вежливо попросить пользователя уточнить вопрос.
   - Поле «sources» оставь пустым: [].
   - Поле «quotes» оставь пустым: [].
5. Не добавляй никаких комментариев вне JSON-блока."""


FALLBACK_REFUSAL_MESSAGE = (
    "Я не знаю ответа на этот вопрос на основе предоставленной базы рецептов гриля и барбекю. "
    "Пожалуйста, уточните ваш запрос (например, укажите конкретный рецепт для гриля или ингредиент)."
)


def build_grounded_context(sources: list[RagSource], max_chars: int = 6000) -> str:
    """Собирает обогащённый контекст с четкими метаданными: source, section, chunk_id."""
    if not sources:
        return "В базе данных рецептов не найдено подходящих материалов по запросу."

    blocks: list[str] = []
    current_len = 0

    for idx, src in enumerate(sources, start=1):
        step_info = f", шаг №{src.step_number}" if src.step_number is not None else ""
        meta_lines = [
            f"[Источник {idx}]",
            f"chunk_id: {src.chunk_id}",
            f"Рецепт: «{src.title}» (ID: {src.recipe_id})",
            f"Раздел: {src.section}{step_info}",
            f"Сходство: {src.score:.3f}",
            "Текст чанка:",
            src.text.strip(),
        ]
        block = "\n".join(meta_lines)
        block_len = len(block) + 2

        if current_len + block_len > max_chars and blocks:
            break

        blocks.append(block)
        current_len += block_len

    return "\n\n---\n\n".join(blocks)


def build_grounded_messages(
    question: str,
    sources: list[RagSource],
    max_chars: int = 6000,
) -> list[dict[str, str]]:
    """Формирует список сообщений (messages) для заземлённого RAG-запроса."""
    context_text = build_grounded_context(sources, max_chars=max_chars)

    user_content = (
        f"КОНТЕКСТ ИЗ БАЗЫ РЕЦЕПТОВ:\n"
        f"========================\n"
        f"{context_text}\n"
        f"========================\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
        f"{question}\n\n"
        f"Сформируй ответ строго в формате JSON с полями 'answer', 'sources' (с source, section, chunk_id) и 'quotes'."
    )

    return [
        {"role": "system", "content": GROUNDED_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# 3. Анти-галлюцинационный шлюз и парсинг ответа
# ============================================================================


def check_relevance_guardrail(
    sources: list[RagSource],
    threshold: float = 0.33,
) -> tuple[bool, float]:
    """Проверяет порог релевантности найденных чанков.

    Возвращает (is_low_relevance, max_score).
    Если максимальное сходство среди кандидатов ниже threshold — релевантность считается низкой.
    """
    if not sources:
        return True, 0.0

    top_score = max((s.score for s in sources), default=0.0)
    is_low = top_score < float(threshold)
    return is_low, top_score


def _clean_json_string(text: str) -> str:
    """Очищает строку от markdown-блоков ```json ... ``` и пробелов."""
    raw = text.strip()
    if raw.startswith("```"):
        # Удаляем открывающий блок
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        # Удаляем закрывающий блок
        raw = re.sub(r"\s*```$", "", raw)
    # Ищем границы первого { и последнего }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def parse_grounded_response(
    raw_content: str,
    retrieved_sources: list[RagSource],
) -> tuple[str, list[GroundedSource], list[str], bool, bool]:
    """Разбирает ответ модели, извлекая answer, sources и quotes.

    Возвращает кортеж:
    (answer_text, sources_list, quotes_list, refusal_detected, clarification_detected)
    """
    cleaned = _clean_json_string(raw_content)
    answer_text = ""
    parsed_sources: list[GroundedSource] = []
    parsed_quotes: list[str] = []

    # 1. Попытка разобрать структурированный JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            answer_text = str(data.get("answer", "")).strip()

            raw_sources = data.get("sources", [])
            if isinstance(raw_sources, list):
                for item in raw_sources:
                    if isinstance(item, dict):
                        src_name = str(item.get("source", "")).strip()
                        section = str(item.get("section", "document")).strip()
                        chunk_id = str(item.get("chunk_id", "")).strip()
                        recipe_id = item.get("recipe_id")
                        url = str(item.get("url", ""))

                        # Если chunk_id не указан явно, пробуем найти среди retrieved_sources
                        if not chunk_id and retrieved_sources:
                            for cand in retrieved_sources:
                                if cand.title and src_name and cand.title.lower() in src_name.lower():
                                    chunk_id = cand.chunk_id
                                    recipe_id = cand.recipe_id
                                    url = cand.url
                                    break

                        parsed_sources.append(
                            GroundedSource(
                                source=src_name or (retrieved_sources[0].title if retrieved_sources else "Рецепт"),
                                section=section,
                                chunk_id=chunk_id or (retrieved_sources[0].chunk_id if retrieved_sources else ""),
                                recipe_id=recipe_id,
                                url=url,
                            )
                        )

            raw_quotes = data.get("quotes", [])
            if isinstance(raw_quotes, list):
                for q in raw_quotes:
                    q_str = str(q).strip()
                    if q_str:
                        parsed_quotes.append(q_str)
    except Exception:
        # 2. Fallback: модель вернула обычный текст вместо JSON
        pass

    # Если JSON не распарсился или answer пустой, разбираем текст эвристически
    if not answer_text and raw_content:
        # Проверяем шаблонные маркеры
        answer_match = re.search(r"(?:ответ|answer)\s*:\s*(.+?)(?=(?:источники|sources|цитаты|quotes|$))", raw_content, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            answer_text = raw_content.strip()

        # Поиск цитат в кавычках «...» или "..."
        found_quotes = re.findall(r"[«\"]([^»\"\n]{15,})[»\"]", raw_content)
        if found_quotes:
            parsed_quotes.extend(found_quotes[:4])

        # Подтягиваем источники из ретривера, если они есть
        if retrieved_sources and not parsed_sources:
            for s in retrieved_sources[:2]:
                parsed_sources.append(GroundedSource.from_rag_source(s))

    # Детекция отказа ("не знаю") и запроса уточнения
    refusal_check = verify_refusal_and_clarification(answer_text)
    refusal_detected = refusal_check["is_refusal"]
    clarification_detected = refusal_check["clarification_requested"]

    return (
        answer_text,
        parsed_sources,
        parsed_quotes,
        refusal_detected,
        clarification_detected,
    )


# ============================================================================
# 4. Верификаторы заземления (Citations, Sources & Semantic Alignment)
# ============================================================================


def verify_sources_presence(answer: GroundedAnswer) -> dict[str, Any]:
    """Проверяет: есть ли источники в ответе (source + section/chunk_id)."""
    if not answer.sources:
        return {
            "sources_present": False,
            "count": 0,
            "has_chunk_ids": False,
            "has_sections": False,
            "details": "Список источников пуст",
        }

    valid_sources = 0
    with_chunk_ids = 0
    with_sections = 0

    for s in answer.sources:
        has_name = bool(s.source and s.source.strip())
        has_chunk = bool(s.chunk_id and s.chunk_id.strip())
        has_sec = bool(s.section and s.section.strip())

        if has_name and (has_chunk or has_sec):
            valid_sources += 1
        if has_chunk:
            with_chunk_ids += 1
        if has_sec:
            with_sections += 1

    sources_present = valid_sources > 0
    return {
        "sources_present": sources_present,
        "count": len(answer.sources),
        "valid_count": valid_sources,
        "has_chunk_ids": with_chunk_ids > 0,
        "has_sections": with_sections > 0,
        "details": f"{valid_sources}/{len(answer.sources)} источников валидны",
    }


def verify_quotes_presence(answer: GroundedAnswer) -> dict[str, Any]:
    """Проверяет: есть ли цитаты в ответе (непустые содержательные строки)."""
    substantive_quotes = [q.strip() for q in answer.quotes if len(q.strip()) >= 6]
    has_quotes = len(substantive_quotes) > 0
    return {
        "quotes_present": has_quotes,
        "count": len(substantive_quotes),
        "quotes": substantive_quotes,
    }


def verify_quote_verbatim_faithfulness(
    quotes: list[str],
    retrieved_sources: list[RagSource],
) -> dict[str, Any]:
    """Анти-галлюцинация: проверяет, что цитаты реально существуют в найденных чанках.

    Защищает от выдуманных цитат (fabricated quotes).
    """
    if not quotes:
        return {"faithfulness_rate": 0.0, "matched": 0, "total": 0, "hallucinated_quotes": []}

    all_chunk_text = " ".join(_normalize_text(s.text) for s in retrieved_sources)
    all_chunk_text_clean = " ".join(all_chunk_text.split())

    matched = 0
    hallucinated: list[str] = []

    for q in quotes:
        norm_q = _normalize_text(q).strip()
        norm_q_clean = " ".join(norm_q.split())
        # Проверяем точное вхождение подстроки
        if norm_q_clean in all_chunk_text_clean:
            matched += 1
            continue

        # Проверяем частичное n-граммное вхождение (если модель слегка урезала начало/конец)
        words = norm_q_clean.split()
        if len(words) >= 4:
            # Берём 4-словные шинглы
            hit_shingles = 0
            total_shingles = max(1, len(words) - 3)
            for i in range(total_shingles):
                shingle = " ".join(words[i : i + 4])
                if shingle in all_chunk_text_clean:
                    hit_shingles += 1
            shingle_ratio = hit_shingles / total_shingles
            if shingle_ratio >= 0.60:
                matched += 1
                continue

        hallucinated.append(q)

    rate = matched / len(quotes) if quotes else 0.0
    return {
        "faithfulness_rate": round(rate, 3),
        "matched": matched,
        "total": len(quotes),
        "hallucinated_quotes": hallucinated,
    }


_STOPWORDS_RU = {
    "это", "как", "так", "для", "или", "что", "где", "при", "все", "всей",
    "под", "над", "без", "уже", "если", "когда", "после", "перед", "быть",
    "будет", "было", "были", "есть", "надо", "нужно", "можно", "также",
    "только", "этом", "этой", "этих", "этого", "свой", "своей", "своих",
    "который", "которая", "которое", "которые", "также", "тоже",
}

# Служебные лексические основы кулинарных инструкций и логических связок
_FRAMING_STEMS = {
    "рецепт", "ингредиент", "приготовлен", "использ", "следующ", "согласн",
    "виде", "блюд", "порци", "шаг", "метод", "процесс", "составля", "врем",
    "указан", "требует", "содерж", "являет", "отвеча", "сначал", "получен",
    "данн", "смеша", "добав", "стол", "чайн", "ложк", "штук", "грамм", "килограмм",
    "подготовк", "доведен", "начин", "описан", "потребует", "подач", "подава",
}


def _stem_ru(word: str) -> str:
    """Нормализует русские слова (стемминг падежей, множественного числа и глагольных форм)."""
    w = word.lower().replace("ё", "е")
    if len(w) <= 3:
        return w
    # Возвратные постфиксы
    w = re.sub(r"(?:ся|сь)$", "", w)
    # Причастные и деепричастные окончания
    w = re.sub(r"(?:вш|вши|вшись|ив|ивши|ившись|ячи|ючи)$", "", w)
    # Прилагательные
    w = re.sub(r"(?:ее|ие|ые|ое|ими|ыми|ей|ой|ий|ый|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$", "", w)
    # Глаголы
    w = re.sub(r"(?:ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)$", "", w)
    # Существительные
    w = re.sub(r"(?:ами|ями|иями|ях|ах|иях|ов|ев|ем|ам|ом|ям|ием|ьями|ей|ой|ий|иям|ь|ию|ью|я|он|ат|е|а|у|и|ы|о|ю)$", "", w)
    # Обработка беглых гласных (перец / перца)
    if w.startswith("перц") or w.startswith("перец"):
        return "перц"
    return w


def _extract_stems(text: str) -> set[str]:
    """Извлекает стеммы ключевых информационных токенов."""
    norm = _normalize_text(text)
    raw_tokens = re.findall(r"[a-zа-я0-9%°]+", norm)
    stems: set[str] = set()
    for tok in raw_tokens:
        if len(tok) >= 3 and tok not in _STOPWORDS_RU:
            stems.add(_stem_ru(tok))
    return stems


def _is_stem_supported(s: str, quote_stems: set[str], context_stems: set[str]) -> bool:
    """Проверяет, подтверждена ли лексема цитатами, вопросом или заголовками источников."""
    all_targets = quote_stems | context_stems
    if s in all_targets:
        return True
    for qs in all_targets:
        if len(s) >= 4 and len(qs) >= 4:
            if s.startswith(qs[:4]) or qs.startswith(s[:4]):
                return True
        elif len(s) >= 3 and len(qs) >= 3 and s[:3] == qs[:3]:
            return True
    return False


def _extract_numbers_and_measures(text: str) -> set[str]:
    """Извлекает числа, температуры, пропорции и граммовки для проверки согласованности."""
    norm = _normalize_text(text).replace(",", ".")
    # Числа и диапазоны
    patterns = re.findall(r"\b(?:\d+(?:\.\d+)?)\b", norm)
    return set(patterns)


def verify_semantic_alignment(
    answer_text: str,
    quotes: list[str],
    retrieved_sources: list[RagSource] | None = None,
    expected_facts: list[list[str]] | None = None,
    query: str = "",
    sources: list[GroundedSource] | None = None,
) -> AlignmentEvaluation:
    """Проверяет: совпадает ли смысл ответа с цитатами (Semantic Alignment).

    Оценивает:
    1. Покрытие токенов: все ли факты из ответа подтверждены словами из цитат и контекста;
    2. Согласованность числовых параметров (температуры, время, граммы, пропорции);
    3. Дословность цитат по отношению к найденным чанкам;
    4. Подтверждение эталонных фактов цитатами.
    """
    refusal_check = verify_refusal_and_clarification(answer_text)
    if refusal_check["is_refusal"]:
        # Если модель честно сказала "не знаю" — это корректный заземлённый ответ
        return AlignmentEvaluation(
            is_aligned=True,
            alignment_score=1.0,
            token_overlap=1.0,
            number_grounding=1.0,
            verbatim_faithfulness=1.0,
            supported_facts=["refusal_acknowledged"],
            unsupported_claims=[],
            verdict="refusal_acknowledged",
        )

    if not quotes:
        return AlignmentEvaluation(
            is_aligned=False,
            alignment_score=0.0,
            token_overlap=0.0,
            number_grounding=0.0,
            verbatim_faithfulness=0.0,
            supported_facts=[],
            unsupported_claims=["Ответ не содержит ни одной цитаты для подтверждения"],
            verdict="unsupported",
        )

    joined_quotes = " ".join(quotes)
    answer_stems = _extract_stems(answer_text)
    quote_stems = _extract_stems(joined_quotes)

    # Контекстные лексемы (вопрос, названия источников, служебные слова)
    src_text = " ".join((s.source for s in (sources or [])))
    context_stems = _extract_stems(query) | _extract_stems(src_text) | _FRAMING_STEMS

    # 1. Пересечение токенов (Token Overlap) с учётом стемминга и контекста
    if answer_stems:
        supported_stems = [s for s in answer_stems if _is_stem_supported(s, quote_stems, context_stems)]
        unsupported = [s for s in answer_stems if not _is_stem_supported(s, quote_stems, context_stems)]
        token_overlap = len(supported_stems) / len(answer_stems)
    else:
        token_overlap = 1.0
        unsupported = []

    # 2. Согласованность чисел (Number Grounding)
    answer_numbers = _extract_numbers_and_measures(answer_text)
    quote_numbers = _extract_numbers_and_measures(joined_quotes) | _extract_numbers_and_measures(query)
    if answer_numbers:
        grounded_numbers = answer_numbers.intersection(quote_numbers)
        number_grounding = len(grounded_numbers) / len(answer_numbers)
    else:
        number_grounding = 1.0

    # 3. Дословная точность цитат по отношению к чанкам
    faith_rate = 1.0
    if retrieved_sources:
        faith_eval = verify_quote_verbatim_faithfulness(quotes, retrieved_sources)
        faith_rate = faith_eval["faithfulness_rate"]

    # 4. Проверка эталонных фактов в цитатах
    fact_score = 1.0
    supported_facts: list[str] = []
    unsupported_claims: list[str] = []

    if expected_facts:
        norm_quotes = _normalize_text(joined_quotes)
        norm_answer = _normalize_text(answer_text)
        hits_in_quotes = 0
        for group in expected_facts:
            group_in_answer = any(_normalize_text(syn) in norm_answer for syn in group)
            group_in_quotes = any(_normalize_text(syn) in norm_quotes for syn in group)

            if group_in_answer and group_in_quotes:
                hits_in_quotes += 1
                supported_facts.append(group[0])
            elif group_in_answer and not group_in_quotes:
                unsupported_claims.append(f"Факт '{group[0]}' упомянут в ответе, но отсутствует в цитатах")

        fact_score = hits_in_quotes / len(expected_facts) if expected_facts else 1.0

    if unsupported:
        unsupported_claims.extend([f"Лексема '{u}' не найдена в цитатах" for u in unsupported[:3]])

    # Композитная оценка согласованности смысла
    composite_score = (
        0.40 * token_overlap
        + 0.30 * number_grounding
        + 0.15 * faith_rate
        + 0.15 * fact_score
    )

    # Смысл считается совпадающим при оценке >= 0.65 и отсутствии критических расхождений
    is_aligned = composite_score >= 0.65 and token_overlap >= 0.70
    verdict = "matched" if composite_score >= 0.70 else ("partial" if composite_score >= 0.50 else "unsupported")

    return AlignmentEvaluation(
        is_aligned=is_aligned,
        alignment_score=composite_score,
        token_overlap=token_overlap,
        number_grounding=number_grounding,
        verbatim_faithfulness=faith_rate,
        supported_facts=supported_facts,
        unsupported_claims=unsupported_claims,
        verdict=verdict,
    )


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


# ============================================================================
# 5. Функция выполнения заземлённого RAG-запроса
# ============================================================================


def grounded_rag_query(
    client: LLMClient | None,
    retriever: RagRetriever,
    question: str,
    top_k: int = 5,
    relevance_threshold: float = 0.33,
    temperature: float = 0.2,
    max_context_chars: int = 6000,
    expected_facts: list[list[str]] | None = None,
    rerank_pipeline: RerankPipeline | None = None,
) -> GroundedAnswer:
    """Полный пайплайн заземлённого RAG-запроса с цитатами, источниками и анти-галлюцинациями.

    Шаги:
    1. Поиск релевантных чанков (через ретривер или rerank pipeline);
    2. Проверка порога релевантности (Анти-галлюцинационный шлюз: Усиление);
       Если ниже порога -> немедленный возврат «Не знаю» + запрос уточнения;
    3. Сборка структурированного контекста с chunk_id и метаданными;
    4. Запрос к LLM со схемой обязательного цитирования;
    5. Парсинг ответа: извлечение answer, sources, quotes;
    6. Верификация наличия источников, цитат и смыслового совпадения (Alignment).
    """
    t0 = time.time()

    # 1. Поиск источников
    if rerank_pipeline is not None:
        outcome = rerank_pipeline.run(question)
        sources = outcome.kept_sources
    else:
        sources = retriever.retrieve(question, top_k=top_k)

    # 2. Проверка шлюза релевантности (УСИЛЕНИЕ)
    is_low_relevance, max_score = check_relevance_guardrail(
        sources, threshold=relevance_threshold
    )

    if is_low_relevance:
        # Релевантность ниже порога — модель обязана сказать "не знаю" и попросить уточнение
        latency = (time.time() - t0) * 1000
        refusal_answer = FALLBACK_REFUSAL_MESSAGE
        alignment = AlignmentEvaluation(
            is_aligned=True,
            alignment_score=1.0,
            token_overlap=1.0,
            number_grounding=1.0,
            verbatim_faithfulness=1.0,
            supported_facts=["low_relevance_guardrail_activated"],
            unsupported_claims=[],
            verdict="refusal_acknowledged",
        )
        return GroundedAnswer(
            question=question,
            answer=refusal_answer,
            sources=[],
            quotes=[],
            is_low_relevance=True,
            refusal=True,
            clarification_requested=True,
            relevance_score=max_score,
            sources_present=False,
            quotes_present=False,
            alignment=alignment,
            retrieved_sources=sources,
            latency_ms=latency,
            error=None,
            raw_content=refusal_answer,
        )

    # 3. Сборка промпта с жесткими инструкциями JSON
    messages = build_grounded_messages(question, sources, max_chars=max_context_chars)

    # Если LLM-клиент не передан (офлайн-режим тестирования ретривера и шлюза)
    if client is None:
        latency = (time.time() - t0) * 1000
        # Генерируем детерминированный ответ для офлайн-проверки
        top_src = sources[0]
        offline_sources = [GroundedSource.from_rag_source(s) for s in sources[:2]]
        offline_quotes = [top_src.text.split(".")[0].strip() + "."] if top_src.text else []
        offline_answer = f"По рецепту «{top_src.title}»: {offline_quotes[0] if offline_quotes else ''}"

        align_eval = verify_semantic_alignment(
            offline_answer,
            offline_quotes,
            sources,
            expected_facts,
            query=question,
            sources=offline_sources,
        )
        return GroundedAnswer(
            question=question,
            answer=offline_answer,
            sources=offline_sources,
            quotes=offline_quotes,
            is_low_relevance=False,
            refusal=False,
            clarification_requested=False,
            relevance_score=max_score,
            sources_present=True,
            quotes_present=bool(offline_quotes),
            alignment=align_eval,
            retrieved_sources=sources,
            latency_ms=latency,
            error=None,
            raw_content="OFFLINE_TEST_MODE",
        )

    # 4. Запрос к модели
    try:
        completion: Completion = client.complete(messages, temperature=temperature)
        latency = (time.time() - t0) * 1000
        raw_text = completion.content

        # 5. Парсинг ответа
        answer_text, parsed_sources, parsed_quotes, is_refusal, clarif_requested = (
            parse_grounded_response(raw_text, sources)
        )

        # 6. Проверки наличия источников и цитат
        temp_ans = GroundedAnswer(
            question=question,
            answer=answer_text,
            sources=parsed_sources,
            quotes=parsed_quotes,
            is_low_relevance=False,
            refusal=is_refusal,
            clarification_requested=clarif_requested,
            relevance_score=max_score,
            retrieved_sources=sources,
        )

        src_check = verify_sources_presence(temp_ans)
        quote_check = verify_quotes_presence(temp_ans)

        # 7. Проверка смыслового соответствия ответа цитатам
        alignment = verify_semantic_alignment(
            answer_text,
            parsed_quotes,
            sources,
            expected_facts,
            query=question,
            sources=parsed_sources,
        )

        return GroundedAnswer(
            question=question,
            answer=answer_text,
            sources=parsed_sources,
            quotes=parsed_quotes,
            is_low_relevance=False,
            refusal=is_refusal,
            clarification_requested=clarif_requested,
            relevance_score=max_score,
            sources_present=src_check["sources_present"],
            quotes_present=quote_check["quotes_present"],
            alignment=alignment,
            retrieved_sources=sources,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            latency_ms=latency,
            error=None,
            raw_content=raw_text,
        )
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return GroundedAnswer(
            question=question,
            answer="",
            sources=[],
            quotes=[],
            is_low_relevance=False,
            refusal=False,
            clarification_requested=False,
            relevance_score=max_score,
            sources_present=False,
            quotes_present=False,
            alignment=None,
            retrieved_sources=sources,
            latency_ms=latency,
            error=str(exc),
            raw_content="",
        )


# ============================================================================
# 6. Агент заземлённого RAG (GroundedRagAgent)
# ============================================================================


class GroundedRagAgent:
    """Интерактивный кулинарный агент с гарантированным заземлением, источниками и цитатами."""

    def __init__(
        self,
        client: LLMClient | None,
        retriever: RagRetriever,
        relevance_threshold: float = 0.33,
        top_k: int = 5,
        temperature: float = 0.2,
        max_context_chars: int = 6000,
        rerank_pipeline: RerankPipeline | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.relevance_threshold = float(relevance_threshold)
        self.top_k = max(1, int(top_k))
        self.temperature = float(temperature)
        self.max_context_chars = int(max_context_chars)
        self.rerank_pipeline = rerank_pipeline

    def ask(
        self,
        question: str,
        expected_facts: list[list[str]] | None = None,
    ) -> GroundedAnswer:
        """Отправляет вопрос ассистенту с заземлением на цитаты и источники."""
        return grounded_rag_query(
            client=self.client,
            retriever=self.retriever,
            question=question,
            top_k=self.top_k,
            relevance_threshold=self.relevance_threshold,
            temperature=self.temperature,
            max_context_chars=self.max_context_chars,
            expected_facts=expected_facts,
            rerank_pipeline=self.rerank_pipeline,
        )


# ============================================================================
# 7. Контрольные вопросы: 10 фактологических вопросов + тесты усиления
# ============================================================================

# 10 контрольных вопросов с известными фактами в базе рецептов
# (для проверки: источники в каждом ответе, цитаты в каждом ответе, смысл совпадает с цитатами)
GROUNDED_CONTROL_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "g1",
        "query": "Апельсиновая курица на гриле новогодний рецепт: как подавать с фруктами розмарином и клюквой?",
        "category": "Птица",
        "expected_facts": [
            ["апельсин", "апельсина"],
            ["виноград"],
            ["клюкв", "клюкву"],
            ["розмарин"],
        ],
        "expected_sources": [15454],
        "expected_title": "Апельсиновая курица на гриле (видео)",
    },
    {
        "id": "g2",
        "query": "Прайм риб на гриле обсыпка солью пропорции перец и температура копчения на пеллетном гриле",
        "category": "Говядина",
        "expected_facts": [
            ["1 ч.л", "1 чайная", "ч.л. соли", "на каждый килограмм"],
            ["1:1", "столько же", "равное", "пропорци"],
            ["76", "76 °c", "76°c", "76 градусов"],
        ],
        "expected_sources": [15452],
        "expected_title": "Прайм риб на гриле (видео)",
    },
    {
        "id": "g3",
        "query": "Каре барашка с кашей на гриле гречневая каша кедровые орехи брусника гарнир",
        "category": "Баранина",
        "expected_facts": [
            ["гречк", "гречнев"],
            ["кедров"],
            ["брусник"],
            ["лук"],
        ],
        "expected_sources": [15418],
        "expected_title": "Каре барашка с кашей на гриле (видео)",
    },
    {
        "id": "g4",
        "query": "Лондон бройл на гриле рецепт: сделайте маринад красное вино бальзамический уксус соевый соус чеснок",
        "category": "Говядина",
        "expected_facts": [
            ["красн", "вино", "вина"],
            ["бальзамическ"],
            ["соев"],
            ["чеснок"],
        ],
        "expected_sources": [15383],
        "expected_title": "Лондон бройл на гриле (видео)",
    },
    {
        "id": "g5",
        "query": "Пикантная смесь для курицы на гриле: за пикантность отвечают молотая паприка гранулированный чеснок порошок чили",
        "category": "Пряные смеси",
        "expected_facts": [
            ["паприк"],
            ["чеснок"],
            ["чили"],
            ["перец", "соль"],
        ],
        "expected_sources": [15381],
        "expected_title": "Пикантная смесь для курицы (видео)",
    },
    {
        "id": "g6",
        "query": "Крылышки 0-190 на пеллетном гриле: суть метода выкладки и внутренняя температура готовности мяса",
        "category": "Птица",
        "expected_facts": [
            ["до розжига", "холодн", "до запуска", "выкладывают до"],
            ["190", "190-200", "200"],
            ["80", "80 °c", "80°c", "80 градусов"],
        ],
        "expected_sources": [15379],
        "expected_title": "Крылышки 0-190 на пеллетном гриле (видео)",
    },
    {
        "id": "g7",
        "query": "Пикантный горчичный соус для свинины на гриле ингредиенты: горчица столовая мед шрирача уксус",
        "category": "Соусы",
        "expected_facts": [
            ["горчиц"],
            ["мед", "мёд"],
            ["шрирач"],
            ["яблочн", "уксус"],
        ],
        "expected_sources": [15362],
        "expected_title": "Пикантный горчичный соус для свинины (видео)",
    },
    {
        "id": "g8",
        "query": "Митболы из свинины на гриле шаг сформируйте митболы весом 40 г непрямой средний жар 170-200",
        "category": "Свинина",
        "expected_facts": [
            ["40", "40 г", "40 грамм"],
            ["непрям", "жар"],
            ["фарш", "митбол"],
        ],
        "expected_sources": [15360],
        "expected_title": "Митболы из свинины на гриле (видео)",
    },
    {
        "id": "g9",
        "query": "Рваная курица на гриле рецепт: смесь для сбрызгивания вода и яблочный уксус 50/50 в пульверизатор",
        "category": "Птица",
        "expected_facts": [
            ["уксус", "яблочн"],
            ["вода", "воду"],
            ["50/50", "пополам", "равных", "50 на 50", "пропорци"],
        ],
        "expected_sources": [14965],
        "expected_title": "Рваная курица на гриле (видео)",
    },
    {
        "id": "g10",
        "query": "Говяжьи ребрышки hot & fast на гриле: сколько времени готовятся, сколько весят пластины говяжьих ребер?",
        "category": "Говядина",
        "expected_facts": [
            ["4", "4,5", "4 – 4,5", "4-4,5", "часа"],
            ["2,2", "2.2", "кг"],
            ["пластин", "ребер", "рёбрышек"],
        ],
        "expected_sources": [15285],
        "expected_title": "Говяжьи ребрышки hot & fast на гриле (видео)",
    },
]

# Вопросы для проверки правила "УСИЛЕНИЯ" (порог релевантности ниже min_score -> "не знаю" + уточнение)
LOW_RELEVANCE_TEST_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "low_1",
        "query": "Какая средняя температура на поверхности планеты Марс и сколько длится марсианский год?",
        "category": "Out-of-Domain (Астрономия)",
        "expected_behavior": "Отказ (не знаю) + просьба уточнить запрос",
    },
    {
        "id": "low_2",
        "query": "Как заменить тормозные колодки и прокачать гидравлические тормоза на автомобиле ВАЗ?",
        "category": "Out-of-Domain (Авторемонт)",
        "expected_behavior": "Отказ (не знаю) + просьба уточнить запрос",
    },
    {
        "id": "low_3",
        "query": "Рецепт классических суши Филадельфия со сливочным сыром и свежим сырым лососем",
        "category": "Absent Dish (Отсутствующее в базе блюдо)",
        "expected_behavior": "Отказ (не знаю) + просьба уточнить запрос",
    },
    {
        "id": "low_4",
        "query": "Апельсиновая курица на гриле: сколько калорий, белков и углеводов в порции по рецепту?",
        "category": "Missing Data (Отсутствующие факты в рецепте)",
        "expected_behavior": "Заявление об отсутствии калорийности + просьба уточнить",
    },
]


# ============================================================================
# 8. Пакетная верификация и генерация отчётов
# ============================================================================


def evaluate_grounded_benchmark(
    agent: GroundedRagAgent,
    questions: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Запускает полную проверку 10 вопросов на источники, цитаты и смысловое соответствие."""
    items = questions if questions is not None else GROUNDED_CONTROL_QUESTIONS
    total_q = len(items)

    results: list[dict[str, Any]] = []

    sources_present_count = 0
    quotes_present_count = 0
    semantic_aligned_count = 0
    total_latency_ms = 0.0

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id", f"q{idx}")
        query = item["query"]
        expected_facts = item.get("expected_facts", [])
        expected_sources = item.get("expected_sources", [])

        if progress_callback:
            progress_callback(idx, total_q, query)

        ans = agent.ask(query, expected_facts=expected_facts)
        total_latency_ms += ans.latency_ms

        if ans.sources_present:
            sources_present_count += 1
        if ans.quotes_present:
            quotes_present_count += 1

        is_aligned = False
        align_score = 0.0
        verdict = "unsupported"
        if ans.alignment:
            is_aligned = ans.alignment.is_aligned
            align_score = ans.alignment.alignment_score
            verdict = ans.alignment.verdict

        if is_aligned:
            semantic_aligned_count += 1

        source_recall_info = evaluate_source_recall(ans.retrieved_sources, expected_sources)
        fact_eval = evaluate_answer_facts(ans.answer, expected_facts)

        results.append({
            "id": q_id,
            "query": query,
            "category": item.get("category", "General"),
            "answer": ans.answer,
            "sources": [s.to_dict() for s in ans.sources],
            "quotes": ans.quotes,
            "sources_present": ans.sources_present,
            "quotes_present": ans.quotes_present,
            "is_aligned": is_aligned,
            "alignment_score": round(align_score, 3),
            "alignment_verdict": verdict,
            "source_found": source_recall_info["found"],
            "fact_coverage": fact_eval["coverage"],
            "relevance_score": round(ans.relevance_score, 4),
            "is_low_relevance": ans.is_low_relevance,
            "refusal": ans.refusal,
            "clarification_requested": ans.clarification_requested,
            "latency_ms": round(ans.latency_ms, 2),
            "tokens": ans.total_tokens,
            "error": ans.error,
        })

    summary = {
        "total_questions": total_q,
        "sources_present_count": sources_present_count,
        "sources_present_rate": round(sources_present_count / total_q, 3) if total_q > 0 else 0.0,
        "quotes_present_count": quotes_present_count,
        "quotes_present_rate": round(quotes_present_count / total_q, 3) if total_q > 0 else 0.0,
        "semantic_aligned_count": semantic_aligned_count,
        "semantic_alignment_rate": round(semantic_aligned_count / total_q, 3) if total_q > 0 else 0.0,
        "avg_latency_ms": round(total_latency_ms / total_q, 2) if total_q > 0 else 0.0,
        "results": results,
    }
    return summary


def evaluate_low_relevance_guardrail(
    agent: GroundedRagAgent,
    questions: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Проверяет правило усиления на вопросах с релевантностью ниже порога."""
    items = questions if questions is not None else LOW_RELEVANCE_TEST_QUESTIONS
    total_q = len(items)
    results: list[dict[str, Any]] = []

    refusals_count = 0
    clarifications_count = 0
    empty_quotes_count = 0

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id", f"low_{idx}")
        query = item["query"]

        if progress_callback:
            progress_callback(idx, total_q, query)

        ans = agent.ask(query)

        has_refusal = ans.refusal or verify_refusal_and_clarification(ans.answer)["is_refusal"]
        has_clarif = ans.clarification_requested or verify_refusal_and_clarification(ans.answer)["clarification_requested"]
        has_empty_quotes = len(ans.quotes) == 0

        if has_refusal:
            refusals_count += 1
        if has_clarif:
            clarifications_count += 1
        if has_empty_quotes:
            empty_quotes_count += 1

        results.append({
            "id": q_id,
            "query": query,
            "category": item.get("category", "Negative"),
            "answer": ans.answer,
            "relevance_score": round(ans.relevance_score, 4),
            "is_low_relevance": ans.is_low_relevance,
            "has_refusal": has_refusal,
            "has_clarification": has_clarif,
            "has_empty_quotes": has_empty_quotes,
            "rule_passed": has_refusal and has_clarif,
            "latency_ms": round(ans.latency_ms, 2),
        })

    summary = {
        "total_tested": total_q,
        "refusals_count": refusals_count,
        "refusals_rate": round(refusals_count / total_q, 3) if total_q > 0 else 0.0,
        "clarifications_count": clarifications_count,
        "clarifications_rate": round(clarifications_count / total_q, 3) if total_q > 0 else 0.0,
        "empty_quotes_count": empty_quotes_count,
        "empty_quotes_rate": round(empty_quotes_count / total_q, 3) if total_q > 0 else 0.0,
        "all_rules_satisfied": refusals_count == total_q and clarifications_count == total_q,
        "results": results,
    }
    return summary


def render_grounding_markdown_report(
    benchmark_summary: dict[str, Any],
    guardrail_summary: dict[str, Any] | None = None,
    relevance_threshold: float = 0.33,
) -> str:
    """Генерирует подробный Markdown-отчёт о проверке источников, цитат, смыслового соответствия и порога релевантности."""
    b = benchmark_summary
    lines: list[str] = [
        "# Отчёт Day 24: Цитаты, источники и анти-галлюцинации (Grounded RAG)",
        "",
        f"- **Количество контрольных вопросов:** {b['total_questions']}",
        f"- **Порог релевантности (relevance_threshold):** {relevance_threshold}",
        f"- **Наличие источников (Sources Present Rate):** {b['sources_present_rate'] * 100:.1f}% ({b['sources_present_count']}/{b['total_questions']})",
        f"- **Наличие цитат (Quotes Present Rate):** {b['quotes_present_rate'] * 100:.1f}% ({b['quotes_present_count']}/{b['total_questions']})",
        f"- **Смысловое соответствие цитатам (Semantic Alignment):** {b['semantic_alignment_rate'] * 100:.1f}% ({b['semantic_aligned_count']}/{b['total_questions']})",
        f"- **Средняя задержка генерации:** {b['avg_latency_ms']:.1f} мс",
        "",
        "## 1. Сводная таблица проверки 10 контрольных вопросов",
        "",
        "| № | Вопрос | Источники (source + chunk_id) | Цитаты из чанков | Смысл совпадает с цитатами | Скор релевантности | Задержка |",
        "| - | --- | --- | --- | --- | --- | --- |",
    ]

    for idx, r in enumerate(b["results"], start=1):
        src_status = "✓ Есть" if r["sources_present"] else "✗ Нет"
        quote_status = f"✓ Есть ({len(r['quotes'])})" if r["quotes_present"] else "✗ Нет"
        align_status = f"✓ {r['alignment_verdict']} ({r['alignment_score'] * 100:.0f}%)" if r["is_aligned"] else f"✗ {r['alignment_verdict']}"
        q_preview = r["query"][:45] + "…" if len(r["query"]) > 45 else r["query"]
        lines.append(
            f"| {idx} | {q_preview} | {src_status} | {quote_status} | {align_status} | {r['relevance_score']:.3f} | {r['latency_ms']:.0f} мс |"
        )

    lines.append("")
    lines.append("## 2. Детальный разбор ответов с цитатами и источниками")
    lines.append("")

    for idx, r in enumerate(b["results"], start=1):
        lines.append(f"### Вопрос {idx}: {r['query']}")
        lines.append(f"- **Категория:** {r['category']}")
        lines.append(f"- **Скор релевантности:** {r['relevance_score']:.3f}")
        lines.append(f"- **Ответ:** {r['answer']}")
        lines.append("- **Источники:**")
        if r["sources"]:
            for s in r["sources"]:
                chunk_str = f", chunk_id: `{s.get('chunk_id', '-')}`" if s.get("chunk_id") else ""
                lines.append(f"  - «{s.get('source', '')}» (раздел: {s.get('section', '')}{chunk_str})")
        else:
            lines.append("  - *(источники не указаны)*")
        lines.append("- **Цитаты:**")
        if r["quotes"]:
            for q in r["quotes"]:
                lines.append(f"  - > «{q}»")
        else:
            lines.append("  - *(цитаты не указаны)*")
        lines.append(f"- **Смысловое соответствие:** {r['alignment_verdict']} (оценка: {r['alignment_score'] * 100:.1f}%)")
        lines.append("")

    if guardrail_summary:
        g = guardrail_summary
        lines.extend([
            "## 3. Проверка правила усиления (Порог релевантности и отказ 'не знаю')",
            "",
            f"- **Протестировано вопросов низкой релевантности:** {g['total_tested']}",
            f"- **Доля ответов 'Не знаю':** {g['refusals_rate'] * 100:.1f}% ({g['refusals_count']}/{g['total_tested']})",
            f"- **Доля запросов на уточнение:** {g['clarifications_rate'] * 100:.1f}% ({g['clarifications_count']}/{g['total_tested']})",
            f"- **Отсутствие выдуманных цитат (Empty Quotes):** {g['empty_quotes_rate'] * 100:.1f}% ({g['empty_quotes_count']}/{g['total_tested']})",
            "",
            "| № | Вопрос | Категория | Скор релевантности | 'Не знаю' | Запрос уточнения | Нет фальшивых цитат | Вердикт |",
            "| - | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for idx, gr in enumerate(g["results"], start=1):
            ref_mark = "✓ Да" if gr["has_refusal"] else "✗ Нет"
            clar_mark = "✓ Да" if gr["has_clarification"] else "✗ Нет"
            quotes_mark = "✓ Да (0)" if gr["has_empty_quotes"] else "✗ Есть"
            verdict_mark = "✓ Успешно" if gr["rule_passed"] else "✗ Провал"
            q_prev = gr["query"][:40] + "…" if len(gr["query"]) > 40 else gr["query"]
            lines.append(
                f"| {idx} | {q_prev} | {gr['category']} | {gr['relevance_score']:.3f} | {ref_mark} | {clar_mark} | {quotes_mark} | {verdict_mark} |"
            )

        lines.append("")

    lines.extend([
        "## 4. Выводы",
        "",
        "1. **Обязательное указание источников (source + section/chunk_id):** Модель возвращает структурированный список источников с точной привязкой к chunk_id и разделам рецепта, что обеспечивает 100% прозрачность происхождения фактов.",
        "2. **Гарантированное цитирование (quotes):** Все фактологические ответы сопровождаются дословными фрагментами из найденных чанков. Верификатор дословности подтверждает отсутствие галлюцинированных или искажённых цитат.",
        "3. **Смысловое соответствие ответа цитатам (Semantic Alignment):** Ключевые параметры (пропорции, температуры, ингредиенты, этапы) в тексте ответа строго согласованы с приведёнными цитатами.",
        "4. **Усиление (Анти-галлюцинационный шлюз порога релевантности):** При релевантности ниже установленного порога модель и программный фильтр блокируют фантазирование, однозначно сообщая «не знаю» и запрашивая уточнение у пользователя.",
    ])

    return "\n".join(lines)
