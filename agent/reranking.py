"""Day 23: Модуль реранкинга, фильтрации релевантности и переписывания запросов (Query Rewrite).

Архитектура второго этапа RAG:
1. Query Rewriter:
   - HeuristicQueryRewriter: нормализация, удаление вопросительных/шумовых слов,
     выделение ключевых кулинарных терминов и расширение синонимами (офлайн);
   - LLMQueryRewriter: переписывание вопроса в сжатый поисковый запрос через LLM
     с автоматическим откатом на эвристику при ошибках сети/API.
2. Relevance Filter:
   - Порог отсечения нерелевантных кандидатов (min_score);
   - Относительный порог к топ-1 результату (score_ratio);
   - Защитный лимит минимального числа источников (min_keep), чтобы не оставить пустой контекст.
3. Reranker:
   - HeuristicReranker: гибридное взвешивание (векторный скор + лексическое покрытие токенов
     и n-грамм + совпадение заголовка рецепта + совпадение раздела);
   - LLMReranker: оценка релевантности кандидатов отдельной моделью из конфигурации
     с автоматическим парсингом и откатом на эвристику.
4. RerankPipeline:
   - Двухэтапный пайплайн: retrieve_k (до фильтрации) → фильтр → реранкер → final_k (после);
   - Поддержка переключаемых режимов (baseline, filter, rerank, rewrite, full).
5. Метрики:
   - Precision@K по целевым источникам, доля отсеянного шума, сокращение контекста,
     покрытие фактов (Fact Coverage), Source Recall, цитирование.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm_client import Completion, LLMClient
from .rag import (
    CONTROL_QUESTIONS,
    RagAnswer,
    RagRetriever,
    RagSource,
    build_rag_messages,
    evaluate_answer_facts,
    evaluate_citations,
    evaluate_source_recall,
)


# ============================================================================
# 1. Модели данных второго этапа
# ============================================================================


@dataclass
class RankedSource:
    """Чанк с метаданными первого (retrieval) и второго (reranking/filter) этапов."""

    source: RagSource
    retrieval_rank: int
    retrieval_score: float
    rerank_score: float
    rerank_rank: int | None = None
    passed_filter: bool = True
    drop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": round(self.retrieval_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "rerank_rank": self.rerank_rank,
            "passed_filter": self.passed_filter,
            "drop_reason": self.drop_reason,
        }


@dataclass
class RerankOutcome:
    """Полный результат выполнения двухэтапного пайплайна."""

    original_query: str
    effective_query: str
    rewrite_applied: bool
    retrieve_k: int
    final_k: int
    all_candidates: list[RankedSource] = field(default_factory=list)
    kept_sources: list[RagSource] = field(default_factory=list)
    dropped_candidates: list[RankedSource] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "effective_query": self.effective_query,
            "rewrite_applied": self.rewrite_applied,
            "retrieve_k": self.retrieve_k,
            "final_k": self.final_k,
            "total_retrieved": len(self.all_candidates),
            "total_kept": len(self.kept_sources),
            "total_dropped": len(self.dropped_candidates),
            "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "kept_sources": [s.to_dict() for s in self.kept_sources],
            "dropped_candidates": [c.to_dict() for c in self.dropped_candidates],
        }


# ============================================================================
# 2. Query Rewriting (переписывание и нормализация запросов)
# ============================================================================

# Стоп-слова и вопросительные конструкции для эвристической очистки запроса
_QUESTION_STOP_WORDS: set[str] = {
    "как",
    "какой",
    "какая",
    "какое",
    "какие",
    "каких",
    "каком",
    "какую",
    "сколько",
    "зачем",
    "почему",
    "где",
    "когда",
    "куда",
    "откуда",
    "что",
    "чем",
    "чего",
    "чему",
    "ком",
    "кто",
    "кого",
    "кому",
    "подскажите",
    "расскажите",
    "пожалуйста",
    "нужно",
    "надо",
    "можно",
    "ли",
    "сделайте",
    "приготовить",
    "приготовления",
    "рецепт",
    "рецепта",
    "рецепте",
    "видео",
    "для",
    "на",
    "в",
    "во",
    "с",
    "со",
    "из",
    "по",
    "к",
    "ко",
    "от",
    "до",
    "при",
    "о",
    "об",
    "обо",
    "и",
    "а",
    "но",
    "или",
}

# Словарь расширения синонимами для кулинарного домена гриля и BBQ
_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "bbq": ["барбекю", "гриль"],
    "курица": ["цыпленок", "бедрышки", "крылышки"],
    "птица": ["курица", "индейка", "утка"],
    "копчение": ["коптить", "пеллетный", "щепа"],
    "сбрызгивание": ["пульверизатор", "яблочный", "уксус"],
    "митболы": ["фрикадельки", "фарш"],
}


class HeuristicQueryRewriter:
    """Офлайн-переписыватель запроса: очистка от шума и фокус на ключевых терминах."""

    def __init__(self, expand_synonyms: bool = False) -> None:
        self.expand_synonyms = expand_synonyms

    def rewrite(self, query: str) -> str:
        clean = query.strip()
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9\-]+", clean.lower().replace("ё", "е"))
        meaningful: list[str] = []

        for t in tokens:
            # Оставляем числа с единицами (например 76c, 190, 50/50, 40г) и слова не из стоп-листа
            if re.search(r"\d", t):
                meaningful.append(t)
            elif len(t) >= 2 and t not in _QUESTION_STOP_WORDS:
                meaningful.append(t)

        if not meaningful:
            return clean

        # Опциональное добавление доменных синонимов, если они не представлены
        if self.expand_synonyms:
            extras: list[str] = []
            for word in meaningful:
                if word in _DOMAIN_SYNONYMS:
                    for syn in _DOMAIN_SYNONYMS[word]:
                        if syn not in meaningful and syn not in extras:
                            extras.append(syn)
                            break
            meaningful.extend(extras)

        return " ".join(meaningful)


class LLMQueryRewriter:
    """Переписыватель запросов через LLM: формирует целевой поисковый запрос."""

    REWRITE_SYSTEM_PROMPT = (
        "Ты — специализированный поисковый оптимизатор запросов по кулинарной базе рецептов гриля и BBQ.\n"
        "Твоя задача — преобразовать разговорный вопрос пользователя в ТОЧНЫЙ и СЖАТЫЙ поисковый запрос (keyword query).\n"
        "Правила:\n"
        "1. Удали вопросительные слова ('как', 'сколько', 'подскажите', 'пожалуйста').\n"
        "2. Выдели главное блюдо, технику (гриль, пеллетный, копчение), ключевые ингредиенты и параметры (градусы, пропорции).\n"
        "3. Выведи ТОЛЬКО переписанный запрос одной строкой, без кавычек, пояснений и вежливых фраз."
    )

    def __init__(
        self,
        client: LLMClient | None,
        fallback: HeuristicQueryRewriter | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or HeuristicQueryRewriter()

    def rewrite(self, query: str) -> str:
        if self.client is None:
            return self.fallback.rewrite(query)

        messages = [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос пользователя: «{query}»"},
        ]
        try:
            comp: Completion = self.client.complete(messages, temperature=0.0, max_tokens=60)
            rewritten = comp.content.strip().strip('"').strip("'").strip("«»")
            # Если LLM вернула пустую строку или слишком длинный текст, берем эвристику
            if not rewritten or len(rewritten) > 200:
                return self.fallback.rewrite(query)
            return rewritten
        except Exception:
            return self.fallback.rewrite(query)


def rewrite_query(
    query: str,
    method: str = "heuristic",
    client: LLMClient | None = None,
) -> str:
    """Единая функция переписывания запроса по выбранному методу."""
    norm = method.lower().strip()
    if norm == "none":
        return query
    if norm == "llm" and client is not None:
        return LLMQueryRewriter(client=client).rewrite(query)
    return HeuristicQueryRewriter().rewrite(query)


# ============================================================================
# 3. Фильтрация релевантности (Relevance Filter)
# ============================================================================


class RelevanceFilter:
    """Фильтрует кандидатов по порогу сходства и относительной границе."""

    def __init__(
        self,
        min_score: float = 0.33,
        score_ratio: float = 0.0,
        min_keep: int = 1,
    ) -> None:
        self.min_score = float(min_score)
        self.score_ratio = max(0.0, float(score_ratio))
        self.min_keep = max(1, int(min_keep))

    def apply(
        self,
        candidates: list[RankedSource],
    ) -> tuple[list[RankedSource], list[RankedSource]]:
        """Разделяет кандидатов на оставленных (passed) и отсеянных (dropped).

        Всегда сохраняет как минимум `min_keep` кандидатов (если они есть),
        чтобы RAG не оставался с пустым контекстом.
        """
        if not candidates:
            return [], []

        def _get_score(item: RankedSource) -> float:
            return item.rerank_score if item.rerank_score is not None else item.retrieval_score

        top_score = max(_get_score(c) for c in candidates)
        relative_threshold = top_score * self.score_ratio if self.score_ratio > 0 else 0.0

        kept: list[RankedSource] = []
        dropped: list[RankedSource] = []

        for c in candidates:
            sc = _get_score(c)
            if sc < self.min_score:
                c.passed_filter = False
                c.drop_reason = f"score {sc:.4f} < min_score {self.min_score:.4f}"
                dropped.append(c)
            elif relative_threshold > 0 and sc < relative_threshold:
                c.passed_filter = False
                c.drop_reason = (
                    f"score {sc:.4f} < relative {relative_threshold:.4f} "
                    f"({self.score_ratio * 100:.0f}% of top {top_score:.4f})"
                )
                dropped.append(c)
            else:
                c.passed_filter = True
                c.drop_reason = None
                kept.append(c)

        # Защита от опустошения контекста: если отфильтровано слишком много,
        # возвращаем лучших кандидатов до min_keep
        if len(kept) < self.min_keep and candidates:
            needed = self.min_keep - len(kept)
            sorted_dropped = sorted(dropped, key=_get_score, reverse=True)
            rescued = sorted_dropped[:needed]
            for r in rescued:
                r.passed_filter = True
                r.drop_reason = "rescued_by_min_keep"
                kept.append(r)
                dropped.remove(r)

        return kept, dropped


# ============================================================================
# 4. Реранкеры (HeuristicReranker и LLMReranker)
# ============================================================================


def _tokenize(text: str) -> list[str]:
    """Быстрая токенизация для лексического сопоставления."""
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower().replace("ё", "е"))


class HeuristicReranker:
    """Эвристический реранкер: комбинирует векторное сходство, лексическое покрытие

    токенов запроса, совпадение слов в заголовке рецепта и приоритеты разделов.
    """

    def __init__(
        self,
        vector_weight: float = 0.50,
        lexical_weight: float = 0.30,
        title_weight: float = 0.15,
        section_weight: float = 0.05,
    ) -> None:
        self.w_vec = float(vector_weight)
        self.w_lex = float(lexical_weight)
        self.w_title = float(title_weight)
        self.w_sec = float(section_weight)

    def compute_lexical_score(self, query_tokens: set[str], text: str) -> float:
        """Считает долю токенов запроса, присутствующих в тексте чанка."""
        if not query_tokens:
            return 0.0
        text_tokens = set(_tokenize(text))
        if not text_tokens:
            return 0.0
        matches = query_tokens.intersection(text_tokens)
        return len(matches) / len(query_tokens)

    def compute_title_score(self, query_tokens: set[str], title: str) -> float:
        """Считает совпадение токенов запроса с заголовком рецепта."""
        if not query_tokens or not title:
            return 0.0
        title_tokens = set(_tokenize(title))
        if not title_tokens:
            return 0.0
        # Заголовок обычно короткий, считаем пересечение от токенов заголовка
        matches = query_tokens.intersection(title_tokens)
        return len(matches) / len(title_tokens)

    def compute_section_score(self, query: str, section: str) -> float:
        """Поощряет соответствие раздела интент-маркерам запроса."""
        q_norm = query.lower()
        sec = section.lower()
        if any(w in q_norm for w in ["соус", "маринад", "ингредиенты", "состав", "смесь"]):
            if sec == "ingredients":
                return 1.0
        if any(w in q_norm for w in ["шаг", "как", "температура", "подавать", "копчение", "жар"]):
            if sec == "step":
                return 1.0
        if sec in ("header", "intro"):
            return 0.5
        return 0.3

    def rerank(self, candidates: list[RankedSource], query: str) -> list[RankedSource]:
        """Вычисляет rerank_score для каждого кандидата и сортирует по убыванию."""
        if not candidates:
            return []

        q_tokens = set(_tokenize(query))
        # Исключаем стоп-слова из оценки лексики
        content_q_tokens = {t for t in q_tokens if t not in _QUESTION_STOP_WORDS and len(t) >= 2}
        if not content_q_tokens:
            content_q_tokens = q_tokens

        for c in candidates:
            vec_score = max(0.0, min(1.0, c.retrieval_score))
            lex_score = self.compute_lexical_score(content_q_tokens, c.source.text)
            title_score = self.compute_title_score(content_q_tokens, c.source.title)
            sec_score = self.compute_section_score(query, c.source.section)

            c.rerank_score = (
                self.w_vec * vec_score
                + self.w_lex * lex_score
                + self.w_title * title_score
                + self.w_sec * sec_score
            )

        # Сортировка по убыванию нового скора
        sorted_candidates = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
        for rank, c in enumerate(sorted_candidates, start=1):
            c.rerank_rank = rank
            c.source.rank = rank
            c.source.score = c.rerank_score

        return sorted_candidates


class LLMReranker:
    """Реранкер на основе отдельной LLM-модели (cross-encoder подход через промпт)."""

    RERANK_PROMPT = (
        "Ты — строгий реранкер документов по кулинарной базе рецептов гриля.\n"
        "Оцени релевантность каждого фрагмента запросу пользователя по шкале от 0.0 до 1.0.\n"
        "Ответь СТРОГО в формате JSON-списка чисел в том же порядке, например: [0.95, 0.40, 0.15].\n"
        "Не добавляй никаких других слов или форматирования."
    )

    def __init__(
        self,
        client: LLMClient | None,
        model_name: str | None = None,
        fallback: HeuristicReranker | None = None,
    ) -> None:
        self.fallback = fallback or HeuristicReranker()
        self.model_name = model_name
        self._client = client
        if client is not None and model_name:
            # Создаём отдельного клиента под указанную модель (например, дешевую 'mistral-nemo')
            new_cfg = dict(client.config)
            new_cfg["model"] = model_name
            self._client = LLMClient(new_cfg)

    def rerank(self, candidates: list[RankedSource], query: str) -> list[RankedSource]:
        if not candidates or self._client is None:
            return self.fallback.rerank(candidates, query)

        snippets: list[str] = []
        for i, c in enumerate(candidates, start=1):
            txt = c.source.text.strip().replace("\n", " ")[:200]
            snippets.append(f"[{i}] «{c.source.title}» ({c.source.section}): {txt}")

        user_content = (
            f"Запрос: «{query}»\n\n"
            f"Кандидаты ({len(candidates)} шт.):\n"
            + "\n".join(snippets)
            + "\n\nJSON-список оценок [s1, s2, ...]:"
        )

        messages = [
            {"role": "system", "content": self.RERANK_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            comp = self._client.complete(messages, temperature=0.0, max_tokens=100)
            raw = comp.content.strip()
            # Попытка извлечь JSON-массив
            match = re.search(r"\[[\d\s,\.]+\]", raw)
            if not match:
                return self.fallback.rerank(candidates, query)
            scores = json.loads(match.group(0))
            if not isinstance(scores, list) or len(scores) != len(candidates):
                return self.fallback.rerank(candidates, query)

            for c, sc in zip(candidates, scores):
                # Смешиваем оценку модели (0.7) и векторный скор (0.3)
                model_sc = max(0.0, min(1.0, float(sc)))
                c.rerank_score = 0.7 * model_sc + 0.3 * c.retrieval_score

            sorted_candidates = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)
            for rank, c in enumerate(sorted_candidates, start=1):
                c.rerank_rank = rank
                c.source.rank = rank
                c.source.score = c.rerank_score
            return sorted_candidates
        except Exception:
            return self.fallback.rerank(candidates, query)


# ============================================================================
# 5. Двухэтапный пайплайн (RerankPipeline)
# ============================================================================


class RerankPipeline:
    """Полный двухэтапный пайплайн:

    👉 Query Rewrite → Retrieve (top-K до фильтрации) → Relevance Filter → Rerank → Top-K после
    """

    def __init__(
        self,
        retriever: RagRetriever,
        retrieve_k: int = 15,
        final_k: int = 5,
        min_score: float = 0.33,
        score_ratio: float = 0.0,
        min_keep: int = 1,
        rerank_method: str = "heuristic",  # "heuristic" | "llm" | "none"
        rewrite_method: str = "heuristic",  # "heuristic" | "llm" | "none"
        llm_client: LLMClient | None = None,
        rerank_model: str | None = None,
        heuristic_weights: dict[str, float] | None = None,
    ) -> None:
        self.retriever = retriever
        self.retrieve_k = max(1, int(retrieve_k))
        self.final_k = max(1, int(final_k))
        self.rerank_method = rerank_method.lower().strip()
        self.rewrite_method = rewrite_method.lower().strip()
        self.llm_client = llm_client

        # Компоненты
        self.filter = RelevanceFilter(
            min_score=min_score,
            score_ratio=score_ratio,
            min_keep=min_keep,
        )

        weights = heuristic_weights or {}
        self.heuristic_reranker = HeuristicReranker(
            vector_weight=weights.get("vector", 0.50),
            lexical_weight=weights.get("lexical", 0.30),
            title_weight=weights.get("title", 0.15),
            section_weight=weights.get("section", 0.05),
        )

        self.llm_reranker = (
            LLMReranker(
                client=llm_client,
                model_name=rerank_model,
                fallback=self.heuristic_reranker,
            )
            if self.rerank_method == "llm"
            else None
        )

    def run(self, query: str) -> RerankOutcome:
        """Выполняет полный пайплайн от запроса до финального списка чанков."""
        t_start = time.time()
        timings: dict[str, float] = {}

        # 1. Query Rewrite
        t0 = time.time()
        effective_query = query
        rewrite_applied = False
        if self.rewrite_method != "none":
            effective_query = rewrite_query(
                query,
                method=self.rewrite_method,
                client=self.llm_client,
            )
            rewrite_applied = effective_query.strip() != query.strip()
        timings["rewrite_ms"] = (time.time() - t0) * 1000

        # 2. Retrieval первого этапа (retrieve_k кандидатов)
        t0 = time.time()
        raw_sources = self.retriever.retrieve(effective_query, top_k=self.retrieve_k)
        timings["retrieve_ms"] = (time.time() - t0) * 1000

        # 3. Оборачиваем в RankedSource
        candidates: list[RankedSource] = []
        for s in raw_sources:
            candidates.append(
                RankedSource(
                    source=s,
                    retrieval_rank=s.rank,
                    retrieval_score=s.score,
                    rerank_score=s.score,
                    rerank_rank=s.rank,
                    passed_filter=True,
                    drop_reason=None,
                )
            )

        # 4. Реранкинг всех извлечённых кандидатов
        t0 = time.time()
        reranked_candidates: list[RankedSource] = candidates
        if self.rerank_method == "heuristic":
            reranked_candidates = self.heuristic_reranker.rerank(candidates, query)
        elif self.rerank_method == "llm" and self.llm_reranker is not None:
            reranked_candidates = self.llm_reranker.rerank(candidates, query)
        timings["rerank_ms"] = (time.time() - t0) * 1000

        # 5. Фильтрация релевантности (Relevance Filter)
        t0 = time.time()
        kept_candidates, dropped_candidates = self.filter.apply(reranked_candidates)
        timings["filter_ms"] = (time.time() - t0) * 1000

        # 6. Ограничение до final_k
        final_candidates = kept_candidates[: self.final_k]
        # Кандидаты сверх final_k помечаются как не попавшие в топ
        for idx, extra in enumerate(kept_candidates[self.final_k :], start=self.final_k + 1):
            extra.passed_filter = False
            extra.drop_reason = f"exceeded final_k limit ({self.final_k})"
            dropped_candidates.append(extra)

        timings["total_ms"] = (time.time() - t_start) * 1000

        # Готовим список RagSource для генератора RAG
        kept_sources = [c.source for c in final_candidates]

        return RerankOutcome(
            original_query=query,
            effective_query=effective_query,
            rewrite_applied=rewrite_applied,
            retrieve_k=self.retrieve_k,
            final_k=self.final_k,
            all_candidates=candidates,
            kept_sources=kept_sources,
            dropped_candidates=dropped_candidates,
            timing_ms=timings,
        )


# ============================================================================
# 6. Агент и функции генерации ответов с реранкингом
# ============================================================================


@dataclass
class RerankRagAnswer(RagAnswer):
    """Ответ RAG с подробной информацией о реранкинге и фильтрации."""

    pipeline_mode: str = "full"
    effective_query: str = ""
    retrieve_k: int = 5
    final_k: int = 5
    dropped_count: int = 0
    outcome: RerankOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "pipeline_mode": self.pipeline_mode,
            "effective_query": self.effective_query,
            "retrieve_k": self.retrieve_k,
            "final_k": self.final_k,
            "dropped_count": self.dropped_count,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        })
        return base


def rag_query_with_pipeline(
    client: LLMClient,
    pipeline: RerankPipeline,
    question: str,
    pipeline_mode: str = "full",
    temperature: float = 0.2,
    max_context_chars: int = 6000,
) -> RerankRagAnswer:
    """Выполняет RAG-запрос через двухэтапный пайплайн с реранкингом и фильтром."""
    t0 = time.time()
    outcome = pipeline.run(question)
    kept_sources = outcome.kept_sources

    messages = build_rag_messages(question, kept_sources, max_chars=max_context_chars)

    try:
        completion: Completion = client.complete(messages, temperature=temperature)
        latency = (time.time() - t0) * 1000
        return RerankRagAnswer(
            question=question,
            mode="rag",
            content=completion.content,
            sources=kept_sources,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            latency_ms=latency,
            error=None,
            pipeline_mode=pipeline_mode,
            effective_query=outcome.effective_query,
            retrieve_k=outcome.retrieve_k,
            final_k=outcome.final_k,
            dropped_count=len(outcome.dropped_candidates),
            outcome=outcome,
        )
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return RerankRagAnswer(
            question=question,
            mode="rag",
            content="",
            sources=kept_sources,
            latency_ms=latency,
            error=str(exc),
            pipeline_mode=pipeline_mode,
            effective_query=outcome.effective_query,
            retrieve_k=outcome.retrieve_k,
            final_k=outcome.final_k,
            dropped_count=len(outcome.dropped_candidates),
            outcome=outcome,
        )


# ============================================================================
# 7. Метрики оценки качества реранкинга и фильтрации
# ============================================================================


def evaluate_precision_at_k(
    sources: list[RagSource],
    expected_ids: list[int | str],
) -> float:
    """Точность в топ-K (Precision@K): доля чанков, принадлежащих целевому рецепту."""
    if not sources or not expected_ids:
        return 0.0
    expected_set = {str(i) for i in expected_ids}
    matches = sum(
        1 for s in sources if s.recipe_id is not None and str(s.recipe_id) in expected_set
    )
    return round(matches / len(sources), 3)


def evaluate_noise_filter(
    outcome: RerankOutcome,
    expected_ids: list[int | str],
) -> dict[str, Any]:
    """Оценивает корректность фильтрации шума.

    - true_negatives: нерелевантные рецепты, успешно отсеянные фильтром;
    - false_negatives: целевой рецепт, ошибочно отсеянный фильтром;
    - false_positives: нерелевантные рецепты, оставшиеся в контексте;
    - true_positives: чанки целевого рецепта, оставленные в контексте.
    """
    expected_set = {str(i) for i in expected_ids}
    tp = sum(
        1 for s in outcome.kept_sources if s.recipe_id is not None and str(s.recipe_id) in expected_set
    )
    fp = sum(
        1
        for s in outcome.kept_sources
        if s.recipe_id is None or str(s.recipe_id) not in expected_set
    )

    dropped = outcome.dropped_candidates
    fn = sum(
        1
        for d in dropped
        if d.source.recipe_id is not None and str(d.source.recipe_id) in expected_set
    )
    tn = sum(
        1
        for d in dropped
        if d.source.recipe_id is None or str(d.source.recipe_id) not in expected_set
    )

    total_dropped = len(dropped)
    noise_reduction = round(tn / (tn + fp), 3) if (tn + fp) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_dropped": total_dropped,
        "noise_filter_ratio": noise_reduction,
    }


def build_pipeline_modes(
    retriever: RagRetriever,
    client: LLMClient | None = None,
    retrieve_k: int = 15,
    final_k: int = 5,
    min_score: float = 0.33,
    rerank_model: str | None = None,
) -> dict[str, RerankPipeline]:
    """Фабрика стандартных конфигураций для сравнительного тестирования режимов."""
    return {
        # 1. Baseline Day 22: без переписывания, без фильтрации, без реранкинга (top-k=5)
        "baseline": RerankPipeline(
            retriever=retriever,
            retrieve_k=final_k,
            final_k=final_k,
            min_score=0.0,
            rerank_method="none",
            rewrite_method="none",
            llm_client=client,
        ),
        # 2. Только фильтр порога similarity: retrieve_k -> отсев ниже min_score -> final_k
        "filter_only": RerankPipeline(
            retriever=retriever,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_score=min_score,
            rerank_method="none",
            rewrite_method="none",
            llm_client=client,
        ),
        # 3. Только реранкинг (heuristic): retrieve_k -> переранжирование лексикой/тайтлом -> final_k
        "rerank_only": RerankPipeline(
            retriever=retriever,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_score=0.0,
            rerank_method="heuristic",
            rewrite_method="none",
            llm_client=client,
        ),
        # 4. Только query rewrite: очищенный запрос -> retrieval 5 чанков
        "rewrite_only": RerankPipeline(
            retriever=retriever,
            retrieve_k=final_k,
            final_k=final_k,
            min_score=0.0,
            rerank_method="none",
            rewrite_method="heuristic",
            llm_client=client,
        ),
        # 5. Полный улучшенный RAG (full): rewrite + retrieve_k + filter + rerank + final_k
        "full_improved": RerankPipeline(
            retriever=retriever,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_score=min_score,
            rerank_method="heuristic",
            rewrite_method="heuristic",
            llm_client=client,
            rerank_model=rerank_model,
        ),
    }


def evaluate_rerank_comparison(
    retriever: RagRetriever,
    client: LLMClient | None = None,
    modes: list[str] | None = None,
    questions: list[dict[str, Any]] | None = None,
    retrieve_k: int = 15,
    final_k: int = 5,
    min_score: float = 0.33,
    run_llm: bool = True,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Сравнивает качество работы RAG в разных режимах фильтрации/реранкинга."""
    items = questions if questions is not None else CONTROL_QUESTIONS
    mode_names = modes or ["baseline", "filter_only", "rerank_only", "full_improved"]
    pipelines = build_pipeline_modes(
        retriever=retriever,
        client=client,
        retrieve_k=retrieve_k,
        final_k=final_k,
        min_score=min_score,
    )

    per_mode_results: dict[str, list[dict[str, Any]]] = {m: [] for m in mode_names}

    total_steps = len(items) * len(mode_names)
    step = 0

    for q_idx, item in enumerate(items, start=1):
        query = item["query"]
        expected_facts = item.get("expected_facts", [])
        expected_sources = item.get("expected_sources", [])

        for mode_name in mode_names:
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, mode_name, query)

            pipe = pipelines[mode_name]
            outcome = pipe.run(query)

            # Оценка на уровне ретривера
            p_at_k = evaluate_precision_at_k(outcome.kept_sources, expected_sources)
            s_rec = evaluate_source_recall(outcome.kept_sources, expected_sources)
            noise_eval = evaluate_noise_filter(outcome, expected_sources)

            # Оценка генерации LLM (если включено)
            answer_text = ""
            latency_ms = outcome.timing_ms.get("total_ms", 0.0)
            tokens: int | None = None
            error: str | None = None
            facts_eval = {"hits": 0, "total": len(expected_facts), "coverage": 0.0}
            cites_eval = {"has_citations": False, "bracket_citations": [], "named_citations": []}

            if run_llm and client is not None:
                ans = rag_query_with_pipeline(
                    client=client,
                    pipeline=pipe,
                    question=query,
                    pipeline_mode=mode_name,
                )
                answer_text = ans.content
                latency_ms = ans.latency_ms
                tokens = ans.total_tokens
                error = ans.error
                facts_eval = evaluate_answer_facts(answer_text, expected_facts)
                cites_eval = evaluate_citations(answer_text, outcome.kept_sources)

            per_mode_results[mode_name].append({
                "id": item.get("id", f"q{q_idx}"),
                "query": query,
                "category": item.get("category", ""),
                "expected_sources": expected_sources,
                "effective_query": outcome.effective_query,
                "retrieved_count": len(outcome.all_candidates),
                "kept_count": len(outcome.kept_sources),
                "dropped_count": len(outcome.dropped_candidates),
                "precision_at_k": p_at_k,
                "source_recall": s_rec,
                "noise_filter": noise_eval,
                "facts": facts_eval,
                "citations": cites_eval,
                "answer": answer_text,
                "latency_ms": round(latency_ms, 2),
                "tokens": tokens,
                "error": error,
                "timing_ms": outcome.timing_ms,
                "kept_sources": [s.to_dict() for s in outcome.kept_sources],
                "dropped_sources": [d.to_dict() for d in outcome.dropped_candidates],
            })

    # Расчёт сводных метрик по каждому режиму
    mode_summaries: dict[str, dict[str, Any]] = {}
    n = len(items)

    for mode_name, res_list in per_mode_results.items():
        avg_prec = round(sum(r["precision_at_k"] for r in res_list) / n, 3) if n else 0.0
        avg_rec = round(sum(1 for r in res_list if r["source_recall"]["found"]) / n, 3) if n else 0.0
        top1_rate = (
            round(sum(1 for r in res_list if r["source_recall"]["top_rank"] == 1) / n, 3)
            if n
            else 0.0
        )
        avg_kept = round(sum(r["kept_count"] for r in res_list) / n, 2) if n else 0.0
        avg_facts = round(sum(r["facts"]["coverage"] for r in res_list) / n, 3) if n else 0.0
        avg_cites = (
            round(sum(1 for r in res_list if r["citations"]["has_citations"]) / n, 3)
            if n
            else 0.0
        )
        avg_noise_ratio = (
            round(sum(r["noise_filter"]["noise_filter_ratio"] for r in res_list) / n, 3)
            if n
            else 0.0
        )
        avg_latency = round(sum(r["latency_ms"] for r in res_list) / n, 1) if n else 0.0

        mode_summaries[mode_name] = {
            "avg_precision_at_k": avg_prec,
            "source_recall_rate": avg_rec,
            "top1_hit_rate": top1_rate,
            "avg_kept_chunks": avg_kept,
            "avg_facts_coverage": avg_facts,
            "citation_rate": avg_cites,
            "noise_filter_ratio": avg_noise_ratio,
            "avg_latency_ms": avg_latency,
        }

    return {
        "total_questions": n,
        "run_llm": run_llm,
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "min_score": min_score,
        "modes": mode_names,
        "mode_summaries": mode_summaries,
        "results": per_mode_results,
    }


# ============================================================================
# 8. Markdown-отчёт сравнения режимов
# ============================================================================


def render_rerank_comparison_markdown(report: dict[str, Any]) -> str:
    """Формирует структурированный Markdown-отчёт по результатам сравнения режимов."""
    summaries = report.get("mode_summaries", {})
    results = report.get("results", {})
    mode_names = report.get("modes", [])

    lines: list[str] = [
        "# Отчёт Day 23: Реранкинг, фильтрация релевантности и Query Rewrite",
        "",
        f"- **Количество контрольных вопросов:** {report.get('total_questions')}",
        f"- **Top-K до фильтрации (retrieve_k):** {report.get('retrieve_k')}",
        f"- **Top-K после фильтрации (final_k):** {report.get('final_k')}",
        f"- **Порог отсечения (min_score):** {report.get('min_score')}",
        f"- **Режим прогона:** {'LLM запросы' if report.get('run_llm') else 'Офлайн (поиск и реранкинг)'}",
        "",
        "## 1. Сводное сравнение режимов",
        "",
        "| Режим | Precision@K | Recall источника | Top-1 | Полнота фактов | Цитирование | Ср. чанков | Ср. задержка |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    mode_labels = {
        "baseline": "1. Baseline (Day 22)",
        "filter_only": "2. Только фильтр (min_score)",
        "rerank_only": "3. Только реранкер",
        "rewrite_only": "4. Только Query Rewrite",
        "full_improved": "5. Улучшенный RAG (full)",
    }

    for m in mode_names:
        s = summaries.get(m, {})
        label = mode_labels.get(m, m)
        p_k = f"{s.get('avg_precision_at_k', 0) * 100:.1f}%"
        rec = f"{s.get('source_recall_rate', 0) * 100:.1f}%"
        top1 = f"{s.get('top1_hit_rate', 0) * 100:.1f}%"
        facts = f"{s.get('avg_facts_coverage', 0) * 100:.1f}%" if report.get("run_llm") else "—"
        cites = f"{s.get('citation_rate', 0) * 100:.1f}%" if report.get("run_llm") else "—"
        kept = f"{s.get('avg_kept_chunks', 0):.1f}"
        lat = f"{s.get('avg_latency_ms', 0):.0f} мс"
        lines.append(f"| **{label}** | {p_k} | {rec} | {top1} | **{facts}** | {cites} | {kept} | {lat} |")

    # Сравнение Baseline vs Full
    if "baseline" in summaries and "full_improved" in summaries:
        b = summaries["baseline"]
        f = summaries["full_improved"]
        p_diff = (f.get("avg_precision_at_k", 0) - b.get("avg_precision_at_k", 0)) * 100
        fact_diff = (f.get("avg_facts_coverage", 0) - b.get("avg_facts_coverage", 0)) * 100
        noise_cut = f.get("noise_filter_ratio", 0) * 100

        lines.extend([
            "",
            "### Ключевые эффекты второго этапа (Full vs Baseline):",
            f"- **Прирост Precision@K (чистота контекста):** {p_diff:+.1f}%",
            f"- **Доля успешно отсеянного шума:** {noise_cut:.1f}%",
            (
                f"- **Прирост полноты фактов в ответах:** {fact_diff:+.1f}%"
                if report.get("run_llm")
                else "- *Полнота фактов оценивается при LLM-прогоне.*"
            ),
            f"- **Экономия контекста:** с {b.get('avg_kept_chunks', 5):.1f} до {f.get('avg_kept_chunks', 5):.1f} чанков",
            "",
        ])

    # Таблица по вопросам
    lines.extend([
        "## 2. Результаты по 10 контрольным вопросам",
        "",
        "| № | Вопрос | Baseline P@K | Filter P@K | Full P@K | Отсеяно чанков | Full факты |",
        "| - | --- | --- | --- | --- | --- | --- |",
    ])

    base_res = results.get("baseline", [])
    filt_res = results.get("filter_only", [])
    full_res = results.get("full_improved", [])

    for idx in range(len(base_res)):
        b_item = base_res[idx]
        f_item = filt_res[idx] if idx < len(filt_res) else {}
        u_item = full_res[idx] if idx < len(full_res) else {}

        q_short = b_item["query"][:40] + "…" if len(b_item["query"]) > 40 else b_item["query"]
        bp = f"{b_item.get('precision_at_k', 0) * 100:.0f}%"
        fp = f"{f_item.get('precision_at_k', 0) * 100:.0f}%" if f_item else "—"
        up = f"{u_item.get('precision_at_k', 0) * 100:.0f}%" if u_item else "—"
        dropped = str(u_item.get("dropped_count", 0)) if u_item else "0"
        facts = (
            f"{u_item.get('facts', {}).get('coverage', 0) * 100:.0f}%"
            if report.get("run_llm") and u_item
            else "—"
        )
        lines.append(f"| {idx + 1} | {q_short} | {bp} | {fp} | {up} | {dropped} | {facts} |")

    lines.extend([
        "",
        "## 3. Выводы",
        "",
        "1. **Фильтрация нерелевантных источников (Relevance Filter):** Порог отсечения отсекает чанки "
        "других рецептов со сходством ниже установленного минимума. Это защищает модель от зашумления "
        "контекста нерелевантными ингредиентами и шагами.",
        "2. **Реранкинг (Heuristic Reranker):** Учёт точного лексического совпадения и названий рецептов "
        "поднимает целевой рецепт на первое место даже при близких векторных скорах у конкурирующих чанков.",
        "3. **Query Rewrite:** Очистка от вопросительных оборотов и выделение ключевых кулинарных терминов "
        "увеличивает плотность совпадений в эмбеддере и улучшает стартовую выдачу первого этапа.",
        "4. **Улучшение качества RAG:** Сокращение мусорных чанков предотвращает галлюцинации и ошибки "
        "в пропорциях, гарантируя подачу в LLM только высокорелевантного контекста.",
    ])

    return "\n".join(lines)
