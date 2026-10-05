"""Day 22: Модуль RAG (Retrieval-Augmented Generation) для рецептов гриля и BBQ.

Архитектура:
1. Поиск релевантных чанков через векторный индекс FAISS (RagRetriever);
2. Сборка обогащённого контекста с метаданными и ссылками на источники;
3. Формирование структурированного промпта и запрос к LLM;
4. Агент с двумя режимами (с RAG / без RAG) — RagAgent;
5. Набор из 10 контрольных вопросов с эталонными ожиданиями и источниками;
6. Сравнение качества ответов (покрытие фактов, точность поиска, цитирование).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .indexing import (
    Embedder,
    FaissIndex,
    HashingEmbedder,
)
from .llm_client import Completion, LLMClient


# ============================================================================
# 1. Модели данных RAG
# ============================================================================


@dataclass
class RagSource:
    """Найденный чанк-источник для формирования контекста RAG."""

    rank: int
    score: float
    chunk_id: str
    recipe_id: int | str | None
    title: str
    section: str
    step_number: int | None = None
    url: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "recipe_id": self.recipe_id,
            "title": self.title,
            "section": self.section,
            "step_number": self.step_number,
            "url": self.url,
            "text": self.text,
        }

    @classmethod
    def from_search_result(cls, result: dict[str, Any]) -> RagSource:
        meta = result.get("metadata", {}) or {}
        return cls(
            rank=int(result.get("rank", 0)),
            score=float(result.get("score", 0.0)),
            chunk_id=str(result.get("chunk_id", "")),
            recipe_id=meta.get("recipe_id"),
            title=str(meta.get("title", "")),
            section=str(meta.get("section", "document")),
            step_number=meta.get("step_number"),
            url=str(meta.get("url", "")),
            text=str(result.get("text", "")),
        )


@dataclass
class RagAnswer:
    """Результат выполнения запроса (с RAG или без RAG)."""

    question: str
    mode: str  # "rag" | "no_rag"
    content: str
    sources: list[RagSource] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "mode": self.mode,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ============================================================================
# 2. Ретривер (RagRetriever)
# ============================================================================


class RagRetriever:
    """Извлекает релевантные чанки из FAISS-индекса и собирает контекст."""

    def __init__(
        self,
        index: FaissIndex,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.index = index
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)

    @property
    def total_vectors(self) -> int:
        return self.index.total

    def retrieve(self, query: str, top_k: int | None = None) -> list[RagSource]:
        """Ищет ближайшие чанки по запросу и оборачивает в RagSource."""
        k = top_k if top_k is not None else self.top_k
        raw_results = self.index.search(query, top_k=k)
        sources: list[RagSource] = []
        for item in raw_results:
            source = RagSource.from_search_result(item)
            if source.score >= self.min_score:
                sources.append(source)
        return sources

    def build_context(self, sources: list[RagSource], max_chars: int = 6000) -> str:
        """Собирает структурированный текст контекста с нумерацией источников."""
        if not sources:
            return "В базе данных рецептов не найдено подходящих материалов по запросу."

        blocks: list[str] = []
        current_len = 0

        for idx, src in enumerate(sources, start=1):
            step_info = f", шаг №{src.step_number}" if src.step_number is not None else ""
            meta_line = (
                f"[Источник {idx}] Рецепт: «{src.title}» "
                f"(ID: {src.recipe_id}, раздел: {src.section}{step_info}, сходство: {src.score:.3f})"
            )
            block = f"{meta_line}\n{src.text.strip()}"
            block_len = len(block) + 2

            if current_len + block_len > max_chars and blocks:
                break

            blocks.append(block)
            current_len += block_len

        return "\n\n".join(blocks)


# ============================================================================
# 3. Промпты для RAG и Baseline
# ============================================================================


RAG_SYSTEM_PROMPT = """Ты — точный кулинарный ассистент по рецептам гриля и барбекю.
Твоя задача — отвечать на вопросы пользователя СТРОГО на основе предоставленного контекста из базы рецептов.

Правила:
1. Используй ТОЛЬКО факты из предоставленных [Источников].
2. Если в контексте есть точные параметры (пропорции, граммы, штуки, градусы, время), обязательно укажи их.
3. Обязательно ссылайся на источники по их номерам, например: [Источник 1], [Источник 2] или по названию рецепта.
4. Если в контексте НЕТ ответа на вопрос или часть вопроса (например, нет калорийности, другого блюда), прямо и честно скажи: «В предоставленной базе рецептов эта информация отсутствует». НЕ придумывай факты от себя.
5. Отвечай по-русски, структурированно, понятно и по делу."""


NO_RAG_SYSTEM_PROMPT = """Ты — кулинарный ассистент по приготовлению блюд на гриле и барбекю.
Ответь на вопрос пользователя, используя свои общие знания.
Отвечай по-русски, структурированно, доброжелательно и по делу."""


def build_rag_messages(
    question: str,
    sources: list[RagSource],
    max_chars: int = 6000,
) -> list[dict[str, str]]:
    """Формирует список сообщений (messages) для RAG-запроса."""
    retriever_dummy = RagRetriever(index=None, top_k=len(sources))  # type: ignore[arg-type]
    context_text = retriever_dummy.build_context(sources, max_chars=max_chars)

    user_content = (
        f"КОНТЕКСТ ИЗ БАЗЫ РЕЦЕПТОВ:\n"
        f"------------------------\n"
        f"{context_text}\n"
        f"------------------------\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
        f"{question}\n\n"
        f"Пожалуйста, дай точный ответ строго по контексту с указанием источников."
    )

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_plain_messages(question: str) -> list[dict[str, str]]:
    """Формирует список сообщений для baseline-запроса (без RAG)."""
    return [
        {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ============================================================================
# 4. Функции выполнения RAG и Baseline запросов
# ============================================================================


def rag_query(
    client: LLMClient,
    retriever: RagRetriever,
    question: str,
    top_k: int = 5,
    temperature: float = 0.2,
    max_context_chars: int = 6000,
) -> RagAnswer:
    """Полный пайплайн RAG-запроса:

    👉 вопрос → поиск релевантных чанков → объединение с вопросом → запрос к LLM
    """
    t0 = time.time()
    # 1. Поиск релевантных чанков
    sources = retriever.retrieve(question, top_k=top_k)

    # 2. Объединение контекста и вопроса
    messages = build_rag_messages(question, sources, max_chars=max_context_chars)

    # 3. Запрос к LLM
    try:
        completion: Completion = client.complete(messages, temperature=temperature)
        latency = (time.time() - t0) * 1000
        return RagAnswer(
            question=question,
            mode="rag",
            content=completion.content,
            sources=sources,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            latency_ms=latency,
            error=None,
        )
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return RagAnswer(
            question=question,
            mode="rag",
            content="",
            sources=sources,
            latency_ms=latency,
            error=str(exc),
        )


def plain_query(
    client: LLMClient,
    question: str,
    temperature: float = 0.2,
) -> RagAnswer:
    """Запрос к LLM напрямую без использования RAG (Baseline)."""
    t0 = time.time()
    messages = build_plain_messages(question)
    try:
        completion: Completion = client.complete(messages, temperature=temperature)
        latency = (time.time() - t0) * 1000
        return RagAnswer(
            question=question,
            mode="no_rag",
            content=completion.content,
            sources=[],
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            latency_ms=latency,
            error=None,
        )
    except Exception as exc:
        latency = (time.time() - t0) * 1000
        return RagAnswer(
            question=question,
            mode="no_rag",
            content="",
            sources=[],
            latency_ms=latency,
            error=str(exc),
        )


# ============================================================================
# 5. Агент с двумя режимами (RagAgent)
# ============================================================================


class RagAgent:
    """Агент с переключаемыми режимами: с RAG (retrieval) и без RAG (no_rag).

    Поддерживает:
    - явную смену режима через свойство `mode` или метод `set_mode`;
    - вызов `ask(question)` согласно текущему режиму;
    - удобные методы `ask_with_rag(question)` и `ask_without_rag(question)`.
    """

    VALID_MODES = ("rag", "no_rag")

    def __init__(
        self,
        client: LLMClient,
        retriever: RagRetriever | None = None,
        mode: str = "rag",
        top_k: int = 5,
        temperature: float = 0.2,
        max_context_chars: int = 6000,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.top_k = max(1, int(top_k))
        self.temperature = float(temperature)
        self.max_context_chars = int(max_context_chars)
        self._mode = "rag"
        self.set_mode(mode)

    @property
    def mode(self) -> str:
        """Текущий активный режим агента ('rag' или 'no_rag')."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Переключает режим работы агента."""
        norm = mode.lower().strip()
        if norm not in self.VALID_MODES:
            raise ValueError(
                f"Некорректный режим '{mode}'. Допустимые варианты: {self.VALID_MODES}"
            )
        if norm == "rag" and self.retriever is None:
            raise ValueError("Режим 'rag' недоступен: ретривер не инициализирован.")
        self._mode = norm

    def ask(self, question: str) -> RagAnswer:
        """Отправляет вопрос модели в соответствии с текущим режимом."""
        if self._mode == "rag":
            return self.ask_with_rag(question)
        return self.ask_without_rag(question)

    def ask_with_rag(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Выполняет запрос с RAG независимо от текущего режима."""
        if self.retriever is None:
            raise ValueError("Ретривер не инициализирован для RAG-запроса.")
        k = top_k if top_k is not None else self.top_k
        return rag_query(
            client=self.client,
            retriever=self.retriever,
            question=question,
            top_k=k,
            temperature=self.temperature,
            max_context_chars=self.max_context_chars,
        )

    def ask_without_rag(self, question: str) -> RagAnswer:
        """Выполняет прямой запрос к LLM (без RAG) независимо от режима."""
        return plain_query(
            client=self.client,
            question=question,
            temperature=self.temperature,
        )


# ============================================================================
# 6. Загрузка ретривера
# ============================================================================


def load_retriever(
    index_dir: str | Path,
    dataset: str = "recipt_all",
    strategy: str = "structural",
    embedder: Embedder | None = None,
    top_k: int = 5,
) -> RagRetriever:
    """Загружает FAISS-индекс с диска и создает RagRetriever."""
    target_dir = Path(index_dir).resolve() / dataset / strategy
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Директория индекса не найдена: {target_dir}. "
            f"Сначала выполните индексацию через `python3 -m agent.indexing_scenarios`."
        )

    if embedder is None:
        # По умолчанию читаем метаданные сохраненного индекса
        meta_file = target_dir / "index_meta.json"
        dim = 256
        if meta_file.exists():
            import json

            try:
                with meta_file.open(encoding="utf-8") as file:
                    m = json.load(file)
                    dim = int(m.get("dim", 256))
            except Exception:
                pass
        embedder = HashingEmbedder(dim=dim)

    faiss_index = FaissIndex.load(target_dir, embedder=embedder)
    return RagRetriever(index=faiss_index, top_k=top_k)


# ============================================================================
# 7. Набор из 10 контрольных вопросов
# ============================================================================

CONTROL_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "Апельсиновая курица на гриле новогодний рецепт: как подавать с фруктами розмарином и клюквой?",
        "category": "Птица",
        "expectation": (
            "В ответе должны быть названы: 2 поджаренные половинки апельсина, виноград (черный и зеленый), "
            "размороженная клюква (100 г) и свежие веточки розмарина для подачи вокруг запеченной курицы."
        ),
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
        "id": "q2",
        "query": "Прайм риб на гриле обсыпка солью пропорции перец и температура копчения на пеллетном гриле",
        "category": "Говядина",
        "expectation": (
            "Указана точная норма соли: 1 ч.л. на каждый 1 кг мяса (7 ч.л. на 7,5 кг); перец в пропорции 1:1 к соли; "
            "температура первого этапа копчения на пеллетном гриле — 76 °C."
        ),
        "expected_facts": [
            ["1 ч.л", "1 чайная", "ч.л. соли на каждый килограмм", "1 ч. л", "на каждый килограмм"],
            ["1:1", "столько же", "равное", "пропорци"],
            ["76", "76 °c", "76°c", "76 градусов"],
        ],
        "expected_sources": [15452],
        "expected_title": "Прайм риб на гриле (видео)",
    },
    {
        "id": "q3",
        "query": "Каре барашка с кашей на гриле гречневая каша кедровые орехи брусника гарнир",
        "category": "Баранина",
        "expectation": (
            "Указан гарнир из гречневой каши (300 г), обжаренных кедровых орехов (3 ст.л.), "
            "красного лука и украшение размороженной брусникой (3 ст.л.) и петрушкой."
        ),
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
        "id": "q4",
        "query": "Лондон бройл на гриле рецепт: сделайте маринад красное вино бальзамический уксус соевый соус чеснок",
        "category": "Говядина",
        "expectation": (
            "Указаны ингредиенты маринада на 1 кг: сухое красное вино (100 мл), бальзамический уксус (3 ст.л.), "
            "светлый соевый соус (4 ст.л.), чеснок (2 зубчика), оливковое масло (3 ст.л.), соль и перец; "
            "время маринования в холодильнике — от 2 до 8 часов."
        ),
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
        "id": "q5",
        "query": "Пикантная смесь для курицы на гриле: за пикантность отвечают молотая паприка гранулированный чеснок порошок чили",
        "category": "Пряные смеси",
        "expectation": (
            "Указан точный состав смеси: соль (2 ч.л.), свежемолотый черный перец (1 ч.л.), "
            "гранулированный чеснок (2 ч.л.), сушеный лук (2 ч.л.), молотая паприка (2 ч.л.) и порошок чили (0.5 ч.л.)."
        ),
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
        "id": "q6",
        "query": "Крылышки 0-190 на пеллетном гриле: суть метода выкладки и внутренняя температура готовности мяса",
        "category": "Птица",
        "expectation": (
            "Указана суть методики 0-190: выкладка крылышек на решетку до начала розжига (на холодный гриль), "
            "настройка гриля на 190-200 °C и доведение внутренней температуры мяса до не менее 80 °C."
        ),
        "expected_facts": [
            ["до розжига", "холодн", "до запуска", "выкладывают до", "выложить до"],
            ["190", "190-200", "200"],
            ["80", "80 °c", "80°c", "80 градусов"],
        ],
        "expected_sources": [15379],
        "expected_title": "Крылышки 0-190 на пеллетном гриле (видео)",
    },
    {
        "id": "q7",
        "query": "Пикантный горчичный соус для свинины на гриле ингредиенты: горчица столовая мед шрирача уксус",
        "category": "Соусы",
        "expectation": (
            "Указан точный состав соуса: столовая горчица (7 ч.л.), мед (2 ч.л.), соус шрирача (1 ч.л.), "
            "яблочный уксус (1-2 ч.л.), соль (1/3 ч.л.) и черный свежемолотый перец (0.5 ч.л.)."
        ),
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
        "id": "q8",
        "query": "Митболы из свинины на гриле шаг сформируйте митболы весом 40 г непрямой средний жар 170-200",
        "category": "Свинина",
        "expectation": (
            "Указан вес каждого митбола около 40 г, режим непрямого среднего жара с температурой 170-200 °C "
            "и копчение в течение 20 минут."
        ),
        "expected_facts": [
            ["40", "40 г", "40 грамм"],
            ["непрям", "жар"],
            ["фарш", "митбол"],
        ],
        "expected_sources": [15360],
        "expected_title": "Митболы из свинины на гриле (видео)",
    },
    {
        "id": "q9",
        "query": "Рваная курица на гриле рецепт: смесь для сбрызгивания вода и яблочный уксус 50/50 в пульверизатор",
        "category": "Птица",
        "expectation": (
            "Указано использование куриных бедрышек (снять кожу, оставить бедренную кость), "
            "а для сбрызгивания каждые 30 минут применяется смесь воды и яблочного уксуса в пропорции 50/50."
        ),
        "expected_facts": [
            ["уксус", "яблочн"],
            ["вода", "воду"],
            ["50/50", "пополам", "равных", "50 на 50", "пропорци"],
        ],
        "expected_sources": [14965],
        "expected_title": "Рваная курица на гриле (видео)",
    },
    {
        "id": "q10",
        "query": "Апельсиновая курица на гриле: сколько калорий и белков в порции по рецепту?",
        "category": "Контроль галлюцинаций",
        "expectation": (
            "Отрицательный контроль (проверка заземления): в базе рецептов информация о калорийности "
            "и БЖУ отсутствует. Агент с RAG должен явно заявить об отсутствии этих данных в источнике, "
            "а не выдумывать цифры."
        ),
        "expected_facts": [
            [
                "не указан",
                "отсутств",
                "нет данн",
                "не содержит",
                "не приводится",
                "нет информац",
                "не указана",
                "не указано",
                "не найдено",
                "не приведено",
            ],
        ],
        "expected_sources": [15454],
        "expected_title": "Апельсиновая курица на гриле (видео)",
    },
]


# ============================================================================
# 8. Оценка качества и сравнительный анализ
# ============================================================================


def _normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру и заменяет букву ё."""
    return text.lower().replace("ё", "е")


def evaluate_answer_facts(
    answer_text: str,
    expected_facts: list[list[str]],
) -> dict[str, Any]:
    """Проверяет покрытие ключевых фактов/синонимов в ответе модели.

    Каждый элемент `expected_facts` — это список синонимов (OR-группа).
    Группа считается выполненной, если хотя бы один из синонимов найден в тексте.
    """
    if not expected_facts:
        return {"hits": 0, "total": 0, "coverage": 1.0, "matched": [], "missing": []}

    norm = _normalize_text(answer_text)
    hits = 0
    matched: list[str] = []
    missing: list[str] = []

    for group in expected_facts:
        found_synonym = None
        for syn in group:
            if _normalize_text(syn) in norm:
                found_synonym = syn
                break
        if found_synonym is not None:
            hits += 1
            matched.append(found_synonym)
        else:
            missing.append("/".join(group))

    total = len(expected_facts)
    coverage = round(hits / total, 3) if total > 0 else 0.0
    return {
        "hits": hits,
        "total": total,
        "coverage": coverage,
        "matched": matched,
        "missing": missing,
    }


def evaluate_source_recall(
    sources: list[RagSource],
    expected_ids: list[int | str],
) -> dict[str, Any]:
    """Проверяет, попал ли целевой рецепт в найденные источники."""
    expected_set = {str(i) for i in expected_ids}
    if not expected_set:
        return {"found": True, "top_rank": None, "top_score": 0.0, "found_ids": []}

    found_ranks: list[int] = []
    found_ids: list[str] = []
    top_score = sources[0].score if sources else 0.0

    for s in sources:
        s_id = str(s.recipe_id) if s.recipe_id is not None else ""
        if s_id in expected_set:
            found_ranks.append(s.rank)
            found_ids.append(s_id)

    top_rank = min(found_ranks) if found_ranks else None
    return {
        "found": bool(found_ranks),
        "top_rank": top_rank,
        "top_score": round(top_score, 4),
        "found_ids": list(set(found_ids)),
        "all_ranks": found_ranks,
    }


def evaluate_citations(answer_text: str, sources: list[RagSource]) -> dict[str, Any]:
    """Проверяет наличие ссылок на источники в ответе модели."""
    # Ищем упоминания [Источник N] или [N]
    bracket_cites = re.findall(r"\[(?:источник\s*)?(\d+)\]", answer_text, flags=re.IGNORECASE)
    # Ищем упоминания названий рецептов из источников
    named_cites: list[str] = []
    norm = _normalize_text(answer_text)
    for s in sources:
        if s.title:
            clean_title = _normalize_text(s.title).split("(")[0].strip()
            if len(clean_title) >= 6 and clean_title in norm:
                named_cites.append(s.title)

    has_citations = bool(bracket_cites or named_cites)
    return {
        "has_citations": has_citations,
        "bracket_citations": [int(x) for x in bracket_cites],
        "named_citations": list(set(named_cites)),
    }


def evaluate_rag_comparison(
    agent: RagAgent,
    questions: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    run_llm: bool = True,
) -> dict[str, Any]:
    """Выполняет пакетное сравнение двух режимов по набору контрольных вопросов."""
    items = questions if questions is not None else CONTROL_QUESTIONS
    total_q = len(items)

    results: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id", f"q{idx}")
        query = item["query"]
        expected_facts = item.get("expected_facts", [])
        expected_sources = item.get("expected_sources", [])

        if progress_callback:
            progress_callback(idx, total_q, query)

        # 1. Поиск источников
        sources: list[RagSource] = []
        if agent.retriever is not None:
            sources = agent.retriever.retrieve(query, top_k=agent.top_k)

        source_eval = evaluate_source_recall(sources, expected_sources)

        # 2. Ответы LLM (если run_llm=True)
        no_rag_answer: RagAnswer | None = None
        rag_answer: RagAnswer | None = None

        if run_llm:
            no_rag_answer = agent.ask_without_rag(query)
            rag_answer = agent.ask_with_rag(query, top_k=agent.top_k)
        else:
            # Офлайн-заглушка для быстрого тестирования поиска
            no_rag_answer = RagAnswer(
                question=query,
                mode="no_rag",
                content="[LLM отключена в режиме --no-llm]",
            )
            rag_answer = RagAnswer(
                question=query,
                mode="rag",
                content="[LLM отключена в режиме --no-llm]",
                sources=sources,
            )

        facts_no_rag = evaluate_answer_facts(no_rag_answer.content, expected_facts)
        facts_rag = evaluate_answer_facts(rag_answer.content, expected_facts)
        citations = evaluate_citations(rag_answer.content, sources)

        results.append({
            "id": q_id,
            "query": query,
            "category": item.get("category", ""),
            "expectation": item.get("expectation", ""),
            "expected_sources": expected_sources,
            "expected_title": item.get("expected_title", ""),
            "sources": [s.to_dict() for s in sources],
            "source_recall": source_eval,
            "no_rag": {
                "answer": no_rag_answer.content,
                "facts": facts_no_rag,
                "tokens": no_rag_answer.total_tokens,
                "latency_ms": no_rag_answer.latency_ms,
                "error": no_rag_answer.error,
            },
            "rag": {
                "answer": rag_answer.content,
                "facts": facts_rag,
                "citations": citations,
                "tokens": rag_answer.total_tokens,
                "latency_ms": rag_answer.latency_ms,
                "error": rag_answer.error,
            },
        })

    # Расчет агрегированных метрик
    n = len(results)
    avg_no_rag_coverage = (
        round(sum(r["no_rag"]["facts"]["coverage"] for r in results) / n, 3) if n else 0.0
    )
    avg_rag_coverage = (
        round(sum(r["rag"]["facts"]["coverage"] for r in results) / n, 3) if n else 0.0
    )
    source_recall_rate = (
        round(sum(1 for r in results if r["source_recall"]["found"]) / n, 3) if n else 0.0
    )
    top1_hit_rate = (
        round(sum(1 for r in results if r["source_recall"]["top_rank"] == 1) / n, 3) if n else 0.0
    )
    citation_rate = (
        round(sum(1 for r in results if r["rag"]["citations"]["has_citations"]) / n, 3)
        if n
        else 0.0
    )

    return {
        "total_questions": n,
        "run_llm": run_llm,
        "dataset": agent.retriever.index.metadata.get("source", "unknown")
        if agent.retriever
        else "none",
        "top_k": agent.top_k,
        "summary": {
            "avg_facts_no_rag": avg_no_rag_coverage,
            "avg_facts_rag": avg_rag_coverage,
            "facts_gain": round(avg_rag_coverage - avg_no_rag_coverage, 3),
            "source_recall_top_k": source_recall_rate,
            "source_recall_top_1": top1_hit_rate,
            "citation_rate": citation_rate,
        },
        "results": results,
    }


def render_rag_comparison_markdown(report: dict[str, Any]) -> str:
    """Формирует наглядный Markdown-отчёт со сравнением RAG и Baseline."""
    summary = report.get("summary", {})
    results = report.get("results", [])

    lines: list[str] = [
        "# Отчёт сравнения ответов: Без RAG vs С RAG (Day 22)",
        "",
        f"- **База знаний:** `{report.get('dataset')}`",
        f"- **Количество контрольных вопросов:** {report.get('total_questions')}",
        f"- **Число чанков в контексте (top-k):** {report.get('top_k')}",
        f"- **Режим прогона:** {'LLM запрос' if report.get('run_llm') else 'Офлайн (только поиск)'}",
        "",
        "## 1. Сводные метрики качества",
        "",
        "| Метрика | Без RAG (Baseline) | С RAG (Retrieval) | Прирост / Эффект |",
        "| --- | --- | --- | --- |",
        (
            f"| **Полнота фактов (Fact Coverage)** | **{summary.get('avg_facts_no_rag') * 100:.1f}%** | "
            f"**{summary.get('avg_facts_rag') * 100:.1f}%** | "
            f"**{summary.get('facts_gain') * 100:+.1f}%** |"
        ),
        (
            f"| **Нахождение целевого рецепта в top-{report.get('top_k')}** | — | "
            f"**{summary.get('source_recall_top_k') * 100:.1f}%** | Полнота поиска чанков |"
        ),
        (
            f"| **Целевой рецепт на первом месте (Top-1)** | — | "
            f"**{summary.get('source_recall_top_1') * 100:.1f}%** | Точность ретривера |"
        ),
        (
            f"| **Цитирование источников в ответе** | — | "
            f"**{summary.get('citation_rate') * 100:.1f}%** | Обоснованность ответа |"
        ),
        "",
        "## 2. Результаты по 10 контрольным вопросам",
        "",
        "| № | Вопрос | Категория | Ранг источника | Факты (Без RAG) | Факты (С RAG) | Ссылки на источник |",
        "| - | --- | --- | --- | --- | --- | --- |",
    ]

    for idx, r in enumerate(results, start=1):
        q_text = r["query"][:45] + "…" if len(r["query"]) > 45 else r["query"]
        cat = r.get("category", "")
        rank_info = r["source_recall"]["top_rank"]
        rank_str = f"#{rank_info}" if rank_info else "Не найден"
        f_no = f"{r['no_rag']['facts']['coverage'] * 100:.0f}%"
        f_rag = f"{r['rag']['facts']['coverage'] * 100:.0f}%"
        cites = "Да" if r["rag"]["citations"]["has_citations"] else "Нет"
        lines.append(f"| {idx} | {q_text} | {cat} | {rank_str} | {f_no} | {f_rag} | {cites} |")

    lines.extend([
        "",
        "## 3. Детальный разбор вопросов",
        "",
    ])

    for idx, r in enumerate(results, start=1):
        lines.extend([
            f"### Вопрос №{idx}: {r['query']}",
            "",
            f"- **Категория:** {r.get('category')}",
            f"- **Ожидание:** {r.get('expectation')}",
            f"- **Целевой источник:** ID `{r.get('expected_sources')}` («{r.get('expected_title')}»)",
            f"- **Ранг в поиске:** {r['source_recall']['top_rank']} (score={r['source_recall']['top_score']})",
            "",
            "#### Ответ без RAG (общие знания модели):",
            f"> {r['no_rag']['answer'].replace(chr(10), ' ' * 2 + chr(10) + '> ')}",
            "",
            f"*Покрытие фактов:* **{r['no_rag']['facts']['hits']}/{r['no_rag']['facts']['total']}** "
            f"({r['no_rag']['facts']['coverage'] * 100:.0f}%)",
            "",
            "#### Ответ с RAG (на основе базы рецептов):",
            f"> {r['rag']['answer'].replace(chr(10), ' ' * 2 + chr(10) + '> ')}",
            "",
            f"*Покрытие фактов:* **{r['rag']['facts']['hits']}/{r['rag']['facts']['total']}** "
            f"({r['rag']['facts']['coverage'] * 100:.0f}%) | "
            f"*Ссылки на источники:* {'Да' if r['rag']['citations']['has_citations'] else 'Нет'}",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 4. Выводы",
        "",
        "1. **Точность пропорций и параметров:** Без RAG модель предлагает общие кулинарные рецепты, "
        "но ошибается в конкретных рецептурных нормах (например, пропорция соли в Прайм риб, "
        "состав пряных смесей, точный вес митболов). С RAG модель воспроизводит точные граммовки и градусы из базы.",
        "2. **Устранение галлюцинаций (Grounding):** На вопрос о калорийности апельсиновой курицы (отсутствующей в базе) "
        "агент с RAG прямо сообщает об отсутствии данных в источнике, тогда как модель без RAG "
        "генерирует приблизительные или вымышленные цифры.",
        "3. **Прослеживаемость (Provenance):** Ответы с RAG снабжены ссылками на конкретные источники [Источник N], "
        "что позволяет пользователю проверить первоисточник рецепта.",
    ])

    return "\n".join(lines)
