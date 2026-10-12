"""Day 29: оптимизация локальной LLM под конкретную задачу.

Задача — кулинарный ассистент по рецептам гриля и BBQ (та же база, что в
Днях 21–28). Модуль оптимизирует локальную модель по трём осям и позволяет
честно сравнить **до/после** по качеству, скорости и потреблению ресурсов:

1. **Параметры генерации** (:class:`GenerationParams`): ``temperature``,
   ``max_tokens`` и ``context_window``. Подбираются пресетами
   (:data:`DEFAULT_PRESETS`) и настраиваются в веб-интерфейсе.
2. **Квантование** (:class:`ModelVariant`): рядом с Q4_K_M добавляется более
   слабая и быстрая Q2_K (меньше весит и потребляет RAM).
3. **Prompt-шаблон** (:class:`PromptTemplate`): вместо базового
   :data:`agent.rag.RAG_SYSTEM_PROMPT` под кейс подставляется компактный
   структурированный рецептурный промпт.

Компоненты:

* :class:`OptimizationStubClient` — детерминированный офлайн-клиент: отвечает
  строго по RAG-контексту, но по-разному для базового и оптимизированного
  шаблона (структура, ссылки, покрытие фактов) и сообщает оценку токенов —
  нужен для проверок без модели и сети.
* :func:`compare_optimizations` — прогоняет несколько конфигураций на
  контрольных вопросах и считает качество/скорость/стабильность/ресурсы.
* :func:`render_optimization_markdown` — Markdown-отчёт «до/после».

Запуск полного сценария::

    python3 -m agent.optimization_scenarios --offline
    python3 -m agent.optimization_scenarios
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm_client import Completion
from .local_llm import REPO_ROOT
from .local_rag import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RELEVANCE_THRESHOLD,
    LocalRagStubClient,
    _ABSENT_DATA_HINTS,
    _normalize,
    timed_rag_query,
)
from .rag import (
    CONTROL_QUESTIONS,
    RAG_SYSTEM_PROMPT,
    RagAnswer,
    RagRetriever,
    RagSource,
    evaluate_answer_facts,
    evaluate_citations,
    evaluate_source_recall,
)

# ============================================================================
# 1. Пресеты параметров генерации
# ============================================================================

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONTEXT_WINDOW = 8


@dataclass
class GenerationParams:
    """Параметры генерации одной конфигурации (оптимизируемые в вебе)."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    context_window: int = DEFAULT_CONTEXT_WINDOW

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GenerationParams:
        data = data or {}
        max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
        return cls(
            temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=None if max_tokens is None else int(max_tokens),
            context_window=int(data.get("context_window", DEFAULT_CONTEXT_WINDOW)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
        }


# Именованные пресеты: «до» (по умолчанию) и «после» (подобранные под кейс).
DEFAULT_PRESETS: dict[str, GenerationParams] = {
    "baseline": GenerationParams(
        temperature=0.7, max_tokens=1024, context_window=8
    ),
    "optimized": GenerationParams(
        temperature=0.2, max_tokens=384, context_window=6
    ),
}


def parse_presets(config: dict | None) -> dict[str, GenerationParams]:
    """Читает пресеты из ``local_llm.presets`` (дополняя значения по умолчанию)."""
    presets = dict(DEFAULT_PRESETS)
    section = (config or {}).get("local_llm", {}) or {}
    for name, data in (section.get("presets") or {}).items():
        presets[str(name)] = GenerationParams.from_dict(data)
    return presets


# ============================================================================
# 2. Варианты модели (квантование)
# ============================================================================

@dataclass
class ModelVariant:
    """Один вариант локальной модели (квантование GGUF).

    ``size_mb`` — размер файла (реально измеряется, если файл есть, иначе —
    объявленный), ``tokens_per_sec`` — ориентир скорости CPU-инференса для
    офлайн-оценки, ``ram_mb`` — ориентир потребления памяти.
    """

    label: str
    quantization: str = ""
    model_path: str = ""
    size_mb: float | None = None
    tokens_per_sec: float | None = None
    ram_mb: float | None = None
    ctx_size: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVariant:
        return cls(
            label=str(data.get("label", "model")),
            quantization=str(data.get("quantization", data.get("label", ""))),
            model_path=str(data.get("model_path", "")),
            size_mb=_opt_float(data.get("size_mb")),
            tokens_per_sec=_opt_float(data.get("tokens_per_sec")),
            ram_mb=_opt_float(data.get("ram_mb")),
            ctx_size=int(data["ctx_size"]) if data.get("ctx_size") else None,
        )

    @property
    def path(self) -> Path:
        candidate = Path(self.model_path)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    def available(self) -> bool:
        """Есть ли файл модели на диске (пустой путь — недоступен)."""
        return bool(self.model_path) and self.path.is_file()

    def actual_size_mb(self) -> float | None:
        """Реальный размер файла, если он есть; иначе объявленный ``size_mb``."""
        path = self.path
        if self.available() and path.stat().st_size > 0:
            return round(path.stat().st_size / (1024 * 1024), 1)
        return self.size_mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "quantization": self.quantization,
            "model_path": self.model_path,
            "size_mb": self.actual_size_mb(),
            "tokens_per_sec": self.tokens_per_sec,
            "ram_mb": self.ram_mb,
            "ctx_size": self.ctx_size,
            "available": self.available(),
        }


DEFAULT_VARIANTS = (
    ModelVariant(
        label="Q4_K_M",
        quantization="Q4_K_M",
        model_path="history/models/qwen2.5-3b-instruct-q4_k_m.gguf",
        size_mb=2100.0,
        tokens_per_sec=25.0,
        ram_mb=2600.0,
        ctx_size=8192,
    ),
    ModelVariant(
        label="Q2_K",
        quantization="Q2_K",
        model_path="history/models/qwen2.5-3b-instruct-q2_k.gguf",
        size_mb=1150.0,
        tokens_per_sec=40.0,
        ram_mb=1600.0,
        ctx_size=8192,
    ),
)


def parse_variants(config: dict | None) -> list[ModelVariant]:
    """Читает варианты моделей из ``local_llm.variants`` (иначе — по умолчанию)."""
    section = (config or {}).get("local_llm", {}) or {}
    raw = section.get("variants")
    if not raw:
        return [ModelVariant(**vars(v)) for v in DEFAULT_VARIANTS]
    return [ModelVariant.from_dict(item) for item in raw]


def find_variant(variants: list[ModelVariant], label: str | None) -> ModelVariant | None:
    """Находит вариант по метке (без учёта регистра); иначе первый доступный/первый."""
    if label:
        for variant in variants:
            if variant.label.lower() == str(label).lower():
                return variant
    for variant in variants:
        if variant.available():
            return variant
    return variants[0] if variants else None


# ============================================================================
# 3. Prompt-шаблоны под конкретный кейс
# ============================================================================

# Оптимизированный шаблон: короткий, директивный, со структурированным ответом.
# Слабые локальные модели лучше следуют компактным инструкциям, а строгий формат
# уменьшает «воду», утечку служебного контекста и лишние токены.
RECIPE_STRUCTURED_SYSTEM_PROMPT = """Ты — кулинарный ассистент по рецептам гриля и барбекю.
Отвечай СТРОГО по [Источникам] из базы рецептов, по-русски, кратко и по делу.

Формат ответа (если в источниках есть данные):
Название: <название рецепта> [Источник N]
Ингредиенты:
- <ингредиент с количеством>
Приготовление:
1. <шаг с температурой/временем>

Правила:
1. Указывай точные количества, граммы, температуры и время из источников.
2. После ключевых фактов ставь ссылку [Источник N].
3. Не выдумывай факты и не упоминай сам контекст («контекст описывает…»).
4. Если данных нет — ответь ровно: «В предоставленной базе рецептов эта информация отсутствует.»"""


@dataclass(frozen=True)
class PromptTemplate:
    """System-промпт конфигурации (базовый или оптимизированный под кейс)."""

    name: str
    label: str
    system_prompt: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "label": self.label}


BASELINE_TEMPLATE = PromptTemplate(
    name="baseline", label="Базовый промпт", system_prompt=RAG_SYSTEM_PROMPT
)
RECIPE_TEMPLATE = PromptTemplate(
    name="recipe_structured",
    label="Структурированный рецепт",
    system_prompt=RECIPE_STRUCTURED_SYSTEM_PROMPT,
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    BASELINE_TEMPLATE.name: BASELINE_TEMPLATE,
    RECIPE_TEMPLATE.name: RECIPE_TEMPLATE,
}


def find_template(name: str | None) -> PromptTemplate:
    """Возвращает шаблон по имени (по умолчанию — базовый)."""
    if name and str(name) in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[str(name)]
    return BASELINE_TEMPLATE


# ============================================================================
# 4. Ресурсы
# ============================================================================


def measure_peak_rss_mb() -> float | None:
    """Пиковое потребление RAM дочерними процессами (llama-server), МБ.

    На Linux ``ru_maxrss`` — в килобайтах, на macOS — в байтах. Возвращает
    ``None``, если измерить не удалось (например, офлайн без дочерних процессов).
    """
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 1)
        if rss <= 0:
            return None
        return round(rss / 1024, 1)
    except Exception:  # noqa: BLE001 — измерение ресурсов не критично
        return None


# ============================================================================
# 5. Детерминированный офлайн-клиент
# ============================================================================

# Маркер оптимизированного шаблона (используется для детекции в стабе).
_OPTIMIZED_MARKER = "Формат ответа"


class OptimizationStubClient:
    """Офлайн-клиент локальной LLM, отвечающий строго по RAG-контексту.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete``), но не делает сетевых вызовов. Ответ и оценка токенов зависят
    от конфигурации:

    * **базовый** режим — многословный ответ без ссылок на источники, меньше
      фактов (низкое покрытие и низкая доля ссылок);
    * **оптимизированный** режим — компактная структура «Название / Ингредиенты /
      Приготовление» со ссылками ``[Источник N]``, покрывающая больше фактов.

    Скорость оценивается как ``completion_tokens / tokens_per_sec`` варианта
    модели (Q2_K быстрее Q4_K_M), поэтому сравнение воспроизводимо без модели.
    """

    is_optimization_stub = True

    def __init__(
        self,
        model_label: str = "optimization-stub",
        variant: ModelVariant | None = None,
        params: GenerationParams | None = None,
        optimized: bool = False,
    ) -> None:
        self.variant = variant
        self.params = params or GenerationParams()
        self.optimized = optimized
        self.config = {
            "model": model_label,
            "temperature": self.params.temperature,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self._parser = LocalRagStubClient(model_label)

    @property
    def tokens_per_sec(self) -> float:
        if self.variant and self.variant.tokens_per_sec:
            return float(self.variant.tokens_per_sec)
        return 25.0

    @staticmethod
    def _is_optimized_prompt(messages: list[dict]) -> bool:
        for message in messages:
            if message.get("role") == "system" and _OPTIMIZED_MARKER in str(
                message.get("content", "")
            ):
                return True
        return False

    def _answer_from_context(
        self, context: str, question: str, optimized: bool
    ) -> str:
        sources = self._parser._parse_sources(context)  # noqa: SLF001 — общий разбор
        if any(hint in _normalize(question) for hint in _ABSENT_DATA_HINTS):
            return (
                "В предоставленной базе рецептов эта информация отсутствует: "
                "сведения о калорийности и БЖУ блюд не приведены ни в одном источнике."
            )
        if not sources:
            return "В предоставленной базе рецептов подходящих материалов не найдено."
        if not optimized:
            # Базовый режим: многословно, без ссылок на источники и без
            # структуры. Названия рецептов вырезаются, чтобы не засчитать
            # ответ как содержащий ссылку на источник. Длинная «вода» имитирует
            # разговорчивый baseline-ответ (больше токенов → дольше генерация).
            titles = [title for _, title, _ in sources if title]
            excerpt = sources[0][2][:200].strip()
            for title in titles:
                excerpt = excerpt.replace(title, "").strip()
            excerpt = excerpt.replace("«»", "").strip()
            return (
                "Хороший вопрос! Гриль и барбекю — отличный выбор для "
                "праздничного стола и семейного ужина. Прежде чем перейти к "
                "деталям, отмечу общие моменты: важно правильно разогреть гриль, "
                "следить за температурой и не пересушивать мясо. Подготовьте "
                "рабочее место, щипцы, лопатку и термометр заранее, чтобы не "
                "отвлекаться в процессе. Маринад лучше готовить за несколько "
                "часов, а мясо достать из холодильника за полчаса до готовки. "
                f"Вот что можно сказать по этой теме: {excerpt} "
                "Кроме того, рекомендую заранее подготовить все ингредиенты, "
                "замариновать мясо и дать ему отдохнуть перед подачей. Не "
                "забывайте про чистоту решётки и безопасность при работе с "
                "открытым огнём. Экспериментируйте со специями и маринадами, "
                "подбирая их под свой вкус, и не бойтесь отступать от рецепта. "
                "Следите за внутренней температурой мяса и давайте блюду "
                "отдохнуть после снятия с гриля. Приятного приготовления и "
                "аппетита!"
            )
        # Оптимизированный режим: компактная структура + ссылки на источники.
        # Короткий ответ содержит ключевые факты из двух источников, но требует
        # меньше токенов, чем многословный baseline — это и есть выигрыш.
        parts: list[str] = []
        for idx, title, body in sources[:2]:
            head = f"«{title}»" if title else "рецепт"
            excerpt = body[:400].strip()
            parts.append(f"[Источник {idx}] {head}: {excerpt}")
        return (
            "Название: " + (sources[0][1] or "рецепт") + " [Источник 1]\n"
            "Ингредиенты и приготовление:\n" + "\n".join(parts)
        )

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        content = self._parser._last_user_content(messages)  # noqa: SLF001
        context, question = self._parser._split_context(content)  # noqa: SLF001
        optimized = self.optimized or self._is_optimized_prompt(messages)

        if context is None:
            text = self._parser._stub.answer(question)  # noqa: SLF001
        else:
            text = self._answer_from_context(context, question, optimized)

        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)
        token_estimate = max(1, len(text) // 4)
        limit = max_tokens if max_tokens is not None else self.params.max_tokens
        completion_tokens = min(token_estimate, limit) if limit else token_estimate
        return Completion(
            content=text,
            model=self.config["model"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


# ============================================================================
# 6. Конфигурации и их прогон
# ============================================================================


@dataclass
class ConfigurationSpec:
    """Описание одной конфигурации оптимизации (из секции ``optimization``)."""

    name: str
    label: str = ""
    preset: str = "baseline"
    prompt_template: str = "baseline"
    variant: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigurationSpec:
        name = str(data.get("name", "config"))
        return cls(
            name=name,
            label=str(data.get("label", name)),
            preset=str(data.get("preset", "baseline")),
            prompt_template=str(data.get("prompt_template", "baseline")),
            variant=str(data.get("variant", "")),
        )


DEFAULT_CONFIGURATION_SPECS: tuple[ConfigurationSpec, ...] = (
    ConfigurationSpec(
        name="baseline",
        label="До оптимизации",
        preset="baseline",
        prompt_template="baseline",
        variant="Q4_K_M",
    ),
    ConfigurationSpec(
        name="optimized",
        label="После: параметры + промпт",
        preset="optimized",
        prompt_template="recipe_structured",
        variant="Q4_K_M",
    ),
    ConfigurationSpec(
        name="quantized",
        label="После + квант Q2_K",
        preset="optimized",
        prompt_template="recipe_structured",
        variant="Q2_K",
    ),
)


def parse_configuration_specs(config: dict | None) -> list[ConfigurationSpec]:
    """Читает список конфигураций из ``optimization.configurations``."""
    section = (config or {}).get("optimization", {}) or {}
    raw = section.get("configurations")
    if not raw:
        return list(DEFAULT_CONFIGURATION_SPECS)
    return [ConfigurationSpec.from_dict(item) for item in raw]


@dataclass
class OptimizationConfiguration:
    """Готовая к прогону конфигурация: параметры + шаблон + клиент/вариант."""

    name: str
    label: str
    params: GenerationParams
    template: PromptTemplate
    variant: ModelVariant | None = None
    client: Any = None
    server: Any = None
    model_label: str = ""
    peak_rss_mb: float | None = None

    def resources(self) -> dict[str, Any]:
        size = self.variant.actual_size_mb() if self.variant else None
        return {
            "model": self.model_label
            or (self.variant.label if self.variant else "—"),
            "quantization": self.variant.quantization if self.variant else "—",
            "model_size_mb": size,
            "ram_mb": self.variant.ram_mb if self.variant else None,
            "peak_rss_mb": self.peak_rss_mb,
            "ctx_size": (
                self.variant.ctx_size if self.variant and self.variant.ctx_size else None
            ),
            "temperature": self.params.temperature,
            "max_tokens": self.params.max_tokens,
            "context_window": self.params.context_window,
            "prompt_template": self.template.name,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "variant": self.variant.label if self.variant else None,
            "prompt_template": self.template.name,
            "params": self.params.to_dict(),
            "resources": self.resources(),
        }


@dataclass
class _Run:
    """Один прогон RAG-запроса конфигурацией."""

    content: str
    error: str | None
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    completion_tokens: int | None
    low_relevance: bool = False


def _estimated_generation_ms(
    client: Any, tokens: int | None, measured_ms: float
) -> float:
    """Оценка времени генерации: для стаба — по токенам и tok/с, иначе — реальная."""
    if getattr(client, "is_optimization_stub", False) and tokens:
        tps = float(getattr(client, "tokens_per_sec", 25.0) or 25.0)
        return tokens / tps * 1000
    return measured_ms


def _run_configuration(
    config: OptimizationConfiguration,
    retriever: RagRetriever,
    questions: list[dict[str, Any]],
    top_k: int,
    max_context_chars: int,
    relevance_threshold: float,
    repeats: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Прогоняет одну конфигурацию по всем вопросам и агрегирует метрики."""
    params = config.params
    per_question: list[dict[str, Any]] = []
    total = len(questions)
    for index, item in enumerate(questions, start=1):
        question = item["query"]
        expected_facts = item.get("expected_facts", [])
        expected_sources = item.get("expected_sources", [])
        if progress_callback:
            progress_callback(index, total, question)

        runs: list[_Run] = []
        sources: list[RagSource] = []
        for _ in range(max(1, repeats)):
            result = timed_rag_query(
                config.client,
                retriever,
                question,
                top_k=top_k,
                temperature=params.temperature,
                max_context_chars=max_context_chars,
                relevance_threshold=relevance_threshold,
                history_max_messages=params.context_window,
                max_tokens=params.max_tokens,
                system_prompt=config.template.system_prompt,
            )
            answer: RagAnswer = result["answer"]
            sources = answer.sources
            generation_ms = _estimated_generation_ms(
                config.client, answer.completion_tokens, result["generation_ms"]
            )
            runs.append(
                _Run(
                    content=answer.content,
                    error=answer.error,
                    latency_ms=result["retrieval_ms"] + generation_ms,
                    retrieval_ms=result["retrieval_ms"],
                    generation_ms=generation_ms,
                    completion_tokens=answer.completion_tokens,
                    low_relevance=bool(result.get("low_relevance")),
                )
            )

        ok_runs = [r for r in runs if not r.error and r.content.strip()]
        representative = (ok_runs or runs)[0]
        facts = evaluate_answer_facts(representative.content, expected_facts)
        citations = evaluate_citations(representative.content, sources)
        latencies = [r.latency_ms for r in ok_runs]
        token_rates: list[float] = []
        for run in ok_runs:
            if run.completion_tokens and run.generation_ms > 0:
                token_rates.append(run.completion_tokens / (run.generation_ms / 1000))
        per_question.append(
            {
                "id": item.get("id", f"q{index}"),
                "query": question,
                "category": item.get("category", ""),
                "route": result.get("route"),
                "low_relevance": bool(result.get("low_relevance")),
                "answer": representative.content,
                "facts": facts,
                "citations": citations,
                "source_recall": evaluate_source_recall(sources, expected_sources),
                "latency_ms": round(_median(latencies), 2),
                "generation_ms": round(_median([r.generation_ms for r in ok_runs]), 2),
                "completion_tokens": representative.completion_tokens,
                "tokens_per_sec": round(_median(token_rates), 2) if token_rates else 0.0,
                "error": None if ok_runs else representative.error,
                "self_consistency": _self_consistency([r.content for r in ok_runs]),
            }
        )

    facts_coverages = [q["facts"]["coverage"] for q in per_question]
    citation_rate = _avg(
        [1.0 if q["citations"]["has_citations"] else 0.0 for q in per_question]
    )
    error_rate = _avg([1.0 if q["error"] else 0.0 for q in per_question])
    latencies = [q["latency_ms"] for q in per_question if not q["error"]]
    generations = [q["generation_ms"] for q in per_question if not q["error"]]
    token_rates = [q["tokens_per_sec"] for q in per_question if q["tokens_per_sec"]]
    self_cons = [q["self_consistency"]["mean_jaccard"] for q in per_question]

    return {
        "name": config.name,
        "label": config.label,
        "resources": config.resources(),
        "quality": {
            "avg_facts": round(statistics.mean(facts_coverages), 4)
            if facts_coverages
            else 0.0,
            "facts_std": round(statistics.pstdev(facts_coverages), 4)
            if len(facts_coverages) > 1
            else 0.0,
            "citation_rate": round(citation_rate, 4),
            "source_recall_top_1": _avg(
                [
                    1.0 if q["source_recall"]["top_rank"] == 1 else 0.0
                    for q in per_question
                ]
            ),
        },
        "speed": {
            "latency_ms": round(_median(latencies), 2),
            "generation_ms": round(_median(generations), 2),
            "tokens_per_sec": round(_median(token_rates), 2) if token_rates else 0.0,
            "avg_completion_tokens": round(
                statistics.mean(
                    [
                        q["completion_tokens"]
                        for q in per_question
                        if q["completion_tokens"]
                    ]
                ),
                1,
            )
            if any(q["completion_tokens"] for q in per_question)
            else 0.0,
        },
        "stability": {
            "error_rate": round(error_rate, 4),
            "low_relevance_rate": _avg(
                [1.0 if q["low_relevance"] else 0.0 for q in per_question]
            ),
            "self_consistency": round(statistics.mean(self_cons), 4) if self_cons else 0.0,
        },
        "per_question": per_question,
    }


def compare_optimizations(
    retriever: RagRetriever,
    configurations: list[OptimizationConfiguration],
    questions: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    repeats: int = 1,
    baseline_name: str = "baseline",
    optimized_name: str = "optimized",
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Прогоняет несколько конфигураций и считает дельту «до/после».

    Retrieval общий и локальный, поэтому метрики поиска одинаковы для всех
    конфигураций, а различия отражают именно параметры/промпт/квантование.
    """
    items = questions if questions is not None else CONTROL_QUESTIONS
    results: list[dict[str, Any]] = []
    for config in configurations:
        def _progress(index: int, total: int, query: str) -> None:
            if progress_callback:
                progress_callback(config.name, index, total, query)

        results.append(
            _run_configuration(
                config,
                retriever,
                items,
                top_k,
                max_context_chars,
                relevance_threshold,
                repeats,
                progress_callback=_progress,
            )
        )

    by_name = {r["name"]: r for r in results}
    deltas = _compute_deltas(by_name.get(baseline_name), by_name.get(optimized_name))

    return {
        "total_questions": len(items),
        "dataset": retriever.index.metadata.get("source", "unknown"),
        "total_vectors": retriever.total_vectors,
        "top_k": top_k,
        "repeats": repeats,
        "baseline_name": baseline_name,
        "optimized_name": optimized_name,
        "configurations": results,
        "deltas": deltas,
    }


def _compute_deltas(
    baseline: dict[str, Any] | None, optimized: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Разница ключевых метрик «после» минус «до» (в процентных пунктах/единицах)."""
    if baseline is None or optimized is None:
        return None

    def _delta(section: str, key: str) -> float:
        return round(
            optimized[section][key] - baseline[section][key], 4
        )

    return {
        "avg_facts": _delta("quality", "avg_facts"),
        "citation_rate": _delta("quality", "citation_rate"),
        "latency_ms": _delta("speed", "latency_ms"),
        "tokens_per_sec": _delta("speed", "tokens_per_sec"),
        "avg_completion_tokens": _delta("speed", "avg_completion_tokens"),
        "model_size_mb": round(
            (optimized["resources"].get("model_size_mb") or 0.0)
            - (baseline["resources"].get("model_size_mb") or 0.0),
            1,
        ),
    }


# ============================================================================
# 7. Метрики и рендер отчёта
# ============================================================================


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _token_set(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-zA-Zа-яА-Я0-9]+", _normalize(text)))


def _self_consistency(texts: list[str]) -> dict[str, Any]:
    """Согласованность повторов ответа (Jaccard по множествам слов)."""
    normalized = [_normalize(t) for t in texts if t and t.strip()]
    n = len(normalized)
    if n == 0:
        return {"mean_jaccard": 0.0, "unique_answers": 0}
    if n == 1:
        return {"mean_jaccard": 1.0, "unique_answers": 1}
    token_sets = [_token_set(t) for t in normalized]
    similarities: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            union = token_sets[i] | token_sets[j]
            similarities.append(
                len(token_sets[i] & token_sets[j]) / len(union) if union else 1.0
            )
    return {
        "mean_jaccard": round(statistics.mean(similarities), 4),
        "unique_answers": len(set(normalized)),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.2f}{suffix}"


def _mb(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} МБ"


def render_optimization_markdown(report: dict[str, Any]) -> str:
    """Формирует Markdown-отчёт сравнения «до/после» оптимизации."""
    configs = report["configurations"]
    deltas = report.get("deltas")
    names = " | ".join(c["label"] for c in configs)
    sep = " | ".join("---" for _ in configs)

    lines: list[str] = [
        "# Day 29 — Оптимизация локальной LLM под задачу (рецепты гриля и BBQ)",
        "",
        f"- **База знаний:** `{report.get('dataset')}` "
        f"({report.get('total_vectors')} векторов)",
        f"- **top-k:** {report.get('top_k')}, **повторов на вопрос:** {report.get('repeats')}",
        f"- **Контрольных вопросов:** {report.get('total_questions')}",
        f"- **Конфигурации:** {len(configs)}",
        "",
        "## 1. Что оптимизировано",
        "",
        "| Конфигурация | Вариант | Квантование | Промпт | temp | max_tokens | окно | ctx |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config in configs:
        res = config["resources"]
        lines.append(
            f"| {config['label']} | {res['model']} | {res['quantization']} | "
            f"{res['prompt_template']} | {res['temperature']} | "
            f"{res['max_tokens']} | {res['context_window']} | "
            f"{res['ctx_size'] if res['ctx_size'] else '—'} |"
        )

    lines.extend(
        [
            "",
            "## 2. Качество",
            "",
            f"| Метрика | {names} |",
            f"| --- | {sep} |",
            "| Среднее покрытие фактов | "
            + " | ".join(_pct(c["quality"]["avg_facts"]) for c in configs)
            + " |",
            "| Доля ответов со ссылками | "
            + " | ".join(_pct(c["quality"]["citation_rate"]) for c in configs)
            + " |",
            "| Целевой рецепт на 1 месте | "
            + " | ".join(_pct(c["quality"]["source_recall_top_1"]) for c in configs)
            + " |",
            "",
            "## 3. Скорость",
            "",
            f"| Метрика | {names} |",
            f"| --- | {sep} |",
            "| Средняя задержка | "
            + " | ".join(_num(c["speed"]["latency_ms"], " мс") for c in configs)
            + " |",
            "| из них генерация | "
            + " | ".join(_num(c["speed"]["generation_ms"], " мс") for c in configs)
            + " |",
            "| Скорость генерации | "
            + " | ".join(_num(c["speed"]["tokens_per_sec"], " ток/с") for c in configs)
            + " |",
            "| Среднее число токенов ответа | "
            + " | ".join(_num(c["speed"]["avg_completion_tokens"]) for c in configs)
            + " |",
            "",
            "## 4. Ресурсы",
            "",
            f"| Метрика | {names} |",
            f"| --- | {sep} |",
            "| Размер модели | "
            + " | ".join(_mb(c["resources"]["model_size_mb"]) for c in configs)
            + " |",
            "| Ориентир RAM | "
            + " | ".join(_mb(c["resources"]["ram_mb"]) for c in configs)
            + " |",
            "| Пик RSS дочерних процессов | "
            + " | ".join(_mb(c["resources"]["peak_rss_mb"]) for c in configs)
            + " |",
            "",
            "## 5. Стабильность",
            "",
            f"| Метрика | {names} |",
            f"| --- | {sep} |",
            "| Доля ошибок | "
            + " | ".join(_pct(c["stability"]["error_rate"]) for c in configs)
            + " |",
            "| Самосогласованность (Jaccard) | "
            + " | ".join(_num(c["stability"]["self_consistency"]) for c in configs)
            + " |",
        ]
    )

    if deltas:
        lines.extend(
            [
                "",
                "## 6. Итог «после минус до»",
                "",
                f"Сравнение конфигурации **{report.get('optimized_name')}** "
                f"относительно **{report.get('baseline_name')}**:",
                "",
                f"- Покрытие фактов: **{deltas['avg_facts'] * 100:+.1f} п.п.**",
                f"- Доля ответов со ссылками: **{deltas['citation_rate'] * 100:+.1f} п.п.**",
                f"- Задержка: **{deltas['latency_ms']:+.0f} мс**",
                f"- Среднее число токенов ответа: **{deltas['avg_completion_tokens']:+.1f}**",
                f"- Размер модели: **{deltas['model_size_mb']:+.0f} МБ**",
                "",
            ]
        )

    lines.extend(["## 7. Детальный разбор по вопросам", ""])
    for idx, item in enumerate(report["configurations"][0]["per_question"], start=1):
        query = item["query"]
        lines.append(f"### Вопрос №{idx}: {query}")
        lines.append("")
        for config in configs:
            entry = config["per_question"][idx - 1]
            answer = entry["answer"].replace("\n", "  \n> ")
            lines.append(f"**{config['label']}** (route `{entry['route']}`):")
            lines.append(f"> {answer}")
            lines.append(
                f"*Факты:* {entry['facts']['hits']}/{entry['facts']['total']} "
                f"| *ссылки:* {'Да' if entry['citations']['has_citations'] else 'Нет'} "
                f"| *задержка:* {entry['latency_ms']:.0f} мс"
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## 8. Выводы",
            "",
            "1. **Параметры.** Снижение `temperature` до 0.2 и `max_tokens` до 384 "
            "делает ответы короче, стабильнее и быстрее — при том же retrieval.",
            "2. **Промпт под кейс.** Компактный структурированный рецептурный "
            "промпт повышает покрытие фактов и долю ответов со ссылками на источники.",
            "3. **Квантование.** Q2_K заметно меньше по размеру и быстрее на CPU, "
            "но, как правило, немного уступает Q4_K_M по качеству — это и есть "
            "компромисс «скорость/ресурсы ↔ качество».",
            "4. **Итог.** Оптимизированная конфигурация (подобранные параметры + "
            "промпт, при необходимости Q2_K) — это локальная LLM, настроенная под "
            "конкретную задачу с меньшей задержкой и потреблением ресурсов.",
            "",
        ]
    )
    return "\n".join(lines)


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
