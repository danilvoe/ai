#!/usr/bin/env python3
"""Day 24: Сквозной сценарий цитирования, источников и анти-галлюцинаций (Grounded RAG).

Сценарий реализует:
1. Заземлённый RAG (Grounded RAG):
   👉 Ответ (answer)
   👉 Список источников (sources: source + section/chunk_id)
   👉 Цитаты (quotes: дословные фрагменты из найденных чанков)
2. Проверка на 10 контрольных вопросах:
   👉 Есть ли источники в каждом ответе (sources_present)
   👉 Есть ли цитаты в каждом ответе (quotes_present)
   👉 Совпадает ли смысл ответа с цитатами (semantic_alignment)
3. Усиление (Анти-галлюцинационный шлюз):
   👉 Если релевантность ниже порога — ассистент обязан сказать «не знаю» и попросить уточнение
   👉 Отсутствие выдуманных цитат при нерелевантных вопросах
4. Экспорт структурированного JSON и Markdown-отчёта в history/rag_grounding/.

Запуск::

    # Полное тестирование 10 вопросов + проверка порога релевантности + генерация отчёта
    python3 -m agent.grounding_scenarios

    # Быстрый офлайн-прогон без обращений к API (проверка структуры, источников и цитат)
    python3 -m agent.grounding_scenarios --no-llm

    # Демонстрация на одиночном вопросе (вывод ответа, источников с chunk_id и цитат)
    python3 -m agent.grounding_scenarios --question "Как подавать апельсиновую курицу на гриле?"

    # Ограничение числа вопросов (например, первые 3 вопроса)
    python3 -m agent.grounding_scenarios --limit 3

    # Настройка порога релевантности (relevance threshold)
    python3 -m agent.grounding_scenarios --min-relevance 0.35
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import load_config
from agent.grounding import (
    GROUNDED_CONTROL_QUESTIONS,
    LOW_RELEVANCE_TEST_QUESTIONS,
    GroundedRagAgent,
    evaluate_grounded_benchmark,
    evaluate_low_relevance_guardrail,
    render_grounding_markdown_report,
)
from agent.llm_client import LLMClient
from agent.rag import load_retriever
from agent.reranking import RerankPipeline


def _preview(text: str, limit: int = 100) -> str:
    """Однострочный предпросмотр текста."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def main() -> None:
    # 1. Загрузка конфигурации
    config = load_config()
    rag_cfg = config.get("rag", {}) or {}
    idx_cfg = config.get("indexing", {}) or {}
    rerank_cfg = config.get("reranking", {}) or {}
    ground_cfg = config.get("grounding", {}) or {}

    default_dataset = ground_cfg.get("dataset", rerank_cfg.get("dataset", rag_cfg.get("dataset", "recipt_all")))
    default_strategy = ground_cfg.get("strategy", rerank_cfg.get("strategy", rag_cfg.get("strategy", "structural")))
    default_index_dir = ground_cfg.get("index_dir", idx_cfg.get("index_dir", "history/index"))
    default_threshold = float(ground_cfg.get("relevance_threshold", rerank_cfg.get("min_score", 0.33)))
    default_top_k = int(ground_cfg.get("top_k", rerank_cfg.get("final_k", 5)))
    default_retrieve_k = int(rerank_cfg.get("retrieve_k", 15))
    default_report_dir = ground_cfg.get("report_dir", "history/rag_grounding")

    parser = argparse.ArgumentParser(
        description="Day 24: Grounded RAG — цитаты, источники и анти-галлюцинации"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Задать одиночный вопрос и вывести структурированный ответ с источниками и цитатами",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Офлайн-режим: выполнить проверку без вызовов API LLM",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество тестируемых вопросов",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=default_threshold,
        help=f"Порог релевантности для правила усиления (по умолчанию: {default_threshold})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help=f"Количество чанков в контексте (по умолчанию: {default_top_k})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help=f"Датасет рецептов (по умолчанию: {default_dataset})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=default_strategy,
        help=f"Стратегия чанкинга (по умолчанию: {default_strategy})",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=default_report_dir,
        help=f"Каталог для сохранения отчётов (по умолчанию: {default_report_dir})",
    )
    args = parser.parse_args()

    # 2. Инициализация ретривера
    print("=" * 80)
    print("DAY 24: ЦИТАТЫ, ИСТОЧНИКИ И АНТИ-ГАЛЛЮЦИНАЦИИ (GROUNDED RAG)")
    print("=" * 80)
    print(f"Датасет: {args.dataset} | Стратегия: {args.strategy}")
    print(f"Порог релевантности (Усиление): {args.min_relevance} | Top-K: {args.top_k}")
    print(f"Режим LLM: {'ВЫКЛЮЧЕН (--no-llm)' if args.no_llm else 'ВКЛЮЧЕН (API)'}")

    try:
        retriever = load_retriever(
            index_dir=default_index_dir,
            dataset=args.dataset,
            strategy=args.strategy,
            top_k=args.top_k,
        )
        print(f"✓ FAISS-индекс загружен: {retriever.total_vectors} векторов в базе")
    except Exception as exc:
        print(f"✗ Ошибка загрузки индекса: {exc}", file=sys.stderr)
        sys.exit(1)

    # Двухэтапный пайплайн реранкинга для высокого качества кандидатов
    pipeline = RerankPipeline(
        retriever=retriever,
        retrieve_k=default_retrieve_k,
        final_k=args.top_k,
        min_score=args.min_relevance,
        rerank_method="heuristic",
        rewrite_method="heuristic",
    )

    # 3. Инициализация LLM-клиента и агента
    client: LLMClient | None = None
    if not args.no_llm:
        try:
            client = LLMClient(config)
            print(f"✓ LLM-клиент инициализирован (модель: {config.get('model', 'default')})")
        except Exception as exc:
            print(f"⚠️ Ошибка инициализации LLM: {exc}. Переключение в --no-llm режим.")
            client = None

    agent = GroundedRagAgent(
        client=client,
        retriever=retriever,
        relevance_threshold=args.min_relevance,
        top_k=args.top_k,
        temperature=float(ground_cfg.get("temperature", 0.2)),
        max_context_chars=int(ground_cfg.get("max_context_chars", 6000)),
        rerank_pipeline=pipeline,
    )

    # 4. Режим одиночного вопроса
    if args.question:
        print(f"\n[Запрос пользователя]: {args.question}")
        print("-" * 80)
        answer = agent.ask(args.question)
        print(answer.format_display())
        return

    # 5. Пакетная верификация 10 контрольных вопросов
    questions = GROUNDED_CONTROL_QUESTIONS
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    print("\n" + "=" * 80)
    print(f"ЭТАП 1: ПРОВЕРКА НА {len(questions)} КОНТРОЛЬНЫХ ВОПРОСАХ")
    print("Параметры проверки:")
    print("  👉 Есть ли источники в каждом ответе (source + section/chunk_id)")
    print("  👉 Есть ли цитаты в каждом ответе (фрагменты из найденных чанков)")
    print("  👉 Совпадает ли смысл ответа с цитатами (Semantic Alignment)")
    print("=" * 80)

    def _progress(current: int, total: int, query: str) -> None:
        print(f"[{current}/{total}] {_preview(query, 65)} ...", end=" ", flush=True)

    benchmark_summary = evaluate_grounded_benchmark(
        agent=agent,
        questions=questions,
        progress_callback=lambda cur, tot, q: print(f"[{cur}/{tot}] {_preview(q, 60)} ...", end=" ", flush=True) or sys.stdout.write(""),
    )
    print("✓ Завершено!\n")

    # Вывод результатов по 10 вопросам
    print("-" * 80)
    print(f"{'№':<3} | {'Вопрос':<40} | {'Источники':<12} | {'Цитаты':<10} | {'Смысл':<15} | {'Скор':<6}")
    print("-" * 80)
    for idx, r in enumerate(benchmark_summary["results"], start=1):
        src_status = "✓ Есть" if r["sources_present"] else "✗ Нет"
        quote_status = f"✓ ({len(r['quotes'])})" if r["quotes_present"] else "✗ Нет"
        align_str = f"✓ {r['alignment_verdict']}" if r["is_aligned"] else f"✗ {r['alignment_verdict']}"
        print(
            f"{idx:<3} | {_preview(r['query'], 38):<40} | {src_status:<12} | {quote_status:<10} | {align_str:<15} | {r['relevance_score']:<6.3f}"
        )
    print("-" * 80)
    print(
        f"ИТОГИ ЭТАПА 1:\n"
        f"  • Наличие источников: {benchmark_summary['sources_present_rate'] * 100:.1f}% "
        f"({benchmark_summary['sources_present_count']}/{benchmark_summary['total_questions']})\n"
        f"  • Наличие цитат:      {benchmark_summary['quotes_present_rate'] * 100:.1f}% "
        f"({benchmark_summary['quotes_present_count']}/{benchmark_summary['total_questions']})\n"
        f"  • Совпадение смысла:  {benchmark_summary['semantic_alignment_rate'] * 100:.1f}% "
        f"({benchmark_summary['semantic_aligned_count']}/{benchmark_summary['total_questions']})\n"
        f"  • Средняя задержка:   {benchmark_summary['avg_latency_ms']:.1f} мс"
    )

    # 6. Проверка правила усиления (порог релевантности)
    print("\n" + "=" * 80)
    print("ЭТАП 2: ПРОВЕРКА УСИЛЕНИЯ (ПОРОГ РЕЛЕВАНТНОСТИ И ОТКАЗ «НЕ ЗНАЮ»)")
    print(f"Порог отсечения: {args.min_relevance}")
    print("Правило: если релевантность ниже порога — сказать «не знаю» и попросить уточнение")
    print("=" * 80)

    guardrail_questions = LOW_RELEVANCE_TEST_QUESTIONS
    if args.limit and args.limit > 0:
        guardrail_questions = guardrail_questions[: args.limit]

    guardrail_summary = evaluate_low_relevance_guardrail(
        agent=agent,
        questions=guardrail_questions,
    )

    print("-" * 80)
    print(f"{'№':<3} | {'Вопрос':<40} | {'Скор':<6} | {'«Не знаю»':<10} | {'Уточнение':<10} | {'Цитаты':<8}")
    print("-" * 80)
    for idx, gr in enumerate(guardrail_summary["results"], start=1):
        ref_mark = "✓ Да" if gr["has_refusal"] else "✗ Нет"
        clar_mark = "✓ Да" if gr["has_clarification"] else "✗ Нет"
        quotes_mark = "✓ Пусто" if gr["has_empty_quotes"] else "✗ Выдумано"
        print(
            f"{idx:<3} | {_preview(gr['query'], 38):<40} | {gr['relevance_score']:<6.3f} | {ref_mark:<10} | {clar_mark:<10} | {quotes_mark:<8}"
        )
    print("-" * 80)
    print(
        f"ИТОГИ ЭТАПА 2:\n"
        f"  • Срабатывание «Не знаю»:       {guardrail_summary['refusals_rate'] * 100:.1f}% "
        f"({guardrail_summary['refusals_count']}/{guardrail_summary['total_tested']})\n"
        f"  • Запрос на уточнение:         {guardrail_summary['clarifications_rate'] * 100:.1f}% "
        f"({guardrail_summary['clarifications_count']}/{guardrail_summary['total_tested']})\n"
        f"  • Отсутствие ложных цитат:     {guardrail_summary['empty_quotes_rate'] * 100:.1f}% "
        f"({guardrail_summary['empty_quotes_count']}/{guardrail_summary['total_tested']})"
    )

    # 7. Сохранение отчётов
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "grounding_results.json"
    full_export = {
        "benchmark": benchmark_summary,
        "guardrail": guardrail_summary,
        "metadata": {
            "dataset": args.dataset,
            "strategy": args.strategy,
            "relevance_threshold": args.min_relevance,
            "top_k": args.top_k,
            "no_llm": args.no_llm,
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_export, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON-данные сохранены: {json_path}")

    md_report = render_grounding_markdown_report(
        benchmark_summary=benchmark_summary,
        guardrail_summary=guardrail_summary,
        relevance_threshold=args.min_relevance,
    )
    md_path = report_dir / "grounding_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"✓ Markdown-отчёт сохранён: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
