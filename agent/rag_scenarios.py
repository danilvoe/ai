#!/usr/bin/env python3
"""Day 22: Сквозной сценарий первого RAG-запроса и сравнения с Baseline.

Сценарий реализует:
1. Вопрос → Поиск релевантных чанков (FAISS) → Объединение с вопросом → Запрос к LLM.
2. Сравнение ответов:
   - Ответ модели без RAG (общие знания, Baseline);
   - Ответ модели с RAG (обогащение контекстом из базы рецептов).
3. Мини-набор из 10 контрольных вопросов с зафиксированными ожиданиями и источниками.
4. Агент с двумя переключаемыми режимами (RagAgent).
5. Расчёт метрик качества (Fact Coverage, Source Recall@k, цитирование) и выгрузка отчёта.

Запуск::

    # Полный сценарий: 10 контрольных вопросов в двух режимах + Markdown-отчёт
    python3 -m agent.rag_scenarios

    # Демонстрация на одиночном вопросе (сравнение без RAG и с RAG)
    python3 -m agent.rag_scenarios --question "Сколько соли нужно на 1 кг мяса для прайм риб на гриле?"

    # Офлайн-прогон без сетевых вызовов к LLM (проверка поиска и ретривера)
    python3 -m agent.rag_scenarios --no-llm

    # Быстрый прогон на первых 3 вопросах
    python3 -m agent.rag_scenarios --limit 3

    # Выбор датасета и параметров
    python3 -m agent.rag_scenarios --dataset recipt_all --strategy structural --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import load_config
from agent.indexing import HashingEmbedder
from agent.llm_client import LLMClient
from agent.rag import (
    CONTROL_QUESTIONS,
    RagAgent,
    RagAnswer,
    evaluate_answer_facts,
    evaluate_citations,
    evaluate_rag_comparison,
    load_retriever,
    render_rag_comparison_markdown,
)


def _preview(text: str, limit: int = 140) -> str:
    """Однострочный предпросмотр текста."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _print_sources(sources: list, title: str = "Найденные источники (top-k)") -> None:
    print(f"\n   [{title}]:")
    if not sources:
        print("     (источники не найдены)")
        return
    for s in sources:
        sec = s.section
        if s.step_number is not None:
            sec = f"{sec} #{s.step_number}"
        print(
            f"     #{s.rank} [score={s.score:.4f}] [{sec}] "
            f"«{s.title}» (ID {s.recipe_id})"
        )
        print(f"        {_preview(s.text, 120)}")


def main() -> None:
    # 1. Чтение конфигурации
    config = load_config()
    rag_cfg = config.get("rag", {}) or {}
    idx_cfg = config.get("indexing", {}) or {}

    default_dataset = rag_cfg.get("dataset", "recipt_all")
    default_strategy = rag_cfg.get("strategy", "structural")
    default_index_dir = idx_cfg.get("index_dir", "history/index")
    default_top_k = int(rag_cfg.get("top_k", 5))
    default_temp = float(rag_cfg.get("temperature", 0.2))
    default_report_dir = rag_cfg.get("report_dir", "history/rag")
    default_max_chars = int(rag_cfg.get("max_context_chars", 6000))

    parser = argparse.ArgumentParser(
        description="Day 22: Первый RAG-запрос, сравнение ответов с/без RAG и 10 контрольных вопросов."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=default_dataset,
        help=f"имя датасета индекса (по умолчанию: '{default_dataset}')",
    )
    parser.add_argument(
        "--strategy",
        default=default_strategy,
        choices=["structural", "fixed"],
        help=f"стратегия чанкинга индекса (по умолчанию: '{default_strategy}')",
    )
    parser.add_argument(
        "--index-dir",
        default=default_index_dir,
        help=f"директория индексов (по умолчанию: '{default_index_dir}')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=default_top_k,
        help=f"количество чанков в контексте (по умолчанию: {default_top_k})",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=default_temp,
        help=f"температура генерации LLM (по умолчанию: {default_temp})",
    )
    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="одиночный пользовательский вопрос для демонстрации",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "rag", "no_rag"],
        default="both",
        help="режим генерации для одиночного вопроса (по умолчанию: both)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="ограничить число контрольных вопросов (по умолчанию: все 10)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="офлайн-режим: выполнить только поиск источников без запросов к LLM",
    )
    parser.add_argument(
        "--report-dir",
        default=default_report_dir,
        help=f"директория для сохранения отчётов (по умолчанию: '{default_report_dir}')",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="не сохранять отчёты на диск",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("Day 22: Первый RAG-запрос — Поиск → Контекст → Запрос к LLM")
    print("=" * 72)

    # 2. Инициализация ретривера
    print(f"\n[Шаг 1/4] Загрузка векторного индекса ({args.dataset}/{args.strategy})")
    embedder = HashingEmbedder(dim=int(idx_cfg.get("embedding_dim", 256)))
    try:
        retriever = load_retriever(
            index_dir=args.index_dir,
            dataset=args.dataset,
            strategy=args.strategy,
            embedder=embedder,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Ошибка загрузки ретривера: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"   База рецептов     : {args.dataset}")
    print(f"   Стратегия чанков  : {args.strategy}")
    print(f"   Всего векторов    : {retriever.total_vectors}")
    print(f"   Эмбеддер          : {embedder.name} (dim={embedder.dim})")
    print(f"   Размер контекста  : top-{args.top_k} чанков")

    # 3. Инициализация LLM-клиента и RagAgent
    client = LLMClient(config)
    agent = RagAgent(
        client=client,
        retriever=retriever,
        mode="rag",
        top_k=args.top_k,
        temperature=args.temperature,
        max_context_chars=default_max_chars,
    )
    print(f"   Модель LLM        : {config.get('model', 'deepseek-v4-flash')}")
    print(f"   Температура       : {args.temperature}")
    print(f"   Агент инициализирован в режиме: {agent.mode}")

    # 4. Режим одиночного вопроса
    if args.question:
        q = args.question
        print(f"\n[Шаг 2/4] Демонстрация одиночного вопроса: «{q}»")

        sources = retriever.retrieve(q, top_k=args.top_k)
        _print_sources(sources)

        if not args.no_llm:
            if args.mode in ("both", "no_rag"):
                print("\n" + "-" * 72)
                print("1. ОТВЕТ БЕЗ RAG (общие знания модели / Baseline):")
                print("-" * 72)
                ans_no = agent.ask_without_rag(q)
                if ans_no.error:
                    print(f"Ошибка вызова LLM: {ans_no.error}")
                else:
                    print(ans_no.content.strip())
                    print(f"\n[Токены: {ans_no.total_tokens}, время: {ans_no.latency_ms:.0f} мс]")

            if args.mode in ("both", "rag"):
                print("\n" + "-" * 72)
                print("2. ОТВЕТ С RAG (на основе контекста из базы рецептов):")
                print("-" * 72)
                ans_rag = agent.ask_with_rag(q)
                if ans_rag.error:
                    print(f"Ошибка вызова LLM: {ans_rag.error}")
                else:
                    print(ans_rag.content.strip())
                    cites = evaluate_citations(ans_rag.content, sources)
                    print(
                        f"\n[Токены: {ans_rag.total_tokens}, время: {ans_rag.latency_ms:.0f} мс | "
                        f"Ссылки на источники: {'Да' if cites['has_citations'] else 'Нет'}]"
                    )
        else:
            print("\n[--no-llm]: Запрос к LLM пропущен, отображены только результаты поиска.")

        print("\n" + "=" * 72)
        print("Готово!")
        print("=" * 72 + "\n")
        return

    # 5. Режим пакетного тестирования 10 контрольных вопросов
    questions = CONTROL_QUESTIONS
    if args.limit:
        questions = questions[: args.limit]

    print("\n[Шаг 2/4] Демонстрация первого RAG-запроса на вопросе №1")
    q1 = questions[0]
    print(f"   Вопрос: «{q1['query']}»")
    print(f"   Ожидание: {_preview(q1['expectation'], 120)}")
    demo_sources = retriever.retrieve(q1["query"], top_k=args.top_k)
    _print_sources(demo_sources, f"Чанки для вопроса №1 (top-{args.top_k})")

    demo_ans_no: RagAnswer | None = None
    demo_ans_rag: RagAnswer | None = None

    if not args.no_llm:
        print("\n   [Запрос к модели: режим БЕЗ RAG]")
        demo_ans_no = agent.ask_without_rag(q1["query"])
        print(f"   Ответ без RAG:\n   {_preview(demo_ans_no.content, 180)}\n")

        print("   [Запрос к модели: режим С RAG]")
        demo_ans_rag = agent.ask_with_rag(q1["query"])
        print(f"   Ответ с RAG:\n   {_preview(demo_ans_rag.content, 180)}\n")

        f_no = evaluate_answer_facts(demo_ans_no.content, q1.get("expected_facts", []))
        f_rag = evaluate_answer_facts(demo_ans_rag.content, q1.get("expected_facts", []))
        print(
            f"   Сравнение покрытия фактов: Без RAG = {f_no['coverage'] * 100:.0f}% "
            f"({f_no['hits']}/{f_no['total']}) | С RAG = {f_rag['coverage'] * 100:.0f}% ({f_rag['hits']}/{f_rag['total']})"
        )

    # 6. Прогон полного набора контрольных вопросов
    print(
        f"\n[Шаг 3/4] Прогон мини-набора из {len(questions)} контрольных вопросов "
        f"({'LLM запрос' if not args.no_llm else 'офлайн/только поиск'})"
    )

    def _progress(current: int, total: int, query_text: str) -> None:
        print(f"   [{current:>2}/{total}] {_preview(query_text, 68)}")

    report = evaluate_rag_comparison(
        agent=agent,
        questions=questions,
        progress_callback=_progress,
        run_llm=not args.no_llm,
    )

    # 7. Вывод результатов в консоль
    print("\n[Шаг 4/4] Сводная таблица качества ответов")
    summary = report["summary"]
    results = report["results"]

    print("\n" + "=" * 78)
    print(
        f"{'№':<3} | {'Вопрос':<36} | {'Ранг':<8} | {'Без RAG':<10} | {'С RAG':<10} | {'Ссылки'}"
    )
    print("-" * 78)
    for idx, r in enumerate(results, start=1):
        q_short = r["query"][:34] + "…" if len(r["query"]) > 34 else r["query"]
        rank_val = r["source_recall"]["top_rank"]
        rank_str = f"#{rank_val}" if rank_val else "—"
        f_no = f"{r['no_rag']['facts']['coverage'] * 100:.0f}%"
        f_rag = f"{r['rag']['facts']['coverage'] * 100:.0f}%"
        cites = "Да" if r["rag"]["citations"]["has_citations"] else "Нет"
        print(
            f"{idx:<3} | {q_short:<36} | {rank_str:<8} | {f_no:<10} | {f_rag:<10} | {cites}"
        )
    print("=" * 78)

    print("\n   Агрегированные показатели:")
    print(f"     - Среднее покрытие фактов БЕЗ RAG : {summary['avg_facts_no_rag'] * 100:.1f}%")
    print(f"     - Среднее покрытие фактов С RAG   : {summary['avg_facts_rag'] * 100:.1f}%")
    print(f"     - Прирост точности фактов         : {summary['facts_gain'] * 100:+.1f}%")
    print(f"     - Нахождение источника в top-{args.top_k}  : {summary['source_recall_top_k'] * 100:.1f}%")
    print(f"     - Нахождение источника на 1 месте : {summary['source_recall_top_1'] * 100:.1f}%")
    print(f"     - Доля ответов со ссылками        : {summary['citation_rate'] * 100:.1f}%")

    # 8. Сохранение отчётов
    if not args.skip_save:
        out_dir = Path(args.report_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / "rag_comparison.md"
        json_path = out_dir / "rag_results.json"

        md_text = render_rag_comparison_markdown(report)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_text)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n   Markdown-отчёт сохранён в : {md_path}")
        print(f"   JSON-результаты сохранены в: {json_path}")

    print("\n" + "=" * 72)
    print("Сценарий RAG успешно завершён!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
