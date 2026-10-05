#!/usr/bin/env python3
"""Day 23: Сквозной сценарий реранкинга, фильтрации релевантности и Query Rewrite.

Сценарий реализует:
1. Двухэтапный RAG:
   👉 Query Rewrite (очистка/нормализация/LLM)
   👉 Векторный поиск кандидатов (retrieve_k, top-K до фильтрации)
   👉 Фильтр релевантности (порог отсечения нерелевантных чанков min_score)
   👉 Реранкер (Heuristic / LLM: взвешивание лексики, названий и структуры)
   👉 Финальный топ-K для контекста LLM (final_k, top-K после фильтрации).
2. Сравнение режимов:
   - baseline: без фильтра и rewriting (Day 22);
   - filter_only: только отсечение по порогу similarity;
   - rerank_only: только лексико-семантический реранкер;
   - full_improved: связка rewrite + filter + reranker.
3. Метрики:
   - Precision@K по целевым источникам, доля отсеянного шума, сокращение контекста,
   - Покрытие фактов (Fact Coverage) и цитирование в ответах LLM.
4. Выгрузка Markdown-отчёта и JSON-структуры в history/rag_rerank/.

Запуск::

    # Полное сравнительное тестирование всех режимов на 10 вопросах + Markdown-отчёт
    python3 -m agent.reranking_scenarios

    # Быстрый офлайн-прогон без обращений к LLM (проверка ретривера, фильтра и реранкера)
    python3 -m agent.reranking_scenarios --no-llm

    # Демонстрация на одиночном вопросе (показ кандидатов до/после фильтрации и ответа)
    python3 -m agent.reranking_scenarios --question "Сколько соли нужно на 1 кг мяса для прайм риб на гриле?"

    # Ограничение числа вопросов (например, первые 3 вопроса)
    python3 -m agent.reranking_scenarios --limit 3

    # Настройка параметров отсечения и top-K
    python3 -m agent.reranking_scenarios --retrieve-k 15 --final-k 5 --min-score 0.33
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
    evaluate_citations,
    load_retriever,
)
from agent.reranking import (
    RerankPipeline,
    evaluate_rerank_comparison,
    rag_query_with_pipeline,
    render_rerank_comparison_markdown,
    rewrite_query,
)


def _preview(text: str, limit: int = 120) -> str:
    """Однострочный предпросмотр текста."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _print_candidate(c, prefix: str = "     ") -> None:
    src = c.source
    sec = src.section
    if src.step_number is not None:
        sec = f"{sec} #{src.step_number}"
    passed_mark = "✓" if c.passed_filter else "✗"
    status = f"[{passed_mark}]"
    drop = f" ({c.drop_reason})" if c.drop_reason else ""
    print(
        f"{prefix}{status} #{c.retrieval_rank} -> rerank #{c.rerank_rank or '-'} "
        f"[vec={c.retrieval_score:.3f}, rerank={c.rerank_score:.3f}] "
        f"«{src.title}» (ID {src.recipe_id}, {sec}){drop}"
    )
    print(f"{prefix}    {_preview(src.text, 110)}")


def main() -> None:
    # 1. Загрузка конфигурации
    config = load_config()
    rag_cfg = config.get("rag", {}) or {}
    idx_cfg = config.get("indexing", {}) or {}
    rerank_cfg = config.get("reranking", {}) or {}

    default_dataset = rerank_cfg.get("dataset", rag_cfg.get("dataset", "recipt_all"))
    default_strategy = rerank_cfg.get("strategy", rag_cfg.get("strategy", "structural"))
    default_index_dir = rerank_cfg.get("index_dir", idx_cfg.get("index_dir", "history/index"))
    default_retrieve_k = int(rerank_cfg.get("retrieve_k", 15))
    default_final_k = int(rerank_cfg.get("final_k", rag_cfg.get("top_k", 5)))
    default_min_score = float(rerank_cfg.get("min_score", 0.33))
    default_score_ratio = float(rerank_cfg.get("score_ratio", 0.0))
    default_method = rerank_cfg.get("method", "heuristic")
    default_rewrite = rerank_cfg.get("rewrite", "heuristic")
    default_report_dir = rerank_cfg.get("report_dir", "history/rag_rerank")
    default_temp = float(rerank_cfg.get("temperature", rag_cfg.get("temperature", 0.2)))

    parser = argparse.ArgumentParser(
        description="Day 23: Реранкинг, фильтрация релевантности и Query Rewrite для RAG."
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
        "--retrieve-k",
        type=int,
        default=default_retrieve_k,
        help=f"число кандидатов первого этапа (top-K до фильтрации, по умолчанию: {default_retrieve_k})",
    )
    parser.add_argument(
        "--final-k",
        "-k",
        type=int,
        default=default_final_k,
        help=f"число чанков в контексте (top-K после фильтрации, по умолчанию: {default_final_k})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=default_min_score,
        help=f"порог отсечения нерелевантных чанков (по умолчанию: {default_min_score})",
    )
    parser.add_argument(
        "--score-ratio",
        type=float,
        default=default_score_ratio,
        help=f"относительный порог к лучшему результату (по умолчанию: {default_score_ratio})",
    )
    parser.add_argument(
        "--method",
        choices=["heuristic", "llm", "none"],
        default=default_method,
        help=f"метод реранкинга (по умолчанию: '{default_method}')",
    )
    parser.add_argument(
        "--rewrite",
        choices=["heuristic", "llm", "none"],
        default=default_rewrite,
        help=f"метод Query Rewrite (по умолчанию: '{default_rewrite}')",
    )
    parser.add_argument(
        "--modes",
        default="baseline,filter_only,rerank_only,rewrite_only,full_improved",
        help="список сравниваемых режимов через запятую (по умолчанию: baseline,filter_only,rerank_only,rewrite_only,full_improved)",
    )
    parser.add_argument(
        "--question",
        "-q",
        default=None,
        help="одиночный вопрос для пошаговой демонстрации этапов RAG",
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
        help="офлайн-режим: выполнить только поиск, фильтрацию и реранкинг без вызовов к LLM",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=default_temp,
        help=f"температура генерации ответов LLM (по умолчанию: {default_temp})",
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

    print("=" * 76)
    print("Day 23: Реранкинг, фильтрация релевантности и Query Rewrite (Двухэтапный RAG)")
    print("=" * 76)

    # 2. Инициализация ретривера
    print(f"\n[Шаг 1/4] Загрузка векторного индекса ({args.dataset}/{args.strategy})")
    embedder = HashingEmbedder(dim=int(idx_cfg.get("embedding_dim", 256)))
    try:
        retriever = load_retriever(
            index_dir=args.index_dir,
            dataset=args.dataset,
            strategy=args.strategy,
            embedder=embedder,
            top_k=args.retrieve_k,
        )
    except Exception as exc:
        print(f"Ошибка загрузки ретривера: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"   База рецептов       : {args.dataset}")
    print(f"   Стратегия чанков    : {args.strategy}")
    print(f"   Всего векторов      : {retriever.total_vectors}")
    print(f"   Top-K до фильтрации : {args.retrieve_k} кандидатов (retrieve_k)")
    print(f"   Порог отсечения     : min_score >= {args.min_score}")
    print(f"   Top-K после фильтра : {args.final_k} источников в контексте (final_k)")
    print(f"   Реранкер            : {args.method}")
    print(f"   Query Rewrite       : {args.rewrite}")

    # 3. Инициализация клиента LLM (если нужен)
    client: LLMClient | None = None
    if not args.no_llm:
        client = LLMClient(config)
        print(f"   Модель LLM          : {config.get('model', 'gemini-3.5-flash-lite')}")
    else:
        print("   Режим LLM           : отключён (--no-llm, только офлайн-анализ качества)")

    # 4. Демонстрация одиночного вопроса
    if args.question:
        q = args.question
        print(f"\n[Шаг 2/4] Демонстрация двухэтапного RAG на вопросе: «{q}»")

        # А. Query Rewrite
        rw_query = rewrite_query(q, method=args.rewrite, client=client)
        print(f"\n   [1. Query Rewrite ({args.rewrite})]:")
        print(f"     Исходный запрос     : «{q}»")
        print(f"     Очищенный/поисковый : «{rw_query}»")

        # Б. Pipeline
        pipeline = RerankPipeline(
            retriever=retriever,
            retrieve_k=args.retrieve_k,
            final_k=args.final_k,
            min_score=args.min_score,
            score_ratio=args.score_ratio,
            rerank_method=args.method,
            rewrite_method=args.rewrite,
            llm_client=client,
        )
        outcome = pipeline.run(q)

        print(f"\n   [2. Первый этап: Поиск top-{args.retrieve_k} кандидатов (FAISS)]:")
        print(f"     Найдено кандидатов  : {len(outcome.all_candidates)}")
        print(
            f"     Время поиска        : {outcome.timing_ms.get('retrieve_ms', 0):.1f} мс "
            f"(rewrite: {outcome.timing_ms.get('rewrite_ms', 0):.1f} мс)"
        )

        print(f"\n   [3. Второй этап: Фильтрация (порог {args.min_score}) и Реранкинг ({args.method})]:")
        for c in outcome.all_candidates:
            _print_candidate(c)

        print(
            f"\n   [Итог отбора]: оставлено {len(outcome.kept_sources)} из {len(outcome.all_candidates)} "
            f"(отсеяно шума: {len(outcome.dropped_candidates)})"
        )

        # В. Запрос к LLM (если включено)
        if not args.no_llm and client is not None:
            print("\n" + "-" * 76)
            print("ОТВЕТ УЛУЧШЕННОГО RAG (после фильтрации и реранкинга):")
            print("-" * 76)
            answer = rag_query_with_pipeline(
                client=client,
                pipeline=pipeline,
                question=q,
                temperature=args.temperature,
            )
            if answer.error:
                print(f"Ошибка вызова LLM: {answer.error}")
            else:
                print(answer.content.strip())
                cites = evaluate_citations(answer.content, outcome.kept_sources)
                print(
                    f"\n[Токены: {answer.total_tokens}, время: {answer.latency_ms:.0f} мс | "
                    f"Источников в контексте: {len(outcome.kept_sources)} | "
                    f"Ссылки: {'Да' if cites['has_citations'] else 'Нет'}]"
                )

        print("\n" + "=" * 76)
        print("Готово!")
        print("=" * 76 + "\n")
        return

    # 5. Пакетное тестирование на контрольных вопросах
    questions = CONTROL_QUESTIONS
    if args.limit:
        questions = questions[: args.limit]

    modes_to_test = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(
        f"\n[Шаг 2/4] Запуск сравнения режимов на {len(questions)} контрольных вопросах "
        f"({'LLM запросы' if not args.no_llm else 'офлайн/только ранжирование'})"
    )
    print(f"   Тестируемые режимы: {', '.join(modes_to_test)}")

    def _progress(step: int, total: int, mode_name: str, query_text: str) -> None:
        print(f"   [{step:>2}/{total}] ({mode_name:<13}) {_preview(query_text, 55)}")

    report = evaluate_rerank_comparison(
        retriever=retriever,
        client=client,
        modes=modes_to_test,
        questions=questions,
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        min_score=args.min_score,
        run_llm=not args.no_llm,
        progress_callback=_progress,
    )

    # 6. Сводная таблица сравнения в консоли
    print("\n[Шаг 3/4] Сводная таблица качества по режимам")
    summaries = report["mode_summaries"]

    print("\n" + "=" * 88)
    print(
        f"{'Режим':<18} | {'P@K':<8} | {'Recall':<8} | {'Top-1':<8} | "
        f"{'Факты':<10} | {'Ссылки':<8} | {'Ср.чанков':<10} | {'Шум cut'}"
    )
    print("-" * 88)

    mode_labels = {
        "baseline": "1. Baseline (D22)",
        "filter_only": "2. Filter only",
        "rerank_only": "3. Rerank only",
        "rewrite_only": "4. Rewrite only",
        "full_improved": "5. Full improved",
    }

    for m in modes_to_test:
        s = summaries.get(m, {})
        label = mode_labels.get(m, m)
        p_k = f"{s.get('avg_precision_at_k', 0) * 100:.1f}%"
        rec = f"{s.get('source_recall_rate', 0) * 100:.1f}%"
        top1 = f"{s.get('top1_hit_rate', 0) * 100:.1f}%"
        facts = f"{s.get('avg_facts_coverage', 0) * 100:.1f}%" if not args.no_llm else "—"
        cites = f"{s.get('citation_rate', 0) * 100:.1f}%" if not args.no_llm else "—"
        kept = f"{s.get('avg_kept_chunks', 0):.2f}"
        noise = f"{s.get('noise_filter_ratio', 0) * 100:.1f}%"
        print(
            f"{label:<18} | {p_k:<8} | {rec:<8} | {top1:<8} | "
            f"{facts:<10} | {cites:<8} | {kept:<10} | {noise}"
        )
    print("=" * 88)

    # Ключевой эффект
    if "baseline" in summaries and "full_improved" in summaries:
        b = summaries["baseline"]
        f = summaries["full_improved"]
        p_gain = (f.get("avg_precision_at_k", 0) - b.get("avg_precision_at_k", 0)) * 100
        fact_gain = (f.get("avg_facts_coverage", 0) - b.get("avg_facts_coverage", 0)) * 100
        noise_drop = f.get("noise_filter_ratio", 0) * 100
        print("\n   Эффект фильтрации и реранкинга (Full vs Baseline):")
        print(f"     - Прирост Precision@K (чистота контекста) : {p_gain:+.1f}%")
        print(f"     - Доля отсеянного шума (Noise Reduction)   : {noise_drop:.1f}%")
        if not args.no_llm:
            print(f"     - Прирост полноты фактов в ответах LLM    : {fact_gain:+.1f}%")
        print(f"     - Оптимизация размера контекста            : с {b.get('avg_kept_chunks', 5):.1f} до {f.get('avg_kept_chunks', 5):.1f} чанков")

    # 7. Сохранение отчётов
    print(f"\n[Шаг 4/4] Сохранение отчётов в {args.report_dir}")
    if not args.skip_save:
        out_dir = Path(args.report_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / "reranking_comparison.md"
        json_path = out_dir / "reranking_results.json"

        md_text = render_rerank_comparison_markdown(report)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_text)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"   Markdown-отчёт сохранён в : {md_path}")
        print(f"   JSON-результаты сохранены в: {json_path}")
    else:
        print("   (--skip-save указан: сохранение пропущено)")

    print("\n" + "=" * 76)
    print("Сценарий реранкинга и фильтрации успешно завершён!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
