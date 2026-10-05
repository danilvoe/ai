#!/usr/bin/env python3
"""Day 21: Сквозной сценарий индексации документов и сравнения стратегий chunking.

Сценарий демонстрирует полный пайплайн:
1. Загрузка документов из JSON (по умолчанию recipt_sample.json, поддерживается любой файл через --source).
2. Две стратегии chunking:
   - по фиксированному размеру окна со сдвигом;
   - по логической структуре рецепта (шапка, ингредиенты, описание, шаги, оборудование).
3. Обогащение чанков метаданными (усиление): ID, раздел, номер шага, автор, ссылка, токены.
4. Построение детерминированных эмбеддингов (HashingEmbedder, офлайн) или через OpenAI API.
5. Создание, сохранение и проверка перезагрузки векторных индексов FAISS (IndexFlatIP).
6. Сравнение стратегий по метрикам структуры и качества поиска (Recall@k, MRR@k).

Запуск::

    # Базовый прогон по умолчанию (recipt_sample.json, офлайн)
    python3 -m agent.indexing_scenarios

    # Промышленный прогон по другому (более полному) файлу рецептов
    python3 -m agent.indexing_scenarios --source path/to/full_recipes.json

    # Пользовательский поисковый запрос по готовым индексам
    python3 -m agent.indexing_scenarios --query "маринад для говядины с вином"

    # Настройка параметров фиксированного окна и top-k
    python3 -m agent.indexing_scenarios --chunk-size 350 --overlap 70 --top-k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.config import load_config
from agent.indexing import (
    DEFAULT_PROBE_QUERIES,
    Chunk,
    Embedder,
    FaissIndex,
    HashingEmbedder,
    OpenAIEmbedder,
    compare_strategies,
    load_recipes,
    render_comparison_markdown,
)


def _preview(text: str, limit: int = 140) -> str:
    """Компактное однострочное превью текста."""
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _print_chunk_sample(title: str, chunk: Chunk) -> None:
    """Красиво выводит образец чанка и его структурированные метаданные."""
    print(f"   [{title}] ID: {chunk.id}")
    print(f"     Текст: {_preview(chunk.text, 150)}")
    print("     Метаданные:")
    meta_items = [
        f"рецепт #{chunk.metadata.get('recipe_id')}",
        f"раздел: {chunk.metadata.get('section')}",
    ]
    if chunk.metadata.get("step_number") is not None:
        meta_items.append(f"шаг №{chunk.metadata['step_number']}")
    if chunk.metadata.get("ingredient_count") is not None:
        meta_items.append(f"ингредиентов: {chunk.metadata['ingredient_count']}")
    meta_items.extend([
        f"символов: {chunk.metadata.get('n_chars')}",
        f"токенов ≈ {chunk.metadata.get('n_tokens')}",
    ])
    print(f"       {', '.join(meta_items)}")
    print()


def _print_search_results(results: list[dict], title: str) -> None:
    print(f"   Результаты для {title}:")
    for r in results:
        meta = r["metadata"]
        step_str = f" [шаг {meta['step_number']}]" if meta.get("step_number") else ""
        sec_str = f"[{meta.get('section')}{step_str}]"
        print(
            f"     #{r['rank']} score={r['score']:.4f} | {sec_str} "
            f"«{meta.get('title')}» (ID {meta.get('recipe_id')})"
        )
        print(f"        {_preview(r['text'], 130)}")
    print()


def main() -> None:
    # 1. Чтение конфигурации
    config = load_config()
    idx_cfg = config.get("indexing", {}) or {}

    default_source = idx_cfg.get("source_path", "recipt_sample.json")
    default_index_dir = idx_cfg.get("index_dir", "history/index")
    default_chunk_size = int(idx_cfg.get("chunk_size", 400))
    default_overlap = int(idx_cfg.get("chunk_overlap", 80))
    default_top_k = int(idx_cfg.get("top_k", 5))
    default_embedder = idx_cfg.get("embedder", "local")

    parser = argparse.ArgumentParser(
        description="Day 21: Пайплайн индексации документов и сравнение стратегий chunking."
    )
    parser.add_argument(
        "--source",
        "-s",
        default=default_source,
        help=(
            f"путь к JSON-файлу с рецептами (по умолчанию: '{default_source}'; "
            "для промышленного прогона укажите полный датасет)"
        ),
    )
    parser.add_argument(
        "--index-dir",
        "-o",
        default=default_index_dir,
        help=f"корневая директория для сохранения индексов FAISS (по умолчанию: '{default_index_dir}')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=default_chunk_size,
        help=f"размер скользящего окна в символах для fixed chunking (по умолчанию: {default_chunk_size})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=default_overlap,
        help=f"перекрытие скользящего окна в символах (по умолчанию: {default_overlap})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help=f"число возвращаемых соседей при поиске (по умолчанию: {default_top_k})",
    )
    parser.add_argument(
        "--embedder",
        choices=["local", "api"],
        default=default_embedder,
        help="тип генератора эмбеддингов: 'local' (HashingEmbedder офлайн) или 'api' (OpenAI)",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="произвольный поисковый запрос для проверки обоих сохранённых индексов",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="не сохранять индексы на диск (только расчет метрик в памяти)",
    )
    args = parser.parse_args()

    # 2. Подготовка генератора эмбеддингов
    embedder: Embedder
    if args.embedder == "api":
        api_key = config.get("api_key")
        base_url = config.get("base_url", "https://openai.bothub.chat/v1")
        if not api_key:
            print("Ошибка: для embedder=api требуется api_key в config.json.", file=sys.stderr)
            raise SystemExit(1)
        embedder = OpenAIEmbedder(api_key=api_key, base_url=base_url)
    else:
        embedder = HashingEmbedder(dim=int(idx_cfg.get("embedding_dim", 256)))

    print("=" * 68)
    print("Day 21: Пайплайн индексации документов и FAISS-поиск")
    print("=" * 68)

    # 3. Шаг 1: Загрузка документов
    print("\n[Шаг 1/5] Загрузка документов из источника")
    source_path = Path(args.source)
    try:
        recipes, file_meta = load_recipes(source_path)
    except Exception as exc:
        print(f"Ошибка загрузки источника {source_path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"   Файл источника      : {file_meta['source_path']}")
    print(f"   Размер файла        : {file_meta['source_size_bytes']} байт")
    print(f"   Заявлено рецептов   : {file_meta['total_declared']}")
    print(f"   Успешно загружено   : {file_meta['total_loaded']}")
    if file_meta['total_loaded'] == 0:
        print("В источнике не найдено рецептов для индексации.", file=sys.stderr)
        raise SystemExit(1)

    dataset_name = source_path.stem

    # 4. Шаг 2: Выполнение chunking по 2 стратегиям
    print("\n[Шаг 2/5] Разбиение на чанки (Chunking): 2 стратегии + метаданные")
    print(f"   Стратегия А (Fixed)     : окно={args.chunk_size} симв., перекрытие={args.overlap} симв.")
    print("   Стратегия Б (Structural): разделы (header, ingredients, intro, step, equipment)")
    print(f"   Эмбеддер                : {embedder.name} (размерность {embedder.dim})")

    report = compare_strategies(
        recipes=recipes,
        source_name=dataset_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        embedder=embedder,
        top_k=args.top_k,
        queries=DEFAULT_PROBE_QUERIES,
    )

    fixed_info = report["fixed"]
    struct_info = report["structural"]
    f_chunks: list[Chunk] = fixed_info["chunks"]
    s_chunks: list[Chunk] = struct_info["chunks"]

    print("\n   Чанков получено:")
    print(f"     - Fixed     : {len(f_chunks)} (время {fixed_info['chunking_time_ms']} мс)")
    print(f"     - Structural: {len(s_chunks)} (время {struct_info['chunking_time_ms']} мс)")

    print("\n   Примеры чанков со структурированными метаданными:")
    if f_chunks:
        _print_chunk_sample("Стратегия Fixed: Образец №1", f_chunks[0])
    if s_chunks:
        # Покажем ингредиенты и один шаг
        sample_ing = next((c for c in s_chunks if c.metadata.get("section") == "ingredients"), s_chunks[0])
        sample_step = next((c for c in s_chunks if c.metadata.get("section") == "step"), s_chunks[min(1, len(s_chunks) - 1)])
        _print_chunk_sample("Стратегия Structural: Ингредиенты", sample_ing)
        _print_chunk_sample("Стратегия Structural: Шаг рецепта", sample_step)

    # 5. Шаг 3: Сохранение и проверка индексов FAISS
    print("[Шаг 3/5] Создание и сохранение индексов FAISS на диске")
    print("   Индексация векторов (FAISS IndexFlatIP):")
    print(f"     - Fixed     : {fixed_info['index'].total} векторов ({fixed_info['indexing_time_ms']} мс)")
    print(f"     - Structural: {struct_info['index'].total} векторов ({struct_info['indexing_time_ms']} мс)")

    base_index_dir = Path(args.index_dir).resolve() / dataset_name
    fixed_dir = base_index_dir / "fixed"
    struct_dir = base_index_dir / "structural"

    if not args.skip_save:
        fixed_info["index"].save(fixed_dir)
        struct_info["index"].save(struct_dir)
        print(f"   Индекс Fixed сохранён      : {fixed_dir}")
        print(f"   Индекс Structural сохранён : {struct_dir}")

        # Проверка перезагрузки с диска
        loaded_fixed = FaissIndex.load(fixed_dir, embedder)
        loaded_struct = FaissIndex.load(struct_dir, embedder)
        print(f"   Проверка целостности : успешно загружено {loaded_fixed.total} и {loaded_struct.total} векторов.")
    else:
        print("   Сохранение пропущено (--skip-save).")

    # 6. Шаг 4: Сравнение качества поиска и статистик
    print("\n[Шаг 4/5] Сравнение двух стратегий chunking")
    f_stats = fixed_info["stats"]
    s_stats = struct_info["stats"]
    f_ret = fixed_info["retrieval"]
    s_ret = struct_info["retrieval"]

    table_rows = [
        ("Количество чанков", f"{f_stats['count']}", f"{s_stats['count']}"),
        ("Средняя длина чанка", f"{f_stats['avg_chars']} симв.", f"{s_stats['avg_chars']} симв."),
        ("Мин / Макс длина", f"{f_stats['min_chars']} / {f_stats['max_chars']} симв.", f"{s_stats['min_chars']} / {s_stats['max_chars']} симв."),
        ("Всего токенов (оценка)", f"{f_stats['total_tokens']}", f"{s_stats['total_tokens']}"),
        ("Recall@1 (топ-1 точен)", f"{f_ret['recall_at_1'] * 100:.1f}%", f"{s_ret['recall_at_1'] * 100:.1f}%"),
        (f"Recall@{args.top_k} (в топ-{args.top_k})", f"{f_ret[f'recall_at_{args.top_k}'] * 100:.1f}%", f"{s_ret[f'recall_at_{args.top_k}'] * 100:.1f}%"),
        (f"MRR@{args.top_k} (обратный ранг)", f"{f_ret[f'mrr_at_{args.top_k}']:.3f}", f"{s_ret[f'mrr_at_{args.top_k}']:.3f}"),
        ("Средний score топ-1", f"{f_ret['avg_top_score']:.4f}", f"{s_ret['avg_top_score']:.4f}"),
    ]

    col1_w = max(len(r[0]) for r in table_rows) + 2
    col2_w = max(len(r[1]) for r in table_rows) + 2
    col3_w = max(len(r[2]) for r in table_rows) + 2

    header = f"   {'Параметр':<{col1_w}} | {'Fixed (окно)':<{col2_w}} | {'Structural (разделы)':<{col3_w}}"
    sep = f"   {'-' * col1_w}-+-{'-' * col2_w}-+-{'-' * col3_w}"
    print(sep)
    print(header)
    print(sep)
    for p, v1, v2 in table_rows:
        print(f"   {p:<{col1_w}} | {v1:<{col2_w}} | {v2:<{col3_w}}")
    print(sep)

    print("\n   Распределение разделов в Structural:")
    for sec, cnt in sorted(s_stats["sections"].items(), key=lambda x: -x[1]):
        print(f"     - {sec:<14}: {cnt:>3} чанков")

    # 7. Пользовательский запрос (если задан)
    test_q = args.query or "Апельсиновая курица на гриле новогодний рецепт"
    print(f"\n[Шаг 5/5] Демонстрация поиска: «{test_q}»")
    res_fixed = fixed_info["index"].search(test_q, top_k=3)
    res_struct = struct_info["index"].search(test_q, top_k=3)

    _print_search_results(res_fixed, "Fixed Chunking")
    _print_search_results(res_struct, "Structural Chunking")

    # Сохранение Markdown-отчёта
    if not args.skip_save:
        report_md = render_comparison_markdown(report)
        report_file = base_index_dir / "comparison.md"
        with report_file.open("w", encoding="utf-8") as file:
            file.write(report_md)
        print(f"   Подробный отчёт сохранён в: {report_file}")

    print("\n" + "=" * 68)
    print("Индексация и сравнение завершены успешно!")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
