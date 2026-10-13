#!/usr/bin/env python3
"""CLI пересборки индекса рецептов.

Примеры::

    # скопировать готовый индекс из репозитория (по умолчанию)
    python3 rebuild_index.py --copy

    # пересобрать из датасета (та же структура JSON), стратегия structural
    python3 rebuild_index.py --source index/dataset/recipt_all.json \
        --dataset recipt_all --strategy structural

    # пересобрать все стратегии
    python3 rebuild_index.py --source index/dataset/recipt_all.json --strategy all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.index_build import (
    DEFAULT_INDEX_DIR,
    STRATEGIES,
    IndexBuildError,
    build_all_strategies,
    build_index,
    copy_index,
    resolve_dataset_source,
)

DEFAULT_COPY_SOURCE = Path(__file__).resolve().parent.parent / "history" / "index"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пересборка/копирование индекса рецептов.")
    parser.add_argument("--copy", action="store_true", help="скопировать готовый индекс")
    parser.add_argument("--source", default=None, help="JSON-датасет для пересборки")
    parser.add_argument("--dataset", default="recipt_all", help="имя датасета")
    parser.add_argument(
        "--strategy",
        default="structural",
        choices=(*STRATEGIES, "all"),
        help="стратегия чанкинга (или all)",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="каталог индекса")
    parser.add_argument("--dim", type=int, default=256, help="размерность хэш-эмбеддера")
    parser.add_argument(
        "--copy-source",
        default=str(DEFAULT_COPY_SOURCE),
        help="каталог с готовыми индексами для --copy",
    )
    return parser.parse_args(argv)


def _run_copy(args: argparse.Namespace) -> None:
    src_root = Path(args.copy_source).resolve()
    src = src_root / args.dataset / "structural"
    dst = Path(args.index_dir).resolve() / args.dataset / "structural"
    target = copy_index(src, dst)
    print(f"Индекс скопирован: {src} → {target}")


def _run_build(args: argparse.Namespace) -> None:
    source = Path(args.source) if args.source else resolve_dataset_source(
        args.dataset, args.index_dir
    )
    if not source.exists():
        fallback = resolve_dataset_source(args.dataset, args.index_dir)
        print(f"Датасет {source} не найден — использую {fallback}")
        source = fallback
    print(f"Источник датасета: {source}")
    if args.strategy == "all":
        targets = build_all_strategies(
            source, dataset=args.dataset, index_dir=args.index_dir, dim=args.dim
        )
        for target in targets:
            print(f"Индекс пересобран: {target}")
    else:
        target = build_index(
            source,
            dataset=args.dataset,
            strategy=args.strategy,
            index_dir=args.index_dir,
            dim=args.dim,
        )
        print(f"Индекс пересобран: {target}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.copy or not args.source:
            try:
                _run_copy(args)
            except IndexBuildError as exc:
                # На чистом клоне готового индекса в репозитории нет —
                # пересобираем из доступного датасета.
                print(f"Копирование недоступно ({exc}). Пересобираю из датасета.")
                _run_build(args)
        else:
            _run_build(args)
    except (IndexBuildError, FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
