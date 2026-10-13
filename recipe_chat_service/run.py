#!/usr/bin/env python3
"""Точка входа сервиса рецептов.

Поднимает Flask-приложение и (при необходимости) локальный ``llama-server``.
Индекс можно пересобрать перед запуском: ``--rebuild-index`` (всегда) или
``--rebuild-if-missing`` (по умолчанию, только если индекса нет).

Примеры::

    python3 run.py --offline            # UI на детерминированном стабе
    python3 run.py                       # живая локальная модель
    python3 run.py --rebuild-index       # пересобрать индекс и запустить
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app import create_app
from app.config import ServiceConfig
from core.index_build import build_index, resolve_dataset_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Веб-сервис рецептов с локальным RAG.")
    parser.add_argument("--offline", action="store_true", help="офлайн-стаб вместо llama-server")
    parser.add_argument("--host", default=None, help="хост Flask (по умолчанию из конфига)")
    parser.add_argument("--port", type=int, default=None, help="порт Flask")
    parser.add_argument("--model", default=None, help="путь к GGUF-модели")
    parser.add_argument("--index-dir", default=None, help="каталог индекса")
    parser.add_argument("--dataset", default=None, help="имя датасета (по умолчанию recipt_all)")
    parser.add_argument("--strategy", default=None, help="стратегия чанкинга")
    parser.add_argument("--source", default=None, help="JSON-датасет для пересборки")
    parser.add_argument("--config", default=None, help="путь к config.json")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="пересобрать индекс перед запуском",
    )
    parser.add_argument(
        "--rebuild-if-missing",
        dest="rebuild_if_missing",
        action="store_true",
        default=True,
        help="пересобрать индекс, только если его нет (по умолчанию)",
    )
    parser.add_argument(
        "--no-rebuild-if-missing",
        dest="rebuild_if_missing",
        action="store_false",
        help="не пересобирать индекс автоматически",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: ServiceConfig, args: argparse.Namespace) -> None:
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.index_dir:
        config.index_dir = Path(args.index_dir).resolve()
    if args.dataset:
        config.dataset = args.dataset
    if args.strategy:
        config.strategy = args.strategy
    if args.model:
        config.local_llm["model_path"] = args.model


def _index_exists(config: ServiceConfig) -> bool:
    target = config.index_dir / config.dataset / config.strategy / "faiss.index"
    return target.exists()


def _rebuild(config: ServiceConfig, source: str | None) -> None:
    dataset_source = source or resolve_dataset_source(config.dataset, config.index_dir)
    print(f"Пересборка индекса из {dataset_source} …")
    target = build_index(
        dataset_source,
        dataset=config.dataset,
        strategy=config.strategy,
        index_dir=config.index_dir,
    )
    print(f"Индекс готов: {target}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ServiceConfig.from_file(args.config) if args.config else ServiceConfig.from_file()
    _apply_overrides(config, args)

    if args.rebuild_index:
        _rebuild(config, args.source)
    elif args.rebuild_if_missing and not _index_exists(config):
        print("Индекс не найден — пересобираю.")
        _rebuild(config, args.source)

    app = create_app(config, offline=args.offline)
    engine = app.extensions["engine"]
    mode = "офлайн-стаб" if args.offline else engine.model_label
    print(f"Сервис запущен на http://{config.host}:{config.port} (модель: {mode})")
    try:
        app.run(host=config.host, port=config.port, debug=config.debug, use_reloader=False)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
