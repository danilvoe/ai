#!/usr/bin/env python3
"""Точка входа: веб-чат с локальной и облачной LLM.

Поднимает Flask-приложение (см. :mod:`agent.web_chat`), которое отправляет
запросы в выбранную модель — локальный ``llama-server`` из Дня 26 или облачный
OpenAI-совместимый API — и отображает ответы в браузере.

RAG-возможность (база рецептов, День 28) включена по умолчанию: при старте
загружается локальный FAISS-индекс, а режим («рецепты» или без RAG) выбирается
пользователем в выпадающем списке веб-интерфейса на каждое сообщение. Если индекс
недоступен, чат работает в обычном режиме, а «рецепты» вернут ошибку.

Помимо локальной модели в интерфейсе можно выбрать **облачную**: она доступна,
если в конфигурации заданы ``api_key`` и ``model`` (любой OpenAI-совместимый API).
Локальный retrieval при этом сохраняется — облачная модель отвечает по той же
базе рецептов. Отключить облако можно флагом ``--no-cloud``.

Запуск::

    python3 web_local.py                 # живой llama-server (+ облако) + RAG-выбор
    python3 web_local.py --offline       # веб-чат на детерминированном стабе
    python3 web_local.py --no-cloud      # только локальная модель
    python3 web_local.py --no-rag        # отключить RAG-возможность
    python3 web_local.py --port 8000     # порт веб-приложения
    python3 web_local.py --llm-port 8090 # порт llama-server
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from agent.config import load_config
from agent.local_llm import LocalLLMError
from agent.optimization import find_variant, parse_variants
from agent.web_chat import WebChatBackend, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Веб-чат с локальной и облачной LLM")
    parser.add_argument("--host", default=None, help="Адрес веб-приложения")
    parser.add_argument("--port", type=int, default=None, help="Порт веб-приложения")
    parser.add_argument("--llm-port", type=int, default=None, help="Порт llama-server")
    parser.add_argument("--model", default=None, help="Путь к GGUF-модели")
    parser.add_argument("--offline", action="store_true", help="Стаб вместо llama.cpp")
    parser.add_argument(
        "--rag", action="store_true", help="(устарело) RAG-возможность включена по умолчанию"
    )
    parser.add_argument("--no-rag", action="store_true", help="Отключить RAG-возможность")
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Отключить облачную модель (только локальная)",
    )
    parser.add_argument("--dataset", default=None, help="Датасет индекса (например, recipt_all)")
    parser.add_argument(
        "--strategy", default=None, choices=["structural", "fixed"], help="Стратегия чанков"
    )
    parser.add_argument("--top-k", type=int, default=None, help="Число источников в контексте")
    parser.add_argument("--index-dir", default=None, help="Каталог FAISS-индексов")
    parser.add_argument(
        "--variant", default=None, help="Вариант локальной модели (квантование), напр. Q2_K"
    )
    parser.add_argument(
        "--preset", default=None, help="Пресет параметров генерации (baseline/optimized)"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Лимит токенов ответа")
    parser.add_argument(
        "--context-window", type=int, default=None, help="Окно контекста (число сообщений)"
    )
    parser.add_argument("--ctx-size", type=int, default=None, help="Размер контекста llama-server (-c)")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер")
    args = parser.parse_args()

    config = load_config()
    section = config.get("web_chat", {}) or {}
    host = args.host or section.get("host", "127.0.0.1")
    port = args.port or int(section.get("port", 8000))

    # День 29: начальные настройки оптимизации из CLI.
    if args.variant:
        variant = find_variant(parse_variants(config), args.variant)
        if variant is None:
            print(f"✗ Неизвестный вариант модели: {args.variant}", file=sys.stderr)
            return 1
        config.setdefault("local_llm", {})["model_path"] = variant.model_path
        config["local_llm"]["model_label"] = variant.label
        if variant.ctx_size and not args.ctx_size:
            config["local_llm"]["ctx_size"] = variant.ctx_size
    if args.ctx_size:
        config.setdefault("local_llm", {})["ctx_size"] = args.ctx_size
    if args.max_tokens is not None:
        config.setdefault("local_llm", {})["max_response_tokens"] = args.max_tokens
    if args.temperature is not None:
        config.setdefault("local_rag", {})["temperature"] = args.temperature
    if args.context_window is not None:
        config.setdefault("web_chat", {})["window_size"] = args.context_window
    if args.preset:
        config.setdefault("optimization", {})["default_preset"] = args.preset

    if args.dataset or args.strategy or args.top_k:
        rag_section = config.setdefault("local_rag", {})
        if args.dataset:
            rag_section["dataset"] = args.dataset
        if args.strategy:
            rag_section["strategy"] = args.strategy
        if args.top_k:
            rag_section["top_k"] = args.top_k
    if args.index_dir:
        config.setdefault("indexing", {})["index_dir"] = args.index_dir

    backend = WebChatBackend(
        config,
        offline=args.offline,
        model_path=args.model,
        port=args.llm_port,
        top_k=args.top_k,
        enable_rag=not args.no_rag,
        cloud=not args.no_cloud,
    )

    print("=" * 78)
    print("🌐 ВЕБ-ЧАТ: ЛОКАЛЬНАЯ И ОБЛАЧНАЯ МОДЕЛЬ")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой llama.cpp'}")
    if not args.offline:
        print(f"Локальная модель: {backend.local.model_file}")
        print("Запуск llama-server...")

    try:
        backend.start()
    except (LocalLLMError, OSError) as exc:
        print(f"✗ Ошибка запуска модели: {exc}", file=sys.stderr)
        return 1

    app = create_app(backend)
    url = f"http://{host}:{port}/"
    print(f"✓ Локальная модель готова: {backend.endpoint}")
    if backend.cloud_enabled:
        print(f"✓ Облачная модель доступна: {backend.cloud_model_label or '—'}")
    else:
        print("⚠ Облачная модель недоступна (нет api_key/model или выключена)")
    if backend.rag_enabled:
        status = backend.status()
        print(
            f"✓ RAG: база '{status.get('dataset')}', {status.get('vectors')} векторов, "
            f"top-{status.get('top_k')} (режим и модель выбираются в UI)"
        )
    else:
        print(f"⚠ RAG недоступен: {backend.rag_error or 'возможность отключена'}")
    gen = backend.generation_params.to_dict()
    print(
        f"✓ Оптимизация: вариант {backend.local_variant_label or '—'}, "
        f"промпт {backend.prompt_template_label}, ctx {backend.ctx_size}, "
        f"temperature {gen['temperature']}, max_tokens {gen['max_tokens']} "
        f"(настраивается в UI)"
    )
    print(f"✓ Веб-чат: {url}")
    print("Остановка — Ctrl+C.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — браузер не критичен
            pass

    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
