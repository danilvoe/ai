#!/usr/bin/env python3
"""Точка входа: веб-чат с локальной LLM (без облачных моделей).

Поднимает Flask-приложение (см. :mod:`agent.web_chat`), которое отправляет
запросы в локальный ``llama-server`` из Дня 26 и отображает ответы в браузере.

Запуск::

    python3 web_local.py                 # живой локальный llama-server
    python3 web_local.py --offline       # веб-чат на детерминированном стабе
    python3 web_local.py --port 8000     # порт веб-приложения
    python3 web_local.py --llm-port 8090 # порт llama-server
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from agent.config import load_config
from agent.local_llm import LocalLLMError
from agent.web_chat import WebChatBackend, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Веб-чат с локальной LLM")
    parser.add_argument("--host", default=None, help="Адрес веб-приложения")
    parser.add_argument("--port", type=int, default=None, help="Порт веб-приложения")
    parser.add_argument("--llm-port", type=int, default=None, help="Порт llama-server")
    parser.add_argument("--model", default=None, help="Путь к GGUF-модели")
    parser.add_argument("--offline", action="store_true", help="Стаб вместо llama.cpp")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер")
    args = parser.parse_args()

    config = load_config()
    section = config.get("web_chat", {}) or {}
    host = args.host or section.get("host", "127.0.0.1")
    port = args.port or int(section.get("port", 8000))

    backend = WebChatBackend(
        config,
        offline=args.offline,
        model_path=args.model,
        port=args.llm_port,
    )

    print("=" * 78)
    print("🌐 ВЕБ-ЧАТ С ЛОКАЛЬНОЙ LLM")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой llama.cpp'}")
    if not args.offline:
        print(f"Модель: {backend.local.model_file}")
        print("Запуск llama-server...")

    try:
        backend.start()
    except (LocalLLMError, OSError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    app = create_app(backend)
    url = f"http://{host}:{port}/"
    print(f"✓ Модель готова: {backend.endpoint}")
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
