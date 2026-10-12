#!/usr/bin/env python3
"""Точка входа: локальный чат с LLM через llama.cpp (без облачного API).

Поднимает ``llama-server`` из сборки llama.cpp и направляет на него обычного
агента (:class:`agent.agent.Agent`) через OpenAI-совместимый endpoint ``/v1``.

Запуск::

    python3 chat_local.py
    python3 chat_local.py --prompt "Сколько будет 2+2?"
    python3 chat_local.py --port 8090
"""

from __future__ import annotations

import argparse
import sys

from agent.agent import Agent
from agent.config import load_config
from agent.context import SlidingWindowStrategy
from agent.conversation import SessionStore
from agent.local_llm import LocalLLMConfig, LocalLLMError, LocalLLMServer
from agent.tokens import ContextOverflowError


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный чат с LLM через llama.cpp")
    parser.add_argument("--prompt", default=None, help="Один запрос и выход (без REPL)")
    parser.add_argument("--port", type=int, default=None, help="Порт llama-server")
    parser.add_argument("--model", default=None, help="Путь к GGUF-модели")
    args = parser.parse_args()

    config = load_config()
    local = LocalLLMConfig.from_config(config)
    if args.port:
        local.port = args.port
    if args.model:
        local.model_path = args.model

    print("=" * 78)
    print("🔌 ЛОКАЛЬНЫЙ ЧАТ С LLM (llama.cpp)")
    print("=" * 78)
    if not local.enabled:
        print("✗ Локальная LLM отключена (local_llm.enabled = false).", file=sys.stderr)
        return 1
    print(f"Модель: {local.model_file}")
    print(f"Endpoint: http://{local.host}:{local.port}/v1")
    print("Запуск llama-server...")

    server = LocalLLMServer(local)
    try:
        server.start()
    except (LocalLLMError, OSError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    try:
        print(f"✓ Сервер готов (модели: {', '.join(server.list_models())})\n")
        conversation = SessionStore().create()
        agent = Agent(
            server.build_client(),
            conversation,
            # Клиентский лимит под реальный контекст llama-server: переполнение
            # ловится заранее (ContextOverflowError), а не как HTTP 400.
            max_context_tokens=server.max_context_tokens,
            # Окно не даёт истории расти бесконечно в длинном диалоге.
            context_strategy=SlidingWindowStrategy(size=8),
        )

        if args.prompt:
            reply = agent.ask(args.prompt)
            print(f"Агент: {reply.content}")
            return 0

        print("Введите запрос, '/clear' для сброса истории или 'exit' для выхода.\n")
        while True:
            try:
                user_input = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nДо свидания!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("До свидания!")
                break
            if user_input.lower() in ("/clear", "/reset"):
                conversation.clear()
                print("История диалога очищена.\n")
                continue
            try:
                reply = agent.ask(user_input)
            except ContextOverflowError:
                print(
                    "⚠️ Контекст переполнен. Сбросьте историю командой /clear "
                    "или увеличьте local_llm.ctx_size в конфиге.\n",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — наглядная ошибка локального запроса
                print(f"⚠️ Ошибка: {exc}", file=sys.stderr)
                continue
            print(f"\nАгент: {reply.content}\n")
        return 0
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
