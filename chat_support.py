#!/usr/bin/env python3
"""Точка входа: ассистент поддержки пользователей (День 33).

Ассистент отвечает на вопросы о продукте по FAQ (RAG) и учитывает контекст
пользователя и тикета из JSON-CRM через MCP-сервер ``mcp/support_server.py``.

Запуск::

    python3 chat_support.py
    python3 chat_support.py --offline --question "Почему не работает авторизация?" --ticket T-1024
    python3 chat_support.py --offline --ticket T-1024
"""

import sys

from agent.support_cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nАссистент поддержки завершён.")
        sys.exit(0)
