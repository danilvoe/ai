#!/usr/bin/env python3
"""Точка входа: ассистент разработчика (День 31).

Ассистент отвечает на вопросы о проекте по документации (README + ``docs/``) и
контексту из git через MCP-сервер ``mcp/project_server.py``. Основная команда —
``/help``.

Запуск::

    python3 chat_dev.py
    python3 chat_dev.py --no-llm
    python3 chat_dev.py --question "где находится MCP-сервер проекта?"
"""

import sys

from agent.dev_assistant_cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nАссистент завершён.")
        sys.exit(0)
