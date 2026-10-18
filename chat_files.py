#!/usr/bin/env python3
"""Точка входа: ассистент для работы с файлами проекта (День 34).

Ассистент получает цель и сам выполняет операции с файлами через MCP-сервер
``mcp/fs_server.py``: читает файлы, ищет по нескольким файлам, анализирует
содержимое и создаёт/изменяет файлы. По умолчанию изменения только выводятся как
diff (реальная запись — с флагом ``--apply``).

Запуск::

    python3 chat_files.py
    python3 chat_files.py --offline --goal "найди, где используется McpToolRuntime"
    python3 chat_files.py --offline --apply --goal "сгенерируй каталог MCP-серверов"
"""

import sys

from agent.file_cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nАссистент файловых операций завершён.")
        sys.exit(0)
