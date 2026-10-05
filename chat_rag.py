#!/usr/bin/env python3
"""Точка входа для запуска RAG-чата с памятью задачи.

Запуск:
    python3 chat_rag.py
    python3 chat_rag.py --new
    python3 chat_rag.py --no-llm
"""

import sys
from agent.rag_chat_cli import main

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nЧат завершён.")
        sys.exit(0)
