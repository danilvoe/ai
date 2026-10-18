#!/usr/bin/env python3
"""Точка входа: автоматическое ревью PR (День 32).

Ассистент получает diff и изменённые файлы PR через MCP, ищет релевантную
документацию и код в локальных RAG-индексах и выдаёт текст ревью: потенциальные
баги, архитектурные проблемы и рекомендации. Используется в GitHub Action и
локально.

Запуск::

    python3 review_pr.py --event "$GITHUB_EVENT_PATH" --post-comment
    python3 review_pr.py --base HEAD~1 --head HEAD --offline
    python3 review_pr.py --base day31 --head day32 --offline --output review.md
"""

import sys

from agent.review_cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nРевью прервано.")
        sys.exit(0)
