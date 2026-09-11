"""CLI-интерфейс: принимает запрос пользователя и выводит ответ агента."""

import json
import sys
import urllib.error

from .agent import Agent
from .config import load_config
from .llm_client import LLMClient


def build_agent() -> Agent:
    return Agent(LLMClient(load_config()))


def main() -> None:
    try:
        agent = build_agent()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}. Проверьте config.json.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Агент запущен. Введите 'exit' или Ctrl+C для выхода.\n")

    while True:
        try:
            user_request = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_request:
            continue
        if user_request.lower() in ("exit", "quit"):
            print("До свидания!")
            break

        try:
            reply = agent.ask(user_request)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue

        print(f"\nАгент: {reply.content}\n")


if __name__ == "__main__":
    main()
