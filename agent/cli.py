"""CLI-интерфейс: принимает запрос пользователя и выводит ответ агента."""

import json
import sys
import urllib.error

from .agent import Agent
from .config import load_config
from .conversation import SessionStore
from .llm_client import LLMClient


def build_agent(conversation) -> Agent:
    return Agent(LLMClient(load_config()), conversation)


def pick_session(store: SessionStore) -> Agent:
    """Показывает список сессий и возвращает агента для выбранной или новой."""
    sessions = store.list_sessions()
    print("Доступные сессии:")
    for index, session_id in enumerate(sessions, start=1):
        print(f"  {index}. {session_id}")

    choice = input(
        f"Выберите сессию (1-{len(sessions)}) или 'new' для новой "
        f"['new']: " if sessions else "Сессий нет. Введите 'new' для новой ['new']: "
    ).strip()

    if choice.lower() in ("new", "") or not sessions:
        conversation = store.create()
        print(f"Новая сессия: {conversation.session_id}\n")
    else:
        try:
            conversation = store.get(sessions[int(choice) - 1])
        except (ValueError, IndexError):
            print(f"Неверный выбор, создаю новую сессию.")
            conversation = store.create()
        print(f"Открыта сессия: {conversation.session_id}\n")

    return build_agent(conversation)


def main() -> None:
    store = SessionStore()
    try:
        agent = pick_session(store)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}. Проверьте config.json.", file=sys.stderr)
        raise SystemExit(1) from exc

    if agent.history:
        print(f"Восстановлен контекст: {len(agent.history)} сообщений.\n")
    print("Агент запущен. Введите 'exit' или Ctrl+C для выхода.\n")

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
        if user_request.lower() in ("clear", "очистить"):
            agent.conversation.clear()
            print("История диалога очищена.\n")
            continue

        try:
            reply = agent.ask(user_request)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue

        print(f"\nАгент: {reply.content}\n")


if __name__ == "__main__":
    main()
