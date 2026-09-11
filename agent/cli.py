"""CLI-интерфейс: принимает запрос пользователя и выводит ответ агента."""

import json
import sys
import urllib.error

from .agent import Agent
from .compression import CompressionConfig
from .config import load_config
from .conversation import SessionStore
from .llm_client import LLMClient
from .tokens import ContextOverflowError


def _compression_from_config(config: dict) -> CompressionConfig:
    """Читает настройки сжатия истории из конфигурации."""
    section = config.get("compression", {}) or {}
    return CompressionConfig(
        keep_recent=section.get("keep_recent", 10),
        summarize_every=section.get("summarize_every", 10),
        max_tokens=section.get("max_tokens"),
    )


def build_agent(conversation) -> Agent:
    config = load_config()
    client = LLMClient(config)
    return Agent(
        client,
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=_compression_from_config(config),
    )


def print_usage(agent: Agent) -> None:
    """Печатает токены и стоимость последнего хода и всей сессии."""
    last = agent.usage.turns[-1]
    print(
        f"Токены хода: {last.total_tokens} "
        f"(история {last.history_tokens}, запрос {last.request_tokens}, "
        f"ответ {last.response_tokens or 0})"
    )
    print(
        f"Токены за всю сессию: {agent.usage.total_tokens} "
        f"(запросы {agent.usage.total_request_tokens}, ответы {agent.usage.total_response_tokens})"
    )
    if agent.usage.total_cost is None:
        print("Стоимость сессии: н/д (укажите тарифы в config.json)")
    else:
        print(f"Стоимость сессии: ${agent.usage.total_cost:.6f}")
    print()


def print_history(agent: Agent) -> None:
    """Выводит полную историю диалога, помечая сообщения, свёрнутые в summary."""
    conv = agent.conversation
    if not conv.messages:
        print("История пуста.\n")
        return
    print(f"История диалога ({len(conv.messages)} сообщений):")
    for index, message in enumerate(conv.messages):
        is_summarized = index < conv.summarized
        marker = " (в summary)" if is_summarized else ""
        who = "Вы" if message.get("role") == "user" else "Агент"
        print(f"[{index + 1}] {who}{marker}: {message.get('content', '')}")
    if conv.summary:
        print(f"\nКраткое содержание свёрнутой части:\n{conv.summary}")
    print()


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
        print(f"Восстановлен контекст: {len(agent.history)} сообщений.")
    if agent.compression.enabled:
        conv = agent.conversation
        if conv.summary:
            print(
                f"Сжатие истории: включено ({len(conv.messages)} сообщений, "
                f"из них {conv.summarized} свёрнуты в summary)."
            )
        else:
            print("Сжатие истории: включено.")
    print()

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
        if user_request.lower() in ("history", "история", "показать историю"):
            print_history(agent)
            continue

        try:
            reply = agent.ask(user_request)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        except ContextOverflowError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue

        print(f"\nАгент: {reply.content}\n")
        print_usage(agent)


if __name__ == "__main__":
    main()
