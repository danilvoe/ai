"""День 33: интерактивный CLI ассистента поддержки.

Мини-сервис поддержки: ассистент отвечает на вопросы о продукте по FAQ (RAG) и
учитывает данные пользователя и тикета из JSON-CRM через MCP
(``mcp/support_server.py``). Команды дают прямой доступ к MCP-инструментам.

Запуск::

    python3 -m agent.support_cli
    python3 chat_support.py
    python3 chat_support.py --offline --question "Почему не работает авторизация?" --ticket T-1024
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config
from .support_assistant import SupportAnswer, build_support_assistant
from .support_faq import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATASET,
    DEFAULT_FAQ_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_faq_index,
    load_faq_retriever,
)

HELP_TEXT = """
Доступные команды:
  /ticket <id>     — загрузить тикет и сделать его контекстом (MCP get_ticket_context)
  /user <id>       — загрузить профиль пользователя (MCP get_user)
  /tickets [uid]   — список тикетов (MCP list_tickets)
  /sources         — источники FAQ последнего ответа
  /new             — сбросить историю диалога и текущий тикет
  /reindex         — пересобрать индекс FAQ
  /help            — показать эту справку
  /exit или quit   — завершить

Любой обычный вопрос (без «/») уходит ассистенту поддержки.

Примеры вопросов:
  • Почему не работает авторизация?
  • Как пересобрать индекс рецептов?
  • Модель отвечает очень долго, что делать?
  • Где хранятся данные и как сделать бэкап?
"""

EXAMPLE_TICKETS = ("T-1024", "T-1025", "T-1029")


def _llm_ready(config: dict) -> bool:
    """Есть ли в конфиге (или окружении) всё для живого LLM."""
    base_url = config.get("base_url") or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = config.get("model") or os.environ.get("LLM_MODEL")
    return bool(base_url and api_key and model)


def print_banner(assistant) -> None:
    print("=" * 78)
    print("🎧 АССИСТЕНТ ПОДДЕРЖКИ (ДЕНЬ 33): FAQ (RAG) + ТИКЕТЫ ЧЕРЕЗ MCP")
    print("=" * 78)
    print(f"Индекс FAQ: {assistant.total_vectors} чанков")
    print("Команды: /ticket <id>, /user <id>, /tickets, /sources, /new, /help, /exit")
    print(f"Пример тикета: {EXAMPLE_TICKETS[0]}. Введите вопрос или команду.\n")


def _print_faq_sources(sources) -> None:
    if not sources:
        print("📚 Источников FAQ нет (ответа в базе не найдено).")
        return
    print("📚 Источники FAQ:")
    for idx, src in enumerate(sources, start=1):
        print(f"  [{idx}] [{src.recipe_id}] «{src.title}» — {src.section} (сходство {src.score:.3f})")


def _print_answer(answer: SupportAnswer) -> None:
    if answer.error:
        print(f"\n⚠️ Ошибка: {answer.error}\n", file=sys.stderr)
        return
    ticket = f" | тикет: {answer.ticket_id}" if answer.ticket_id else ""
    escalate = " | ⚠️ эскалация в поддержку" if answer.escalate else ""
    print(f"\nАссистент [{answer.answer_kind}{ticket}{escalate}]:\n{answer.content}\n")
    _print_faq_sources(answer.faq_sources)
    print()


def _print_ticket(data: dict) -> None:
    if data.get("error"):
        print(f"⚠️ Ошибка: {data['error']}\n")
        return
    print(f"\n🎫 Тикет {data.get('ticket_id') or data.get('id')}: «{data.get('subject', '')}»")
    print(
        f"   Статус: {data.get('status')}; категория: {data.get('category')}; "
        f"приоритет: {data.get('priority')}; версия: {data.get('version')}"
    )
    user = data.get("user") or {}
    if user:
        print(
            f"   Пользователь: {user.get('name', '')} ({user.get('username', '')}), "
            f"план: {user.get('plan')}, статус: {user.get('status')}"
        )
    messages = data.get("messages") or []
    if messages:
        print("   Переписка:")
        for message in messages:
            print(
                f"     [{message.get('created_at', '')}] {message.get('author')} "
                f"({message.get('role')}): {message.get('text')}"
            )
    print()


def _print_user(data: dict) -> None:
    if data.get("error"):
        print(f"⚠️ Ошибка: {data['error']}\n")
        return
    print(
        f"\n👤 {data.get('name', '')} ({data.get('username', '')}, {data.get('email', '')})"
    )
    print(f"   План: {data.get('plan')}; версия: {data.get('version')}; статус: {data.get('status')}\n")


def _print_tickets(data: dict) -> None:
    if data.get("error"):
        print(f"⚠️ Ошибка: {data['error']}\n")
        return
    tickets = data.get("tickets") or []
    print(f"\n🎫 Тикеты (всего {data.get('total', 0)}):")
    for item in tickets:
        print(f"   {item.get('id')}: «{item.get('subject')}» — {item.get('status')}")
    print()


def handle_command(raw: str, assistant, state: dict) -> bool:
    """Обрабатывает slash-команду. True — ввод был командой."""
    text = raw.strip()
    if not text.startswith("/"):
        return False

    parts = text[1:].split(None, 1)
    action = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if action in ("help", "h", "?"):
        if arg:
            answer = assistant.ask(arg, ticket_id=state.get("ticket_id"), history=state.get("history"))
            state["last_sources"] = answer.faq_sources
            _print_answer(answer)
        else:
            print(HELP_TEXT)
        return True

    if action == "ticket":
        if not arg:
            print(f"ℹ️ Укажите id тикета, например /ticket {EXAMPLE_TICKETS[0]}.\n")
            return True
        data = assistant.get_ticket(arg)
        _print_ticket(data)
        if not data.get("error"):
            state["ticket_id"] = data.get("ticket_id") or arg
            state["user_id"] = data.get("user_id") or state.get("user_id")
        return True

    if action == "user":
        if not arg.isdigit():
            print("ℹ️ Укажите числовой id пользователя, например /user 1.\n")
            return True
        data = assistant.get_user(int(arg))
        _print_user(data)
        if not data.get("error"):
            state["user_id"] = int(arg)
        return True

    if action == "tickets":
        user_id = int(arg) if arg.isdigit() else state.get("user_id")
        _print_tickets(assistant.list_tickets(user_id=user_id))
        return True

    if action == "sources":
        sources = state.get("last_sources") or []
        if not sources:
            print("ℹ️ В последнем ответе источников FAQ не было.\n")
        else:
            print(f"\n📚 Источники последнего ответа ({len(sources)}):")
            for idx, src in enumerate(sources, start=1):
                print(f"  [{idx}] [{src.recipe_id}] «{src.title}» — {src.section} (сходство {src.score:.3f})")
            print()
        return True

    if action in ("new", "clear", "reset"):
        state["history"] = []
        state["last_sources"] = []
        state["ticket_id"] = None
        print("✓ История диалога и текущий тикет сброшены.\n")
        return True

    if action in ("reindex", "reload"):
        build_faq_index(
            faq_path=state["faq_path"],
            index_dir=state["index_dir"],
            dataset=state["dataset"],
            strategy=state["strategy"],
            chunk_size=state["chunk_size"],
            overlap=state["overlap"],
        )
        assistant.retriever = load_faq_retriever(
            index_dir=state["index_dir"],
            dataset=state["dataset"],
            strategy=state["strategy"],
            top_k=assistant.top_k,
            min_score=assistant.min_score,
        )
        print(f"✓ Индекс FAQ пересобран: {assistant.total_vectors} чанков.\n")
        return True

    print(f"⚠️ Неизвестная команда: /{action}. Введите /help.\n")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ассистент поддержки: RAG по FAQ + тикеты/пользователи через MCP"
    )
    parser.add_argument("--offline", "--no-llm", dest="offline", action="store_true",
                        help="офлайн-режим без обращения к LLM API (детерминированный стаб)")
    parser.add_argument("--question", "-q", default=None, help="задать один вопрос и выйти")
    parser.add_argument("--ticket", default=None, help="id тикета для контекста (например T-1024)")
    parser.add_argument("--user", type=int, default=None, help="id пользователя для контекста")
    parser.add_argument("--top-k", type=int, default=5, help="сколько карточек FAQ брать")
    parser.add_argument("--min-score", type=float, default=0.0, help="минимальное сходство чанка")
    parser.add_argument("--relevance-threshold", type=float, default=0.30,
                        help="порог релевантности FAQ (ниже — эскалация)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--faq-path", default=DEFAULT_FAQ_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--no-mcp", action="store_true", help="не подключать MCP-сервер поддержки")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = load_config()
    except (OSError, ValueError) as exc:  # noqa: BLE001 — конфиг опционален
        print(f"ℹ️  Конфиг не загружен ({type(exc).__name__}); офлайн-режим.", file=sys.stderr)
        config = {}

    if args.offline or not _llm_ready(config):
        offline = True
        if not args.offline:
            print("⚠️  LLM-ключ не найден — использую офлайн-стаб.", file=sys.stderr)
    else:
        offline = False

    mode = "офлайн-стаб" if offline else f"живой LLM ({config.get('model')})"
    print(f"Режим: {mode}")

    print("Инициализация индекса FAQ и MCP-сервера поддержки...")
    try:
        assistant = build_support_assistant(
            config=config,
            offline=offline,
            faq_path=args.faq_path,
            index_dir=args.index_dir,
            dataset=args.dataset,
            strategy=args.strategy,
            top_k=args.top_k,
            min_score=args.min_score,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            relevance_threshold=args.relevance_threshold,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            with_support_mcp=not args.no_mcp,
        )
    except Exception as exc:  # noqa: BLE001 — понятная ошибка инициализации
        print(f"✗ Ошибка инициализации: {exc}", file=sys.stderr)
        return 1

    state = {
        "history": [],
        "last_sources": [],
        "ticket_id": args.ticket,
        "user_id": args.user,
        "faq_path": args.faq_path,
        "index_dir": args.index_dir,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }

    if args.ticket:
        _print_ticket(assistant.get_ticket(args.ticket))

    if args.question:
        answer = assistant.ask(
            args.question, ticket_id=state["ticket_id"], user_id=state["user_id"]
        )
        _print_answer(answer)
        return 0 if not answer.error else 1

    print_banner(assistant)

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit", "/q"):
            print("До свидания!")
            break

        if handle_command(user_input, assistant, state):
            continue

        answer = assistant.ask(
            user_input,
            ticket_id=state.get("ticket_id"),
            user_id=state.get("user_id"),
            history=state["history"],
        )
        state["last_sources"] = answer.faq_sources
        _print_answer(answer)
        if not answer.error:
            state["history"].append({"role": "user", "content": user_input})
            state["history"].append({"role": "assistant", "content": answer.content})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
