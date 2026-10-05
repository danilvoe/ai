"""Интерактивный терминальный CLI-чат с RAG и памятью задачи (Task State).

Поддерживает:
- Сохранение истории и состояния задачи в файл сессии history/<session_id>.json;
- Поиск в RAG на каждый вопрос и обязательный вывод источников (chunk_id + URL);
- Удержание цели диалога, зафиксированных ограничений и терминов;
- Команды управления состоянием:
  /state, /goal, /constraint, /clarify, /term, /sources, /history, /clear, /exit.

Запуск:
    python3 -m agent.rag_chat_cli
    python3 chat_rag.py
    python3 -m agent.rag_chat_cli --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .conversation import SessionStore
from .rag_chat import RagChatSession, build_rag_chat_agent


HELP_TEXT = """
Доступные команды управления чатом и памятью задачи:
  /state                     — показать текущую карточку задачи (цель, ограничения, термины)
  /goal <текст>             — вручную установить или обновить цель диалога
  /constraint <текст>       — зафиксировать новое ограничение (например, «без томатов»)
  /clarify <ключ: значение> — добавить параметр уточнения (например, «гриль: котел Kettle»)
  /term <термин: значение>  — зафиксировать кулинарный термин или технологию
  /sources                  — показать подробный список и фрагменты последних источников RAG
  /history                  — показать историю сообщений текущей сессии
  /clear или /reset         — сбросить историю сообщений и состояние задачи
  /help                     — показать эту справку
  /exit или quit            — завершить диалог
"""


def print_banner(session: RagChatSession) -> None:
    """Выводит стартовую заставку и состояние сессии."""
    print("=" * 78)
    print("🤖 RAG-ЧАТ С ПАМЯТЬЮ ЗАДАЧИ И ИСТОЧНИКАМИ (GRILL & BBQ ASSISTANT)")
    print(f"Сессия: {session.session_id} (файл: history/{session.session_id}.json)")
    print("=" * 78)
    print(session.state.render_banner())
    print("Введите ваш вопрос, команду (например, /state, /help) или 'exit' для выхода.\n")


def handle_command(cmd: str, session: RagChatSession) -> bool:
    """Обрабатывает slash-команды CLI.

    Возвращает True, если ввод был командой (не передавать в RAG), иначе False.
    """
    raw = cmd.strip()
    if not raw.startswith("/"):
        return False

    parts = raw[1:].split(None, 1)
    action = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if action in ("help", "h", "?"):
        print(HELP_TEXT)
        return True

    if action == "state":
        print("\n" + session.state.render_banner() + "\n")
        return True

    if action == "goal":
        if not arg:
            print("⚠️ Укажите цель: /goal <описание цели>")
        else:
            session.state.set_goal(arg)
            session.save()
            print(f"✓ Цель обновлена: «{arg}»\n")
            print(session.state.render_banner() + "\n")
        return True

    if action in ("constraint", "limit"):
        if not arg:
            print("⚠️ Укажите ограничение: /constraint <ограничение>")
        else:
            session.state.add_constraint(arg)
            session.save()
            print(f"✓ Ограничение зафиксировано: «{arg}»\n")
            print(session.state.render_banner() + "\n")
        return True

    if action in ("clarify", "set"):
        if ":" not in arg:
            print("⚠️ Формат: /clarify <ключ: значение> (например, /clarify гриль: Kettle 57 см)")
        else:
            k, v = arg.split(":", 1)
            session.state.add_clarification(k.strip(), v.strip())
            session.save()
            print(f"✓ Параметр зафиксирован: [{k.strip()}] = {v.strip()}\n")
            print(session.state.render_banner() + "\n")
        return True

    if action in ("term", "concept"):
        if ":" not in arg and "—" not in arg and "-" not in arg:
            print("⚠️ Формат: /term <термин: определение> (например, /term Hot & Fast: копчение при 135-150°C)")
        else:
            delim = ":" if ":" in arg else ("—" if "—" in arg else "-")
            t, d = arg.split(delim, 1)
            session.state.add_term(t.strip(), d.strip())
            session.save()
            print(f"✓ Термин зафиксирован: [{t.strip()}] = {d.strip()}\n")
            print(session.state.render_banner() + "\n")
        return True

    if action == "sources":
        if not session.last_sources:
            print("ℹ️ В последнем ответе не было найдено источников или поиск ещё не выполнялся.\n")
        else:
            print(f"\n📚 ПОСЛЕДНИЕ ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ ({len(session.last_sources)}):")
            for idx, s in enumerate(session.last_sources, 1):
                print(s.format_line(idx))
                excerpt = " ".join(s.text.split())[:200]
                print(f"     Цитата: «{excerpt}…»\n")
        return True

    if action == "history":
        messages = session.messages
        if not messages:
            print("ℹ️ История сообщений пуста.\n")
        else:
            print(f"\n💬 ИСТОРИЯ ДИАЛОГА ({len(messages)} сообщений):")
            for m in messages:
                role = "👤 Пользователь" if m["role"] == "user" else "🤖 Ассистент"
                print(f"{role}:\n{m['content']}\n{'-' * 40}")
        return True

    if action in ("clear", "reset"):
        session.clear()
        print("✓ Сессия очищена: история и состояние задачи сброшены.\n")
        print(session.state.render_banner() + "\n")
        return True

    print(f"⚠️ Неизвестная команда: /{action}. Введите /help для справки.\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI RAG-чат с памятью задачи и источниками")
    parser.add_argument("--session", type=str, default=None, help="Идентификатор существующей сессии")
    parser.add_argument("--new", action="store_true", help="Всегда создавать новую сессию")
    parser.add_argument("--no-llm", action="store_true", help="Офлайн-режим без обращения к LLM API")
    parser.add_argument("--top-k", type=int, default=4, help="Количество чанков в поиске (по умолчанию: 4)")
    args = parser.parse_args()

    config = load_config()
    history_dir = Path(__file__).resolve().parent.parent / "history"
    store = SessionStore(history_dir=history_dir)

    if args.session:
        conversation = store.get(args.session)
    elif args.new:
        conversation = store.create()
    else:
        # Открываем последнюю или создаём новую
        sessions = store.list_sessions()
        if sessions:
            conversation = store.get(sessions[-1])
        else:
            conversation = store.create()

    session = RagChatSession(conversation)

    print("Инициализация RAG-базы и модели...")
    try:
        agent = build_rag_chat_agent(
            config=config,
            top_k=args.top_k,
            no_llm=args.no_llm,
        )
    except Exception as exc:
        print(f"✗ Ошибка инициализации RAG-агента: {exc}", file=sys.stderr)
        sys.exit(1)

    print_banner(session)

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nСессия сохранена. До свидания!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Сессия сохранена. До свидания!")
            break

        # Проверка и обработка slash-команд
        if handle_command(user_input, session):
            continue

        # Запрос к RAG-агенту
        try:
            resp = agent.ask(user_input, session)
            print(f"\nАссистент:\n{resp.full_reply_text}\n")
        except Exception as exc:
            print(f"\n⚠️ Ошибка обработки: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
