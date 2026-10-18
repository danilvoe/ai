"""День 31: интерактивный CLI ассистента разработчика.

Ассистент отвечает на вопросы о проекте по документации (README + ``docs/``) и
живому контексту из git через MCP (``mcp/project_server.py``). Команда ``/help``
— основной способ задать вопрос о проекте; остальные команды дают прямой доступ
к MCP-инструментам.

Запуск::

    python3 -m agent.dev_assistant_cli
    python3 chat_dev.py
    python3 chat_dev.py --no-llm
    python3 chat_dev.py --question "где лежит MCP-сервер проекта?"
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .dev_assistant import DevAssistant, build_dev_assistant
from .project_docs import (
    DEFAULT_DATASET,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_documents_index,
    load_project_retriever,
)

HELP_TEXT = """
Доступные команды:
  /help <вопрос>   — ответить на вопрос о проекте (документация + git-контекст)
  /help            — показать эту справку и примеры вопросов
  /branch          — текущая git-ветка (MCP get_current_branch)
  /files [префикс] — список файлов проекта (MCP list_project_files)
  /diff [путь]     — diff рабочего дерева (MCP get_diff)
  /sources         — источники последнего ответа (файл + раздел)
  /reindex         — пересобрать индекс документации
  /clear           — сбросить историю текущей сессии
  /exit или quit   — завершить

Любой обычный вопрос (без «/») тоже уходит ассистенту как вопрос о проекте.

Примеры вопросов:
  • как устроена подсистема RAG и где хранится индекс?
  • какие MCP-серверы есть в проекте и какие у них инструменты?
  • где находится MCP-сервер проекта и какие у него инструменты?
  • как запустить офлайн-проверки?
  • что изменилось в рабочем дереве? (запросит diff)
"""

EXAMPLE_QUESTIONS = (
    "Как устроена подсистема RAG и где хранится индекс?",
    "Какие MCP-серверы есть в проекте и какие у них инструменты?",
    "Где находится MCP-сервер проекта и какие у него инструменты?",
    "Как запустить офлайн-проверки без затрат на API?",
)


def print_banner(assistant: DevAssistant) -> None:
    print("=" * 78)
    print("🛠️  АССИСТЕНТ РАЗРАБОТЧИКА (ДЕНЬ 31): ДОКУМЕНТАЦИЯ + GIT-КОНТЕКСТ")
    print("=" * 78)
    print(f"Индекс документации: {assistant.total_vectors} чанков")
    branch = assistant.get_branch()
    if branch.get("branch"):
        print(f"Текущая ветка: {branch['branch']} (коммит {branch.get('commit', '?')})")
    else:
        print(f"Git-контекст недоступен: {branch.get('error', 'неизвестно')}")
    print("Введите вопрос или команду (/help). 'exit' — выход.\n")


def _print_answer(answer) -> None:
    if answer.error:
        print(f"\n⚠️ Ошибка: {answer.error}\n", file=sys.stderr)
        return
    print(f"\nАссистент:\n{answer.content}\n")
    if answer.sources:
        print("📚 Источники:")
        for idx, src in enumerate(answer.sources, start=1):
            print(f"  [{idx}] {src.url} — «{src.section}» (сходство {src.score:.3f})")
        print()


def _print_files(data: dict) -> None:
    if data.get("error"):
        print(f"⚠️ Ошибка: {data['error']}\n")
        return
    print(f"\nФайлы (всего {data.get('total', 0)}):")
    for name in data.get("files", []):
        print(f"  {name}")
    print()


def _print_diff(data: dict) -> None:
    if data.get("error"):
        print(f"⚠️ Ошибка: {data['error']}\n")
        return
    if not data.get("diff"):
        print("ℹ️ Изменений нет.\n")
        return
    print(f"\nDiff ({data.get('base')}, {data.get('chars', 0)} символов):\n")
    print(data["diff"])
    if data.get("truncated"):
        print("\n… (diff усечён)")
    print()


def handle_command(raw: str, assistant: DevAssistant, state: dict) -> bool:
    """Обрабатывает slash-команду. True — ввод был командой."""
    text = raw.strip()
    if not text.startswith("/"):
        return False

    parts = text[1:].split(None, 1)
    action = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if action in ("help", "h", "?"):
        if arg:
            answer = assistant.answer_help(arg, history=state.get("history"))
            state["last_sources"] = answer.sources
            _print_answer(answer)
        else:
            print(HELP_TEXT)
        return True

    if action == "branch":
        data = assistant.get_branch()
        if data.get("error"):
            print(f"⚠️ Ошибка: {data['error']}\n")
        else:
            print(f"\nТекущая ветка: {data.get('branch')} (коммит {data.get('commit')})\n")
        return True

    if action == "files":
        _print_files(assistant.list_files(path=arg, limit=100))
        return True

    if action in ("diff", "changes"):
        _print_diff(assistant.get_diff(path=arg, max_chars=4000))
        return True

    if action == "sources":
        sources = state.get("last_sources") or []
        if not sources:
            print("ℹ️ В последнем ответе источников не было.\n")
        else:
            print(f"\n📚 Источники последнего ответа ({len(sources)}):")
            for idx, src in enumerate(sources, start=1):
                print(f"  [{idx}] {src.url} — «{src.section}» (сходство {src.score:.3f})")
            print()
        return True

    if action in ("reindex", "reload"):
        build_documents_index(
            index_dir=state["index_dir"],
            dataset=state["dataset"],
            strategy=state["strategy"],
            chunk_size=state["chunk_size"],
            overlap=state["overlap"],
        )
        assistant.retriever = load_project_retriever(
            index_dir=state["index_dir"],
            dataset=state["dataset"],
            strategy=state["strategy"],
            top_k=assistant.top_k,
            min_score=assistant.min_score,
        )
        print(f"✓ Индекс пересобран: {assistant.total_vectors} чанков.\n")
        return True

    if action in ("clear", "reset"):
        state["history"] = []
        state["last_sources"] = []
        print("✓ История сессии очищена.\n")
        return True

    print(f"⚠️ Неизвестная команда: /{action}. Введите /help.\n")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ассистент разработчика: RAG по документации + git-контекст через MCP"
    )
    parser.add_argument("--no-llm", "--offline", dest="offline", action="store_true",
                        help="офлайн-режим без обращения к LLM API (детерминированный стаб)")
    parser.add_argument("--question", "-q", default=None, help="задать один вопрос и выйти")
    parser.add_argument("--top-k", type=int, default=5, help="сколько разделов документации брать")
    parser.add_argument("--min-score", type=float, default=0.0, help="минимальное сходство чанка")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--no-mcp", action="store_true", help="не подключать MCP-сервер проекта")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()

    print("Инициализация индекса документации и MCP-сервера проекта...")
    try:
        assistant = build_dev_assistant(
            config=config,
            offline=args.offline,
            index_dir=args.index_dir,
            dataset=args.dataset,
            strategy=args.strategy,
            top_k=args.top_k,
            min_score=args.min_score,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            with_project_mcp=not args.no_mcp,
        )
    except Exception as exc:  # noqa: BLE001 — понятная ошибка инициализации
        print(f"✗ Ошибка инициализации: {exc}", file=sys.stderr)
        return 1

    state = {
        "history": [],
        "last_sources": [],
        "index_dir": args.index_dir,
        "dataset": args.dataset,
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }

    if args.question:
        answer = assistant.ask(args.question)
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
        if user_input.lower() in ("exit", "quit", "q"):
            print("До свидания!")
            break

        if handle_command(user_input, assistant, state):
            continue

        answer = assistant.ask(user_input, history=state["history"])
        state["last_sources"] = answer.sources
        _print_answer(answer)
        if not answer.error:
            state["history"].append({"role": "user", "content": user_input})
            state["history"].append({"role": "assistant", "content": answer.content})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
