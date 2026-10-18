"""День 34: интерактивный CLI ассистента для работы с файлами.

Ассистент получает **цель** («найди, где используется X», «сгенерируй changelog»,
«обнови документацию», «проверь правила», «подготовь diff») и сам решает, какие
файлы читать, искать и изменять. Операции выполняются через MCP-сервер
``mcp/fs_server.py``; по умолчанию изменения только показываются как diff
(реальная запись — после ``/apply on`` или флага ``--apply``).

Запуск::

    python3 -m agent.file_cli
    python3 chat_files.py
    python3 chat_files.py --offline --goal "найди, где используется McpToolRuntime"
    python3 chat_files.py --offline --workspace history/day34/sandbox/demo --apply
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config
from .file_assistant import (
    FileAssistant,
    FileTaskResult,
    build_file_assistant,
)

HELP_TEXT = """
Доступные команды:
  /find <символ>        — найти все места использования символа (по многим файлам)
  /search <запрос>      — поиск текста по файлам проекта (MCP search_files)
  /read <путь>          — прочитать файл (MCP read_file)
  /analyze <путь>       — разобрать структуру файла (MCP analyze_file)
  /files [префикс]      — список файлов проекта (MCP list_files)
  /generate <kind> [цель] — сгенерировать файл: catalog | adr | changelog
  /update <документ>    — обновить документ по исходному коду
  /check                — проверить файлы на соответствие правилам (инварианты)
  /diff [путь]          — показать изменения (MCP make_diff)
  /apply on|off         — включить/выключить реальную запись (по умолчанию off)
  /help                 — показать эту справку
  /exit или quit        — завершить

Любой обычный текст уходит ассистенту как цель (ассистент сам выбирает файлы).

Примеры целей:
  • найди, где используется build_support_runtime
  • сгенерируй каталог MCP-серверов
  • проверь соответствие правилам проекта
  • обнови документацию по файловым операциям
"""

EXAMPLE_GOALS = (
    "найди, где используется McpToolRuntime",
    "сгенерируй каталог MCP-серверов",
    "проверь соответствие правилам проекта",
    "подготовь diff изменений",
)


def _llm_ready(config: dict) -> bool:
    """Есть ли в конфиге (или окружении) всё для живого LLM."""
    base_url = config.get("base_url") or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = config.get("api_key") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = config.get("model") or os.environ.get("LLM_MODEL")
    return bool(base_url and api_key and model)


def print_banner(assistant: FileAssistant) -> None:
    print("=" * 78)
    print("🗂️  АССИСТЕНТ ФАЙЛОВЫХ ОПЕРАЦИЙ (ДЕНЬ 34)")
    print("=" * 78)
    print(f"Рабочая область: {assistant.workspace_root}")
    print(f"Режим записи: {'включена (apply)' if assistant.apply else 'dry-run (только diff)'}")
    print("Команды: /find, /search, /read, /analyze, /files, /generate, /update, /check, /diff, /apply")
    print(f"Пример цели: «{EXAMPLE_GOALS[0]}». Введите цель или команду.\n")


def _print_result(result: FileTaskResult) -> None:
    if result.error:
        print(f"\n⚠️ Ошибка: {result.error}\n", file=sys.stderr)
        return
    print(f"\nАссистент [{result.task_type}] (файлов затронуто: {len(result.files_touched)}):")
    print(result.content)
    if result.findings:
        print(f"\n🔎 Находки ({len(result.findings)}):")
        for finding in result.findings[:20]:
            location = f"{finding.path}:{finding.line}" if finding.line else finding.path
            print(f"  - [{finding.severity}] {location} — {finding.message}")
        if len(result.findings) > 20:
            print(f"  … и ещё {len(result.findings) - 20}")
    for artifact in result.artifacts:
        status = artifact.get("status")
        if status == "diff":
            continue
        print(f"\n📝 {artifact.get('path')} ({status}):")
        diff = artifact.get("diff") or ""
        print(diff if diff else "  (без изменений)")
    print()


def handle_command(raw: str, assistant: FileAssistant, state: dict) -> bool:
    """Обрабатывает slash-команду. True — ввод был командой."""
    text = raw.strip()
    if not text.startswith("/"):
        return False

    parts = text[1:].split(None, 1)
    action = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if action in ("help", "h", "?"):
        print(HELP_TEXT)
        return True

    if action == "apply":
        if arg.lower() in ("on", "1", "true", "yes", "вкл"):
            assistant.apply = True
            print("✓ Реальная запись включена.\n")
        elif arg.lower() in ("off", "0", "false", "no", "выкл"):
            assistant.apply = False
            print("✓ Реальная запись выключена (только diff).\n")
        else:
            print(f"ℹ️ Режим записи: {'включена' if assistant.apply else 'dry-run'}.\n")
        return True

    if action == "find":
        if not arg:
            print("ℹ️ Укажите символ, например /find build_support_runtime.\n")
            return True
        _print_result(assistant.run(f"найди, где используется {arg}", symbol=arg))
        return True

    if action == "search":
        if not arg:
            print("ℹ️ Укажите запрос, например /search McpToolRuntime.\n")
            return True
        _print_result(assistant.run(f"найди, где используется {arg}", symbol=arg))
        return True

    if action == "read":
        data = assistant.read_file(arg) if arg else {"error": "не указан путь"}
        if data.get("error"):
            print(f"⚠️ Ошибка: {data['error']}\n")
        else:
            print(f"\n📄 {data.get('path')} ({data.get('total_lines', 0)} строк):\n{data.get('content', '')}\n")
        return True

    if action == "analyze":
        data = assistant.analyze_file(arg) if arg else {"error": "не указан путь"}
        if data.get("error"):
            print(f"⚠️ Ошибка: {data['error']}\n")
        else:
            print(f"\n🔬 {data.get('path')} ({data.get('kind')}): {data.get('total_lines')} строк")
            if data.get("functions"):
                print("   функции: " + ", ".join(item["name"] for item in data["functions"][:20]))
            if data.get("classes"):
                print("   классы: " + ", ".join(item["name"] for item in data["classes"][:20]))
            if data.get("headings"):
                print("   заголовки: " + ", ".join(item["title"] for item in data["headings"][:10]))
            print()
        return True

    if action == "files":
        data = assistant.list_files(path=arg, limit=200)
        if data.get("error"):
            print(f"⚠️ Ошибка: {data['error']}\n")
        else:
            print(f"\nФайлы (всего {data.get('total', 0)}):")
            for name in data.get("files", []):
                print(f"  {name}")
            print()
        return True

    if action == "generate":
        kind, _, target = arg.partition(" ")
        kind = kind or "catalog"
        _print_result(assistant.run(f"сгенерируй {kind}", kind=kind, target=target.strip()))
        return True

    if action == "update":
        doc = arg or "docs/architecture.md"
        _print_result(assistant.run("обнови документацию по коду", doc_path=doc))
        return True

    if action == "check":
        _print_result(assistant.run("проверь соответствие правилам проекта"))
        return True

    if action in ("diff", "changes"):
        data = assistant.get_diff(path=arg, max_chars=8000)
        if data.get("error"):
            print(f"⚠️ Ошибка: {data['error']}\n")
        elif not data.get("diff"):
            print("ℹ️ Изменений нет.\n")
        else:
            print(f"\nDiff ({data.get('mode')}, файлов: {data.get('changed', 0)}):\n{data.get('diff')}\n")
        return True

    print(f"⚠️ Неизвестная команда: /{action}. Введите /help.\n")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ассистент файловых операций: чтение, поиск, анализ и запись через MCP"
    )
    parser.add_argument("--offline", "--no-llm", dest="offline", action="store_true",
                        help="офлайн-режим без обращения к LLM API (детерминированный стаб)")
    parser.add_argument("--goal", "-g", default=None, help="задать одну цель и выйти")
    parser.add_argument("--workspace", default=None, help="рабочая область (по умолчанию корень проекта)")
    parser.add_argument("--apply", action="store_true", help="разрешить реальную запись файлов")
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--no-mcp", action="store_true", help="не подключать MCP-сервер файлов")
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

    try:
        assistant = build_file_assistant(
            config=config,
            offline=offline,
            workspace_root=args.workspace or "",
            apply=args.apply,
            max_chars=args.max_chars,
            with_fs_mcp=not args.no_mcp,
        )
    except Exception as exc:  # noqa: BLE001 — понятная ошибка инициализации
        print(f"✗ Ошибка инициализации: {exc}", file=sys.stderr)
        return 1

    if args.goal:
        result = assistant.run(args.goal)
        _print_result(result)
        return 0 if not result.error else 1

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

        if handle_command(user_input, assistant, state={}):
            continue

        _print_result(assistant.run(user_input))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
