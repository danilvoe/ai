#!/usr/bin/env python3
"""Day 19: сквозной сценарий композиции MCP-инструментов.

Сценарий показывает, как несколько независимых MCP-инструментов собираются в
**автоматический пайплайн** ``search → summarize → saveToFile``:

1. **Инструменты сервера** — подключение к ``mcp/pipeline_server.py`` и список
   инструментов (``search``, ``summarize``, ``saveToFile``, ``run_pipeline``).
2. **Автоматический пайплайн** — декларативная цепочка выполняется движком
   (``agent/pipeline.py``) шаг за шагом, данные передаются автоматически.
3. **Проверка передачи данных** — ``verify_pipeline_run`` убеждается, что
   ``summarize`` получил ``items`` от ``search``, а ``saveToFile`` — ``summary``
   от ``summarize``, и что файл сохранён.
4. **Композиция на стороне сервера** — один вызов ``run_pipeline`` выполняет всю
   цепочку внутри MCP-сервера и возвращает отчёт по шагам.
5. **Файл результата** — путь и превью сохранённого отчёта.
6. **Вызов агентом** (``--agent``) — агент сам вызывает составной инструмент.

API GitFlic требует авторизацию: токен (scope ``PROJECT_READ``) в переменной
окружения ``GITFLIC_TOKEN`` или в секции ``pipeline``/``tools`` конфигурации.

Запуск::

    GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios
    GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --query docs --size 5
    GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --skip-composite
    GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --agent "собери отчёт по проектам gitflic про docs"
"""

import argparse
import json
import sys
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import LLMClient
from agent.pipeline import (
    gitflic_summary_pipeline,
    pipeline_runtime_from_config,
    verify_pipeline_run,
)
from agent.tokens import estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_TOOLS = ("search", "summarize", "saveToFile", "run_pipeline")


def _preview(value, limit: int = 120) -> str:
    """Короткое текстовое представление значения для отчёта."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_tools(runtime) -> None:
    print("=" * 62)
    print("1. MCP-сервер пайплайна: доступные инструменты")
    print("=" * 62)
    specs = runtime.list_tools(refresh=True)
    for index, spec in enumerate(specs, start=1):
        mark = " *" if spec.name in PIPELINE_TOOLS else "  "
        first = spec.description.splitlines()[0] if spec.description else ""
        print(f"{index}.{mark} {spec.name} — {first}")
    print("\n   (* — инструменты и композиция дня 19)\n")


def _print_step(step) -> None:
    status = "ошибка" if step.is_error else "ok"
    print(f"   Шаг {step.name} → {step.tool} [{status}]")
    print("     аргументы:")
    for key, value in step.arguments.items():
        print(f"       {key} = {_preview(value)}")
    keys = ", ".join(step.output.keys()) or "—"
    print(f"     выход: {keys}")
    if step.is_error:
        detail = step.result.content or step.output.get("error") or "ошибка"
        print(f"     ошибка: {_preview(detail)}")
    print()


def _run_chain(runtime, query: str, size: int, max_sentences: int, output: str):
    print("=" * 62)
    print("2. Автоматическое выполнение цепочки search → summarize → saveToFile")
    print("=" * 62)
    pipeline = gitflic_summary_pipeline(
        query=query, output_path=output, size=size, max_sentences=max_sentences
    )
    print(f"Пайплайн: шагов {len(pipeline.steps)}, движок выполняет их автоматически.")
    print("Связи: summarize.items ← ${search.items}; "
          "saveToFile.content ← ${summarize.summary}\n")
    run = pipeline.run(runtime)
    for step in run.steps:
        _print_step(step)
    print(run.summary())
    print()
    return run


def _verify(run) -> bool:
    print("=" * 62)
    print("3. Проверка корректности передачи данных между инструментами")
    print("=" * 62)
    problems = verify_pipeline_run(run)
    if problems:
        print("Проверка не пройдена:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        return False
    print("Передача данных корректна:")
    print("  - search.items  →  summarize.items (список проектов совпадает)")
    print("  - summarize.summary  →  saveToFile.content (текст совпадает)")
    print("  - файл сохранён и его содержимое совпадает с content")
    print()
    return True


def _run_composite(runtime, query: str, size: int, max_sentences: int, output: str) -> None:
    print("=" * 62)
    print("4. Композиция на стороне сервера: один вызов run_pipeline")
    print("=" * 62)
    arguments = {
        "query": query,
        "output_path": output,
        "size": size,
        "max_sentences": max_sentences,
    }
    print(f"Вызов: run_pipeline {json.dumps(arguments, ensure_ascii=False)}\n")
    result = runtime.call("run_pipeline", arguments)
    data = result.structured or {}
    if data.get("error"):
        print(f"Ошибка: {data['error']}\n")
        return
    for step in data.get("steps") or []:
        name = step.get("name")
        tool = step.get("tool")
        out = step.get("result") or {}
        keys = ", ".join(out.keys()) or "—"
        print(f"   {name} → {tool} [ok] выход: {keys}")
    print(f"\n   Сохранено: {data.get('savedPath')}")
    summary = (data.get("summary") or {}).get("summary")
    if summary:
        print(f"   Выжимка: {_preview(summary, 240)}")
    print()


def _print_file(run) -> None:
    print("=" * 62)
    print("5. Файл результата")
    print("=" * 62)
    path = run.saved_path
    if not path:
        print("Файл не сохранён.\n")
        return
    target = Path(path)
    if not target.exists():
        print(f"Файл не найден: {path}\n")
        return
    text = target.read_text(encoding="utf-8")
    print(f"Путь : {target}")
    print(f"Байт : {len(text.encode('utf-8'))}")
    print("Содержимое:")
    print(text if len(text) <= 800 else text[:800] + "\n…")
    print()


def _run_agent(request: str, runtime) -> None:
    print("=" * 62)
    print("6. Вызов составного инструмента агентом")
    print("=" * 62)
    config = load_config()
    if not config.get("model"):
        config["model"] = "deepseek-v4-flash"
    conversation = SessionStore().create()
    agent = Agent(
        LLMClient(config),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        tools=runtime,
    )
    print(f"Запрос пользователю: {request}\n")
    reply = agent.ask(request)
    call = agent.last_tool_call
    if call is None:
        print("Агент не стал вызывать инструмент.")
    else:
        print(f"MCP-вызов агента: {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")
    print(f"\nОтвет агента:\n{reply.content}\n")
    print(f"(токенов в ответе ≈ {estimate_tokens(reply.content)})\n")


def main() -> None:
    config = load_config()
    runtime = pipeline_runtime_from_config(config)
    parser = argparse.ArgumentParser(
        description="Композиция MCP-инструментов: search → summarize → saveToFile"
    )
    parser.add_argument("--query", default="", help="поисковый запрос для search")
    parser.add_argument("--size", type=int, default=5, help="сколько проектов запросить")
    parser.add_argument(
        "--max-sentences", type=int, default=3, help="сколько предложений в выжимке"
    )
    parser.add_argument(
        "--output",
        default=(config.get("pipeline", {}) or {}).get(
            "output_path", "history/pipeline_result.md"
        ),
        help="куда сохранить отчёт (saveToFile)",
    )
    parser.add_argument(
        "--skip-composite", action="store_true", help="не вызывать серверный run_pipeline"
    )
    parser.add_argument(
        "--agent",
        nargs="?",
        const="Собери отчёт по публичным проектам GitFlic",
        default=None,
        help="вызвать составной инструмент силами агента (нужен API-ключ LLM)",
    )
    args = parser.parse_args()

    if runtime is None:
        print(
            "Пайплайн выключен: включите секцию pipeline в config и укажите "
            "GITFLIC_TOKEN.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        _print_tools(runtime)
        run = _run_chain(runtime, args.query, args.size, args.max_sentences, args.output)
        ok = _verify(run)
        if not args.skip_composite:
            _run_composite(runtime, args.query, args.size, args.max_sentences, args.output)
        _print_file(run)
    except Exception as exc:  # noqa: BLE001 — наглядная ошибка подключения/вызова
        print(f"Ошибка MCP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.agent is not None:
        _run_agent(args.agent, runtime)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
