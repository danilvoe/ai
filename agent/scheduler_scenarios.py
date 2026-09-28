#!/usr/bin/env python3
"""Day 18: планировщик и фоновые задачи — подсчёт публичных проектов GitFlic.

Сценарий показывает MCP-инструмент с отложенным/периодическим выполнением:

1. **Инструменты планировщика** — подключение к MCP-серверу и список
   инструментов: ``count_public_projects``, ``schedule_project_count``,
   ``list_count_jobs``, ``cancel_count_job``, ``get_project_count_report``.
2. **Отложенная задача** — постановка задачи через MCP
   (``schedule_project_count`` с ``delay_seconds``) и её сохранение в JSON.
3. **Запуск по cron** — внешний запуск ``mcp/gitflic_cron.py --run-due``
   выполняет готовые задачи, сохраняет снимок в JSON и печатает агрегат.
4. **Агрегированный результат** — ``get_project_count_report`` читает
   JSON-хранилище и возвращает последний агрегат (всего, языки, владельцы)
   и историю снимков.

API GitFlic требует авторизацию: токен (scope ``PROJECT_READ``) в
переменной окружения ``GITFLIC_TOKEN`` или в секции ``tools`` конфигурации.

Запуск::

    GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios
    GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios --delay 2 --interval 60
    GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios --skip-cron
    GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios --agent "сколько публичных проектов в gitflic"
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import LLMClient
from agent.mcp_tools import runtime_from_config
from agent.tokens import estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRON_SCRIPT = PROJECT_ROOT / "mcp" / "gitflic_cron.py"
SCHEDULER_TOOLS = (
    "count_public_projects",
    "schedule_project_count",
    "list_count_jobs",
    "cancel_count_job",
    "get_project_count_report",
)


def _print_tools(runtime) -> None:
    print("=" * 62)
    print("1. MCP-сервер: инструменты подсчёта и планировщика")
    print("=" * 62)
    specs = runtime.list_tools(refresh=True)
    for spec in specs:
        mark = " *" if spec.name in SCHEDULER_TOOLS else "  "
        first = spec.description.splitlines()[0] if spec.description else ""
        print(f"{mark} {spec.name} — {first}")
    print("\n   (* — инструменты дня 18)\n")


def _schedule(runtime, delay: float, interval: float | None) -> dict:
    print("=" * 62)
    print("2. Постановка отложенной/периодической задачи через MCP")
    print("=" * 62)
    args: dict = {"delay_seconds": delay}
    if interval is not None:
        args["interval_seconds"] = interval
    result = runtime.call("schedule_project_count", args)
    job = result.structured or {}
    print(json.dumps(job, ensure_ascii=False, indent=2))
    print()
    return job


def _run_cron(command: list[str], note: str, delay: float = 0.0) -> dict | None:
    print("=" * 62)
    print("3. Запуск по cron (тот же JSON-скрипт, что зовёт crontab)")
    print("=" * 62)
    print(note)
    if delay > 0:
        print(f"Ждём {delay:.0f} c до срока задачи ...")
        time.sleep(delay)
    full = [sys.executable, str(CRON_SCRIPT), *command]
    print("Команда: " + " ".join(full) + "\n")
    completed = subprocess.run(
        full,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if output:
        try:
            data = json.loads(output)
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(output)
    if completed.returncode != 0:
        print(f"(cron вернул код {completed.returncode}) {completed.stderr.strip()}")
    print()
    return None


def _print_report(runtime) -> None:
    print("=" * 62)
    print("4. Агрегированный результат из JSON-хранилища")
    print("=" * 62)
    result = runtime.call("get_project_count_report", {"history_limit": 5})
    report = result.structured or {}
    latest = report.get("latest") or {}
    print(f"Файл снимков : {report.get('storePath')}")
    print(f"Всего снимков: {report.get('snapshotCount')}")
    print(f"Публичных проектов (итог API) : {latest.get('totalPublicProjects')}")
    print(f"Разобрано в выборку           : {latest.get('countedProjects')}")
    print(f"Последний подсчёт             : {latest.get('countedAt')}")
    languages = latest.get("languages") or {}
    if languages:
        top = list(languages.items())[:5]
        print("Топ языков (выборка): " + ", ".join(f"{k}={v}" for k, v in top))
    owners = latest.get("topOwners") or []
    if owners:
        print("Топ владельцев: " + ", ".join(f"{o['name']}={o['count']}" for o in owners[:5]))
    print(f"История: {json.dumps(report.get('history') or [], ensure_ascii=False)}")
    print()


def _run_agent(request: str, runtime) -> None:
    print("=" * 62)
    print("5. Вызов инструмента агентом")
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
    runtime = runtime_from_config(config)
    parser = argparse.ArgumentParser(
        description="Планировщик: подсчёт публичных проектов GitFlic по расписанию"
    )
    parser.add_argument("--delay", type=float, default=2.0, help="задержка задачи, c")
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="период повторения задачи, c (по умолчанию — один отложенный запуск)",
    )
    parser.add_argument("--skip-cron", action="store_true", help="не звать cron-скрипт")
    parser.add_argument(
        "--agent",
        nargs="?",
        const="Сколько публичных проектов в GitFlic?",
        default=None,
        help="вызвать инструмент силами агента (нужен API-ключ LLM)",
    )
    args = parser.parse_args()

    if runtime is None:
        print(
            "MCP-инструменты выключены: включите секцию tools в config и укажите "
            "GITFLIC_TOKEN.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _print_tools(runtime)
    job = _schedule(runtime, args.delay, args.interval)

    if not args.skip_cron:
        if args.interval is not None:
            # Периодическая задача: срок уже наступил — выполняем и снимаем.
            _run_cron(
                ["--run-due"],
                "Выполняем периодическую задачу (crontab зовёт это каждую минуту).",
            )
            runtime.call("cancel_count_job", {"job_id": job.get("jobId")})
        else:
            _run_cron(
                ["--run-due"],
                "Ждём срок и выполняем отложенную задачу.",
                delay=args.delay,
            )

    _print_report(runtime)

    if args.agent is not None:
        _run_agent(args.agent, runtime)


if __name__ == "__main__":
    main()
