#!/usr/bin/env python3
"""Day 20: сквозной сценарий оркестрации нескольких MCP-серверов.

Сценарий показывает, как агент ведёт **длинный флоу** между тремя MCP-серверами
одного каталога:

1. **Регистрация серверов** — подключаются ``gitflic`` (проекты и подсчёт),
   ``pipeline`` (поиск → выжимка → сохранение) и ``report`` (журнал и индекс
   отчётов); печатается состав каждого сервера.
2. **Единый каталог** — инструменты всех серверов видны агенту как
   ``сервер.инструмент``; по нему LLM выбирает следующие вызовы.
3. **Длинный флоу агента** — на одном запросе агент сам выбирает и вызывает
   инструменты **разных серверов** в нужном порядке (подсчёт → сборка отчёта →
   запись в журнал → индекс). Печатается трейс: шаг, сервер, инструмент,
   аргументы и результат.
4. **Проверка** — ``verify_agent_flow`` подтверждает: задействовано ≥ 2 (фактически
   3) сервера, каждый вызов ушёл на нужный сервер, а порядок функционально верен
   (сначала данные, потом обработка/сохранение, затем журнал и агрегация).
5. **Файлы результата** — сохранённый отчёт и индекс отчётов.

Режимы:

* по умолчанию — живой прогон: запрос к LLM (выбор инструментов) + вызовы
  настоящих MCP-серверов (нужен доступ к GitFlic и токен);
* ``--offline`` — тот же цикл агента, но серверы заменены на детерминированные
  заглушки, а роль LLM играет скриптованный план: воспроизводится без сети.

Запуск::

    python3 -m agent.orchestration_scenarios --offline
    GITFLIC_TOKEN=<token> python3 -m agent.orchestration_scenarios
    GITFLIC_TOKEN=<token> python3 -m agent.orchestration_scenarios --query docs --size 5
"""

import argparse
import json
import sys
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.orchestrator import (
    McpOrchestrator,
    StaticToolRuntime,
    describe_flow,
    orchestrator_from_config,
    verify_agent_flow,
)
from agent.tokens import estimate_tokens
from agent.mcp_tools import ToolSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERY = "docs"
DEFAULT_OUTPUT = "history/reports/gitflic_report.md"
DEFAULT_INDEX = "history/reports/index.md"


def _preview(value, limit: int = 160) -> str:
    """Короткое текстовое представление значения для отчёта."""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_spec(name: str, description: str, properties: dict, required: list[str]) -> ToolSpec:
    """Собирает компактную JSON-схему инструмента для заглушки/сценария."""
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _offline_orchestrator(query: str, output: str, index_path: str) -> McpOrchestrator:
    """Собирает оркестратор на детерминированных серверах-заглушках (без сети)."""

    def count_projects(arguments: dict) -> dict:
        return {
            "query": arguments.get("query", ""),
            "totalPublicProjects": 7,
            "byLanguage": [{"language": "Python", "count": 4}, {"language": "Go", "count": 3}],
            "source": "offline",
        }

    def run_pipeline(arguments: dict) -> dict:
        target = PROJECT_ROOT / (arguments.get("output_path") or output)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "# Отчёт по проектам GitFlic (offline)\n\n"
            f"Поисковый запрос: `{arguments.get('query', '')}`\n\n"
            "## Выжимка\n\n"
            "Публичные проекты по запросу сгруппированы, ключевые описания выделены.\n"
        )
        target.write_text(text, encoding="utf-8")
        return {
            "ok": True,
            "savedPath": str(target.relative_to(PROJECT_ROOT)),
            "summary": {"summary": "Публичные проекты по запросу.", "itemCount": 5},
        }

    def record_run(arguments: dict) -> dict:
        store = PROJECT_ROOT / "history" / "report_runs.json"
        store.parent.mkdir(parents=True, exist_ok=True)
        data = {"createdAt": "offline", "runs": []}
        if store.exists():
            try:
                data = json.loads(store.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        runs = data.setdefault("runs", [])
        record = {
            "id": len(runs) + 1,
            "query": arguments.get("query", ""),
            "reportPath": arguments.get("report_path", ""),
            "totalProjects": int(arguments.get("total_projects") or 0),
            "tags": arguments.get("tags") or [],
            "recordedAt": "offline",
        }
        runs.append(record)
        data["runCount"] = len(runs)
        store.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"recorded": True, "runCount": len(runs), "record": record}

    def build_index(arguments: dict) -> dict:
        target = PROJECT_ROOT / (arguments.get("path") or index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        reports = sorted(
            item.name
            for item in (PROJECT_ROOT / "history" / "reports").glob("*.md")
            if item.name != target.name
        )
        lines = ["# Индекс отчётов (offline)", ""]
        lines += [f"- {name}" for name in reports] or ["- (нет отчётов)"]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "relativePath": str(target.relative_to(PROJECT_ROOT)),
            "reportCount": len(reports),
        }

    def list_reports(arguments: dict) -> dict:
        directory = PROJECT_ROOT / (arguments.get("directory") or "history/reports")
        reports = [item.name for item in directory.glob("*.md")] if directory.is_dir() else []
        return {"count": len(reports), "reports": reports}

    gitflic = StaticToolRuntime(
        specs=[
            _tool_spec(
                "count_public_projects",
                "Посчитать количество публичных проектов GitFlic по запросу.",
                {"query": {"type": "string"}},
                [],
            )
        ],
        handlers={"count_public_projects": count_projects},
    )
    pipeline = StaticToolRuntime(
        specs=[
            _tool_spec(
                "run_pipeline",
                "Собрать и сохранить отчёт по публичным проектам GitFlic одним вызовом.",
                {
                    "query": {"type": "string"},
                    "output_path": {"type": "string"},
                    "size": {"type": "integer"},
                },
                ["query"],
            )
        ],
        handlers={"run_pipeline": run_pipeline},
    )
    report = StaticToolRuntime(
        specs=[
            _tool_spec(
                "record_run",
                "Записать запуск флоу в журнал отчётов.",
                {
                    "query": {"type": "string"},
                    "report_path": {"type": "string"},
                    "total_projects": {"type": "integer"},
                },
                [],
            ),
            _tool_spec(
                "build_report_index",
                "Собрать единый индекс отчётов из сохранённых отчётов и журнала.",
                {"directory": {"type": "string"}, "path": {"type": "string"}},
                [],
            ),
            _tool_spec(
                "list_reports",
                "Перечислить сохранённые отчёты.",
                {"directory": {"type": "string"}},
                [],
            ),
        ],
        handlers={
            "record_run": record_run,
            "build_report_index": build_index,
            "list_reports": list_reports,
        },
    )
    return McpOrchestrator(
        [("gitflic", gitflic), ("pipeline", pipeline), ("report", report)]
    )


class ScriptedLLMClient:
    """Заглушка LLM для ``--offline``: заранее заданный план вызовов.

    Отвечает на запросы планировщика (в system-сообщении есть слово
    «диспетчер») следующим пунктом плана, извлекая уже полученные результаты из
    текста запроса. На основной вопрос отдаёт заранее заданный ответ.
    """

    def __init__(self, plan, final: str = "Готово: отчёт собран, запуск записан, индекс обновлён.") -> None:
        self._plan = plan
        self._final = final
        self.config = {"model": "scripted", "temperature": 0.0}
        self.plans: list[str] = []

    @staticmethod
    def _extract_results(system: str) -> list[dict]:
        marker = "Уже выполненные шаги"
        position = system.find(marker)
        if position < 0:
            return []
        tail = system[position:]
        start = tail.find("[")
        if start < 0:
            return []
        try:
            data = json.loads(tail[start:])
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:  # noqa: ARG002
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        if "диспетчер" in system:
            results = self._extract_results(system)
            step = self._plan(results)
            plan = step if step is not None else {"tool": None}
            content = json.dumps(plan, ensure_ascii=False)
            self.plans.append(content)
        else:
            content = self._final
        return Completion(
            content=content,
            model="scripted",
            prompt_tokens=0,
            completion_tokens=estimate_tokens(content),
        )


def _scripted_plan(query: str, output: str, index_path: str):
    """План длинного флоу для офлайн-режима (по шагам, с данными из результатов)."""

    def plan(results: list[dict]):
        step = len(results)
        if step == 0:
            return {"tool": "gitflic.count_public_projects", "arguments": {"query": query}}
        if step == 1:
            return {
                "tool": "pipeline.run_pipeline",
                "arguments": {"query": query, "output_path": output, "size": 5},
            }
        if step == 2:
            total = (results[0].get("result") or {}).get("totalPublicProjects") or 0
            saved = (results[1].get("result") or {}).get("savedPath") or output
            return {
                "tool": "report.record_run",
                "arguments": {
                    "query": query,
                    "report_path": saved,
                    "total_projects": total,
                },
            }
        if step == 3:
            return {
                "tool": "report.build_report_index",
                "arguments": {"directory": "history/reports", "path": index_path},
            }
        return None

    return plan


def _print_servers(orchestrator: McpOrchestrator) -> None:
    print("=" * 64)
    print("1. Зарегистрированные MCP-серверы и их инструменты")
    print("=" * 64)
    print(orchestrator.summarize())
    print()


def _print_catalog(orchestrator: McpOrchestrator) -> None:
    print("=" * 64)
    print("2. Единый каталог инструментов (по серверам)")
    print("=" * 64)
    catalog = orchestrator.catalog()
    for item in catalog:
        required = set(item["parameters"].get("required") or [])
        params = ", ".join(
            f"{name}{'*' if name in required else ''}"
            for name in (item["parameters"].get("properties") or {})
        )
        print(f"  [{item['server']:<8}] {item['name']:<22} параметры: {params or '—'}")
    print(f"\n  Всего инструментов: {len(catalog)} "
          f"на {len(orchestrator.server_names)} серверах.\n")


def _build_agent(orchestrator: McpOrchestrator, client, max_steps: int) -> Agent:
    conversation = SessionStore().create()
    return Agent(
        client,
        conversation,
        max_context_tokens=None,
        tools=orchestrator,
        max_tool_steps=max_steps,
    )


def _run_flow(agent: Agent, task: str, max_steps: int) -> tuple[list, list, str]:
    print("=" * 64)
    print("3. Длинный флоу: агент сам выбирает инструменты разных серверов")
    print("=" * 64)
    print(f"Задача: {task}")
    print(f"Лимит шагов: {max_steps}\n")
    reply = agent.ask(task)
    calls = agent.last_tool_calls
    results = agent.last_tool_results

    if not calls:
        print("Агент не вызвал ни одного инструмента.\n")
    else:
        print("Трейс вызовов (порядок важен):")
        print("\n".join(describe_flow(calls, results)))
        print()
        for index, (call, result) in enumerate(zip(calls, results), start=1):
            print(f"   Шаг {index}: {call.name}")
            print(f"     аргументы: {_preview(call.arguments)}")
            if result.is_error:
                print(f"     ошибка: {_preview(result.content)}")
            else:
                keys = ", ".join((result.structured or {}).keys()) or "—"
                print(f"     результат: {keys}")
        print()
    print(f"Ответ агента: {_preview(reply.content, 240)}\n")
    return calls, results, reply.content


def _verify(calls: list, results: list) -> bool:
    print("=" * 64)
    print("4. Проверка выбора серверов и порядка вызовов")
    print("=" * 64)
    problems = verify_agent_flow(
        calls,
        results,
        required_servers=["gitflic", "pipeline", "report"],
        min_servers=3,
    )
    servers: list[str] = []
    for result in results:
        if result.server and result.server not in servers:
            servers.append(result.server)
    print(f"Задействованные серверы: {', '.join(servers) or '—'}")
    if problems:
        print("Проверка не пройдена:")
        for problem in problems:
            print(f"  - {problem}")
        print()
        return False
    print("Проверка пройдена:")
    print("  - инструменты вызваны с 3 разных серверов;")
    print("  - каждый вызов маршрутизирован на сервер, объявивший инструмент;")
    print("  - порядок функционально верный: данные → обработка/сохранение → "
          "журнал → индекс.\n")
    return True


def _collect_paths(results: list, fallback: list[str]) -> list[str]:
    """Собирает пути сохранённых файлов из результатов инструментов."""
    paths: list[str] = []
    seen: set[str] = set()
    for result in results:
        data = result.structured or {}
        for key in ("savedPath", "relativePath", "path"):
            value = data.get(key)
            if not value:
                continue
            resolved = Path(str(value))
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            marker = str(resolved.resolve())
            if marker not in seen:
                seen.add(marker)
                paths.append(str(value))
    return paths or fallback


def _print_files(paths: list[str]) -> None:
    print("=" * 64)
    print("5. Файлы результата")
    print("=" * 64)
    for relative in paths:
        target = Path(relative)
        if not target.is_absolute():
            target = PROJECT_ROOT / relative
        if not target.exists():
            print(f"Не найден: {relative}\n")
            continue
        text = target.read_text(encoding="utf-8")
        print(f"{relative} ({len(text.encode('utf-8'))} байт)")
        print(text if len(text) <= 500 else text[:500] + "\n…")
        print()


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Оркестрация нескольких MCP-серверов: длинный флоу агента"
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="поисковый запрос")
    parser.add_argument("--size", type=int, default=5, help="сколько проектов в отчёте")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="куда сохранить отчёт",
    )
    parser.add_argument("--index", default=DEFAULT_INDEX, help="куда сохранить индекс")
    parser.add_argument(
        "--max-steps", type=int, default=6, help="максимум шагов флоу за один ход"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="офлайн-прогон на заглушках (без сети и LLM)",
    )
    args = parser.parse_args()

    output = args.output
    index_path = args.index

    if args.offline:
        orchestrator = _offline_orchestrator(args.query, output, index_path)
        client = ScriptedLLMClient(_scripted_plan(args.query, output, index_path))
    else:
        orchestrator = orchestrator_from_config(config)
        if orchestrator is None:
            print(
                "Оркестратор выключен: включите секцию orchestrator в config и "
                "укажите GITFLIC_TOKEN (или запустите с --offline).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not config.get("model"):
            config["model"] = "deepseek-v4-flash"
        client = LLMClient(config)

    task = (
        f"Подготовь отчёт по публичным проектам GitFlic по запросу '{args.query}': "
        "посчитай их общее количество, собери и сохрани отчёт по этим проектам, "
        "зафиксируй запуск в журнале отчётов и собери единый индекс отчётов."
    )

    try:
        _print_servers(orchestrator)
        _print_catalog(orchestrator)
        agent = _build_agent(orchestrator, client, args.max_steps)
        calls, results, _ = _run_flow(agent, task, args.max_steps)
        ok = _verify(calls, results)
        _print_files(_collect_paths(results, [output, index_path]))
    except Exception as exc:  # noqa: BLE001 — наглядная ошибка подключения/вызова
        print(f"Ошибка оркестрации MCP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
