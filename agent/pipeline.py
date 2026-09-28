#!/usr/bin/env python3
"""Day 19: движок композиции MCP-инструментов.

Модуль собирает несколько независимых MCP-инструментов в **автоматический
пайплайн**. Шаги описываются декларативно (:class:`PipelineStep`), а движок
(:class:`Pipeline`) выполняет их по порядку, **передавая данные** из результата
одного шага в аргументы следующего.

Связь между шагами задаётся ссылками вида ``${имя_шага.путь.к.полю}`` прямо в
аргументах. Перед вызовом инструмента движок разрешает ссылки по контексту
(результаты уже выполненных шагов), поэтому, например, ``summarize`` получает
ровно тот список ``items``, который вернул ``search``, а ``saveToFile`` —
ровно ту ``summary``, которую вернул ``summarize``.

Проверка корректности передачи данных — :func:`verify_pipeline_run`: она
сравнивает фактические аргументы шагов с выходами предыдущих шагов и убеждается,
что файл действительно сохранён.

Запуск самопроверки движка без сети::

    python3 -m agent.pipeline
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .mcp_tools import McpToolRuntime, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_SCRIPT = PROJECT_ROOT / "mcp" / "pipeline_server.py"

# Ссылка на данные предыдущего шага: ${step.path.to.field}.
REF_RE = re.compile(r"\$\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}")


@dataclass
class PipelineStep:
    """Один шаг пайплайна: какой инструмент и с какими аргументами вызвать."""

    name: str
    tool: str
    arguments: dict = field(default_factory=dict)


@dataclass
class StepResult:
    """Результат выполнения шага: фактические аргументы и ответ инструмента."""

    name: str
    tool: str
    arguments: dict
    result: ToolResult

    @property
    def is_error(self) -> bool:
        return self.result.is_error

    @property
    def output(self) -> dict:
        """Структурированный результат шага (пустой, если его нет)."""
        return self.result.structured or {}


@dataclass
class PipelineRun:
    """Результат всего пайплайна: шаги по порядку и итоговый выход."""

    query: str
    steps: list[StepResult]

    @property
    def ok(self) -> bool:
        """Все шаги выполнены без ошибок и пайплайн не пуст."""
        return bool(self.steps) and all(step.is_error is False for step in self.steps)

    @property
    def final(self) -> dict:
        """Структурированный выход последнего шага."""
        return self.steps[-1].output if self.steps else {}

    @property
    def saved_path(self) -> str | None:
        """Путь сохранённого файла (из последнего шага, если он пишет файл)."""
        for step in reversed(self.steps):
            path = step.output.get("path")
            if path:
                return path
        return None

    def step(self, name: str) -> StepResult | None:
        """Шаг по имени (None, если такого шага нет)."""
        return next((step for step in self.steps if step.name == name), None)

    def summary(self) -> str:
        """Человекочитаемая сводка по шагам пайплайна."""
        lines = [f"Пайплайн: шагов {len(self.steps)}, статус: {'ok' if self.ok else 'ошибка'}"]
        for index, step in enumerate(self.steps, start=1):
            status = "ошибка" if step.is_error else "ok"
            keys = ", ".join(step.output.keys()) or "—"
            lines.append(
                f"  {index}. {step.name} → {step.tool} [{status}] выход: {keys}"
            )
        return "\n".join(lines)


def resolve_refs(value, context: dict):
    """Подставляет ``${step.path}`` в аргументы, используя результаты шагов.

    Строка, целиком состоящая из одной ссылки, заменяется на исходный объект
    (список/словарь), а не на текстовое представление — так между инструментами
    передаются структуры, а не их строки. Частичные ссылки внутри текста
    заменяются на подставленное значение.
    """
    if isinstance(value, dict):
        return {key: resolve_refs(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_refs(item, context) for item in value]
    if not isinstance(value, str):
        return value

    match = REF_RE.fullmatch(value)
    if match:
        return _lookup(context, match.group(1))

    def _replace(found: re.Match) -> str:
        resolved = _lookup(context, found.group(1))
        if isinstance(resolved, (dict, list)):
            return json.dumps(resolved, ensure_ascii=False)
        return str(resolved)

    return REF_RE.sub(_replace, value)


def _lookup(context: dict, path: str):
    """Достаёт значение по пути ``step.field.sub`` из контекста шагов."""
    parts = path.split(".")
    current = context
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return ""
    return current


class Pipeline:
    """Пайплайн из MCP-инструментов, выполняемый автоматически по шагам."""

    def __init__(self, steps: list[PipelineStep], query: str = "") -> None:
        self.steps = list(steps)
        self.query = query

    def run(self, runtime: McpToolRuntime, initial: dict | None = None) -> PipelineRun:
        """Выполняет шаги по порядку, передавая выходы в следующие аргументы.

        ``runtime`` — клиент MCP-сервера (см. ``agent/mcp_tools.py``). Для каждого
        шага ссылки ``${...}`` в аргументах разрешаются по контексту уже
        выполненных шагов, затем инструмент вызывается через MCP, а его
        структурированный результат кладётся в контекст под именем шага.
        """
        context: dict = dict(initial or {})
        results: list[StepResult] = []
        for step in self.steps:
            arguments = resolve_refs(step.arguments, context)
            result = runtime.call(step.tool, arguments)
            results.append(
                StepResult(
                    name=step.name,
                    tool=step.tool,
                    arguments=arguments,
                    result=result,
                )
            )
            context[step.name] = result.structured or {}
        return PipelineRun(query=self.query, steps=results)


def gitflic_summary_pipeline(
    query: str,
    output_path: str = "history/pipeline_result.md",
    size: int = 5,
    max_sentences: int = 3,
) -> Pipeline:
    """Собирает пайплайн ``search → summarize → saveToFile`` для GitFlic.

    Связи между инструментами:

    * ``summarize.items`` ← ``${search.items}`` (данные из первого шага);
    * ``saveToFile.content`` ← ``${summarize.summary}`` (результат обработки).
    """
    return Pipeline(
        query=query,
        steps=[
            PipelineStep(
                name="search",
                tool="search",
                arguments={"query": query, "page": 0, "size": size},
            ),
            PipelineStep(
                name="summarize",
                tool="summarize",
                arguments={
                    "items": "${search.items}",
                    "title": query,
                    "max_sentences": max_sentences,
                },
            ),
            PipelineStep(
                name="saveToFile",
                tool="saveToFile",
                arguments={
                    "content": "${summarize.summary}",
                    "path": output_path,
                    "format": "markdown",
                },
            ),
        ],
    )


def verify_pipeline_run(run: PipelineRun, check_file: bool = True) -> list[str]:
    """Проверяет автоматическое выполнение и передачу данных между шагами.

    Возвращает список проблем; пустой список означает, что всё корректно:

    * все шаги завершились без ошибок;
    * ``summarize`` получил ровно те ``items``, что вернул ``search``;
    * ``saveToFile`` получил ровно ту ``summary``, что вернул ``summarize``;
    * файл сохранён и его содержимое совпадает с переданным ``content``.
    """
    problems: list[str] = []

    if not run.steps:
        return ["Пайплайн не выполнен: нет шагов."]

    for step in run.steps:
        if step.is_error:
            detail = step.result.content or step.output.get("error") or "ошибка"
            problems.append(f"Шаг {step.name} завершился ошибкой: {detail}")

    search = run.step("search")
    summarize = run.step("summarize")
    save = run.step("saveToFile")

    if search is not None and summarize is not None:
        sent_items = summarize.arguments.get("items")
        produced_items = search.output.get("items")
        if not isinstance(sent_items, list) or not produced_items:
            problems.append("summarize не получил items от search.")
        elif len(sent_items) != len(produced_items) or sent_items != produced_items:
            problems.append(
                "Данные search → summarize переданы неверно: items не совпадают."
            )

    if summarize is not None and save is not None:
        sent_content = save.arguments.get("content")
        produced_summary = summarize.output.get("summary")
        if not isinstance(sent_content, str) or not sent_content:
            problems.append("saveToFile не получил summary от summarize.")
        elif sent_content != produced_summary:
            problems.append(
                "Данные summarize → saveToFile переданы неверно: summary не совпадает."
            )

    if check_file and save is not None and not save.is_error:
        path = save.output.get("path")
        if not path:
            problems.append("saveToFile не вернул путь к сохранённому файлу.")
        else:
            target = Path(path)
            if not target.exists():
                problems.append(f"Файл результата не найден: {path}")
            else:
                expected = save.arguments.get("content") or ""
                actual = target.read_text(encoding="utf-8").strip()
                if expected.strip() != actual:
                    problems.append("Содержимое файла не совпадает с переданным content.")

    return problems


def pipeline_runtime_from_config(config: dict) -> McpToolRuntime | None:
    """Создаёт рантайм пайплайна из секции ``pipeline`` конфигурации.

    Токен GitFlic берётся из секции ``pipeline``, а если его там нет — из секции
    ``tools`` (совместимо с конфигурацией дней 17–18). ``None`` — выключено.
    """
    section = config.get("pipeline", {}) or {}
    if not section.get("enabled", False):
        return None

    tools_section = config.get("tools", {}) or {}
    env: dict[str, str] = {}
    token = section.get("gitflic_token") or tools_section.get("gitflic_token")
    if token:
        env["GITFLIC_TOKEN"] = str(token)
    api_url = section.get("gitflic_api_url") or tools_section.get("gitflic_api_url")
    if api_url:
        env["GITFLIC_API_URL"] = str(api_url)

    args = section.get("args")
    cwd = section.get("cwd")
    if cwd:
        cwd = str((PROJECT_ROOT / cwd).resolve()) if not Path(cwd).is_absolute() else cwd
    else:
        cwd = str(PROJECT_ROOT)

    return McpToolRuntime(
        command=section.get("command") or sys.executable,
        args=args if args else [str(DEFAULT_PIPELINE_SCRIPT)],
        cwd=cwd,
        env=env,
        timeout=float(section.get("timeout", 60)),
    )


class _FakeRuntime:
    """Мини-рантайм для самопроверки движка без MCP и сети."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        arguments = arguments or {}
        self.calls.append((name, arguments))
        if name == "search":
            payload = {
                "query": arguments.get("query", ""),
                "count": 2,
                "items": [
                    {"id": 1, "title": "Alpha", "description": "Первый проект."},
                    {"id": 2, "title": "Beta", "description": "Второй проект."},
                ],
            }
        elif name == "summarize":
            items = arguments.get("items") or []
            payload = {
                "summary": "Alpha. Beta.",
                "bullets": [item.get("title") for item in items],
                "keywords": ["проект"],
                "itemCount": len(items),
            }
        elif name == "saveToFile":
            payload = {
                "path": str(PROJECT_ROOT / arguments.get("path", "history/pipeline_result.md")),
                "bytes": len(str(arguments.get("content", "")).encode("utf-8")),
            }
        else:
            payload = {"error": f"unknown tool {name}"}
        return ToolResult(
            name=name,
            arguments=arguments,
            content=json.dumps(payload, ensure_ascii=False),
            structured=payload,
            is_error=bool(payload.get("error")),
        )


def _self_test() -> int:
    """Проверяет разрешение ссылок и передачу данных на фиктивном рантайме."""
    runtime = _FakeRuntime()
    pipeline = gitflic_summary_pipeline("demo", output_path="history/self_test.md", size=2)
    run = pipeline.run(runtime)
    problems = verify_pipeline_run(run, check_file=False)

    print("Самопроверка движка композиции (без MCP и сети)")
    print("=" * 62)
    print(run.summary())
    print()
    print("Разрешённые аргументы шагов:")
    for step in run.steps:
        printable = {
            key: (f"<{type(value).__name__} len={len(value)}>" if isinstance(value, (list, dict)) else value)
            for key, value in step.arguments.items()
        }
        print(f"  {step.name}: {json.dumps(printable, ensure_ascii=False)}")
    print()
    if problems:
        print("Проблемы передачи данных:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Передача данных корректна: search.items → summarize → saveToFile.content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
