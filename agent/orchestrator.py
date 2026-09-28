#!/usr/bin/env python3
"""Day 20: оркестрация нескольких MCP-серверов.

Модуль регистрирует **несколько MCP-серверов** сразу и даёт агенту единый
каталог их инструментов. За routing отвечает :class:`McpOrchestrator`:

* инструменты всех серверов собираются в один каталог с меткой сервера
  (``сервер.инструмент``);
* :meth:`McpOrchestrator.resolve` находит, на какой сервер отправить вызов
  (по полному имени ``сервер.инструмент`` или по уникальному короткому имени);
* :meth:`McpOrchestrator.call` маршрутизирует вызов ровно на этот сервер и
  помечает результат полем ``server``.

Планирование длинного флоу ведёт LLM: на каждом шаге агенту показывают каталог
всех серверов и уже полученные результаты, а он выбирает **следующий**
инструмент (``plan_messages`` + ``parse_plan``). Цикл вызовов живёт в
``Agent._maybe_call_tool``; здесь же лежит проверка результата —
:func:`verify_agent_flow`, которая подтверждает, что в флоу реально участвовали
инструменты с разных серверов, каждый вызов ушёл на нужный сервер, а порядок
шагов функционально корректен (сначала получение данных, потом обработка,
потом сохранение и финальная агрегация).

Самопроверка без MCP и сети::

    python3 -m agent.orchestrator
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .mcp_tools import McpToolRuntime, ToolCall, ToolResult, ToolSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Инструменты по ролям — нужны для проверки порядка вызовов.
FETCH_TOOLS = {
    "search",
    "get_public_projects",
    "count_public_projects",
    "get_project_count_report",
    "list_count_jobs",
    "list_reports",
}
PROCESS_TOOLS = {"summarize"}
# run_pipeline на стороне сервера тоже сохраняет отчёт (search → summarize →
# saveToFile одним вызовом), поэтому считается производящим файл.
SAVE_TOOLS = {"saveToFile", "run_pipeline"}
JOURNAL_TOOLS = {"record_run"}
FINALIZE_TOOLS = {"build_report_index"}

FLOW_PLAN_PROMPT = """Ты — диспетчер MCP-инструментов с нескольких серверов.
Инструменты названы как "сервер.инструмент". Запрос пользователя может требовать
НЕСКОЛЬКО последовательных вызовов с разных серверов.

Реши, какой ОДИН инструмент нужно вызвать СЛЕДУЮЩИМ. Используй результаты уже
выполненных шагов как аргументы (например, передай items от поиска в обработку).

Ответь СТРОГО одним JSON-объектом без пояснений:
{"tool": "<сервер>.<инструмент>", "arguments": {<аргументы>}}

Если для выполнения запроса больше НЕ нужно вызывать инструменты, ответь:
{"tool": null}

Доступные инструменты (по серверам):
"""


class ToolRoutingError(Exception):
    """Вызов не удалось сопоставить ни одному инструменту известных серверов."""


@dataclass
class ServerSpec:
    """Описание одного MCP-сервера из конфигурации."""

    name: str
    command: str = sys.executable
    args: list[str] = field(default_factory=list)
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0

    def build_runtime(self) -> McpToolRuntime:
        """Создаёт клиент MCP-сервера (stdio) по этому описанию."""
        cwd = self.cwd
        if cwd:
            cwd = str((PROJECT_ROOT / cwd).resolve()) if not Path(cwd).is_absolute() else str(cwd)
        else:
            cwd = str(PROJECT_ROOT)
        args = self.args or [str(PROJECT_ROOT / "mcp" / f"{self.name}_server.py")]
        return McpToolRuntime(
            command=self.command or sys.executable,
            args=args,
            cwd=cwd,
            env=self.env,
            timeout=self.timeout,
        )


@dataclass
class RoutedTool:
    """Инструмент с указанием сервера, который его объявляет."""

    server: str
    spec: ToolSpec

    @property
    def name(self) -> str:
        """Короткое имя инструмента без сервера."""
        return self.spec.name

    @property
    def qualified(self) -> str:
        """Полное имя вида ``сервер.инструмент``."""
        return f"{self.server}.{self.spec.name}"

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def input_schema(self) -> dict:
        return self.spec.input_schema


class StaticToolRuntime:
    """Рантайм-заглушка без MCP: фиксированные инструменты и их обработчики.

    Используется в офлайн-самопроверке и сценарии, чтобы прогнать оркестрацию и
    проверку маршрутизации без сети и без LLM-сервера.
    """

    def __init__(
        self,
        specs: list[ToolSpec],
        handlers: dict[str, Callable[[dict], dict]],
    ) -> None:
        self._specs = list(specs)
        self._handlers = dict(handlers)
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self, refresh: bool = False) -> list[ToolSpec]:  # noqa: ARG002
        return list(self._specs)

    def has_tool(self, name: str) -> bool:
        return any(spec.name == name for spec in self._specs)

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        arguments = arguments or {}
        self.calls.append((name, arguments))
        handler = self._handlers.get(name)
        if handler is None:
            payload = {"error": f"unknown tool {name}"}
        else:
            payload = handler(arguments)
        return ToolResult(
            name=name,
            arguments=arguments,
            content=json.dumps(payload, ensure_ascii=False),
            structured=payload,
            is_error=bool(payload.get("error")),
        )


class McpOrchestrator:
    """Регистрирует несколько MCP-серверов и маршрутизирует вызовы инструментов."""

    def __init__(self, servers: list[tuple[str, object]]) -> None:
        self._runtimes: dict[str, object] = {}
        for name, runtime in servers:
            if not name:
                raise ValueError("У каждого MCP-сервера должно быть имя.")
            self._runtimes[name] = runtime

    def __bool__(self) -> bool:
        return bool(self._runtimes)

    @property
    def server_names(self) -> list[str]:
        return list(self._runtimes)

    def runtime(self, server: str):
        return self._runtimes.get(server)

    def list_tools(self, server: str, refresh: bool = False) -> list[ToolSpec]:
        runtime = self._runtimes.get(server)
        if runtime is None:
            raise ToolRoutingError(f"Неизвестный MCP-сервер: {server!r}.")
        return runtime.list_tools(refresh=refresh)

    def list_routes(self, refresh: bool = False) -> list[RoutedTool]:
        """Все инструменты всех серверов с меткой сервера."""
        routes: list[RoutedTool] = []
        for server, runtime in self._runtimes.items():
            for spec in runtime.list_tools(refresh=refresh):
                routes.append(RoutedTool(server=server, spec=spec))
        return routes

    def catalog(self, refresh: bool = False) -> list[dict]:
        """Каталог инструментов для LLM: сервер, имя, описание и параметры."""
        return [
            {
                "server": route.server,
                "name": route.name,
                "tool": route.qualified,
                "description": route.description,
                "parameters": route.input_schema,
            }
            for route in self.list_routes(refresh=refresh)
        ]

    def resolve(self, tool: str) -> RoutedTool:
        """Находит, какой сервер объявляет инструмент ``tool``.

        Принимает полное имя ``сервер.инструмент`` или короткое имя, если оно
        уникально среди всех серверов. Неизвестное или неоднозначное имя — ошибка.
        """
        name = (tool or "").strip()
        if not name:
            raise ToolRoutingError("Пустое имя инструмента.")

        if "." in name:
            server, _, bare = name.partition(".")
            runtime = self._runtimes.get(server)
            if runtime is not None:
                for spec in runtime.list_tools():
                    if spec.name == bare:
                        return RoutedTool(server=server, spec=spec)
            raise ToolRoutingError(
                f"Инструмент {name!r} не найден ни на одном MCP-сервере."
            )

        matches = [route for route in self.list_routes() if route.name == name]
        if not matches:
            raise ToolRoutingError(
                f"Инструмент {name!r} не найден. Доступно серверов: "
                f"{', '.join(self.server_names)}."
            )
        if len(matches) > 1:
            servers = ", ".join(route.server for route in matches)
            raise ToolRoutingError(
                f"Инструмент {name!r} объявлен несколькими серверами ({servers}); "
                f"укажите полное имя 'сервер.инструмент'."
            )
        return matches[0]

    def has_tool(self, tool: str) -> bool:
        try:
            self.resolve(tool)
        except ToolRoutingError:
            return False
        return True

    def call(self, tool: str, arguments: dict | None = None) -> ToolResult:
        """Маршрутизирует вызов инструмента на нужный сервер и выполняет его."""
        route = self.resolve(tool)
        runtime = self._runtimes[route.server]
        result = runtime.call(route.name, arguments or {})
        result.server = route.server
        return result

    def summarize(self) -> str:
        """Человекочитаемый список серверов и их инструментов."""
        lines = [f"MCP-серверов зарегистрировано: {len(self._runtimes)}"]
        for server, runtime in self._runtimes.items():
            command = getattr(runtime, "command", type(runtime).__name__)
            args = getattr(runtime, "args", None)
            where = f"{command} {' '.join(args)}" if args else str(command)
            specs = runtime.list_tools()
            lines.append(f"\n[{server}] инструментов: {len(specs)}  ({where})")
            for index, spec in enumerate(specs, start=1):
                required = set(spec.input_schema.get("required") or [])
                properties = spec.input_schema.get("properties") or {}
                params = ", ".join(
                    f"{name}{'*' if name in required else ''}" for name in properties
                )
                first = spec.description.splitlines()[0] if spec.description else ""
                lines.append(f"  {index}. {server}.{spec.name} — {first}")
                if params:
                    lines.append(f"     параметры: {params}")
        return "\n".join(lines)

    def plan_messages(
        self, user_request: str, results: list[dict] | None = None
    ) -> list[dict]:
        """Сообщения для выбора СЛЕДУЮЩЕГО инструмента под запрос.

        В каталоге видно, какой сервер объявляет инструмент; если переданы
        результаты уже выполненных шагов, они тоже уходят в контекст, чтобы LLM
        мог связать данные между серверами.
        """
        catalog = self.catalog()
        if not catalog:
            return []
        prompt = FLOW_PLAN_PROMPT + json.dumps(catalog, ensure_ascii=False, indent=2)
        if results:
            prompt += (
                "\n\nУже выполненные шаги (используй их результаты в аргументах):\n"
                + json.dumps(results, ensure_ascii=False, indent=2)
            )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_request},
        ]

    @staticmethod
    def parse_plan(text: str) -> ToolCall | None:
        """Разбирает JSON-план модели. None — инструменты больше не нужны."""
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        name = data.get("tool") or data.get("name")
        if not name:
            return None
        server = data.get("server")
        name = str(name)
        if server and "." not in name:
            name = f"{server}.{name}"
        arguments = data.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        server_name = name.partition(".")[0] if "." in name else None
        return ToolCall(name=name, arguments=arguments, server=server_name)

    @staticmethod
    def result_message(result: ToolResult) -> dict:
        """System-сообщение с результатом вызова (видно основной модели)."""
        label = f"{result.server}.{result.name}" if result.server else result.name
        payload = result.structured
        detail = (
            json.dumps(payload, ensure_ascii=False, indent=2)
            if payload is not None
            else result.content
        )
        return {
            "role": "system",
            "content": (
                f"Результат вызова MCP-инструмента {label} "
                f"с аргументами {json.dumps(result.arguments, ensure_ascii=False)}:\n"
                f"{detail}\n"
                "Используй эти данные в ответе пользователю."
            ),
        }


def orchestrator_from_config(config: dict) -> McpOrchestrator | None:
    """Создаёт оркестратор из секции ``orchestrator`` (None — выключено).

    Токен GitFlic и адрес API подтягиваются из секций ``tools``/``pipeline``,
    если не заданы в окружении конкретного сервера.
    """
    section = config.get("orchestrator", {}) or {}
    if not section.get("enabled", False):
        return None

    tools_section = config.get("tools", {}) or {}
    pipeline_section = config.get("pipeline", {}) or {}
    common_env: dict[str, str] = {}
    token = tools_section.get("gitflic_token") or pipeline_section.get("gitflic_token")
    if token:
        common_env["GITFLIC_TOKEN"] = str(token)
    api_url = tools_section.get("gitflic_api_url") or pipeline_section.get("gitflic_api_url")
    if api_url:
        common_env["GITFLIC_API_URL"] = str(api_url)

    servers: list[tuple[str, McpToolRuntime]] = []
    for item in section.get("servers") or []:
        name = item.get("name")
        if not name:
            continue
        env = dict(common_env)
        env.update({str(k): str(v) for k, v in (item.get("env") or {}).items()})
        spec = ServerSpec(
            name=str(name),
            command=item.get("command") or sys.executable,
            args=list(item.get("args") or []),
            cwd=item.get("cwd"),
            env=env,
            timeout=float(item.get("timeout", 60)),
        )
        servers.append((spec.name, spec.build_runtime()))
    if not servers:
        return None
    return McpOrchestrator(servers)


# --------------------------------------------------------------------------
# Проверка длинного флоу: участие разных серверов и порядок вызовов.
# --------------------------------------------------------------------------


def _bare(name: str) -> str:
    """Короткое имя инструмента без префикса сервера."""
    return name.partition(".")[2] if "." in name else name


def _first_index(names: list[str], candidates: set[str]) -> int | None:
    for index, name in enumerate(names):
        if name in candidates:
            return index
    return None


def verify_agent_flow(
    trace: list[ToolCall],
    results: list[ToolResult],
    required_servers: list[str] | None = None,
    min_servers: int = 2,
) -> list[str]:
    """Проверяет длинный флоу агента по разным MCP-серверам.

    Возвращает список проблем; пустой список — всё корректно:

    * агент вызвал хотя бы один инструмент, и все вызовы прошли без ошибок;
    * в флоу участвовали инструменты минимум ``min_servers`` разных серверов
      (и все серверы из ``required_servers``);
    * каждый вызов маршрутизирован на сервер, который объявляет этот инструмент;
    * порядок функционально верен: получение данных идёт до обработки, обработка
      до сохранения, сохранение до записи в журнал и финальной агрегации.
    """
    problems: list[str] = []
    if not trace:
        return ["Агент не вызвал ни одного MCP-инструмента."]
    if len(trace) != len(results):
        problems.append("Число вызовов и результатов не совпадает.")

    for index, (call, result) in enumerate(zip(trace, results), start=1):
        if result.is_error:
            detail = result.content or "ошибка"
            problems.append(
                f"Шаг {index} ({call.name}) завершился ошибкой: {detail}"
            )

    servers: list[str] = []
    for result in results:
        server = result.server or (
            result.name.partition(".")[0] if "." in result.name else None
        )
        if server and server not in servers:
            servers.append(server)

    if len(servers) < min_servers:
        problems.append(
            f"В флоу участвовал только {len(servers)} сервер(а): {servers or '—'}; "
            f"ожидалось минимум {min_servers}."
        )
    for server in required_servers or []:
        if server not in servers:
            problems.append(f"Сервер {server!r} не был задействован во флоу.")

    names = [_bare(call.name) for call in trace]

    summarize_at = _first_index(names, PROCESS_TOOLS)
    if summarize_at is not None:
        if _first_index(names[: summarize_at + 1], FETCH_TOOLS) is None:
            problems.append("Обработка (summarize) вызвана до получения данных.")

    save_at = _first_index(names, {"saveToFile"})
    if save_at is not None and summarize_at is None:
        problems.append("Сохранение (saveToFile) вызвано без предварительной обработки.")

    journal_at = _first_index(names, JOURNAL_TOOLS)
    if journal_at is not None:
        if _first_index(names[: journal_at + 1], SAVE_TOOLS) is None:
            problems.append("Запись в журнал (record_run) вызвана до сохранения отчёта.")

    finalize_at = _first_index(names, FINALIZE_TOOLS)
    if finalize_at is not None:
        producers = FETCH_TOOLS | SAVE_TOOLS | JOURNAL_TOOLS
        if _first_index(names[: finalize_at + 1], producers) is None:
            problems.append(
                "Финальная агрегация (build_report_index) вызвана без данных."
            )

    return problems


def describe_flow(trace: list[ToolCall], results: list[ToolResult]) -> list[str]:
    """Пошаговое описание флоу: номер, сервер, инструмент и ключи результата."""
    lines: list[str] = []
    for index, (call, result) in enumerate(zip(trace, results), start=1):
        server = result.server or (
            call.name.partition(".")[0] if "." in call.name else "—"
        )
        tool = _bare(call.name)
        status = "ошибка" if result.is_error else "ok"
        keys = ", ".join((result.structured or {}).keys()) or "—"
        lines.append(f"  {index}. [{server}] {tool} [{status}] → {keys}")
    return lines


# --------------------------------------------------------------------------
# Самопроверка без MCP и сети.
# --------------------------------------------------------------------------


def _fake_tool(name: str, description: str = "") -> ToolSpec:
    return ToolSpec(name=name, description=description or name, input_schema={"type": "object"})


def _self_test() -> int:
    """Проверяет реестр, роутинг и проверку порядка без MCP и сети."""
    gitflic = StaticToolRuntime(
        specs=[
            _fake_tool("get_public_projects"),
            _fake_tool("count_public_projects"),
            _fake_tool("get_project_count_report"),
        ],
        handlers={
            "get_public_projects": lambda a: {"items": [{"id": 1}], "count": 1},
            "count_public_projects": lambda a: {"totalPublicProjects": 42},
            "get_project_count_report": lambda a: {"latest": {"totalPublicProjects": 42}},
        },
    )
    pipeline = StaticToolRuntime(
        specs=[_fake_tool("search"), _fake_tool("summarize"), _fake_tool("saveToFile")],
        handlers={
            "search": lambda a: {"items": [{"id": 1, "title": "Alpha"}]},
            "summarize": lambda a: {"summary": "Alpha.", "itemCount": len(a.get("items") or [])},
            "saveToFile": lambda a: {"path": "history/reports/x.md", "bytes": 7},
        },
    )
    report = StaticToolRuntime(
        specs=[_fake_tool("record_run"), _fake_tool("build_report_index")],
        handlers={
            "record_run": lambda a: {"recorded": True, "runCount": 1},
            "build_report_index": lambda a: {"relativePath": "history/reports/index.md"},
        },
    )
    orchestrator = McpOrchestrator(
        [("gitflic", gitflic), ("pipeline", pipeline), ("report", report)]
    )

    print("Самопроверка оркестратора MCP (без MCP и сети)")
    print("=" * 62)
    print(orchestrator.summarize())
    print()

    errors: list[str] = []

    # Роутинг по полному и короткому имени.
    route = orchestrator.resolve("pipeline.summarize")
    if route.server != "pipeline" or route.name != "summarize":
        errors.append("resolve('pipeline.summarize') вернул неверный маршрут.")
    route = orchestrator.resolve("count_public_projects")
    if route.server != "gitflic":
        errors.append("resolve('count_public_projects') вернул неверный сервер.")

    for bad in ("nope", "gitflic.nope", "unknown_server.tool"):
        try:
            orchestrator.resolve(bad)
            errors.append(f"resolve({bad!r}) должен был упасть.")
        except ToolRoutingError:
            pass

    # Прямой вызов маршрутизируется и помечается сервером.
    result = orchestrator.call("gitflic.count_public_projects", {"query": "docs"})
    if result.server != "gitflic":
        errors.append("call() не пометил результат сервером gitflic.")

    # Проверка длинного флоу по трём серверам.
    trace = [
        ToolCall(name="gitflic.count_public_projects", arguments={"query": "docs"}, server="gitflic"),
        ToolCall(name="pipeline.search", arguments={"query": "docs"}, server="pipeline"),
        ToolCall(name="pipeline.summarize", arguments={"items": []}, server="pipeline"),
        ToolCall(name="pipeline.saveToFile", arguments={"content": "x"}, server="pipeline"),
        ToolCall(name="report.record_run", arguments={"report_path": "x"}, server="report"),
        ToolCall(name="report.build_report_index", arguments={}, server="report"),
    ]
    results: list[ToolResult] = []
    for call in trace:
        server, _, bare = call.name.partition(".")
        payload = orchestrator.runtime(server).call(bare, call.arguments)
        payload.server = server
        results.append(payload)

    problems = verify_agent_flow(
        trace, results, required_servers=["gitflic", "pipeline", "report"]
    )
    print("Флоу:")
    print("\n".join(describe_flow(trace, results)))
    print()

    # Нарушение порядка должно ловиться.
    bad_trace = [
        ToolCall(name="pipeline.summarize", arguments={}, server="pipeline"),
        ToolCall(name="pipeline.search", arguments={}, server="pipeline"),
    ]
    bad_problems = verify_agent_flow(bad_trace, [ToolResult(name="pipeline.summarize", arguments={}, content="{}", server="pipeline"), ToolResult(name="pipeline.search", arguments={}, content="{}", server="pipeline")])
    if not bad_problems:
        errors.append("Проверка не заметила обработку раньше получения данных.")

    if errors or problems:
        print("Проблемы:")
        for problem in errors + problems:
            print(f"  - {problem}")
        return 1
    print("Роутинг и порядок вызовов корректны: 3 сервера задействованы, "
          "порядок функционально верный.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
