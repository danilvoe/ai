"""Day 17: MCP-инструменты в агенте.

Модуль подключает к агенту собственный MCP-сервер (``mcp/gitflic_server.py``)
и позволяет вызывать его инструменты. Используется официальный SDK ``mcp``
с транспортом stdio: клиент запускает сервер отдельным процессом, обменивается
с ним JSON-RPC сообщениями и получает результат инструмента.

Агент вызывает инструмент в два шага (см. ``Agent._maybe_call_tool``):
сначала коротким запросом к LLM выбирает инструмент и аргументы
(``plan_messages`` + ``parse_plan``), затем выполняет вызов через MCP
(``call``) и подмешивает полученный результат в контекст, чтобы модель
построила ответ по реальным данным.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "mcp" / "gitflic_server.py"


@dataclass
class ToolSpec:
    """Инструмент, объявленный MCP-сервером."""

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """Выбранный агентом вызов инструмента: имя и аргументы.

    ``server`` заполняется, когда вызов идёт через оркестратор нескольких
    серверов (Day 20); для одного сервера остаётся ``None``.
    """

    name: str
    arguments: dict
    server: str | None = None


@dataclass
class ToolResult:
    """Результат вызова MCP-инструмента."""

    name: str
    arguments: dict
    content: str
    structured: dict | None = None
    is_error: bool = False
    server: str | None = None


PLAN_PROMPT = """Ты — диспетчер инструментов. Тебе дан список доступных MCP-инструментов
и запрос пользователя. Реши, нужно ли вызвать какой-либо инструмент, чтобы
ответить на запрос.

Ответь СТРОГО одним JSON-объектом без пояснений:
{"tool": "<имя инструмента>", "arguments": {<аргументы>}}

Если ни один инструмент не подходит, ответь:
{"tool": null}

Доступные инструменты:
"""


class McpToolRuntime:
    """Клиент MCP-сервера: список инструментов и их вызов."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._command = command or sys.executable
        if args is None:
            args = [str(DEFAULT_SERVER_SCRIPT)]
        self._args = args
        self._cwd = str(cwd) if cwd else None
        self._env = env or {}
        self._timeout = timeout
        self._specs: list[ToolSpec] | None = None

    @property
    def command(self) -> str:
        return self._command

    @property
    def args(self) -> list[str]:
        return list(self._args)

    def _server_params(self) -> StdioServerParameters:
        # Передаём серверу текущее окружение плюс свои переменные
        # (например, GITFLIC_TOKEN и GITFLIC_API_URL).
        env = dict(os.environ)
        env.update(self._env)
        return StdioServerParameters(
            command=self._command,
            args=list(self._args),
            cwd=self._cwd,
            env=env,
        )

    async def _list_async(self) -> list[ToolSpec]:
        params = self._server_params()
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                return [
                    ToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    for tool in listed.tools
                ]

    async def _call_async(self, name: str, arguments: dict) -> ToolResult:
        params = self._server_params()
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        text_parts = [
            part.text
            for part in result.content
            if getattr(part, "type", None) == "text"
        ]
        content = "\n".join(text_parts).strip()
        structured = getattr(result, "structuredContent", None)
        if structured is None and content:
            # FastMCP может вернуть результат только текстом (JSON-строка) —
            # разбираем её, чтобы агенту было удобнее читать поля.
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    structured = parsed
            except json.JSONDecodeError:
                pass
        if not content and structured is not None:
            content = json.dumps(structured, ensure_ascii=False)
        structured_dict = structured if isinstance(structured, dict) else None
        is_error = bool(getattr(result, "isError", False)) or bool(
            structured_dict and structured_dict.get("error")
        )
        return ToolResult(
            name=name,
            arguments=arguments,
            content=content,
            structured=structured_dict,
            is_error=is_error,
        )

    def list_tools(self, refresh: bool = False) -> list[ToolSpec]:
        """Возвращает инструменты MCP-сервера (с кэшированием)."""
        if self._specs is None or refresh:
            self._specs = asyncio.run(self._list_async())
        return self._specs

    def call(self, name: str, arguments: dict | None = None) -> ToolResult:
        """Вызывает MCP-инструмент по имени и возвращает результат."""
        return asyncio.run(self._call_async(name, arguments or {}))

    def has_tool(self, name: str) -> bool:
        return any(spec.name == name for spec in self.list_tools())

    def summarize(self) -> str:
        """Человекочитаемое описание инструментов (для CLI/сценария)."""
        specs = self.list_tools()
        lines = [
            f"MCP-инструменты ({len(specs)}):",
            f"  сервер : {self._command} {' '.join(self._args)}",
        ]
        for index, spec in enumerate(specs, start=1):
            required = set(spec.input_schema.get("required") or [])
            properties = spec.input_schema.get("properties") or {}
            params = ", ".join(
                f"{name}{'*' if name in required else ''}" for name in properties
            )
            lines.append(f"\n  {index}. {spec.name}")
            if spec.description:
                for line in spec.description.splitlines():
                    lines.append(f"     {line}")
            if params:
                lines.append(f"     параметры: {params}  (* — обязательный)")
        return "\n".join(lines)

    def plan_messages(
        self, user_request: str, results: list[dict] | None = None
    ) -> list[dict]:
        """Сообщения для выбора инструмента и аргументов под запрос.

        Если переданы ``results`` — результаты уже выполненных шагов, они уходят
        в контекст, чтобы модель могла связать данные между вызовами (длинный
        флоу, Day 20).
        """
        specs = self.list_tools()
        if not specs:
            return []
        catalog = []
        for spec in specs:
            catalog.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                }
            )
        prompt = PLAN_PROMPT + json.dumps(catalog, ensure_ascii=False, indent=2)
        if results:
            prompt += (
                "\n\nУже выполненные шаги (используй их результаты):\n"
                + json.dumps(results, ensure_ascii=False, indent=2)
            )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_request},
        ]

    @staticmethod
    def parse_plan(text: str) -> ToolCall | None:
        """Разбирает JSON-план модели. None — инструмент вызывать не нужно."""
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        name = data.get("tool")
        if not name:
            return None
        arguments = data.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ToolCall(name=str(name), arguments=arguments)

    def result_message(self, result: ToolResult) -> dict:
        """System-сообщение с результатом вызова, которое видит основная модель."""
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


def runtime_from_config(config: dict) -> McpToolRuntime | None:
    """Создаёт MCP-рантайм из секции ``tools`` конфигурации (None — выключено)."""
    section = config.get("tools", {}) or {}
    if not section.get("enabled", False):
        return None
    args = section.get("args")
    env: dict[str, str] = {}
    if section.get("gitflic_token"):
        env["GITFLIC_TOKEN"] = str(section["gitflic_token"])
    if section.get("gitflic_api_url"):
        env["GITFLIC_API_URL"] = str(section["gitflic_api_url"])
    cwd = section.get("cwd")
    if cwd:
        cwd = str((PROJECT_ROOT / cwd).resolve()) if not Path(cwd).is_absolute() else cwd
    else:
        # Сервер ищется относительно корня проекта, поэтому запускаем из него.
        cwd = str(PROJECT_ROOT)
    return McpToolRuntime(
        command=section.get("command") or sys.executable,
        args=args if args else [str(DEFAULT_SERVER_SCRIPT)],
        cwd=cwd,
        env=env,
        timeout=float(section.get("timeout", 60)),
    )
