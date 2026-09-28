#!/usr/bin/env python3
"""Day 16: подключение к MCP и получение списка инструментов.

Минимальный MCP-клиент на официальном SDK ``mcp`` (транспорт Streamable HTTP).
Он делает ровно то, что требуется в задаче:

1. устанавливает MCP-соединение (``streamablehttp_client`` + ``ClientSession``);
2. выполняет ``initialize`` и получает описание сервера;
3. запрашивает ``tools/list`` и выводит список доступных инструментов.

По умолчанию используется публичный сервер документации ОС Аврора
``https://developer.auroraos.ru/api/mcp``. Сервер требует заголовок
``User-Agent`` как у браузера, поэтому он подставляется автоматически
(и может быть переопределён флагом ``--user-agent``).

Установка SDK::

    pip install mcp

Запуск::

    python3 -m agent.mcp_client
    python3 -m agent.mcp_client --url https://developer.auroraos.ru/api/mcp
    python3 -m agent.mcp_client --json
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_MCP_URL = "https://developer.auroraos.ru/api/mcp"

# Сервер отдаёт ответ только браузерным клиентам, поэтому подставляем
# обычный User-Agent браузера.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


@dataclass
class McpTool:
    """Инструмент, объявленный MCP-сервером."""

    name: str
    description: str
    input_schema: dict


@dataclass
class McpConnection:
    """Результат подключения: сервер и его инструменты."""

    server_name: str
    server_version: str
    protocol_version: str
    instructions: str
    tools: list[McpTool]


async def connect_and_list_tools(
    url: str = DEFAULT_MCP_URL,
    user_agent: str = BROWSER_USER_AGENT,
    timeout: float = 30.0,
) -> McpConnection:
    """Устанавливает MCP-соединение и возвращает список инструментов.

    Это минимальный код подключения: открываем Streamable HTTP-сессию,
    инициализируем её и запрашиваем ``tools/list``.
    """
    headers = {"User-Agent": user_agent}
    # terminate_on_close=False: сервер не поддерживает DELETE-завершение сессии
    # и отвечает 403, поэтому закрываем соединение без этого запроса.
    async with streamablehttp_client(
        url, headers=headers, timeout=timeout, terminate_on_close=False
    ) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            tools = [
                McpTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                )
                for tool in listed.tools
            ]
            return McpConnection(
                server_name=init.serverInfo.name,
                server_version=init.serverInfo.version,
                protocol_version=init.protocolVersion,
                instructions=init.instructions or "",
                tools=tools,
            )


def print_connection(url: str, connection: McpConnection) -> None:
    """Печатает состояние соединения и список инструментов."""
    print("=" * 62)
    print("MCP: подключение и список инструментов")
    print("=" * 62)
    print(f"URL сервера     : {url}")
    print(f"Протокол        : {connection.protocol_version}")
    print(f"Сервер          : {connection.server_name} {connection.server_version}")
    print("Соединение      : установлено")
    print(f"Инструментов    : {len(connection.tools)}")
    if connection.instructions:
        print(f"\nИнструкции сервера:\n  {connection.instructions}\n")
    print("Доступные инструменты:")
    for index, tool in enumerate(connection.tools, start=1):
        print(f"\n  {index}. {tool.name}")
        if tool.description:
            print(f"     {tool.description}")
        properties = tool.input_schema.get("properties") or {}
        required = set(tool.input_schema.get("required") or [])
        if properties:
            params = ", ".join(
                f"{name}{'*' if name in required else ''}" for name in properties
            )
            print(f"     параметры: {params}  (* — обязательный)")
    print()


def verify_connection(connection: McpConnection) -> list[tuple[str, bool]]:
    """Проверяет, что соединение установлено, а инструменты вернулись корректно."""
    checks = [
        ("сервер сообщил имя и версию", bool(connection.server_name)),
        ("протокол согласован", bool(connection.protocol_version)),
        ("список инструментов не пуст", len(connection.tools) > 0),
        (
            "у каждого инструмента есть имя",
            all(tool.name for tool in connection.tools),
        ),
        (
            "у каждого инструмента есть схема",
            all(isinstance(tool.input_schema, dict) for tool in connection.tools),
        ),
    ]
    return checks


def connection_as_dict(connection: McpConnection) -> dict:
    """Представление результата для машинного вывода (JSON)."""
    return {
        "server": {
            "name": connection.server_name,
            "version": connection.server_version,
        },
        "protocol_version": connection.protocol_version,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in connection.tools
        ],
    }


def load_mcp_config() -> dict:
    """Читает секцию ``mcp`` конфигурации, если она доступна."""
    try:
        from agent.config import load_config

        return load_config().get("mcp", {})
    except Exception:  # noqa: BLE001 — конфиг не обязателен для этого клиента
        return {}


def main() -> None:
    mcp_config = load_mcp_config()
    parser = argparse.ArgumentParser(
        description="Подключение к MCP-серверу и вывод списка инструментов"
    )
    parser.add_argument(
        "--url",
        default=mcp_config.get("url", DEFAULT_MCP_URL),
        help="адрес MCP-сервера",
    )
    parser.add_argument(
        "--user-agent",
        default=mcp_config.get("user_agent", BROWSER_USER_AGENT),
        help="User-Agent для запроса (по умолчанию браузерный)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=mcp_config.get("timeout", 30.0),
        help="таймаут, сек",
    )
    parser.add_argument(
        "--json", action="store_true", help="вывести список инструментов в JSON"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="проверить соединение и корректность списка инструментов",
    )
    args = parser.parse_args()

    try:
        connection = asyncio.run(
            connect_and_list_tools(args.url, args.user_agent, args.timeout)
        )
    except Exception as exc:  # noqa: BLE001 — наглядная ошибка подключения
        print(f"Не удалось подключиться к MCP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.check:
        checks = verify_connection(connection)
        for label, ok in checks:
            print(f"[{'OK' if ok else 'FAIL'}] {label}")
        failed = [label for label, ok in checks if not ok]
        if failed:
            print(f"\nПроверка не пройдена: {', '.join(failed)}", file=sys.stderr)
            raise SystemExit(1)
        print(f"\nВсё в порядке: подключение установлено, "
              f"инструментов — {len(connection.tools)}.")
    elif args.json:
        print(json.dumps(connection_as_dict(connection), ensure_ascii=False, indent=2))
    else:
        print_connection(args.url, connection)


if __name__ == "__main__":
    main()
