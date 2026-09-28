#!/usr/bin/env python3
"""Day 17: первый MCP-инструмент — публичные проекты GitFlic.

Сценарий показывает полный цикл работы с собственным MCP-сервером
(``mcp/gitflic_server.py``):

1. **Регистрация инструмента** — клиент подключается к MCP-серверу и получает
   список инструментов (``tools/list``): имя, описание и входные параметры.
2. **Описание входных параметров** — печатается JSON-схема инструмента
   ``get_public_projects`` (query / page / size).
3. **Прямой вызов и результат** — инструмент вызывается напрямую
   (``tools/call``), возвращая публичные проекты GitFlic.
4. **Вызов агентом** (``--agent``) — агент сам выбирает инструмент под запрос,
   вызывает его через MCP и строит ответ по полученным данным.

API GitFlic требует авторизацию, поэтому нужен access token (scope
``PROJECT_READ``) в переменной окружения ``GITFLIC_TOKEN`` или в секции
``tools`` конфигурации.

Запуск::

    GITFLIC_TOKEN=<token> python3 -m agent.tool_scenarios
    GITFLIC_TOKEN=<token> python3 -m agent.tool_scenarios --query docs --size 5
    GITFLIC_TOKEN=<token> python3 -m agent.tool_scenarios --agent "покажи публичные проекты gitflic"
"""

import argparse
import json
import sys

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import LLMClient
from agent.mcp_tools import runtime_from_config
from agent.tokens import estimate_tokens


def _print_tools(runtime) -> None:
    print("=" * 62)
    print("1. Подключение к MCP-серверу и регистрация инструмента")
    print("=" * 62)
    print(runtime.summarize())
    print()


def _print_schema(runtime, tool_name: str) -> None:
    print("=" * 62)
    print("2. Описание входных параметров инструмента")
    print("=" * 62)
    spec = next((s for s in runtime.list_tools() if s.name == tool_name), None)
    if spec is None:
        print(f"Инструмент {tool_name} не найден на сервере.\n")
        return
    print(f"Инструмент : {spec.name}")
    print(f"Описание   : {spec.description.splitlines()[0] if spec.description else ''}")
    print("Схема входа (inputSchema):")
    print(json.dumps(spec.input_schema, ensure_ascii=False, indent=2))
    print()


def _call_and_print(runtime, tool_name: str, arguments: dict) -> dict | None:
    print("=" * 62)
    print("3. Прямой вызов инструмента и возврат результата")
    print("=" * 62)
    print(f"Вызов: {tool_name} {json.dumps(arguments, ensure_ascii=False)}")
    result = runtime.call(tool_name, arguments)
    if result.structured is not None:
        print(json.dumps(result.structured, ensure_ascii=False, indent=2))
    else:
        print(result.content)
    print()
    return result.structured


def _run_agent(request: str, runtime) -> None:
    print("=" * 62)
    print("4. Вызов инструмента агентом и использование результата")
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
        print("Агент не стал вызывать инструмент (ответ построен без него).")
    else:
        result = agent.last_tool_result
        status = "ошибка" if (result and result.is_error) else "результат получен"
        print(f"MCP-вызов агента: {call.name} {json.dumps(call.arguments, ensure_ascii=False)} — {status}")
        if result is not None:
            projects = (result.structured or {}).get("projects") or []
            print(f"Инструмент вернул проектов: {len(projects)}")
    print(f"\nОтвет агента:\n{reply.content}\n")
    print(f"(токенов в ответе ≈ {estimate_tokens(reply.content)})\n")


def main() -> None:
    config = load_config()
    runtime = runtime_from_config(config)
    parser = argparse.ArgumentParser(
        description="MCP-инструмент get_public_projects: подключение, описание, вызов"
    )
    parser.add_argument("--query", default="", help="поисковый запрос (название проекта)")
    parser.add_argument("--page", type=int, default=0, help="страница (с 0)")
    parser.add_argument("--size", type=int, default=5, help="сколько проектов вернуть")
    parser.add_argument(
        "--agent",
        nargs="?",
        const="Покажи публичные проекты GitFlic",
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

    try:
        _print_tools(runtime)
        _print_schema(runtime, "get_public_projects")
        _call_and_print(
            runtime,
            "get_public_projects",
            {"query": args.query, "page": args.page, "size": args.size},
        )
    except Exception as exc:  # noqa: BLE001 — наглядная ошибка подключения/вызова
        print(f"Ошибка MCP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.agent is not None:
        _run_agent(args.agent, runtime)


if __name__ == "__main__":
    main()
