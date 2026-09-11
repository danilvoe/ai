#!/usr/bin/env python3
"""Day 8: считаем токены и смотрим, как они влияют на поведение агента.

Запускает три сценария и показывает, как растут токены и стоимость по мере
диалога и что происходит, когда контекст превышает лимит модели.

По умолчанию работает в офлайн-режиме (ответы модели имитируются локально,
чтобы не тратить токены). Флаг --online отправляет реальные запросы к API.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.tokens import ContextOverflowError, estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: считает токены локально, не отправляя запросы в сеть."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages: list[dict]) -> Completion:
        last = messages[-1]["content"]
        content = "Имитация ответа модели на сообщение: " + last[:120]
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


def _new_agent(config: dict, offline: bool, max_context_tokens: int | None = None):
    """Создаёт агента в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    agent = Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=max_context_tokens,
    )
    return agent, conversation


def _format_turn(agent: Agent) -> str:
    last = agent.usage.turns[-1]
    return (
        f"    токены хода: {last.total_tokens} "
        f"(история {last.history_tokens}, запрос {last.request_tokens}, "
        f"ответ {last.response_tokens or 0})"
    )


def _format_total(agent: Agent) -> str:
    cost = agent.usage.total_cost
    cost_str = "н/д" if cost is None else f"${cost:.6f}"
    return (
        f"    за всю сессию: {agent.usage.total_tokens} токенов "
        f"(запросы {agent.usage.total_request_tokens}, ответы "
        f"{agent.usage.total_response_tokens}), стоимость {cost_str}"
    )


def scenario_short(config: dict, offline: bool) -> None:
    print("=== 1. Короткий диалог (2 реплики) ===")
    agent, _ = _new_agent(config, offline)
    for message in ("Привет, как дела?", "Расскажи про ОС Аврора в двух словах."):
        agent.ask(message)
        print(_format_turn(agent))
    print(_format_total(agent))
    print()


def scenario_long(config: dict, offline: bool) -> None:
    print(f"=== 2. Длинный диалог ({config.get('long_turns', 8)} реплик) ===")
    agent, _ = _new_agent(config, offline)
    turns = config.get("long_turns", 8)
    for index in range(1, turns + 1):
        agent.ask(f"Вопрос №{index}: детально объясни понятие '{'пример' * 5}'.")
        print(_format_turn(agent))
    print(_format_total(agent))
    print()


def scenario_overflow(config: dict, offline: bool) -> None:
    print("=== 3. Диалог, превышающий лимит модели ===")
    limit = config.get("overflow_limit_tokens", 200)
    agent, conversation = _new_agent(config, offline, max_context_tokens=limit)
    print(f"    лимит контекста: {limit} токенов")
    filler = "Длинный фрагмент истории. " * 8
    for _ in range(4):
        conversation.append("user", filler)
        conversation.append("assistant", filler)
    print(f"    история уже занимает {agent.context_tokens} токенов")

    try:
        agent.ask("Добавь ещё один вопрос")
        print("    ...запрос отправлен (лимит не превышен)")
    except ContextOverflowError as exc:
        print("    что ломается:")
        print(f"      -> {exc}")
    print(_format_total(agent))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online",
        action="store_true",
        help="отправлять реальные запросы к API вместо локальной имитации",
    )
    parser.add_argument(
        "--config", default=None, help="путь к config.json (по умолчанию config.one.json)"
    )
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config)) if args.config else load_config()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    scenario_short(config, not args.online)
    scenario_long(config, not args.online)
    scenario_overflow(config, not args.online)

    if not args.online:
        print("Режим: офлайн (имитация). Запустите с --online для реальных запросов к API.")


if __name__ == "__main__":
    main()
