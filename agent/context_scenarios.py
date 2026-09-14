#!/usr/bin/env python3
"""Day 10: стратегии управления контекстом — живое сравнение без хардкода.

Запускает 4 агента (по одному на каждую стратегию и один без стратегии) и
прогоняет через них один и тот же ввод пользователя в реальном времени. Для
каждой реплики показывается ответ агента, размер контекста и накопленные
токены — так видно, как стратегии по-разному управляют контекстом.

Стратегии:
1. **Sliding Window** — храним только последние N сообщений, остальное отбрасываем.
2. **Sticky Facts (Key-Value Memory)** — важные данные выносим в блок ``facts``
   (ключ-значение), который отправляется вместе с последними N сообщениями.
3. **Branching** — checkpoint, независимые ветки, переключение между ними.

Команды (не являются репликами):
- ``checkpoint``        — сохранить точку ветвления (только у Branching).
- ``branch <имя>``      — создать и активировать ветку (Branching).
- ``switch <имя>``      — переключить активную ветку (Branching).
- ``facts``             — показать накопленные липкие факты (Sticky Facts).
- ``status``            — сводка по всем стратегиям (контекст, токены, стоимость).
- ``exit`` / ``quit``   — завершить.

По умолчанию работает в офлайн-режиме (ответы имитируются локально), флаг
--online отправляет реальные запросы к API.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.context import (
    BranchingStrategy,
    ContextStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.tokens import ContextOverflowError, estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: считает токены локально, не отправляя запросы в сеть."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
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


def new_agent(config: dict, offline: bool, strategy: ContextStrategy | None) -> Agent:
    """Создаёт агента для одной стратегии в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    return Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=strategy,
    )


class Runner:
    """Держит по одному агенту на каждую стратегию и прогоняет общий ввод."""

    def __init__(self, config: dict, offline: bool, window: int) -> None:
        self._offline = offline
        self._agents: dict[str, Agent] = {
            "БЕЗ стратегии": new_agent(config, offline, None),
            "Sliding Window": new_agent(config, offline, SlidingWindowStrategy(size=window)),
            "Sticky Facts": new_agent(
                config, offline, StickyFactsStrategy(window_size=window)
            ),
            "Branching": new_agent(config, offline, BranchingStrategy()),
        }

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents

    def handle_command(self, command: str) -> bool:
        """Обрабатывает служебную команду. Возвращает True, если это команда."""
        word = command.strip().split(maxsplit=1)
        verb = word[0].lower()
        rest = word[1].strip() if len(word) > 1 else ""

        if verb in ("checkpoint", "чекпоинт"):
            conv = self._agents["Branching"].conversation
            count = conv.checkpoint()
            print(f"  Branching: checkpoint сохранён ({count} сообщений в общей части).\n")
            return True
        if verb in ("branch", "ветв", "ветка"):
            conv = self._agents["Branching"].conversation
            if not rest:
                print("  Укажите имя ветки: branch <имя>\n")
            elif not conv.checkpoint:
                print("  Сначала сохраните checkpoint.\n")
            elif conv.create_branch(rest):
                print(f"  Branching: создана и активирована ветка '{rest}'.\n")
            else:
                print(f"  Branching: не удалось создать ветку '{rest}'.\n")
            return True
        if verb in ("switch", "переключ", "переключить"):
            conv = self._agents["Branching"].conversation
            if not rest:
                print("  Укажите имя ветки: switch <имя>\n")
            elif conv.switch_branch(rest):
                print(f"  Branching: активна ветка '{rest}'.\n")
            else:
                print(f"  Branching: ветка '{rest}' не найдена.\n")
            return True
        if verb in ("facts", "факт", "факты"):
            facts = self._agents["Sticky Facts"].conversation.facts
            print("  Липкие факты (Sticky Facts):")
            if facts:
                for key, value in facts.items():
                    print(f"    - {key}: {value}")
            else:
                print("    (пока пусто)")
            print()
            return True
        if verb in ("status", "сводка", "итог"):
            self.print_status()
            return True
        return False

    def ask_all(self, user_request: str) -> None:
        """Отправляет реплику всем агентам и печатает сравнение."""
        print(f"\nВы: {user_request}")
        for title, agent in self._agents.items():
            try:
                reply = agent.ask(user_request)
            except ContextOverflowError as exc:
                print(f"  [{title}] Ошибка: {exc}")
                continue
            print(f"  --- {title} ---")
            print(f"    Ответ: {reply.content}")
            print(f"    Контекст: {agent.context_tokens} токенов, "
                  f"история: {len(agent.history)} сообщений")
            print(f"    Накоплено за сессию: {agent.usage.total_tokens} токенов, "
                  f"стоимость: ${agent.usage.total_cost:.6f}"
                  if agent.usage.total_cost is not None else
                  f"    Накоплено за сессию: {agent.usage.total_tokens} токенов")
            print()

    def print_status(self) -> None:
        """Сводка по всем стратегиям: контекст, токены, стоимость."""
        print("\n=== Сводка по стратегиям ===")
        for title, agent in self._agents.items():
            cost = agent.usage.total_cost
            cost_str = "н/д" if cost is None else f"${cost:.6f}"
            conv = agent.conversation
            branch_info = ""
            if conv.has_branches:
                branch_info = f", активна ветка '{conv.active_branch}'"
            print(f"  {title}:")
            print(f"    контекст: {agent.context_tokens} токенов, "
                  f"история: {len(agent.history)} сообщений{branch_info}")
            print(f"    токены за сессию (вход/всего): "
                  f"{agent.usage.total_request_tokens}/{agent.usage.total_tokens}, "
                  f"стоимость: {cost_str}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online", action="store_true",
        help="отправлять реальные запросы к API вместо локальной имитации",
    )
    parser.add_argument(
        "--window", type=int, default=6,
        help="размер окна для Sliding Window и Sticky Facts (по умолчанию 6)",
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

    offline = not args.online
    print(f"Режим: {'офлайн (имитация)' if offline else 'онлайн (реальный API)'}")
    print(f"Окно: {args.window} сообщений для Sliding Window и Sticky Facts.")
    print("Вводите реплики — они прогоняются через все стратегии сразу.")
    print("Команды: checkpoint, branch <имя>, switch <имя>, facts, status, exit.\n")

    runner = Runner(config, offline, args.window)

    while True:
        try:
            line = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("До свидания!")
            break
        if runner.handle_command(line):
            continue
        runner.ask_all(line)


if __name__ == "__main__":
    main()
