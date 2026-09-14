#!/usr/bin/env python3
"""Day 10: стратегии управления контекстом — сравниваем 3 стратегии без summary.

Прогоняет один и тот же диалог под разными стратегиями управления контекстом и
сравнивает, как растёт контекст, сколько токенов и стоимости уходит за сессию,
и удерживает ли агент ранние факты (качество ответов). Используются стратегии:

1. **Sliding Window** — храним только последние N сообщений, остальное отбрасываем.
2. **Sticky Facts (Key-Value Memory)** — важные данные выносим в блок ``facts``
   (ключ-значение), который отправляется вместе с последними N сообщениями;
   факты обновляются после каждого сообщения пользователя.
3. **Branching** — сохраняем checkpoint, создаём от него независимые ветки,
   продолжаем беседу в каждой отдельно и переключаемся между ними.

Для сравнения дополнительно прогоняется базовый вариант без стратегии (вся
история уходит в запрос как есть). По умолчанию работает в офлайн-режиме
(ответы имитируются локально), флаг --online отправляет реальные запросы.
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


def new_agent(config: dict, offline: bool, strategy: ContextStrategy | None):
    """Создаёт агента в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    agent = Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=strategy,
    )
    return agent


# Ранние факты, которые должны удерживаться стратегиями до конца диалога.
EARLY_FACTS = [
    "Запомни: проект = Север",
    "Цель: написать статью",
    "Предпочтение: писать кратко",
    "Решение: использовать FastAPI",
]


def dialog_turns(turns: int) -> list[str]:
    """Скрипт диалога: ранние факты + вопросы-наполнители."""
    script = list(EARLY_FACTS)
    script += [
        f"Вопрос №{i}: детально объясни понятие '{'пример' * 5}'."
        for i in range(1, turns - len(EARLY_FACTS) + 1)
    ]
    return script


def run_dialog(config: dict, offline: bool, strategy: ContextStrategy | None, turns: int):
    """Проводит диалог, возвращая агента и пошаговый рост контекста."""
    agent = new_agent(config, offline, strategy)
    growth: list[int] = []
    stopped = False
    for message in dialog_turns(turns):
        try:
            agent.ask(message)
        except ContextOverflowError:
            stopped = True
            break
        growth.append(agent.context_tokens)
    return agent, growth, stopped


def context_contains(agent: Agent, text: str) -> bool:
    """Есть ли текст в сообщениях, отправляемых в модель (для офлайн-проверки)."""
    blob = "\n".join(m.get("content", "") for m in agent.context_messages)
    return text.lower() in blob.lower()


def report(title: str, agent: Agent, growth: list[int]) -> None:
    """Печатает сводку по одной стратегии."""
    cost = agent.usage.total_cost
    cost_str = "н/д" if cost is None else f"${cost:.6f}"
    print(f"--- {title} ---")
    print(f"  сообщений в истории: {len(agent.history)}")
    print(f"  контекст в конце (оценка): {agent.context_tokens} токенов")
    print(f"  максимальный контекст за диалог: {max(growth)} токенов")
    print(f"  токены за сессию (вход):  {agent.usage.total_request_tokens}")
    print(f"  токены за сессию (ответы): {agent.usage.total_response_tokens}")
    print(f"  токены за сессию (всего):  {agent.usage.total_tokens}")
    print(f"  стоимость сессии: {cost_str}")
    print()


def run_quality_probe(agent: Agent, fact: str, offline: bool) -> tuple[bool, str]:
    """Задаёт контрольный вопрос про ранний факт и оценивает сохранность."""
    question = f"Какое кодовое имя было у проекта в начале диалога?"
    try:
        reply = agent.ask(question)
    except ContextOverflowError:
        return False, "не задан (переполнение контекста)"
    if offline:
        return context_contains(agent, fact), "offline (по сохранности факта в контексте)"
    return fact.lower() in reply.content.lower(), "online (по наличию факта в ответе)"


def run_branching_demo(config: dict, offline: bool) -> None:
    """Демонстрирует стратегию Branching: checkpoint, 2 ветки, переключение."""
    agent = new_agent(config, offline, BranchingStrategy())
    conv = agent.conversation
    print("=== Демонстрация стратегии Branching ===")
    for message in EARLY_FACTS[:2]:
        agent.ask(message)
    print(f"  Общая часть: {len(conv.messages)} сообщений, "
          f"контекст {agent.context_tokens} токенов.")

    count = conv.checkpoint()
    print(f"  Checkpoint сохранён: {count} сообщений в общей части.")

    conv.create_branch("кратко")
    agent.ask("Расскажи о проекте кратко, одним абзацем.")
    print(f"  Ветка 'кратко': {len(conv.branches['кратко']['messages'])} сообщений.")

    conv.create_branch("подробно")
    agent.ask("Теперь распиши проект максимально подробно, с деталями.")
    print(f"  Ветка 'подробно': {len(conv.branches['подробно']['messages'])} сообщений.")

    print(f"  Активна ветка: {conv.active_branch} "
          f"(контекст {agent.context_tokens} токенов).")

    conv.switch_branch("кратко")
    probe = "кратко"
    print(f"  Переключаюсь на 'кратко' (контекст {agent.context_tokens} токенов).")
    conv.switch_branch("подробно")
    print(f"  Переключаюсь на 'подробно' (контекст {agent.context_tokens} токенов).")
    print(f"  Ветки независимы: 'кратко' содержит факт о проекте — "
          f"{context_contains(agent, 'Север')}.")
    print(f"  Факт 'проект = Север' сохранён в общей части — "
          f"{context_contains(agent, 'проект = Север')}.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online", action="store_true",
        help="отправлять реальные запросы к API вместо локальной имитации",
    )
    parser.add_argument(
        "--turns", type=int, default=20,
        help="сколько всего реплик в диалоге (по умолчанию 20)",
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
    print(f"Диалог: {args.turns} реплик, окно: {args.window} сообщений.\n")

    strategies = [
        ("БЕЗ стратегии (вся история)", None),
        ("Sliding Window", SlidingWindowStrategy(size=args.window)),
        ("Sticky Facts", StickyFactsStrategy(window_size=args.window)),
        ("Branching", BranchingStrategy()),
    ]

    agents = {}
    for title, strategy in strategies:
        agent, growth, stopped = run_dialog(config, offline, strategy, args.turns)
        agents[title] = agent
        report(title, agent, growth)
        if stopped:
            print(f"  (диалог прервался из-за переполнения контекста)\n")

    print("=== Контрольный вопрос (качество: ранний факт 'проект = Север') ===")
    for title, strategy in strategies:
        agent = agents[title]
        ok, mode = run_quality_probe(agent, "Север", offline)
        print(f"  {title}: {'факт сохранён' if ok else 'факт утерян'} ({mode})")

    print("\n=== Расход токенов на запросы (вход) ===")
    for title, strategy in strategies:
        agent = agents[title]
        print(f"  {title}: {agent.usage.total_request_tokens} токенов")

    print("\n=== Сравнение размеров контекста в конце ===")
    for title, strategy in strategies:
        agent = agents[title]
        print(f"  {title}: {agent.context_tokens} токенов, "
              f"{len(agent.history)} сообщений в истории")

    print("\n" + "=" * 40)
    run_branching_demo(config, offline)


if __name__ == "__main__":
    main()
