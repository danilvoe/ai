#!/usr/bin/env python3
"""Day 11: модель памяти агента — три отдельных слоя.

Агент с явной моделью памяти разделяет информацию на три слоя, каждый хранится
отдельно, а то, что попадает в каждый слой, выбирается явно:

- **Краткосрочная (short-term)** — текущий диалог: сообщения, которыми обменялись
  пользователь и агент в этой сессии.
- **Рабочая (working)** — данные текущей задачи: то, что ввёл пользователь,
  промежуточные результаты, ограничения именно этой задачи. Слой живёт в рамках
  сессии и очищается при смене задачи (``clearworking``).
- **Долговременная (long-term)** — профиль, решения и знания, которые агент
  сохраняет в отдельный файл и помнит даже в новой сессии.

Сценарий показывает на живом примере, какие данные попадают в каждый слой, как
это влияет на запрос к модели (блоки памяти добавляются как system-сообщения) и
как агент отвечает, когда часть слоёв заполнена или очищена.

Команды (не являются репликами к модели, начинаются со ``/``):
- ``/memory``                              — показать содержимое всех слоёв памяти.
- ``/remember short <ключ> = <значение>``  — добавить сообщение в диалог (краткосрочная).
- ``/remember working <ключ> = <значение>``— записать данные текущей задачи (рабочая).
- ``/remember <profile|decision|knowledge> <ключ> = <значение>`` — долговременная.
- ``/forget <категория> <ключ>``           — удалить запись долговременной памяти.
- ``/clearworking``                        — очистить рабочую память (смена задачи).
- ``/memory_reset``                        — стереть долговременную память целиком.
- ``/context``                             — показать, что реально уйдёт в модель.
- ``/help`` / ``/exit``                    — помощь / завершить.

По умолчанию офлайн-режим (ответы имитируются локально), флаг --online шлёт
реальные запросы к API.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.memory import LONG_TERM_KINDS, MemoryLayers
from agent.tokens import estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: считает токены локально и отражает память в ответе."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        # Показываем, какие блоки памяти (system) и сколько сообщений ушло в модель.
        system_blocks = [m for m in messages if m.get("role") == "system"]
        memory_lines = []
        for block in system_blocks:
            for line in block["content"].splitlines():
                if line.startswith("- "):
                    memory_lines.append(line)
        seen = [m for m in messages if m.get("role") == "user"]
        header = ""
        if memory_lines:
            header = "Агент видит память: " + "; ".join(memory_lines) + ". "
        content = header + f"Получено сообщений от пользователя: {len(seen)}"
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


def profile_path_from_config(config: dict) -> Path:
    """Путь к файлу долговременной памяти из конфигурации (или по умолчанию)."""
    section = config.get("memory", {}) or {}
    path = section.get("profile_path")
    if path:
        return Path(path)
    return Path(__file__).resolve().parent.parent / "history" / "profile.json"


def new_agent(config: dict, offline: bool) -> tuple[Agent, Path]:
    """Создаёт агента с моделью памяти; долговременная память живёт в постоянном файле."""
    tmp = tempfile.TemporaryDirectory()
    profile_path = profile_path_from_config(config)
    conversation = SessionStore(Path(tmp.name)).create()
    memory = MemoryLayers(conversation, long_term_path=profile_path)
    agent = Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=None,
        memory=memory,
    )
    return agent, profile_path


def show_memory(agent: Agent) -> None:
    """Печатает содержимое всех слоёв памяти."""
    print("=" * 60)
    print(agent.memory.summarize())
    print("=" * 60)


def show_context(agent: Agent) -> None:
    """Печатает сообщения, которые реально уйдут в модель (с блоками памяти)."""
    print("Контекст, отправляемый в модель:")
    for message in agent.context_messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            print(f"  [system] {content.splitlines()[0][:90]}...")
        else:
            print(f"  [{role}] {content[:90]}")
    print()


def demo(
    config: dict,
    offline: bool,
    turns: list[str],
    long_term_path: Path | None = None,
) -> None:
    """Прогоняет сценарий и в конце показывает, что в каком слое осталось."""
    tmp = tempfile.TemporaryDirectory()
    profile_path = long_term_path or (Path(tmp.name) / "profile.json")
    conversation = SessionStore(Path(tmp.name)).create()
    memory = MemoryLayers(conversation, long_term_path=profile_path)
    agent = Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=None,
        memory=memory,
    )

    print("Запуск агента с явной моделью памяти (3 слоя).")
    show_memory(agent)

    for i, request in enumerate(turns, start=1):
        print(f"\n--- Ход {i}: {request}")
        if request.startswith("/memory"):
            show_memory(agent)
            continue
        if request == "/context":
            show_context(agent)
            continue
        if request.startswith("/remember "):
            command = request[len("/remember "):].strip()
            layer, _, rest = command.partition(" ")
            key, _, value = rest.partition("=")
            key, value = key.strip(), value.strip()
            if layer in ("short",):
                memory.remember_short_term("user", f"{key} = {value}")
                print(f"  -> добавлено в диалог (краткосрочная): {key} = {value}")
            elif layer == "working":
                memory.remember_working(key, value)
                print(f"  -> записано в рабочую память: {key} = {value}")
            elif layer in LONG_TERM_KINDS:
                memory.remember_long_term(layer, key, value)
                print(f"  -> записано в долговременную память ({layer}): {key} = {value}")
            show_context(agent)
            continue
        if request == "/clearworking":
            memory.working.clear()
            print("  -> рабочая память очищена")
            show_context(agent)
            continue

        reply = agent.ask(request)
        print(f"  Агент: {reply.content}")

    print("\n=== Что осталось в каждом слое после сценария ===")
    show_memory(agent)
    print(f"Файл долговременной памяти: {profile_path}")


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам решает, что и в какой слой сохранять."""
    agent, profile_path = new_agent(config, offline)
    print("Модель памяти включена (краткосрочная / рабочая / долговременная).")
    print("Команды со слэшем: /help /memory /context /remember /forget /clearworking /memory_reset")
    show_memory(agent)
    while True:
        try:
            line = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        if not line.startswith("/"):
            try:
                reply = agent.ask(line)
            except Exception as exc:  # noqa: BLE001 — офлайн/онлайн общая обработка
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            print(f"Агент: {reply.content}")
            continue

        cmd = line[1:].strip()
        verb = cmd.split(maxsplit=1)[0].lower()
        if verb in ("help", "?"):
            print(
                "/help  /memory  /context  /clearworking  /memory_reset\n"
                "/remember <слой> <ключ> = <значение>  /forget <категория> <ключ>\n"
                "слои: short | working | profile | decision | knowledge\n"
            )
            continue
        if verb in ("memory", "память"):
            show_memory(agent)
            continue
        if verb in ("context", "контекст"):
            show_context(agent)
            continue
        if verb in ("clearworking", "clear_working", "новая_задача"):
            agent.memory.working.clear()
            print("Рабочая память очищена.\n")
            continue
        if verb in ("memory_reset", "стереть"):
            agent.memory.long_term.clear()
            print("Долговременная память стёрта.\n")
            continue
        if verb in ("remember", "запомни"):
            rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
            layer, _, right = rest.partition(" ")
            key, _, value = right.partition("=")
            key, value = key.strip(), value.strip()
            if layer in ("short",):
                agent.memory.remember_short_term("user", f"{key} = {value}")
                print(f"Добавлено в диалог: {key} = {value}\n")
            elif layer == "working":
                agent.memory.remember_working(key, value)
                print(f"Записано в рабочую память: {key} = {value}\n")
            elif layer in LONG_TERM_KINDS:
                agent.memory.remember_long_term(layer, key, value)
                print(f"Записано в долговременную память ({layer}): {key} = {value}\n")
            else:
                print(f"Неизвестный слой: {layer}. "
                      f"Допустимо: short, working, {', '.join(LONG_TERM_KINDS)}.\n")
            continue
        if verb in ("forget", "забыть"):
            parts = cmd.split(None, 2)
            if len(parts) < 3:
                print("Использование: /forget <категория> <ключ>\n")
                continue
            kind, key = parts[1], parts[2]
            if agent.memory.long_term.forget(kind, key):
                print(f"Удалено ({kind}): {key}\n")
            else:
                print(f"Не найдено ({kind}): {key}\n")
            continue
        print(f"Неизвестная команда: /{verb}. Введите /help.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Модель памяти агента: 3 слоя")
    parser.add_argument("--online", action="store_true", help="реальные запросы к API")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="прогнать фиксированный сценарий вместо живого ввода",
    )
    args = parser.parse_args()

    config = load_config()
    if args.demo:
        demo(
            config,
            offline=not args.online,
            turns=[
                "Привет! Меня зовут Иван, я тестирую модели.",
                "/remember profile имя = Иван",
                "/remember working текущая_задача = сравнение моделей",
                "/remember working лимит_бюджета = 5 долларов",
                "/remember decision подход = предпочитать слабую модель",
                "Что ты знаешь обо мне и о задаче?",
                "/context",
                "/clearworking",
                "Что ты помнишь обо мне после смены задачи?",
                "/context",
            ],
        )
    else:
        run_interactive(config, offline=not args.online)


if __name__ == "__main__":
    main()
