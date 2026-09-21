#!/usr/bin/env python3
"""Day 13: состояние задачи как конечный автомат.

Состояние задачи больше не свободный текст, а **конечный автомат**: текущий
этап, шаг и ожидаемое действие. Этапы переходят по строгим правилам
``planning → execution → validation → done``, и нельзя перепрыгнуть этап или
пойти назад по недопустимому переходу (например, ``planning → validation``).

Сценарий показывает:

1. Как формализовано состояние задачи (этап / шаг / ожидаемое действие).
2. Полный жизненный цикл: планирование → выполнение → проверка → готово.
3. Что **недопустимые переходы отклоняются** автоматом.
4. Что при неудачной проверке можно вернуться на выполнение
   (``validation → execution``).
5. Что **пауза возможна на любом этапе** и **возобновление идёт с того же
   места без повторных объяснений** — описание и ожидаемое действие остаются
   в состоянии и уходят в модель.

По умолчанию офлайн-режим: ответы имитируются локально, а состояние задачи
подмешивается в запрос. Флаг --online шлёт реальные запросы к API.

Команды (не являются репликами к модели, начинаются со ``/``):
- ``/task``                              — показать этап, шаг, ожидаемое действие.
- ``/task_new <описание>``               — начать новую задачу (этап planning).
- ``/task_next``                         — следующий этап по автомату.
- ``/task_stage <этап>``                 — перейти на конкретный этап.
- ``/task_step <N>`` / ``/task_expected <действие>`` — шаг / ожидание.
- ``/task_pause`` / ``/task_resume``     — пауза / возобновление.
- ``/task_done``                         — перевести в done.
- ``/context``                           — показать, что реально уйдёт в модель.
- ``/help`` / ``/exit``                  — помощь / завершить.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.task_state import TaskStateMachine
from agent.tokens import estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: показывает, какое состояние задачи ушло в модель."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        system_blocks = [m for m in messages if m.get("role") == "system"]
        task_block = next(
            (m["content"] for m in system_blocks if "Текущая задача:" in m["content"]),
            None,
        )
        user_messages = [m for m in messages if m.get("role") == "user"]
        header = ""
        if task_block is not None:
            first_line = task_block.splitlines()[0]
            header = f"Вижу состояние задачи: {first_line} · "
        content = (
            header
            + f"system-блоков: {len(system_blocks)}; "
            + f"сообщений от пользователя: {len(user_messages)}"
        )
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


def new_agent(config: dict, offline: bool, task: TaskStateMachine) -> Agent:
    """Создаёт агента с конечным автоматом задачи в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    return Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=None,
        memory=None,
        personalization=None,
        task=task,
    )


def show_context(agent: Agent) -> None:
    """Печатает сообщения, которые реально уйдут в модель (состояние + история)."""
    print("Контекст, отправляемый в модель:")
    for message in agent.context_messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            print(f"  [system] {content.splitlines()[0][:90]}...")
        else:
            print(f"  [{role}] {content[:90]}")
    print()


def run_scenario(config: dict, offline: bool) -> None:
    """Прогоняет фиксированный сценарий: жизненный цикл задачи + пауза."""
    print("=" * 62)
    print("Состояние задачи как конечный автомат")
    print("=" * 62)
    print(
        "Этапы: planning → execution → validation → done\n"
        "Недопустимые переходы отклоняются; пауза возможна на любом этапе.\n"
    )

    task = TaskStateMachine.begin(
        description="Написать отчёт о продажах за месяц",
        expected_action="уточнить период и данные",
    )
    agent = new_agent(config, offline, task)

    print("--- 1. Начало задачи (planning) ---")
    print(task.summarize())
    print()
    print("Запрос модели (состояние уже уходит в контекст):")
    reply = agent.ask("Начинаем готовить отчёт")
    print(f"  ответ: {reply.content}")
    print()

    print("--- 2. Попытка перейти на выполнение без завершения планирования ---")
    try:
        task.advance()
    except ValueError as exc:
        print(f"  отклонено автоматом: {exc}")
    print()

    print("--- 3. Планирование выполнено -> переход в выполнение (execution) ---")
    task.complete()
    agent.save_task()
    task.advance()
    agent.save_task()
    task.set_step(1)
    agent.save_task()
    task.set_expected_action("собрать данные и оформить отчёт")
    agent.save_task()
    print(task.summarize())
    print()

    print("--- 4. Попытка недопустимого перехода (execution -> planning) ---")
    try:
        task.set_stage("planning")
    except ValueError as exc:
        print(f"  отклонено автоматом: {exc}")
    print()

    print("--- 5. Выполнение завершено -> переход в проверку (validation) ---")
    task.complete()
    agent.save_task()
    task.advance()
    agent.save_task()
    task.set_expected_action("проверить цифры и соответствие данным")
    agent.save_task()
    print(task.summarize())
    print()

    print("--- 6. Пауза на этапе проверки (на любом этапе) ---")
    task.pause()
    agent.save_task()
    print(task.summarize())
    print("Состояние в запросе (пауза не ломает автомат):")
    reply = agent.ask("Поставь на паузу")
    print(f"  ответ: {reply.content}")
    print()

    print("--- 7. Возобновление с того же места ---")
    task.resume()
    agent.save_task()
    print(task.summarize())
    print("После resume описание и ожидание на месте — без повторных объяснений:")
    print("  " + task.system_message().replace("\n", "\n  "))
    print()

    print("--- 8. Проверка провалена -> возврат на выполнение ---")
    try:
        task.set_stage("execution")
        agent.save_task()
        task.set_expected_action("исправить найденные расхождения")
        agent.save_task()
    except ValueError as exc:
        print(f"  отклонено: {exc}")
    print(task.summarize())
    print()

    print("--- 9. Переход из execution в done напрямую запрещён ---")
    try:
        task.set_stage("done")
    except ValueError as exc:
        print(f"  отклонено автоматом: {exc}")
    print()

    print("--- 10. Повторная проверка -> done ---")
    task.complete()
    agent.save_task()
    task.set_stage("validation")
    agent.save_task()
    print(task.summarize())
    print()
    task.complete()
    agent.save_task()
    task.set_stage("done")
    agent.save_task()
    print(task.summarize())
    print()
    print("Готово: задача завершена, переходов дальше нет.")
    reply = agent.ask("Итог по задаче")
    print(f"  ответ: {reply.content}")
    print()

    print("--- Проверка: состояние задачи уходит в контекст ---")
    show_context(agent)


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам управляет конечным автоматом задачи."""
    task = TaskStateMachine()
    agent = new_agent(config, offline, task)

    print("Состояние задачи (конечный автомат) включено.")
    print("Начните: /task_new <описание>, далее /task_complete, /task_next")
    print("Команды со слэшем: /help /task /task_new /task_complete /task_next "
          "/task_stage /task_step /task_expected /task_pause /task_resume "
          "/task_done /task_clear /context")
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
        rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
        if verb in ("help", "?"):
            print(
                "/task  /task_new <описание>  /task_complete  /task_next\n"
                "/task_stage <этап>  /task_step <N>  /task_expected <действие>\n"
                "/task_pause  /task_resume  /task_done  /task_clear  /context  /exit\n"
            )
            continue
        if verb in ("task", "задача", "статус"):
            print(task.summarize() + "\n")
            continue
        if verb in ("task_new", "новая_задача"):
            if not rest:
                print("Укажите описание: task_new <описание>\n")
                continue
            task.reset(description=rest)
            agent.save_task()
            print(task.summarize() + "\n")
            continue
        if verb in ("task_complete", "выполнено", "готово_этап"):
            if task.complete():
                agent.save_task()
                print("Этап выполнен — можно /task_next.\n")
            else:
                print("Этап уже выполнен.\n")
            print(task.summarize() + "\n")
            continue
        if verb in ("task_next", "следующий_этап", "далее"):
            try:
                task.advance()
                agent.save_task()
                print(task.summarize() + "\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("task_stage", "этап"):
            try:
                task.set_stage(rest)
                agent.save_task()
                print(task.summarize() + "\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("task_step", "шаг"):
            try:
                task.set_step(int(rest))
                agent.save_task()
                print(task.summarize() + "\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("task_expected", "ожидание"):
            task.set_expected_action(rest)
            agent.save_task()
            print(task.summarize() + "\n")
            continue
        if verb in ("task_pause", "пауза", "приостановить"):
            task.pause()
            agent.save_task()
            print(task.summarize() + "\n")
            continue
        if verb in ("task_resume", "продолжить", "возобновить"):
            task.resume()
            agent.save_task()
            print(task.summarize() + "\n")
            continue
        if verb in ("task_done", "готово", "завершить"):
            try:
                task.set_stage("done")
                agent.save_task()
                print(task.summarize() + "\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("task_clear", "сбросить"):
            task.reset()
            agent.save_task()
            print("Состояние сброшено.\n")
            continue
        if verb in ("context", "контекст"):
            show_context(agent)
            continue
        print(f"Неизвестная команда: /{verb}. Введите /help.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Состояние задачи как конечный автомат")
    parser.add_argument("--online", action="store_true", help="реальные запросы к API")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="прогнать фиксированный сценарий вместо живого ввода",
    )
    args = parser.parse_args()

    config = load_config()
    if args.demo:
        run_scenario(config, offline=not args.online)
    else:
        run_interactive(config, offline=not args.online)


if __name__ == "__main__":
    main()
