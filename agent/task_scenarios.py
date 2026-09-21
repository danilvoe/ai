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
from agent.task_state import TaskState, TaskStateMachine
from agent.tokens import estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: показывает состояние задачи и отвечает на проверку этапа.

    На обычные запросы показывает, какое состояние задачи ушло в модель. На
    короткую проверку завершения этапа отвечает «да», если пользователь дал
    содержательное сообщение (данные предоставлены), иначе «нет».
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        # Короткая проверка завершения этапа (маркер в system-сообщении).
        if any(m.get("content") == "проверяешь, выполнено ли ожидаемое действие"
               for m in messages):
            return self._completion_verdict(messages)

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

    def _completion_verdict(self, messages: list[dict]) -> Completion:
        """Возвращает «да», если в запросе пользователя есть данные (не служебное)."""
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        verbose = len(text.split()) > 5  # не пустышка и не команда — есть данные
        verdict = "да" if verbose else "нет"
        return Completion(
            content=verdict,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(verdict),
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
    """Прогоняет фиксированный сценарий: автопродвижение + пауза + валидация."""
    print("=" * 62)
    print("Состояние задачи как конечный автомат (автопродвижение)")
    print("=" * 62)
    print(
        "Этапы: planning → execution → validation → done\n"
        "Автопродвижение: задача сама переходит дальше, когда данные предоставлены.\n"
        "Пауза возможна на любом этапе; недопустимые переходы отклоняются.\n"
    )

    task = TaskStateMachine(
        TaskState.begin(
            description="Написать отчёт о продажах за месяц",
            expected_action="уточнить период и данные",
        ),
        auto_advance=True,
    )
    agent = new_agent(config, offline, task)

    print("--- 1. Начало задачи (planning). Пользователь даёт данные ---")
    print(task.summarize())
    print()
    reply = agent.ask("Нужен отчёт по продажам за март, вот данные: 120, 145, 133.")
    print(f"  ответ: {reply.content}")
    print(f"  состояние после хода: этап = {task.stage} (задача сдвинулась сама)\n")

    print("--- 2. Пользователь даёт ещё данные — выполнение завершено сам ---")
    reply = agent.ask("Добавь в отчёт сводку по регионам и итоговый вывод.")
    print(f"  ответ: {reply.content}")
    print(f"  состояние после хода: этап = {task.stage}\n")

    print("--- 3. Пауза на любом этапе ---")
    task.pause()
    agent.save_task()
    print(task.summarize())
    print()

    print("--- 4. Возобновление с того же места ---")
    task.resume()
    agent.save_task()
    print("После resume описание и ожидание на месте — без повторных объяснений:")
    print("  " + task.system_message().replace("\n", "\n  "))
    print()

    print("--- 5. Недопустимый переход отклоняется ---")
    try:
        task.set_stage("planning")
    except ValueError as exc:
        print(f"  отклонено автоматом: {exc}")
    print()

    print("--- 6. Корректирующий возврат validation -> execution разрешён ---")
    task.set_stage("validation")
    agent.save_task()
    print(task.summarize())
    print()
    task.set_stage("execution")
    agent.save_task()
    task.set_expected_action("исправить найденные расхождения")
    agent.save_task()
    print(task.summarize())
    print()

    print("--- 7. Финал: execution -> validation -> done ---")
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

    print("--- Проверка: состояние задачи уходит в контекст ---")
    show_context(agent)


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам управляет конечным автоматом задачи."""
    task = TaskStateMachine(auto_advance=True)
    agent = new_agent(config, offline, task)

    print("Состояние задачи (конечный автомат) включено, автопродвижение включено.")
    print("Дайте данные — задача сама перейдёт на следующий этап.")
    print("Начните: /task_new <описание>, далее просто пишите данные")
    print("Команды со слэшем: /help /task /task_new /task_complete /task_next "
          "/task_auto /task_stage /task_step /task_expected /task_pause "
          "/task_resume /task_done /task_clear /context")
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
                "/task  /task_new <описание>  /task_complete  /task_next  /task_auto\n"
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
        if verb in ("task_auto", "автопродвижение"):
            task.auto_advance = not task.auto_advance
            agent.save_task()
            state = "включено" if task.auto_advance else "выключено"
            print(f"Автопродвижение: {state}.\n")
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
