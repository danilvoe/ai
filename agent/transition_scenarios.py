#!/usr/bin/env python3
"""Day 15: контролируемые переходы состояний задачи.

Задача проходит жизненный цикл по строгому конечному автомату
``planning → execution → validation → done``, и ассистент **не может перепрыгнуть
этап**. Переходы контролируются на двух уровнях:

1. **Допустимые состояния и разрешённые переходы** (``TRANSITIONS`` и
   ``STAGE_SCOPE`` в agent/task_state.py): этап ``planning`` не переходит прямо в
   ``validation``, а ``execution`` не переходит в ``done`` без ``validation``.
   Пока текущий этап не отмечен выполненным, переход вперёд запрещён.
2. **Ассистент не может перепрыгнуть этап.** Каждый этап описывает, что сейчас
   *разрешено*, а что *запрещено* (относится к более поздним этапам). Если
   пользователь просит сделать работу следующего этапа (например, реализацию до
   утверждённого плана или финал без проверки) — ассистент **отказывается** и
   удерживает этап. Если сам ассистент выдаёт результат более позднего этапа —
   короткая проверка (``transition_guard_messages`` + ``parse_transition_verdict``)
   заменяет его ответ отказом.

Сценарий показывает:

1. Допустимые состояния и разрешённые переходы автомата.
2. Что нельзя делать реализацию до утверждённого плана.
3. Что нельзя сдавать финал без проверки.
4. Попытку перейти в недопустимое состояние — отклоняется автоматом.
5. Реакцию ассистента (отказ), когда его ответ «перепрыгивает» этап.
6. Корректность продолжения после паузы — задача идёт с того же этапа.

По умолчанию офлайн-режим: ответы имитируются локально. Флаг --online шлёт
реальные запросы к API.

Команды (не являются репликами к модели, начинаются со ``/``):
- ``/task``                              — показать этап, шаг, ожидаемое действие.
- ``/task_new <описание>``               — начать новую задачу (этап planning).
- ``/task_next``                         — следующий этап по автомату.
- ``/task_stage <этап>``                 — перейти на конкретный этап.
- ``/task_strict``                       — вкл/выкл контроль переходов.
- ``/task_pause`` / ``/task_resume``     — пауза / возобновление.
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
from agent.task_state import (
    STAGE_LABELS,
    STAGE_SCOPE,
    TRANSITIONS,
    TaskState,
    TaskStateMachine,
)
from agent.tokens import estimate_messages_tokens, estimate_tokens

# Ключевые слова «перепрыгивания» в ответе ассистента — для проверки перехода.
_JUMP_MARKERS = ("итоговый", "готово, можно сдавать", "выполнил задачу", "сдаю финал")


class FakeClient:
    """Имитирует LLM: показывает состояние задачи и отвечает на проверки.

    На обычные запросы показывает, какое состояние задачи ушло в модель. На
    короткую проверку завершения этапа отвечает «да», если пользователь дал
    содержательное сообщение. На короткую проверку перепрыгивания этапа отвечает
    «НАРУШЕНИЕ», если ответ ассистента содержит маркеры более позднего этапа.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._jump_reply = False

    def make_jump(self) -> None:
        """Включает режим, в котором обычный ответ ассистента «перепрыгивает» этап."""
        self._jump_reply = True

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        if any(m.get("content") == "проверяешь, выполнено ли ожидаемое действие"
               for m in messages):
            return self._completion_verdict(messages)
        if any(m.get("content") == "проверяешь, не перепрыгнул ли ассистент этап"
               for m in messages):
            return self._transition_verdict(messages)
        return self._ordinary_reply(messages)

    def _ordinary_reply(self, messages: list[dict]) -> Completion:
        system_blocks = [m for m in messages if m.get("role") == "system"]
        task_block = next(
            (m["content"] for m in system_blocks if "Текущая задача:" in m["content"]),
            None,
        )
        stage = ""
        if task_block is not None:
            for line in task_block.splitlines():
                if line.startswith("- Этап:"):
                    stage = line.split("(", 1)[-1].rstrip(")")
        header = f"Вижу этап задачи: {stage or '—'}. "
        if self._jump_reply:
            content = header + "Готово, я выполнил задачу, вот итоговый результат."
        else:
            content = header + (
                "Уточняю цель и собираю требования, чтобы составить план."
            )
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )

    def _completion_verdict(self, messages: list[dict]) -> Completion:
        text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        verbose = len(text.split()) > 5
        verdict = "да" if verbose else "нет"
        return Completion(
            content=verdict,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(verdict),
        )

    def _transition_verdict(self, messages: list[dict]) -> Completion:
        prompt = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"),
            "",
        )
        # Ответ ассистента идёт сразу после метки и заканчивается перед
        # следующей инструкцией «Если ответ ассистента...».
        _, _, after = prompt.partition("Ответ ассистента:")
        assistant_reply = after.split("\n\nЕсли", 1)[0].strip()
        jumped = any(marker in assistant_reply.lower() for marker in _JUMP_MARKERS)
        verdict = (
            "НАРУШЕНИЕ: да\nПРИЧИНА: ассистент выдал результат более позднего "
            "этапа, не завершив текущий."
            if jumped
            else "ОК"
        )
        return Completion(
            content=verdict,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(verdict),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


def new_agent(config: dict, offline: bool, task: TaskStateMachine) -> Agent:
    """Создаёт агента с контролируемым жизненным циклом задачи."""
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


def print_states() -> None:
    """Печатает допустимые состояния и разрешённые переходы автомата."""
    print("Допустимые состояния задачи:")
    for stage, label in STAGE_LABELS.items():
        allowed = ", ".join(TRANSITIONS[stage]) or "нет"
        scope = STAGE_SCOPE[stage]
        print(f"  - {stage} ({label}): далее -> [{allowed}]")
        print(f"      разрешено: {scope['allowed']}")
        print(f"      запрещено сейчас: {scope['forbidden']}")
    print()


def run_scenario(config: dict, offline: bool) -> None:
    """Прогоняет фиксированный сценарий контролируемых переходов."""
    print("=" * 62)
    print("Контролируемые переходы состояний задачи")
    print("=" * 62)
    print(
        "Жизненный цикл: planning → execution → validation → done.\n"
        "Ассистент не может перепрыгнуть этап: реализация требует плана,\n"
        "финал требует проверки. Строгий контроль переходов включён.\n"
    )
    print_states()

    task = TaskStateMachine(
        TaskState.begin(
            description="Написать отчёт о продажах за месяц",
            expected_action="уточнить цель и составить план",
        ),
        auto_advance=False,
        strict=True,
    )
    agent = new_agent(config, offline, task)
    print(f"Старт: {task.summarize()}\n")

    print("--- 1. Нельзя делать реализацию до утверждённого плана ---")
    reply = agent.ask("Напиши код и сразу сделай итоговый отчёт.")
    print(f"  пользователь: Напиши код и сразу сделай итоговый отчёт.")
    print(f"  ассистент: {reply.content}")
    print(f"  этап остался: {task.stage} (перепрыгнуть не удалось)\n")

    print("--- 2. Допустимое действие: согласовать план ---")
    reply = agent.ask("Цель: отчёт по продажам за март. Данные: 120, 145, 133.")
    print(f"  ассистент: {reply.content}")
    print(f"  этап остался: {task.stage}\n")

    print("--- 3. Переход в недопустимое состояние отклоняется ---")
    try:
        task.set_stage("validation")
    except ValueError as exc:
        print(f"  отклонено автоматом: {exc}")
    print()

    print("--- 4. План согласован — переход на выполнение разрешён ---")
    task.complete()
    agent.save_task()
    task.set_stage("execution")
    agent.save_task()
    print(task.summarize())
    print()

    print("--- 5. Нельзя сдавать финал без проверки ---")
    reply = agent.ask("Сдай финальный отчёт, без проверки.")
    print(f"  пользователь: Сдай финальный отчёт, без проверки.")
    print(f"  ассистент: {reply.content}")
    print(f"  этап остался: {task.stage}\n")

    print("--- 6. Ассистент сам пытается перепрыгнуть этап -> отказ ---")
    agent._client.make_jump()
    reply = agent.ask("Добавь в отчёт вывод по регионам.")
    print(f"  ассистент (модель хотела сдать финал): {reply.content}")
    print(f"  перепрыгивание поймано: {agent.last_transition_violation}")
    print(f"  этап остался: {task.stage}\n")

    print("--- 7. Пауза и корректное продолжение с того же этапа ---")
    task.pause()
    agent.save_task()
    print("пауза ->")
    print("  " + task.system_message().replace("\n", "\n  "))
    task.resume()
    agent.save_task()
    print("resume -> этап, описание и ожидание на месте:")
    print("  " + task.system_message().replace("\n", "\n  "))
    print()

    print("--- 8. Контролируемый финал: выполнение -> проверка -> done ---")
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
    print("Готово: задача завершена по контролируемым переходам, дальше переходов нет.")

    print("--- Контекст: состояние задачи с зоной разрешённого/запрещённого ---")
    show_context(agent)


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам управляет контролируемым жизненным циклом."""
    task = TaskStateMachine(auto_advance=True, strict=True)
    agent = new_agent(config, offline, task)

    print("Контролируемые переходы включены (strict).")
    print("Попробуйте перепрыгнуть этап — ассистент удержит его.")
    print("Начните: /task_new <описание>, далее просто пишите данные")
    print("Команды со слэшем: /help /task /task_new /task_complete /task_next "
          "/task_auto /task_strict /task_stage /task_step /task_expected "
          "/task_pause /task_resume /task_done /task_clear /context")
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
            if agent.last_transition_violation:
                print("(ассистент отказался: попытка перепрыгнуть этап)")
            continue

        cmd = line[1:].strip()
        verb = cmd.split(maxsplit=1)[0].lower()
        rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
        if verb in ("help", "?"):
            print(
                "/task  /task_new <описание>  /task_complete  /task_next  /task_auto\n"
                "/task_strict  /task_stage <этап>  /task_step <N>\n"
                "/task_expected <действие>  /task_pause  /task_resume\n"
                "/task_done  /task_clear  /context  /exit\n"
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
        if verb in ("task_strict", "контроль_этапов"):
            task.strict = not task.strict
            agent.save_task()
            state = "включён" if task.strict else "выключен"
            print(f"Контроль переходов: {state}.\n")
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
    parser = argparse.ArgumentParser(description="Контролируемые переходы состояний задачи")
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
