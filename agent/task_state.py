"""Состояние задачи как конечный автомат (Finite State Machine).

Задача агента описывается не свободным текстом, а **формализованным
состоянием**: текущий этап, шаг внутри этапа и ожидаемое действие. Состояние
переходит между этапами по строгим правилам — конечному автомату, поэтому
агент всегда знает, на каком этапе находится задача и что делать дальше.

Состояние задачи содержит три поля:

- **Этап** (``stage``) — где сейчас задача: ``planning`` → ``execution`` →
  ``validation`` → ``done``. Это узлы автомата.
- **Текущий шаг** (``step``) — номер шага внутри этапа (или внутри всей задачи).
- **Ожидаемое действие** (``expected_action``) — что агент должен сделать
  сейчас: что уточнить, что проверить, что сдавать.

Этапы образуют автомат с допустимыми переходами (см. :data:`TRANSITIONS`):
нельзя перепрыгнуть этап или пойти назад по недопустимому переходу. Если
проверка провалилась, состояние может вернуться на этап выполнения
(``validation`` → ``execution``) — это тоже описано в автомате.

Отдельно поддерживается **пауза**: приостановить задачу можно на любом этапе,
и это не ломает автомат. При ``pause``/``resume`` состояние сохраняется целиком,
поэтому после возобновления агент продолжает с того же места **без повторных
объяснений** — описание задачи и ожидаемое действие остаются в состоянии.

Состояние задачи подмешивается к запросу к модели как system-сообщение
(см. :meth:`TaskStateMachine.system_message`), поэтому модель видит текущий этап
и ожидаемое действие и не требует пересказывать задачу заново.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Допустимые переходы конечного автомата: этап -> кортеж следующих этапов.
# None ('' ) — задача завершена, переходов нет.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "planning": ("execution",),
    "execution": ("validation",),
    "planning_review": ("execution",),
    "validation": ("done", "execution"),
    "done": (),
}

# Человекочитаемые названия этапов.
STAGE_LABELS = {
    "planning": "Планирование",
    "execution": "Выполнение",
    "validation": "Проверка",
    "done": "Готово",
}

# Этап по умолчанию для новой задачи.
DEFAULT_STAGE = "planning"


def _validate_stage(stage: str) -> str:
    stage = str(stage).strip().lower()
    if stage not in TRANSITIONS:
        raise ValueError(
            f"Неизвестный этап задачи: {stage!r}. "
            f"Допустимо: {', '.join(TRANSITIONS)}."
        )
    return stage


@dataclass
class TaskState:
    """Формализованное состояние задачи: этап, шаг, ожидаемое действие.

    ``paused`` — задача приостановлена на любом этапе. При паузе состояние не
    меняется: ``description`` и ``expected_action`` остаются в поле, поэтому
    возобновление не требует повторных объяснений.
    """

    stage: str = DEFAULT_STAGE
    step: int = 0
    expected_action: str = ""
    description: str = ""
    paused: bool = False
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = _validate_stage(self.stage)

    @classmethod
    def begin(cls, description: str, expected_action: str = "") -> "TaskState":
        """Создаёт состояние только что начатой задачи (этап ``planning``)."""
        return cls(
            stage=DEFAULT_STAGE,
            step=0,
            expected_action=expected_action or "уточнить цель и составить план",
            description=description,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        """Восстанавливает состояние из словаря (например, из JSON-сессии)."""
        data = data or {}
        return cls(
            stage=str(data.get("stage", DEFAULT_STAGE)),
            step=int(data.get("step", 0) or 0),
            expected_action=str(data.get("expected_action", "")),
            description=str(data.get("description", "")),
            paused=bool(data.get("paused", False)),
            log=[str(item) for item in (data.get("log") or [])],
        )

    def to_dict(self) -> dict:
        """Сериализует состояние в словарь для сохранения в JSON."""
        return {
            "stage": self.stage,
            "step": self.step,
            "expected_action": self.expected_action,
            "description": self.description,
            "paused": self.paused,
            "log": list(self.log),
        }

    def _record(self, event: str) -> None:
        self.log.append(event)

    def advance(self) -> str:
        """Переводит задачу на следующий этап (по автомату)."""
        next_stages = TRANSITIONS[self.stage]
        if not next_stages:
            raise ValueError(
                f"Задача уже завершена (этап {self.stage!r}), переходить некуда."
            )
        previous = self.stage
        self.stage = next_stages[0]
        self._record(f"{previous} -> {self.stage}")
        return self.stage

    def set_stage(self, stage: str) -> str:
        """Явно переводит задачу на указанный этап, проверяя допустимость.

        Возвращает новый этап. Бросает ValueError при недопустимом переходе.
        """
        target = _validate_stage(stage)
        if target not in TRANSITIONS[self.stage] and target != self.stage:
            raise ValueError(
                f"Недопустимый переход: {self.stage!r} -> {target!r}. "
                f"Допустимо: {', '.join(TRANSITIONS[self.stage]) or 'нет'}."
            )
        if target != self.stage:
            previous = self.stage
            self.stage = target
            self._record(f"{previous} -> {self.stage}")
        return self.stage

    def set_step(self, step: int) -> int:
        """Устанавливает номер текущего шага (только вперёд, не меньше текущего).

        Шаги — линейный прогресс задачи, поэтому откат назад запрещён:
        значение меньше текущего шага отклоняется. Исключение — сброс задачи
        через :meth:`TaskStateMachine.reset`, который начинает с шага 0.
        """
        step = max(0, int(step))
        if step < self.step:
            raise ValueError(
                f"Нельзя откатить шаг назад: текущий {self.step}, "
                f"запрошен {step}. Шаги идут только вперёд."
            )
        self.step = step
        return self.step

    def set_expected_action(self, action: str) -> str:
        """Устанавливает ожидаемое действие."""
        self.expected_action = str(action).strip()
        return self.expected_action

    def pause(self) -> bool:
        """Приостанавливает задачу. Разрешено на любом этапе."""
        if self.paused:
            return False
        self.paused = True
        self._record("pause")
        return True

    def resume(self) -> bool:
        """Возобновляет задачу с того же места (без повторных объяснений)."""
        if not self.paused:
            return False
        self.paused = False
        self._record("resume")
        return True

    def is_paused(self) -> bool:
        """Приостановлена ли задача."""
        return self.paused

    def is_done(self) -> bool:
        """Завершена ли задача."""
        return self.stage == "done"

    def stage_label(self) -> str:
        """Человекочитаемое название текущего этапа."""
        return STAGE_LABELS.get(self.stage, self.stage)

    def system_message(self) -> str:
        """System-сообщение с текущим состоянием задачи (для подстановки в запрос).

        Включает описание, этап, шаг и ожидаемое действие. Если задача
        приостановлена — сообщает об этом. Если состояние пустое (нет описания
        и нет ожидаемого действия), возвращает пустую строку.
        """
        if not self.description and not self.expected_action and not self.step:
            return ""
        parts: list[str] = ["Текущая задача:"]
        if self.description:
            parts.append(f"- Описание: {self.description}")
        parts.append(f"- Этап: {self.stage} ({self.stage_label()})")
        if self.step:
            parts.append(f"- Текущий шаг: {self.step}")
        if self.expected_action:
            parts.append(f"- Ожидаемое действие: {self.expected_action}")
        if self.paused:
            parts.append("- Статус: задача приостановлена (resume — продолжить)")
        return "\n".join(parts)

    def summarize(self) -> str:
        """Краткая человекочитаемая сводка состояния задачи."""
        lines = [
            f"Этап: {self.stage} ({self.stage_label()})",
            f"Шаг: {self.step}",
        ]
        if self.expected_action:
            lines.append(f"Ожидаемое действие: {self.expected_action}")
        if self.description:
            lines.append(f"Описание: {self.description}")
        lines.append("Статус: приостановлена" if self.paused else "Статус: активна")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TaskState(stage={self.stage!r}, step={self.step}, "
            f"paused={self.paused}, expected_action={self.expected_action!r})"
        )


class TaskStateMachine:
    """Конечный автомат состояния задачи.

    Обёртка над :class:`TaskState`: хранит текущее состояние и обеспечивает
    переходы между этапами по правилам автомата. Присоединяется к сессии и
    переживает перезапуск (состояние сохраняется в JSON-файле сессии).
    """

    def __init__(self, state: TaskState | None = None) -> None:
        self._state = state or TaskState()

    @classmethod
    def begin(cls, description: str, expected_action: str = "") -> "TaskStateMachine":
        """Создаёт автомат для только что начатой задачи."""
        return cls(TaskState.begin(description, expected_action))

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskStateMachine":
        """Восстанавливает автомат из словаря (None — пустое состояние)."""
        if not data:
            return cls()
        return cls(TaskState.from_dict(data))

    @property
    def state(self) -> TaskState:
        """Текущее состояние задачи."""
        return self._state

    @property
    def stage(self) -> str:
        """Текущий этап задачи."""
        return self._state.stage

    @property
    def step(self) -> int:
        """Текущий шаг задачи."""
        return self._state.step

    @property
    def expected_action(self) -> str:
        """Ожидаемое действие на текущем шаге."""
        return self._state.expected_action

    @property
    def description(self) -> str:
        """Описание задачи."""
        return self._state.description

    @property
    def is_paused(self) -> bool:
        """Приостановлена ли задача."""
        return self._state.is_paused()

    @property
    def is_done(self) -> bool:
        """Завершена ли задача."""
        return self._state.is_done()

    def has_task(self) -> bool:
        """Есть ли у автомата осмысленная задача (есть описание или этап)."""
        return bool(self._state.description) or bool(self._state.expected_action) \
            or self._state.stage != DEFAULT_STAGE

    def advance(self) -> str:
        """Переводит задачу на следующий этап по автомату."""
        return self._state.advance()

    def set_stage(self, stage: str) -> str:
        """Явно переводит задачу на указанный этап (с проверкой перехода)."""
        return self._state.set_stage(stage)

    def set_step(self, step: int) -> int:
        """Устанавливает номер текущего шага."""
        return self._state.set_step(step)

    def set_expected_action(self, action: str) -> str:
        """Устанавливает ожидаемое действие."""
        return self._state.set_expected_action(action)

    def pause(self) -> bool:
        """Приостанавливает задачу на текущем этапе."""
        return self._state.pause()

    def resume(self) -> bool:
        """Возобновляет задачу с того же места."""
        return self._state.resume()

    def reset(self, description: str = "", expected_action: str = "") -> None:
        """Сбрасывает автомат в начальное состояние новой задачи."""
        self._state = TaskState.begin(description, expected_action)

    def system_message(self) -> str:
        """System-сообщение с текущим состоянием задачи."""
        return self._state.system_message()

    def summarize(self) -> str:
        """Краткая сводка состояния задачи."""
        return self._state.summarize()

    def to_dict(self) -> dict:
        """Сериализует автомат в словарь для сохранения в JSON."""
        return self._state.to_dict()

    def __repr__(self) -> str:
        return f"TaskStateMachine({self._state!r})"
