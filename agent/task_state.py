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

# Ожидаемое действие по умолчанию для каждого этапа. Подставляется
# автоматически при переходе на этап (если не задано своё через /task_expected).
STAGE_DEFAULT_ACTIONS = {
    "planning": "уточнить цель и составить план",
    "planning_review": "согласовать план с пользователем",
    "execution": "выполнить задачу и оформить результат",
    "validation": "проверить результат на соответствие требованиям",
    "done": "задача завершена",
}

# Критерий «этап выполнен» для каждого этапа. Задаёт проверке, что считать
# завершением этапа, с учётом того, что пользователь мог перескочить этап
# (напр. сразу запросить выполнение, минуя планирование).
STAGE_COMPLETION_CRITERIA = {
    "planning": (
        "этап закрыт, когда цель задачи ясна и есть план, ЛИБО пользователь "
        "сразу предоставил всё нужное для перехода к выполнению (сразу просит "
        "сделать/оформить) — тогда планирование считается выполненным"
    ),
    "planning_review": (
        "этап закрыт, когда план согласован пользователем или пользователь "
        "уже дал задание на выполнение"
    ),
    "execution": (
        "этап закрыт, когда создан финальный результат (деливерабл), "
        "заполненный реальными данными пользователя — не черновик, не шаблон "
        "и не ожидание данных"
    ),
    "validation": (
        "этап закрыт, когда результат проверен и подтверждён (пользователь "
        "одобрил, ошибки исправлены)"
    ),
    "done": "этап закрыт — задача завершена",
}

# Этап по умолчанию для новой задачи.
DEFAULT_STAGE = "planning"

# Промпт короткой проверки: решает LLM, выполнено ли ожидаемое действие.
COMPLETION_PROMPT = (
    "Ты проверяешь, выполнен ли текущий этап задачи.\n"
    "Текущий этап: {stage} ({stage_label}).\n"
    "Ожидаемое действие: {expected_action}.\n"
    "Критерий выполнения этапа: {criterion}.\n\n"
    "Недавний ход диалога (последние сообщения):\n{recent}\n\n"
    "Последний запрос пользователя:\n{user_request}\n\n"
    "Твой последний ответ:\n{assistant_reply}\n\n"
    "Ответь «да», если по критерию этап выполнен. Учитывай весь недавний диалог:\n"
    "- если результат уже выдан и одобрен пользователем, а последний ход лишь "
    "подтверждает это — «да»;\n"
    "- если пользователь сразу перескочил этот этап (сразу просит выполнить "
    "задачу) — для этапа планирования это «да».\n"
    "Если ассистент всё ещё уточняет детали, задаёт вопросы, предложил "
    "шаблон/черновик или ждёт данных — «нет».\n"
    "Ответь ровно одним словом: «да» или «нет». Больше ничего не пиши."
)

_COMPLETION_MARKER = "проверяешь, выполнено ли ожидаемое действие"


def parse_completion_verdict(text: str) -> bool:
    """Разбирает ответ модели на проверку завершения этапа.

    True, если модель ответила «да»/«yes» (ожидаемое действие выполнено).
    """
    token = (text or "").strip().lower()
    return any(word in token for word in ("да", "yes", "выполнено", "готово"))

# Порядок этапов жизненного цикла — для определения, куда ведёт переход.
# Переход "вперёд" (к более позднему этапу) требует, чтобы текущий этап был
# выполнен. "Назад" (корректирующий возврат, напр. validation -> execution при
# провале проверки) разрешён без отметки выполненным.
_STAGE_ORDER = {
    "planning": 0,
    "planning_review": 1,
    "execution": 2,
    "validation": 3,
    "done": 4,
}


def _is_forward(from_stage: str, to_stage: str) -> bool:
    """Является ли переход ``from -> to`` движением вперёд по жизненному циклу."""
    return _STAGE_ORDER[to_stage] > _STAGE_ORDER[from_stage]


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
    completed: bool = False
    log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = _validate_stage(self.stage)

    @classmethod
    def begin(cls, description: str, expected_action: str = "") -> "TaskState":
        """Создаёт состояние только что начатой задачи (этап ``planning``)."""
        return cls(
            stage=DEFAULT_STAGE,
            step=0,
            expected_action=expected_action or STAGE_DEFAULT_ACTIONS[DEFAULT_STAGE],
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
            completed=bool(data.get("completed", False)),
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
            "completed": self.completed,
            "log": list(self.log),
        }

    def _apply_stage_default_action(self) -> None:
        """Подставляет ожидаемое действие по умолчанию для текущего этапа.

        Вызывается при переходе на новый этап, поэтому каждый этап получает
        своё ожидание (planning — план, execution — выполнить, validation —
        проверить). Явное ``/task_expected`` задаёт действие на текущем этапе
        и действует до следующего перехода.
        """
        default = STAGE_DEFAULT_ACTIONS.get(self.stage)
        if default:
            self.expected_action = default

    def _record(self, event: str) -> None:
        self.log.append(event)

    def complete(self) -> bool:
        """Отмечает текущий этап как выполненный (завершён).

        Только после этого допустим переход на следующий этап. Возвращает
        False, если этап уже отмечен выполненным.
        """
        if self.completed:
            return False
        self.completed = True
        self._record(f"stage {self.stage} completed")
        return True

    def advance(self) -> str:
        """Переводит задачу на следующий этап (по автомату).

        Разрешено только если текущий этап отмечен выполненным (см.
        :meth:`complete`). Переход сбрасывает флаг ``completed``: новый этап
        снова нужно отработать и подтвердить.
        """
        if not self.completed:
            raise ValueError(
                f"Этап {self.stage!r} ещё не выполнен. Отметьте его через "
                f"complete(), прежде чем переходить дальше."
            )
        next_stages = TRANSITIONS[self.stage]
        if not next_stages:
            raise ValueError(
                f"Задача уже завершена (этап {self.stage!r}), переходить некуда."
            )
        previous = self.stage
        self.stage = next_stages[0]
        self.completed = False
        self.step += 1
        self._apply_stage_default_action()
        self._record(f"{previous} -> {self.stage}")
        return self.stage

    def set_stage(self, stage: str) -> str:
        """Явно переводит задачу на указанный этап, проверяя допустимость.

        Возвращает новый этап. Бросает ValueError при недопустимом переходе.
        """
        target = _validate_stage(stage)
        # Вперёд можно только после выполнения текущего этапа; корректирующий
        # возврат (напр. validation -> execution при провале проверки) — без него.
        if target != self.stage and _is_forward(self.stage, target) and not self.completed:
            raise ValueError(
                f"Этап {self.stage!r} ещё не выполнен. Отметьте его через "
                f"complete(), прежде чем переходить дальше."
            )
        if target not in TRANSITIONS[self.stage] and target != self.stage:
            raise ValueError(
                f"Недопустимый переход: {self.stage!r} -> {target!r}. "
                f"Допустимо: {', '.join(TRANSITIONS[self.stage]) or 'нет'}."
            )
        if target != self.stage:
            previous = self.stage
            self.stage = target
            self.completed = False
            self.step += 1
            self._apply_stage_default_action()
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
        elif self.completed:
            parts.append("- Статус: этап выполнен (переход на следующий этап разрешён)")
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
        if self.paused:
            lines.append("Статус: приостановлена")
        elif self.completed:
            lines.append("Статус: этап выполнен — можно /task_next")
        else:
            lines.append("Статус: активна (выполните ожидаемое действие, затем /task_complete)")
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

    def __init__(self, state: TaskState | None = None, auto_advance: bool = False) -> None:
        self._state = state or TaskState()
        self._auto_advance = auto_advance

    @classmethod
    def begin(cls, description: str, expected_action: str = "",
              auto_advance: bool = False) -> "TaskStateMachine":
        """Создаёт автомат для только что начатой задачи."""
        return cls(TaskState.begin(description, expected_action), auto_advance)

    @classmethod
    def from_dict(cls, data: dict | None, auto_advance: bool = False) -> "TaskStateMachine":
        """Восстанавливает автомат из словаря (None — пустое состояние)."""
        if not data:
            return cls(auto_advance=auto_advance)
        return cls(TaskState.from_dict(data), auto_advance)

    @property
    def auto_advance(self) -> bool:
        """Автоматически ли продвигать задачу, когда этап выполнен."""
        return self._auto_advance

    @auto_advance.setter
    def auto_advance(self, value: bool) -> None:
        self._auto_advance = bool(value)

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

    @property
    def is_completed(self) -> bool:
        """Выполнен ли текущий этап (можно ли переходить дальше)."""
        return self._state.completed

    def has_task(self) -> bool:
        """Есть ли у автомата осмысленная задача (есть описание или этап)."""
        return bool(self._state.description) or bool(self._state.expected_action) \
            or self._state.stage != DEFAULT_STAGE

    def advance(self) -> str:
        """Переводит задачу на следующий этап по автомату."""
        return self._state.advance()

    def complete(self) -> bool:
        """Отмечает текущий этап выполненным (разрешает переход дальше)."""
        return self._state.complete()

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

    def completion_check_messages(self, user_request: str, assistant_reply: str,
                                  recent_messages: list[dict] | None = None) -> list[dict]:
        """Сообщения короткой проверки: выполнено ли ожидаемое действие этапа.

        Включает не только последний ход, но и несколько недавних сообщений
        диалога, чтобы модель видела ранее выданный результат и его одобрение.
        Возвращает пустой список, если задачи нет или она завершена.
        """
        if not self.has_task() or self._state.is_done():
            return []
        recent = self._format_recent(recent_messages or [])
        content = COMPLETION_PROMPT.format(
            stage=self._state.stage,
            stage_label=self._state.stage_label(),
            expected_action=self._state.expected_action or "не задано",
            criterion=STAGE_COMPLETION_CRITERIA.get(
                self._state.stage, "этап выполнен, когда задача в нём доведена до результата"
            ),
            recent=recent or "—",
            user_request=user_request,
            assistant_reply=assistant_reply,
        )
        return [
            {"role": "system", "content": _COMPLETION_MARKER},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _format_recent(messages: list[dict]) -> str:
        """Сворачивает недавние сообщения в компактную текстовую стенограмму."""
        lines = []
        for message in messages[-6:]:
            role = message.get("role")
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content[:1200]}")
        return "\n".join(lines)

    def summarize(self) -> str:
        """Краткая сводка состояния задачи."""
        return self._state.summarize()

    def to_dict(self) -> dict:
        """Сериализует автомат в словарь для сохранения в JSON."""
        return self._state.to_dict()

    def __repr__(self) -> str:
        return f"TaskStateMachine({self._state!r})"
