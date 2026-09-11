"""Подсчёт токенов и учёт стоимости по мере развития диалога агента."""

from dataclasses import dataclass, field
from math import ceil


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов в тексте (~4 символа на токен).

    Аппроксимация не зависит от конкретного токенизатора модели и
    одинаково ведёт себя для кириллицы и латиницы. Точные значения
    для отправляемого контекста берутся из поля usage ответа API.
    """
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Сумма токенов по всем сообщениям истории диалога."""
    return sum(estimate_tokens(message.get("content", "")) for message in messages)


class ContextOverflowError(RuntimeError):
    """Поднимается, когда диалог превышает контекстный лимит модели."""


@dataclass
class TurnUsage:
    """Учёт токенов и стоимости одного хода (запрос + ответ модели)."""

    request_tokens: int
    history_tokens: int
    response_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None

    @property
    def total_request_tokens(self) -> int:
        """Токены, отправленные в модель: история + текущий запрос."""
        return self.history_tokens + self.request_tokens

    @property
    def total_tokens(self) -> int:
        """Все токены хода: запрос + ответ модели."""
        return self.total_request_tokens + (self.response_tokens or 0)


@dataclass
class UsageReport:
    """Накопительная статистика токенов и стоимости всей сессии."""

    input_price: float | None = None
    output_price: float | None = None
    turns: list[TurnUsage] = field(default_factory=list)

    @property
    def total_request_tokens(self) -> int:
        return sum(turn.total_request_tokens for turn in self.turns)

    @property
    def total_response_tokens(self) -> int:
        return sum(turn.response_tokens or 0 for turn in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_request_tokens + self.total_response_tokens

    @property
    def total_cost(self) -> float | None:
        if self.input_price is None or self.output_price is None:
            return None
        return sum(turn.cost or 0 for turn in self.turns)

    def record(
        self,
        request_tokens: int,
        history_tokens: int,
        response_tokens: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> TurnUsage:
        """Добавляет ход в отчёт и считает его стоимость по тарифам."""
        cost = None
        if self.input_price is not None and self.output_price is not None:
            prompt = prompt_tokens if prompt_tokens is not None else history_tokens + request_tokens
            completion = completion_tokens if completion_tokens is not None else (response_tokens or 0)
            cost = (prompt * self.input_price + completion * self.output_price) / 1_000_000
        turn = TurnUsage(
            request_tokens=request_tokens,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )
        self.turns.append(turn)
        return turn
