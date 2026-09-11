"""Агент — отдельная сущность, принимающая запрос и вызывающая LLM."""

from .conversation import Conversation, SessionStore
from .llm_client import Completion, LLMClient
from .tokens import (
    ContextOverflowError,
    UsageReport,
    estimate_messages_tokens,
    estimate_tokens,
)


class Agent:
    """Инкапсулирует логику запроса к LLM и возврата ответа."""

    def __init__(
        self,
        client: LLMClient,
        conversation: Conversation,
        max_context_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._max_context_tokens = max_context_tokens
        self._last_prompt_tokens: int | None = None
        self._usage = UsageReport(
            input_price=client.config.get("input_cost_per_million"),
            output_price=client.config.get("output_cost_per_million"),
        )

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def history(self) -> list[dict]:
        """История диалога текущей сессии."""
        return self._conversation.messages

    @property
    def usage(self) -> UsageReport:
        """Накопительная статистика токенов и стоимости сессии."""
        return self._usage

    @property
    def context_tokens(self) -> int:
        """Токены текущей истории диалога.

        Приоритет у реального значения prompt_tokens, которое модель
        вернула в прошлом ответе: оно учитывает реальный токенизатор и не
        занижает оценку. Пока таких данных нет — используется локальная
        оценка (~4 символа на токен).
        """
        return self._last_prompt_tokens or estimate_messages_tokens(self._conversation.messages)

    def ask(self, user_request: str) -> Completion:
        history_tokens = self.context_tokens
        request_tokens = estimate_tokens(user_request)
        if self._max_context_tokens is not None:
            total = history_tokens + request_tokens
            if total > self._max_context_tokens:
                raise ContextOverflowError(
                    f"Контекст переполнен: {total} токенов > лимита "
                    f"{self._max_context_tokens}. История диалога не была отправлена."
                )

        self._conversation.append("user", user_request)
        try:
            reply = self._client.complete(self._conversation.messages)
        except Exception:
            self._conversation.pop_last()
            raise

        self._last_prompt_tokens = reply.prompt_tokens

        response_tokens = (
            reply.completion_tokens
            if reply.completion_tokens is not None
            else estimate_tokens(reply.content)
        )
        self._usage.record(
            request_tokens=request_tokens,
            history_tokens=history_tokens,
            response_tokens=response_tokens,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        self._conversation.append("assistant", reply.content)
        return reply
