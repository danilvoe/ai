"""Агент — отдельная сущность, принимающая запрос и вызывающая LLM."""

from .conversation import Conversation, SessionStore
from .llm_client import Completion, LLMClient


class Agent:
    """Инкапсулирует логику запроса к LLM и возврата ответа."""

    def __init__(self, client: LLMClient, conversation: Conversation) -> None:
        self._client = client
        self._conversation = conversation

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def history(self) -> list[dict]:
        """История диалога текущей сессии."""
        return self._conversation.messages

    def ask(self, user_request: str) -> Completion:
        self._conversation.append("user", user_request)
        try:
            reply = self._client.complete(self._conversation.messages)
        except Exception:
            self._conversation.pop_last()
            raise
        self._conversation.append("assistant", reply.content)
        return reply
