"""Агент — отдельная сущность, принимающая запрос и вызывающая LLM."""

from .llm_client import Completion, LLMClient


class Agent:
    """Инкапсулирует логику запроса к LLM и возврата ответа."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def ask(self, user_request: str) -> Completion:
        messages = [{"role": "user", "content": user_request}]
        return self._client.complete(messages)
