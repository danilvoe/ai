"""HTTP-клиент для запросов к LLM через OpenAI-совместимый API."""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    """Ответ LLM: текст и информация об использовании токенов."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient:
    """Отправляет запросы к API и разбирает ответ."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._url = config["base_url"].rstrip("/") + "/chat/completions"
        # Повторы при сетевых сбоях/таймауте чтения (сервер LLM может "затупить").
        self._max_retries = max(0, int(config.get("max_retries", 2)))

    @property
    def config(self) -> dict:
        """Конфигурация клиента, переданная при создании."""
        return self._config

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        body = {
            "model": self._config["model"],
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self._config.get("temperature", 0.7),
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                "Authorization": f"Bearer {self._config['api_key']}",
            },
            method="POST",
        )
        timeout = self._config.get("timeout", 120)
        data = self._request_json(request, timeout)
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return Completion(
            content=choice["content"],
            model=data.get("model", self._config["model"]),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _request_json(self, request: urllib.request.Request, timeout: float) -> dict:
        """Выполняет запрос с повтором при таймауте/сетевой ошибке.

        HTTP-ошибки (4xx/5xx) не повторяются — они сразу поднимаются наружу;
        повторяются только таймауты чтения и сетевые сбои.
        """
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt >= self._max_retries:
                    raise
                attempt += 1
                time.sleep(min(2.0 * attempt, 5.0))
