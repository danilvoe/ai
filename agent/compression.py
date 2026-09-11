"""Сжатие истории диалога: старые сообщения заменяются кратким summary.

Принцип работы:

- Последние ``keep_recent`` сообщений хранятся "как есть".
- Когда количество сообщений в окне превышает ``keep_recent``, самые старые из
  них выносятся за пределы окна, а их краткое содержание запрашивается у LLM и
  сохраняется в ``Conversation.summary`` отдельно от сообщений.
- В запрос к модели подставляется ``[system: summary] + окно последних
  сообщений`` вместо полной истории. Таким образом контекст не растёт
  неограниченно, а токены экономятся.
"""

from dataclasses import dataclass

from .conversation import Conversation
from .tokens import estimate_messages_tokens, estimate_tokens

SUMMARY_SYSTEM = (
    "Ты — модуль сжатия истории диалога. Перепиши старую часть диалога "
    "в одно связное краткое содержание (summary) на русском языке. "
    "Сохрани ключевые факты, решения, имена и договорённости, которые могут "
    "понадобиться в дальнейшем. Не добавляй от себя ничего, чего не было в "
    "диалоге. Ответь только текстом summary, без предисловий и заголовков."
)


@dataclass
class CompressionConfig:
    """Настройки сжатия истории."""

    keep_recent: int = 10
    """Сколько последних сообщений хранить без изменений."""
    summarize_every: int = 10
    """Через сколько сообщений в окне пересжимать старую часть."""
    max_tokens: int | None = None
    """Ограничение на длину summary (передаётся в запрос, если задано)."""

    @property
    def enabled(self) -> bool:
        """Сжатие включено, если окно сообщений ограничено."""
        return self.keep_recent > 0


def build_context_messages(conversation: Conversation) -> list[dict]:
    """Собирает сообщения для отправки в модель.

    Если есть summary, он подставляется отдельным system-сообщением, а сами
    сообщения окна идут следом. Это и есть "сжатая" история, которая уходит
    в запрос вместо полной.
    """
    messages: list[dict] = []
    if conversation.summary:
        messages.append(
            {
                "role": "system",
                "content": "Краткое содержание предыдущей части диалога:\n"
                + conversation.summary,
            }
        )
    messages.extend(conversation.messages)
    return messages


def build_summary_prompt(messages: list[dict], old_summary: str = "") -> list[dict]:
    """Формирует сообщения для запроса на создание/обновление summary."""
    prompt = []
    if old_summary:
        prompt.append(
            {
                "role": "system",
                "content": (
                    "Ниже уже есть краткое содержание предыдущей части диалога. "
                    "Учти его и дополни/уточни новыми сообщениями."
                ),
            }
        )
        prompt.append(
            {"role": "system", "content": "Текущий summary:\n" + old_summary}
        )
    prompt.append({"role": "system", "content": SUMMARY_SYSTEM})
    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        prompt.append({"role": role, "content": message.get("content", "")})
    return prompt


def summarize_messages(
    client,
    messages: list[dict],
    old_summary: str = "",
    max_tokens: int | None = None,
) -> str:
    """Просит LLM сжать список сообщений в краткое содержание.

    Возвращает текст summary. Стоимость/токены этого служебного запроса не
    включаются в учёт сессии агента — это отдельный технический вызов.
    """
    prompt = build_summary_prompt(messages, old_summary)
    if max_tokens is not None:
        reply = client.complete(prompt, max_tokens=max_tokens)
    else:
        reply = client.complete(prompt)
    return (reply.content or "").strip()


def should_compress(conversation: Conversation, config: CompressionConfig) -> bool:
    """Пора ли сжимать: сообщений в окне стало заметно больше keep_recent."""
    return config.enabled and len(conversation.messages) > config.keep_recent + config.summarize_every


def compress(conversation: Conversation, client, config: CompressionConfig) -> bool:
    """Сжимает историю: старые сообщения заменяются обновлённым summary.

    Возвращает True, если сжатие реально выполнено (было что выносить).
    """
    if not should_compress(conversation, config):
        return False

    messages = conversation.messages
    keep = messages[-config.keep_recent:]
    to_summarize = messages[: len(messages) - config.keep_recent]
    if not to_summarize:
        return False

    new_summary = summarize_messages(
        client,
        to_summarize,
        old_summary=conversation.summary,
        max_tokens=config.max_tokens,
    )
    conversation.set_summary(new_summary)
    conversation._messages = keep
    conversation.save()
    return True


def compressed_context_tokens(conversation: Conversation) -> int:
    """Оценка токенов сжатого контекста: summary + окно сообщений."""
    total = estimate_messages_tokens(build_context_messages(conversation))
    if conversation.summary:
        total += estimate_tokens(conversation.summary)
    return total
