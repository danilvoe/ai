"""Сжатие истории диалога: старые сообщения заменяются кратким summary.

Принцип работы:

- Полная история хранится в ``Conversation.messages`` и доступна для просмотра.
- Когда свежих (не свёрнутых) сообщений становится больше ``keep_recent``, их
  краткое содержание запрашивается у LLM и дополняет ``Conversation.summary``.
- В запрос к модели подставляется ``[system: summary] + свежие сообщения``
  вместо полной истории: старые сообщения остаются на диске, но в запрос не
  уходят. Так контекст не растёт неограниченно, а токены экономятся.
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
    """Сколько последних свежих сообщений хранить без сжатия."""
    summarize_every: int = 10
    """Насколько свежих сообщений может накопиться до следующего сжатия."""
    max_tokens: int | None = None
    """Ограничение на длину summary (передаётся в запрос, если задано)."""

    @property
    def enabled(self) -> bool:
        """Сжатие включено, если окно свежих сообщений ограничено."""
        return self.keep_recent > 0


def build_context_messages(conversation: Conversation) -> list[dict]:
    """Собирает сообщения для отправки в модель.

    Если есть summary, он подставляется отдельным system-сообщением, а следом
    идут только свежие (не свёрнутые) сообщения. Это и есть "сжатая" история,
    которая уходит в запрос вместо полной.
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
    messages.extend(conversation.recent_messages)
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
    """Пора ли сжимать: свежих сообщений накопилось заметно больше keep_recent."""
    return config.enabled and len(conversation.recent_messages) > config.keep_recent + config.summarize_every


def compress(conversation: Conversation, client, config: CompressionConfig) -> bool:
    """Сжимает историю: свежие сообщения сворачиваются в обновлённый summary.

    Старые сообщения остаются в ``Conversation.messages``, а их количество
    помечается в ``Conversation.summarized``. Возвращает True, если сжатие
    реально выполнено (было что сворачивать).
    """
    if not should_compress(conversation, config):
        return False

    messages = conversation.messages
    keep = len(conversation.recent_messages) - config.keep_recent
    to_summarize = conversation.recent_messages[:keep]
    if not to_summarize:
        return False

    new_summary = summarize_messages(
        client,
        to_summarize,
        old_summary=conversation.summary,
        max_tokens=config.max_tokens,
    )
    conversation.set_summary(new_summary, summarized=conversation.summarized + len(to_summarize))
    return True


def compressed_context_tokens(conversation: Conversation) -> int:
    """Оценка токенов сжатого контекста: summary + свежие сообщения."""
    total = estimate_messages_tokens(build_context_messages(conversation))
    if conversation.summary:
        total += estimate_tokens(conversation.summary)
    return total
