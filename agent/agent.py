"""Агент — отдельная сущность, принимающая запрос и вызывающая LLM."""

from .compression import (
    CompressionConfig,
    build_context_messages,
    compressed_context_tokens,
    compress,
)
from .context import ContextStrategy
from .conversation import Conversation, SessionStore
from .llm_client import Completion, LLMClient
from .memory import MemoryLayers
from .personalization import Personalization
from .task_state import TaskStateMachine, parse_completion_verdict
from .tokens import (
    ContextOverflowError,
    UsageReport,
    estimate_messages_tokens,
    estimate_tokens,
)


class Agent:
    """Инкапсулирует логику запроса к LLM и возврата ответа.

    При включённом сжатии истории (``compression.keep_recent``) старые сообщения
    заменяются summary, которое подставляется в запрос вместо полной истории —
    см. agent/compression.py.

    Альтернативно можно задать ``context_strategy`` — стратегию управления
    контекстом без summary (Sliding Window / Sticky Facts / Branching),
    см. agent/context.py. Если стратегия задана, она определяет, какие сообщения
    уходят в модель, и имеет приоритет над сжатием.

    Если задана ``memory`` (см. agent/memory.py), агент использует явную модель
    памяти из трёх слоёв: краткосрочная (текущий диалог), рабочая (данные
    текущей задачи) и долговременная (профиль, решения, знания). Слои хранятся
    отдельно, а их содержимое добавляется к запросу в виде system-сообщений.

    Если задан ``personalization`` (см. agent/personalization.py), активный
    профиль пользователя автоматически подмешивается к каждому запросу:
    его system-сообщение вставляется первым и учитывается в контексте.

    Если задан ``task`` (см. agent/task_state.py), агент держит состояние
    текущей задачи как конечный автомат (этап / шаг / ожидаемое действие).
    Состояние подмешивается к запросу в виде system-сообщения и сохраняется
    в сессию, поэтому при паузе/возобновлении задача продолжается без
    повторных объяснений.
    """

    def __init__(
        self,
        client: LLMClient,
        conversation: Conversation,
        max_context_tokens: int | None = None,
        compression: CompressionConfig | None = None,
        context_strategy: ContextStrategy | None = None,
        memory: MemoryLayers | None = None,
        personalization: Personalization | None = None,
        task: TaskStateMachine | None = None,
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._max_context_tokens = max_context_tokens
        self._compression = compression or CompressionConfig(keep_recent=0)
        self._context_strategy = context_strategy
        self._memory = memory
        self._personalization = personalization
        self._task = task
        self._last_prompt_tokens: int | None = None
        self._usage = UsageReport(
            input_price=client.config.get("input_cost_per_million"),
            output_price=client.config.get("output_cost_per_million"),
        )

    @property
    def context_strategy(self) -> ContextStrategy | None:
        """Стратегия управления контекстом (None — история как есть)."""
        return self._context_strategy

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
    def compression(self) -> CompressionConfig:
        """Настройки сжатия истории."""
        return self._compression

    @property
    def memory(self) -> MemoryLayers | None:
        """Явная модель памяти агента (None — память не используется)."""
        return self._memory

    @property
    def personalization(self) -> Personalization | None:
        """Персонализация агента: активный профиль пользователя (None — выключена)."""
        return self._personalization

    @property
    def task(self) -> TaskStateMachine | None:
        """Конечный автомат состояния задачи (None — состояние не используется)."""
        return self._task

    def save_task(self) -> None:
        """Сохраняет текущее состояние задачи в сессию (чтобы оно пережило перезапуск)."""
        if self._task is not None:
            self._conversation.set_task(self._task.to_dict())

    @property
    def profile_prompt(self) -> str:
        """System-сообщение активного профиля (пусто, если профиля нет/пуст)."""
        if self._personalization is None:
            return ""
        return self._personalization.system_message()

    def _profile_messages(self) -> list[dict]:
        """Список system-сообщений активного профиля (пусто, если профиль пуст)."""
        prompt = self.profile_prompt
        if not prompt:
            return []
        return [{"role": "system", "content": prompt}]

    def _task_messages(self) -> list[dict]:
        """Список system-сообщений состояния задачи (пусто, если задача пуста)."""
        if self._task is None:
            return []
        prompt = self._task.system_message()
        if not prompt:
            return []
        return [{"role": "system", "content": prompt}]

    @property
    def context_messages(self) -> list[dict]:
        """Сообщения, которые будут отправлены в модель.

        Определяется активной стратегией управления контекстом, а если её нет —
        сжатием истории (summary + свежее окно), см. agent/compression.py.
        Если задана модель памяти, к полученным сообщениям добавляется блок
        system-сообщений из всех слоёв памяти (см. agent/memory.py).
        Если задана персонализация, её профиль вставляется первым — он идёт
        раньше памяти, так как задаёт общий стиль и ограничения.
        Если задано состояние задачи, оно вставляется следом за профилем —
        это текущая цель, поверх стиля.
        """
        if self._context_strategy is not None:
            messages = self._context_strategy.build_messages(self._conversation)
        else:
            messages = build_context_messages(self._conversation)
        if self._memory is not None:
            messages = self._memory.build_memory_context() + messages
        return self._profile_messages() + self._task_messages() + messages

    @property
    def context_tokens(self) -> int:
        """Токены текущего контекста, отправляемого в модель.

        Приоритет у реального значения prompt_tokens, которое модель вернула в
        прошлом ответе. Пока таких данных нет — используется локальная оценка.
        """
        if self._last_prompt_tokens is not None:
            return self._last_prompt_tokens
        tokens = 0
        if self._context_strategy is not None:
            tokens += self._context_strategy.context_tokens(self._conversation)
        elif self._compression.enabled:
            tokens += compressed_context_tokens(self._conversation)
        else:
            tokens += estimate_messages_tokens(self._conversation.messages)
        if self._memory is not None:
            tokens += estimate_messages_tokens(self._memory.build_memory_context())
        tokens += estimate_messages_tokens(self._profile_messages())
        tokens += estimate_messages_tokens(self._task_messages())
        return tokens

    def ask(self, user_request: str) -> Completion:
        # Контекст, который реально уйдёт в модель, зависит от сжатия.
        context_messages = self.context_messages
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
        if self._context_strategy is not None:
            self._context_strategy.on_user_message(self._conversation, user_request)
        context_messages = self.context_messages
        try:
            reply = self._client.complete(context_messages)
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
        if self._context_strategy is not None:
            self._context_strategy.on_reply(self._conversation, reply)

        self._maybe_auto_advance_task(user_request, reply.content)

        if self._compression.enabled:
            compress(self._conversation, self._client, self._compression)

        return reply

    def _maybe_auto_advance_task(self, user_request: str, assistant_reply: str) -> None:
        """Автоматически продвигает задачу, если этап выполнен.

        Работает, только если включён ``task.auto_advance``. Агент коротким
        запросом к LLM выясняет, выполнено ли ожидаемое действие текущего
        этапа; если да — завершает этап и переходит на следующий по автомату.
        """
        task = self._task
        if task is None or not task.auto_advance or task.is_done:
            return
        check_messages = task.completion_check_messages(user_request, assistant_reply)
        if not check_messages:
            return
        try:
            verdict = self._client.complete(check_messages)
        except Exception:
            # Проверка не критична: если она не удалась, задачу не двигаем.
            return
        if not parse_completion_verdict(verdict.content):
            return
        task.complete()
        try:
            task.advance()
            self.save_task()
        except ValueError:
            self.save_task()
