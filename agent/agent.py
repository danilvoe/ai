"""Агент — отдельная сущность, принимающая запрос и вызывающая LLM."""

import json

from .compression import (
    CompressionConfig,
    build_context_messages,
    compressed_context_tokens,
    compress,
)
from .context import ContextStrategy
from .conversation import Conversation, SessionStore
from .invariants import Invariants, parse_guard_verdict
from .llm_client import Completion, LLMClient
from .mcp_tools import McpToolRuntime, ToolCall, ToolResult
from .memory import MemoryLayers
from .personalization import Personalization
from .task_state import (
    TaskStateMachine,
    parse_completion_verdict,
    parse_transition_verdict,
)
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

    Если заданы ``invariants`` (см. agent/invariants.py), агент работает в
    рамках жёстких ограничений — архитектура, принятые решения, стек,
    бизнес-правила. Инварианты хранятся отдельно от диалога и подмешиваются
    к каждому запросу как system-сообщение, поэтому модель учитывает их в
    рассуждении. Если ``invariants.enforce`` включён, агент после ответа
    прогоняет короткую проверку: не нарушает ли предложенное решение
    инвариант. Если нарушает — ответ заменяется отказом с объяснением.

    Если задан ``tools`` (см. agent/mcp_tools.py), агент может вызывать
    инструменты MCP-сервера. Перед основным ответом он коротким запросом к
    LLM выбирает инструмент и аргументы под запрос пользователя, выполняет
    вызов через MCP и подмешивает результат в контекст, поэтому основная
    модель отвечает по реальным данным (например, публичные проекты GitFlic).

    Вместо одного сервера можно передать оркестратор нескольких MCP-серверов
    (``McpOrchestrator``, см. agent/orchestrator.py). Тогда агент ведёт
    **длинный флоу**: на каждом шаге LLM выбирает следующий инструмент любого
    из серверов, вызов маршрутизируется на нужный сервер, а его результат
    возвращается в контекст следующего шага. Число шагов ограничено
    ``max_tool_steps``. Все вызовы доступны через ``last_tool_calls``.
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
        invariants: Invariants | None = None,
        tools: McpToolRuntime | None = None,
        max_tool_steps: int = 6,
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._max_context_tokens = max_context_tokens
        self._compression = compression or CompressionConfig(keep_recent=0)
        self._context_strategy = context_strategy
        self._memory = memory
        self._personalization = personalization
        self._task = task
        self._invariants = invariants
        self._tools = tools
        self._max_tool_steps = max(1, int(max_tool_steps))
        self._last_prompt_tokens: int | None = None
        self._last_guard_violation: bool = False
        self._last_transition_violation: bool = False
        self._last_tool_call: ToolCall | None = None
        self._last_tool_result: ToolResult | None = None
        self._last_tool_calls: list[ToolCall] = []
        self._last_tool_results: list[ToolResult] = []
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

    @property
    def invariants(self) -> Invariants | None:
        """Инварианты агента (None — ограничения не используются)."""
        return self._invariants

    @property
    def last_guard_violation(self) -> bool:
        """Нарушал ли последний ответ агента инварианты (True — был отказ)."""
        return self._last_guard_violation

    @property
    def last_transition_violation(self) -> bool:
        """Перепрыгивал ли последний ответ агента этап задачи (True — был отказ)."""
        return self._last_transition_violation

    @property
    def tools(self) -> McpToolRuntime | None:
        """Рантайм MCP-инструментов (None — инструменты не подключены)."""
        return self._tools

    @property
    def last_tool_call(self) -> ToolCall | None:
        """Какой MCP-инструмент вызвал агент на последнем ходу (None — не вызывал)."""
        return self._last_tool_call

    @property
    def last_tool_result(self) -> ToolResult | None:
        """Результат последнего вызова MCP-инструмента (None — вызова не было)."""
        return self._last_tool_result

    @property
    def max_tool_steps(self) -> int:
        """Максимум шагов длинного флоу инструментов за один ход."""
        return self._max_tool_steps

    @property
    def last_tool_calls(self) -> list[ToolCall]:
        """Все вызовы MCP-инструментов за последний ход, в порядке выполнения."""
        return list(self._last_tool_calls)

    @property
    def last_tool_results(self) -> list[ToolResult]:
        """Результаты всех вызовов инструментов за последний ход, по порядку."""
        return list(self._last_tool_results)

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

    def _invariant_messages(self) -> list[dict]:
        """Список system-сообщений инвариантов (пусто, если инвариантов нет)."""
        if self._invariants is None:
            return []
        prompt = self._invariants.system_message()
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
        Если заданы инварианты, они вставляются сразу после профиля, но перед
        состоянием задачи: это жёсткие рамки, которые модель обязана соблюдать
        в любом решении, поэтому они идут раньше цели.
        """
        if self._context_strategy is not None:
            messages = self._context_strategy.build_messages(self._conversation)
        else:
            messages = build_context_messages(self._conversation)
        if self._memory is not None:
            messages = self._memory.build_memory_context() + messages
        return (
            self._profile_messages()
            + self._invariant_messages()
            + self._task_messages()
            + messages
        )

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
        tokens += estimate_messages_tokens(self._invariant_messages())
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

        # MCP-инструменты: если запрос требует данных извне, агент вызывает
        # инструмент и подмешивает его результат в контекст (Day 17).
        tool_messages = self._maybe_call_tool(user_request)
        if tool_messages:
            context_messages = context_messages + tool_messages

        # Контролируемые переходы: не пытается ли пользователь перепрыгнуть этап.
        guard_reason = self._stage_guard_reason(user_request)
        if guard_reason is not None:
            reply = self._stage_refusal(guard_reason)
            self._record_reply(reply, user_request, history_tokens)
            return reply

        try:
            reply = self._client.complete(context_messages)
        except Exception:
            self._conversation.pop_last()
            raise

        self._last_prompt_tokens = reply.prompt_tokens

        reply = self._enforce_invariants(user_request, reply)
        reply = self._enforce_stage_guard(user_request, reply)

        self._record_reply(reply, user_request, history_tokens)

        self._maybe_auto_advance_task(user_request, reply.content)

        if self._compression.enabled:
            compress(self._conversation, self._client, self._compression)

        return reply

    def _record_reply(self, reply: Completion, user_request: str, history_tokens: int) -> None:
        """Фиксирует ответ в истории и в накопительной статистике токенов."""
        response_tokens = (
            reply.completion_tokens
            if reply.completion_tokens is not None
            else estimate_tokens(reply.content)
        )
        self._usage.record(
            request_tokens=estimate_tokens(user_request),
            history_tokens=history_tokens,
            response_tokens=response_tokens,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        self._conversation.append("assistant", reply.content)
        if self._context_strategy is not None:
            self._context_strategy.on_reply(self._conversation, reply)

    def _stage_guard_reason(self, user_request: str) -> str | None:
        """Причина отказа, если пользователь пытается перепрыгнуть этап (иначе None)."""
        task = self._task
        if task is None or not task.strict:
            return None
        return task.guard_request(user_request)

    def _stage_refusal(self, reason: str) -> Completion:
        """Отказ ассистента: этап нельзя перепрыгнуть (детерминированная проверка)."""
        content = self._task.refusal_message(reason) if self._task is not None else reason
        return Completion(
            content=content,
            model=self._client.config.get("model"),
            prompt_tokens=0,
            completion_tokens=estimate_tokens(content),
        )

    def _enforce_stage_guard(self, user_request: str, reply: Completion) -> Completion:
        """Проверяет ответ агента на перепрыгивание этапа задачи.

        Работает, только если состояние задачи задано и ``task.strict`` включён.
        Коротким запросом к LLM агент выясняет, не перепрыгнул ли ответ текущий
        этап (не выдал ли он результат более позднего этапа). Если перепрыгнул —
        ответ заменяется отказом с объяснением, иначе — остаётся как есть.
        """
        self._last_transition_violation = False
        task = self._task
        if task is None or not task.strict:
            return reply
        check_messages = task.transition_guard_messages(user_request, reply.content)
        if not check_messages:
            return reply
        try:
            verdict_text = self._client.complete(check_messages).content
        except Exception:
            # Проверка не критична: если она не удалась, ответ оставляем как есть.
            return reply
        verdict = parse_transition_verdict(verdict_text)
        if not verdict.violated:
            return reply
        self._last_transition_violation = True
        return Completion(
            content=task.refusal_message(verdict.reason),
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=estimate_tokens(task.refusal_message(verdict.reason)),
        )

    def _enforce_invariants(self, user_request: str, reply: Completion) -> Completion:
        """Проверяет ответ агента на нарушение инвариантов.

        Работает, только если инварианты заданы и ``invariants.enforce`` включён.
        Коротким запросом к LLM агент выясняет, не нарушает ли предложенное
        решение инвариант. Если нарушает — возвращает заменённый ответ с отказом
        и объяснением (см. Invariants.refusal_message), иначе — исходный ответ.
        """
        self._last_guard_violation = False
        invariants = self._invariants
        if invariants is None or not invariants.enforce or not invariants:
            return reply
        check_messages = invariants.guard_check_messages(user_request, reply.content)
        if not check_messages:
            return reply
        try:
            verdict_text = self._client.complete(check_messages).content
        except Exception:
            # Проверка не критична: если она не удалась, ответ оставляем как есть.
            return reply
        verdict = parse_guard_verdict(verdict_text)
        if not verdict.violated:
            return reply
        self._last_guard_violation = True
        return Completion(
            content=invariants.refusal_message(verdict),
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=estimate_tokens(invariants.refusal_message(verdict)),
        )

    def _maybe_call_tool(self, user_request: str) -> list[dict]:
        """Выбирает MCP-инструменты под запрос, вызывает их и возвращает результаты.

        Работает, только если подключён рантайм инструментов (``tools``) —
        один MCP-сервер (``McpToolRuntime``) или оркестратор нескольких серверов
        (``McpOrchestrator``).

        Это **длинный флоу**: на каждом шаге коротким запросом к LLM агент
        выбирает следующий инструмент и аргументы (``plan_messages`` +
        ``parse_plan``), зная каталог всех серверов и уже полученные результаты.
        Вызов маршрутизируется (у оркестратора — на нужный сервер), результат
        кладётся в контекст следующего шага. Цикл длится до ``max_tool_steps``
        шагов, пока LLM не вернёт ``{"tool": null}``, пока не случится ошибка
        или повтор вызова. Возвращает system-сообщения со всеми результатами.
        """
        self._last_tool_call = None
        self._last_tool_result = None
        self._last_tool_calls = []
        self._last_tool_results = []
        runtime = self._tools
        if runtime is None:
            return []

        messages: list[dict] = []
        results_log: list[dict] = []
        seen: set[str] = set()

        for _ in range(self._max_tool_steps):
            try:
                plan_messages = runtime.plan_messages(user_request, results_log)
                if not plan_messages:
                    break
                plan_text = self._client.complete(
                    plan_messages, temperature=0.0
                ).content
                call = runtime.parse_plan(plan_text)
            except Exception:
                # Инструменты не критичны: при сбое выбора ответ даёт модель.
                break
            if call is None:
                break

            key = f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"
            if key in seen:
                break
            seen.add(key)

            self._last_tool_call = call
            try:
                result = runtime.call(call.name, call.arguments)
            except Exception as exc:  # noqa: BLE001 — показываем ошибку инструмента
                result = ToolResult(
                    name=call.name,
                    arguments=call.arguments,
                    content=f"Ошибка вызова MCP-инструмента: {exc}",
                    is_error=True,
                )
            self._last_tool_result = result
            self._last_tool_calls.append(call)
            self._last_tool_results.append(result)
            messages.append(runtime.result_message(result))

            payload = result.structured if result.structured is not None else result.content
            results_log.append(
                {
                    "step": len(self._last_tool_calls),
                    "tool": call.name,
                    "server": result.server,
                    "arguments": call.arguments,
                    "result": payload,
                    "error": result.is_error,
                }
            )
            if result.is_error:
                break

        return messages

    def _maybe_auto_advance_task(self, user_request: str, assistant_reply: str) -> None:
        """Автоматически продвигает задачу, если этап выполнен.

        Работает, только если включён ``task.auto_advance``. Агент коротким
        запросом к LLM выясняет, выполнено ли ожидаемое действие текущего
        этапа; если да — завершает этап и переходит на следующий по автомату.
        """
        task = self._task
        if task is None or not task.auto_advance or task.is_done:
            return
        check_messages = task.completion_check_messages(
            user_request,
            assistant_reply,
            recent_messages=self._conversation.messages,
        )
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
