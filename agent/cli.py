"""CLI-интерфейс: принимает запрос пользователя и выводит ответ агента."""

import json
import sys
import urllib.error
from pathlib import Path

from .agent import Agent
from .compression import CompressionConfig
from .config import load_config
from .context import strategy_from_config
from .conversation import SessionStore
from .invariants import INVARIANT_CATEGORIES, Invariants
from .llm_client import LLMClient
from .mcp_tools import McpToolRuntime, runtime_from_config
from .memory import LONG_TERM_KINDS, MemoryLayers
from .orchestrator import orchestrator_from_config
from .personalization import PREFERENCE_GROUPS, Personalization, UserProfile
from .task_state import TaskStateMachine
from .tokens import ContextOverflowError


def _compression_from_config(config: dict) -> CompressionConfig:
    """Читает настройки сжатия истории из конфигурации."""
    section = config.get("compression", {}) or {}
    return CompressionConfig(
        keep_recent=section.get("keep_recent", 10),
        summarize_every=section.get("summarize_every", 10),
        max_tokens=section.get("max_tokens"),
    )


def build_agent(conversation) -> Agent:
    config = load_config()
    client = LLMClient(config)
    return Agent(
        client,
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=_compression_from_config(config),
        context_strategy=strategy_from_config(config),
        memory=_memory_from_config(conversation),
        personalization=_personalization_from_config(),
        task=_task_from_config(conversation),
        invariants=_invariants_from_config(),
        tools=_tools_from_config(),
        max_tool_steps=int((config.get("orchestrator", {}) or {}).get("max_steps", 6)),
    )


def _tools_from_config() -> McpToolRuntime | None:
    """Создаёт рантайм MCP-инструментов (None — выключено).

    Если включена секция ``orchestrator``, возвращается оркестратор нескольких
    MCP-серверов (Day 20); иначе — рантайм одного сервера из секции ``tools``.
    """
    try:
        config = load_config()
        orchestrator = orchestrator_from_config(config)
        if orchestrator is not None:
            return orchestrator
        return runtime_from_config(config)
    except Exception:  # noqa: BLE001 — инструменты не должны ломать запуск
        return None


def _invariants_from_config() -> Invariants | None:
    """Создаёт хранилище инвариантов из секции ``invariants`` (None — выключено)."""
    config = load_config()
    section = config.get("invariants", {}) or {}
    if not section.get("enabled", False):
        return None
    path = section.get("path")
    return Invariants(
        path=Path(path) if path else None,
        enforce=section.get("enforce", True),
    )


def _task_from_config(conversation) -> TaskStateMachine | None:
    """Создаёт конечный автомат состояния задачи из секции ``task`` (None — выключен)."""
    config = load_config()
    section = config.get("task", {}) or {}
    if not section.get("enabled", False):
        return None
    return TaskStateMachine.from_dict(
        conversation.task,
        auto_advance=section.get("auto_advance", False),
        strict=section.get("strict", True),
    )


def _personalization_from_config() -> Personalization | None:
    """Создаёт персонализацию из секции ``personalization`` (None — выключена)."""
    config = load_config()
    section = config.get("personalization", {}) or {}
    if not section.get("enabled", False):
        return None
    path = section.get("profiles_path")
    return Personalization(path=path or None)


def _memory_from_config(conversation) -> MemoryLayers | None:
    """Создаёт модель памяти из секции ``memory`` конфигурации (None — выключена)."""
    config = load_config()
    section = config.get("memory", {}) or {}
    if not section.get("enabled", False):
        return None
    profile_path = section.get("profile_path")
    return MemoryLayers(
        conversation,
        long_term_path=profile_path or None,
    )


def print_usage(agent: Agent) -> None:
    """Печатает токены и стоимость последнего хода и всей сессии."""
    last = agent.usage.turns[-1]
    if last.prompt_tokens is not None:
        # Реальные входные токены, которые модель вернула в прошлом ответе.
        print(
            f"Токены хода: {last.total_tokens} "
            f"(вход {last.prompt_tokens}, "
            f"ответ {last.completion_tokens or last.response_tokens or 0})"
        )
    else:
        print(
            f"Токены хода: {last.total_tokens} "
            f"(история {last.history_tokens}, запрос {last.request_tokens}, "
            f"ответ {last.response_tokens or 0})"
        )
    real_prompt = agent.usage.total_prompt_tokens
    real_completion = agent.usage.total_completion_tokens
    if real_prompt is not None:
        print(
            f"Токены за всю сессию: {real_prompt + (real_completion or 0)} "
            f"(вход {real_prompt}, выход {real_completion or 0})"
        )
    else:
        print(
            f"Токены за всю сессию: {agent.usage.total_tokens} "
            f"(запросы {agent.usage.total_request_tokens}, "
            f"ответы {agent.usage.total_response_tokens})"
        )
    if agent.usage.total_cost is None:
        print("Стоимость сессии: н/д (укажите тарифы в config.json)")
    else:
        print(f"Стоимость сессии: ${agent.usage.total_cost:.6f}")
    print()


def print_history(agent: Agent) -> None:
    """Выводит полную историю диалога, помечая сообщения, свёрнутые в summary."""
    conv = agent.conversation
    if not conv.messages:
        print("История пуста.\n")
        return
    print(f"История диалога ({len(conv.messages)} сообщений):")
    for index, message in enumerate(conv.messages):
        is_summarized = index < conv.summarized
        marker = " (в summary)" if is_summarized else ""
        who = "Вы" if message.get("role") == "user" else "Агент"
        print(f"[{index + 1}] {who}{marker}: {message.get('content', '')}")
    if conv.summary:
        print(f"\nКраткое содержание свёрнутой части:\n{conv.summary}")
    print()


def pick_session(store: SessionStore) -> Agent:
    """Показывает список сессий и возвращает агента для выбранной или новой."""
    sessions = store.list_sessions()
    print("Доступные сессии:")
    for index, session_id in enumerate(sessions, start=1):
        print(f"  {index}. {session_id}")

    choice = input(
        f"Выберите сессию (1-{len(sessions)}) или 'new' для новой "
        f"['new']: " if sessions else "Сессий нет. Введите 'new' для новой ['new']: "
    ).strip()

    if choice.lower() in ("new", "") or not sessions:
        conversation = store.create()
        print(f"Новая сессия: {conversation.session_id}\n")
    else:
        try:
            conversation = store.get(sessions[int(choice) - 1])
        except (ValueError, IndexError):
            print(f"Неверный выбор, создаю новую сессию.")
            conversation = store.create()
        print(f"Открыта сессия: {conversation.session_id}\n")

    return build_agent(conversation)


def print_memory_status(agent: Agent) -> None:
    """Печатает сводку по всем слоям памяти агента."""
    memory = agent.memory
    if memory is None:
        print("Модель памяти: выключена (см. секцию memory в config.json).\n")
        return
    print(f"Модель памяти: включена ({memory.long_term.path.name}).")
    print(memory.summarize())
    print()


def _parse_memory_command(command: str) -> tuple[str, str, str] | None:
    """Разбирает команду ``remember <layer> <key> = <value>``.

    Слой: short | working | profile | decision | knowledge. Возвращает кортеж
    (layer, key, value) или None, если команда не распознана.
    """
    rest = command.strip()
    if not rest:
        return None
    parts = rest.split("=", maxsplit=1)
    if len(parts) != 2:
        return None
    lhs = parts[0].strip()
    rhs = parts[1].strip()
    if not lhs or not rhs:
        return None
    layer, _, key = lhs.partition(" ")
    key = key.strip()
    if not key:
        return None
    return layer.strip().lower(), key, rhs


def _handle_memory_command(agent: Agent, command: str) -> bool:
    """Обрабатывает команды модели памяти. Возвращает True, если команда взята."""
    memory = agent.memory
    if memory is None:
        return False

    parts = command.strip().split(maxsplit=2)
    verb = parts[0].lower()
    rest = command.strip()[len(parts[0]):].strip()

    if verb in ("memory", "память", "память?"):
        print_memory_status(agent)
        return True

    if verb in ("remember", "запомни"):
        parsed = _parse_memory_command(rest)
        if parsed is None:
            print("Использование: remember <short|working|profile|decision|knowledge> <ключ> = <значение>\n")
            return True
        layer, key, value = parsed
        if layer in ("working", "рабочая"):
            memory.remember_working(key, value)
            print(f"Записано в рабочую память: {key} = {value}\n")
        elif layer in ("short", "short_term", "краткосрочная"):
            memory.remember_short_term("user", f"{key} = {value}")
            print(f"Добавлено в диалог (краткосрочная память): {key} = {value}\n")
        elif layer in LONG_TERM_KINDS:
            memory.remember_long_term(layer, key, value)
            print(f"Записано в долговременную память ({layer}): {key} = {value}\n")
        else:
            print(
                f"Неизвестный слой памяти: {layer!r}. "
                f"Допустимо: short, working, {', '.join(LONG_TERM_KINDS)}.\n"
            )
        return True

    if verb in ("forget", "забыть", "удалить"):
        if len(parts) < 3:
            print("Использование: forget <profile|decision|knowledge> <ключ>\n")
            return True
        kind = parts[1].lower()
        key = parts[2]
        if kind not in LONG_TERM_KINDS:
            print(f"Неизвестная категория: {kind!r}.\n")
            return True
        if memory.long_term.forget(kind, key):
            print(f"Удалено из долговременной памяти ({kind}): {key}\n")
        else:
            print(f"Запись {key!r} ({kind}) не найдена.\n")
        return True

    if verb in ("clearworking", "clear_working", "новая_задача", "сменить_задачу"):
        memory.working.clear()
        print("Рабочая память очищена (текущая задача завершена).\n")
        return True

    return False


def print_task_status(agent: Agent) -> None:
    """Печатает формализованное состояние текущей задачи (конечный автомат)."""
    task = agent.task
    if task is None:
        print("Состояние задачи: выключено (см. секцию task в config.json).\n")
        return
    if not task.has_task():
        print("Состояние задачи: нет активной задачи (команда /task_new <описание>).\n")
        return
    print(f"Состояние задачи ({task.stage}):")
    for line in task.summarize().splitlines():
        print(f"  {line}")
    print()


def _handle_task_command(agent: Agent, command: str) -> bool:
    """Обрабатывает команды конечного автомата задачи. True, если команда взята."""
    task = agent.task
    if task is None:
        print("Состояние задачи: выключено (см. секцию task в config.json).\n")
        return True

    parts = command.strip().split(maxsplit=1)
    verb = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb in ("task", "задача", "статус"):
        print_task_status(agent)
        return True

    if verb in ("task_new", "новая_задача", "задача_new"):
        if not rest:
            print("Укажите описание задачи: task_new <описание>\n")
            return True
        task.reset(description=rest)
        agent.save_task()
        print("Новая задача начата (этап planning).")
        print_task_status(agent)
        return True

    if verb in ("task_complete", "выполнено", "готово_этап"):
        if not task.has_task():
            print("Нет активной задачи. Начните через /task_new <описание>.\n")
            return True
        if task.complete():
            agent.save_task()
            print("Этап отмечен выполненным — теперь можно /task_next.")
        else:
            print("Этап уже отмечен выполненным.\n")
        print_task_status(agent)
        return True

    if verb in ("task_next", "следующий_этап", "далее"):
        if not task.has_task():
            print("Нет активной задачи. Начните через /task_new <описание>.\n")
            return True
        try:
            task.advance()
            agent.save_task()
            print("Задача переведена на следующий этап по автомату.")
            print_task_status(agent)
        except ValueError as exc:
            print(f"{exc}\n")
        return True

    if verb in ("task_stage", "этап"):
        if not rest:
            print("Укажите этап: task_stage <planning|execution|validation|done>\n")
            return True
        try:
            task.set_stage(rest)
            agent.save_task()
            print_task_status(agent)
        except ValueError as exc:
            print(f"{exc}\n")
        return True

    if verb in ("task_step", "шаг"):
        if not rest:
            print("Укажите номер шага: task_step <N>\n")
            return True
        try:
            task.set_step(int(rest))
            agent.save_task()
            print_task_status(agent)
        except ValueError as exc:
            print(f"{exc}\n")
        return True

    if verb in ("task_expected", "ожидание"):
        if not rest:
            print("Укажите ожидаемое действие: task_expected <что нужно сейчас>\n")
            return True
        task.set_expected_action(rest)
        agent.save_task()
        print_task_status(agent)
        return True

    if verb in ("task_pause", "пауза", "приостановить"):
        if task.pause():
            agent.save_task()
            print("Задача приостановлена (можно на любом этапе).")
        else:
            print("Задача уже приостановлена.\n")
        print_task_status(agent)
        return True

    if verb in ("task_resume", "продолжить", "возобновить"):
        if task.resume():
            agent.save_task()
            print("Задача возобновлена с того же места (без повторных объяснений).")
        else:
            print("Задача не приостановлена.\n")
        print_task_status(agent)
        return True

    if verb in ("task_done", "готово", "завершить"):
        try:
            task.set_stage("done")
            agent.save_task()
            print("Задача переведена в состояние done.")
            print_task_status(agent)
        except ValueError as exc:
            print(f"{exc}\n")
        return True

    if verb in ("task_auto", "автопродвижение"):
        task.auto_advance = not task.auto_advance
        agent.save_task()
        state = "включено" if task.auto_advance else "выключено"
        print(f"Автопродвижение задачи: {state} "
              f"(задача сама переходит дальше, когда этап выполнен).\n")
        return True

    if verb in ("task_strict", "контроль_этапов"):
        task.strict = not task.strict
        agent.save_task()
        state = "включён" if task.strict else "выключен"
        print(f"Контроль переходов (запрет перепрыгивания этапа): {state}.\n")
        return True

    if verb in ("task_clear", "сбросить"):
        task.reset()
        agent.save_task()
        print("Состояние задачи сброшено.\n")
        return True

    return False


def print_invariant_status(agent: Agent) -> None:
    """Печатает все установленные инварианты агента."""
    invariants = agent.invariants
    if invariants is None:
        print("Инварианты: выключены (см. секцию invariants в config.json).\n")
        return
    print(invariants.summarize())
    print()


def _handle_invariant_command(agent: Agent, command: str) -> bool:
    """Обрабатывает команды инвариантов. Возвращает True, если команда взята."""
    invariants = agent.invariants
    if invariants is None:
        print("Инварианты: выключены (см. секцию invariants в config.json).\n")
        return True

    parts = command.strip().split(maxsplit=2)
    verb = parts[0].lower()
    rest = command.strip()[len(parts[0]):].strip()

    if verb in ("invariant", "инвариант", "ограничения", "правила"):
        print_invariant_status(agent)
        return True

    if verb in ("invariant_add", "добавить_инвариант", "новый_инвариант"):
        if not rest:
            print(
                "Использование: invariant_add <категория> <описание> "
                "[-- почему]\n"
            )
            return True
        category, _, remainder = rest.partition(" ")
        category = category.strip().lower()
        rationale = ""
        description = remainder.strip()
        if "--" in description:
            description, _, rationale = description.partition("--")
            description, rationale = description.strip(), rationale.strip()
        if category not in INVARIANT_CATEGORIES:
            print(
                f"Неизвестная категория: {category!r}. "
                f"Допустимо: {', '.join(INVARIANT_CATEGORIES)}.\n"
            )
            return True
        if not description:
            print("Укажите описание инварианта после категории.\n")
            return True
        invariant = invariants.add(category, description, rationale)
        print(
            f"Добавлен инвариант {invariant.id} "
            f"({invariant.category_label}): {invariant.description}"
        )
        if invariant.rationale:
            print(f"  Причина: {invariant.rationale}")
        print()
        return True

    if verb in ("invariant_del", "удалить_инвариант", "снять_инвариант"):
        if not rest:
            print("Использование: invariant_del <id>\n")
            return True
        if invariants.remove(rest):
            print(f"Инвариант {rest} удалён.\n")
        else:
            print(f"Инвариант {rest} не найден.\n")
        print_invariant_status(agent)
        return True

    if verb in ("invariant_clear", "очистить_инварианты"):
        invariants.clear()
        print("Все инварианты удалены.\n")
        return True

    if verb in ("invariant_enforce", "контроль"):
        invariants.enforce = not invariants.enforce
        state = "включена" if invariants.enforce else "выключена"
        print(f"Проверка ответа на нарушение инвариантов: {state}.\n")
        return True

    return False


def _handle_profile_command(agent: Agent, command: str) -> bool:
    """Обрабатывает команды персонализации. Возвращает True, если команда взята."""
    personalization = agent.personalization
    if personalization is None:
        print("Персонализация: выключена (см. секцию personalization в config.json).\n")
        return True

    parts = command.strip().split(maxsplit=2)
    verb = parts[0].lower()
    rest = command.strip()[len(parts[0]):].strip()

    if verb in ("profile", "профиль"):
        active = personalization.active_name
        print(f"Персонализация: {personalization.path.name}.")
        print(personalization.summarize())
        if active:
            profile = personalization.active()
            if profile is not None:
                print(f"\nАктивный профиль: {active}")
                prompt = profile.system_message()
                if prompt:
                    print(prompt)
                else:
                    print("  (профиль пуст — ничего не подмешивается)")
        print()
        return True

    if verb in ("profile_new", "новый_профиль"):
        if not rest:
            print("Укажите имя профиля: profile_new <имя>\n")
            return True
        name = rest
        if personalization.has(name):
            print(f"Профиль {name} уже существует.\n")
            return True
        profile = UserProfile(name=name)
        personalization.save(name, profile)
        personalization.set_active(name)
        print(f"Создан и активирован профиль: {name} (пустой). "
              f"Заполните его через /preference.\n")
        return True

    if verb in ("profile_use", "использовать", "активный"):
        if not rest:
            print("Укажите имя профиля: profile_use <имя>\n")
            return True
        if personalization.set_active(rest):
            print(f"Активный профиль: {rest}\n")
        else:
            print(f"Профиль {rest} не найден.\n")
        return True

    if verb in ("profile_del", "удалить_профиль"):
        if not rest:
            print("Укажите имя профиля: profile_del <имя>\n")
            return True
        if personalization.delete(rest):
            print(f"Профиль {rest} удалён.\n")
        else:
            print(f"Профиль {rest} не найден.\n")
        return True

    if verb in ("profile_off", "отключить"):
        personalization.set_active(None)
        print("Персонализация отключена (активного профиля нет).\n")
        return True

    if verb in ("preference", "предпочтение"):
        # preference <style|format|constraints> <ключ> = <значение>
        if personalization.active_name is None:
            print("Нет активного профиля. Активируйте через /profile_use <имя>.\n")
            return True
        profile = personalization.active()
        group, _, right = rest.partition(" ")
        key, _, value = right.partition("=")
        group, key, value = group.strip().lower(), key.strip(), value.strip()
        if group not in PREFERENCE_GROUPS:
            print(f"Неизвестная группа: {group!r}. "
                  f"Допустимо: {', '.join(PREFERENCE_GROUPS)}.\n")
            return True
        if not key or not value:
            print("Использование: preference <style|format|constraints> <ключ> = <значение>\n")
            return True
        profile.set_preference(group, key, value)
        personalization.save(personalization.active_name, profile)
        print(f"{PREFERENCE_GROUPS} → {key} = {value}\n")
        return True

    if verb in ("preference_del", "убрать_предпочтение"):
        group, _, key = rest.partition(" ")
        group, key = group.strip().lower(), key.strip()
        if personalization.active_name is None:
            print("Нет активного профиля.\n")
            return True
        if group not in PREFERENCE_GROUPS or not key:
            print("Использование: preference_del <style|format|constraints> <ключ>\n")
            return True
        profile = personalization.active()
        if profile.remove_preference(group, key):
            personalization.save(personalization.active_name, profile)
            print(f"Убрано: {group} → {key}\n")
        else:
            print(f"Предпочтение {key!r} ({group}) не найдено.\n")
        return True

    return False


def print_context_status(agent: Agent) -> None:
    """Печатает активную стратегию управления контекстом и состояние веток."""
    strategy = agent.context_strategy
    if strategy is not None:
        print(f"Стратегия управления контекстом: {strategy.label} ({strategy.name}).")
    else:
        print("Стратегия управления контекстом: нет (вся история как есть).")
    conv = agent.conversation
    if conv.facts:
        print("Липкие факты:")
        for key, value in conv.facts.items():
            print(f"  - {key}: {value}")
    if conv.has_branches:
        active = conv.active_branch
        print(f"Общая часть (checkpoint): {len(conv.messages) - _branch_len(conv)} сообщений.")
        for branch_id, branch in conv.branches.items():
            marker = " (активна)" if branch_id == active else ""
            print(f"  - {branch_id}: {len(branch['messages'])} сообщений{marker}")
    print()


def _branch_len(conv) -> int:
    """Число сообщений активной ветки (для заголовка checkpoint)."""
    if conv.active_branch is not None:
        return len(conv.branches[conv.active_branch]["messages"])
    return 0


def _handle_branch_command(agent: Agent, command: str) -> None:
    """Обрабатывает команды ветвления: checkpoint, branch <name>, switch <name>."""
    conv = agent.conversation
    parts = command.strip().split(maxsplit=1)
    verb = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb in ("checkpoint", "чекпоинт"):
        count = conv.checkpoint()
        print(f"Checkpoint сохранён: {count} сообщений в общей части.\n")
        return

    if verb in ("branch", "ветв", "ветка"):
        if not rest:
            print("Укажите имя ветки: branch <имя>\n")
            return
        if not conv.checkpoint:
            print("Сначала сохраните checkpoint (команда checkpoint).\n")
            return
        if conv.create_branch(rest):
            print(f"Создана и активирована ветка: {rest}\n")
        else:
            print(f"Не удалось создать ветку {rest} (уже существует или нет checkpoint).\n")
        return

    if verb in ("switch", "переключ", "переключить"):
        if not rest:
            print("Укажите имя ветки: switch <имя>\n")
            return
        if conv.switch_branch(rest):
            print(f"Активна ветка: {rest}\n")
        else:
            print(f"Ветка {rest} не найдена.\n")
        return


def _handle_tool_command(agent: Agent, command: str) -> bool:
    """Обрабатывает команды MCP-инструментов. Возвращает True, если команда взята."""
    runtime = agent.tools
    if runtime is None:
        print("MCP-инструменты: выключены (см. секцию tools в config.json).\n")
        return True

    parts = command.strip().split(maxsplit=2)
    verb = parts[0].lower()
    rest = command.strip()[len(parts[0]):].strip()

    if verb in ("tools", "инструменты", "mcp"):
        try:
            print(runtime.summarize())
        except Exception as exc:  # noqa: BLE001 — наглядная ошибка подключения
            print(f"Не удалось получить список MCP-инструментов: {exc}")
        print()
        return True

    if verb in ("tool", "вызов", "tool_call"):
        name, _, raw = rest.partition(" ")
        name = name.strip()
        if not name:
            print("Использование: tool <имя> [<json-аргументы>]\n")
            return True
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            print('Аргументы должны быть JSON-объектом: tool get_public_projects {"query": "docs"}\n')
            return True
        if not isinstance(arguments, dict):
            print("Аргументы должны быть JSON-объектом.\n")
            return True
        try:
            result = runtime.call(name, arguments)
        except Exception as exc:  # noqa: BLE001 — наглядная ошибка вызова
            print(f"Ошибка вызова MCP-инструмента: {exc}\n")
            return True
        status = "ошибка" if result.is_error else "ok"
        print(f"MCP-инструмент {result.name} → {status}:")
        if result.structured is not None:
            print(json.dumps(result.structured, ensure_ascii=False, indent=2))
        else:
            print(result.content)
        print()
        return True

    return False


def print_help() -> None:
    """Печатает список доступных команд агента."""
    print("""Доступные команды (вводятся со слэшем):

Общие:
  /exit /quit            — завершить
  /help                  — показать этот список
  /clear                 — стереть историю диалога
  /history               — показать полную историю диалога

Контекст и ветвление:
  /context               — показать стратегию контекста, факты, ветки
  /checkpoint            — сохранить точку ветвления
  /branch <имя>          — создать и активировать ветку
  /switch <имя>          — переключить активную ветку

Память (3 слоя):
  /memory                — показать содержимое всех слоёв памяти
  /remember <слой> <ключ> = <значение>
                         — записать в слой: short | working | profile | decision | knowledge
  /forget <категория> <ключ>
                         — удалить запись долговременной памяти (profile/decision/knowledge)
  /clearworking          — очистить рабочую память (смена задачи)

Персонализация (профиль пользователя):
  /profile               — показать профили и активный профиль
  /profile_new <имя>     — создать и активировать новый профиль
  /profile_use <имя>     — сделать профиль активным
  /profile_del <имя>     — удалить профиль
  /profile_off           — отключить персонализацию
  /preference <группа> <ключ> = <значение>
                         — задать предпочтение (группа: style | format | constraints)
  /preference_del <группа> <ключ> — убрать предпочтение

Состояние задачи (конечный автомат):
  /task                  — показать этап, шаг и ожидаемое действие
  /task_new <описание>   — начать новую задачу (этап planning)
  /task_expected <действие> — задать ожидаемое действие
  /task_complete         — отметить этап выполненным (разрешает /task_next)
  /task_auto             — вкл/выкл автопродвижение (задача сама идёт дальше)
  /task_strict           — вкл/выкл контроль переходов (запрет перепрыгивания этапа)
  /task_next             — перевести задачу на следующий этап по автомату
  /task_stage <этап>     — перейти на этап: planning | execution | validation | done
  /task_step <N>         — установить номер текущего шага (только вперёд)
  /task_pause            — приостановить задачу (можно на любом этапе)
  /task_resume           — продолжить задачу с того же места
  /task_done             — перевести задачу в состояние done
  /task_clear            — сбросить состояние задачи

Инварианты (ограничения, которые ассистент не может нарушать):
  /invariant             — показать все инварианты
  /invariant_add <категория> <описание> [-- причина]
                         — добавить инвариант (категория: architecture |
                           tech_decision | stack | business_rule | custom)
  /invariant_del <id>    — удалить инвариант
  /invariant_clear       — удалить все инварианты
  /invariant_enforce     — вкл/выкл проверку ответа на нарушение инвариантов

MCP-инструменты:
  /tools                 — показать серверы MCP и их инструменты с параметрами
  /tool <имя> [<json>]   — вызвать инструмент напрямую и показать результат
                           (например: /tool get_public_projects {"query": "docs"})
                           для нескольких серверов имя: /tool pipeline.search {"query": "docs"}

Любая строка без слэша уходит модели как сообщение пользователя.
""")


def _dispatch_command(agent: Agent, line: str) -> bool:
    """Обрабатывает строку-команду (начинается со ``/``). Возвращает True, если это команда."""
    cmd = line[1:].strip()
    if not cmd:
        return True
    verb = cmd.split(maxsplit=1)[0].lower()
    if verb in ("exit", "quit"):
        print("До свидания!")
        raise SystemExit(0)
    if verb in ("help", "?", "помощь"):
        print_help()
        return True
    if verb in ("clear", "очистить"):
        agent.conversation.clear()
        print("История диалога очищена.\n")
        return True
    if verb in ("history", "история", "показать историю"):
        print_history(agent)
        return True
    if verb in ("context", "контекст", "факты", "факт"):
        print_context_status(agent)
        return True
    if verb in ("checkpoint", "чекпоинт", "ветв", "branch", "switch", "переключ"):
        _handle_branch_command(agent, cmd)
        return True
    if verb in ("memory", "память", "remember", "запомни", "forget", "забыть",
                "clearworking", "clear_working", "новая_задача", "сменить_задачу"):
        return _handle_memory_command(agent, cmd)
    if verb in ("profile", "профиль", "profile_new", "новый_профиль",
                "profile_use", "использовать", "активный", "profile_del",
                "удалить_профиль", "profile_off", "отключить",
                "preference", "предпочтение", "preference_del",
                "убрать_предпочтение"):
        return _handle_profile_command(agent, cmd)
    if verb in ("task", "задача", "статус", "task_new", "новая_задача",
                "задача_new", "task_next", "следующий_этап", "далее",
                "task_stage", "этап", "task_step", "шаг",
                "task_expected", "ожидание", "task_complete", "выполнено",
                "готово_этап", "task_auto", "автопродвижение",
                "task_strict", "контроль_этапов",
                "task_pause", "пауза",
                "приостановить", "task_resume", "продолжить", "возобновить",
                "task_done", "готово", "завершить", "task_clear", "сбросить"):
        return _handle_task_command(agent, cmd)
    if verb in ("invariant", "инвариант", "ограничения", "правила",
                "invariant_add", "добавить_инвариант", "новый_инвариант",
                "invariant_del", "удалить_инвариант", "снять_инвариант",
                "invariant_clear", "очистить_инварианты",
                "invariant_enforce", "контроль"):
        return _handle_invariant_command(agent, cmd)
    if verb in ("tools", "инструменты", "mcp", "tool", "вызов", "tool_call"):
        return _handle_tool_command(agent, cmd)
    print(f"Неизвестная команда: /{verb}. Введите /help для списка команд.\n")
    return True


def main() -> None:
    store = SessionStore()
    try:
        agent = pick_session(store)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}. Проверьте config.json.", file=sys.stderr)
        raise SystemExit(1) from exc

    if agent.history:
        print(f"Восстановлен контекст: {len(agent.history)} сообщений.")
    if agent.context_strategy is not None:
        print(
            f"Стратегия управления контекстом: "
            f"{agent.context_strategy.label} ({agent.context_strategy.name})."
        )
    if agent.compression.enabled:
        conv = agent.conversation
        if conv.summary:
            print(
                f"Сжатие истории: включено ({len(conv.messages)} сообщений, "
                f"из них {conv.summarized} свёрнуты в summary)."
            )
        else:
            print("Сжатие истории: включено.")
    if agent.memory is not None:
        print("Модель памяти: включена (краткосрочная / рабочая / долговременная).")
        print("  команды: /memory, /remember <слой> <ключ> = <значение>, "
              "/forget <категория> <ключ>, /clearworking")
    if agent.personalization is not None:
        active = agent.personalization.active_name
        if active:
            print(f"Персонализация: включена, активный профиль: {active}.")
        else:
            print("Персонализация: включена, активного профиля нет.")
        print("  команды: /profile, /profile_new, /profile_use, /preference")
    if agent.task is not None:
        print("Состояние задачи: включено (конечный автомат: этап/шаг/ожидание).")
        print("  команды: /task, /task_new, /task_next, /task_pause, /task_resume")
        if agent.task.has_task():
            print_task_status(agent)
    if agent.invariants is not None:
        print("Инварианты: включены (ограничения, которые нельзя нарушать).")
        print("  команды: /invariant, /invariant_add, /invariant_del, "
              "/invariant_enforce")
        print_invariant_status(agent)
    if agent.tools is not None:
        servers = getattr(agent.tools, "server_names", None)
        if servers:
            print(
                f"MCP-серверы: {len(servers)} ({', '.join(servers)}); "
                "агент сам ведёт длинный флоу между ними."
            )
        else:
            print("MCP-инструменты: подключены (агент вызывает их сам под задачу).")
        print("  команды: /tools, /tool <имя> [<json-аргументы>]")
    print("Введите /help для списка команд.\n")

    while True:
        try:
            user_request = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_request:
            continue
        if user_request.startswith("/"):
            try:
                _dispatch_command(agent, user_request)
            except SystemExit:
                break
            continue

        try:
            reply = agent.ask(user_request)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        except ContextOverflowError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue

        print(f"\nАгент: {reply.content}\n")
        calls = agent.last_tool_calls
        if agent.tools is not None and calls:
            if len(calls) == 1:
                call = calls[0]
                result = agent.last_tool_results[0]
                status = "ошибка" if result.is_error else "результат получен"
                print(
                    f"(MCP-вызов: {call.name} "
                    f"{json.dumps(call.arguments, ensure_ascii=False)} — {status})\n"
                )
            else:
                print(f"(MCP-флоу: вызовов {len(calls)})")
                for index, (call, result) in enumerate(
                    zip(calls, agent.last_tool_results), start=1
                ):
                    server = result.server or "—"
                    status = "ошибка" if result.is_error else "ok"
                    print(
                        f"  {index}. [{server}] {call.name} — {status} "
                        f"{json.dumps(call.arguments, ensure_ascii=False)}"
                    )
                print()
        if agent.invariants is not None and agent.last_guard_violation:
            print("(ассистент отказался: предложенное решение нарушало инвариант)\n")
        if agent.task is not None and agent.last_transition_violation:
            print("(ассистент отказался: попытка перепрыгнуть этап задачи)\n")
        print_usage(agent)
        if agent.task is not None:
            print_task_status(agent)


if __name__ == "__main__":
    main()
