"""CLI-интерфейс: принимает запрос пользователя и выводит ответ агента."""

import json
import sys
import urllib.error

from .agent import Agent
from .compression import CompressionConfig
from .config import load_config
from .context import strategy_from_config
from .conversation import SessionStore
from .llm_client import LLMClient
from .memory import LONG_TERM_KINDS, MemoryLayers
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
    )


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
        print_usage(agent)


if __name__ == "__main__":
    main()
