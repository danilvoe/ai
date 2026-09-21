#!/usr/bin/env python3
"""Day 12: персонализация ассистента поверх модели памяти.

Персонализация — отдельный слой над моделью памяти (см. agent/memory.py).
Агент держит **профиль пользователя**: личность и три группы предпочтений —
**стиль** (как отвечать), **формат** (как структурировать) и **ограничения**
(чего избегать). Профиль **автоматически** подмешивается к каждому запросу:
его system-сообщение вставляется первым, и ассистент учитывает его без того,
чтобы пользователь повторял требования каждый раз.

Сценарий показывает:

1. Как выглядит профиль (личность + стиль + формат + ограничения).
2. Что профиль уходит в **каждый** запрос (system-сообщение видно в контексте).
3. Что **разные профили дают разные ответы** на один и тот же вопрос —
   ассистент адаптируется автоматически.

Сценарий запускает один и тот же вопрос через несколько профилей:
- **Инженер** — кратко, по делу, без воды, с кодом и списками.
- **Менеджер** — структурировано, с заголовками и итогами, дружелюбно.
- **Новичок** — просто, пошагово, без жаргона, с примерами.

По умолчанию офлайн-режим: ответы имитируются локально и видно, какой профиль
был подмешан. Флаг --online шлёт реальные запросы к API, и ассистент реально
отвечает по-разному в зависимости от профиля.

Команды (не являются репликами к модели, начинаются со ``/``):
- ``/profiles``                              — показать все профили и активный.
- ``/use <имя>``                             — сделать профиль активным.
- ``/preference <группа> <ключ> = <значение>``— задать предпочтение.
- ``/context``                               — показать, что реально уйдёт в модель.
- ``/help`` / ``/exit``                      — помощь / завершить.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.personalization import Personalization, UserProfile
from agent.tokens import estimate_messages_tokens, estimate_tokens


# Три разных профиля: один вопрос, разные предпочтения.
BUILTIN_PROFILES = {
    "инженер": UserProfile(
        name="Инженер",
        identity="Опытный разработчик, ценит точность и лаконичность.",
        style={
            "тон": "нейтральный, технический",
            "краткость": "отвечай коротко и по делу, без воды",
            "эмодзи": "не используй",
        },
        format={
            "структура": "маркированные списки и код",
            "код": "примеры кода в блоках",
        },
        constraints={
            "вода": "не разворачивай вступление",
            "детали": "не перечисляй очевидное",
        },
    ),
    "менеджер": UserProfile(
        name="Менеджер",
        identity="Руководитель проекта, любит итоги и рекомендации.",
        style={
            "тон": "дружелюбный, деловой",
            "краткость": "средняя длина, но по существу",
            "эмодзи": "допустимы изредка",
        },
        format={
            "структура": "заголовки, в конце — краткий итог",
            "таблицы": "сравнения своди в таблицу",
        },
        constraints={
            "вода": "не углубляйся в технические детали",
            "рекомендация": "всегда давай вывод и следующий шаг",
        },
    ),
    "новичок": UserProfile(
        name="Новичок",
        identity="Начинающий, объясняй просто и без жаргона.",
        style={
            "тон": "тёплый, терпеливый",
            "краткость": "достаточно подробно",
        },
        format={
            "структура": "пошагово, с примерами",
            "термины": "жаргон объясняй сразу",
        },
        constraints={
            "жаргон": "не используй сленг без пояснения",
        },
    ),
}


class FakeClient:
    """Имитирует LLM: показывает, какой профиль и сколько сообщений ушло в модель."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        # Профиль — первое system-сообщение (если есть).
        system_blocks = [m for m in messages if m.get("role") == "system"]
        profile_block = system_blocks[0] if system_blocks else None
        user_messages = [m for m in messages if m.get("role") == "user"]
        header = ""
        if profile_block is not None:
            first_line = profile_block["content"].splitlines()[0]
            header = f"Вижу профиль: {first_line} · "
        content = (
            header
            + f"Подмешанных system-блоков: {len(system_blocks)}; "
            + f"сообщений от пользователя: {len(user_messages)}"
        )
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


def profiles_path_from_config(config: dict) -> Path:
    """Путь к файлу профилей из конфигурации (или по умолчанию)."""
    section = config.get("personalization", {}) or {}
    path = section.get("profiles_path")
    if path:
        return Path(path)
    return Path(__file__).resolve().parent.parent / "history" / "profiles.json"


def new_agent(config: dict, offline: bool, personalization: Personalization) -> Agent:
    """Создаёт агента с персонализацией в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    return Agent(
        build_client(config, offline),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=None,
        memory=None,
        personalization=personalization,
    )


def show_context(agent: Agent) -> None:
    """Печатает сообщения, которые реально уйдут в модель (профиль + история)."""
    print("Контекст, отправляемый в модель:")
    for message in agent.context_messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            print(f"  [system] {content.splitlines()[0][:90]}...")
        else:
            print(f"  [{role}] {content[:90]}")
    print()


def run_scenario(config: dict, offline: bool, query: str) -> None:
    """Прогоняет один вопрос через каждый профиль и показывает результат."""
    print("=" * 62)
    print("Персонализация ассистента поверх модели памяти")
    print("=" * 62)
    print(f"Вопрос, который задаётся каждому профилю: {query!r}\n")

    personalization = Personalization(path=profiles_path_from_config(config))
    for name, profile in BUILTIN_PROFILES.items():
        personalization.save(name, profile)
    personalization.set_active("инженер")

    print("Профили для сравнения:")
    for name, profile in BUILTIN_PROFILES.items():
        entries = len(profile.style) + len(profile.format) + len(profile.constraints)
        print(f"  - {name}: {entries} предпочтений")
    print()

    results: dict[str, str] = {}
    for name in BUILTIN_PROFILES:
        personalization.set_active(name)
        agent = new_agent(config, offline, personalization)

        print("=" * 62)
        print(f"Профиль: {name} — {BUILTIN_PROFILES[name].identity}")
        print("-" * 62)
        prompt = personalization.system_message()
        print("system-сообщение профиля (подмешивается к запросу):")
        for line in prompt.splitlines():
            print(f"  {line}")
        print("-" * 62)

        reply = agent.ask(query)
        print(f"Ответ ассистента: {reply.content}")
        print(f"Токены запроса с профилем: {agent.context_tokens}")
        print()
        results[name] = reply.content

    # Показываем, что разным профилям уходят разные system-сообщения.
    print("=" * 62)
    print("Вывод: профиль автоматически учитывается в каждом запросе.")
    print("Каждому профилю ушло своё system-сообщение, поэтому и ответы отличаются.")
    for name, content in results.items():
        print(f"  {name}: {content[:70]}")
    print()

    # Финальный профиль остаётся активным — покажем контекст следующего запроса.
    personalization.set_active("новичок")
    agent = new_agent(config, offline, personalization)
    print("Проверка: без повторных инструкций профиль уже ушёл в следующий запрос.")
    reply = agent.ask(query)
    print(f"Ответ (профиль {personalization.active_name}): {reply.content}")
    show_context(agent)


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам переключает профили и смотрит контекст."""
    personalization = Personalization(path=profiles_path_from_config(config))
    for name, profile in BUILTIN_PROFILES.items():
        personalization.save(name, profile)
    personalization.set_active("инженер")
    agent = new_agent(config, offline, personalization)

    print("Персонализация включена. Профили: " + ", ".join(BUILTIN_PROFILES))
    print("Активный профиль: " + personalization.active_name)
    print("Команды со слэшем: /help /profiles /use <имя> /preference /context")
    while True:
        try:
            line = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        if not line.startswith("/"):
            try:
                reply = agent.ask(line)
            except Exception as exc:  # noqa: BLE001 — офлайн/онлайн общая обработка
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            print(f"Агент: {reply.content}")
            continue

        cmd = line[1:].strip()
        verb = cmd.split(maxsplit=1)[0].lower()
        if verb in ("help", "?"):
            print(
                "/profiles  /use <имя>  /preference <группа> <ключ> = <значение>\n"
                "/context  /exit\n"
            )
            continue
        if verb in ("profiles", "профиль"):
            print(personalization.summarize())
            if personalization.active_name:
                print("Активный профиль:\n" + personalization.system_message())
            print()
            continue
        if verb in ("use", "использовать", "активный"):
            name = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
            if personalization.set_active(name):
                print(f"Активный профиль: {name}\n")
            else:
                print(f"Профиль {name} не найден. Есть: "
                      f"{', '.join(personalization.profiles())}\n")
            continue
        if verb in ("preference", "предпочтение"):
            if personalization.active_name is None:
                print("Нет активного профиля.\n")
                continue
            rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
            group, _, right = rest.partition(" ")
            key, _, value = right.partition("=")
            group, key, value = group.strip().lower(), key.strip(), value.strip()
            profile = personalization.active()
            try:
                profile.set_preference(group, key, value)
                personalization.save(personalization.active_name, profile)
                print(f"Задано: {group} → {key} = {value}\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("context", "контекст"):
            show_context(agent)
            continue
        print(f"Неизвестная команда: /{verb}. Введите /help.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Персонализация ассистента")
    parser.add_argument("--online", action="store_true", help="реальные запросы к API")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="прогнать фиксированный сценарий вместо живого ввода",
    )
    parser.add_argument(
        "--query",
        default="Расскажи, как начать работать с новым проектом?",
        help="вопрос, который задаётся каждому профилю",
    )
    args = parser.parse_args()

    config = load_config()
    if args.demo:
        run_scenario(config, offline=not args.online, query=args.query)
    else:
        run_interactive(config, offline=not args.online)


if __name__ == "__main__":
    main()
