#!/usr/bin/env python3
"""Day 14: инварианты и ограничения состояния ассистента.

Инварианты — жёсткие рамки, в которых работает ассистент: выбранная архитектура,
принятые технические решения, ограничения по стеку, бизнес-правила. В отличие от
диалога, они **хранятся отдельно** (в ``invariants.json``) и не зависят от темы
разговора. Ассистент обязан их учитывать и **отказываться** от решений, которые
их нарушают.

Сценарий показывает три вещи:

1. **Хранение отдельно от диалога** — инварианты живут в своём файле и не
   попадают в сообщения сессии.
2. **Явный учёт в рассуждении** — их system-сообщение подмешивается к каждому
   запросу (видно в контексте), поэтому модель проверяет решение до того, как
   его предложить.
3. **Отказ при конфликте** — если запрос требует решения, нарушающего инвариант,
   ассистент отказывается и объясняет, какой именно инвариант нарушен и почему.
   Отказ работает двумя путями: сама модель отказывается (следуя system-инструкции),
   и, если модель всё же попыталась нарушить инвариант, — короткая проверка
   (``enforce``) ловит нарушение и заменяет ответ отказом.

По умолчанию офлайн-режим: ответы имитируются локально, и видно, какой инвариант
был подмешан и как ассистент отказался. Флаг --online шлёт реальные запросы к API.

Команды (не являются репликами к модели, начинаются со ``/``):
- ``/invariant``                             — показать все инварианты.
- ``/invariant_add <категория> <описание>`` — добавить инвариант.
- ``/invariant_del <id>``                    — удалить инвариант.
- ``/invariant_enforce``                     — вкл/выкл проверку ответа.
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
from agent.invariants import Invariants
from agent.llm_client import Completion, LLMClient
from agent.tokens import estimate_messages_tokens, estimate_tokens


# Примеры инвариантов: архитектура, техрешения, стек, бизнес-правила.
# По каким словам в запросе офлайн-имитация определяет конфликт — по категории
# (каждая категория в демо встречается один раз).
DEMO_INVARIANTS = [
    {
        "category": "architecture",
        "description": "Сервис строится на микросервисной архитектуре",
        "rationale": "переход на монолит потребует переписывания всей системы",
    },
    {
        "category": "tech_decision",
        "description": "PostgreSQL — основная и единственная БД",
        "rationale": "команда уже приняла это решение, миграция не планируется",
    },
    {
        "category": "stack",
        "description": "Бэкенд — Python 3 + FastAPI",
        "rationale": "стек согласован и его нельзя размывать сторонними языками",
    },
    {
        "category": "business_rule",
        "description": "Персональные данные не покидают территорию страны",
        "rationale": "требование регулятора, нарушение ведёт к штрафам",
    },
]

# Известные технологии/архитектурные понятия, которые офлайн-имитатор умеет
# распознавать. Это нужно, чтобы понять: запрос предлагает сущность, которой
# НЕТ в описании инварианта (значит, инвариант нарушается). Слова с границей
# (``_ENTITIES_WORD``) сопоставляются как целое; длинные русские основы
# (``_ENTITIES_SUBSTR``) — по подстроке, чтобы поймать словоформы.
_ENTITIES_WORD = {
    "python", "php", "mysql", "postgres", "postgresql", "mongodb", "nosql",
    "node.js", "nodejs", "javascript", "typescript", "go", "golang", "rust",
    "c++", "java", "kotlin", "swift", "django", "flask", "fastapi", "spring",
    "react", "vue", "kafka", "redis", "docker", "kubernetes", "aws", "gcp",
    "azure", "microservice", "monolith",
}
_ENTITIES_SUBSTR = {"монолит", "микросервис"}

# Категории, в которых инвариант фиксирует набор технологий/сущностей.

# Домен (категория инварианта) каждой сущности: к какому ограничению она относится.
# Нужно, чтобы конфликт искали в «правильном» инварианте (стек -> stack,
# БД -> tech_decision, архитектура -> architecture, облако -> business_rule),
# а не в произвольном первом.
_ENTITY_DOMAINS = {
    # языки и фреймворки -> стек
    "c++": "stack", "python": "stack", "php": "stack", "javascript": "stack",
    "typescript": "stack", "go": "stack", "golang": "stack", "rust": "stack",
    "java": "stack", "kotlin": "stack", "swift": "stack", "node.js": "stack",
    "nodejs": "stack", "django": "stack", "flask": "stack", "fastapi": "stack",
    "spring": "stack", "react": "stack", "vue": "stack",
    # базы данных -> техрешение
    "mysql": "tech_decision", "postgres": "tech_decision", "postgresql": "tech_decision",
    "mongodb": "tech_decision", "nosql": "tech_decision", "redis": "tech_decision",
    "kafka": "tech_decision",
    # инфраструктура -> техрешение / бизнес-правило
    "docker": "tech_decision", "kubernetes": "tech_decision",
    "aws": "business_rule", "gcp": "business_rule", "azure": "business_rule",
    # архитектура
    "монолит": "architecture", "микросервис": "architecture",
    "microservice": "architecture", "monolith": "architecture",
}


class FakeClient:
    """Имитирует LLM: видит инварианты и умеет отказывать.

    ``forcing_violation`` — режим «слабой» модели: она предлагает решение,
    нарушающее инвариант, и тогда нарушение ловит короткая проверка (enforce).
    В обычном режиме модель, следуя system-инструкции, отказывается сама.
    """

    def __init__(self, config: dict, invariants: Invariants | None = None) -> None:
        self.config = config
        self.invariants = invariants
        self.forcing_violation = False
        self._last_conflict = None

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        if "проверяешь, не нарушает ли ответ инварианты" in joined:
            return self._guard_completion()
        return self._main_completion(joined)

    def _guard_completion(self) -> Completion:
        if self.forcing_violation and self._last_conflict is not None:
            invariant = self._last_conflict
            content = (
                f"НАРУШЕНИЕ: {invariant.id}\n"
                f"ПРИЧИНА: предложен вариант, противоречащий "
                f"{invariant.category_label} ({invariant.description})."
            )
        else:
            content = "ОК"
        return self._completion(content)

    def _main_completion(self, joined: str) -> Completion:
        user_text = self._last_user(joined)
        conflict = self._conflict(user_text)
        self._last_conflict = conflict
        if self.forcing_violation:
            if conflict is not None:
                content = (
                    "Предлагаю переработать систему: откажемся от текущего "
                    f"подхода и выберем вариант с {conflict.description} — "
                    "это проще и быстрее в реализации."
                )
            else:
                content = "Решение не противоречит инвариантам. Готов описать план."
        else:
            if conflict is not None:
                content = self._refusal(conflict)
            else:
                content = (
                    "Решение укладывается в установленные инварианты. "
                    "Могу описать план реализации."
                )
        return self._completion(content)

    def _refusal(self, invariant) -> str:
        reason = f" Причина: {invariant.rationale}." if invariant.rationale else ""
        return (
            f"Я не могу предложить такое решение — оно нарушает инвариант "
            f"{invariant.id} ({invariant.category_label}): {invariant.description}."
            f"{reason} Предложите решение в рамках инвариантов."
        )

    def _conflict(self, user_text: str):
        """Находит инвариант, который нарушает запрос.

        Офлайн-имитация: запрос считается нарушающим, если он предлагает
        технологию/сущность, а инвариант её категории её НЕ разрешает. Например,
        «добавить с++» нарушает инвариант категории stack. Решение ищется только
        в инварианте той же категории, что и сущность (см. ``_ENTITY_DOMAINS``),
        поэтому обычные вопросы про разрешённый стек конфликтом не считаются.
        Это приближение реальной проверки — в online-режиме конфликт определяет
        сама модель.
        """
        if self.invariants is None:
            return None
        requested = self._entities(user_text)
        if not requested:
            return None
        for entity in requested:
            domain = _ENTITY_DOMAINS.get(entity)
            if domain is None:
                continue
            invariant = self._first_invariant_of(domain)
            if invariant is None:
                # Нет инварианта этой категории — ограничения на сущность нет.
                continue
            if entity in self._entities(invariant.description):
                # Инвариант явно разрешает сущность — конфликта нет.
                continue
            return invariant
        return None

    def _first_invariant_of(self, category: str):
        for invariant in self.invariants.all():
            if invariant.category == category:
                return invariant
        return None

    @classmethod
    def _entities(cls, text: str) -> set[str]:
        """Набор известных сущностей (технологий/понятий), найденных в тексте."""
        import re
        low = (text or "").lower().replace("с++", "c++")
        found: set[str] = set()
        # Сопоставление по границам слов (lookaround), чтобы "go" не ловилось
        # внутри "mongodb", а "c++" на конце строки всё же находилось.
        for entity in _ENTITIES_WORD:
            if re.search(r"(?<![a-z0-9а-я])" + re.escape(entity) + r"(?![a-z0-9а-я])", low):
                found.add(entity)
        for entity in _ENTITIES_SUBSTR:
            if entity in low:
                found.add(entity)
        return found

    @staticmethod
    def _last_user(joined: str) -> str:
        lines = joined.splitlines()
        return lines[-1] if lines else ""

    def _completion(self, content: str) -> Completion:
        return Completion(
            content=content,
            model=self.config.get("model", "fake"),
            prompt_tokens=estimate_tokens(content) + 100,
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool, invariants: Invariants | None) -> object:
    if offline:
        return FakeClient(config, invariants)
    return LLMClient(config)


def new_agent(config: dict, offline: bool, invariants: Invariants | None,
              client: object | None = None) -> Agent:
    """Создаёт агента с инвариантами в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    return Agent(
        client or build_client(config, offline, invariants),
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=None,
        context_strategy=None,
        memory=None,
        personalization=None,
        task=None,
        invariants=invariants,
    )


def show_context(agent: Agent) -> None:
    """Печатает сообщения, которые реально уйдут в модель (инварианты + история)."""
    print("Контекст, отправляемый в модель:")
    for message in agent.context_messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            print(f"  [system] {content.splitlines()[0][:90]}...")
        else:
            print(f"  [{role}] {content[:90]}")
    print()


def setup_invariants(path: Path) -> Invariants:
    """Создаёт хранилище инвариантов и наполняет его демо-примерами."""
    invariants = Invariants(path=path, enforce=True)
    invariants.clear()
    for item in DEMO_INVARIANTS:
        invariants.add(item["category"], item["description"], item["rationale"])
    return invariants


def run_scenario(config: dict, offline: bool) -> None:
    print("=" * 62)
    print("Инварианты и ограничения состояния ассистента")
    print("=" * 62)

    # 1. Инварианты хранятся отдельно от диалога.
    tmp = tempfile.TemporaryDirectory()
    invariants_path = Path(tmp.name) / "invariants.json"
    invariants = setup_invariants(invariants_path)
    print("1. Инварианты хранятся отдельно от диалога.")
    print(f"   Файл: {invariants_path.name}")
    print("   В диалоге (сообщениях сессии) инвариантов нет — они в своём файле.")
    print()
    print(invariants.summarize())
    print()

    # 2. Явный учёт в рассуждении: system-сообщение уходит в каждый запрос.
    print("=" * 62)
    print("2. Явный учёт в рассуждении — инварианты уходят в запрос.")
    print("-" * 62)
    fake = FakeClient(config, invariants)
    agent = new_agent(config, offline, invariants, client=fake)
    prompt = invariants.system_message()
    print("system-сообщение инвариантов (подмешивается к каждому запросу):")
    for line in prompt.splitlines():
        print(f"  {line}")
    print()
    show_context(agent)

    # 3а. Конфликт: модель сама отказывается, следуя инвариантам.
    print("=" * 62)
    print("3. Конфликт запроса и инварианта — ассистент отказывается.")
    print("-" * 62)
    conflict_request = "А давайте перейдём на монолитную архитектуру?"
    print(f"Запрос: {conflict_request!r}")
    print()
    reply = agent.ask(conflict_request)
    print(f"Ответ ассистента:\n{reply.content}")
    print(f"(нарушение зафиксировано проверкой: {agent.last_guard_violation})\n")
    show_context(agent)

    # 3б. Слабая модель нарушила инвариант — проверка (enforce) ловит и отбраковывает.
    print("=" * 62)
    print("3б. Слабая модель пытается нарушить инвариант — проверка отбраковывает.")
    print("-" * 62)
    fake.forcing_violation = True
    reply = agent.ask(conflict_request)
    print(f"Ответ ассистента (после проверки):\n{reply.content}")
    print(f"(нарушение зафиксировано проверкой: {agent.last_guard_violation})")
    print("Диалог сессии (проверка, что нарушающий вариант не попал в историю):")
    for message in agent.history:
        if message.get("role") == "assistant":
            print(f"  [assistant] {message.get('content', '')[:90]}...")
    fake.forcing_violation = False
    print()

    # 4. Что происходит при снятии инварианта: запрет исчезает.
    print("=" * 62)
    print("4. Если инвариант устарел и его сняли — запрос становится допустим.")
    print("-" * 62)
    first = invariants.all()[0]
    print(f"Снимаем инвариант {first.id}: {first.description}")
    invariants.remove(first.id)
    print()
    reply = agent.ask(conflict_request)
    print(f"Ответ ассистента:\n{reply.content}\n")


def run_interactive(config: dict, offline: bool) -> None:
    """Живой ввод: пользователь сам добавляет инварианты и смотрит отказы."""
    tmp = tempfile.TemporaryDirectory()
    invariants = Invariants(path=Path(tmp.name) / "invariants.json", enforce=True)
    client = build_client(config, offline, invariants)
    agent = new_agent(config, offline, invariants, client=client)

    print("Инварианты включены. Добавьте свои через /invariant_add.")
    print("Команды со слэшем: /help /invariant /invariant_add /invariant_del "
          "/invariant_enforce /context")
    if offline:
        print("Офлайн-режим: нарушение инварианта определяется локально "
              "(по сущностям в описании).")
        print("  Пример: /invariant_add stack Бэкенд только Python и FastAPI")
        print("  Тогда запрос с упоминанием Node.js будет отклонён.")
    else:
        print("Online-режим: нарушение инварианта определяет сама модель —")
        print("  она видит инварианты в каждом запросе и отказывается, а")
        print("  короткая проверка (enforce) ловит нарушение и подставляет отказ.")
    print()
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
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            print(f"Агент: {reply.content}")
            if agent.last_guard_violation:
                print("  (отказ: решение нарушало инвариант)")
            continue

        cmd = line[1:].strip()
        verb = cmd.split(maxsplit=1)[0].lower()
        if verb in ("help", "?"):
            print(
                "/invariant  /invariant_add <категория> <описание>\n"
                "/invariant_del <id>  /invariant_enforce\n"
                "/context  /exit\n"
            )
            continue
        if verb in ("invariant", "инвариант", "ограничения"):
            print(invariants.summarize())
            print()
            continue
        if verb in ("invariant_add", "добавить_инвариант"):
            rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
            category, _, description = rest.partition(" ")
            try:
                invariant = invariants.add(category, description.strip())
                print(f"Добавлен: {invariant.id} ({invariant.category_label}) "
                      f"{invariant.description}\n")
            except ValueError as exc:
                print(f"{exc}\n")
            continue
        if verb in ("invariant_del", "удалить_инвариант"):
            rest = cmd.split(maxsplit=1)[1] if len(cmd.split()) > 1 else ""
            if invariants.remove(rest):
                print(f"Удалён: {rest}\n")
            else:
                print(f"Не найден: {rest}\n")
            continue
        if verb in ("invariant_enforce", "контроль"):
            invariants.enforce = not invariants.enforce
            print(f"Проверка ответа: {'включена' if invariants.enforce else 'выключена'}\n")
            continue
        if verb in ("context", "контекст"):
            show_context(agent)
            continue
        print(f"Неизвестная команда: /{verb}. Введите /help.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Инварианты и ограничения состояния")
    parser.add_argument("--online", action="store_true", help="реальные запросы к API")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="прогнать фиксированный сценарий вместо живого ввода",
    )
    args = parser.parse_args()

    config = load_config()
    if args.demo:
        run_scenario(config, offline=not args.online)
    else:
        run_interactive(config, offline=not args.online)


if __name__ == "__main__":
    main()
