"""Инварианты и ограничения состояния агента — то, что ассистент не может нарушать.

Инварианты — это жёсткие рамки, в которых работает ассистент: выбранная
архитектура, принятые технические решения, ограничения по стеку, бизнес-правила.
Они **не являются частью диалога**: хранятся в отдельном файле ``invariants.json``
и не зависят от того, о чём идёт разговор. Диалог может менять тему, а инварианты
остаются неизменными.

Инварианты учитываются в рассуждении двумя способами:

1. **Явно**: их system-сообщение подмешивается к каждому запросу (см.
   ``Invariants.system_message``), поэтому модель видит инварианты до того, как
   предложит решение, и должна проверить его на соответствие.
2. **Контролем**: после ответа ассистента агент может прогнать короткую проверку
   (``Invariants.guard_check_messages`` + ``parse_guard_verdict``) — не нарушает ли
   предложенное решение инварианты. Если нарушает, ответ заменяется отказом с
   объяснением (``Invariants.refusal_message``).

Инварианты структурированы: каждый имеет ``category`` (архитектура / техническое
решение / стек / бизнес-правило / прочее), описание и обоснование (зачем его
нельзя нарушать). Обоснование попадает и в запрос, и в текст отказа — ассистент
не просто отказывает, а объясняет, **какой именно** инвариант нарушен и почему.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Категории инвариантов.
INVARIANT_CATEGORIES = (
    "architecture",       # выбранная архитектура
    "tech_decision",      # принятые технические решения
    "stack",              # ограничения по стеку
    "business_rule",      # бизнес-правила
    "custom",             # прочее
)

INVARIANT_LABELS = {
    "architecture": "Архитектура",
    "tech_decision": "Техническое решение",
    "stack": "Ограничение стека",
    "business_rule": "Бизнес-правило",
    "custom": "Прочее",
}

# Промпт короткой проверки: решает LLM, нарушает ли ответ инварианты.
GUARD_PROMPT = (
    "Ты проверяешь, не нарушает ли ответ ассистента установленные инварианты.\n"
    "Инварианты — это жёсткие рамки, которые нельзя нарушать ни при каких "
    "обстоятельствах.\n\n"
    "Установленные инварианты:\n{invariants}\n\n"
    "Запрос пользователя:\n{user_request}\n\n"
    "Ответ ассистента:\n{assistant_reply}\n\n"
    "Если ответ предлагает решение, которое нарушает хотя бы один инвариант, "
    "ответь строго в две строки:\n"
    "НАРУШЕНИЕ: <id нарушенного инварианта, через запятую>\n"
    "ПРИЧИНА: <объяснение, что именно и какой инвариант нарушено>\n"
    "Если нарушений нет, ответь ровно одним словом: ОК."
)

_GUARD_MARKER = "проверяешь, не нарушает ли ответ инварианты"


def default_invariants_path(history_dir: Path | None = None) -> Path:
    """Путь к файлу инвариантов (по умолчанию рядом с историей)."""
    base = history_dir or (Path(__file__).resolve().parent.parent / "history")
    return base / "invariants.json"


def parse_guard_verdict(text: str) -> "GuardVerdict":
    """Разбирает ответ модели на проверку нарушения инвариантов.

    Возвращает :class:`GuardVerdict`: ``violated`` — нарушен ли инвариант,
    ``invariant_ids`` — id нарушенных инвариантов, ``reason`` — объяснение.
    """
    token = (text or "").strip()
    if not token:
        return GuardVerdict(violated=False, invariant_ids=[], reason="")
    upper = token.upper()
    if "НАРУШЕНИЕ" not in upper:
        return GuardVerdict(violated=False, invariant_ids=[], reason="")
    ids: list[str] = []
    reason = ""
    for line in token.splitlines():
        line = line.strip()
        if line.upper().startswith("НАРУШЕНИЕ"):
            _, _, value = line.partition(":")
            ids = [item.strip() for item in value.split(",") if item.strip()]
        elif line.upper().startswith("ПРИЧИНА"):
            _, _, value = line.partition(":")
            reason = value.strip()
    return GuardVerdict(
        violated=bool(ids) or bool(reason),
        invariant_ids=ids,
        reason=reason,
    )


@dataclass
class GuardVerdict:
    """Вердикт проверки ответа на нарушение инвариантов."""

    violated: bool
    invariant_ids: list[str]
    reason: str


@dataclass
class Invariant:
    """Один инвариант: жёсткое ограничение, которое ассистент не может нарушить.

    ``category`` — из :data:`INVARIANT_CATEGORIES`. ``rationale`` — обоснование,
    зачем это нельзя нарушать (показывается и в запросе, и в тексте отказа).
    ``id`` — короткий идентификатор (например ``inv1``), используется в командах
    и в вердикте проверки.
    """

    id: str
    category: str = "custom"
    description: str = ""
    rationale: str = ""
    severity: str = "hard"  # hard | soft — жёсткий инвариант нельзя обойти

    def __post_init__(self) -> None:
        self.category = _validate_category(self.category)
        self.id = str(self.id).strip()
        self.description = str(self.description).strip()
        self.rationale = str(self.rationale).strip()

    @property
    def category_label(self) -> str:
        return INVARIANT_LABELS.get(self.category, self.category)

    def system_line(self) -> str:
        """Строка для system-сообщения: инвариант в человекочитаемом виде."""
        line = f"- [{self.category}] {self.id}: {self.description}"
        if self.rationale:
            line += f" (причина: {self.rationale})"
        return line

    def summary_line(self) -> str:
        """Строка для сводки (для вывода пользователю)."""
        line = f"{self.id} ({self.category_label}): {self.description}"
        if self.rationale:
            line += f" — {self.rationale}"
        return line

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "rationale": self.rationale,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Invariant":
        data = data or {}
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "custom")),
            description=str(data.get("description", "")),
            rationale=str(data.get("rationale", "")),
            severity=str(data.get("severity", "hard")),
        )

    def __repr__(self) -> str:
        return f"Invariant(id={self.id!r}, category={self.category!r}, description={self.description!r})"


def _validate_category(category: str) -> str:
    category = str(category).strip().lower()
    if category not in INVARIANT_CATEGORIES:
        raise ValueError(
            f"Неизвестная категория инварианта: {category!r}. "
            f"Допустимо: {', '.join(INVARIANT_CATEGORIES)}."
        )
    return category


class Invariants:
    """Хранилище инвариантов, отделённое от диалога.

    Инварианты живут в собственном JSON-файле ``invariants.json`` и не зависят от
    сообщений сессии. Их можно добавлять/удалять через команды, а их
    system-сообщение подмешивается к каждому запросу (см. agent/agent.py), так что
    ассистент учитывает их в рассуждении. Опционально включается проверка ответа
    (``enforce``), которая отбраковывает решения, нарушающие инварианты.
    """

    def __init__(self, path: Path = default_invariants_path(), enforce: bool = True) -> None:
        self._path = Path(path)
        self._enforce = bool(enforce)
        self._invariants: list[Invariant] = []
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def enforce(self) -> bool:
        """Проверять ли ответ агента на нарушение инвариантов."""
        return self._enforce

    @enforce.setter
    def enforce(self, value: bool) -> None:
        self._enforce = bool(value)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        items = data.get("invariants", [])
        if isinstance(items, list):
            self._invariants = [Invariant.from_dict(item) for item in items if isinstance(item, dict)]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(
                {"invariants": [inv.to_dict() for inv in self._invariants]},
                file,
                ensure_ascii=False,
                indent=2,
            )

    def add(self, category: str, description: str, rationale: str = "",
            invariant_id: str | None = None) -> Invariant:
        """Добавляет инвариант и возвращает его. id генерируется автоматически."""
        category = _validate_category(category)
        if not str(description).strip():
            raise ValueError("Описание инварианта не может быть пустым.")
        if invariant_id is None:
            invariant_id = f"inv{len(self._invariants) + 1}"
        invariant = Invariant(
            id=invariant_id,
            category=category,
            description=description,
            rationale=rationale,
        )
        self._invariants.append(invariant)
        self._save()
        return invariant

    def remove(self, invariant_id: str) -> bool:
        """Удаляет инвариант по id. Возвращает True, если он был."""
        before = len(self._invariants)
        self._invariants = [inv for inv in self._invariants if inv.id != invariant_id]
        if len(self._invariants) != before:
            self._save()
            return True
        return False

    def get(self, invariant_id: str) -> Invariant | None:
        """Инвариант по id (None, если не найден)."""
        for invariant in self._invariants:
            if invariant.id == invariant_id:
                return invariant
        return None

    def all(self) -> list[Invariant]:
        """Все инварианты в порядке добавления."""
        return list(self._invariants)

    def clear(self) -> None:
        """Удаляет все инварианты."""
        if self._invariants:
            self._invariants = []
            self._save()

    def __bool__(self) -> bool:
        return bool(self._invariants)

    def __len__(self) -> int:
        return len(self._invariants)

    def system_message(self) -> str:
        """System-сообщение с инвариантами (подмешивается к каждому запросу).

        Модель получает инварианты до ответа и обязана проверить предлагаемое
        решение. Если решение нарушает инвариант — отказаться и объяснить.
        Возвращает пустую строку, если инвариантов нет.
        """
        if not self._invariants:
            return ""
        lines = [
            "Установленные инварианты (нарушать их НЕЛЬЗЯ ни при каких обстоятельствах):"
        ]
        for invariant in self._invariants:
            lines.append(invariant.system_line())
        lines.append("")
        lines.append(
            "Перед тем как предложить решение, явно проверь его на соответствие "
            "каждому инварианту. Если предлагаемое решение нарушает хотя бы один "
            "инвариант — НЕ предлагай его. Откажись и объясни, какой именно "
            "инвариант нарушен и почему. Нельзя «немного нарушить», обойти "
            "инвариант или предложить его как компромисс."
        )
        return "\n".join(lines)

    def guard_check_messages(self, user_request: str, assistant_reply: str) -> list[dict]:
        """Сообщения короткой проверки: нарушает ли ответ инварианты.

        Возвращает пустой список, если инвариантов нет (проверять нечего).
        """
        if not self._invariants:
            return []
        invariants_block = "\n".join(inv.summary_line() for inv in self._invariants)
        content = GUARD_PROMPT.format(
            invariants=invariants_block,
            user_request=user_request,
            assistant_reply=assistant_reply,
        )
        return [
            {"role": "system", "content": _GUARD_MARKER},
            {"role": "user", "content": content},
        ]

    def refusal_message(self, verdict: GuardVerdict) -> str:
        """Текст отказа: объясняет, какой инвариант нарушен и почему."""
        lines = [
            "Я не могу предложить такое решение — оно нарушает установленные инварианты."
        ]
        for invariant_id in verdict.invariant_ids:
            invariant = self.get(invariant_id)
            if invariant is not None:
                lines.append(
                    f"- Инвариант {invariant.id} ({invariant.category_label}): "
                    f"{invariant.description}"
                )
                if invariant.rationale:
                    lines.append(f"  Причина: {invariant.rationale}")
        if verdict.reason:
            lines.append(f"- Проверка: {verdict.reason}")
        lines.append(
            "Предложите решение в рамках инвариантов, либо измените сам инвариант "
            "(если он устарел) через команду /invariant_del."
        )
        return "\n".join(lines)

    def summarize(self) -> str:
        """Краткая человекочитаемая сводка всех инвариантов."""
        if not self._invariants:
            return "Инвариантов нет."
        lines = [f"Инварианты ({len(self._invariants)}):"]
        for invariant in self._invariants:
            lines.append(f"  - {invariant.summary_line()}")
        lines.append(f"Проверка ответа: {'включена' if self._enforce else 'выключена'}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Invariants({len(self._invariants)} invariants, enforce={self._enforce})"
