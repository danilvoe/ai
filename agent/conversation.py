"""Сессии диалога агента: история каждой беседы сохраняется в отдельный JSON.

Сессия хранит полную историю ``messages`` (для просмотра пользователем) и
отдельное поле ``summary`` — краткое содержание старой части диалога. Когда
включается сжатие, в запрос к модели подставляется ``summary + последние
сообщения``, но сами старые сообщения не удаляются, а остаются в истории.
Счётчик ``summarized`` отмечает, сколько первых сообщений уже свернуто в
summary (см. agent/compression.py).

Помимо этого сессия поддерживает данные для стратегий управления контекстом
(см. agent/context.py):

- ``facts`` — блок "липких фактов" (ключ-значение) с важными данными диалога;
- ``branches``/``checkpoint`` — ветвление диалога: общая часть до чекпоинта
  плюс независимые ветки, между которыми можно переключаться.
"""

import json
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent.parent / "history"
MAX_MESSAGES = 100


def _new_session_id(existing: set[str] | None = None) -> str:
    """Создаёт уникальный идентификатор новой сессии по текущему времени."""
    existing = existing or set()
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    session_id = base
    counter = 1
    while session_id in existing:
        session_id = f"{base}-{counter}"
        counter += 1
    return session_id


class Conversation:
    """История одной сессии диалога, хранящаяся в JSON-файле.

    В файле хранится объект вида::

        {
            "summary": "...",        # краткое содержание старой части
            "summarized": 42,        # сколько первых сообщений свернуто в summary
            "messages": [...],       # активная ветка диалога (для просмотра)
            "facts": {...},          # липкие факты (ключ-значение)
            "working": {...},        # рабочая память: данные текущей задачи
            "checkpoint": [...],     # общая часть диалога до точки ветвления
            "branches": {...},       # независимые ветки (id -> {name, messages})
            "active_branch": "id",   # какая ветка сейчас активна
        }

    Формат старой версии (простой массив сообщений или объект без ``summarized``)
    поддерживается при чтении, чтобы существующие сессии продолжали открываться.
    """

    def __init__(self, session_id: str, history_dir: Path = HISTORY_DIR) -> None:
        self._session_id = session_id
        self._path = history_dir / f"{session_id}.json"
        self._messages: list[dict] = []
        self._summary = ""
        self._summarized = 0
        self._facts: dict[str, str] = {}
        self._working: dict[str, str] = {}
        self._checkpoint: list[dict] = []
        self._branches: dict[str, dict] = {}
        self._active_branch: str | None = None
        self.load()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[dict]:
        """Активный диалог: общая часть (checkpoint) + сообщения активной ветки."""
        if self._active_branch is not None:
            return list(self._checkpoint) + list(self._branches[self._active_branch]["messages"])
        return list(self._checkpoint)

    @property
    def summary(self) -> str:
        """Краткое содержание старой части диалога (пусто, если сжатия не было)."""
        return self._summary

    @property
    def summarized(self) -> int:
        """Сколько первых сообщений истории уже свёрнуто в summary."""
        return self._summarized

    @property
    def recent_messages(self) -> list[dict]:
        """Сообщения, которые ещё не свёрнуты в summary (свежее окно)."""
        return self.messages[self._summarized:]

    @property
    def facts(self) -> dict[str, str]:
        """Блок "липких фактов" (ключ-значение), накопленных за диалог."""
        return dict(self._facts)

    @property
    def working(self) -> dict[str, str]:
        """Рабочая память: данные текущей задачи (ключ-значение)."""
        return dict(self._working)

    @property
    def active_branch(self) -> str | None:
        """Имя активной ветки диалога (None, если ветвление не начато)."""
        return self._active_branch

    @property
    def branches(self) -> dict[str, dict]:
        """Все ветки диалога: id -> {name, messages} (без общих сообщений)."""
        return {
            branch_id: {"name": branch["name"], "messages": list(branch["messages"])}
            for branch_id, branch in self._branches.items()
        }

    @property
    def has_branches(self) -> bool:
        """Есть ли в диалоге созданные ветки."""
        return bool(self._branches)

    def _reset(self) -> None:
        self._summary = ""
        self._summarized = 0
        self._facts = {}
        self._working = {}
        self._checkpoint = []
        self._branches = {}
        self._active_branch = None

    def load(self) -> None:
        """Загружает историю из JSON-файла сессии."""
        if not self._path.exists():
            self._reset()
            return
        try:
            with self._path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            self._reset()
            return

        if isinstance(data, dict):
            self._summary = data.get("summary", "") or ""
            self._summarized = data.get("summarized", 0) or 0
            self._facts = data.get("facts", {}) or {}
            self._working = data.get("working", {}) or {}
            self._branches = data.get("branches", {}) or {}
            self._active_branch = data.get("active_branch")
            checkpoint = data.get("checkpoint")
            messages = data.get("messages", []) or []
            # Старый формат не хранит checkpoint: вся история в messages.
            self._checkpoint = checkpoint if checkpoint is not None else messages
            if self._active_branch and self._active_branch not in self._branches:
                self._active_branch = None
            if self._summarized > len(self.messages):
                self._summarized = len(self.messages)
        else:
            self._reset()
            self._checkpoint = data

    def save(self) -> None:
        """Сохраняет историю в JSON-файл сессии."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "summary": self._summary,
                    "summarized": self._summarized,
                    "messages": self.messages,
                    "facts": self._facts,
                    "working": self._working,
                    "checkpoint": self._checkpoint,
                    "branches": self._branches,
                    "active_branch": self._active_branch,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def append(self, role: str, content: str) -> None:
        """Добавляет сообщение в активную ветку (или общую часть) и сохраняет."""
        message = {"role": role, "content": content}
        if self._active_branch is not None:
            self._branches[self._active_branch]["messages"].append(message)
        else:
            self._checkpoint.append(message)
        self._trim()
        self.save()

    def set_summary(self, summary: str, summarized: int | None = None) -> None:
        """Устанавливает краткое содержание старой части диалога.

        ``summarized`` — сколько первых сообщений теперь покрыто summary.
        """
        self._summary = summary
        if summarized is not None:
            self._summarized = min(max(0, summarized), len(self.messages))
        self.save()

    def pop_last(self) -> None:
        """Удаляет последнее сообщение активной ветки и сохраняет."""
        if self._active_branch is not None:
            branch = self._branches[self._active_branch]
            if branch["messages"]:
                branch["messages"].pop()
                self.save()
        elif self._checkpoint:
            self._checkpoint.pop()
            self.save()

    def clear(self) -> None:
        """Очищает историю, summary, факты и ветки сессии."""
        self._reset()
        self.save()

    def trim_to(self, keep: int) -> None:
        """Жёстко отбрасывает всё, кроме последних ``keep`` сообщений (без summary).

        Используется стратегией Sliding Window: старые сообщения не просто не
        отправляются, а физически удаляются из активной ветки диалога.
        """
        if self._active_branch is not None:
            branch = self._branches[self._active_branch]
            total = len(self._checkpoint) + len(branch["messages"])
            if total <= keep:
                return
            to_remove = total - keep
            if len(branch["messages"]) >= to_remove:
                branch["messages"] = branch["messages"][to_remove:]
            else:
                remainder = to_remove - len(branch["messages"])
                branch["messages"] = []
                self._checkpoint = self._checkpoint[remainder:]
        else:
            if len(self._checkpoint) > keep:
                self._checkpoint = self._checkpoint[-keep:]
        self.save()

    def set_fact(self, key: str, value: str) -> None:
        """Записывает один липкий факт и сохраняет."""
        self._facts[key] = value
        self.save()

    def update_facts(self, facts: dict[str, str]) -> None:
        """Добавляет/обновляет несколько липких фактов и сохраняет."""
        if facts:
            self._facts.update(facts)
            self.save()

    def set_working(self, key: str, value: str) -> None:
        """Записывает одну запись в рабочую память (данные текущей задачи)."""
        self._working[key] = value
        self.save()

    def update_working(self, items: dict[str, str]) -> None:
        """Добавляет/обновляет несколько записей рабочей памяти."""
        if items:
            self._working.update(items)
            self.save()

    def clear_working(self) -> None:
        """Очищает рабочую память (например, при смене текущей задачи)."""
        if self._working:
            self._working = {}
            self.save()

    def checkpoint(self) -> int:
        """Сохраняет текущее состояние как общую точку ветвления.

        Все сообщения становятся общим префиксом (checkpoint), существующие
        ветки сбрасываются. Возвращает число сообщений в общей части.
        """
        self._checkpoint = self.messages
        self._branches = {}
        self._active_branch = None
        self._summarized = 0
        self.save()
        return len(self._checkpoint)

    def create_branch(self, name: str) -> bool:
        """Создаёт независимую ветку от точки ветвления и переключается на неё.

        Каждая ветка начинается с чистой общей части (checkpoint). Возвращает
        False, если checkpoint ещё не создан, имя пустое или ветка уже есть.
        """
        if not self._checkpoint or not name or name in self._branches:
            return False
        self._branches[name] = {"name": name, "messages": []}
        self._active_branch = name
        self.save()
        return True

    def switch_branch(self, name: str) -> bool:
        """Переключает активную ветку. Возвращает False, если такой ветки нет."""
        if name not in self._branches:
            return False
        self._active_branch = name
        self.save()
        return True

    def _trim(self) -> None:
        """Обрезает активную ветку до последних MAX_MESSAGES сообщений."""
        total = len(self.messages)
        if total <= MAX_MESSAGES:
            return
        self.trim_to(MAX_MESSAGES)
        if self._summarized > MAX_MESSAGES:
            self._summarized = MAX_MESSAGES


class SessionStore:
    """Управляет списком доступных сессий диалога."""

    def __init__(self, history_dir: Path = HISTORY_DIR) -> None:
        self._history_dir = history_dir

    def list_sessions(self) -> list[str]:
        """Возвращает идентификаторы сессий, отсортированные от старых к новым."""
        if not self._history_dir.exists():
            return []
        return sorted(p.stem for p in self._history_dir.glob("*.json"))

    def create(self) -> Conversation:
        """Создаёт и возвращает новую пустую сессию."""
        return Conversation(_new_session_id(set(self.list_sessions())), self._history_dir)

    def get(self, session_id: str) -> Conversation:
        """Открывает существующую сессию."""
        return Conversation(session_id, self._history_dir)
