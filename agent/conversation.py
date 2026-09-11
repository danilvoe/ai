"""Сессии диалога агента: история каждой беседы сохраняется в отдельный JSON."""

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
    """История одной сессии диалога, хранящаяся в JSON-файле."""

    def __init__(self, session_id: str, history_dir: Path = HISTORY_DIR) -> None:
        self._session_id = session_id
        self._path = history_dir / f"{session_id}.json"
        self._messages: list[dict] = []
        self.load()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[dict]:
        return self._messages

    def load(self) -> None:
        """Загружает историю из JSON-файла сессии."""
        if not self._path.exists():
            self._messages = []
            return
        try:
            with self._path.open(encoding="utf-8") as file:
                self._messages = json.load(file)
        except (OSError, json.JSONDecodeError):
            self._messages = []

    def save(self) -> None:
        """Сохраняет историю в JSON-файл сессии."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(self._messages, file, ensure_ascii=False, indent=2)

    def append(self, role: str, content: str) -> None:
        """Добавляет сообщение и сохраняет историю."""
        self._messages.append({"role": role, "content": content})
        self._trim()
        self.save()

    def pop_last(self) -> None:
        """Удаляет последнее сообщение и сохраняет историю."""
        if self._messages:
            self._messages.pop()
            self.save()

    def clear(self) -> None:
        """Очищает историю сессии."""
        self._messages = []
        self.save()

    def _trim(self) -> None:
        """Обрезает историю до последних MAX_MESSAGES сообщений."""
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]


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
