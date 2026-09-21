"""Персонализация ассистента поверх модели памяти.

Персонализация — это отдельный слой над моделью памяти (см. agent/memory.py):
агент не просто помнит факты о пользователе, а держит **профиль** — описание
пользователя и его предпочтений, и **автоматически** подмешивает его к каждому
запросу к модели. Пользователю не нужно повторять "отвечай коротко и по делу",
"избегай воды" — эти требования уже зашиты в профиль и уходят в модель сами.

Профиль описывает три группы предпочтений:

- **Стиль** (``style``) — как отвечать: тон, формальность, лаконичность,
  эмодзи, юмор.
- **Формат** (``format``) — как структурировать ответ: маркированные списки,
  заголовки, таблицы, markdown, код, пошаговость.
- **Ограничения** (``constraints``) — чего избегать: вода, лишние детали,
  определённые темы, длина ответа, язык.

Плюс блок **личности** (``identity``) — кто пользователь и его контекст, чтобы
ассистент говорил с ним на подходящем уровне.

Профили хранятся в отдельном файле ``profiles.json`` и переживают сессии.
Объект :class:`Personalization` хранит все профили и активный профиль; метод
:meth:`Personalization.system_message` собирает из активного профиля system-
сообщение, которое агент добавляет к каждому запросу (см. agent/agent.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Группы предпочтений профиля.
PREFERENCE_GROUPS = ("style", "format", "constraints")

PREFERENCE_LABELS = {
    "style": "Стиль (как отвечать)",
    "format": "Формат (как структурировать)",
    "constraints": "Ограничения (чего избегать)",
}

# Шаблоны по умолчанию для нового профиля.
DEFAULT_PROFILE = {
    "name": "",
    "identity": "",
    "style": {},
    "format": {},
    "constraints": {},
}


def default_profiles_path(history_dir: Path | None = None) -> Path:
    """Путь к файлу профилей (по умолчанию рядом с историей)."""
    base = history_dir or (Path(__file__).resolve().parent.parent / "history")
    return base / "profiles.json"


@dataclass
class UserProfile:
    """Профиль пользователя: личность + предпочтения (стиль/формат/ограничения)."""

    name: str = ""
    identity: str = ""
    style: dict[str, str] = field(default_factory=dict)
    format: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Восстанавливает профиль из словаря (например, из JSON-файла)."""
        data = data or {}
        return cls(
            name=str(data.get("name", "")),
            identity=str(data.get("identity", "")),
            style={str(k): str(v) for k, v in (data.get("style") or {}).items()},
            format={str(k): str(v) for k, v in (data.get("format") or {}).items()},
            constraints={
                str(k): str(v) for k, v in (data.get("constraints") or {}).items()
            },
        )

    def to_dict(self) -> dict:
        """Сериализует профиль в словарь для сохранения в JSON."""
        return {
            "name": self.name,
            "identity": self.identity,
            "style": dict(self.style),
            "format": dict(self.format),
            "constraints": dict(self.constraints),
        }

    def set_preference(self, group: str, key: str, value: str) -> None:
        """Записывает одно предпочтение в нужную группу (style/format/constraints)."""
        group = self._validate_group(group)
        getattr(self, group)[key] = value

    def remove_preference(self, group: str, key: str) -> bool:
        """Удаляет предпочтение. Возвращает True, если оно было."""
        group = self._validate_group(group)
        return getattr(self, group).pop(key, None) is not None

    def _validate_group(self, group: str) -> str:
        group = str(group).strip().lower()
        if group not in PREFERENCE_GROUPS:
            raise ValueError(
                f"Неизвестная группа предпочтений: {group!r}. "
                f"Допустимо: {', '.join(PREFERENCE_GROUPS)}."
            )
        return group

    def _preference_lines(self, group: str) -> list[str]:
        entries = getattr(self, group)
        if not entries:
            return []
        lines = []
        for key, value in entries.items():
            if value.strip().lower() in ("да", "true", "1", "+"):
                lines.append(f"- {key}")
            elif value.strip():
                lines.append(f"- {key}: {value}")
        return lines

    def system_message(self) -> str:
        """Собирает system-сообщение профиля для подстановки в запрос.

        Если профиль пустой (нет ни личности, ни предпочтений), возвращает
        пустую строку — блок не добавляется.
        """
        parts: list[str] = []
        if self.name:
            parts.append(f"Профиль пользователя: {self.name}.")
        if self.identity:
            parts.append(f"Пользователь: {self.identity}.")

        for group in PREFERENCE_GROUPS:
            lines = self._preference_lines(group)
            if not lines:
                continue
            parts.append(f"{PREFERENCE_LABELS[group]}:\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        """Пустой ли профиль (нечего подмешивать в запрос)."""
        return not (
            self.name
            or self.identity
            or self.style
            or self.format
            or self.constraints
        )

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        return (
            f"UserProfile(name={self.name!r}, identity={self.identity!r}, "
            f"style={self.style!r}, format={self.format!r}, "
            f"constraints={self.constraints!r})"
        )


class Personalization:
    """Управляет профилями пользователя и активным профилем.

    Хранит несколько именованных профилей в файле ``profiles.json`` и помнит,
    какой из них активен. Активный профиль подмешивается к каждому запросу
    агента через :meth:`system_message`.
    """

    def __init__(self, path: Path = default_profiles_path()) -> None:
        self._path = path
        self._profiles: dict[str, UserProfile] = {}
        self._active: str | None = None
        self._load()

    @property
    def path(self) -> Path:
        """Файл, в котором хранятся профили."""
        return self._path

    @property
    def active_name(self) -> str | None:
        """Имя активного профиля (None — персонализация не активна)."""
        return self._active

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
        for name, profile_data in (data.get("profiles") or {}).items():
            if isinstance(profile_data, dict):
                profile = UserProfile.from_dict(profile_data)
                if not profile.name:
                    profile.name = str(name)
                self._profiles[str(name)] = profile
        active = data.get("active")
        if active in self._profiles:
            self._active = str(active)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "profiles": {
                        name: profile.to_dict()
                        for name, profile in self._profiles.items()
                    },
                    "active": self._active,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def profiles(self) -> dict[str, UserProfile]:
        """Все сохранённые профили: имя -> профиль."""
        return dict(self._profiles)

    def has(self, name: str) -> bool:
        """Есть ли профиль с таким именем."""
        return name in self._profiles

    def get(self, name: str) -> UserProfile | None:
        """Профиль по имени (None, если такого нет)."""
        return self._profiles.get(name)

    def save(self, name: str, profile: UserProfile) -> None:
        """Сохраняет профиль под указанным именем."""
        self._profiles[name] = profile
        self._save()

    def delete(self, name: str) -> bool:
        """Удаляет профиль. Возвращает True, если он был."""
        if name not in self._profiles:
            return False
        del self._profiles[name]
        if self._active == name:
            self._active = None
        self._save()
        return True

    def set_active(self, name: str | None) -> bool:
        """Делает профиль активным (None — отключить персонализацию)."""
        if name is not None and name not in self._profiles:
            return False
        self._active = name
        self._save()
        return True

    def active(self) -> UserProfile | None:
        """Активный профиль (None — персонализация не активна)."""
        if self._active is None:
            return None
        return self._profiles.get(self._active)

    def system_message(self) -> str:
        """System-сообщение активного профиля (пусто, если профиля нет/пуст)."""
        profile = self.active()
        if profile is None or profile.is_empty():
            return ""
        return profile.system_message()

    def summarize(self) -> str:
        """Краткое описание всех профилей и активного (для вывода пользователю)."""
        lines = []
        if not self._profiles:
            lines.append("Профилей нет. Создайте профиль через /profile_new <имя>.")
            return "\n".join(lines)
        for name, profile in self._profiles.items():
            marker = " (активный)" if name == self._active else ""
            entries = len(profile.style) + len(profile.format) + len(profile.constraints)
            lines.append(f"- {name}{marker}: {entries} предпочтений")
        return "\n".join(lines)
