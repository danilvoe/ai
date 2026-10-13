"""Аутентификация и безопасность сервиса.

Пароли хешируются ``werkzeug.security``; сессия — серверная подписанная cookie
Flask. Ключ подписи берётся из ``security.secret_key`` конфига или создаётся
в ``data/secret.key`` при первом запуске.
"""

from __future__ import annotations

import secrets
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from flask import jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

MIN_PASSWORD_LENGTH = 6
SESSION_USER_KEY = "user_id"


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def validate_credentials(username: str, password: str) -> str | None:
    """Возвращает текст ошибки или ``None``, если данные корректны."""
    if not username or len(username.strip()) < 3:
        return "Имя пользователя должно содержать минимум 3 символа."
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Пароль должен содержать минимум {MIN_PASSWORD_LENGTH} символов."
    return None


def get_or_create_secret_key(secret_path: Path, configured: str = "") -> str:
    """Возвращает ключ подписи сессии: из конфига или из ``secret.key``."""
    if configured:
        return configured
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        content = secret_path.read_text(encoding="utf-8").strip()
        if content:
            return content
    key = secrets.token_hex(32)
    secret_path.write_text(key, encoding="utf-8")
    try:
        secret_path.chmod(0o600)
    except OSError:
        pass
    return key


def login_user(user_id: int) -> None:
    session[SESSION_USER_KEY] = int(user_id)


def logout_user() -> None:
    session.pop(SESSION_USER_KEY, None)


def current_user_id() -> int | None:
    value = session.get(SESSION_USER_KEY)
    return int(value) if value is not None else None


def login_required(view: Callable) -> Callable:
    """Защищает страницы: без логина — редирект на ``/login``."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        if current_user_id() is None:
            return redirect(url_for("pages.login", next=request.path))
        return view(*args, **kwargs)

    return wrapper


def api_login_required(view: Callable) -> Callable:
    """Защищает API: без логина — JSON-ошибка 401."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any):
        if current_user_id() is None:
            return jsonify({"error": "Требуется авторизация."}), 401
        return view(*args, **kwargs)

    return wrapper
