"""Аутентификация: регистрация, вход, выход и текущий пользователь."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..chat_service import ChatServiceError
from ..security import (
    api_login_required,
    current_user_id,
    login_user,
    logout_user,
)
from . import current_service

bp = Blueprint("auth", __name__, url_prefix="/api")


def _public_user(user: dict) -> dict:
    return {"id": user["id"], "username": user["username"]}


@bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", ""))
    password = str(data.get("password", ""))
    service = current_service()
    user = service.register(username, password)
    login_user(user["id"])
    return jsonify({"user": _public_user(user)}), 201


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", ""))
    password = str(data.get("password", ""))
    user = current_service().authenticate(username, password)
    if user is None:
        raise ChatServiceError("Неверное имя пользователя или пароль.", 401)
    login_user(user["id"])
    return jsonify({"user": _public_user(user)})


@bp.post("/logout")
def logout():
    logout_user()
    return jsonify({"ok": True})


@bp.get("/me")
@api_login_required
def me():
    user = current_service().get_user(current_user_id())
    if user is None:
        raise ChatServiceError("Пользователь не найден.", 404)
    return jsonify({"user": _public_user(user)})
