"""Blueprints и общие помощники HTTP-слоя."""

from __future__ import annotations

from flask import Flask, current_app


def current_service():
    """Возвращает общий :class:`app.chat_service.ChatService` приложения."""
    return current_app.extensions["service"]


def current_engine():
    """Возвращает общий :class:`app.engine.RagEngine` приложения."""
    return current_app.extensions["engine"]


def register_blueprints(app: Flask) -> None:
    """Регистрирует все blueprint-ы сервиса."""
    from . import api, auth, pages

    app.register_blueprint(auth.bp)
    app.register_blueprint(api.bp)
    app.register_blueprint(pages.bp)
