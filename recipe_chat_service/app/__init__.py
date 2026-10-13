"""Flask-приложение сервиса рецептов: фабрика :func:`create_app`."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .chat_service import ChatService, ChatServiceError
from .config import ServiceConfig
from .db import Database
from .engine import RagEngine
from .routes import register_blueprints
from .security import get_or_create_secret_key


def create_app(
    config: ServiceConfig | None = None,
    engine: RagEngine | None = None,
    db: Database | None = None,
    offline: bool | None = None,
) -> Flask:
    """Создаёт и настраивает Flask-приложение сервиса.

    ``engine``/``db`` можно передать готовыми (используется в офлайн-сценариях
    и тестах); иначе они создаются из конфигурации.
    """
    config = config or ServiceConfig.from_file()
    config.ensure_directories()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = get_or_create_secret_key(config.secret_path, config.secret_key)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    database = db or Database(config.database_path)
    database.init_schema()

    rag_engine = engine
    if rag_engine is None:
        rag_engine = RagEngine(config, offline=bool(offline))

    service = ChatService(database, rag_engine, config)

    app.extensions["service"] = service
    app.extensions["engine"] = rag_engine
    app.extensions["service_config"] = config
    app.extensions["database"] = database

    register_blueprints(app)

    @app.errorhandler(ChatServiceError)
    def _handle_service_error(error: ChatServiceError):
        return jsonify({"error": error.message}), error.status

    @app.errorhandler(404)
    def _handle_not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Не найдено."}), 404
        return "<h1>404 — страница не найдена</h1>", 404

    return app


__all__ = ["create_app"]
