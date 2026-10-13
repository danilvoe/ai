"""JSON API чатов, сообщений и статуса сервиса."""

from __future__ import annotations

import threading

from flask import Blueprint, current_app, jsonify, request

from ..chat_service import ChatServiceError
from ..security import api_login_required, current_user_id
from . import current_engine, current_service

bp = Blueprint("api", __name__, url_prefix="/api")


def _chat_payload(chat: dict) -> dict:
    return {
        "id": chat["id"],
        "title": chat["title"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"],
    }


@bp.get("/chats")
@api_login_required
def list_chats():
    chats = current_service().list_chats(current_user_id())
    return jsonify({"chats": [_chat_payload(chat) for chat in chats]})


@bp.post("/chats")
@api_login_required
def create_chat():
    data = request.get_json(silent=True) or {}
    chat = current_service().create_chat(current_user_id(), data.get("title"))
    return jsonify({"chat": _chat_payload(chat)}), 201


@bp.patch("/chats/<int:chat_id>")
@api_login_required
def rename_chat(chat_id: int):
    data = request.get_json(silent=True) or {}
    chat = current_service().rename_chat(
        current_user_id(), chat_id, str(data.get("title", ""))
    )
    return jsonify({"chat": _chat_payload(chat)})


@bp.delete("/chats/<int:chat_id>")
@api_login_required
def delete_chat(chat_id: int):
    current_service().delete_chat(current_user_id(), chat_id)
    return jsonify({"ok": True})


@bp.get("/chats/<int:chat_id>/messages")
@api_login_required
def list_messages(chat_id: int):
    messages = current_service().get_messages(current_user_id(), chat_id)
    return jsonify(
        {
            "messages": [
                {
                    "id": item["id"],
                    "role": item["role"],
                    "content": item["content"],
                    "meta": item.get("meta"),
                    "created_at": item["created_at"],
                }
                for item in messages
            ]
        }
    )


@bp.post("/chats/<int:chat_id>/messages")
@api_login_required
def post_message(chat_id: int):
    data = request.get_json(silent=True) or {}
    use_rag = bool(data.get("use_rag", True))
    result = current_service().post_message(
        current_user_id(), chat_id, str(data.get("message", "")), use_rag=use_rag
    )
    return jsonify(result)


@bp.post("/chats/<int:chat_id>/reset")
@api_login_required
def reset_chat(chat_id: int):
    current_service().reset_chat(current_user_id(), chat_id)
    return jsonify({"ok": True})


@bp.get("/status")
@api_login_required
def status():
    return jsonify(current_service().status())


@bp.post("/admin/reindex")
@api_login_required
def admin_reindex():
    config = current_app.extensions["service_config"]
    if not config.allow_reindex:
        raise ChatServiceError("Пересборка индекса отключена.", 403)
    engine = current_engine()

    def _rebuild() -> None:
        from core.index_build import build_index, resolve_dataset_source

        source = resolve_dataset_source(config.dataset, config.index_dir)
        build_index(
            source,
            dataset=config.dataset,
            strategy=config.strategy,
            index_dir=config.index_dir,
        )
        engine.reload()

    threading.Thread(target=_rebuild, daemon=True).start()
    return jsonify({"ok": True, "started": True}), 202
