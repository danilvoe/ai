"""HTML-страницы: вход, регистрация и чат."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, url_for

from ..security import current_user_id, login_required

bp = Blueprint("pages", __name__)


@bp.get("/")
def index():
    if current_user_id() is None:
        return redirect(url_for("pages.login"))
    return redirect(url_for("pages.chat"))


@bp.get("/login")
def login():
    if current_user_id() is not None:
        return redirect(url_for("pages.chat"))
    return render_template("login.html")


@bp.get("/register")
def register():
    if current_user_id() is not None:
        return redirect(url_for("pages.chat"))
    return render_template("register.html")


@bp.get("/chat")
@login_required
def chat():
    return render_template("chat.html")
