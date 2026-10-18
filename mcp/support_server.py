#!/usr/bin/env python3
"""День 33: MCP-сервер поддержки — CRM с пользователями и тикетами.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и работает **локально**,
без сети: все инструменты читают два JSON-файла (синтетическая CRM-выгрузка) из
каталога данных. Он даёт ассистенту поддержки всё, что нужно, чтобы учитывать
контекст пользователя и тикета:

* ``get_user`` — профиль пользователя (план, версия, статус);
* ``find_user`` — поиск пользователя по логину/e-mail/имени;
* ``get_ticket`` — полный тикет вместе с перепиской;
* ``list_tickets`` — тикеты пользователя (с фильтром по статусу);
* ``search_tickets`` — поиск тикетов по тексту (связка «вопрос → тикет»);
* ``get_ticket_context`` — готовый человекочитаемый контекст для промпта.

Каталог данных задаётся переменной окружения ``SUPPORT_DATA_DIR`` (по умолчанию —
``mcp/support_data/``). Ожидаемые файлы: ``users.json`` и ``tickets.json``.

Запуск (транспорт по умолчанию — stdio)::

    python3 mcp/support_server.py

Проверка из терминала без MCP::

    python3 mcp/support_server.py --check --ticket T-1024 --user 1
    python3 mcp/support_server.py --check --list --search "авторизация"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "mcp" / "support_data"


def data_dir() -> Path:
    """Каталог с JSON-выгрузкой CRM (можно переопределить через env)."""
    override = os.environ.get("SUPPORT_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DATA_DIR


_cache: dict[str, tuple[float, dict]] = {}


def _load(name: str) -> dict:
    """Читает JSON-файл из каталога данных (с кэшем по mtime)."""
    path = data_dir() / name
    if not path.exists():
        return {"error": f"файл данных не найден: {path}"}
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:  # pragma: no cover — редкий случай
        return {"error": f"{type(exc).__name__}: {exc}"}
    cached = _cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(data, dict):
        return {"error": f"ожидался JSON-объект в {path}"}
    _cache[str(path)] = (mtime, data)
    return data


def _users() -> list[dict]:
    return list(_load("users.json").get("users") or [])


def _tickets() -> list[dict]:
    return list(_load("tickets.json").get("tickets") or [])


def _short_ticket(ticket: dict) -> dict:
    """Краткая форма тикета (без переписки) для списков."""
    return {
        key: ticket.get(key)
        for key in (
            "id", "user_id", "subject", "status", "category",
            "priority", "product", "version", "created_at", "updated_at", "tags",
        )
    }


def render_ticket_context(ticket: dict, user: dict | None = None) -> str:
    """Человекочитаемый контекст «пользователь + тикет + переписка»."""
    if not ticket or ticket.get("error"):
        return "Данные тикета недоступны."

    lines = [f"Тикет {ticket.get('id')}: «{ticket.get('subject', '')}»"]
    lines.append(
        f"Статус: {ticket.get('status', '?')}; категория: {ticket.get('category', '?')}; "
        f"приоритет: {ticket.get('priority', '?')}; версия продукта: {ticket.get('version', '?')}"
    )
    if ticket.get("tags"):
        lines.append("Теги: " + ", ".join(str(tag) for tag in ticket["tags"]))
    if user:
        lines.append(
            f"Пользователь: {user.get('name', '')} ({user.get('username', '')}, "
            f"e-mail: {user.get('email', '')}); план: {user.get('plan', '?')}; "
            f"статус аккаунта: {user.get('status', '?')}; версия: {user.get('version', '?')}"
        )
    elif ticket.get("user_id") is not None:
        lines.append(f"Пользователь (id): {ticket.get('user_id')}")
    messages = ticket.get("messages") or []
    if messages:
        lines.append("Переписка:")
        for message in messages:
            author = message.get("author", "?")
            role = message.get("role", "")
            stamp = message.get("created_at", "")
            lines.append(f"  - [{stamp}] {author} ({role}): {message.get('text', '')}")
    return "\n".join(lines)


mcp = FastMCP(
    "support",
    log_level="WARNING",
    instructions=(
        "Инструменты поддержки пользователей по локальной JSON-CRM: "
        "get_user возвращает профиль пользователя, find_user ищет пользователя по "
        "логину/e-mail/имени, get_ticket — полный тикет с перепиской, list_tickets — "
        "тикеты пользователя, search_tickets — поиск тикетов по тексту, "
        "get_ticket_context — готовый контекст тикета для ответа. Используй их, "
        "чтобы учитывать данные пользователя и тикета при ответе на вопрос."
    ),
)


@mcp.tool()
def get_user(
    user_id: Annotated[int, Field(description="Идентификатор пользователя (id).")],
) -> dict:
    """Получить профиль пользователя по идентификатору.

    Возвращает логин, e-mail, имя, план, версию продукта, статус аккаунта и дату
    последнего входа. Если пользователь не найден — ``error``.
    """
    for user in _users():
        if str(user.get("id")) == str(user_id):
            return dict(user)
    return {"error": f"пользователь не найден: id={user_id}", "user_id": user_id}


@mcp.tool()
def find_user(
    query: Annotated[str, Field(description="Подстрока для поиска: логин, e-mail или имя.")],
    limit: Annotated[int, Field(description="Сколько результатов вернуть (по умолчанию 5).")] = 5,
) -> dict:
    """Найти пользователя по логину, e-mail или имени.

    Поиск регистронезависимый, по подстроке. Возвращает ``users`` (до ``limit``) и
    общее число совпадений ``count``.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return {"error": "пустой поисковый запрос", "users": [], "count": 0}
    found = [
        dict(user)
        for user in _users()
        if needle in str(user.get("username", "")).lower()
        or needle in str(user.get("email", "")).lower()
        or needle in str(user.get("name", "")).lower()
    ]
    count = max(1, int(limit))
    return {"query": query, "count": len(found), "users": found[:count]}


@mcp.tool()
def get_ticket(
    ticket_id: Annotated[str, Field(description="Идентификатор тикета, например 'T-1024'.")],
) -> dict:
    """Получить полный тикет по идентификатору (включая переписку).

    Возвращает все поля тикета и ``messages``. Если тикет не найден — ``error``.
    """
    target = (ticket_id or "").strip().upper()
    if not target:
        return {"error": "не указан идентификатор тикета"}
    for ticket in _tickets():
        if str(ticket.get("id", "")).upper() == target:
            return dict(ticket)
    return {"error": f"тикет не найден: {ticket_id}", "ticket_id": ticket_id}


@mcp.tool()
def list_tickets(
    user_id: Annotated[str, Field(description="Фильтр по id пользователя (пусто — все).")] = "",
    status: Annotated[str, Field(description="Фильтр по статусу (пусто — все).")] = "",
    limit: Annotated[int, Field(description="Сколько тикетов вернуть (по умолчанию 20).")] = 20,
) -> dict:
    """Перечислить тикеты (краткая форма) с фильтрами по пользователю и статусу.

    Возвращает ``tickets`` (до ``limit``), число ``count`` и общее число
    совпадений ``total``. Переписка в краткую форму не входит.
    """
    items = _tickets()
    if str(user_id).strip():
        items = [t for t in items if str(t.get("user_id")) == str(user_id).strip()]
    if str(status).strip():
        wanted = str(status).strip().lower()
        items = [t for t in items if str(t.get("status", "")).lower() == wanted]
    count = max(1, int(limit))
    return {
        "user_id": str(user_id),
        "status": str(status),
        "total": len(items),
        "count": min(len(items), count),
        "tickets": [_short_ticket(t) for t in items[:count]],
    }


@mcp.tool()
def search_tickets(
    query: Annotated[str, Field(description="Текст для поиска по теме, тегам и переписке.")],
    limit: Annotated[int, Field(description="Сколько результатов вернуть (по умолчанию 10).")] = 10,
    scope: Annotated[
        str,
        Field(description="Где искать: 'all' (тема/теги/категория/переписка) или 'meta' (без переписки)."),
    ] = "all",
) -> dict:
    """Найти тикеты по тексту: тема, теги и (при scope='all') переписка.

    Регистронезависимый поиск по подстроке. Возвращает ``tickets`` (краткая форма
    + ``matched_in`` — где нашлось совпадение) и ``count``. ``scope='meta'``
    ограничивает поиск темой, тегами и категорией (без переписки) — удобно для
    привязки короткого вопроса к тикету.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return {"error": "пустой поисковый запрос", "tickets": [], "count": 0}

    include_messages = str(scope).strip().lower() != "meta"
    results: list[dict] = []
    for ticket in _tickets():
        matched_in: list[str] = []
        if needle in str(ticket.get("subject", "")).lower():
            matched_in.append("subject")
        tags = " ".join(str(tag) for tag in ticket.get("tags") or []).lower()
        if needle in tags:
            matched_in.append("tags")
        if needle in str(ticket.get("category", "")).lower():
            matched_in.append("category")
        if include_messages and not matched_in:
            for message in ticket.get("messages") or []:
                if needle in str(message.get("text", "")).lower():
                    matched_in.append("messages")
                    break
        if matched_in:
            item = _short_ticket(ticket)
            item["matched_in"] = sorted(set(matched_in))
            results.append(item)

    count = max(1, int(limit))
    return {"query": query, "scope": "all" if include_messages else "meta", "count": len(results), "tickets": results[:count]}


@mcp.tool()
def get_ticket_context(
    ticket_id: Annotated[str, Field(description="Идентификатор тикета, например 'T-1024'.")],
) -> dict:
    """Собрать готовый контекст тикета: пользователь + тикет + переписка.

    Главный инструмент для ассистента: возвращает поля тикета, профиль
    пользователя и текст ``context`` — человекочитаемый блок, который можно
    подставить в промпт.
    """
    ticket = get_ticket(ticket_id)
    if ticket.get("error"):
        return ticket
    user = get_user(int(ticket["user_id"])) if ticket.get("user_id") is not None else {}
    if user.get("error"):
        user = {}
    return {
        "ticket_id": ticket.get("id"),
        "user_id": ticket.get("user_id"),
        "subject": ticket.get("subject"),
        "status": ticket.get("status"),
        "category": ticket.get("category"),
        "priority": ticket.get("priority"),
        "version": ticket.get("version"),
        "tags": ticket.get("tags") or [],
        "user": user,
        "message_count": len(ticket.get("messages") or []),
        "context": render_ticket_context(ticket, user),
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструменты напрямую (без MCP) и печатает JSON-результат."""
    result: dict = {"dataDir": str(data_dir())}
    ok = True
    if args.user is not None:
        user = get_user(args.user)
        result["user"] = user
        ok = ok and not user.get("error")
    if args.find:
        found = find_user(args.find)
        result["find"] = found
        ok = ok and not found.get("error")
    if args.ticket:
        ticket = get_ticket(args.ticket)
        result["ticket"] = ticket
        ok = ok and not ticket.get("error")
    if args.context:
        context = get_ticket_context(args.context)
        result["context"] = context
        ok = ok and not context.get("error")
    if args.list:
        listed = list_tickets(user_id=args.list_user, status=args.status, limit=args.limit)
        result["list"] = listed
        ok = ok and not listed.get("error")
    if args.search:
        found = search_tickets(args.search, limit=args.limit)
        result["search"] = found
        ok = ok and not found.get("error")

    if not (args.user is not None or args.find or args.ticket or args.context or args.list or args.search):
        # По умолчанию — сводка по данным.
        result["users"] = {"count": len(_users())}
        result["tickets"] = {"count": len(_tickets())}

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер поддержки: пользователи и тикеты из локальной JSON-CRM"
    )
    parser.add_argument(
        "--check", action="store_true", help="вызвать инструменты напрямую (без MCP)"
    )
    parser.add_argument("--user", type=int, default=None, help="id для get_user")
    parser.add_argument("--find", default="", help="запрос для find_user")
    parser.add_argument("--ticket", default="", help="id для get_ticket")
    parser.add_argument("--context", default="", help="id для get_ticket_context")
    parser.add_argument("--list", action="store_true", help="вызвать list_tickets")
    parser.add_argument("--list-user", default="", help="фильтр user_id для list_tickets")
    parser.add_argument("--status", default="", help="фильтр статуса для list_tickets")
    parser.add_argument("--search", default="", help="запрос для search_tickets")
    parser.add_argument("--limit", type=int, default=10, help="лимит результатов")
    parser.add_argument("--data-dir", default="", help="каталог данных (иначе SUPPORT_DATA_DIR)")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["SUPPORT_DATA_DIR"] = args.data_dir

    if args.check or args.user is not None or args.find or args.ticket or args.context or args.list or args.search:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: stdout — канал MCP, подсказку пишем в stderr.
    print(
        "MCP-сервер поддержки запущен (stdio). Для разовой проверки используйте "
        "--check (например: --check --ticket T-1024 --user 1).",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
