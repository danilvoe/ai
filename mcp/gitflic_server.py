#!/usr/bin/env python3
"""Day 17: собственный MCP-сервер вокруг публичного API GitFlic.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``). Он регистрирует
инструмент ``get_public_projects`` — получение списка публичных проектов
GitFlic через ``GET {base}/project``.

Инструмент возвращает результат: список проектов (id, название, владелец,
язык, ссылки на клонирование, топики) и данные пагинации. Входные параметры
описаны типами и docstring — MCP автоматически строит из них JSON-схему
инструмента (``tools/list``), которую видит клиент.

API GitFlic требует авторизацию: в заголовке ``Authorization: token <token>``.
Токен (scope ``PROJECT_READ``) берётся из переменной окружения
``GITFLIC_TOKEN``. Базовый адрес API — ``GITFLIC_API_URL`` (по умолчанию
``https://api.gitflic.ru``).

Запуск (транспорт по умолчанию — stdio)::

    GITFLIC_TOKEN=<token> python3 mcp/gitflic_server.py

Для проверки из терминала есть отдельный режим ``--check``: он выполняет
инструмент напрямую и печатает результат, не поднимая MCP-транспорт.
"""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

DEFAULT_API_URL = "https://api.gitflic.ru"
REQUEST_TIMEOUT = 30.0

mcp = FastMCP(
    "gitflic",
    instructions=(
        "Инструменты для работы с публичным API GitFlic. "
        "Инструмент get_public_projects возвращает публичные проекты GitFlic."
    ),
)


def _api_url() -> str:
    """Базовый адрес API GitFlic (можно переопределить через окружение)."""
    return os.environ.get("GITFLIC_API_URL", DEFAULT_API_URL).rstrip("/")


def _token() -> str:
    """Токен доступа GitFlic из переменной окружения (может быть пустым)."""
    return os.environ.get("GITFLIC_TOKEN", "").strip()


def _request_json(path: str, params: dict | None = None) -> dict:
    """Выполняет GET-запрос к API GitFlic и возвращает разобранный JSON."""
    url = _api_url() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/json",
        "User-Agent": "ai-advent-mcp-gitflic/1.0",
    }
    token = _token()
    if token:
        headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_project(raw: dict) -> dict:
    """Приводит проект GitFlic к компактному, удобному агенту виду."""
    owner = raw.get("owner") or {}
    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "alias": raw.get("alias"),
        "description": raw.get("description") or "",
        "language": raw.get("language") or "",
        "private": bool(raw.get("private", False)),
        "owner": {"alias": owner.get("alias"), "type": owner.get("type")},
        "webUrl": raw.get("forkUrl") or "",
        "httpTransportUrl": raw.get("httpTransportUrl") or "",
        "sshTransportUrl": raw.get("sshTransportUrl") or "",
        "defaultBranch": raw.get("defaultBranch") or "",
        "topics": list(raw.get("topics") or []),
    }


@mcp.tool()
def get_public_projects(
    query: Annotated[
        str, Field(description="Необязательная подстрока названия проекта для поиска.")
    ] = "",
    page: Annotated[
        int, Field(description="Номер страницы, начиная с 0 (первая страница — 0).")
    ] = 0,
    size: Annotated[
        int, Field(description="Сколько проектов вернуть на странице (1..100).")
    ] = 10,
) -> dict:
    """Получить список публичных проектов GitFlic.

    Возвращает публичные проекты GitFlic: их id, название, владельца, язык,
    ссылки на клонирование и топики, а также данные пагинации.
    """
    size = max(1, min(int(size), 100))
    page = max(0, int(page))
    params = {"page": page, "size": size}
    if query:
        params["q"] = query
    try:
        data = _request_json("/project", params)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            detail = (
                "GitFlic API отклонил запрос: нужен access token "
                "(scope PROJECT_READ) в GITFLIC_TOKEN."
            )
        elif exc.code == 404:
            detail = "GitFlic API: данные по запросу не найдены."
        else:
            detail = f"GitFlic API вернул ошибку HTTP {exc.code}."
        return {"error": detail, "status": exc.code, "projects": [], "count": 0}
    except urllib.error.URLError as exc:
        return {
            "error": f"Не удалось обратиться к GitFlic API: {exc.reason}",
            "projects": [],
            "count": 0,
        }

    raw_projects = (data.get("_embedded") or {}).get("projectList") or []
    projects = [_normalize_project(item) for item in raw_projects]
    page_info = data.get("page") or {}
    return {
        "query": query,
        "count": len(projects),
        "total": page_info.get("totalElements", len(projects)),
        "page": {
            "size": page_info.get("size", size),
            "number": page_info.get("number", page),
            "totalPages": page_info.get("totalPages", 1),
        },
        "projects": projects,
    }


def _run_check(query: str, page: int, size: int) -> int:
    """Выполняет инструмент напрямую (без MCP) и печатает JSON-результат."""
    result = get_public_projects(query=query, page=page, size=size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("error") else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер GitFlic (get_public_projects)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструмент напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument(
        "--query", default="", help="поисковый запрос для режима --check"
    )
    parser.add_argument("--page", type=int, default=0, help="страница (с 0)")
    parser.add_argument("--size", type=int, default=10, help="сколько проектов")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(_run_check(args.query, args.page, args.size))

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout
    # с MCP-клиентом (agent/mcp_tools.py или mcp_client).
    mcp.run()


if __name__ == "__main__":
    main()
