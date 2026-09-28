#!/usr/bin/env python3
"""Day 18: общий слой доступа к публичному API GitFlic.

Модуль выделяет низкоуровневую работу с API GitFlic (запрос, разбор ответа,
нормализация проекта) в одно место, чтобы им пользовались и MCP-сервер
(``mcp/gitflic_server.py``), и планировщик фоновых задач
(``mcp/gitflic_scheduler.py``), и запуск по cron (``mcp/gitflic_cron.py``).

Авторизация: заголовок ``Authorization: token <token>``; токен (scope
``PROJECT_READ``) берётся из переменной окружения ``GITFLIC_TOKEN``. Базовый
адрес API — ``GITFLIC_API_URL`` (по умолчанию ``https://api.gitflic.ru``).
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_API_URL = "https://api.gitflic.ru"
REQUEST_TIMEOUT = 30.0


def api_url() -> str:
    """Базовый адрес API GitFlic (можно переопределить через окружение)."""
    return os.environ.get("GITFLIC_API_URL", DEFAULT_API_URL).rstrip("/")


def token() -> str:
    """Токен доступа GitFlic из переменной окружения (может быть пустым)."""
    return os.environ.get("GITFLIC_TOKEN", "").strip()


def request_json(path: str, params: dict | None = None) -> dict:
    """Выполняет GET-запрос к API GitFlic и возвращает разобранный JSON."""
    url = api_url() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/json",
        "User-Agent": "ai-advent-mcp-gitflic/1.0",
    }
    access_token = token()
    if access_token:
        headers["Authorization"] = f"token {access_token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_project(raw: dict) -> dict:
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


def fetch_public_projects(query: str = "", page: int = 0, size: int = 100) -> dict:
    """Возвращает одну страницу публичных проектов GitFlic и данные пагинации."""
    size = max(1, min(int(size), 100))
    page = max(0, int(page))
    params = {"page": page, "size": size}
    if query:
        params["q"] = query
    data = request_json("/project", params)
    raw_projects = (data.get("_embedded") or {}).get("projectList") or []
    projects = [normalize_project(item) for item in raw_projects]
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
