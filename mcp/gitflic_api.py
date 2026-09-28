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
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_API_URL = "https://api.gitflic.ru"
REQUEST_TIMEOUT = 30.0
TIMEOUT_ENV = "GITFLIC_TIMEOUT"
RETRIES_ENV = "GITFLIC_RETRIES"


def api_url() -> str:
    """Базовый адрес API GitFlic (можно переопределить через окружение)."""
    return os.environ.get("GITFLIC_API_URL", DEFAULT_API_URL).rstrip("/")


def token() -> str:
    """Токен доступа GitFlic из переменной окружения (может быть пустым)."""
    return os.environ.get("GITFLIC_TOKEN", "").strip()


def request_timeout() -> float:
    """Таймаут одного HTTP-запроса к GitFlic (сек), с учётом ``GITFLIC_TIMEOUT``."""
    try:
        return max(1.0, float(os.environ.get(TIMEOUT_ENV, REQUEST_TIMEOUT)))
    except ValueError:
        return REQUEST_TIMEOUT


def request_retries() -> int:
    """Сколько раз повторять запрос при таймауте/сетевой ошибке (``GITFLIC_RETRIES``)."""
    try:
        return max(0, int(os.environ.get(RETRIES_ENV, "1")))
    except ValueError:
        return 1


def request_json(path: str, params: dict | None = None) -> dict:
    """Выполняет GET-запрос к API GitFlic и возвращает разобранный JSON.

    При таймауте чтения или сетевом сбое запрос повторяется ``GITFLIC_RETRIES``
    раз (по умолчанию 1). HTTP-ошибки (4xx/5xx) не повторяются.
    """
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

    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(
                request, timeout=request_timeout()
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt >= request_retries():
                raise
            attempt += 1
            time.sleep(min(1.5 * attempt, 4.0))


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
