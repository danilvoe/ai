#!/usr/bin/env python3
"""Day 17–18: собственный MCP-сервер вокруг публичного API GitFlic.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``). Он регистрирует
инструменты вокруг публичного API GitFlic:

* ``get_public_projects`` — получение списка публичных проектов
  (``GET {base}/project``);
* ``count_public_projects`` — подсчёт количества публичных проектов с
  агрегацией (по языкам, владельцам, топикам) и сохранением снимка в JSON;
* ``schedule_project_count`` / ``list_count_jobs`` / ``cancel_count_job`` —
  фоновые задачи с отложенным и периодическим выполнением;
* ``get_project_count_report`` — агрегированный результат последнего
  подсчёта и история снимков.

Каждый инструмент возвращает результат, а входные параметры описаны типами и
docstring — MCP автоматически строит из них JSON-схему (``tools/list``).

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
import urllib.error
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

from gitflic_api import fetch_public_projects
from gitflic_scheduler import ProjectCounter, ProjectCountScheduler

mcp = FastMCP(
    "gitflic",
    instructions=(
        "Инструменты для работы с публичным API GitFlic. "
        "get_public_projects возвращает публичные проекты, "
        "count_public_projects считает их количество и сохраняет снимок в JSON, "
        "schedule_project_count запускает подсчёт по расписанию (отложенно или "
        "периодически), get_project_count_report возвращает агрегированный "
        "результат и историю."
    ),
)

# Один планировщик на процесс сервера: фоновые задачи живут в daemon-потоке
# и продолжают работать, пока сервер обслуживает MCP-клиента.
_scheduler = ProjectCountScheduler()


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
    try:
        return fetch_public_projects(query=query, page=page, size=size)
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


def _error_result(exc: Exception) -> dict:
    """Единый формат ошибки GitFlic API для инструментов подсчёта."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            detail = (
                "GitFlic API отклонил запрос: нужен access token "
                "(scope PROJECT_READ) в GITFLIC_TOKEN."
            )
        elif exc.code == 404:
            detail = "GitFlic API: данные по запросу не найдены."
        else:
            detail = f"GitFlic API вернул ошибку HTTP {exc.code}."
        return {"error": detail, "status": exc.code, "totalPublicProjects": None}
    if isinstance(exc, urllib.error.URLError):
        return {
            "error": f"Не удалось обратиться к GitFlic API: {exc.reason}",
            "totalPublicProjects": None,
        }
    return {"error": f"{type(exc).__name__}: {exc}", "totalPublicProjects": None}


@mcp.tool()
def count_public_projects(
    query: Annotated[
        str, Field(description="Необязательная подстрока названия проекта для поиска.")
    ] = "",
    max_pages: Annotated[
        int,
        Field(
            description=(
                "Сколько страниц обойти максимум (0 — по умолчанию 5 или "
                "значение переменной окружения GITFLIC_MAX_PAGES)."
            )
        ),
    ] = 0,
) -> dict:
    """Посчитать количество публичных проектов GitFlic и сохранить данные в JSON.

    Инструмент обходит страницы публичных проектов GitFlic, агрегирует
    результат (всего, по языкам, по владельцам, по топикам) и **сохраняет
    снимок в JSON-файл** (``history/gitflic_counts.json``). Возвращает
    агрегированный результат последнего подсчёта.
    """
    counter = (
        ProjectCounter(max_pages=max_pages)
        if max_pages
        else _scheduler.counter
    )
    try:
        return counter.count_now(query=query)
    except Exception as exc:  # noqa: BLE001 — инструмент возвращает ошибку данными
        return _error_result(exc)


@mcp.tool()
def schedule_project_count(
    query: Annotated[
        str, Field(description="Необязательная подстрока названия проекта для поиска.")
    ] = "",
    delay_seconds: Annotated[
        float, Field(description="Задержка перед первым подсчётом, секунды.")
    ] = 0.0,
    interval_seconds: Annotated[
        float | None,
        Field(description="Период повторения в секундах (None — один отложенный запуск)."),
    ] = None,
    run_immediately: Annotated[
        bool, Field(description="Выполнить первый подсчёт сразу, не дожидаясь задержки.")
    ] = False,
) -> dict:
    """Запустить подсчёт публичных проектов по расписанию (в фоне).

    Поддерживает **отложенное** исполнение (``delay_seconds``) и
    **периодическое** (``interval_seconds``) — например, раз в час. Возвращает
    описание задачи с ``jobId`` сразу, не дожидаясь запроса к GitFlic; статус
    и результат забираются через ``list_count_jobs`` и
    ``get_project_count_report``.
    """
    return _scheduler.schedule(
        query=query,
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        run_immediately=run_immediately,
    )


@mcp.tool()
def list_count_jobs() -> dict:
    """Показать фоновые задачи подсчёта: статус, следующий запуск, число прогонов."""
    jobs = _scheduler.list_jobs()
    return {"count": len(jobs), "jobs": jobs}


@mcp.tool()
def cancel_count_job(
    job_id: Annotated[str, Field(description="Идентификатор задачи из list_count_jobs.")]
) -> dict:
    """Отменить фоновую периодическую задачу подсчёта по её jobId."""
    return {"jobId": job_id, "cancelled": _scheduler.cancel(job_id)}


@mcp.tool()
def get_project_count_report(
    history_limit: Annotated[
        int, Field(description="Сколько последних снимков вернуть в истории.")
    ] = 20,
) -> dict:
    """Вернуть агрегированный результат подсчёта и историю сохранённых снимков.

    Читает JSON-хранилище, отдаёт последний агрегат (всего, по языкам, по
    владельцам, по топикам) и краткую историю, чтобы видеть динамику.
    """
    store = _scheduler.counter.store
    latest = store.latest()
    data = store.load()
    return {
        "snapshotCount": len(data.get("snapshots") or []),
        "updatedAt": data.get("updatedAt"),
        "storePath": str(store.path),
        "latest": latest,
        "history": store.history(history_limit),
        "jobs": _scheduler.list_jobs(),
    }


def _run_check(query: str, page: int, size: int, count: bool, max_pages: int) -> int:
    """Выполняет инструмент напрямую (без MCP) и печатает JSON-результат."""
    if count:
        result = count_public_projects(query=query, max_pages=max_pages)
    else:
        result = get_public_projects(query=query, page=page, size=size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("error") else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер GitFlic (проекты и подсчёт по расписанию)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструмент напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="в режиме --check посчитать количество проектов (агрегат + JSON)",
    )
    parser.add_argument(
        "--query", default="", help="поисковый запрос для режима --check"
    )
    parser.add_argument("--page", type=int, default=0, help="страница (с 0)")
    parser.add_argument("--size", type=int, default=10, help="сколько проектов")
    parser.add_argument(
        "--max-pages", type=int, default=0, help="лимит страниц для --count"
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(
            _run_check(args.query, args.page, args.size, args.count, args.max_pages)
        )

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout
    # с MCP-клиентом (agent/mcp_tools.py или mcp_client).
    mcp.run()


if __name__ == "__main__":
    main()
