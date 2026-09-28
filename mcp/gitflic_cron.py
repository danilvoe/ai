#!/usr/bin/env python3
"""Day 18: запуск подсчёта публичных проектов GitFlic по cron.

Скрипт — точка входа для планировщика ``cron``. У него несколько режимов:

* по умолчанию (``--once``) — один подсчёт публичных проектов GitFlic:
  сохраняет снимок в JSON и печатает агрегированный результат;
* ``--run-due`` — выполняет отложенные/периодические задачи, поставленные
  MCP-инструментом ``schedule_project_count`` (задачи хранятся в JSON);
* ``--daemon`` — то же, но в цикле (без cron), выполняя задачи каждую секунду.

Ненулевой код возврата означает ошибку — удобно для логов cron.

Пример строк crontab::

    # каждый час считать публичные проекты GitFlic
    5 * * * * cd /path/to/ai_advent && GITFLIC_TOKEN=<token> \
        python3 mcp/gitflic_cron.py >> history/gitflic_cron.log 2>&1
    # каждую минуту выполнять отложенные/периодические задачи
    * * * * * cd /path/to/ai_advent && GITFLIC_TOKEN=<token> \
        python3 mcp/gitflic_cron.py --run-due >> history/gitflic_cron.log 2>&1

Готовые строки печатает ``python3 mcp/gitflic_cron.py --print-crontab``.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

from gitflic_scheduler import (
    CountStore,
    JobStore,
    ProjectCounter,
    ProjectCountScheduler,
    now_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "history" / "gitflic_cron.log"


def _ensure_token() -> str:
    """Берёт токен из окружения, а если его нет — из config проекта.

    Так cron работает без экспорта переменной: скрипт сам подставит
    ``tools.gitflic_token`` из ``config.one.json`` / ``config.json``.
    """
    token = os.environ.get("GITFLIC_TOKEN", "").strip()
    if token:
        return token
    for name in ("config.one.json", "config.json"):
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        try:
            section = (json.loads(path.read_text(encoding="utf-8")) or {}).get("tools") or {}
        except (json.JSONDecodeError, OSError):
            continue
        token = str(section.get("gitflic_token") or "").strip()
        if token:
            os.environ["GITFLIC_TOKEN"] = token
            if section.get("gitflic_api_url"):
                os.environ.setdefault("GITFLIC_API_URL", str(section["gitflic_api_url"]))
            return token
    return ""


def make_scheduler(store: str | None = None, jobs: str | None = None) -> ProjectCountScheduler:
    counter = ProjectCounter(store=CountStore(store))
    return ProjectCountScheduler(counter=counter, jobs_store=JobStore(jobs))


def run_once(
    query: str = "", max_pages: int = 0, store: str | None = None
) -> dict:
    """Один подсчёт с сохранением снимка; возвращает агрегированный результат.

    Ошибки сети/авторизации возвращаются полем ``error`` (а не исключением),
    чтобы cron писал понятную строку в лог.
    """
    counter = (
        ProjectCounter(store=CountStore(store), max_pages=max_pages)
        if max_pages
        else ProjectCounter(store=CountStore(store))
    )
    try:
        return counter.count_now(query=query, trigger="cron")
    except urllib.error.HTTPError as exc:
        detail = (
            "GitFlic API отклонил запрос: нужен access token "
            "(scope PROJECT_READ) в GITFLIC_TOKEN или tools.gitflic_token."
            if exc.code in (401, 403)
            else f"GitFlic API вернул ошибку HTTP {exc.code}."
        )
        return {
            "countedAt": now_iso(),
            "trigger": "cron",
            "error": detail,
            "status": exc.code,
            "totalPublicProjects": None,
        }
    except urllib.error.URLError as exc:
        return {
            "countedAt": now_iso(),
            "trigger": "cron",
            "error": f"Не удалось обратиться к GitFlic API: {exc.reason}",
            "totalPublicProjects": None,
        }


def crontab_lines() -> list[str]:
    """Готовые строки crontab для периодического подсчёта и выполнения задач."""
    root = PROJECT_ROOT
    script = Path(__file__).resolve()
    py = sys.executable
    return [
        f"5 * * * * cd {root} && GITFLIC_TOKEN=<token> {py} {script} "
        f">> {DEFAULT_LOG} 2>&1",
        f"* * * * * cd {root} && GITFLIC_TOKEN=<token> {py} {script} "
        f"--run-due >> {DEFAULT_LOG} 2>&1",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подсчёт публичных проектов GitFlic по cron (сохранение в JSON)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="выполнить один подсчёт (режим по умолчанию для cron)",
    )
    parser.add_argument(
        "--run-due",
        action="store_true",
        help="выполнить отложенные/периодические задачи, готовые к запуску",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="выполнять задачи в цикле (без cron), каждую секунду",
    )
    parser.add_argument("--query", default="", help="поисковый запрос (название проекта)")
    parser.add_argument(
        "--max-pages", type=int, default=0, help="лимит страниц (0 — по умолчанию)"
    )
    parser.add_argument("--store", default=None, help="путь к JSON-хранилищу снимков")
    parser.add_argument("--jobs", default=None, help="путь к JSON-хранилищу задач")
    parser.add_argument(
        "--print-crontab",
        action="store_true",
        help="напечатать строки crontab для запуска по расписанию и выйти",
    )
    args = parser.parse_args()

    if args.print_crontab:
        print("\n".join(crontab_lines()))
        return

    if not _ensure_token():
        print(
            "Нет токена GitFlic: задайте GITFLIC_TOKEN или tools.gitflic_token в "
            "config.one.json / config.json.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.run_due or args.daemon:
        scheduler = make_scheduler(args.store, args.jobs)
        if args.daemon:
            print(
                f"Демон планировщика: store={scheduler.counter.store.path}, "
                f"jobs={scheduler.jobs.path}"
            )
            while True:
                for view in scheduler.run_due():
                    print(json.dumps(view, ensure_ascii=False))
                time.sleep(1)
        executed = scheduler.run_due()
        print(
            json.dumps(
                {"executed": executed, "count": len(executed), "at": now_iso()},
                ensure_ascii=False,
                indent=2,
            )
        )
        if any(view.get("lastError") for view in executed):
            raise SystemExit(1)
        return

    result = run_once(args.query, args.max_pages, args.store)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("error") or result.get("totalPublicProjects") is None:
        print(f"Подсчёт не удался: {result.get('error', 'нет данных')}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"OK {now_iso()}: публичных проектов {result['totalPublicProjects']}, "
        f"сохранено в {result['storePath']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
