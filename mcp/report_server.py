#!/usr/bin/env python3
"""Day 20: третий MCP-сервер оркестрации — журнал и индекс отчётов.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и работает **локально**,
без сети. Он обслуживает финальную часть кросс-серверного флоу: агрегирует уже
сохранённые отчёты в единый индекс и ведёт журнал запусков.

Инструменты:

* ``list_reports`` — перечислить сохранённые отчёты в каталоге проекта
  (имя, путь, размер, число строк, заголовок);
* ``record_run`` — записать запуск флоу в JSON-журнал
  (``history/report_runs.json``): запрос, путь отчёта, число проектов, теги;
* ``build_report_index`` — собрать Markdown-индекс из отчётов и журнала
  запусков и сохранить его в файл внутри проекта.

Все пути, которые принимает инструмент, обязаны находиться внутри каталога
проекта — выход наружу отклоняется.

Запуск (транспорт по умолчанию — stdio)::

    python3 mcp/report_server.py

Проверка из терминала без MCP::

    python3 mcp/report_server.py --check --list
    python3 mcp/report_server.py --check --index
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = "history/reports"
DEFAULT_RUNS_STORE = "history/report_runs.json"
DEFAULT_INDEX = "history/reports/index.md"

mcp = FastMCP(
    "report",
    instructions=(
        "Локальные инструменты для финальной обработки отчётов: list_reports "
        "перечисляет сохранённые отчёты, record_run пишет запуск флоу в журнал, "
        "build_report_index собирает единый Markdown-индекс отчётов и запусков."
    ),
)


def now_iso() -> str:
    """Текущее время в ISO-8601 с локальным часовым поясом."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _resolve_inside(path: str | Path, default: str | Path) -> Path:
    """Разрешает путь и не даёт выйти за пределы каталога проекта."""
    candidate = Path(path or default)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if PROJECT_ROOT != resolved and PROJECT_ROOT not in resolved.parents:
        raise ValueError(
            f"Путь должен находиться внутри проекта ({PROJECT_ROOT})."
        )
    return resolved


def _write_json(path: Path, data: dict) -> None:
    """Атомарно пишет JSON-файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: dict) -> dict:
    """Читает JSON-файл; при отсутствии/повреждении возвращает значение по умолчанию."""
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _report_meta(path: Path) -> dict:
    """Метаданные одного отчёта: путь, размер, строки, заголовок."""
    text = path.read_text(encoding="utf-8")
    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
    return {
        "name": path.name,
        "relativePath": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(text.encode("utf-8")),
        "lines": text.count("\n") + 1,
        "title": title,
        "modifiedAt": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).astimezone().isoformat(timespec="seconds"),
    }


def _collect_reports(directory: str) -> list[dict]:
    """Собирает метаданные отчётов заданного каталога, свежие сверху."""
    try:
        target = _resolve_inside(directory, DEFAULT_REPORTS_DIR)
    except ValueError:
        return []
    if not target.is_dir():
        return []
    reports = [
        _report_meta(item)
        for item in target.iterdir()
        if item.is_file() and item.suffix.lower() in (".md", ".txt", ".json")
    ]
    reports.sort(key=lambda item: item["modifiedAt"], reverse=True)
    return reports


@mcp.tool()
def list_reports(
    directory: Annotated[
        str, Field(description="Каталог с отчётами внутри проекта.")
    ] = DEFAULT_REPORTS_DIR,
) -> dict:
    """Перечислить сохранённые отчёты в каталоге проекта.

    Возвращает список отчётов: имя файла, относительный путь, размер, число
    строк, заголовок (первый заголовок Markdown) и время изменения.
    """
    reports = _collect_reports(directory)
    return {"directory": directory, "count": len(reports), "reports": reports}


@mcp.tool()
def record_run(
    query: Annotated[
        str, Field(description="Запрос/тема запуска флоу.")
    ] = "",
    report_path: Annotated[
        str, Field(description="Путь сохранённого отчёта внутри проекта.")
    ] = "",
    total_projects: Annotated[
        int, Field(description="Сколько проектов было в отчёте (0 — неизвестно).")
    ] = 0,
    tags: Annotated[
        list[str] | None, Field(description="Метки запуска (необязательно).")
    ] = None,
) -> dict:
    """Записать запуск флоу в JSON-журнал.

    Журнал (``history/report_runs.json``) хранит записи о запусках: время,
    запрос, путь отчёта, число проектов и метки. Возвращает запись и общее
    число запусков в журнале.
    """
    store = _resolve_inside(DEFAULT_RUNS_STORE, DEFAULT_RUNS_STORE)
    relative = report_path
    if report_path:
        try:
            relative = str(
                _resolve_inside(report_path, report_path).relative_to(PROJECT_ROOT)
            )
        except ValueError as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "recorded": False}

    data = _load_json(store, {"createdAt": now_iso(), "runs": []})
    runs = data.setdefault("runs", [])
    record = {
        "id": len(runs) + 1,
        "query": query,
        "reportPath": relative,
        "totalProjects": total_projects,
        "tags": tags or [],
        "recordedAt": now_iso(),
    }
    runs.append(record)
    data["updatedAt"] = record["recordedAt"]
    data["runCount"] = len(runs)
    try:
        _write_json(store, data)
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "recorded": False}
    return {
        "recorded": True,
        "runCount": len(runs),
        "storePath": str(store.relative_to(PROJECT_ROOT)),
        "record": record,
    }


@mcp.tool()
def build_report_index(
    directory: Annotated[
        str, Field(description="Каталог с отчётами внутри проекта.")
    ] = DEFAULT_REPORTS_DIR,
    path: Annotated[
        str, Field(description="Куда сохранить индекс внутри проекта.")
    ] = DEFAULT_INDEX,
) -> dict:
    """Собрать единый Markdown-индекс отчётов и журнала запусков.

    Читает локальные отчёты (см. ``list_reports``) и журнал запусков
    (``record_run``), формирует Markdown-оглавление со сводкой и атомарно
    сохраняет его в файл внутри проекта. Возвращает путь и статистику.
    """
    reports = _collect_reports(directory)
    store = _resolve_inside(DEFAULT_RUNS_STORE, DEFAULT_RUNS_STORE)
    runs = _load_json(store, {"createdAt": now_iso(), "runs": []}).get("runs") or []

    lines = ["# Индекс отчётов", "", f"_Сформировано: {now_iso()}_", ""]
    lines.append(f"Отчётов: {len(reports)}; запусков: {len(runs)}")
    lines.append("")
    if reports:
        lines.append("## Отчёты")
        lines.append("")
        for report in reports:
            title = report["title"] or report["name"]
            lines.append(
                f"- [{title}]({report['name']}) — "
                f"{report['bytes']} байт, строк {report['lines']}"
            )
        lines.append("")
    else:
        lines.append("_Отчётов пока нет._")
        lines.append("")
    if runs:
        lines.append("## Журнал запусков")
        lines.append("")
        for run in reversed(runs):
            suffix = (
                f" ({run['totalProjects']} проектов)" if run.get("totalProjects") else ""
            )
            lines.append(
                f"- #{run.get('id')} `{run.get('query') or '—'}` → "
                f"{run.get('reportPath') or '—'}{suffix} "
                f"— {run.get('recordedAt')}"
            )
        lines.append("")

    text = "\n".join(lines)
    try:
        target = _resolve_inside(path, DEFAULT_INDEX)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
    except (ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "path": str(target),
        "relativePath": str(target.relative_to(PROJECT_ROOT)),
        "bytes": len(text.encode("utf-8")),
        "reportCount": len(reports),
        "runCount": len(runs),
        "savedAt": now_iso(),
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструмент напрямую (без MCP) и печатает JSON-результат."""
    if args.index:
        result = build_report_index(directory=args.directory, path=args.output)
    else:
        result = list_reports(directory=args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("error") else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер отчётов: журнал запусков и индекс"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструмент напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument(
        "--list", action="store_true", help="в режиме --check перечислить отчёты"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="в режиме --check собрать индекс отчётов",
    )
    parser.add_argument(
        "--directory", default=DEFAULT_REPORTS_DIR, help="каталог с отчётами"
    )
    parser.add_argument(
        "--output", default=DEFAULT_INDEX, help="куда сохранить индекс"
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout с клиентом.
    mcp.run()


if __name__ == "__main__":
    main()
