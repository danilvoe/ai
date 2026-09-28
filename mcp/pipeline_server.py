#!/usr/bin/env python3
"""Day 19: MCP-сервер с композицией инструментов (search → summarize → saveToFile).

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и регистрирует три
независимых инструмента, которые по отдельности решают свою задачу, и один
составной инструмент, который автоматически прогоняет их цепочкой:

* ``search`` — **получает данные**: публичные проекты GitFlic по запросу
  (через ``mcp/gitflic_api.py``);
* ``summarize`` — **обрабатывает данные**: локальная экстрактивная выжимка
  из описаний проектов (``mcp/pipeline_text.py``), без сети и LLM;
* ``saveToFile`` — **сохраняет результат**: атомарно пишет текст/JSON в файл
  внутри каталога проекта (защита от выхода за его пределы);
* ``run_pipeline`` — композиция: одним вызовом выполняет ``search`` →
  ``summarize`` → ``saveToFile`` и возвращает отчёт по каждому шагу, включая
  фактические аргументы, то есть что именно было передано между инструментами.

GitFlic API требует авторизацию: токен (scope ``PROJECT_READ``) в переменной
окружения ``GITFLIC_TOKEN`` (или ``tools.gitflic_token`` конфигурации, когда
сервер запускает ``agent/pipeline_scenarios.py``). Базовый адрес — ``GITFLIC_API_URL``.

Запуск (транспорт по умолчанию — stdio)::

    GITFLIC_TOKEN=<token> python3 mcp/pipeline_server.py

Проверка из терминала без MCP::

    GITFLIC_TOKEN=<token> python3 mcp/pipeline_server.py --check --query docs
    GITFLIC_TOKEN=<token> python3 mcp/pipeline_server.py --check --pipeline --query docs
"""

import argparse
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

from gitflic_api import fetch_public_projects
from pipeline_text import summarize_items

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = "history/pipeline_result.md"

mcp = FastMCP(
    "pipeline",
    instructions=(
        "Инструменты-конструктор пайплайна: search получает публичные проекты "
        "GitFlic, summarize делает локальную выжимку из данных, saveToFile "
        "сохраняет результат в файл. run_pipeline автоматически выполняет "
        "цепочку search → summarize → saveToFile и возвращает отчёт по шагам."
    ),
)


def now_iso() -> str:
    """Текущее время в ISO-8601 с локальным часовым поясом."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _error_result(exc: Exception, items: list | None = None) -> dict:
    """Единый формат ошибки GitFlic API для инструментов."""
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
        return {"error": detail, "status": exc.code, "items": items or [], "count": 0}
    if isinstance(exc, urllib.error.URLError):
        return {
            "error": f"Не удалось обратиться к GitFlic API: {exc.reason}",
            "items": items or [],
            "count": 0,
        }
    return {"error": f"{type(exc).__name__}: {exc}", "items": items or [], "count": 0}


def _normalize_item(project: dict) -> dict:
    """Компактное представление проекта для передачи между инструментами."""
    owner = project.get("owner") or {}
    return {
        "id": project.get("id"),
        "title": project.get("title"),
        "alias": project.get("alias"),
        "description": project.get("description") or "",
        "language": project.get("language") or "",
        "owner": {"alias": owner.get("alias"), "type": owner.get("type")},
        "topics": list(project.get("topics") or []),
        "webUrl": project.get("webUrl") or "",
    }


@mcp.tool()
def search(
    query: Annotated[
        str, Field(description="Подстрока названия проекта для поиска (может быть пустой).")
    ] = "",
    page: Annotated[
        int, Field(description="Номер страницы, начиная с 0 (первая страница — 0).")
    ] = 0,
    size: Annotated[
        int, Field(description="Сколько проектов вернуть на странице (1..100).")
    ] = 10,
) -> dict:
    """Получить публичные проекты GitFlic — первый шаг пайплайна.

    Возвращает нормализованный список проектов (``items``) и данные пагинации.
    Именно ``items`` передаётся дальше в ``summarize``.
    """
    try:
        page_result = fetch_public_projects(query=query, page=page, size=size)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        return _error_result(exc)
    except Exception as exc:  # noqa: BLE001 — инструмент возвращает ошибку данными
        return _error_result(exc)

    items = [_normalize_item(project) for project in page_result.get("projects") or []]
    return {
        "query": query,
        "count": len(items),
        "total": page_result.get("total", len(items)),
        "page": page_result.get("page", {}),
        "source": "gitflic",
        "fetchedAt": now_iso(),
        "items": items,
    }


@mcp.tool()
def summarize(
    items: Annotated[
        list[dict],
        Field(description="Проекты от search: поля title, description, language, owner, topics."),
    ] = None,
    title: Annotated[
        str, Field(description="Заголовок/тема отчёта, подмешивается в текст выжимки.")
    ] = "",
    max_sentences: Annotated[
        int, Field(description="Сколько лучших предложений оставить в выжимке.")
    ] = 3,
    max_keywords: Annotated[
        int, Field(description="Сколько ключевых слов вернуть.")
    ] = 8,
) -> dict:
    """Сделать выжимку из проектов — второй шаг пайплайна.

    Локальная экстрактивная суммаризация (без сети и LLM): выбирает самые
    значимые предложения, собирает ключевые слова и краткие пункты по проектам.
    Возвращает ``summary``, ``bullets``, ``keywords`` и счётчики.
    """
    return summarize_items(
        items or [],
        title=title,
        max_sentences=max(1, int(max_sentences)),
        max_keywords=max(0, int(max_keywords)),
    )


def _resolve_output(path: str) -> Path:
    """Разрешает путь вывода и не даёт выйти за пределы каталога проекта."""
    candidate = Path(path or DEFAULT_OUTPUT)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if PROJECT_ROOT != resolved and PROJECT_ROOT not in resolved.parents:
        raise ValueError(
            f"Путь вывода должен находиться внутри проекта ({PROJECT_ROOT})."
        )
    return resolved


def _render(content, fmt: str) -> str:
    """Готовит содержимое к записи в выбранном формате."""
    if fmt == "json":
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"content": content}
        return json.dumps(content, ensure_ascii=False, indent=2)
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    return content


def _write_file(content, path: str, fmt: str, append: bool) -> dict:
    """Пишет результат в файл и возвращает метаданные (общий помощник)."""
    target = _resolve_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _render(content, fmt)
    if append and target.exists():
        text = target.read_text(encoding="utf-8") + "\n" + text
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return {
        "path": str(target),
        "relativePath": str(target.relative_to(PROJECT_ROOT)),
        "format": fmt,
        "bytes": len(text.encode("utf-8")),
        "lines": text.count("\n") + 1,
        "savedAt": now_iso(),
    }


@mcp.tool(name="saveToFile")
def save_to_file(
    content: Annotated[
        str | dict | list,
        Field(description="Что сохранить: текст, объект или список (для format=json)."),
    ] = "",
    path: Annotated[
        str,
        Field(description="Путь к файлу внутри проекта (по умолчанию history/pipeline_result.md)."),
    ] = DEFAULT_OUTPUT,
    format: Annotated[
        str, Field(description="Формат: markdown | text | json.")
    ] = "markdown",
    append: Annotated[
        bool, Field(description="Дописать в конец файла, а не перезаписать.")
    ] = False,
) -> dict:
    """Сохранить результат — третий шаг пайплайна.

    Атомарно пишет содержимое в файл внутри каталога проекта и возвращает путь,
    размер и время сохранения. Путь вне проекта отклоняется.
    """
    fmt = (format or "markdown").strip().lower()
    if fmt not in ("markdown", "text", "json"):
        return {"error": f"Неизвестный формат: {format!r} (markdown|text|json)."}
    try:
        return _write_file(content, path, fmt, append)
    except (ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _render_report(query: str, summary: dict) -> str:
    """Собирает Markdown-отчёт из результата ``summarize``."""
    lines = ["# Отчёт по проектам GitFlic", ""]
    if query:
        lines.append(f"Поисковый запрос: `{query}`")
        lines.append("")
    lines.append("## Выжимка")
    lines.append("")
    lines.append(summary.get("summary") or "(нет данных)")
    lines.append("")
    bullets = summary.get("bullets") or []
    if bullets:
        lines.append("## Проекты")
        lines.append("")
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")
    keywords = summary.get("keywords") or []
    if keywords:
        lines.append("## Ключевые слова")
        lines.append("")
        lines.append(", ".join(keywords))
        lines.append("")
    lines.append("---")
    lines.append(
        f"_Сформировано: {now_iso()}; проектов: {summary.get('itemCount', 0)}_"
    )
    return "\n".join(lines)


@mcp.tool()
def run_pipeline(
    query: Annotated[
        str, Field(description="Поисковый запрос для первого шага (search).")
    ] = "",
    output_path: Annotated[
        str, Field(description="Куда сохранить отчёт (saveToFile).")
    ] = DEFAULT_OUTPUT,
    size: Annotated[
        int, Field(description="Сколько проектов запросить у GitFlic.")
    ] = 10,
    max_sentences: Annotated[
        int, Field(description="Сколько предложений оставить в выжимке.")
    ] = 3,
) -> dict:
    """Выполнить всю цепочку search → summarize → saveToFile одним вызовом.

    Возвращает отчёт по шагам: имя шага, инструмент, фактические аргументы и
    результат. По нему видно, что данные корректно переданы между инструментами
    (``summarize`` получил ``items`` от ``search``, ``saveToFile`` — ``summary``
    от ``summarize``), а также путь сохранённого файла.
    """
    steps: list[dict] = []

    search_result = search(query=query, page=0, size=size)
    steps.append(
        {
            "name": "search",
            "tool": "search",
            "arguments": {"query": query, "page": 0, "size": size},
            "result": search_result,
        }
    )
    if search_result.get("error") or not search_result.get("items"):
        return {
            "ok": False,
            "query": query,
            "steps": steps,
            "error": search_result.get("error") or "Нет проектов по запросу.",
        }

    summary_result = summarize(
        items=search_result["items"],
        title=query,
        max_sentences=max_sentences,
    )
    steps.append(
        {
            "name": "summarize",
            "tool": "summarize",
            "arguments": {
                "items": search_result["items"],
                "title": query,
                "max_sentences": max_sentences,
            },
            "result": summary_result,
        }
    )

    report = _render_report(query, summary_result)
    saved = save_to_file(content=report, path=output_path, format="markdown")
    steps.append(
        {
            "name": "saveToFile",
            "tool": "saveToFile",
            "arguments": {"content": report, "path": output_path, "format": "markdown"},
            "result": saved,
        }
    )
    if saved.get("error"):
        return {"ok": False, "query": query, "steps": steps, "error": saved["error"]}

    return {
        "ok": True,
        "query": query,
        "savedPath": saved.get("relativePath"),
        "summary": summary_result,
        "steps": steps,
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструмент напрямую (без MCP) и печатает JSON-результат."""
    if args.pipeline:
        result = run_pipeline(
            query=args.query,
            output_path=args.output,
            size=args.size,
            max_sentences=args.max_sentences,
        )
    else:
        result = search(query=args.query, page=0, size=args.size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("error") else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер пайплайна: search → summarize → saveToFile"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструмент напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="в режиме --check выполнить всю цепочку search → summarize → saveToFile",
    )
    parser.add_argument("--query", default="", help="поисковый запрос")
    parser.add_argument("--size", type=int, default=5, help="сколько проектов запросить")
    parser.add_argument(
        "--max-sentences", type=int, default=3, help="предложений в выжимке"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="куда сохранить отчёт"
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout с клиентом.
    mcp.run()


if __name__ == "__main__":
    main()
