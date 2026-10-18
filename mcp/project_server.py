#!/usr/bin/env python3
"""День 31: MCP-сервер, подключающий ассистента к самому проекту.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и работает **локально**,
без сети: все инструменты вызывают ``git`` в каталоге проекта. Он даёт ассистенту
разработчика живой контекст репозитория:

* ``get_current_branch`` — текущая git-ветка (обязательный минимум Дня 31);
* ``get_repo_status`` — ветка, upstream, ahead/behind и изменённые файлы;
* ``list_project_files`` — список файлов под контролем версий (с фильтром по пути);
* ``get_recent_commits`` — последние коммиты;
* ``get_diff`` — diff рабочего дерева или индекса.

Команды ``git`` запускаются без shell (список аргументов), с ограничением вывода,
поэтому инструменты безопасны и предсказуемы.

Запуск (транспорт по умолчанию — stdio)::

    python3 mcp/project_server.py

Проверка из терминала без MCP::

    python3 mcp/project_server.py --check
    python3 mcp/project_server.py --check --status
    python3 mcp/project_server.py --check --files --commits
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mcp = FastMCP(
    "project",
    log_level="WARNING",
    instructions=(
        "Инструменты доступа к текущему git-репозиторию проекта: "
        "get_current_branch возвращает активную ветку, get_repo_status — "
        "состояние рабочего дерева, list_project_files — список файлов, "
        "get_recent_commits — историю коммитов, get_diff — изменения. "
        "Используй их, чтобы отвечать на вопросы о состоянии и структуре проекта."
    ),
)


def _run_git(args: list[str], timeout: float = 20.0) -> tuple[str | None, str | None]:
    """Запускает ``git`` в каталоге проекта. Возвращает (stdout, error)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return None, detail or f"git вернул код {proc.returncode}"
    return proc.stdout, None


@mcp.tool()
def get_current_branch() -> dict:
    """Получить текущую git-ветку проекта.

    Возвращает имя активной ветки, флаг отсоединённого HEAD (``detached``) и
    короткий хеш коммита (``commit``). Если каталог не является git-репозиторием,
    поле ``is_repo`` равно ``false``, а в ``error`` — причина.
    """
    branch_out, branch_err = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch_err:
        return {"is_repo": False, "branch": "", "detached": False, "commit": "", "error": branch_err}

    branch = (branch_out or "").strip()
    detached = branch == "HEAD"
    commit_out, _ = _run_git(["rev-parse", "--short", "HEAD"])
    commit = (commit_out or "").strip()
    return {
        "is_repo": True,
        "branch": branch if not detached else f"(detached HEAD {commit})",
        "detached": detached,
        "commit": commit,
    }


def _parse_status(output: str) -> dict:
    """Разбирает вывод ``git status --porcelain=v1 --branch``."""
    branch = ""
    upstream = ""
    ahead = 0
    behind = 0
    staged: list[str] = []
    modified: list[str] = []
    untracked: list[str] = []

    for line in output.splitlines():
        if line.startswith("## "):
            header = line[3:]
            if "..." in header:
                branch, rest = header.split("...", 1)
                upstream = rest.split(" ", 1)[0]
                if "ahead " in header:
                    ahead = int(header.split("ahead ", 1)[1].split("]", 1)[0].split(",")[0])
                if "behind " in header:
                    behind = int(header.split("behind ", 1)[1].split("]", 1)[0].split(",")[0])
            else:
                branch = header.split(" ", 1)[0]
            continue
        if len(line) < 3:
            continue
        code = line[:2]
        name = line[3:].strip()
        if code == "??":
            untracked.append(name)
            continue
        if code[0] not in (" ", "?"):
            staged.append(name)
        if code[1] not in (" ", "?"):
            modified.append(name)

    return {
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
    }


@mcp.tool()
def get_repo_status(
    limit: Annotated[
        int, Field(description="Сколько файлов показывать в каждом списке (по умолчанию 30).")
    ] = 30,
) -> dict:
    """Получить состояние рабочего дерева проекта.

    Возвращает ветку, upstream, число коммитов ahead/behind и списки файлов:
    подготовленные (``staged``), изменённые (``modified``) и неотслеживаемые
    (``untracked``). Списки обрезаются до ``limit`` элементов.
    """
    output, error = _run_git(["status", "--porcelain=v1", "--branch"])
    if error:
        return {"error": error}
    parsed = _parse_status(output or "")
    count = max(1, int(limit))

    def _cap(items: list[str]) -> tuple[list[str], int]:
        return items[:count], len(items)

    staged, staged_total = _cap(parsed["staged"])
    modified, modified_total = _cap(parsed["modified"])
    untracked, untracked_total = _cap(parsed["untracked"])
    return {
        **{key: parsed[key] for key in ("branch", "upstream", "ahead", "behind")},
        "staged": staged,
        "modified": modified,
        "untracked": untracked,
        "stagedTotal": staged_total,
        "modifiedTotal": modified_total,
        "untrackedTotal": untracked_total,
        "clean": staged_total + modified_total + untracked_total == 0,
    }


@mcp.tool()
def list_project_files(
    path: Annotated[
        str, Field(description="Префикс пути для фильтра (например, 'agent/' или 'docs').")
    ] = "",
    limit: Annotated[
        int, Field(description="Сколько файлов вернуть (по умолчанию 200).")
    ] = 200,
) -> dict:
    """Перечислить файлы проекта под контролем версий.

    Возвращает отсортированный список путей (относительно корня проекта),
    отфильтрованный по префиксу ``path`` и обрезанный до ``limit``.
    """
    output, error = _run_git(["ls-files"])
    if error:
        return {"error": error}
    files = sorted(line.strip() for line in (output or "").splitlines() if line.strip())
    prefix = path.strip().lstrip("./")
    if prefix:
        files = [name for name in files if name.startswith(prefix)]
    count = max(1, int(limit))
    return {
        "path": prefix,
        "total": len(files),
        "returned": min(len(files), count),
        "files": files[:count],
    }


@mcp.tool()
def get_recent_commits(
    limit: Annotated[
        int, Field(description="Сколько последних коммитов вернуть (по умолчанию 10).")
    ] = 10,
) -> dict:
    """Получить последние коммиты репозитория.

    Возвращает список коммитов: полный и короткий хеш, автор, дата и тема.
    """
    count = max(1, int(limit))
    output, error = _run_git(
        ["log", f"-n{count}", "--pretty=format:%H%x1f%h%x1f%an%x1f%ad%x1f%s", "--date=short"]
    )
    if error:
        return {"error": error}
    commits = []
    for line in (output or "").splitlines():
        parts = line.split("\x1f")
        if len(parts) < 5:
            continue
        commits.append(
            {"hash": parts[0], "short": parts[1], "author": parts[2], "date": parts[3], "subject": parts[4]}
        )
    return {"count": len(commits), "commits": commits}


@mcp.tool()
def get_diff(
    base: Annotated[
        str, Field(description="База сравнения (по умолчанию 'HEAD'); пусто — рабочие изменения.")
    ] = "HEAD",
    staged: Annotated[
        bool, Field(description="Показать изменения индекса (--staged), а не рабочего дерева.")
    ] = False,
    path: Annotated[
        str, Field(description="Ограничить diff конкретным путём (пусто — весь проект).")
    ] = "",
    max_chars: Annotated[
        int, Field(description="Максимальная длина diff в символах (по умолчанию 4000).")
    ] = 4000,
) -> dict:
    """Получить diff изменений проекта.

    По умолчанию — изменения рабочего дерева относительно ``HEAD``. Можно
    ограничить конкретным путём (``path``), показать индекс (``staged``) и задать
    лимит длины (``max_chars``). Ответ помечает усечение полем ``truncated``.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    elif base:
        args.append(base)
    if path.strip():
        args.extend(["--", path.strip()])
    output, error = _run_git(args)
    if error:
        return {"error": error}
    diff = output or ""
    limit = max(1, int(max_chars))
    truncated = len(diff) > limit
    return {
        "base": "staged" if staged else (base or "working tree"),
        "path": path.strip(),
        "truncated": truncated,
        "chars": len(diff),
        "diff": diff[:limit],
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструменты напрямую (без MCP) и печатает JSON-результат."""
    result: dict = {"branch": get_current_branch()}
    if args.status:
        result["status"] = get_repo_status(limit=args.limit)
    if args.files:
        result["files"] = list_project_files(path=args.path, limit=args.limit)
    if args.commits:
        result["commits"] = get_recent_commits(limit=args.limit)
    if args.diff:
        result["diff"] = get_diff(path=args.path, max_chars=args.max_chars)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    branch = result["branch"]
    return 0 if branch.get("is_repo") and not branch.get("error") else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер проекта: git-ветка, файлы, коммиты, diff"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструменты напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument("--status", action="store_true", help="в --check добавить get_repo_status")
    parser.add_argument("--files", action="store_true", help="в --check добавить list_project_files")
    parser.add_argument("--commits", action="store_true", help="в --check добавить get_recent_commits")
    parser.add_argument("--diff", action="store_true", help="в --check добавить get_diff")
    parser.add_argument("--path", default="", help="фильтр пути для --files/--diff")
    parser.add_argument("--limit", type=int, default=10, help="лимит для --status/--files/--commits")
    parser.add_argument("--max-chars", type=int, default=2000, help="лимит diff в --check")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout с клиентом.
    mcp.run()


if __name__ == "__main__":
    main()
