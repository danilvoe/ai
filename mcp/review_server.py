#!/usr/bin/env python3
"""День 32: MCP-сервер, отдающий ассистенту данные для ревью PR.

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и работает **локально**,
без сети: все инструменты вызывают ``git`` в каталоге репозитория. Он даёт
ассистенту ревью всё, что нужно для анализа pull request:

* ``get_pr_diff`` — diff между base и head (или изменения рабочего дерева);
* ``list_changed_files`` — список изменённых файлов со статусом и числом строк;
* ``get_file_content`` — содержимое файла на выбранной ревизии;
* ``get_pr_metadata`` — метаданные PR из GitHub-события (если доступны).

Команды ``git`` запускаются без shell (список аргументов), с ограничением вывода,
поэтому инструменты безопасны и предсказуемы.

Каталог репозитория задаётся переменной окружения ``REVIEW_REPO_ROOT`` (по
умолчанию — корень проекта).

Запуск (транспорт по умолчанию — stdio)::

    python3 mcp/review_server.py

Проверка из терминала без MCP::

    python3 mcp/review_server.py --check --diff --files
    python3 mcp/review_server.py --check --content agent/rag.py
    python3 mcp/review_server.py --check --metadata
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Корень проверяемого репозитория (можно переопределить через env)."""
    override = os.environ.get("REVIEW_REPO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


mcp = FastMCP(
    "review",
    log_level="WARNING",
    instructions=(
        "Инструменты для ревью pull request по локальному git-репозиторию: "
        "get_pr_diff возвращает diff между base и head, list_changed_files — "
        "изменённые файлы со статусом и числом строк, get_file_content — "
        "содержимое файла на ревизии, get_pr_metadata — метаданные PR из "
        "GitHub-события. Используй их, чтобы собрать входные данные ревью."
    ),
)


def _run_git(args: list[str], timeout: float = 20.0) -> tuple[str | None, str | None]:
    """Запускает ``git`` в каталоге репозитория. Возвращает (stdout, error)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root()),
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


def _range_args(base: str, head: str) -> str:
    """Собирает спецификацию диапазона для git diff.

    Три точки (``base...head``) дают diff от общего предка — так же, как diff
    pull request в GitHub/GitFlic. Если ``head`` пуст, сравниваем с ``base``
    (пусто — изменения рабочего дерева).
    """
    base = (base or "").strip()
    head = (head or "").strip()
    if head:
        return f"{base or 'HEAD'}...{head}"
    return base or "HEAD"


@mcp.tool()
def get_pr_diff(
    base: Annotated[str, Field(description="База сравнения (SHA/ветка), по умолчанию 'HEAD'.")] = "HEAD",
    head: Annotated[str, Field(description="Верхняя ревизия (SHA/ветка); пусто — рабочее дерево.")] = "",
    path: Annotated[str, Field(description="Ограничить diff конкретным путём (пусто — весь PR).")] = "",
    max_chars: Annotated[
        int, Field(description="Максимальная длина diff в символах (по умолчанию 20000).")
    ] = 20000,
) -> dict:
    """Получить diff pull request (base → head).

    По умолчанию это diff ``base...head`` (от общего предка). Если ``head`` пуст,
    возвращаются изменения рабочего дерева относительно ``base``. Ответ помечает
    усечение полем ``truncated``.
    """
    args = ["diff", _range_args(base, head)]
    if path.strip():
        args.extend(["--", path.strip()])
    output, error = _run_git(args)
    if error:
        return {"error": error}
    diff = output or ""
    limit = max(1, int(max_chars))
    return {
        "base": base or "HEAD",
        "head": head or "working tree",
        "path": path.strip(),
        "truncated": len(diff) > limit,
        "chars": len(diff),
        "diff": diff[:limit],
    }


@mcp.tool()
def list_changed_files(
    base: Annotated[str, Field(description="База сравнения (SHA/ветка), по умолчанию 'HEAD'.")] = "HEAD",
    head: Annotated[str, Field(description="Верхняя ревизия (SHA/ветка); пусто — рабочее дерево.")] = "",
) -> dict:
    """Перечислить изменённые в PR файлы со статусом и числом добавленных строк.

    Возвращает ``files``: путь, статус (``A``/``M``/``D``/``R``), число
    добавленных и удалённых строк, а также суммарные счётчики.
    """
    spec = _range_args(base, head)
    numstat, error = _run_git(["diff", "--numstat", spec])
    if error:
        return {"error": error}
    name_status, status_error = _run_git(["diff", "--name-status", spec])
    if status_error:
        name_status = ""

    statuses: dict[str, str] = {}
    for line in (name_status or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status = parts[0].strip()
            name = parts[-1].strip()
            statuses[name] = status[0] if status else "M"

    files: list[dict] = []
    add_total = 0
    del_total = 0
    for line in (numstat or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        raw_add, raw_del, name = parts[0], parts[1], parts[2]
        additions = int(raw_add) if raw_add.isdigit() else 0
        deletions = int(raw_del) if raw_del.isdigit() else 0
        add_total += additions
        del_total += deletions
        files.append(
            {
                "path": name.strip(),
                "status": statuses.get(name.strip(), "M"),
                "additions": additions,
                "deletions": deletions,
            }
        )

    return {
        "base": base or "HEAD",
        "head": head or "working tree",
        "count": len(files),
        "addTotal": add_total,
        "delTotal": del_total,
        "files": files,
    }


@mcp.tool()
def get_file_content(
    path: Annotated[str, Field(description="Путь к файлу относительно корня репозитория.")],
    ref: Annotated[str, Field(description="Ревизия (SHA/ветка); пусто — рабочий файл.")] = "",
    max_chars: Annotated[
        int, Field(description="Максимальная длина содержимого в символах (по умолчанию 20000).")
    ] = 20000,
) -> dict:
    """Прочитать содержимое файла на выбранной ревизии.

    Если ``ref`` пуст — читается файл из рабочего дерева. Ответ помечает усечение
    полем ``truncated``.
    """
    target = path.strip()
    if not target:
        return {"error": "не указан путь к файлу"}

    if ref.strip():
        output, error = _run_git(["show", f"{ref.strip()}:{target}"])
        if error:
            return {"error": error, "path": target, "ref": ref.strip()}
        content = output or ""
    else:
        file_path = repo_root() / target
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "path": target}

    limit = max(1, int(max_chars))
    return {
        "path": target,
        "ref": ref.strip() or "working tree",
        "truncated": len(content) > limit,
        "chars": len(content),
        "content": content[:limit],
    }


@mcp.tool()
def get_pr_metadata() -> dict:
    """Получить метаданные pull request из GitHub-события.

    Читает файл, на который указывает ``GITHUB_EVENT_PATH`` (payload события
    ``pull_request``), и возвращает номер, заголовок, описание, автора, базовую и
    верхнюю ветки/ревизии и ссылку. Без события (локальный запуск) — поля из
    переменных ``GITHUB_*`` или ``available: false``.
    """
    event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    if event_path and Path(event_path).exists():
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

        pr = payload.get("pull_request") or {}
        if pr:
            base = pr.get("base") or {}
            head = pr.get("head") or {}
            user = pr.get("user") or {}
            repo = (payload.get("repository") or {}).get("full_name", "")
            return {
                "available": True,
                "repository": repo,
                "number": pr.get("number"),
                "title": pr.get("title") or "",
                "body": (pr.get("body") or "")[:2000],
                "author": user.get("login") or "",
                "url": pr.get("html_url") or "",
                "baseRef": base.get("ref") or "",
                "headRef": head.get("ref") or "",
                "baseSha": base.get("sha") or "",
                "headSha": head.get("sha") or "",
                "draft": bool(pr.get("draft")),
            }

    return {
        "available": False,
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "eventName": os.environ.get("GITHUB_EVENT_NAME", ""),
        "refName": os.environ.get("GITHUB_REF_NAME", ""),
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструменты напрямую (без MCP) и печатает JSON-результат."""
    result: dict = {"repoRoot": str(repo_root())}
    ok = True
    if args.metadata:
        result["metadata"] = get_pr_metadata()
    if args.files:
        files = list_changed_files(base=args.base, head=args.head)
        result["files"] = files
        ok = ok and not files.get("error")
    if args.diff:
        diff = get_pr_diff(base=args.base, head=args.head, max_chars=args.max_chars)
        result["diff"] = diff
        ok = ok and not diff.get("error")
    if args.content:
        content = get_file_content(path=args.content, ref=args.ref or args.head)
        result["content"] = content
        ok = ok and not content.get("error")

    if not (args.metadata or args.files or args.diff or args.content):
        diff = get_pr_diff(base=args.base, head=args.head, max_chars=args.max_chars)
        result["diff"] = diff
        ok = not diff.get("error")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер ревью PR: diff, изменённые файлы, содержимое, метаданные"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="вызвать инструменты напрямую и напечатать результат (без MCP)",
    )
    parser.add_argument("--diff", action="store_true", help="в --check добавить get_pr_diff")
    parser.add_argument(
        "--files", action="store_true", help="в --check добавить list_changed_files"
    )
    parser.add_argument(
        "--metadata", action="store_true", help="в --check добавить get_pr_metadata"
    )
    parser.add_argument("--content", default="", help="путь файла для get_file_content")
    parser.add_argument("--ref", default="", help="ревизия для get_file_content (пусто — рабочий файл)")
    parser.add_argument("--base", default="HEAD", help="база сравнения")
    parser.add_argument("--head", default="", help="верхняя ревизия")
    parser.add_argument("--max-chars", type=int, default=4000, help="лимит вывода в --check")
    parser.add_argument("--repo-root", default="", help="корень репозитория (иначе REVIEW_REPO_ROOT)")
    args = parser.parse_args()

    if args.repo_root:
        os.environ["REVIEW_REPO_ROOT"] = args.repo_root

    if args.check:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: сервер общается по stdin/stdout с клиентом.
    mcp.run()


if __name__ == "__main__":
    main()
