#!/usr/bin/env python3
"""День 34: MCP-сервер файловых операций (чтение, поиск, анализ, запись).

Сервер построен на официальном SDK ``mcp`` (``FastMCP``) и работает **локально**,
без сети: все инструменты читают и пишут файлы внутри рабочей области, заданной
переменной окружения ``FS_ROOT`` (по умолчанию — корень проекта). Все пути
проверяются на выход за пределы корня (``..`` и абсолютные пути вне ``FS_ROOT``
отклоняются), поэтому инструменты безопасны и предсказуемы.

Инструменты:

* ``list_files`` — список файлов рабочей области (с фильтром по префиксу/glob);
* ``read_file`` — чтение файла или диапазона строк;
* ``search_files`` — поиск по нескольким файлам (результат по файлам);
* ``analyze_file`` — структура файла (для ``.py`` — через ``ast``);
* ``file_info`` — размер, mtime, число строк;
* ``write_file`` — создать/перезаписать файл;
* ``edit_file`` — точечная замена подстроки;
* ``make_diff`` — diff рабочей области относительно git или baseline-каталога.

Запуск (транспорт по умолчанию — stdio)::

    python3 mcp/fs_server.py

Проверка из терминала без MCP::

    python3 mcp/fs_server.py --check --files
    python3 mcp/fs_server.py --check --search "build_support_runtime" --path agent
    python3 mcp/fs_server.py --check --analyze agent/rag.py
"""

import argparse
import ast
import difflib
import fnmatch
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Каталоги, которые не обходим при листинге/поиске (мусор и артефакты).
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "history", ".idea", ".vscode", "dist", "build",
}

MAX_READ_CHARS = 200_000
MAX_SEARCH_RESULTS = 500


def workspace_root() -> Path:
    """Корень рабочей области (можно переопределить через env ``FS_ROOT``)."""
    override = os.environ.get("FS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


def _resolve(path: str, *, allow_root: bool = False) -> tuple[Path | None, str | None]:
    """Приводит путь к абсолютному внутри корня. Возвращает (path, error)."""
    root = workspace_root()
    raw = (path or "").strip()
    if not raw:
        if allow_root:
            return root, None
        return None, "не указан путь"
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None, f"путь вне рабочей области: {path}"
    return target, None


def _relative(path: Path) -> str:
    try:
        return path.relative_to(workspace_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_files(prefix: str = "", pattern: str = "") -> list[Path]:
    """Обходит рабочую область, пропуская служебные каталоги."""
    start, error = _resolve(prefix, allow_root=True)
    if error or start is None:
        return []
    if start.is_file():
        return [start]
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            found.append(Path(dirpath) / name)
    return found


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """Читает файл как UTF-8. Возвращает (text, error)."""
    try:
        if path.stat().st_size > MAX_READ_CHARS * 8:
            return None, f"файл слишком большой: {path.stat().st_size} байт"
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, f"файл не найден: {_relative(path)}"
    except UnicodeDecodeError:
        return None, f"не текстовый файл (не UTF-8): {_relative(path)}"
    except OSError as exc:  # pragma: no cover — редкий случай
        return None, f"{type(exc).__name__}: {exc}"


mcp = FastMCP(
    "filesystem",
    log_level="WARNING",
    instructions=(
        "Инструменты работы с файлами проекта: list_files перечисляет файлы, "
        "read_file читает файл, search_files ищет по нескольким файлам, "
        "analyze_file разбирает структуру файла, file_info даёт метаданные, "
        "write_file и edit_file создают/изменяют файлы, make_diff показывает "
        "изменения. Используй их, чтобы выполнять реальные операции с файлами."
    ),
)


@mcp.tool()
def list_files(
    path: Annotated[str, Field(description="Префикс пути/каталог (пусто — вся область).")] = "",
    pattern: Annotated[str, Field(description="Glob-шаблон имени файла, например '*.py'.")] = "",
    limit: Annotated[int, Field(description="Сколько файлов вернуть (по умолчанию 200).")] = 200,
) -> dict:
    """Перечислить файлы рабочей области.

    Возвращает отсортированный список путей (относительно корня), отфильтрованный
    по префиксу ``path`` и glob ``pattern``. Служебные каталоги (``.git``,
    ``history``, ``__pycache__`` и т.п.) пропускаются.
    """
    files = _iter_files(path, pattern)
    count = max(1, int(limit))
    names = [_relative(item) for item in files]
    return {
        "path": (path or "").strip(),
        "pattern": pattern,
        "total": len(names),
        "returned": min(len(names), count),
        "files": names[:count],
    }


@mcp.tool()
def read_file(
    path: Annotated[str, Field(description="Путь к файлу относительно корня области.")],
    start_line: Annotated[int, Field(description="Первая строка (1-based, по умолчанию 1).")] = 1,
    end_line: Annotated[int, Field(description="Последняя строка (0 — до конца файла).")] = 0,
    max_chars: Annotated[int, Field(description="Максимум символов в ответе.")] = 8000,
) -> dict:
    """Прочитать файл или диапазон строк.

    Возвращает содержимое (``content``), число строк в файле (``total_lines``),
    запрошенный диапазон и признак усечения ``truncated``.
    """
    target, error = _resolve(path)
    if error or target is None:
        return {"error": error}
    if target.is_dir():
        return {"error": f"это каталог, а не файл: {path}"}
    text, error = _read_text(target)
    if error:
        return {"error": error}

    lines = text.splitlines()
    total = len(lines)
    first = max(1, int(start_line or 1))
    last = int(end_line) if end_line and int(end_line) >= first else total
    selected = "\n".join(lines[first - 1:last])
    limit = max(1, int(max_chars))
    truncated = len(selected) > limit
    return {
        "path": _relative(target),
        "start_line": first,
        "end_line": min(last, total),
        "total_lines": total,
        "truncated": truncated,
        "content": selected[:limit],
    }


@mcp.tool()
def search_files(
    query: Annotated[str, Field(description="Строка или регулярное выражение для поиска.")],
    path: Annotated[str, Field(description="Каталог/префикс для поиска (пусто — вся область).")] = "",
    glob: Annotated[str, Field(description="Glob-шаблон файлов, например '*.py'.")] = "",
    regex: Annotated[bool, Field(description="Считать query регулярным выражением.")] = False,
    case_sensitive: Annotated[bool, Field(description="Учитывать регистр.")] = False,
    limit: Annotated[int, Field(description="Сколько файлов-совпадений вернуть.")] = 100,
    context: Annotated[int, Field(description="Сколько соседних строк показывать (0 — только строка).")] = 0,
) -> dict:
    """Найти текст по нескольким файлам.

    Возвращает ``files`` — список ``{file, count, matches: [{line, text, before,
    after}]}``, отсортированный по числу совпадений, а также общее число
    совпавших файлов ``total_files`` и совпадений ``total_matches``.
    """
    needle = query or ""
    if not needle.strip():
        return {"error": "пустой поисковый запрос"}
    try:
        pattern = re.compile(needle, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return {"error": f"некорректное регулярное выражение: {exc}"}

    results: list[dict] = []
    total_matches = 0
    ctx = max(0, int(context))
    for item in _iter_files(path, glob):
        text, error = _read_text(item)
        if error:
            continue
        lines = text.splitlines()
        matches: list[dict] = []
        for index, line in enumerate(lines):
            hit = pattern.search(line) if regex else (
                needle in line if case_sensitive else needle.lower() in line.lower()
            )
            if not hit:
                continue
            entry: dict = {"line": index + 1, "text": line[:400]}
            if ctx:
                entry["before"] = lines[max(0, index - ctx):index]
                entry["after"] = lines[index + 1:index + 1 + ctx]
            matches.append(entry)
        if matches:
            results.append({"file": _relative(item), "count": len(matches), "matches": matches})
            total_matches += len(matches)

    results.sort(key=lambda entry: (-entry["count"], entry["file"]))
    count = max(1, int(limit))
    return {
        "query": query,
        "path": (path or "").strip(),
        "glob": glob,
        "total_files": len(results),
        "total_matches": total_matches,
        "returned": min(len(results), count),
        "files": results[:count],
    }


@mcp.tool()
def analyze_file(
    path: Annotated[str, Field(description="Путь к файлу для анализа.")],
) -> dict:
    """Проанализировать структуру файла.

    Для Python-файлов разбирает модуль через ``ast`` (докстринг, импорты, классы,
    функции, константы). Для markdown — список заголовков. Для остальных —
    базовую статистику (строки, символы).
    """
    target, error = _resolve(path)
    if error or target is None:
        return {"error": error}
    text, error = _read_text(target)
    if error:
        return {"error": error}
    lines = text.splitlines()
    info: dict = {
        "path": _relative(target),
        "suffix": target.suffix,
        "total_lines": len(lines),
        "chars": len(text),
    }

    if target.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {**info, "error": f"SyntaxError: {exc}", "kind": "python"}
        imports: list[str] = []
        classes: list[dict] = []
        functions: list[dict] = []
        constants: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module or ''}")
            elif isinstance(node, ast.ClassDef):
                classes.append({"name": node.name, "line": node.lineno,
                                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]})
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.append({"name": node.name, "line": node.lineno})
            elif isinstance(node, ast.Assign):
                for target_node in node.targets:
                    if isinstance(target_node, ast.Name) and target_node.id.isupper():
                        constants.append(target_node.id)
        info.update({
            "kind": "python",
            "docstring": ast.get_docstring(tree) or "",
            "imports": sorted(set(filter(None, imports))),
            "classes": classes,
            "functions": functions,
            "constants": sorted(set(constants)),
        })
    elif target.suffix in (".md", ".markdown"):
        headings = [
            {"level": len(match.group(1)), "title": match.group(2).strip(), "line": index + 1}
            for index, line in enumerate(lines)
            if (match := re.match(r"^(#{1,6})\s+(.*)$", line.strip()))
        ]
        info.update({"kind": "markdown", "headings": headings})
    else:
        info["kind"] = "text"
    return info


@mcp.tool()
def file_info(
    path: Annotated[str, Field(description="Путь к файлу.")],
) -> dict:
    """Метаданные файла: размер, время изменения, число строк."""
    target, error = _resolve(path)
    if error or target is None:
        return {"error": error}
    if not target.exists():
        return {"error": f"файл не найден: {path}"}
    try:
        stat = target.stat()
    except OSError as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}
    lines = None
    if target.is_file():
        text, read_error = _read_text(target)
        lines = len(text.splitlines()) if read_error is None else None
    return {
        "path": _relative(target),
        "is_dir": target.is_dir(),
        "size": stat.st_size,
        "lines": lines,
        "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
    }


@mcp.tool()
def write_file(
    path: Annotated[str, Field(description="Путь к файлу для записи.")],
    content: Annotated[str, Field(description="Полное содержимое файла.")],
    overwrite: Annotated[bool, Field(description="Разрешить перезапись существующего файла.")] = True,
    create_dirs: Annotated[bool, Field(description="Создавать недостающие каталоги.")] = True,
) -> dict:
    """Создать или перезаписать файл.

    Если файл существует, а ``overwrite=False`` — запись пропускается (``skipped``).
    Возвращает путь, число записанных символов и признак ``created``.
    """
    target, error = _resolve(path)
    if error or target is None:
        return {"error": error}
    existed = target.exists()
    if existed and not overwrite:
        return {"path": _relative(target), "skipped": True, "reason": "файл существует, overwrite=False"}
    try:
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": _relative(target),
        "created": not existed,
        "chars": len(content),
        "lines": content.count("\n") + (1 if content else 0),
    }


@mcp.tool()
def edit_file(
    path: Annotated[str, Field(description="Путь к изменяемому файлу.")],
    old: Annotated[str, Field(description="Подстрока, которую нужно заменить.")],
    new: Annotated[str, Field(description="Новая подстрока.")],
    count: Annotated[int, Field(description="Сколько вхождений заменить (0 — все).")] = 1,
) -> dict:
    """Заменить подстроку в файле.

    Заменяет ``count`` первых вхождений ``old`` на ``new`` (``count=0`` — все).
    Если ``old`` не найден — ``error``.
    """
    target, error = _resolve(path)
    if error or target is None:
        return {"error": error}
    text, error = _read_text(target)
    if error:
        return {"error": error}
    occurrences = text.count(old)
    if occurrences == 0:
        return {"error": "подстрока не найдена"}
    replacements = occurrences if int(count) == 0 else max(1, int(count))
    updated = text.replace(old, new, replacements)
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": _relative(target),
        "occurrences": occurrences,
        "replaced": min(replacements, occurrences),
        "chars": len(updated),
    }


def _unified_diff(name: str, before: str, after: str) -> str:
    """Unified diff двух версий текста."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )
    )


def _diff_against_baseline(path: Path, baseline_root: Path) -> list[dict]:
    """Сравнивает файл/каталог с baseline-каталогом (difflib)."""
    root = workspace_root()
    skip = {".git", "__pycache__"}
    targets = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    changes: list[dict] = []
    for item in targets:
        if any(part in skip for part in item.parts):
            continue
        rel = item.relative_to(root)
        base = baseline_root / rel
        before = base.read_text(encoding="utf-8") if base.exists() else ""
        after = item.read_text(encoding="utf-8")
        if before == after:
            continue
        changes.append({
            "path": rel.as_posix(),
            "status": "modified" if base.exists() else "added",
            "diff": _unified_diff(rel.as_posix(), before, after),
        })
    return changes


@mcp.tool()
def make_diff(
    path: Annotated[str, Field(description="Файл или каталог (пусто — вся область).")] = "",
    baseline_root: Annotated[
        str, Field(description="Каталог-эталон для сравнения (difflib). Пусто — git diff.")
    ] = "",
    max_chars: Annotated[int, Field(description="Максимум символов diff.")] = 8000,
) -> dict:
    """Показать изменения файлов.

    Если задан ``baseline_root`` — сравнивает файлы рабочей области с эталонным
    каталогом (unified diff через ``difflib``). Иначе пытается выполнить
    ``git diff`` в рабочей области. Возвращает список изменённых файлов и diff.
    """
    target, error = _resolve(path, allow_root=True)
    if error or target is None:
        return {"error": error}
    limit = max(1, int(max_chars))

    if baseline_root.strip():
        base = Path(baseline_root).expanduser().resolve()
        if not base.exists():
            return {"error": f"baseline-каталог не найден: {baseline_root}"}
        changes = _diff_against_baseline(target, base)
        combined = "".join(change["diff"] for change in changes)
        return {
            "mode": "baseline",
            "baseline": str(base),
            "changed": len(changes),
            "files": [change["path"] for change in changes],
            "truncated": len(combined) > limit,
            "diff": combined[:limit],
        }

    try:
        proc = subprocess.run(
            ["git", "diff", "--", str(target)],
            cwd=str(workspace_root()), capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or "git diff недоступен").strip()}
    diff = proc.stdout or ""
    files = sorted({line.split(" b/")[-1] for line in diff.splitlines() if line.startswith("diff --git")})
    return {
        "mode": "git",
        "changed": len(files),
        "files": files,
        "truncated": len(diff) > limit,
        "diff": diff[:limit],
    }


def _run_check(args: argparse.Namespace) -> int:
    """Выполняет инструменты напрямую (без MCP) и печатает JSON-результат."""
    result: dict = {"root": str(workspace_root())}
    ok = True
    if args.files:
        result["files"] = list_files(path=args.path, pattern=args.glob, limit=args.limit)
    if args.read:
        result["read"] = read_file(args.read, max_chars=args.max_chars)
        ok = ok and not result["read"].get("error")
    if args.search:
        result["search"] = search_files(args.search, path=args.path, glob=args.glob, limit=args.limit)
        ok = ok and not result["search"].get("error")
    if args.analyze:
        result["analyze"] = analyze_file(args.analyze)
        ok = ok and not result["analyze"].get("error")
    if args.info:
        result["info"] = file_info(args.info)
        ok = ok and not result["info"].get("error")
    if args.diff:
        result["diff"] = make_diff(path=args.path, max_chars=args.max_chars)
        ok = ok and not result["diff"].get("error")

    if not (args.files or args.read or args.search or args.analyze or args.info or args.diff):
        result["files"] = list_files(limit=args.limit)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP-сервер файловых операций: чтение, поиск, анализ, запись, diff"
    )
    parser.add_argument("--check", action="store_true", help="вызвать инструменты напрямую (без MCP)")
    parser.add_argument("--files", action="store_true", help="в --check добавить list_files")
    parser.add_argument("--read", default="", help="путь для read_file")
    parser.add_argument("--search", default="", help="запрос для search_files")
    parser.add_argument("--analyze", default="", help="путь для analyze_file")
    parser.add_argument("--info", default="", help="путь для file_info")
    parser.add_argument("--diff", action="store_true", help="в --check добавить make_diff")
    parser.add_argument("--path", default="", help="путь/префикс для --files/--search/--diff")
    parser.add_argument("--glob", default="", help="glob-шаблон для --files/--search")
    parser.add_argument("--limit", type=int, default=50, help="лимит результатов")
    parser.add_argument("--max-chars", type=int, default=2000, help="лимит вывода в --check")
    parser.add_argument("--root", default="", help="рабочая область (иначе FS_ROOT)")
    args = parser.parse_args()

    if args.root:
        os.environ["FS_ROOT"] = args.root

    if args.check or args.files or args.read or args.search or args.analyze or args.info or args.diff:
        raise SystemExit(_run_check(args))

    # Транспорт по умолчанию — stdio: stdout — канал MCP, подсказку пишем в stderr.
    print(
        "MCP-сервер файлов запущен (stdio). Для разовой проверки используйте "
        "--check (например: --check --files --search \"build_support_runtime\").",
        file=os.sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
