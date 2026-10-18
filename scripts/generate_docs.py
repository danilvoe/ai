#!/usr/bin/env python3
"""День 31: генерация справочника по модулям проекта (``docs/modules.md``).

Ассистент разработчика должен отвечать на вопросы о структуре проекта, поэтому
документация по коду генерируется автоматически из исходников. Скрипт обходит
пакеты ``agent/`` и ``mcp/``, разбирает модули через ``ast`` (без импорта, чтобы
не тянуть зависимости вроде ``faiss`` и не запускать MCP-серверы) и собирает:

* назначение модуля (первый абзац docstring);
* публичные классы и функции с сигнатурами и кратким описанием;
* количество строк и путь к файлу.

Результат детерминирован, поэтому файл можно пересобирать и индексировать в RAG
(см. :mod:`agent.project_docs`).

Запуск::

    python3 scripts/generate_docs.py
    python3 scripts/generate_docs.py --output docs/modules.md
    python3 scripts/generate_docs.py --check          # проверить, что файл актуален
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGES = ("agent", "mcp")
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "modules.md"

HEADER = (
    "# Справочник по модулям\n\n"
    "> Файл сгенерирован автоматически командой "
    "`python3 scripts/generate_docs.py`.\n"
    "> Не редактируйте его вручную — изменения будут перезаписаны.\n"
)


def _first_paragraph(docstring: str | None) -> str:
    """Возвращает первый абзац docstring одной строкой."""
    if not docstring:
        return ""
    lines: list[str] = []
    for line in docstring.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def _annotation(node: ast.AST | None) -> str:
    """Текстовая аннотация узла (``ast.unparse``), либо пустая строка."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — некорректная аннотация не должна ронять генерацию
        return ""


def _format_args(args: ast.arguments) -> str:
    """Собирает список параметров функции из ``ast.arguments``."""
    parts: list[str] = []
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults = list(args.defaults)
    default_start = len(positional) - len(defaults)

    for index, arg in enumerate(positional):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_annotation(arg.annotation)}"
        if index >= default_start:
            text += f" = {_annotation(defaults[index - default_start])}"
        parts.append(text)

    if args.vararg is not None:
        text = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            text += f": {_annotation(args.vararg.annotation)}"
        parts.append(text)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_annotation(arg.annotation)}"
        if default is not None:
            text += f" = {_annotation(default)}"
        parts.append(text)

    if args.kwarg is not None:
        text = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            text += f": {_annotation(args.kwarg.annotation)}"
        parts.append(text)

    return ", ".join(parts)


def _format_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Сигнатура функции/метода с учётом async и возвращаемого типа."""
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    signature = f"{prefix}{node.name}({_format_args(node.args)})"
    returns = _annotation(node.returns)
    if returns:
        signature += f" -> {returns}"
    return signature


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _collect_module(path: Path) -> dict | None:
    """Разбирает один модуль и возвращает его публичное описание."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    classes: list[dict] = []
    functions: list[dict] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            bases = [_annotation(base) for base in node.bases]
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
                    item.name
                ):
                    methods.append(
                        {
                            "signature": _format_function(item),
                            "summary": _first_paragraph(ast.get_docstring(item)),
                        }
                    )
            classes.append(
                {
                    "name": node.name,
                    "bases": [b for b in bases if b],
                    "summary": _first_paragraph(ast.get_docstring(node)),
                    "methods": methods,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            functions.append(
                {
                    "signature": _format_function(node),
                    "summary": _first_paragraph(ast.get_docstring(node)),
                }
            )

    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "lines": source.count("\n") + 1,
        "summary": _first_paragraph(ast.get_docstring(tree)),
        "classes": classes,
        "functions": functions,
    }


def collect_modules(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, list[dict]]:
    """Собирает описания всех ``*.py``-модулей указанных пакетов."""
    result: dict[str, list[dict]] = {}
    for package in packages:
        package_dir = PROJECT_ROOT / package
        if not package_dir.is_dir():
            continue
        modules: list[dict] = []
        for path in sorted(package_dir.glob("*.py")):
            info = _collect_module(path)
            if info is not None:
                modules.append(info)
        result[package] = modules
    return result


def render(modules: dict[str, list[dict]]) -> str:
    """Формирует Markdown-справочник по собранным модулям."""
    lines: list[str] = [HEADER, ""]
    total = sum(len(items) for items in modules.values())
    lines.append(f"Всего модулей: **{total}**.\n")

    for package, items in modules.items():
        lines.append(f"## Пакет `{package}/`\n")
        for module in items:
            lines.append(f"### `{module['path']}` ({module['lines']} строк)\n")
            if module["summary"]:
                lines.append(f"{module['summary']}\n")
            if module["classes"]:
                lines.append("**Классы:**\n")
                for cls in module["classes"]:
                    base_text = f" ({', '.join(cls['bases'])})" if cls["bases"] else ""
                    summary = f" — {cls['summary']}" if cls["summary"] else ""
                    lines.append(f"- `{cls['name']}`{base_text}{summary}")
                    for method in cls["methods"]:
                        method_summary = f" — {method['summary']}" if method["summary"] else ""
                        lines.append(f"  - `{method['signature']}`{method_summary}")
                lines.append("")
            if module["functions"]:
                lines.append("**Функции:**\n")
                for func in module["functions"]:
                    summary = f" — {func['summary']}" if func["summary"] else ""
                    lines.append(f"- `{func['signature']}`{summary}")
                lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Генерация справочника по модулям проекта (docs/modules.md)"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="куда записать справочник (по умолчанию docs/modules.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="не писать файл, а проверить, что он совпадает со сгенерированным",
    )
    args = parser.parse_args()

    content = render(collect_modules())
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    if args.check:
        if not output.exists():
            print(f"✗ Файл не найден: {output}", file=sys.stderr)
            return 1
        current = output.read_text(encoding="utf-8")
        if current != content:
            print(
                f"✗ {output} устарел. Запустите: python3 scripts/generate_docs.py",
                file=sys.stderr,
            )
            return 1
        print(f"✓ {output} актуален.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"✓ Справочник записан: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
