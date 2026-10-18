"""День 34: ассистент для работы с файлами проекта.

Ассистент получает **цель** («найди, где используется X», «сгенерируй changelog»,
«обнови документацию», «проверь соответствие правилам», «подготовь diff») и
**сам** решает, какие файлы читать, искать и изменять. Реальные операции с файлами
выполняются через MCP-сервер ``mcp/fs_server.py`` (чтение, поиск по нескольким
файлам, анализ, запись, diff).

Компоненты:

* :func:`classify_goal` — определяет тип задачи по формулировке цели;
* :class:`FileAssistant` — оркестратор: цель → вызовы MCP-инструментов →
  результат (находки, действия, артефакты и diff'ы);
* :class:`FileAssistantStubClient` — детерминированный офлайн-генератор
  содержимого (интерфейс ``LLMClient``);
* :func:`build_file_assistant` — фабрика (MCP-рантайм + клиент + настройки).

Безопасность: по умолчанию изменения **не записываются** (``apply=False``) —
ассистент показывает их как unified diff. Запись выполняется только при
``apply=True``. Для воспроизводимых проверок используется песочница
(см. :mod:`agent.file_scenarios`).

Запуск CLI: ``python3 chat_files.py`` (см. :mod:`agent.file_cli`).
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import Completion, LLMClient
from .mcp_tools import McpToolRuntime, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FS_SERVER = PROJECT_ROOT / "mcp" / "fs_server.py"

# Типы задач, которые умеет распознавать ассистент.
TASK_FIND_USAGES = "find_usages"
TASK_GENERATE_FILE = "generate_file"
TASK_UPDATE_DOCS = "update_docs"
TASK_CHECK_INVARIANTS = "check_invariants"
TASK_PREPARE_DIFF = "prepare_diff"
TASK_UNKNOWN = "unknown"

FACTS_MARKER = "ФАКТЫ (JSON):"
GOAL_MARKER = "ЦЕЛЬ:"

FILE_ASSISTANT_SYSTEM_PROMPT = """Ты — ассистент, который работает с файлами проекта.
Ты получаешь цель пользователя и структурированные факты (JSON), собранные из
файлов через MCP-инструменты, и формируешь итоговый результат.

Правила:
1. Опирайся ТОЛЬКО на предоставленные факты — не выдумывай пути, символы и строки.
2. Для генерации документа верни его содержимое в markdown, без пояснений.
3. Отвечай по-русски, структурированно и по делу."""

# Полное чтение файла для анализа/диффа (не для контекста промпта).
FULL_READ_CHARS = 200_000

# Имя инструмента MCP в декораторе: ``@mcp.tool()`` или ``@mcp.tool(name="...")``.
_TOOL_DECORATOR_RE = re.compile(
    r"@mcp\.tool\(\s*(?:name\s*=\s*[\"']([^\"']+)[\"']\s*)?\)\s*\n"
    r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)"
)

# Слова, которые не являются искомым символом при разборе цели.
_STOPWORDS = {
    "найди", "найти", "используется", "использование", "использования", "вызов",
    "вызовы", "ссылки", "ссылк", "проект", "проекта", "файл", "файлах", "файлы",
    "где", "все", "всех", "места", "местах", "код", "коде", "обнови", "обновить",
    "документацию", "документация", "сгенерируй", "создай", "сделай", "проверь",
    "проверить", "соответствие", "правилам", "инвариант", "инвариантам",
    "подготовь", "diff", "изменения", "изменений", "список", "changelog", "adr",
    "readme", "каталог", "отчёт", "отчет",
}


# ============================================================================
# 1. Классификация цели
# ============================================================================


def classify_goal(goal: str) -> str:
    """Определяет тип задачи по формулировке цели.

    Порядок проверок важен: более специфичные задачи (diff, инварианты,
    генерация) распознаются раньше, чем общий поиск использований.
    """
    text = (goal or "").lower()

    def has(*markers: str) -> bool:
        return any(marker in text for marker in markers)

    if has("diff", "диф", "список изменени", "что поменялось", "что изменилось",
           "изменённые файл", "измененные файл"):
        return TASK_PREPARE_DIFF
    if has("инвариант", "соответствие правил", "проверь соответствие",
           "проверить соответствие", "проверь файлы", "линт", "правилам проекта"):
        return TASK_CHECK_INVARIANTS
    if has("сгенерируй", "создай файл", "сгенерировать", "новый файл", "changelog",
           "adr", "каталог серверов", "каталог mcp"):
        return TASK_GENERATE_FILE
    if has("обнови документ", "обновить документ", "актуализируй", "обнови docs",
           "приведи в соответствие документ"):
        return TASK_UPDATE_DOCS
    if has("найди", "где используется", "использовани", "вызов", "ссылк", "найти все"):
        return TASK_FIND_USAGES
    if has("readme", "readme.md"):
        return TASK_GENERATE_FILE
    return TASK_UNKNOWN


_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")


def extract_symbol(goal: str) -> str:
    """Извлекает искомый символ из цели (в бэктиках/кавычках или эвристикой)."""
    for pattern in (r"`([^`]+)`", r"«([^»]+)»", r"\"([^\"]+)\"", r"'([^']+)'"):
        match = re.search(pattern, goal or "")
        if match and match.group(1).strip():
            return match.group(1).strip()

    candidates = [
        token for token in _SYMBOL_RE.findall(goal or "")
        if token.lower() not in _STOPWORDS
    ]
    if not candidates:
        return ""
    # Предпочитаем идентификаторы с подчёркиванием/точкой или CamelCase.
    def rank(token: str) -> tuple[int, int]:
        special = 2 if ("_" in token or "." in token) else (1 if any(c.isupper() for c in token[1:]) else 0)
        return (special, len(token))

    return max(candidates, key=rank)


# ============================================================================
# 2. Модели данных
# ============================================================================


@dataclass
class FileAction:
    """Одно действие ассистента над файлами (вызов MCP-инструмента)."""

    tool: str
    arguments: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments, "summary": self.summary}


@dataclass
class FileFinding:
    """Находка: использование, нарушение правила и т.п."""

    kind: str
    message: str
    path: str = ""
    line: int = 0
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "severity": self.severity,
        }


@dataclass
class FileTaskResult:
    """Результат выполнения цели ассистентом."""

    goal: str
    task_type: str
    content: str = ""
    actions: list[FileAction] = field(default_factory=list)
    findings: list[FileFinding] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    applied: bool = False
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "task_type": self.task_type,
            "content": self.content,
            "actions": [action.to_dict() for action in self.actions],
            "findings": [finding.to_dict() for finding in self.findings],
            "artifacts": self.artifacts,
            "files_touched": self.files_touched,
            "applied": self.applied,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ============================================================================
# 3. Офлайн-клиент (стаб)
# ============================================================================


def _last_user_content(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    rest = text.split(start, 1)[1]
    if end and end in rest:
        return rest.split(end, 1)[0]
    return rest


class FileAssistantStubClient:
    """Детерминированный офлайн-генератор содержимого документов.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete(messages, …)``), но не делает сетевых вызовов: собирает markdown
    документ из структурированных фактов, переданных в промпте (маркер
    ``ФАКТЫ (JSON):``). Подходит для проверок без модели и сети.
    """

    def __init__(self, model_label: str = "file-assistant-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.0,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self.last_kind = ""

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        user = _last_user_content(messages)
        raw_facts = _between(user, FACTS_MARKER, GOAL_MARKER).strip()
        try:
            facts = json.loads(raw_facts) if raw_facts else {}
        except (ValueError, json.JSONDecodeError):
            facts = {}
        kind = str(facts.get("kind", ""))
        self.last_kind = kind
        content = render_document(kind, facts)
        return Completion(content=content, model=self.config["model"])


# ============================================================================
# 4. Рендеринг документов (офлайн-шаблоны)
# ============================================================================


def render_document(kind: str, facts: dict[str, Any]) -> str:
    """Собирает markdown-документ из структурированных фактов (без LLM)."""
    title = facts.get("title") or "Документ"
    lines = [f"# {title}", ""]
    if facts.get("intro"):
        lines += [str(facts["intro"]), ""]

    if kind == "catalog":
        lines += ["## MCP-серверы и инструменты", ""]
        for server in facts.get("servers", []):
            lines.append(f"### `{server.get('path', '?')}`")
            if server.get("docstring"):
                lines.append("")
                lines.append(str(server["docstring"]))
            tools = server.get("tools") or []
            if tools:
                lines.append("")
                lines.append("Инструменты: " + ", ".join(f"`{tool}`" for tool in tools))
            lines.append("")
    elif kind == "adr":
        lines += ["## Контекст", "", facts.get("context", "—"), "",
                  "## Решение", "", facts.get("decision", "—"), "",
                  "## Последствия", "", facts.get("consequences", "—"), ""]
    elif kind == "changelog":
        lines += ["## Изменения", ""]
        for entry in facts.get("entries", []):
            lines.append(f"- {entry}")
        lines.append("")
    elif kind == "module_doc":
        lines += ["## Модуль и символы", ""]
        for module in facts.get("modules", []):
            lines.append(f"### `{module.get('path', '?')}`")
            symbols = module.get("symbols") or []
            if symbols:
                lines.append("")
                lines.append("Символы: " + ", ".join(f"`{symbol}`" for symbol in symbols))
            lines.append("")
    else:
        lines += [str(facts.get("body", "")), ""]
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# 5. Инварианты Python-файлов (AST)
# ============================================================================


def _python_invariants(text: str) -> list[tuple[int, str]]:
    """Находит нарушения правил в Python-коде через ``ast``.

    Возвращает список ``(строка, сообщение)``. Разбор через ``ast`` не даёт
    ложных срабатываний на примерах кода внутри строковых литералов.
    """
    violations: list[tuple[int, str]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return violations
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            violations.append((node.lineno, "голый except без типа исключения"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "breakpoint":
            violations.append((node.lineno, "отладочный breakpoint() в коде"))
    return violations


# ============================================================================
# 6. Ассистент
# ============================================================================

class FileAssistant:
    """Ассистент файловых операций: цель → MCP-инструменты → результат."""

    def __init__(
        self,
        client: Any,
        fs_runtime: McpToolRuntime | None = None,
        workspace_root: str | Path = PROJECT_ROOT,
        apply: bool = False,
        max_chars: int = 8000,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.fs_runtime = fs_runtime
        self.workspace_root = Path(workspace_root)
        self.apply = bool(apply)
        self.max_chars = int(max_chars)
        self.temperature = float(temperature)
        self.system_prompt = system_prompt

    # --- низкоуровневый доступ к MCP ------------------------------------
    def _call_tool(self, name: str, arguments: dict | None = None) -> tuple[dict | None, str | None]:
        """Вызывает MCP-инструмент; возвращает (structured, error)."""
        if self.fs_runtime is None:
            return None, "MCP-сервер файлов не подключён"
        try:
            result: ToolResult = self.fs_runtime.call(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 — понятная ошибка MCP
            return None, f"{type(exc).__name__}: {exc}"
        if result.is_error:
            return None, result.content or "ошибка инструмента"
        return result.structured, None

    def _record(self, result: FileTaskResult, tool: str, arguments: dict, summary: str) -> None:
        result.actions.append(FileAction(tool=tool, arguments=arguments, summary=summary))

    def read_file(self, path: str, max_chars: int | None = None) -> dict[str, Any]:
        data, error = self._call_tool("read_file", {"path": path, "max_chars": max_chars or self.max_chars})
        return {"error": error} if error else (data or {})

    def search_files(self, query: str, path: str = "", glob: str = "*.py",
                     limit: int = 100, context: int = 0) -> dict[str, Any]:
        args = {"query": query, "path": path, "glob": glob, "limit": limit, "context": context}
        data, error = self._call_tool("search_files", args)
        return {"error": error} if error else (data or {})

    def analyze_file(self, path: str) -> dict[str, Any]:
        data, error = self._call_tool("analyze_file", {"path": path})
        return {"error": error} if error else (data or {})

    def list_files(self, path: str = "", pattern: str = "", limit: int = 200) -> dict[str, Any]:
        data, error = self._call_tool("list_files", {"path": path, "pattern": pattern, "limit": limit})
        return {"error": error} if error else (data or {})

    def get_diff(self, path: str = "", baseline_root: str = "", max_chars: int = 8000) -> dict[str, Any]:
        args = {"path": path, "baseline_root": baseline_root, "max_chars": max_chars}
        data, error = self._call_tool("make_diff", args)
        return {"error": error} if error else (data or {})

    def write_file(self, path: str, content: str, overwrite: bool = True) -> dict[str, Any]:
        args = {"path": path, "content": content, "overwrite": overwrite}
        data, error = self._call_tool("write_file", args)
        return {"error": error} if error else (data or {})

    # --- вспомогательные ------------------------------------------------
    def _read_text(self, path: str) -> str:
        """Читает файл целиком (для анализа/диффа, а не для контекста промпта)."""
        data = self.read_file(path, max_chars=FULL_READ_CHARS)
        return str(data.get("content", "")) if not data.get("error") else ""

    @staticmethod
    def _unified_diff(path: str, before: str, after: str) -> str:
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )

    def _stage_write(
        self, result: FileTaskResult, path: str, content: str
    ) -> dict[str, Any]:
        """Готовит запись: считает diff и (при ``apply``) записывает файл.

        Возвращает артефакт ``{path, status, diff}``; ``status`` равен
        ``added``/``modified`` при записи и ``dry-run`` без неё.
        """
        before = self._read_text(path)
        diff = self._unified_diff(path, before, content)
        status = "dry-run"
        if self.apply:
            written = self.write_file(path, content)
            if written.get("error"):
                result.error = written["error"]
                status = "error"
            else:
                status = "added" if written.get("created") else "modified"
            self._record(result, "write_file", {"path": path}, f"{status}: {path}")
        else:
            self._record(result, "write_file", {"path": path, "dry_run": True}, f"diff: {path}")
        artifact = {"path": path, "status": status, "diff": diff}
        result.artifacts.append(artifact)
        if path not in result.files_touched:
            result.files_touched.append(path)
        return artifact

    def _generate_document(self, kind: str, facts: dict[str, Any]) -> str:
        """Формирует документ: офлайн-шаблон или запрос к живому LLM."""
        if isinstance(self.client, FileAssistantStubClient):
            return render_document(kind, facts)
        messages = [
            {"role": "system", "content": self.system_prompt or FILE_ASSISTANT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{FACTS_MARKER}\n"
                    f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
                    f"{GOAL_MARKER}\n"
                    f"Сформируй markdown-документ вида «{kind}» на основе фактов."
                ),
            },
        ]
        completion = self.client.complete(messages, temperature=self.temperature)
        return completion.content or render_document(kind, facts)

    # --- задачи ---------------------------------------------------------
    def _task_find_usages(self, goal: str, symbol: str = "") -> FileTaskResult:
        result = FileTaskResult(goal=goal, task_type=TASK_FIND_USAGES)
        symbol = symbol or extract_symbol(goal)
        if not symbol:
            result.error = "не удалось определить искомый символ в цели"
            return result

        search = self.search_files(symbol, glob="*.py", limit=50, context=0)
        if search.get("error"):
            result.error = search["error"]
            return result
        self._record(result, "search_files", {"query": symbol, "glob": "*.py"},
                     f"найдено файлов: {search.get('total_files', 0)}")

        lines = [f"Использования `{symbol}`: {search.get('total_matches', 0)} совпадений "
                 f"в {search.get('total_files', 0)} файлах.", ""]
        for entry in search.get("files", []):
            path = entry.get("file", "")
            if path not in result.files_touched:
                result.files_touched.append(path)
            lines.append(f"- `{path}` — {entry.get('count', 0)} совпадений")
            for match in entry.get("matches", []):
                result.findings.append(FileFinding(
                    kind="usage", path=path, line=int(match.get("line", 0)),
                    message=str(match.get("text", "")).strip(),
                ))
            # Читаем контекст из найденного файла (ассистент сам решает, что читать).
            context = self.read_file(path, max_chars=4000)
            if not context.get("error"):
                self._record(result, "read_file", {"path": path},
                             f"контекст {path} ({context.get('total_lines', 0)} строк)")
                lines += ["", f"Фрагмент `{path}`:", "```python"]
                lines += str(context.get("content", "")).splitlines()[:40]
                lines.append("```")
            lines.append("")
        result.content = "\n".join(lines).rstrip() + "\n"
        return result

    def _task_generate_file(self, goal: str, kind: str = "catalog", target: str = "") -> FileTaskResult:
        result = FileTaskResult(goal=goal, task_type=TASK_GENERATE_FILE)
        kind = (kind or "catalog").strip().lower()
        if kind in ("changelog", "history"):
            target = target or "history/day34/artifacts/CHANGELOG.md"
            entries = self._collect_changelog_entries()
            facts = {
                "kind": "changelog",
                "title": "Changelog проекта AI Advent",
                "intro": "Изменения, сгенерированные ассистентом файловых операций (День 34).",
                "entries": entries,
            }
        elif kind in ("adr",):
            target = target or "history/day34/artifacts/adr-0001-file-assistant.md"
            facts = {
                "kind": "adr",
                "title": "ADR 0001: ассистент файловых операций",
                "context": "Проекту нужен ассистент, выполняющий реальные операции с файлами.",
                "decision": "Файловые операции вынесены в MCP-сервер fs_server, а ассистент "
                            "оркестрирует чтение, поиск, анализ и запись по цели.",
                "consequences": "Изменения воспроизводимы, запись по умолчанию выключена (dry-run).",
            }
        else:
            kind = "catalog"
            target = target or "history/day34/artifacts/mcp-catalog.md"
            servers = self._collect_mcp_servers(result)
            facts = {
                "kind": "catalog",
                "title": "Каталог MCP-серверов проекта",
                "intro": "Сгенерировано ассистентом файловых операций из исходников mcp/*.py.",
                "servers": servers,
            }

        source_count = len(result.files_touched)
        content = self._generate_document(kind, facts)
        self._stage_write(result, target, content)
        result.content = (
            f"Сгенерирован файл `{target}` (тип: {kind}). "
            f"Исходных файлов прочитано для контекста: {source_count}."
        )
        return result

    def _collect_mcp_servers(self, result: FileTaskResult) -> list[dict[str, Any]]:
        """Читает исходники mcp/*_server.py и собирает их инструменты."""
        listing = self.list_files(path="mcp", pattern="*_server.py", limit=50)
        servers: list[dict[str, Any]] = []
        for path in listing.get("files", []):
            analysis = self.analyze_file(path)
            text = self._read_text(path)
            tools = [match.group(1) or match.group(2) for match in _TOOL_DECORATOR_RE.finditer(text)]
            if path not in result.files_touched:
                result.files_touched.append(path)
            self._record(result, "read_file", {"path": path}, f"контекст {path}")
            servers.append({
                "path": path,
                "docstring": str(analysis.get("docstring", "")).split("\n")[0],
                "tools": tools,
            })
        return servers

    def _collect_changelog_entries(self) -> list[str]:
        """Собирает записи changelog из docstring'ов модулей agent/ и mcp/."""
        listing = self.list_files(path="", pattern="*.py", limit=200)
        entries: list[str] = []
        for path in listing.get("files", []):
            if not (path.startswith("agent/") or path.startswith("mcp/")):
                continue
            analysis = self.analyze_file(path)
            docstring = str(analysis.get("docstring", "")).strip()
            if docstring:
                first = docstring.split("\n")[0].strip()
                entries.append(f"`{path}` — {first}")
        return sorted(entries)

    def _task_update_docs(self, goal: str, doc_path: str = "", source_paths: list[str] | None = None) -> FileTaskResult:
        result = FileTaskResult(goal=goal, task_type=TASK_UPDATE_DOCS)
        doc_path = doc_path or "docs/architecture.md"
        sources = source_paths or ["mcp/fs_server.py", "agent/file_assistant.py"]

        modules: list[dict[str, Any]] = []
        for path in sources:
            analysis = self.analyze_file(path)
            if analysis.get("error"):
                continue
            symbols = [item["name"] for item in analysis.get("functions", [])]
            symbols += [item["name"] for item in analysis.get("classes", [])]
            modules.append({"path": path, "symbols": symbols[:12]})
            if path not in result.files_touched:
                result.files_touched.append(path)

        section = render_document("module_doc", {
            "title": "Файловые операции (День 34)",
            "intro": "Раздел сгенерирован ассистентом по исходному коду.",
            "modules": modules,
        })
        current = self._read_text(doc_path)
        begin, end = "<!-- day34:file-assistant -->", "<!-- /day34:file-assistant -->"
        block = f"{begin}\n{section}{end}\n"
        if begin in current and end in current:
            before = current.split(begin, 1)[0]
            after = current.split(end, 1)[1]
            updated = before + block + after
        else:
            updated = current.rstrip() + "\n\n" + block if current else block
        self._stage_write(result, doc_path, updated)
        result.content = (
            f"Обновлён `{doc_path}` по {len(modules)} исходным файлам: "
            f"{', '.join(module['path'] for module in modules)}."
        )
        return result

    def _task_check_invariants(self, goal: str) -> FileTaskResult:
        result = FileTaskResult(goal=goal, task_type=TASK_CHECK_INVARIANTS)
        listing = self.list_files(path="", pattern="*.py", limit=300)
        files = [path for path in listing.get("files", [])
                 if path.startswith("agent/") or path.startswith("mcp/")]
        checked = 0
        for path in files:
            text = self._read_text(path)
            if not text:
                continue
            checked += 1
            if path not in result.files_touched:
                result.files_touched.append(path)
            for line, message in _python_invariants(text):
                result.findings.append(FileFinding(
                    kind="invariant", severity="major", path=path, line=line, message=message,
                ))
            if path.startswith("mcp/") and path.endswith("_server.py") and "--check" not in text:
                result.findings.append(FileFinding(
                    kind="invariant", severity="minor", path=path,
                    message="у MCP-сервера нет режима --check",
                ))
            analysis = self.analyze_file(path)
            if not str(analysis.get("docstring", "")).strip():
                result.findings.append(FileFinding(
                    kind="invariant", severity="minor", path=path,
                    message="нет докстринга модуля",
                ))
        result.content = (
            f"Проверено файлов: {checked}. Нарушений: {len(result.findings)}."
        )
        return result

    def _task_prepare_diff(self, goal: str, baseline_root: str = "", path: str = "") -> FileTaskResult:
        result = FileTaskResult(goal=goal, task_type=TASK_PREPARE_DIFF)
        diff = self.get_diff(path=path, baseline_root=baseline_root, max_chars=self.max_chars)
        if diff.get("error"):
            result.error = diff["error"]
            return result
        self._record(result, "make_diff", {"path": path, "baseline_root": baseline_root},
                     f"изменённых файлов: {diff.get('changed', 0)}")
        result.files_touched = list(diff.get("files", []))
        result.artifacts.append({
            "path": path or "(вся область)",
            "status": "diff",
            "diff": diff.get("diff", ""),
        })
        result.content = (
            f"Изменённых файлов: {diff.get('changed', 0)}. "
            f"Режим: {diff.get('mode', 'git')}."
        )
        return result

    # --- публичный интерфейс --------------------------------------------
    def run(self, goal: str, *, task_type: str | None = None, apply: bool | None = None,
            symbol: str = "", kind: str = "", target: str = "", doc_path: str = "",
            source_paths: list[str] | None = None, baseline_root: str = "",
            path: str = "") -> FileTaskResult:
        """Выполняет цель ассистента, самостоятельно выбирая файлы и инструменты."""
        started = time.perf_counter()
        previous_apply = self.apply
        if apply is not None:
            self.apply = bool(apply)
        resolved_type = task_type or classify_goal(goal)
        try:
            if resolved_type == TASK_FIND_USAGES:
                result = self._task_find_usages(goal, symbol=symbol)
            elif resolved_type == TASK_GENERATE_FILE:
                result = self._task_generate_file(goal, kind=kind, target=target)
            elif resolved_type == TASK_UPDATE_DOCS:
                result = self._task_update_docs(goal, doc_path=doc_path, source_paths=source_paths)
            elif resolved_type == TASK_CHECK_INVARIANTS:
                result = self._task_check_invariants(goal)
            elif resolved_type == TASK_PREPARE_DIFF:
                result = self._task_prepare_diff(goal, baseline_root=baseline_root, path=path)
            else:
                result = FileTaskResult(
                    goal=goal, task_type=TASK_UNKNOWN,
                    content="Не удалось распознать цель. Уточните задачу (поиск использований, "
                            "генерация файла, обновление документации, проверка правил, diff).",
                )
        except Exception as exc:  # noqa: BLE001 — понятная ошибка выполнения
            result = FileTaskResult(goal=goal, task_type=resolved_type, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.apply = previous_apply

        result.applied = self.apply if apply is None else bool(apply)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result


# ============================================================================
# 7. Фабрика
# ============================================================================


def build_fs_runtime(config: dict | None = None, workspace_root: str | Path | None = None) -> McpToolRuntime | None:
    """Создаёт MCP-рантайм сервера файлов из конфигурации."""
    section = ((config or {}).get("file_assistant", {}) or {}).get("fs_server", {}) or {}
    if section.get("enabled") is False:
        return None
    command = section.get("command") or sys.executable
    args = section.get("args") or [str(DEFAULT_FS_SERVER)]
    cwd = section.get("cwd") or str(PROJECT_ROOT)
    root = workspace_root or section.get("workspace_root") or str(PROJECT_ROOT)
    return McpToolRuntime(
        command=command,
        args=args,
        cwd=cwd,
        env={"FS_ROOT": str(root)},
        timeout=float(section.get("timeout", 30)),
    )


def build_file_assistant(
    config: dict | None = None,
    offline: bool = False,
    workspace_root: str | Path = PROJECT_ROOT,
    apply: bool = False,
    max_chars: int = 8000,
    temperature: float = 0.0,
    with_fs_mcp: bool = True,
) -> FileAssistant:
    """Собирает ассистента файловых операций: MCP-рантайм + клиент."""
    config = config or {}
    root = Path(workspace_root) if workspace_root else PROJECT_ROOT
    if offline:
        client: Any = FileAssistantStubClient()
    else:
        client = LLMClient(config)
    runtime = build_fs_runtime(config, workspace_root=root) if with_fs_mcp else None
    return FileAssistant(
        client=client,
        fs_runtime=runtime,
        workspace_root=root,
        apply=apply,
        max_chars=max_chars,
        temperature=temperature,
    )


def _preview(text: str, limit: int = 200) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _demo() -> int:
    """Небольшая самопроверка модуля без сети (только чтение)."""
    assistant = build_file_assistant(config={}, offline=True, apply=False)
    result = assistant.run("найди, где используется `classify_goal`", symbol="classify_goal")
    print(f"Тип задачи: {result.task_type}; файлов: {len(result.files_touched)}; "
          f"находок: {len(result.findings)}")
    print(_preview(result.content, 600))
    return 0 if result.findings and not result.error else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
