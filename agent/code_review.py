"""День 32: ассистент автоматического ревью кода.

Ассистент анализирует pull request и выдаёт текст ревью с тремя разделами:
потенциальные **баги**, **архитектурные проблемы** и **рекомендации**.

Источники данных:

1. **Diff и изменённые файлы** — из git через MCP-сервер
   (``mcp/review_server.py``): ``get_pr_diff``, ``list_changed_files``,
   ``get_file_content``, ``get_pr_metadata``;
2. **RAG по документации** — индекс Дня 31 (:mod:`agent.project_docs`);
3. **RAG по коду** — индекс исходников (:mod:`agent.code_index`).

Компоненты:

* :class:`ReviewFinding` / :class:`CodeReview` — модель результата ревью;
* :class:`CodeReviewAssistant` — связывает RAG, MCP-рантайм и LLM-клиент;
* :func:`analyze_diff` — детерминированные эвристики по diff (офлайн);
* :class:`ReviewStubClient` — офлайн-клиент для проверок без модели и сети;
* :func:`build_review_messages` — сборка промпта из PR-контекста и RAG;
* :func:`parse_findings` — разбор ответа модели (JSON или markdown).

Запуск ревью: ``python3 review_pr.py`` (см. :mod:`agent.review_cli`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .code_index import (
    DEFAULT_CODE_DATASET,
    DEFAULT_CODE_INDEX_DIR,
    DEFAULT_CODE_STRATEGY,
    build_code_context,
    build_code_index,
    load_code_retriever,
)
from .dev_assistant import SOURCE_PRIOR as DOC_PRIOR
from .llm_client import Completion, LLMClient
from .mcp_tools import McpToolRuntime, ToolResult
from .project_docs import (
    DEFAULT_DATASET as DEFAULT_DOCS_DATASET,
    DEFAULT_INDEX_DIR as DEFAULT_DOCS_INDEX_DIR,
    DEFAULT_STRATEGY as DEFAULT_DOCS_STRATEGY,
    build_docs_context,
    build_documents_index,
    load_project_retriever,
)
from .rag import RagRetriever, RagSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_SERVER = PROJECT_ROOT / "mcp" / "review_server.py"

# --- Категории и серьёзность находок ---------------------------------------
CATEGORY_BUG = "bug"
CATEGORY_ARCHITECTURE = "architecture"
CATEGORY_RECOMMENDATION = "recommendation"
CATEGORIES = (CATEGORY_BUG, CATEGORY_ARCHITECTURE, CATEGORY_RECOMMENDATION)

SEVERITY_CRITICAL = "critical"
SEVERITY_MAJOR = "major"
SEVERITY_MINOR = "minor"
SEVERITY_INFO = "info"
SEVERITIES = (SEVERITY_CRITICAL, SEVERITY_MAJOR, SEVERITY_MINOR, SEVERITY_INFO)

CATEGORY_TITLES = {
    CATEGORY_BUG: "🐞 Потенциальные баги",
    CATEGORY_ARCHITECTURE: "🏗️ Архитектурные проблемы",
    CATEGORY_RECOMMENDATION: "✅ Рекомендации",
}
SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_MAJOR: 1, SEVERITY_MINOR: 2, SEVERITY_INFO: 3}

# --- Маркеры промпта -------------------------------------------------------
PR_METADATA_MARKER = "МЕТАДАННЫЕ PR:"
CHANGED_FILES_MARKER = "ИЗМЕНЁННЫЕ ФАЙЛЫ:"
DIFF_MARKER = "DIFF PR:"
DOCS_CONTEXT_MARKER = "КОНТЕКСТ ДОКУМЕНТАЦИИ (RAG):"
CODE_CONTEXT_MARKER = "КОНТЕКСТ КОДА (RAG):"
REVIEW_REQUEST_MARKER = "ЗАДАНИЕ РЕВЬЮ:"

CODE_REVIEW_SYSTEM_PROMPT = """Ты — старший инженер и ревьюер проекта AI Advent (курс по LLM и агентам).
Твоя задача — провести ревью pull request и найти проблемы до мержа.

Правила:
1. Опирайся ТОЛЬКО на diff, изменённые файлы и предоставленный RAG-контекст
   (документация проекта и код соседних модулей).
2. Верни три обязательных раздела:
   - «Потенциальные баги» — логические ошибки, необработанные исключения,
     небезопасный ввод, ресурсные утечки, гонки;
   - «Архитектурные проблемы» — нарушение границ слоёв, дублирование,
     связность модулей, обход инвариантов проекта;
   - «Рекомендации» — улучшения, тесты, документация, читаемость.
3. У каждой находки укажи серьёзность: critical, major, minor или info.
4. Ссылайся на источники по номерам: [Источник N] для документации и [Код N]
   для кода. Указывай файл и, если известно, строку.
5. Не выдумывай проблемы: если раздел пуст, честно напиши «Ничего критичного не найдено».
6. В конце добавь машиночитаемый JSON-блок (```json … ```) с полем «findings» —
   список объектов {category, severity, title, detail, file, line}.
7. Отвечай по-русски, структурированно и по делу."""


# ============================================================================
# 1. Модель результата
# ============================================================================


@dataclass
class ReviewFinding:
    """Одна находка ревью."""

    category: str
    severity: str
    title: str
    detail: str = ""
    file: str = ""
    line: int | None = None
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "file": self.file,
            "line": self.line,
            "sources": list(self.sources),
        }


@dataclass
class CodeReview:
    """Результат автоматического ревью pull request."""

    base: str
    head: str
    summary: str
    findings: list[ReviewFinding] = field(default_factory=list)
    changed_files: list[dict] = field(default_factory=list)
    diff_stats: dict[str, Any] = field(default_factory=dict)
    sources: list[RagSource] = field(default_factory=list)
    docs_used: int = 0
    code_used: int = 0
    pr: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    def by_category(self, category: str) -> list[ReviewFinding]:
        return [item for item in self.findings if item.category == category]

    def counts(self) -> dict[str, int]:
        result = {category: 0 for category in CATEGORIES}
        for item in self.findings:
            result[item.category] = result.get(item.category, 0) + 1
        return result

    def max_severity(self) -> str | None:
        if not self.findings:
            return None
        return min(self.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9)).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "head": self.head,
            "summary": self.summary,
            "findings": [item.to_dict() for item in self.findings],
            "counts": self.counts(),
            "changed_files": self.changed_files,
            "diff_stats": self.diff_stats,
            "sources": [source.to_dict() for source in self.sources],
            "docs_used": self.docs_used,
            "code_used": self.code_used,
            "pr": self.pr,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }

    def to_markdown(self) -> str:
        """Человекочитаемое ревью (для комментария к PR и артефакта)."""
        stats = self.diff_stats or {}
        files = stats.get("count", len(self.changed_files))
        adds = stats.get("addTotal", 0)
        dels = stats.get("delTotal", 0)

        lines = [
            "## 🤖 Автоматическое ревью PR",
            "",
            f"- **Диапазон:** `{self.base}` → `{self.head}`",
            f"- **Изменённых файлов:** {files} (+{adds} / −{dels})",
            f"- **Источников RAG:** документация — {self.docs_used}, код — {self.code_used}",
        ]
        if self.pr.get("available") and self.pr.get("number"):
            title = self.pr.get("title") or ""
            lines.append(f"- **PR #{self.pr['number']}:** {title}")
        lines.append("")

        if self.error:
            lines += ["⚠️ **Ошибка ревью:** " + self.error, ""]

        lines += ["### Резюме", "", self.summary or "—", ""]

        for category in CATEGORIES:
            items = self.by_category(category)
            lines += [f"### {CATEGORY_TITLES[category]}", ""]
            if not items:
                lines += ["Ничего критичного не найдено.", ""]
                continue
            for item in sorted(items, key=lambda f: SEVERITY_ORDER.get(f.severity, 9)):
                location = ""
                if item.file:
                    location = f" — `{item.file}`"
                    if item.line:
                        location += f":{item.line}"
                lines.append(f"#### `[{item.severity}]` {item.title}{location}")
                if item.detail:
                    lines.append(item.detail)
                if item.sources:
                    lines.append("Источники: " + ", ".join(item.sources))
                lines.append("")

        if self.sources:
            lines += ["### 📚 Использованные источники", ""]
            for idx, src in enumerate(self.sources, start=1):
                lines.append(f"{idx}. `{src.url}` — «{src.section}» (сходство {src.score:.3f})")
            lines.append("")

        lines += [
            "<sub>Сформировано ассистентом ревью (День 32): RAG по документации и коду.</sub>",
        ]
        return "\n".join(lines)


# ============================================================================
# 2. Эвристики по diff (офлайн-режим)
# ============================================================================


@dataclass
class _AddedLine:
    file: str
    number: int
    text: str


def parse_added_lines(diff: str) -> list[_AddedLine]:
    """Разбирает unified diff и возвращает только добавленные строки.

    Каждая запись помечена файлом и номером строки в новой версии. Строки-
    заголовки (``+++``, ``---``) и старый код не попадают в результат: эвристики
    анализируют именно то, что приносит PR.
    """
    added: list[_AddedLine] = []
    current_file = ""
    new_line = 0
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            name = raw[4:].strip()
            if name.startswith("b/"):
                name = name[2:]
            current_file = "" if name == "/dev/null" else name
            continue
        if raw.startswith("--- "):
            continue
        hunk = hunk_re.match(raw)
        if hunk:
            new_line = int(hunk.group(1))
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.append(_AddedLine(file=current_file, number=new_line, text=raw[1:]))
            new_line += 1
            continue
        if raw.startswith("-") and not raw.startswith("---"):
            continue
        if raw.startswith("\\"):
            continue
        new_line += 1
    return added


_SECRET_RE = re.compile(
    r"(?i)\b(api_key|apikey|token|password|passwd|secret)\b\s*[:=]\s*[\"'][^\"']{8,}[\"']"
)
_MUTABLE_DEFAULT_RE = re.compile(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\})")
_PUBLIC_DEF_RE = re.compile(r"^(?:async\s+)?def\s+[a-zA-Z_]\w*\s*\(")
_ANNOTATED_DEF_RE = re.compile(r"^(?:async\s+)?def\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*->")

# Модули, где «низкоуровневые» зависимости допустимы.
_LLM_ALLOWED = {"agent/llm_client.py", "agent/local_llm.py", "agent/web_chat.py"}
_FAISS_ALLOWED = {"agent/indexing.py"}
_SUBPROCESS_ALLOWED_PREFIX = ("mcp/",)
_SUBPROCESS_ALLOWED_FILES = {"agent/local_llm.py"}
_PRINT_SKIP_SUFFIX = ("_cli.py", "_scenarios.py")


def _is_library_code(path: str) -> bool:
    if path.startswith("mcp/"):
        return True
    return path.startswith("agent/") and not path.endswith(_PRINT_SKIP_SUFFIX)


def analyze_diff(diff: str) -> list[ReviewFinding]:
    """Детерминированные эвристики по добавленным строкам diff.

    Используется офлайн-клиентом (:class:`ReviewStubClient`) и как «страховка»,
    если ответ модели пуст. Возвращает находки по трём категориям.
    """
    findings: list[ReviewFinding] = []
    lines = parse_added_lines(diff)
    by_file: dict[str, list[_AddedLine]] = {}
    for item in lines:
        by_file.setdefault(item.file, []).append(item)

    for item in lines:
        text = item.text
        stripped = text.strip()
        path = item.file
        location = {"file": path, "line": item.number}

        # --- Баги ---------------------------------------------------------
        if re.match(r"^except\s*:\s*$", stripped):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_MAJOR,
                    title="Перехват всех исключений без указания типа",
                    detail="`except:` поймает в том числе KeyboardInterrupt/SystemExit и скроет ошибки. Укажите конкретный тип исключения.",
                    **location,
                )
            )
        if re.match(r"^except\s+Exception\s*:\s*$", stripped):
            siblings = [
                by_file[path][i + 1]
                for i, line in enumerate(by_file[path])
                if line is item and i + 1 < len(by_file[path])
            ]
            if siblings and siblings[0].text.strip() == "pass":
                findings.append(
                    ReviewFinding(
                        category=CATEGORY_BUG,
                        severity=SEVERITY_MAJOR,
                        title="Исключение проглатывается (`except Exception: pass`)",
                        detail="Ошибка молча теряется. Логируйте её или обрабатывайте явно.",
                        **location,
                    )
                )
        if re.search(r"\b(eval|exec)\s*\(", text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_CRITICAL,
                    title="Использование eval/exec",
                    detail="Динамическое исполнение кода небезопасно и трудно тестируется. Замените на явный разбор/структуру данных.",
                    **location,
                )
            )
        if "shell=True" in text:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_MAJOR,
                    title="subprocess с `shell=True`",
                    detail="При подстановке пользовательских данных это command injection. Передавайте аргументы списком.",
                    **location,
                )
            )
        if re.search(r"\bos\.system\s*\(", text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_MAJOR,
                    title="Вызов `os.system`",
                    detail="`os.system` запускает команду через shell. Используйте `subprocess.run([...])` без shell.",
                    **location,
                )
            )
        if _SECRET_RE.search(text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_CRITICAL,
                    title="Захардкоженный секрет",
                    detail="Похоже на ключ/пароль прямо в коде. Вынесите в конфиг/переменные окружения и отзовите утёкшее значение.",
                    **location,
                )
            )
        if _MUTABLE_DEFAULT_RE.search(text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_MAJOR,
                    title="Изменяемое значение по умолчанию",
                    detail="Изменяемый аргумент по умолчанию (`=[]`/`={}`) переиспользуется между вызовами. Используйте `None` и создавайте значение внутри.",
                    **location,
                )
            )
        if re.match(r"^assert\s+", stripped) and not path.startswith(("tests/", "recipe_chat_service/scenarios/")):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_BUG,
                    severity=SEVERITY_INFO,
                    title="`assert` в рабочем коде",
                    detail="`assert` отключается флагом `-O`. Для проверок ввода используйте явные исключения.",
                    **location,
                )
            )

        # --- Архитектура --------------------------------------------------
        if re.search(r"(?i)\b(import\s+openai|from\s+openai|import\s+anthropic|from\s+anthropic)\b", text):
            if path not in _LLM_ALLOWED:
                findings.append(
                    ReviewFinding(
                        category=CATEGORY_ARCHITECTURE,
                        severity=SEVERITY_MAJOR,
                        title="Прямая зависимость от LLM-SDK вне клиента",
                        detail="Обращения к модели должны идти через `agent/llm_client.py` — иначе ломаются офлайн-режим, учёт токенов и моки.",
                        **location,
                    )
                )
        if re.search(r"\b(import\s+faiss|from\s+faiss)\b", text) and path not in _FAISS_ALLOWED:
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ARCHITECTURE,
                    severity=SEVERITY_MAJOR,
                    title="Прямое использование faiss вне слоя индексации",
                    detail="Работа с FAISS инкапсулирована в `agent/indexing.py`. Импорт в обход слоя нарушает границы модулей.",
                    **location,
                )
            )
        if re.search(r"\b(import\s+subprocess|from\s+subprocess|subprocess\.)", text) and (
            path not in _SUBPROCESS_ALLOWED_FILES
            and not path.startswith(_SUBPROCESS_ALLOWED_PREFIX)
        ):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ARCHITECTURE,
                    severity=SEVERITY_MAJOR,
                    title="subprocess вне MCP-слоя",
                    detail="Запуск внешних процессов сосредоточен в `mcp/`-серверах. В ядре используйте MCP-инструменты.",
                    **location,
                )
            )
        if re.match(r"^(from|import)\s+recipe_chat_service\b", stripped) and path.startswith("agent/"):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ARCHITECTURE,
                    severity=SEVERITY_MAJOR,
                    title="Обратная зависимость ядра от сервиса Дня 30",
                    detail="`recipe_chat_service` — самодостаточный проект; ядро `agent/` не должно зависеть от него. Вынесите общее в `agent/`.",
                    **location,
                )
            )
        if re.search(r"\bprint\s*\(", text) and _is_library_code(path):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ARCHITECTURE,
                    severity=SEVERITY_MINOR,
                    title="`print` в библиотечном коде",
                    detail="В библиотечных модулях используйте `logging`, а вывод оставьте CLI-слою.",
                    **location,
                )
            )
        if _PUBLIC_DEF_RE.match(text) and not _ANNOTATED_DEF_RE.match(text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_ARCHITECTURE,
                    severity=SEVERITY_MINOR,
                    title="Новая функция без аннотации возвращаемого типа",
                    detail="Добавьте аннотацию `-> …`: это контракт модуля и подсказка для ревьюеров.",
                    **location,
                )
            )

        # --- Рекомендации -------------------------------------------------
        if re.search(r"\b(TODO|FIXME)\b", text):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_RECOMMENDATION,
                    severity=SEVERITY_MINOR,
                    title="Незавершённая работа (TODO/FIXME)",
                    detail="Замените комментарий на issue или завершите изменение до мержа.",
                    **location,
                )
            )

    # Новые модули: напоминание про документацию и индексы.
    for path in by_file:
        if path.startswith(("agent/", "mcp/")) and path.endswith(".py"):
            findings.append(
                ReviewFinding(
                    category=CATEGORY_RECOMMENDATION,
                    severity=SEVERITY_INFO,
                    title="Обновить документацию и пересобрать индексы",
                    detail=f"После изменений в `{path}` прогоните `ruff check`, офлайн-сценарии и пересоберите индексы документации/кода для RAG.",
                    file=path,
                )
            )

    return findings


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


class ReviewStubClient:
    """Детерминированный офлайн-клиент ревью.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete(messages, …)``), но не делает сетевых вызовов: извлекает diff из
    промпта, прогоняет эвристики :func:`analyze_diff` и собирает ревью в формате,
    который разбирает :func:`parse_findings`.
    """

    def __init__(self, model_label: str = "review-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.0,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self.last_findings = 0

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        user = _last_user_content(messages)
        diff = _between(user, DIFF_MARKER, DOCS_CONTEXT_MARKER)
        docs_context = _between(user, DOCS_CONTEXT_MARKER, CODE_CONTEXT_MARKER)
        code_context = _between(user, CODE_CONTEXT_MARKER, REVIEW_REQUEST_MARKER)

        findings = analyze_diff(diff)
        self.last_findings = len(findings)

        sources: list[str] = []
        if code_context.strip():
            sources.append("[Код 1]")
        if docs_context.strip():
            sources.append("[Источник 1]")
        for finding in findings:
            finding.sources = list(sources[:2])

        counts = {category: 0 for category in CATEGORIES}
        for finding in findings:
            counts[finding.category] += 1

        summary = (
            f"Проанализировано изменений: {len(parse_added_lines(diff))} строк кода. "
            f"Найдено: баги — {counts[CATEGORY_BUG]}, "
            f"архитектура — {counts[CATEGORY_ARCHITECTURE]}, "
            f"рекомендации — {counts[CATEGORY_RECOMMENDATION]}."
        )
        if not findings:
            summary = "Существенных проблем в добавленных строках не обнаружено."

        payload = {
            "summary": summary,
            "findings": [
                {
                    "category": finding.category,
                    "severity": finding.severity,
                    "title": finding.title,
                    "detail": finding.detail,
                    "file": finding.file,
                    "line": finding.line,
                    "sources": list(finding.sources),
                }
                for finding in findings
            ],
        }
        content = (
            "### Резюме\n"
            f"{summary}\n\n"
            "Ниже — структурированный результат эвристического анализа.\n\n"
            "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"
        )
        return Completion(content=content, model=self.config["model"])


# ============================================================================
# 4. Разбор ответа модели
# ============================================================================


def _normalize_category(value: str) -> str:
    lowered = (value or "").lower()
    if "баг" in lowered or "bug" in lowered:
        return CATEGORY_BUG
    if "архитект" in lowered or "arch" in lowered:
        return CATEGORY_ARCHITECTURE
    if "рекоменд" in lowered or "recommend" in lowered or "совет" in lowered:
        return CATEGORY_RECOMMENDATION
    return CATEGORY_RECOMMENDATION


def _normalize_severity(value: str) -> str:
    lowered = (value or "").lower()
    if any(marker in lowered for marker in ("critical", "критич", "блок")):
        return SEVERITY_CRITICAL
    if any(marker in lowered for marker in ("major", "серьёз", "важн", "высок")):
        return SEVERITY_MAJOR
    if any(marker in lowered for marker in ("minor", "незнач", "мелк", "низк")):
        return SEVERITY_MINOR
    return SEVERITY_INFO


def _finding_from_dict(data: dict) -> ReviewFinding:
    line = data.get("line")
    try:
        line = int(line) if line not in (None, "", "None") else None
    except (TypeError, ValueError):
        line = None
    sources = data.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]
    return ReviewFinding(
        category=_normalize_category(str(data.get("category", ""))),
        severity=_normalize_severity(str(data.get("severity", "") or data.get("важность", ""))),
        title=str(data.get("title") or data.get("заголовок") or "Находка").strip(),
        detail=str(data.get("detail") or data.get("описание") or "").strip(),
        file=str(data.get("file") or data.get("path") or "").strip(),
        line=line,
        sources=[str(item) for item in sources],
    )


def _extract_json_block(text: str) -> dict | None:
    for match in re.finditer(r"```json\s*(.*?)```", text, re.S):
        try:
            parsed = json.loads(match.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    # Фолбэк: первый {...} с полем findings.
    match = re.search(r"\{.*\"findings\".*\}", text, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def _parse_markdown_findings(text: str) -> tuple[str, list[ReviewFinding]]:
    """Разбирает markdown-ревью по разделам «баги / архитектура / рекомендации»."""
    findings: list[ReviewFinding] = []
    summary_lines: list[str] = []
    current_category: str | None = None
    current: ReviewFinding | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.detail = current.detail.strip()
            findings.append(current)
            current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            flush()
            title = heading.group(1)
            lowered = title.lower()
            matched = None
            if "баг" in lowered or "bug" in lowered:
                matched = CATEGORY_BUG
            elif "архитект" in lowered or "arch" in lowered:
                matched = CATEGORY_ARCHITECTURE
            elif "рекоменд" in lowered or "recommend" in lowered:
                matched = CATEGORY_RECOMMENDATION
            elif "резюме" in lowered or "summary" in lowered:
                matched = None
                current_category = None
                continue
            if matched:
                current_category = matched
                continue
            if current_category is None:
                continue
            # Подзаголовок внутри раздела — это находка.
            item = re.match(r"^[`\[]*\[?(critical|major|minor|info|критич\w*|серьёзн\w*|важн\w*|мелк\w*)\]?[`\]]*\s*(.*)$", title, re.I)
            severity = SEVERITY_INFO
            rest = title
            if item:
                severity = _normalize_severity(item.group(1))
                rest = item.group(2)
            flush()
            current = ReviewFinding(
                category=current_category,
                severity=severity,
                title=rest.strip() or "Находка",
            )
            continue
        if line.startswith(("- ", "* ")):
            if current_category is None:
                summary_lines.append(line[2:].strip())
                continue
            text_item = line[2:].strip()
            item = re.match(r"^\[(critical|major|minor|info|критич\w*|серьёзн\w*|важн\w*|мелк\w*)\]\s*(.*)$", text_item, re.I)
            if item:
                flush()
                current = ReviewFinding(
                    category=current_category,
                    severity=_normalize_severity(item.group(1)),
                    title=item.group(2).strip() or "Находка",
                )
            elif current is not None:
                current.detail = (current.detail + "\n" + text_item).strip()
            else:
                current = ReviewFinding(
                    category=current_category,
                    severity=SEVERITY_INFO,
                    title=text_item or "Находка",
                )
            continue
        if current is not None:
            current.detail = (current.detail + "\n" + line).strip()
        elif current_category is None:
            summary_lines.append(line)

    flush()
    summary = " ".join(summary_lines).strip()
    return summary, findings


def parse_findings(answer_text: str) -> tuple[str, list[ReviewFinding]]:
    """Разбирает ответ модели на резюме и список находок.

    Сначала пробует машиночитаемый JSON-блок, затем — markdown-разделы. Разбор
    толерантный: непонятные заголовки и уровни серьёзности нормализуются.
    """
    text = answer_text or ""
    payload = _extract_json_block(text)
    if payload and isinstance(payload.get("findings"), list):
        findings = [
            _finding_from_dict(item)
            for item in payload["findings"]
            if isinstance(item, dict)
        ]
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            summary, _ = _parse_markdown_findings(text)
        return summary, findings

    return _parse_markdown_findings(text)


# ============================================================================
# 5. Промпт
# ============================================================================


def render_pr_metadata(pr: dict[str, Any] | None) -> str:
    """Человекочитаемые метаданные PR для промпта."""
    pr = pr or {}
    if not pr or pr.get("available") is False:
        return "Метаданные PR недоступны (локальный запуск)."
    lines = []
    if pr.get("number"):
        lines.append(f"PR #{pr['number']}: {pr.get('title', '')}".strip())
    if pr.get("author"):
        lines.append(f"Автор: {pr['author']}")
    if pr.get("baseRef") or pr.get("headRef"):
        lines.append(f"Ветки: {pr.get('baseRef', '?')} → {pr.get('headRef', '?')}")
    if pr.get("url"):
        lines.append(f"Ссылка: {pr['url']}")
    if pr.get("body"):
        lines.append("Описание: " + " ".join(str(pr["body"]).split())[:600])
    return "\n".join(lines) if lines else "Метаданные PR недоступны."


def render_changed_files(changed: list[dict]) -> str:
    """Список изменённых файлов для промпта."""
    if not changed:
        return "Изменённые файлы не определены."
    lines = []
    for item in changed[:100]:
        status = item.get("status", "M")
        adds = item.get("additions", 0)
        dels = item.get("deletions", 0)
        lines.append(f"- [{status}] {item.get('path', '?')} (+{adds}/-{dels})")
    return "\n".join(lines)


def build_review_messages(
    pr: dict[str, Any],
    changed_files: list[dict],
    diff: str,
    docs_sources: list[RagSource],
    code_sources: list[RagSource],
    max_chars: int = 8000,
    max_diff_chars: int = 20000,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Формирует сообщения для ассистента ревью: PR + RAG (доки и код)."""
    docs_context = build_docs_context(docs_sources, max_chars=max_chars)
    code_context = build_code_context(code_sources, max_chars=max_chars)
    diff_text = (diff or "").strip()
    if len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + "\n… (diff усечён)"

    user_content = (
        f"{PR_METADATA_MARKER}\n"
        f"{render_pr_metadata(pr)}\n\n"
        f"{CHANGED_FILES_MARKER}\n"
        f"{render_changed_files(changed_files)}\n\n"
        f"{DIFF_MARKER}\n"
        f"{diff_text or '(изменений нет)'}\n\n"
        f"{DOCS_CONTEXT_MARKER}\n"
        f"{docs_context}\n\n"
        f"{CODE_CONTEXT_MARKER}\n"
        f"{code_context}\n\n"
        f"{REVIEW_REQUEST_MARKER}\n"
        f"Проведи ревью этого pull request. Найди потенциальные баги, "
        f"архитектурные проблемы и дай рекомендации. Ссылайся на источники "
        f"([Источник N] / [Код N]) и добавь JSON-блок с находками."
    )
    return [
        {"role": "system", "content": system_prompt or CODE_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# 6. Ассистент ревью
# ============================================================================


def _query_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ_]+", text.lower())
        if len(token) > 2
    ]


class CodeReviewAssistant:
    """Ассистент ревью: PR-контекст (MCP) + RAG (документация + код) + LLM."""

    def __init__(
        self,
        client: Any,
        docs_retriever: RagRetriever,
        code_retriever: RagRetriever,
        review_runtime: McpToolRuntime | None = None,
        top_k: int = 5,
        max_context_chars: int = 8000,
        max_diff_chars: int = 20000,
        temperature: float = 0.0,
        min_score: float = 0.0,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.docs_retriever = docs_retriever
        self.code_retriever = code_retriever
        self.review_runtime = review_runtime
        self.top_k = max(1, int(top_k))
        self.max_context_chars = int(max_context_chars)
        self.max_diff_chars = int(max_diff_chars)
        self.temperature = float(temperature)
        self.min_score = float(min_score)
        self.system_prompt = system_prompt

    @property
    def total_vectors(self) -> int:
        return self.docs_retriever.total_vectors + self.code_retriever.total_vectors

    def _call_tool(self, name: str, arguments: dict | None = None) -> tuple[dict | None, str | None]:
        """Вызывает MCP-инструмент; возвращает (structured, error)."""
        if self.review_runtime is None:
            return None, "MCP-сервер ревью не подключён"
        try:
            result: ToolResult = self.review_runtime.call(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 — понятная ошибка MCP
            return None, f"{type(exc).__name__}: {exc}"
        if result.is_error:
            return None, result.content or "ошибка инструмента"
        return result.structured, None

    def collect_pr_context(self, base: str, head: str = "") -> dict[str, Any]:
        """Собирает контекст PR: diff, изменённые файлы, метаданные."""
        context: dict[str, Any] = {
            "available": self.review_runtime is not None,
            "diff": "",
            "changed_files": [],
            "pr": {},
            "errors": [],
        }
        if self.review_runtime is None:
            return context

        diff, error = self._call_tool(
            "get_pr_diff", {"base": base, "head": head, "max_chars": self.max_diff_chars}
        )
        if error:
            context["errors"].append(f"get_pr_diff: {error}")
        else:
            context["diff"] = (diff or {}).get("diff", "")
            context["diff_stats"] = {
                key: (diff or {}).get(key) for key in ("base", "head", "chars", "truncated")
            }

        files, error = self._call_tool("list_changed_files", {"base": base, "head": head})
        if error:
            context["errors"].append(f"list_changed_files: {error}")
        else:
            files = files or {}
            context["changed_files"] = files.get("files", [])
            context["diff_stats"] = {
                **(context.get("diff_stats") or {}),
                "count": files.get("count", 0),
                "addTotal": files.get("addTotal", 0),
                "delTotal": files.get("delTotal", 0),
            }

        pr, error = self._call_tool("get_pr_metadata", {})
        if error:
            context["errors"].append(f"get_pr_metadata: {error}")
        else:
            context["pr"] = pr or {}

        return context

    def _queries(self, changed_files: list[dict], diff: str) -> list[str]:
        queries: list[str] = []
        for item in changed_files[:10]:
            path = str(item.get("path", ""))
            if not path:
                continue
            stem = Path(path).stem.replace("_", " ")
            queries.append(f"{path} {stem}")
        # Общие запросы: архитектура/границы слоёв и правила проекта.
        queries.append("архитектура проекта подсистемы границы слоёв модулей")
        identifiers = [token for token in _query_terms(diff) if "_" in token or len(token) > 6]
        if identifiers:
            queries.append(" ".join(dict.fromkeys(identifiers))[:400])
        return queries

    @staticmethod
    def _rerank(
        sources: list[RagSource],
        queries: list[str],
        changed_paths: set[str],
        prior: dict[str, float] | None = None,
        is_code: bool = False,
    ) -> list[RagSource]:
        terms = {token for query in queries for token in _query_terms(query)}
        denominator = len(terms) or 1
        dirs = {str(Path(path).parent) for path in changed_paths}
        scored: list[tuple[float, RagSource]] = []
        for src in sources:
            haystack = f"{src.text} {src.section} {src.title}".lower()
            overlap = sum(1 for term in terms if term in haystack) / denominator
            adjusted = src.score + 0.12 * overlap
            adjusted += (prior or {}).get(src.url, 0.0)
            if is_code:
                if str(Path(src.url).parent) in dirs and src.url not in changed_paths:
                    adjusted += 0.05
                if src.url in changed_paths:
                    adjusted -= 0.03
            scored.append((adjusted, src))
        scored.sort(key=lambda item: item[0], reverse=True)
        for adjusted, src in scored:
            src.score = adjusted
        return [src for _, src in scored]

    def retrieve_context(
        self, changed_files: list[dict], diff: str, top_k: int | None = None
    ) -> tuple[list[RagSource], list[RagSource]]:
        """Извлекает релевантную документацию и код по изменённым файлам."""
        k = top_k or self.top_k
        queries = self._queries(changed_files, diff)
        changed_paths = {str(item.get("path", "")) for item in changed_files}

        docs_map: dict[str, RagSource] = {}
        code_map: dict[str, RagSource] = {}
        for query in queries:
            for src in self.docs_retriever.retrieve(query, top_k=k * 2):
                if src.score >= self.min_score:
                    docs_map.setdefault(src.chunk_id, src)
            for src in self.code_retriever.retrieve(query, top_k=k * 2):
                if src.score >= self.min_score:
                    code_map.setdefault(src.chunk_id, src)

        docs = self._rerank(list(docs_map.values()), queries, changed_paths, prior=DOC_PRIOR)
        code = self._rerank(
            list(code_map.values()), queries, changed_paths, is_code=True
        )
        return docs[:k], code[:k]

    def review(self, base: str = "HEAD", head: str = "", pr: dict | None = None) -> CodeReview:
        """Проводит ревью PR и возвращает структурированный результат."""
        started = time.perf_counter()
        context = self.collect_pr_context(base, head)
        diff = context.get("diff", "")
        changed_files = context.get("changed_files", [])
        pr_meta = pr if pr is not None else context.get("pr", {})

        docs_sources, code_sources = self.retrieve_context(changed_files, diff)
        messages = build_review_messages(
            pr_meta,
            changed_files,
            diff,
            docs_sources,
            code_sources,
            max_chars=self.max_context_chars,
            max_diff_chars=self.max_diff_chars,
            system_prompt=self.system_prompt,
        )

        try:
            completion = self.client.complete(messages, temperature=self.temperature)
        except Exception as exc:  # noqa: BLE001 — понятная ошибка модели
            return CodeReview(
                base=base,
                head=context.get("diff_stats", {}).get("head", head) or head,
                summary="",
                changed_files=changed_files,
                diff_stats=context.get("diff_stats", {}),
                sources=docs_sources + code_sources,
                docs_used=len(docs_sources),
                code_used=len(code_sources),
                pr=pr_meta or {},
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        summary, findings = parse_findings(completion.content)
        if not findings and diff:
            # Страховка: если модель вернула пустой список — добираем эвристиками.
            findings = analyze_diff(diff)

        latency = (time.perf_counter() - started) * 1000
        return CodeReview(
            base=base,
            head=(context.get("diff_stats", {}).get("head") or head or ""),
            summary=summary or "Ревью завершено.",
            findings=findings,
            changed_files=changed_files,
            diff_stats=context.get("diff_stats", {}),
            sources=docs_sources + code_sources,
            docs_used=len(docs_sources),
            code_used=len(code_sources),
            pr=pr_meta or {},
            latency_ms=latency,
        )


# ============================================================================
# 7. Фабрики
# ============================================================================


def build_review_config(config: dict | None = None, env: dict | None = None) -> dict:
    """Собирает конфиг LLM-клиента, подмешивая переменные окружения.

    В CI нет файла ``config.json`` с ключами — значения берутся из окружения:
    ``LLM_BASE_URL``/``OPENAI_BASE_URL``, ``LLM_API_KEY``/``OPENAI_API_KEY``,
    ``LLM_MODEL``.
    """
    env = env if env is not None else os.environ
    merged = dict(config or {})
    base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL")
    api_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
    model = env.get("LLM_MODEL")
    if base_url:
        merged["base_url"] = base_url
    if api_key:
        merged["api_key"] = api_key
    if model:
        merged["model"] = model
    return merged


def has_llm_credentials(config: dict) -> bool:
    """Есть ли в конфиге всё для обращения к LLM API."""
    return bool(config.get("base_url")) and bool(config.get("api_key")) and bool(config.get("model"))


def build_review_runtime(config: dict | None = None) -> McpToolRuntime | None:
    """Создаёт MCP-рантайм сервера ревью из конфигурации."""
    section = ((config or {}).get("code_review", {}) or {}).get("review_server", {}) or {}
    if section.get("enabled") is False:
        return None
    command = section.get("command") or sys.executable
    args = section.get("args") or [str(DEFAULT_REVIEW_SERVER)]
    cwd = section.get("cwd") or str(PROJECT_ROOT)
    env = {"REVIEW_REPO_ROOT": section.get("repo_root", str(PROJECT_ROOT))}
    return McpToolRuntime(
        command=command,
        args=args,
        cwd=cwd,
        env=env,
        timeout=float(section.get("timeout", 30)),
    )


def build_review_assistant(
    config: dict | None = None,
    offline: bool = False,
    docs_index_dir: str = DEFAULT_DOCS_INDEX_DIR,
    docs_dataset: str = DEFAULT_DOCS_DATASET,
    docs_strategy: str = DEFAULT_DOCS_STRATEGY,
    code_index_dir: str = DEFAULT_CODE_INDEX_DIR,
    code_dataset: str = DEFAULT_CODE_DATASET,
    code_strategy: str = DEFAULT_CODE_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    temperature: float = 0.0,
    max_context_chars: int = 8000,
    max_diff_chars: int = 20000,
    build_index_if_missing: bool = True,
    with_review_mcp: bool = True,
) -> CodeReviewAssistant:
    """Собирает ассистента ревью: оба индекса, MCP-рантайм, LLM-клиент."""
    config = config or {}
    doc_retriever = _load_or_build_docs(
        docs_index_dir, docs_dataset, docs_strategy, top_k, min_score, build_index_if_missing
    )
    code_retriever = _load_or_build_code(
        code_index_dir, code_dataset, code_strategy, top_k, min_score, build_index_if_missing
    )

    if offline:
        client: Any = ReviewStubClient()
    else:
        client = LLMClient(build_review_config(config))

    runtime = build_review_runtime(config) if with_review_mcp else None
    return CodeReviewAssistant(
        client=client,
        docs_retriever=doc_retriever,
        code_retriever=code_retriever,
        review_runtime=runtime,
        top_k=top_k,
        max_context_chars=max_context_chars,
        max_diff_chars=max_diff_chars,
        temperature=temperature,
        min_score=min_score,
    )


def _load_or_build_docs(
    index_dir: str,
    dataset: str,
    strategy: str,
    top_k: int,
    min_score: float,
    build_if_missing: bool,
) -> RagRetriever:
    try:
        return load_project_retriever(
            index_dir=index_dir, dataset=dataset, strategy=strategy, top_k=top_k, min_score=min_score
        )
    except FileNotFoundError:
        if not build_if_missing:
            raise
        index, _ = build_documents_index(index_dir=index_dir, dataset=dataset, strategy=strategy)
        return RagRetriever(index=index, top_k=top_k, min_score=min_score)


def _load_or_build_code(
    index_dir: str,
    dataset: str,
    strategy: str,
    top_k: int,
    min_score: float,
    build_if_missing: bool,
) -> RagRetriever:
    try:
        return load_code_retriever(
            index_dir=index_dir, dataset=dataset, strategy=strategy, top_k=top_k, min_score=min_score
        )
    except FileNotFoundError:
        if not build_if_missing:
            raise
        index, _ = build_code_index(index_dir=index_dir, dataset=dataset, strategy=strategy)
        return RagRetriever(index=index, top_k=top_k, min_score=min_score)


def _preview(text: str, limit: int = 200) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _demo() -> int:
    """Небольшая самопроверка модуля без сети."""
    assistant = build_review_assistant(config={}, offline=True)
    review = assistant.review(base="HEAD~1", head="HEAD")
    print(review.to_markdown())
    print(f"\nЧанков в индексах: {assistant.total_vectors}")
    return 0 if review.summary else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
