"""День 31: ассистент разработчика — RAG по документации + контекст проекта через MCP.

Ассистент отвечает на вопросы о самом проекте, комбинируя два источника знаний:

1. **Документация** (``README`` + ``docs/``) — ищется в локальном индексе FAISS
   (см. :mod:`agent.project_docs`) и передаётся модели как контекст с указанием
   файла и раздела;
2. **Контекст проекта** — живые данные из git через MCP-сервер
   (``mcp/project_server.py``): текущая ветка, состояние, файлы, коммиты, diff.

Компоненты:

* :class:`DevAssistant` — связывает ретривер, MCP-рантайм и LLM-клиент;
* :func:`build_dev_assistant` — фабрика (индекс, MCP, клиент, офлайн-стаб);
* :class:`DevAssistantStubClient` — детерминированный офлайн-клиент для проверок;
* :func:`build_dev_messages` — сборка промпта из документации и контекста проекта.

Запуск CLI: ``python3 chat_dev.py`` (см. :mod:`agent.dev_assistant_cli`).
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import Completion, LLMClient
from .mcp_tools import McpToolRuntime, ToolResult
from .project_docs import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATASET,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_docs_context,
    build_documents_index,
    load_project_retriever,
)
from .rag import RagRetriever, RagSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_SERVER = PROJECT_ROOT / "mcp" / "project_server.py"

DOCS_CONTEXT_MARKER = "КОНТЕКСТ ДОКУМЕНТАЦИИ ПРОЕКТА:"
PROJECT_CONTEXT_MARKER = "КОНТЕКСТ ПРОЕКТА (ДАННЫЕ MCP):"
QUESTION_MARKER = "ВОПРОС ПОЛЬЗОВАТЕЛЯ:"

DEV_ASSISTANT_SYSTEM_PROMPT = """Ты — ассистент разработчика по проекту AI Advent (курс по LLM и агентам).
Твоя задача — помогать разбираться в структуре, архитектуре и API этого проекта.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленной документации и контекста проекта.
2. Ссылайся на источники по номерам, например [Источник 1], и указывай файл и раздел.
3. Используй данные MCP (текущая git-ветка, файлы, коммиты, diff), когда они уместны.
4. Если в документации нет ответа — честно скажи об этом и подскажи, где искать.
5. Отвечай по-русски, структурированно и по делу."""


# ============================================================================
# 1. Офлайн-клиент (стаб)
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


def _parse_doc_sources(context: str) -> list[tuple[int, str, str, str, str]]:
    """Разбирает блоки источников документации: (индекс, документ, файл, раздел, текст)."""
    parsed: list[tuple[int, str, str, str, str]] = []
    pattern = re.compile(
        r"\[Источник (\d+)\]\s*Документ:\s*«([^»]*)»,\s*файл:\s*([^,]*),\s*"
        r"раздел:\s*«([^»]*)»[^\n]*\n(.*?)(?=\n\n\[Источник |\Z)",
        re.S,
    )
    for match in pattern.finditer(context):
        parsed.append(
            (
                int(match.group(1)),
                match.group(2).strip(),
                match.group(3).strip(),
                match.group(4).strip(),
                " ".join(match.group(5).split()),
            )
        )
    return parsed


def _parse_branch(project_context: str) -> str:
    match = re.search(r"ветка:\s*([^\n(]+?)\s*(?:\(коммит|\n|$)", project_context)
    if match:
        return match.group(1).strip()
    match = re.search(r'"branch"\s*:\s*"([^"]+)"', project_context)
    return match.group(1) if match else ""


class DevAssistantStubClient:
    """Детерминированный офлайн-клиент ассистента разработчика.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete(messages, ...)``), но не делает сетевых вызовов: извлекает из
    пользовательского сообщения найденные разделы документации и данные проекта
    и собирает экстрактивный ответ. Подходит для проверок без модели и сети.
    """

    def __init__(self, model_label: str = "dev-assistant-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.2,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self.last_sources_used = 0

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        user = _last_user_content(messages)
        docs_context = _between(user, DOCS_CONTEXT_MARKER, PROJECT_CONTEXT_MARKER)
        project_context = _between(user, PROJECT_CONTEXT_MARKER, QUESTION_MARKER)
        question = _between(user, QUESTION_MARKER, "Ответь").strip() or _between(
            user, QUESTION_MARKER, ""
        ).strip()
        sources = _parse_doc_sources(docs_context)
        branch = _parse_branch(project_context)
        self.last_sources_used = min(len(sources), 3)

        parts: list[str] = [f"Вопрос: {question or '(не задан)'}"]
        if sources:
            for idx, title, path, section, body in sources[:3]:
                excerpt = body[:320].strip()
                parts.append(f"[Источник {idx}] {path} — «{section}»: {excerpt}")
        else:
            parts.append(
                "В документации проекта не нашлось подходящих разделов — "
                "уточните вопрос или пересоберите индекс (`/reindex`)."
            )
        if branch:
            parts.append(f"Текущая git-ветка проекта: {branch}.")
        return Completion(content="\n\n".join(parts), model=self.config["model"])


# ============================================================================
# 2. Модели данных
# ============================================================================


@dataclass
class DevAnswer:
    """Ответ ассистента разработчика."""

    question: str
    content: str
    sources: list[RagSource] = field(default_factory=list)
    project_context: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "project_context": self.project_context,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ============================================================================
# 3. Промпт и контекст проекта
# ============================================================================


def render_project_context(context: dict[str, Any]) -> str:
    """Человекочитаемое представление контекста проекта для модели."""
    if not context or context.get("available") is False:
        return "Данные MCP о проекте недоступны."

    lines: list[str] = []
    branch = context.get("branch") or {}
    if branch.get("branch"):
        commit = branch.get("commit", "")
        suffix = f" (коммит {commit})" if commit else ""
        lines.append(f"Текущая git-ветка: {branch['branch']}{suffix}.")
    status = context.get("status") or {}
    if status:
        lines.append(
            f"Состояние дерева: изменено {status.get('modifiedTotal', 0)}, "
            f"новых {status.get('untrackedTotal', 0)}, "
            f"в индексе {status.get('stagedTotal', 0)}."
        )
    files = context.get("files") or {}
    if files.get("files"):
        lines.append(
            f"Файлы (всего {files.get('total', 0)}, показаны первые "
            f"{files.get('returned', 0)}): " + ", ".join(files["files"][:30]) + "."
        )
    commits = context.get("commits") or {}
    if commits.get("commits"):
        rendered = "; ".join(
            f"{item.get('short', '')} {item.get('subject', '')}" for item in commits["commits"][:5]
        )
        lines.append(f"Последние коммиты: {rendered}.")
    diff = context.get("diff") or {}
    if diff.get("diff"):
        lines.append("Diff (фрагмент):\n" + diff["diff"][:1200])
    if context.get("errors"):
        lines.append("Ошибки MCP: " + "; ".join(context["errors"]) + ".")
    return "\n".join(lines) if lines else "Данные MCP о проекте недоступны."


def build_dev_messages(
    question: str,
    sources: list[RagSource],
    project_context: dict[str, Any],
    max_chars: int = 6000,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Формирует сообщения для ассистента: документация + контекст проекта."""
    docs_context = build_docs_context(sources, max_chars=max_chars)
    user_content = (
        f"{DOCS_CONTEXT_MARKER}\n"
        f"------------------------\n"
        f"{docs_context}\n"
        f"------------------------\n\n"
        f"{PROJECT_CONTEXT_MARKER}\n"
        f"{render_project_context(project_context)}\n\n"
        f"{QUESTION_MARKER}\n"
        f"{question}\n\n"
        f"Ответь на вопрос о проекте, опираясь на документацию и контекст проекта, "
        f"со ссылками на источники ([Источник N])."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or DEV_ASSISTANT_SYSTEM_PROMPT}
    ]
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_project_runtime(config: dict | None = None) -> McpToolRuntime | None:
    """Создаёт MCP-рантайм сервера проекта из конфигурации."""
    section = ((config or {}).get("dev_assistant", {}) or {}).get("project_server", {}) or {}
    if section.get("enabled") is False:
        return None
    command = section.get("command") or sys.executable
    args = section.get("args") or [str(DEFAULT_PROJECT_SERVER)]
    cwd = section.get("cwd") or str(PROJECT_ROOT)
    return McpToolRuntime(
        command=command,
        args=args,
        cwd=cwd,
        timeout=float(section.get("timeout", 30)),
    )


def _wants_any(question: str, markers: tuple[str, ...]) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in markers)


FILE_MARKERS = ("файл", "структур", "каталог", "директор", "модул", "где леж", "дерево проекта", "список файлов")
DIFF_MARKERS = ("diff", "диф", "изменени", "правк", "не закоммич")
COMMIT_MARKERS = ("коммит", "commit", "история изменени", "последние изменени")
STATUS_MARKERS = ("статус", "состояние", "ветк", "branch", "изменено")

# Приоритет документов при реранкинге (кураторские доки выше автогенерации).
SOURCE_PRIOR: dict[str, float] = {
    "docs/architecture.md": 0.10,
    "docs/api.md": 0.10,
    "docs/data-schemas.md": 0.08,
    "docs/development.md": 0.08,
    "docs/README.md": 0.04,
    "README.md": 0.02,
    "recipe_chat_service/README.md": 0.01,
    "docs/modules.md": -0.03,
}


# ============================================================================
# 4. Ассистент
# ============================================================================


class DevAssistant:
    """Ассистент разработчика: RAG по документации + контекст проекта через MCP."""

    def __init__(
        self,
        client: Any,
        retriever: RagRetriever,
        project_runtime: McpToolRuntime | None = None,
        top_k: int = 5,
        max_context_chars: int = 6000,
        temperature: float = 0.2,
        min_score: float = 0.0,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.project_runtime = project_runtime
        self.top_k = max(1, int(top_k))
        self.max_context_chars = int(max_context_chars)
        self.temperature = float(temperature)
        self.min_score = float(min_score)
        self.system_prompt = system_prompt

    @property
    def total_vectors(self) -> int:
        return self.retriever.total_vectors

    def retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]:
        k = top_k or self.top_k
        candidates = self.retriever.retrieve(question, top_k=max(k * 3, k))
        candidates = [s for s in candidates if s.score >= self.min_score]
        return self._rerank(question, candidates)[:k]

    @staticmethod
    def _rerank(question: str, sources: list[RagSource]) -> list[RagSource]:
        """Лёгкий реранкинг: векторное сходство + лексическое пересечение + приоритет.

        Кураторские документы (``docs/*.md``, README) чуть выше сгенерированного
        справочника ``docs/modules.md``, чтобы концептуальные вопросы попадали в
        архитектурные разделы, а не в длинный список сигнатур.
        """
        terms = {
            token
            for token in re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ]+", question.lower())
            if len(token) > 2
        }
        denominator = len(terms) or 1
        scored: list[tuple[float, RagSource]] = []
        for src in sources:
            haystack = f"{src.text} {src.section} {src.title}".lower()
            overlap = sum(1 for term in terms if term in haystack) / denominator
            adjusted = src.score + 0.12 * overlap + SOURCE_PRIOR.get(src.url, 0.0)
            scored.append((adjusted, src))
        scored.sort(key=lambda item: item[0], reverse=True)
        for adjusted, src in scored:
            src.score = adjusted
        return [src for _, src in scored]

    def _call_tool(self, name: str, arguments: dict | None = None) -> tuple[dict | None, str | None]:
        """Вызывает MCP-инструмент; возвращает (structured, error)."""
        if self.project_runtime is None:
            return None, "MCP-сервер проекта не подключён"
        try:
            result: ToolResult = self.project_runtime.call(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 — понятная ошибка MCP
            return None, f"{type(exc).__name__}: {exc}"
        if result.is_error:
            return None, result.content or "ошибка инструмента"
        return result.structured, None

    def collect_project_context(self, question: str) -> dict[str, Any]:
        """Собирает живой контекст проекта через MCP-инструменты."""
        if self.project_runtime is None:
            return {"available": False}

        context: dict[str, Any] = {"available": True, "errors": []}

        branch, error = self._call_tool("get_current_branch")
        if error:
            context["errors"].append(f"get_current_branch: {error}")
        else:
            context["branch"] = branch

        if _wants_any(question, STATUS_MARKERS) or _wants_any(question, DIFF_MARKERS):
            status, error = self._call_tool("get_repo_status", {"limit": 30})
            if error:
                context["errors"].append(f"get_repo_status: {error}")
            else:
                context["status"] = status

        if _wants_any(question, FILE_MARKERS):
            files, error = self._call_tool("list_project_files", {"limit": 60})
            if error:
                context["errors"].append(f"list_project_files: {error}")
            else:
                context["files"] = files

        if _wants_any(question, COMMIT_MARKERS):
            commits, error = self._call_tool("get_recent_commits", {"limit": 5})
            if error:
                context["errors"].append(f"get_recent_commits: {error}")
            else:
                context["commits"] = commits

        if _wants_any(question, DIFF_MARKERS):
            diff, error = self._call_tool("get_diff", {"max_chars": 2000})
            if error:
                context["errors"].append(f"get_diff: {error}")
            else:
                context["diff"] = diff

        return context

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> DevAnswer:
        """Отвечает на вопрос о проекте по документации и контексту."""
        started = time.perf_counter()
        try:
            sources = self.retrieve(question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 — ошибка поиска не должна ронять CLI
            return DevAnswer(question=question, content="", error=str(exc))

        project_context = self.collect_project_context(question)
        messages = build_dev_messages(
            question,
            sources,
            project_context,
            max_chars=self.max_context_chars,
            history=history,
            system_prompt=self.system_prompt,
        )
        try:
            completion = self.client.complete(messages, temperature=self.temperature)
        except Exception as exc:  # noqa: BLE001 — понятная ошибка модели
            return DevAnswer(
                question=question,
                content="",
                sources=sources,
                project_context=project_context,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency = (time.perf_counter() - started) * 1000
        return DevAnswer(
            question=question,
            content=completion.content,
            sources=sources,
            project_context=project_context,
            latency_ms=latency,
        )

    def answer_help(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> DevAnswer:
        """Ответ на команду ``/help`` — вопрос о проекте."""
        return self.ask(question, history=history)

    def get_branch(self) -> dict[str, Any]:
        """Текущая git-ветка (для команды ``/branch``)."""
        data, error = self._call_tool("get_current_branch")
        if error:
            return {"error": error}
        return data or {}

    def list_files(self, path: str = "", limit: int = 100) -> dict[str, Any]:
        data, error = self._call_tool("list_project_files", {"path": path, "limit": limit})
        return {"error": error} if error else (data or {})

    def get_diff(self, path: str = "", max_chars: int = 4000) -> dict[str, Any]:
        data, error = self._call_tool("get_diff", {"path": path, "max_chars": max_chars})
        return {"error": error} if error else (data or {})


# ============================================================================
# 5. Фабрика
# ============================================================================


def build_dev_assistant(
    config: dict | None = None,
    offline: bool = False,
    index_dir: str = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    temperature: float = 0.2,
    max_context_chars: int = 6000,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    build_index_if_missing: bool = True,
    with_project_mcp: bool = True,
) -> DevAssistant:
    """Собирает ассистента разработчика: индекс, MCP-рантайм, LLM-клиент."""
    config = config or {}
    try:
        retriever = load_project_retriever(
            index_dir=index_dir, dataset=dataset, strategy=strategy, top_k=top_k, min_score=min_score
        )
    except FileNotFoundError:
        if not build_index_if_missing:
            raise
        index, _ = build_documents_index(
            index_dir=index_dir,
            dataset=dataset,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        retriever = RagRetriever(index=index, top_k=top_k, min_score=min_score)

    if offline:
        client: Any = DevAssistantStubClient()
    else:
        client = LLMClient(config)

    runtime = build_project_runtime(config) if with_project_mcp else None
    return DevAssistant(
        client=client,
        retriever=retriever,
        project_runtime=runtime,
        top_k=top_k,
        max_context_chars=max_context_chars,
        temperature=temperature,
        min_score=min_score,
    )


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _demo() -> int:
    """Небольшая самопроверка модуля без сети."""
    config: dict = {}
    assistant = build_dev_assistant(config=config, offline=True)
    print(f"Индекс документации: {assistant.total_vectors} чанков")
    question = "Как устроена структура проекта и где лежит MCP-сервер проекта?"
    answer = assistant.ask(question)
    print(f"\nВопрос: {question}")
    print(f"Ответ:\n{answer.content}\n")
    print(f"Источников: {len(answer.sources)}")
    for src in answer.sources[:3]:
        print(f"  - {src.url} — «{src.section}» ({src.score:.3f})")
    branch = (answer.project_context.get("branch") or {}).get("branch")
    print(f"Ветка из MCP: {branch}")
    return 0 if answer.content and branch else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
