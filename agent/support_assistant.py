"""День 33: ассистент поддержки пользователей.

Ассистент отвечает на вопросы о продукте, комбинируя два источника знаний:

1. **FAQ о продукте** — ищется в локальном индексе FAISS
   (см. :mod:`agent.support_faq`) и передаётся модели как контекст с указанием
   вопроса и раздела;
2. **Контекст пользователя и тикета** — живые данные из JSON-CRM через
   MCP-сервер (``mcp/support_server.py``): профиль пользователя, статус и версия
   тикета, переписка.

Компоненты:

* :class:`SupportAnswer` — результат ответа;
* :class:`SupportAssistant` — связывает FAQ-ретривер, MCP-рантайм и LLM-клиент;
* :class:`SupportStubClient` — детерминированный офлайн-клиент для проверок;
* :func:`build_support_messages` — сборка промпта (FAQ + тикет + вопрос);
* :func:`resolve_support_ticket` / :meth:`SupportAssistant.resolve_ticket` —
   привязка вопроса к тикету.

Запуск CLI: ``python3 chat_support.py`` (см. :mod:`agent.support_cli`).
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_client import Completion, LLMClient
from .mcp_tools import McpToolRuntime, ToolResult
from .rag import RagRetriever, RagSource
from .support_faq import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATASET,
    DEFAULT_FAQ_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_faq_context,
    build_faq_index,
    load_faq_retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUPPORT_SERVER = PROJECT_ROOT / "mcp" / "support_server.py"
DEFAULT_SUPPORT_DATA_DIR = PROJECT_ROOT / "mcp" / "support_data"

FAQ_CONTEXT_MARKER = "КОНТЕКСТ FAQ (RAG):"
TICKET_CONTEXT_MARKER = "КОНТЕКСТ ПОЛЬЗОВАТЕЛЯ И ТИКЕТА (MCP):"
QUESTION_MARKER = "ВОПРОС ПОЛЬЗОВАТЕЛЯ:"

_TICKET_ID_RE = re.compile(r"\bT-\d{3,}\b", re.IGNORECASE)


SUPPORT_SYSTEM_PROMPT = """Ты — ассистент поддержки пользователей продукта Recipe Chat Service
(веб-сервис поиска рецептов).
Твоя задача — помогать пользователям решать проблемы и отвечать на вопросы о продукте.

Правила:
1. Отвечай на основе предоставленного FAQ ([Источник N]) и данных тикета/пользователя.
2. Обязательно учитывай контекст: статус тикета, версию продукта, план и статус
   аккаунта, содержание переписки. Связывай свой ответ с конкретной ситуацией
   пользователя, а не только с общей инструкцией.
3. Ссылайся на источники FAQ по номерам ([Источник N]).
4. Если в FAQ нет ответа — честно скажи об этом и предложи эскалацию в поддержку.
   НЕ придумывай факты, тарифы, сроки и гарантии.
5. Будь доброжелателен и краток; дай конкретные шаги решения.
6. В конце добавь машиночитаемый JSON-блок (```json … ```) с полями:
   "answer_kind" (faq|ticket|mixed|unknown), "ticket_id", "escalate" (true/false),
   "steps" (список коротких шагов).
7. Отвечай по-русски."""


# ============================================================================
# 1. Модель результата
# ============================================================================


@dataclass
class SupportAnswer:
    """Ответ ассистента поддержки."""

    question: str
    content: str
    faq_sources: list[RagSource] = field(default_factory=list)
    ticket_context: dict[str, Any] = field(default_factory=dict)
    user_context: dict[str, Any] = field(default_factory=dict)
    ticket_id: str | None = None
    user_id: int | None = None
    answer_kind: str = "unknown"
    escalate: bool = False
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "content": self.content,
            "faq_sources": [src.to_dict() for src in self.faq_sources],
            "ticket_context": self.ticket_context,
            "user_context": self.user_context,
            "ticket_id": self.ticket_id,
            "user_id": self.user_id,
            "answer_kind": self.answer_kind,
            "escalate": self.escalate,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ============================================================================
# 2. Контекст тикета
# ============================================================================


def render_ticket_context(context: dict[str, Any]) -> str:
    """Человекочитаемый контекст тикета и пользователя для промпта.

    Если MCP-инструмент ``get_ticket_context`` уже вернул готовую строку
    ``context`` — используется она. Иначе блок собирается из полей.
    """
    if not context or context.get("error"):
        return "Данные тикета/пользователя недоступны."
    prepared = context.get("context")
    if isinstance(prepared, str) and prepared.strip():
        return prepared.strip()

    lines: list[str] = []
    ticket_id = context.get("ticket_id") or context.get("id")
    if ticket_id:
        lines.append(f"Тикет {ticket_id}: «{context.get('subject', '')}»")
    if context.get("status"):
        lines.append(
            f"Статус: {context.get('status')}; категория: {context.get('category', '?')}; "
            f"приоритет: {context.get('priority', '?')}; "
            f"версия продукта: {context.get('version', '?')}"
        )
    user = context.get("user") or context.get("user_context") or {}
    if user:
        lines.append(
            f"Пользователь: {user.get('name', '')} ({user.get('username', '')}, "
            f"e-mail: {user.get('email', '')}); план: {user.get('plan', '?')}; "
            f"статус аккаунта: {user.get('status', '?')}; версия: {user.get('version', '?')}"
        )
    messages = context.get("messages") or []
    if messages:
        lines.append("Переписка:")
        for message in messages:
            lines.append(
                f"  - [{message.get('created_at', '')}] {message.get('author', '?')} "
                f"({message.get('role', '')}): {message.get('text', '')}"
            )
    return "\n".join(lines) if lines else "Данные тикета/пользователя недоступны."


# ============================================================================
# 3. Промпт
# ============================================================================


def build_support_messages(
    question: str,
    faq_sources: list[RagSource],
    ticket_context: dict[str, Any],
    max_chars: int = 6000,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Формирует сообщения ассистента: FAQ + контекст тикета + вопрос."""
    faq_context = build_faq_context(faq_sources, max_chars=max_chars)
    user_content = (
        f"{FAQ_CONTEXT_MARKER}\n"
        f"------------------------\n"
        f"{faq_context}\n"
        f"------------------------\n\n"
        f"{TICKET_CONTEXT_MARKER}\n"
        f"{render_ticket_context(ticket_context)}\n\n"
        f"{QUESTION_MARKER}\n"
        f"{question}\n\n"
        f"Ответь пользователю с учётом FAQ и данных его тикета. Ссылайся на источники "
        f"([Источник N]) и заверши JSON-блоком с полями answer_kind/ticket_id/escalate/steps."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or SUPPORT_SYSTEM_PROMPT}
    ]
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_content})
    return messages


# ============================================================================
# 4. Разбор ответа
# ============================================================================


_VALID_KINDS = ("faq", "ticket", "mixed", "unknown")


def parse_answer_meta(text: str) -> dict[str, Any]:
    """Разбирает машиночитаемый JSON-блок ответа (answer_kind/ticket_id/escalate)."""
    meta: dict[str, Any] = {}
    for match in re.finditer(r"```json\s*(.*?)```", text or "", re.S):
        try:
            parsed = json.loads(match.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            meta = parsed
            break
    if not meta:
        match = re.search(r"\{[^{}]*\"answer_kind\"[^{}]*\}", text or "", re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    meta = parsed
            except (ValueError, json.JSONDecodeError):
                meta = {}
    kind = str(meta.get("answer_kind", "")).strip().lower()
    if kind not in _VALID_KINDS:
        kind = ""
    return {
        "answer_kind": kind,
        "ticket_id": str(meta.get("ticket_id") or "").strip() or None,
        "escalate": bool(meta.get("escalate")),
        "steps": [str(step) for step in meta.get("steps") or []],
    }


# ============================================================================
# 5. Офлайн-клиент (стаб)
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


def _parse_faq_sources(context: str) -> list[tuple[int, str, str, str, float]]:
    """Разбирает блоки FAQ: (индекс, вопрос, раздел, текст ответа, сходство)."""
    parsed: list[tuple[int, str, str, str, float]] = []
    pattern = re.compile(
        r"\[Источник (\d+)\]\s*Вопрос FAQ:\s*«([^»]*)»,\s*раздел:\s*«([^»]*)»"
        r"[^\n(]*\(сходство:\s*([\d.]+)\)[^\n]*\n"
        r"(.*?)(?=\n\n\[Источник |\Z)",
        re.S,
    )
    for match in pattern.finditer(context):
        try:
            score = float(match.group(4))
        except ValueError:
            score = 0.0
        parsed.append(
            (
                int(match.group(1)),
                match.group(2).strip(),
                match.group(3).strip(),
                " ".join(match.group(5).split()),
                score,
            )
        )
    return parsed


def _relevant_faq(
    faq: list[tuple[int, str, str, str, float]], keep: int = 3, relative: float = 0.5
) -> list[tuple[int, str, str, str, float]]:
    """Оставляет верхние карточки и отбрасывает явно нерелевантные по сходству."""
    if not faq:
        return []
    best = max(item[4] for item in faq)
    floor = best * relative if best > 0 else 0.0
    kept = [item for item in faq if item[4] >= floor] or faq[:1]
    return kept[:keep]


def _parse_ticket_context(context: str) -> dict[str, Any]:
    """Извлекает ключевые поля из блока контекста тикета."""
    info: dict[str, Any] = {}
    match = re.search(r"Тикет\s+(\S+):\s*«([^»]*)»", context)
    if match:
        info["ticket_id"] = match.group(1)
        info["subject"] = match.group(2)
    match = re.search(r"Статус:\s*([^;\n]+);", context)
    if match:
        info["status"] = match.group(1).strip()
    match = re.search(r"версия продукта:\s*([^;\n]+)", context)
    if match:
        info["version"] = match.group(1).strip()
    match = re.search(r"план:\s*([^;]+);", context)
    if match:
        info["plan"] = match.group(1).strip()
    user_messages = re.findall(r"-\s*\[[^\]]*\]\s*\S+\s*\(user\):\s*(.+)", context)
    if user_messages:
        info["last_user_message"] = user_messages[-1].strip()
    info["has_context"] = bool(info)
    return info


class SupportStubClient:
    """Детерминированный офлайн-клиент ассистента поддержки.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` +
    ``complete(messages, …)``), но не делает сетевых вызовов: собирает
    экстрактивный ответ из найденных карточек FAQ и полей тикета. Подходит для
    проверок без модели и сети.
    """

    def __init__(self, model_label: str = "support-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.1,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self.last_kind = "unknown"

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        user = _last_user_content(messages)
        faq_context = _between(user, FAQ_CONTEXT_MARKER, TICKET_CONTEXT_MARKER)
        ticket_context = _between(user, TICKET_CONTEXT_MARKER, QUESTION_MARKER)
        question = _between(user, QUESTION_MARKER, "Ответь").strip()

        faq = _parse_faq_sources(faq_context)
        ticket = _parse_ticket_context(ticket_context)

        parts: list[str] = [f"Вопрос: {question or '(не задан)'}"]
        answer_kind = "unknown"
        escalate = False
        steps: list[str] = []

        if faq:
            answer_kind = "faq"
            parts.append("По базе знаний (FAQ):")
            for idx, title, section, body, _score in _relevant_faq(faq):
                excerpt = body[:400].strip()
                parts.append(f"[Источник {idx}] {title} (раздел: {section}): {excerpt}")
            steps.append("Проверьте базовые требования (логин/пароль/сессия) по статье FAQ.")

        if ticket.get("has_context"):
            answer_kind = "mixed" if faq else "ticket"
            summary = f"Данные тикета {ticket.get('ticket_id', '')}"
            if ticket.get("subject"):
                summary += f" — «{ticket['subject']}»"
            if ticket.get("status"):
                summary += f"; статус: {ticket['status']}"
            if ticket.get("version"):
                summary += f"; версия продукта: {ticket['version']}"
            if ticket.get("plan"):
                summary += f"; план: {ticket['plan']}"
            parts.append(summary + ".")
            if ticket.get("last_user_message"):
                parts.append(f"Последнее сообщение пользователя: {ticket['last_user_message']}")
            steps.append("Учтите статус и версию из тикета при проверке проблемы.")

        if not faq and not ticket.get("has_context"):
            answer_kind = "unknown"
            escalate = True
            parts.append(
                "В базе знаний (FAQ) нет ответа на этот вопрос, и тикет не найден. "
                "Проверять детали по данным тикета нечем."
            )
            steps.append("Передайте обращение в поддержку (эскалация).")
        elif not faq:
            steps.append("Ответ дан по данным тикета; общих статей FAQ не найдено.")

        payload = {
            "answer_kind": answer_kind,
            "ticket_id": ticket.get("ticket_id"),
            "escalate": escalate,
            "steps": steps,
        }
        self.last_kind = answer_kind
        content = "\n\n".join(parts) + "\n\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        return Completion(content=content, model=self.config["model"])


# ============================================================================
# 6. Привязка вопроса к тикету
# ============================================================================


def _question_terms(question: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ]+", question.lower())
        if len(token) >= 5 and token not in _STOPWORDS
    ]


# Частотные слова, которые не помогают привязать вопрос к тикету.
_STOPWORDS = {
    "почему", "можно", "нужно", "надо", "стоит", "будет", "делать", "просто",
    "вообще", "сейчас", "когда", "чтобы", "также", "очень", "более", "менее",
    "потом", "если", "либо", "какой", "какие", "какая", "какое", "чего", "чему",
    "этого", "этому", "значит", "подскажите", "здравствуйте", "пожалуйста",
    "снова", "опять", "такое", "такой", "всё", "все", "есть", "быть",
}


_STATUS_RANK = {
    "open": 0,
    "in_progress": 1,
    "waiting_user": 2,
    "resolved": 3,
    "closed": 4,
}


def resolve_support_ticket(
    question: str,
    ticket_id: str | None,
    user_id: int | None,
    call_tool: Any,
) -> tuple[str | None, int | None]:
    """Определяет тикет по вопросу: явный id → тикет пользователя → текстовый поиск.

    ``call_tool(name, args) -> (structured, error)`` — обёртка над MCP-рантаймом.
    Возвращает пару ``(ticket_id, user_id)``; значения могут быть ``None``.
    """
    explicit = (ticket_id or "").strip()
    if not explicit:
        match = _TICKET_ID_RE.search(question or "")
        if match:
            explicit = match.group(0).upper()
    if explicit:
        return explicit, user_id

    if user_id is not None:
        listed, error = call_tool("list_tickets", {"user_id": str(user_id)})
        tickets = (listed or {}).get("tickets") or []
        if not error and tickets:
            tickets = sorted(tickets, key=lambda t: _STATUS_RANK.get(str(t.get("status")), 9))
            return str(tickets[0].get("id")), user_id

    # Текстовый поиск по значимым словам вопроса.
    terms = list(dict.fromkeys(_question_terms(question)))

    scores: dict[str, int] = {}
    matched_terms: dict[str, set[str]] = {}
    tickets_by_id: dict[str, dict] = {}
    for term in terms:
        found, error = call_tool("search_tickets", {"query": term, "limit": 10, "scope": "meta"})
        if error or not found:
            continue
        for ticket in found.get("tickets") or []:
            tid = str(ticket.get("id"))
            scores[tid] = scores.get(tid, 0) + 1
            matched_terms.setdefault(tid, set()).add(term)
            tickets_by_id[tid] = ticket

    # Уверенность: минимум два совпавших слова или одно длинное (специфичное).
    def _confident(tid: str) -> bool:
        words = matched_terms.get(tid, set())
        return len(words) >= 2 or any(len(word) >= 8 for word in words)

    confident = [tid for tid in scores if _confident(tid)]
    if not confident:
        return None, user_id

    ranked = sorted(
        confident,
        key=lambda tid: (
            -scores[tid],
            _STATUS_RANK.get(str(tickets_by_id[tid].get("status")), 9),
        ),
    )
    return ranked[0], user_id


# ============================================================================
# 7. Ассистент поддержки
# ============================================================================


class SupportAssistant:
    """Ассистент поддержки: RAG по FAQ + контекст тикета/пользователя через MCP."""

    def __init__(
        self,
        client: Any,
        retriever: RagRetriever,
        support_runtime: McpToolRuntime | None = None,
        top_k: int = 5,
        max_context_chars: int = 6000,
        temperature: float = 0.1,
        min_score: float = 0.0,
        relevance_threshold: float = 0.30,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.support_runtime = support_runtime
        self.top_k = max(1, int(top_k))
        self.max_context_chars = int(max_context_chars)
        self.temperature = float(temperature)
        self.min_score = float(min_score)
        self.relevance_threshold = float(relevance_threshold)
        self.system_prompt = system_prompt

    @property
    def total_vectors(self) -> int:
        return self.retriever.total_vectors

    def _call_tool(self, name: str, arguments: dict | None = None) -> tuple[dict | None, str | None]:
        """Вызывает MCP-инструмент; возвращает (structured, error)."""
        if self.support_runtime is None:
            return None, "MCP-сервер поддержки не подключён"
        try:
            result: ToolResult = self.support_runtime.call(name, arguments or {})
        except Exception as exc:  # noqa: BLE001 — понятная ошибка MCP
            return None, f"{type(exc).__name__}: {exc}"
        if result.is_error:
            return None, result.content or "ошибка инструмента"
        return result.structured, None

    def retrieve(self, question: str, top_k: int | None = None) -> list[RagSource]:
        """Ищет релевантные карточки FAQ.

        Срабатывает порог релевантности: если лучший найденный чанк слабее
        ``relevance_threshold``, считается, что ответа в FAQ нет (пустой список) —
        ассистент честно скажет об этом и предложит эскалацию.
        """
        k = top_k or self.top_k
        candidates = self.retriever.retrieve(question, top_k=max(k * 3, k))
        candidates = [src for src in candidates if src.score >= self.min_score]
        if not candidates:
            return []
        best = max(src.score for src in candidates)
        if best < self.relevance_threshold:
            return []
        candidates = [src for src in candidates if src.score >= max(self.min_score, 0.5 * best)]
        return self._rerank(question, candidates)[:k]

    @staticmethod
    def _rerank(question: str, sources: list[RagSource]) -> list[RagSource]:
        """Лёгкий реранкинг: векторное сходство + лексическое пересечение."""
        terms = {
            token
            for token in re.findall(r"[0-9a-zA-Zа-яА-ЯёЁ]+", question.lower())
            if len(token) > 2
        }
        denominator = len(terms) or 1
        scored: list[tuple[float, RagSource]] = []
        for src in sources:
            haystack = f"{src.text} {src.title} {src.section}".lower()
            overlap = sum(1 for term in terms if term in haystack) / denominator
            adjusted = src.score + 0.12 * overlap
            scored.append((adjusted, src))
        scored.sort(key=lambda item: item[0], reverse=True)
        for adjusted, src in scored:
            src.score = adjusted
        return [src for _, src in scored]

    def resolve_ticket(
        self, question: str, ticket_id: str | None = None, user_id: int | None = None
    ) -> tuple[str | None, int | None]:
        """Определяет тикет/пользователя для вопроса (через MCP)."""
        return resolve_support_ticket(question, ticket_id, user_id, self._call_tool)

    def collect_ticket_context(
        self, ticket_id: str | None, user_id: int | None = None
    ) -> dict[str, Any]:
        """Собирает контекст тикета и пользователя через MCP."""
        context: dict[str, Any] = {"available": self.support_runtime is not None, "errors": []}
        if self.support_runtime is None:
            return context

        if ticket_id:
            data, error = self._call_tool("get_ticket_context", {"ticket_id": ticket_id})
            if error:
                context["errors"].append(f"get_ticket_context: {error}")
                raw, raw_error = self._call_tool("get_ticket", {"ticket_id": ticket_id})
                if raw_error:
                    context["errors"].append(f"get_ticket: {raw_error}")
                else:
                    context.update(raw or {})
            else:
                context.update(data or {})

        resolved_user_id = user_id
        if resolved_user_id is None and context.get("user_id") is not None:
            try:
                resolved_user_id = int(context["user_id"])
            except (TypeError, ValueError):
                resolved_user_id = None
        if resolved_user_id is not None:
            user, error = self._call_tool("get_user", {"user_id": resolved_user_id})
            if error:
                context["errors"].append(f"get_user: {error}")
            else:
                context["user"] = user
        return context

    def ask(
        self,
        question: str,
        ticket_id: str | None = None,
        user_id: int | None = None,
        top_k: int | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> SupportAnswer:
        """Отвечает на вопрос о продукте с учётом FAQ и данных тикета."""
        started = time.perf_counter()
        try:
            faq_sources = self.retrieve(question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 — ошибка поиска не должна ронять CLI
            return SupportAnswer(question=question, content="", error=str(exc))

        resolved_ticket, resolved_user = self.resolve_ticket(question, ticket_id, user_id)
        ticket_context = self.collect_ticket_context(resolved_ticket, resolved_user)

        messages = build_support_messages(
            question,
            faq_sources,
            ticket_context,
            max_chars=self.max_context_chars,
            history=history,
            system_prompt=self.system_prompt,
        )
        try:
            completion = self.client.complete(messages, temperature=self.temperature)
        except Exception as exc:  # noqa: BLE001 — понятная ошибка модели
            return SupportAnswer(
                question=question,
                content="",
                faq_sources=faq_sources,
                ticket_context=ticket_context,
                ticket_id=resolved_ticket,
                user_id=resolved_user,
                error=f"{type(exc).__name__}: {exc}",
            )

        meta = parse_answer_meta(completion.content)
        latency = (time.perf_counter() - started) * 1000
        return SupportAnswer(
            question=question,
            content=completion.content,
            faq_sources=faq_sources,
            ticket_context=ticket_context,
            user_context=ticket_context.get("user") or {},
            ticket_id=meta.get("ticket_id") or resolved_ticket,
            user_id=resolved_user,
            answer_kind=meta.get("answer_kind") or ("faq" if faq_sources else "unknown"),
            escalate=bool(meta.get("escalate")),
            latency_ms=latency,
        )

    # --- Данные для CLI -------------------------------------------------
    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        data, error = self._call_tool("get_ticket_context", {"ticket_id": ticket_id})
        return {"error": error} if error else (data or {})

    def get_user(self, user_id: int) -> dict[str, Any]:
        data, error = self._call_tool("get_user", {"user_id": user_id})
        return {"error": error} if error else (data or {})

    def list_tickets(self, user_id: int | None = None) -> dict[str, Any]:
        args = {"user_id": str(user_id)} if user_id is not None else {}
        data, error = self._call_tool("list_tickets", args)
        return {"error": error} if error else (data or {})


# ============================================================================
# 8. Фабрики
# ============================================================================


def build_support_runtime(config: dict | None = None) -> McpToolRuntime | None:
    """Создаёт MCP-рантайм сервера поддержки из конфигурации."""
    section = ((config or {}).get("support", {}) or {}).get("support_server", {}) or {}
    if section.get("enabled") is False:
        return None
    command = section.get("command") or sys.executable
    args = section.get("args") or [str(DEFAULT_SUPPORT_SERVER)]
    cwd = section.get("cwd") or str(PROJECT_ROOT)
    data_dir = section.get("data_dir") or str(DEFAULT_SUPPORT_DATA_DIR)
    return McpToolRuntime(
        command=command,
        args=args,
        cwd=cwd,
        env={"SUPPORT_DATA_DIR": str(data_dir)},
        timeout=float(section.get("timeout", 30)),
    )


def build_support_assistant(
    config: dict | None = None,
    offline: bool = False,
    faq_path: str = DEFAULT_FAQ_PATH,
    index_dir: str = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    temperature: float = 0.1,
    max_context_chars: int = 6000,
    relevance_threshold: float = 0.30,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    build_index_if_missing: bool = True,
    with_support_mcp: bool = True,
) -> SupportAssistant:
    """Собирает ассистента поддержки: FAQ-индекс, MCP-рантайм, LLM-клиент."""
    config = config or {}
    try:
        retriever = load_faq_retriever(
            index_dir=index_dir, dataset=dataset, strategy=strategy, top_k=top_k, min_score=min_score
        )
    except FileNotFoundError:
        if not build_index_if_missing:
            raise
        index, _ = build_faq_index(
            faq_path=faq_path,
            index_dir=index_dir,
            dataset=dataset,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        retriever = RagRetriever(index=index, top_k=top_k, min_score=min_score)

    if offline:
        client: Any = SupportStubClient()
    else:
        client = LLMClient(config)

    runtime = build_support_runtime(config) if with_support_mcp else None
    return SupportAssistant(
        client=client,
        retriever=retriever,
        support_runtime=runtime,
        top_k=top_k,
        max_context_chars=max_context_chars,
        temperature=temperature,
        min_score=min_score,
        relevance_threshold=relevance_threshold,
    )


def _preview(text: str, limit: int = 200) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _demo() -> int:
    """Небольшая самопроверка модуля без сети."""
    assistant = build_support_assistant(config={}, offline=True)
    print(f"Индекс FAQ: {assistant.total_vectors} чанков")
    question = "Почему не работает авторизация?"
    answer = assistant.ask(question, ticket_id="T-1024")
    print(f"\nВопрос: {question}")
    print(f"Тикет: {answer.ticket_id}; вид ответа: {answer.answer_kind}; эскалация: {answer.escalate}")
    print(f"Ответ:\n{_preview(answer.content, 600)}\n")
    print(f"Источников FAQ: {len(answer.faq_sources)}")
    for src in answer.faq_sources[:3]:
        print(f"  - [{src.recipe_id}] «{src.title}» ({src.score:.3f})")
    return 0 if answer.content and answer.faq_sources and answer.ticket_id else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
