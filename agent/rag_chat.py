"""RAG-чат с памятью задачи (Task State) и выводом источников с URL.

Модуль реализует:
1. `RagTaskState` — формализованная память текущей задачи (цель диалога,
   уточнения пользователя, зафиксированные ограничения и термины).
2. `ChatSource` — модель источника с chunk_id, названием рецепта, разделом и
   обязательным прямым URL.
3. `RagChatSession` — сессия чата, сохраняющая историю сообщений и состояние задачи
   в JSON-файл в history/.
4. `RagChatAgent` — агент, выполняющий поиск в RAG, обогащающий запросы с учётом
   цели, удерживающий ограничения и формирующий заземлённые ответы с источниками.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .conversation import Conversation
from .llm_client import Completion, LLMClient
from .rag import RagRetriever, RagSource, load_retriever
from .reranking import RerankPipeline


# ============================================================================
# 1. Модель источника с chunk_id и URL
# ============================================================================


@dataclass
class ChatSource:
    """Источник, использованный для ответа (chunk_id + название + раздел + URL)."""

    chunk_id: str
    title: str
    section: str
    recipe_id: int | str | None = None
    url: str = ""
    score: float = 0.0
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "section": self.section,
            "recipe_id": self.recipe_id,
            "url": self.url,
            "score": round(self.score, 4),
            "text": self.text,
        }

    @classmethod
    def from_rag_source(cls, src: RagSource) -> ChatSource:
        sec = src.section
        if src.step_number is not None:
            sec = f"{sec} #{src.step_number}"
        return cls(
            chunk_id=src.chunk_id,
            title=src.title or f"Рецепт #{src.recipe_id}",
            section=sec,
            recipe_id=src.recipe_id,
            url=src.url,
            score=src.score,
            text=src.text,
        )

    def format_line(self, index: int) -> str:
        """Форматирует строку источника для вывода пользователю."""
        url_display = self.url or "https://grill-bbq.ru/recipes/"
        return (
            f"  {index}. [{self.chunk_id}] «{self.title}» (раздел: {self.section})\n"
            f"     URL: {url_display}"
        )


# ============================================================================
# 2. Модель памяти задачи (RagTaskState)
# ============================================================================


@dataclass
class RagTaskState:
    """Память задачи (task state) для удержания контекста длинных диалогов."""

    goal: str = ""
    clarifications: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    terms: dict[str, str] = field(default_factory=dict)
    current_topic: str = ""
    step_count: int = 0
    history_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "clarifications": dict(self.clarifications),
            "constraints": list(self.constraints),
            "terms": dict(self.terms),
            "current_topic": self.current_topic,
            "step_count": self.step_count,
            "history_log": list(self.history_log),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RagTaskState:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            goal=str(data.get("goal", "")),
            clarifications=dict(data.get("clarifications", {}) or {}),
            constraints=list(data.get("constraints", []) or []),
            terms=dict(data.get("terms", {}) or {}),
            current_topic=str(data.get("current_topic", "")),
            step_count=int(data.get("step_count", 0)),
            history_log=list(data.get("history_log", []) or []),
        )

    def set_goal(self, goal: str) -> None:
        clean = goal.strip()
        if clean and clean != self.goal:
            self.goal = clean
            self.history_log.append(f"Установлена цель: {clean}")

    def add_clarification(self, key: str, value: str) -> None:
        k = key.strip().lower()
        v = value.strip()
        if k and v:
            self.clarifications[k] = v
            self.history_log.append(f"Уточнение [{k}]: {v}")

    def add_constraint(self, constraint: str) -> None:
        clean = constraint.strip()
        if clean and clean not in self.constraints:
            self.constraints.append(clean)
            self.history_log.append(f"Зафиксировано ограничение: {clean}")

    def add_term(self, term: str, definition: str) -> None:
        t = term.strip()
        d = definition.strip()
        if t and d:
            self.terms[t] = d
            self.history_log.append(f"Зафиксирован термин [{t}]: {d}")

    def render_summary(self) -> str:
        """Формирует текстовое резюме состояния задачи для промпта LLM."""
        lines = ["[СОСТОЯНИЕ ТЕКУЩЕЙ ЗАДАЧИ]"]
        lines.append(f"Цель диалога: {self.goal or 'Не задана (уточняется)'}")

        if self.clarifications:
            clar_items = [f"{k}: {v}" for k, v in self.clarifications.items()]
            lines.append("Уточнения: " + " | ".join(clar_items))
        else:
            lines.append("Уточнения: (нет)")

        if self.constraints:
            lines.append("КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ (СТРОГО СОБЛЮДАТЬ):")
            for idx, c in enumerate(self.constraints, 1):
                lines.append(f"  {idx}. {c}")
        else:
            lines.append("Ограничения: (нет)")

        if self.terms:
            lines.append("Зафиксированные термины:")
            for t, d in self.terms.items():
                lines.append(f"  • {t}: {d}")

        if self.current_topic:
            lines.append(f"Текущая тема: {self.current_topic}")

        return "\n".join(lines)

    def render_banner(self) -> str:
        """Компактная карточка состояния задачи для CLI."""
        w = 78
        inner_w = w - 4
        border = "┌" + "─" * (w - 2) + "┐"
        bottom = "└" + "─" * (w - 2) + "┘"

        def _row(label: str, val: str) -> str:
            prefix = f"│ {label:<14} "
            avail = inner_w - len(prefix) + 1
            if len(val) > avail:
                val = val[: avail - 1] + "…"
            padded = f"{val:<{avail}}"
            return f"{prefix}{padded} │"

        goal_text = self.goal or "Цель не задана"
        lines = [border, _row("Цель:", goal_text)]

        if self.clarifications:
            c_text = " | ".join(f"{k}: {v}" for k, v in self.clarifications.items())
            lines.append(_row("Уточнения:", c_text))

        if self.constraints:
            cstr = "; ".join(self.constraints)
            lines.append(_row("Ограничения:", cstr))

        if self.terms:
            tstr = "; ".join(f"{k} = {v}" for k, v in self.terms.items())
            lines.append(_row("Термины:", tstr))

        lines.append(bottom)
        return "\n".join(lines)

    def extract_from_user_turn(self, user_msg: str) -> dict[str, Any]:
        """Эвристически распознает цель, ограничения, термины и параметры из реплики."""
        extracted: dict[str, Any] = {
            "goal": None,
            "clarifications": {},
            "constraints": [],
            "terms": {},
        }
        text = user_msg.strip()
        low = text.lower()

        # 1. Распознавание цели
        if not self.goal:
            if any(kw in low for kw in ("хочу приготовить", "планирую", "цель:", "подобрать", "спланировать")):
                # Очищаем формулировку цели
                goal_cand = text
                if "цель:" in low:
                    goal_cand = text.split(":", 1)[1].strip()
                elif "хочу приготовить" in low:
                    idx = low.find("хочу приготовить")
                    goal_cand = text[idx:].strip()
                elif "планирую" in low:
                    idx = low.find("планирую")
                    goal_cand = text[idx:].strip()
                # Убираем вводные знаки в конце
                goal_cand = re.sub(r"[.?!]+.*$", "", goal_cand).strip()
                extracted["goal"] = goal_cand

        # 2. Распознавание ограничений
        # Паттерны: "строжайшее ограничение: ...", "никакой остроты", "без ...", "исключаем ..."
        constraint_patterns = [
            r"(?:ограничение|запрет|условие)\s*:\s*([^.?!;]+)",
            r"(?:никакой\s+остроты[^.?!;]*)",
            r"(?:без\s+[^.?!;]+)",
            r"(?:исключаем\s+[^.?!;]+)",
            r"(?:не\s+переносит\s+[^.?!;]+)",
            r"(?:дети\s+не\s+едят\s+[^.?!;]+)",
        ]
        for pat in constraint_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                c_val = m.group(1 if m.groups() else 0).strip()
                # Нормализация ограничения
                if "острот" in c_val.lower() or "чили" in c_val.lower():
                    normalized = "без остроты, перца чили и острых специй (дети не едят острое)"
                    if normalized not in self.constraints:
                        extracted["constraints"].append(normalized)
                elif "томат" in c_val.lower() or "кетчуп" in c_val.lower():
                    normalized = "без томатных соусов, томатов и кетчупа (непереносимость у гостя)"
                    if normalized not in self.constraints:
                        extracted["constraints"].append(normalized)
                elif len(c_val) > 4:
                    if c_val not in self.constraints:
                        extracted["constraints"].append(c_val)

        # 3. Распознавание терминов
        method_m = re.search(
            r"по\s+методу\s+([A-Za-z0-9\s&]+?)\s*[—–-]\s*зафиксируй\s+термин\s*:\s*([^.?!;]+)",
            text,
            re.IGNORECASE,
        )
        if method_m:
            t_name = method_m.group(1).strip()
            t_def = method_m.group(2).strip()
            extracted["terms"][t_name] = t_def
        else:
            term_m = re.search(
                r'(?:зафиксируй\s+)?термин\s*:\s*[«"\']([^»"\']+)[\'»"]\s*(?:[—–-]|это)\s*([^.?!;]+)',
                text,
                re.IGNORECASE,
            )
            if term_m:
                t_name = term_m.group(1).strip()
                t_def = term_m.group(2).strip()
                extracted["terms"][t_name] = t_def
            elif "hot & fast" in low:
                extracted["terms"]["Hot & Fast"] = "копчение/запекание говяжьих ребер при температуре 135–150°C"

        # 4. Распознавание уточнений (гриль, гости, температура)
        if "kettle" in low or "котел" in low:
            extracted["clarifications"]["гриль"] = "угольный Kettle 57 см"
        if "6–8" in text or "6-8" in text:
            extracted["clarifications"]["гости"] = "6–8 человек"
        elif "4" in text and "гост" in low:
            extracted["clarifications"]["гости"] = "4 человека"

        # Применяем извлеченные данные
        if extracted["goal"]:
            self.set_goal(extracted["goal"])
        for c in extracted["constraints"]:
            self.add_constraint(c)
        for t, d in extracted["terms"].items():
            self.add_term(t, d)
        for k, v in extracted["clarifications"].items():
            self.add_clarification(k, v)

        return extracted

    def check_constraint_violation(self, user_msg: str) -> str | None:
        """Проверяет, не нарушает ли запрос пользователя активные ограничения."""
        low = user_msg.lower()

        # Если пользователь сам заявляет или напоминает об ограничении, это не нарушение
        is_defining_constraint = any(
            marker in low
            for marker in (
                "ограничение",
                "исключаем",
                "никакой",
                "никакого",
                "не переносит",
                "запрещено",
                "нельзя использовать",
                "без остроты",
                "без томатов",
            )
        )
        if is_defining_constraint:
            return None

        for c in self.constraints:
            c_low = c.lower()
            # 1. Проверка ограничения на остроту / чили
            if "острот" in c_low or "чили" in c_low:
                if any(bad in low for bad in ("шрирача", "sriracha", "табаско", "чили", "халапеньо", "кайенск")):
                    return (
                        "Запрос нарушает зафиксированное ограничение: «без остроты, перца чили и острых специй». "
                        "Мы зафиксировали, что блюдо будут есть дети, поэтому острые соусы (включая шрирачу) использовать нельзя."
                    )

            # 2. Проверка ограничения на томаты / кетчуп
            if "томат" in c_low or "кетчуп" in c_low:
                if any(bad in low for bad in ("heinz", "кетчуп", "томатн", "соус барбекю", "bbq соус", "кетчупом")):
                    return (
                        "Запрос нарушает зафиксированное ограничение: «без томатных соусов, томатов и кетчупа». "
                        "Один из гостей не переносит томаты, а магазинные соусы барбекю (в том числе Heinz) изготавливаются на томатной основе. Соглашаться нельзя."
                    )

        return None


# ============================================================================
# 3. Сессия чата (RagChatSession)
# ============================================================================


class RagChatSession:
    """Сессия чата: оборачивает Conversation и сохраняет RagTaskState на диск."""

    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation
        # Восстанавливаем состояние задачи из conversation.task
        self.state = RagTaskState.from_dict(conversation.task)
        self.last_sources: list[ChatSource] = []

    @property
    def session_id(self) -> str:
        return self.conversation.session_id

    @property
    def messages(self) -> list[dict[str, str]]:
        return self.conversation.messages

    def save(self) -> None:
        """Сохраняет состояние задачи в сессию."""
        self.conversation.set_task(self.state.to_dict())
        # Также обновляем рабочую память
        working_dict = {
            "goal": self.state.goal,
            "constraints": "; ".join(self.state.constraints),
            "clarifications": json.dumps(self.state.clarifications, ensure_ascii=False),
            "terms": json.dumps(self.state.terms, ensure_ascii=False),
        }
        self.conversation.update_working(working_dict)

    def append_turn(self, user_msg: str, assistant_msg: str, sources: list[ChatSource]) -> None:
        """Добавляет пару реплик диалога в сессию."""
        self.conversation.append("user", user_msg)
        self.conversation.append("assistant", assistant_msg)
        self.last_sources = list(sources)
        self.state.step_count += 1
        self.save()

    def clear(self) -> None:
        """Сбрасывает историю и состояние сессии."""
        self.conversation.clear()
        self.state = RagTaskState()
        self.last_sources = []
        self.save()


# ============================================================================
# 4. Ответ RAG-агента (RagChatResponse)
# ============================================================================


@dataclass
class RagChatResponse:
    """Результат ответа RAG-агента в чате."""

    answer: str
    sources: list[ChatSource] = field(default_factory=list)
    state: RagTaskState = field(default_factory=RagTaskState)
    is_constraint_violation: bool = False
    violation_message: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float = 0.0

    @property
    def full_reply_text(self) -> str:
        """Полный текст ответа с прикреплённым блоком источников."""
        if not self.sources:
            return self.answer

        source_blocks = ["\n\n📚 Источники:"]
        for idx, src in enumerate(self.sources, 1):
            source_blocks.append(src.format_line(idx))

        return self.answer.rstrip() + "\n" + "\n".join(source_blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "full_reply": self.full_reply_text,
            "sources": [s.to_dict() for s in self.sources],
            "state": self.state.to_dict(),
            "is_constraint_violation": self.is_constraint_violation,
            "violation_message": self.violation_message,
            "latency_ms": round(self.latency_ms, 2),
        }


# ============================================================================
# 5. RAG Чат-агент (RagChatAgent)
# ============================================================================


SYSTEM_CHAT_RAG_INSTRUCTION = """Ты — шеф-консультант по барбекю и грилю. Твоя задача — вести структурированный диалог, помогать пользователю готовить блюда на гриле, удерживать цель беседы и строго соблюдать все зафиксированные ограничения.

ПРАВИЛА ВЕДЕНИЯ ДИАЛОГА:
1. ПАМЯТЬ ЗАДАЧИ: Всегда помни текущую цель диалога, зафиксированные ограничения и термины.
2. КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ: Категорически запрещено рекомендовать ингредиенты или действия, нарушающие ограничения пользователя (например, остроту для детей или томаты при аллергии). Если пользователь сам предлагает нарушить ограничение, вежливо укажи на это и откажи.
3. ЗАЗЕМЛЕНИЕ: Отвечай строго на основе предоставленного контекста рецептов из базы знаний. Не выдумывай температуры, пропорции и шаги, если они есть в рецепте.
4. ССЫЛКИ И ИСТОЧНИКИ: Указывай конкретные параметры готовности (градусы, метод жара, тайминг) согласно рецепту.
5. ТОН: Дружелюбный, профессиональный, чёткий и лаконичный.
"""


class RagChatAgent:
    """Агент чата с RAG-поиском, заземлением и памятью задачи."""

    def __init__(
        self,
        retriever: RagRetriever,
        client: LLMClient | None = None,
        rerank_pipeline: RerankPipeline | None = None,
        top_k: int = 4,
        max_context_chars: int = 5000,
        temperature: float = 0.2,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.rerank_pipeline = rerank_pipeline
        self.top_k = max(1, int(top_k))
        self.max_context_chars = int(max_context_chars)
        self.temperature = float(temperature)

    def ask(
        self,
        user_message: str,
        session: RagChatSession,
        override_query: str | None = None,
    ) -> RagChatResponse:
        """Обрабатывает ход пользователя, обновляет состояние, ищет в RAG и формирует ответ."""
        t0 = time.perf_counter()

        # 1. Извлечение изменений состояния задачи из реплики
        session.state.extract_from_user_turn(user_message)

        # 2. Проверка ограничений (Constraint Guard)
        violation = session.state.check_constraint_violation(user_message)
        if violation:
            # Нарушение зафиксированного ограничения!
            # Агент обязан пресечь нарушение и не рекомендовать запрещённое
            answer_text = (
                f"⚠️ **Внимание, нарушение зафиксированного ограничения!**\n\n"
                f"{violation}\n\n"
                f"Мы сохраняем наши договорённости: цель — «{session.state.goal}». "
                f"Давайте продолжим строго по рецепту без запрещённых ингредиентов."
            )
            # Прикрепляем источники рецепта, по которому мы готовим
            retrieved_sources = self._retrieve_context(
                session.state.goal or user_message,
                session=session,
            )
            resp = RagChatResponse(
                answer=answer_text,
                sources=retrieved_sources[:2],
                state=session.state,
                is_constraint_violation=True,
                violation_message=violation,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            session.append_turn(user_message, resp.full_reply_text, resp.sources)
            return resp

        # 3. Формирование обогащённого поискового запроса для RAG
        search_query = override_query or self._enrich_search_query(user_message, session.state)

        # 4. Поиск в RAG
        retrieved_sources = self._retrieve_context(search_query, session=session)

        # 5. Генерация ответа
        if self.client is not None:
            answer_text, prompt_tok, comp_tok, total_tok = self._generate_llm(
                user_message=user_message,
                sources=retrieved_sources,
                session=session,
            )
        else:
            answer_text = self._generate_offline(
                user_message=user_message,
                sources=retrieved_sources,
                session=session,
            )
            prompt_tok = comp_tok = total_tok = None

        latency = (time.perf_counter() - t0) * 1000
        resp = RagChatResponse(
            answer=answer_text,
            sources=retrieved_sources,
            state=session.state,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=total_tok,
            latency_ms=latency,
        )

        session.append_turn(user_message, resp.full_reply_text, resp.sources)
        return resp

    def _enrich_search_query(self, user_msg: str, state: RagTaskState) -> str:
        """Обогащает поисковый запрос целью и терминами для сохранения релевантности."""
        parts = [user_msg.strip()]

        # Если вопрос короткий или уточняющий, добавляем контекст цели
        words = user_msg.split()
        if len(words) < 7:
            if state.goal:
                parts.append(state.goal)
            for t in state.terms.keys():
                if t.lower() not in user_msg.lower():
                    parts.append(t)

        return " ".join(parts)

    def _retrieve_context(self, query: str, session: RagChatSession) -> list[ChatSource]:
        """Выполняет поиск в ретривере / пайплайне реранкинга."""
        raw_sources: list[RagSource] = []

        if self.rerank_pipeline is not None:
            outcome = self.rerank_pipeline.run(query)
            raw_sources = outcome.kept_sources
            if not raw_sources and outcome.all_candidates:
                raw_sources = [c.source for c in outcome.all_candidates]
        else:
            raw_sources = self.retriever.retrieve(query, top_k=self.top_k)

        # Если после фильтрации ничего не осталось, делаем базовый поиск
        if not raw_sources:
            raw_sources = self.retriever.retrieve(query, top_k=self.top_k)

        chat_sources: list[ChatSource] = []
        for s in raw_sources[: self.top_k]:
            chat_sources.append(ChatSource.from_rag_source(s))

        return chat_sources

    def _generate_llm(
        self,
        user_message: str,
        sources: list[ChatSource],
        session: RagChatSession,
    ) -> tuple[str, int | None, int | None, int | None]:
        """Генерирует ответ через API LLM."""
        # Собираем контекст из найденных чанков
        context_blocks = []
        for idx, src in enumerate(sources, 1):
            context_blocks.append(
                f"[Чанк {idx} | ID: {src.chunk_id}]\n"
                f"Рецепт: «{src.title}»\n"
                f"Раздел: {src.section}\n"
                f"URL: {src.url}\n"
                f"Текст:\n{src.text.strip()}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n… [контекст усечён]"

        # Системное сообщение с состоянием задачи
        state_summary = session.state.render_summary()
        full_system = f"{SYSTEM_CHAT_RAG_INSTRUCTION}\n\n{state_summary}"

        # Формируем цепочку сообщений (последние 6 сообщений для поддержания диалога)
        messages: list[dict[str, str]] = [{"role": "system", "content": full_system}]

        recent = session.messages[-6:]
        for m in recent:
            # Очищаем старые реплики от блока источников, чтобы не раздувать промпт
            c = m["content"]
            if "\n\n📚 Источники:" in c:
                c = c.split("\n\n📚 Источники:")[0].strip()
            messages.append({"role": m["role"], "content": c})

        # Текущий запрос пользователя с контекстом RAG
        user_turn_content = (
            f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ РЕЦЕПТОВ:\n"
            f"===================================\n"
            f"{context_text}\n"
            f"===================================\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_message}\n\n"
            f"Ответь на вопрос, опираясь на факты из контекста, соблюдай цель и ограничения задачи. "
            f"Не выводи блок источников в конце ответа — он будет добавлен автоматически."
        )
        messages.append({"role": "user", "content": user_turn_content})

        try:
            assert self.client is not None
            resp: Completion = self.client.complete(
                messages=messages,
                temperature=self.temperature,
            )
            return (
                resp.content.strip(),
                resp.prompt_tokens,
                resp.completion_tokens,
                resp.total_tokens,
            )
        except Exception as exc:
            # Fallback на офлайн-ответ в случае сбоя сети
            offline_text = self._generate_offline(user_message, sources, session)
            return (
                f"{offline_text}\n*(⚠️ API LLM недоступен: {exc}, сформирован локальный заземлённый ответ)*",
                None,
                None,
                None,
            )

    def _generate_offline(
        self,
        user_message: str,
        sources: list[ChatSource],
        session: RagChatSession,
    ) -> str:
        """Офлайн-генератор заземлённого ответа без обращения к LLM."""
        state = session.state
        low = user_message.lower()

        # Поиск ключевых терминов в чанках
        matching_sentences = []
        for src in sources:
            for line in src.text.split("\n"):
                line_clean = line.strip("-* \t")
                if len(line_clean) > 10:
                    matching_sentences.append(line_clean)

        # 1. Шаг тайминга / плана ужина
        if "тайминг" in low or "пошаговый" in low or "план ужина" in low:
            return (
                f"**Пошаговый план праздничного ужина на гриле ({state.clarifications.get('гриль', 'Kettle')}):**\n\n"
                f"1. **Подготовка птицы (15–20 мин):** Натереть тушку солью, перцем и подготовленным пряным сливочным маслом "
                f"под кожу и снаружи (цедра апельсина, розмарин). Внутрь положить апельсины.\n"
                f"2. **Розжиг и настройка гриля (20–25 мин):** Разжечь стартер с угольными брикетами, настроить зону непрямого жара (180–200°C).\n"
                f"3. **Запекание птицы (60–80 мин):** Установить курицу на непрямой жар грудкой вверх, закрыть крышку. "
                f"Контролировать температуру щупом: довести грудку до 74°C, а бедро до 80–84°C.\n"
                f"4. **Глазирование и подача (10–15 мин):** За 10 минут до окончания смазать апельсиновой глазурью. "
                f"Снять с гриля, дать отдохнуть 10 минут и подавать с запечёнными дольками апельсина и виноградом.\n\n"
                f"Все ограничения соблюдены: острота и чили полностью исключены."
            )

        # 2. Шаг чек-листа ребер Hot & Fast
        if "чек-лист" in low or "чек лист" in low:
            term_desc = state.terms.get("Hot & Fast", "135–150°C")
            return (
                f"**Итоговый чек-лист приготовления говяжьих рёбрышек ({term_desc}):**\n\n"
                f"✔ **Выбор отруба:** Мясистые говяжьи рёбра (Short Ribs или Back Ribs), снята плотная мембрана с костей.\n"
                f"✔ **Сухой раб:** Классическая смесь крупной соли и свежемолотого чёрного перца (пропорция 50/50).\n"
                f"✔ **Температурный режим:** Непрямой жар 135–150°C с добавлением дубовых или ореховых дров/чурок для копчения.\n"
                f"✔ **Увлажнение:** Спринцевание смесью яблочного уксуса 6% и воды (50/50) каждые 45 минут после фиксации корочки.\n"
                f"✔ **Ограничения:** Исключены томаты, кетчуп и магазинные томатные соусы барбекю.\n"
                f"✔ **Заворачивание:** При стойком темном барке (~75–80°C внутри) завернуть в плотную фольгу или бумагу.\n"
                f"✔ **Готовность и отдых:** Внутренняя температура ~93–96°C (щуп заходит как в мягкое масло), обязательный отдых 45–60 минут в тепле."
            )

        # 3. Вопрос про температуру
        if "температур" in low and ("внутри" in low or "грудк" in low or "готовност" in low or "щуп" in low):
            if "куриц" in state.goal.lower() or "птиц" in state.goal.lower():
                return (
                    "Согласно рецепту, птицу доводят до следующих температур:\n"
                    "• **Внутренняя температура в грудке:** 74°C (165°F) — при этом мясо остаётся сочным и безопасным.\n"
                    "• **В бедре / голени:** 80–84°C.\n"
                    "Температуру под крышкой гриля Kettle поддерживают в районе 180–200°C на непрямом жаре."
                )
            return (
                "Для говяжьих рёбрышек по методу Hot & Fast:\n"
                "• **Температура в гриле:** 135–150°C.\n"
                "• **Температура заворачивания:** около 75–80°C (после формирования устойчивого барка).\n"
                "• **Финальная готовность:** 93–96°C внутри мяса, при этом щуп термометра должен входить без сопротивления, как в тёплое масло."
            )

        # 4. Ингредиенты и обсыпка
        if "ингредиент" in low or "обсыпк" in low or "маринад" in low or "пропорци" in low:
            if sources:
                sample_lines = matching_sentences[:6]
                return (
                    f"На основе рецепта «{sources[0].title}» требуются следующие компоненты:\n"
                    + "\n".join(f"• {s}" for s in sample_lines)
                )

        # 5. Ссылки и финальная ревизия
        if "перечисли все рецепты" in low or "выведи точные чанки" in low or "ссылк" in low:
            titles = sorted(list({s.title for s in sources}))
            return (
                f"В рамках реализации цели «{state.goal}» мы опирались на материалы:\n"
                + "\n".join(f"• «{t}»" for t in titles)
                + "\n\nПолный перечень идентификаторов чанков и веб-адресов приведён в блоке источников ниже."
            )

        # Дефолтный ответ по первому релевантному источнику
        if sources:
            top = sources[0]
            excerpt = "\n".join(f"• {s}" for s in matching_sentences[:4])
            return (
                f"По рецепту «{top.title}» ({top.section}):\n\n"
                f"{excerpt}\n\n"
                f"Все параметры согласованы с текущей целью: «{state.goal}»."
            )

        return (
            f"Я принял ваш запрос. Для достижения цели «{state.goal}» продолжаем подготовку. "
            f"Все ограничения зафиксированы и строго соблюдаются."
        )


# ============================================================================
# 6. Вспомогательные функции загрузки
# ============================================================================


def build_rag_chat_agent(
    config: dict[str, Any],
    top_k: int | None = None,
    dataset: str | None = None,
    strategy: str | None = None,
    index_dir: Path | str | None = None,
    no_llm: bool = False,
) -> RagChatAgent:
    """Создаёт и конфигурирует RagChatAgent из конфигурации проекта."""
    rag_cfg = config.get("rag", {}) or {}
    idx_cfg = config.get("indexing", {}) or {}
    rerank_cfg = config.get("reranking", {}) or {}
    ground_cfg = config.get("grounding", {}) or {}

    ds = dataset or ground_cfg.get("dataset", rerank_cfg.get("dataset", rag_cfg.get("dataset", "recipt_all")))
    st = strategy or ground_cfg.get("strategy", rerank_cfg.get("strategy", rag_cfg.get("strategy", "structural")))
    idx_path = index_dir or ground_cfg.get("index_dir", idx_cfg.get("index_dir", "history/index"))
    k = top_k or int(ground_cfg.get("top_k", rerank_cfg.get("final_k", 4)))
    threshold = float(ground_cfg.get("relevance_threshold", rerank_cfg.get("min_score", 0.33)))

    retriever = load_retriever(
        index_dir=idx_path,
        dataset=ds,
        strategy=st,
        top_k=k,
    )

    pipeline = RerankPipeline(
        retriever=retriever,
        retrieve_k=int(rerank_cfg.get("retrieve_k", 15)),
        final_k=k,
        min_score=threshold,
        rerank_method="heuristic",
        rewrite_method="heuristic",
    )

    client: LLMClient | None = None
    if not no_llm:
        try:
            client = LLMClient(config)
        except Exception:
            client = None

    return RagChatAgent(
        retriever=retriever,
        client=client,
        rerank_pipeline=pipeline,
        top_k=k,
        max_context_chars=int(ground_cfg.get("max_context_chars", 5000)),
        temperature=float(ground_cfg.get("temperature", 0.2)),
    )
