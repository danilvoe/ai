#!/usr/bin/env python3
"""Day 9: сжатие истории — сравниваем качество ответов и расход токенов.

Запускает один и тот же диалог в двух вариантах:

- БЕЗ сжатия: вся история уходит в каждый запрос как есть.
- СО сжатием: последние keep_recent сообщений хранятся как есть, а старые
  заменяются summary, которое подставляется в запрос вместо полной истории.

Для каждого варианта показывается, как растёт контекст, сколько токенов
ушло за всю сессию, и какова стоимость. Дополнительно в конце диалога задаётся
"контрольный вопрос" про факт из начала беседы — так сравнивается качество
ответов: удержал ли агент суть диалога после сжатия.

По умолчанию работает в офлайн-режиме (ответы имитируются локально, без
реальных запросов к API, чтобы не тратить токены). Флаг --online отправляет
реальные запросы, и тогда качество проверяется полноценной моделью.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from agent.agent import Agent
from agent.compression import CompressionConfig
from agent.config import load_config
from agent.conversation import SessionStore
from agent.llm_client import Completion, LLMClient
from agent.tokens import ContextOverflowError, estimate_messages_tokens, estimate_tokens


class FakeClient:
    """Имитирует LLM: считает токены локально, не отправляя запросы в сеть.

    Для сжатия возвращает связное "summary" из переданных сообщений, чтобы
    поведение по токенам было похоже на реальное.
    """

    def __init__(self, config: dict) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        if any("модуль сжатия" in m.get("content", "") for m in messages):
            # Имитация качественного сжатия: сохраняем факты и учитываем
            # уже существующий summary (передаётся отдельным system-сообщением).
            texts = [m.get("content", "") for m in messages
                     if m.get("role") in ("user", "assistant")]
            prior = [m.get("content", "") for m in messages
                     if "Текущий summary" in m.get("content", "")]
            chunks = prior + texts
            content = "Summary (факты сохранены): " + " | ".join(chunks)[:400]
        else:
            last = messages[-1]["content"]
            content = "Имитация ответа модели на сообщение: " + last[:120]
        return Completion(
            content=content,
            model=self.config["model"],
            prompt_tokens=estimate_messages_tokens(messages),
            completion_tokens=estimate_tokens(content),
        )


def build_client(config: dict, offline: bool) -> object:
    return FakeClient(config) if offline else LLMClient(config)


class RecordingClient:
    """Обёртка над клиентом: отдельно считает служебные вызовы на сжатие.

    Сжатие истории — это отдельный LLM-запрос (на создание summary), который не
    попадает в учёт сессии агента (agent.usage). Чтобы показать его стоимость,
    обёртка помечает такие вызовы и копит их токены.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.summary_prompt_tokens = 0
        self.summary_completion_tokens = 0
        self.summary_calls = 0

    @property
    def config(self) -> dict:
        return self._client.config

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        is_summary = any("модуль сжатия" in m.get("content", "") for m in messages)
        reply = self._client.complete(messages, temperature=temperature, max_tokens=max_tokens)
        if is_summary:
            self.summary_calls += 1
            self.summary_prompt_tokens += reply.prompt_tokens or 0
            self.summary_completion_tokens += reply.completion_tokens or 0
        return reply


def _new_agent(config: dict, offline: bool, compression: CompressionConfig | None):
    """Создаёт агента в отдельной временной сессии."""
    tmp = tempfile.TemporaryDirectory()
    conversation = SessionStore(Path(tmp.name)).create()
    client = RecordingClient(build_client(config, offline))
    agent = Agent(
        client,
        conversation,
        max_context_tokens=config.get("max_context_tokens"),
        compression=compression,
    )
    return agent, conversation, client


def _format_report(title: str, agent: Agent, context_after: int) -> None:
    cost = agent.usage.total_cost
    cost_str = "н/д" if cost is None else f"${cost:.6f}"
    real_prompt = agent.usage.total_prompt_tokens
    real_completion = agent.usage.total_completion_tokens
    print(f"--- {title} ---")
    print(f"  сообщений в истории: {len(agent.history)} "
          f"(свёрнуто в summary: {agent.conversation.summarized})")
    print(f"  summary: {'есть' if agent.conversation.summary else 'нет'} "
          f"({len(agent.conversation.summary)} символов)")
    print(f"  контекст в конце (оценка): {context_after} токенов")
    if real_prompt is not None:
        print(f"  токены за сессию (вход):  {real_prompt} (реальное значение из API)")
        print(f"  токены за сессию (выход): {real_completion or 0} (реальное значение)")
        print(f"  токены за сессию (всего): {real_prompt + (real_completion or 0)}")
    else:
        print(f"  токены за сессию (запросы): {agent.usage.total_request_tokens}")
        print(f"  токены за сессию (ответы):  {agent.usage.total_response_tokens}")
        print(f"  токены за сессию (всего):   {agent.usage.total_tokens}")
    print(f"  стоимость сессии: {cost_str}")
    print()


def run_dialog(config: dict, offline: bool, compression: CompressionConfig | None,
               turns: int) -> tuple[Agent, object, str, bool]:
    """Проводит длинный диалог и возвращает агента, сессию и контрольный факт."""
    agent, conversation, client = _new_agent(config, offline, compression)
    # Несколько фактов, разбросанных по ранним репликам, чтобы проверить их
    # сохранность после сжатия.
    early_fact = "кодовое имя проекта — 'Север'"
    facts = [
        f"Запомни факт 1: {early_fact}.",
        "Запомни факт 2: любимый цвет — синий.",
        "Запомни факт 3: встреча в четверг в 10 утра.",
    ]
    script = facts + [
        f"Вопрос №{i}: детально объясни понятие '{'пример' * 6}'."
        for i in range(1, turns - len(facts) + 1)
    ]
    stopped = False
    for index, message in enumerate(script, start=1):
        before_summarized = conversation.summarized
        try:
            agent.ask(message)
        except ContextOverflowError:
            stopped = True
            break
        did_compress = conversation.summarized > before_summarized
        fresh = len(conversation.recent_messages)
        print(
            f"    ход {index:>2}: свежих {fresh:>2}, "
            f"свёрнуто {conversation.summarized:>2}"
            + ("  [СЖАТИЕ]" if did_compress else "")
        )
    if compression is not None:
        print(
            f"    вызовов на сжатие: {client.summary_calls}, "
            f"их токены: {client.summary_prompt_tokens} (вход) + "
            f"{client.summary_completion_tokens} (выход)"
        )
    print()
    return agent, conversation, early_fact, stopped, client


def run_quality_probe(agent: Agent, offline: bool, fact: str) -> tuple[bool, str]:
    """Задаёт контрольный вопрос про ранний факт и оценивает ответ."""
    question = f"Какое кодовое имя было у проекта, о котором шла речь в начале диалога?"
    try:
        reply = agent.ask(question)
    except ContextOverflowError:
        return False, "не задан (переполнение контекста)"
    answer = reply.content.lower()
    if offline:
        # В офлайне имитация не знает фактов, поэтому "качество" — это то,
        # сохранился ли факт в summary, переданном в последний запрос.
        context = "\n".join(m.get("content", "") for m in agent.context_messages)
        return fact.lower() in context.lower(), "offline (по сохранности факта в summary)"
    return fact.lower() in answer, "online (по наличию факта в ответе модели)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online",
        action="store_true",
        help="отправлять реальные запросы к API вместо локальной имитации",
    )
    parser.add_argument(
        "--turns", type=int, default=24,
        help="сколько всего реплик в диалоге (по умолчанию 24)",
    )
    parser.add_argument(
        "--keep-recent", type=int, default=None,
        help="сколько последних сообщений хранить без изменений",
    )
    parser.add_argument(
        "--config", default=None, help="путь к config.json (по умолчанию config.one.json)"
    )
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config)) if args.config else load_config()
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    offline = not args.online
    section = config.get("compression", {}) or {}
    keep_recent = args.keep_recent if args.keep_recent is not None else section.get("keep_recent", 10)
    compression = CompressionConfig(
        keep_recent=keep_recent,
        summarize_every=section.get("summarize_every", 10),
        max_tokens=section.get("max_tokens"),
    )

    print(f"Режим: {'офлайн (имитация)' if offline else 'онлайн (реальный API)'}")
    print(f"Диалог: {args.turns} реплик, сжатие: последние {keep_recent} как есть.\n")
    print("Пошагово (сжатие срабатывает, когда свежих сообщений становится "
          f"больше {keep_recent + compression.summarize_every}):\n")

    # 1. Без сжатия.
    agent_plain, _, fact, plain_stopped, _ = run_dialog(config, offline, None, args.turns)
    _format_report("БЕЗ сжатия", agent_plain, agent_plain.context_tokens)

    # 2. Со сжатием.
    agent_compressed, _, fact, comp_stopped, client_comp = run_dialog(config, offline, compression, args.turns)
    _format_report("СО сжатием", agent_compressed, agent_compressed.context_tokens)

    if plain_stopped:
        print("Без сжатия диалог прервался из-за переполнения контекста.\n")

    print("=== Контрольный вопрос (качество ответов) ===")
    ok_plain, mode_plain = run_quality_probe(agent_plain, offline, fact)
    ok_comp, mode_comp = run_quality_probe(agent_compressed, offline, fact)
    print(f"  без сжатия: {'факт сохранён' if ok_plain else 'факт утерян'} ({mode_plain})")
    print(f"  со сжатием: {'факт сохранён' if ok_comp else 'факт утерян'} ({mode_comp})")

    print("\n=== Расход токенов (до/после) ===")
    # Реальные входные токены от модели (считает токенизатор), если доступны.
    plain_in = agent_plain.usage.total_prompt_tokens or agent_plain.usage.total_request_tokens
    comp_in = agent_compressed.usage.total_prompt_tokens or agent_compressed.usage.total_request_tokens
    print(f"  без сжатия: {plain_in} токенов на запросы")
    print(f"  со сжатием: {comp_in} токенов на запросы")
    saved = plain_in - comp_in
    ratio = (saved / plain_in * 100 if plain_in else 0)
    print(f"  экономия: {saved} токенов ({ratio:.1f}%)")
    print("\n  Служебные вызовы на сжатие (отдельно, не входят в 'запросы' выше):")
    print(f"    кол-во: {client_comp.summary_calls}")
    print(f"    токены: {client_comp.summary_prompt_tokens} (вход) + "
          f"{client_comp.summary_completion_tokens} (выход) = "
          f"{client_comp.summary_prompt_tokens + client_comp.summary_completion_tokens} всего")


if __name__ == "__main__":
    main()
