#!/usr/bin/env python3
"""День 33: сквозной сценарий ассистента поддержки (офлайн-проверка).

Сценарий проверяет все три требования Дня 33 без модели и сети:

1. **RAG по FAQ** — строится индекс ``support_faq``, и по каждому контрольному
   вопросу в топ-k находится ожидаемая карточка (полнота поиска);
2. **MCP-CRM** — сервер поддержки по локальной JSON-выгрузке отдаёт пользователя,
   тикет, список тикетов, результаты поиска и готовый контекст тикета;
3. **Ответ с учётом тикета** — ассистент (офлайн-стаб) на вопрос «Почему не
   работает авторизация?» учитывает данные тикета ``T-1024`` (статус, версию) и
   опирается на карточки FAQ; отдельно проверяется отрицательный контроль
   (вопрос вне базы → эскалация, без выдумывания).

Отчёт сохраняется в ``history/support/`` (Markdown + JSON).

Запуск::

    python3 -m agent.support_scenarios --offline
    python3 -m agent.support_scenarios --offline --ticket T-1024
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.config import load_config
from agent.rag import evaluate_answer_facts
from agent.support_assistant import (
    SupportAssistant,
    build_support_assistant,
    build_support_runtime,
)
from agent.support_faq import (
    CONTROL_QUESTIONS,
    DEFAULT_DATASET,
    DEFAULT_FAQ_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_faq_index,
    load_faq_retriever,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEGATIVE_QUESTION = "Сколько стоит подписка и можно ли оплатить картой?"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def check_faq_index(index_dir: str, strategy: str, top_k: int) -> dict:
    """Собирает индекс FAQ и проверяет полноту поиска по контрольным вопросам."""
    print("=" * 78)
    print("1. RAG по FAQ: индекс и полнота поиска")
    print("=" * 78)
    index, items = build_faq_index(index_dir=index_dir, dataset=DEFAULT_DATASET, strategy=strategy)
    retriever = load_faq_retriever(
        index_dir=index_dir, dataset=DEFAULT_DATASET, strategy=strategy, top_k=top_k
    )

    details: list[dict] = []
    hits = 0
    for question in CONTROL_QUESTIONS:
        sources = retriever.retrieve(question["query"], top_k=top_k)
        found_ids = [src.recipe_id for src in sources]
        expected = set(question["expected_faq_ids"])
        recall = (not expected) or bool(expected & set(found_ids))
        hits += recall
        details.append(
            {
                "id": question["id"],
                "query": question["query"],
                "expected_faq_ids": sorted(expected),
                "found_faq_ids": found_ids,
                "recall": recall,
            }
        )

    total = len(CONTROL_QUESTIONS)
    recall_rate = round(hits / total, 3) if total else 0.0
    ok = bool(index.chunks) and recall_rate >= 0.9
    print(f"Карточек FAQ: {len(items)}; чанков: {len(index.chunks)}")
    print(f"Полнота поиска (recall@{top_k}): {hits}/{total} = {recall_rate:.0%}")
    for item in details:
        if not item["recall"]:
            print(f"  MISS {item['id']}: {item['query']} → {item['found_faq_ids']}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "items": len(items),
        "chunks": len(index.chunks),
        "recall": recall_rate,
        "hits": hits,
        "total": total,
        "details": details,
    }


def check_mcp(runtime, ticket_id: str) -> dict:
    """Проверяет MCP-CRM: пользователь, тикет, список, поиск, готовый контекст."""
    print("=" * 78)
    print("2. MCP-сервер поддержки: пользователь, тикет, список, поиск")
    print("=" * 78)
    if runtime is None:
        print("MCP-сервер поддержки не подключён — проверка не пройдена.\n")
        return {"ok": False, "error": "runtime is None"}

    user = runtime.call("get_user", {"user_id": 1}).structured or {}
    ticket = runtime.call("get_ticket", {"ticket_id": ticket_id}).structured or {}
    listed = runtime.call("list_tickets", {"user_id": str(user.get("id", 1))}).structured or {}
    search = runtime.call("search_tickets", {"query": "авторизация", "scope": "meta"}).structured or {}
    context = runtime.call("get_ticket_context", {"ticket_id": ticket_id}).structured or {}

    user_ok = bool(user.get("username")) and not user.get("error")
    ticket_ok = bool(ticket.get("messages")) and not ticket.get("error")
    list_ok = bool(listed.get("tickets")) and not listed.get("error")
    ids = [item.get("id") for item in (search.get("tickets") or [])]
    search_ok = ticket_id in ids
    context_ok = bool(context.get("context")) and not context.get("error")
    ok = user_ok and ticket_ok and list_ok and search_ok and context_ok

    print(f"Пользователь: {user.get('username')} (план: {user.get('plan')}, версия: {user.get('version')})")
    print(f"Тикет {ticket_id}: сообщений={len(ticket.get('messages') or [])}, статус={ticket.get('status')}")
    print(f"Тикетов у пользователя: {listed.get('total')}")
    print(f"Поиск 'авторизация' → {ids}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "user": user.get("username"),
        "user_ok": user_ok,
        "ticket_ok": ticket_ok,
        "messages": len(ticket.get("messages") or []),
        "list_total": listed.get("total"),
        "list_ok": list_ok,
        "search_ids": ids,
        "search_ok": search_ok,
        "context_ok": context_ok,
    }


def check_qa(assistant: SupportAssistant, ticket_id: str, top_k: int) -> dict:
    """Проверяет привязку тикета и сквозной ответ по контрольным вопросам."""
    print("=" * 78)
    print("3. Ответы ассистента: контрольные вопросы и отрицательный контроль")
    print("=" * 78)

    # 3.1. Автопривязка тикета по вопросу.
    auto_ticket, _ = assistant.resolve_ticket("Почему не работает авторизация?")
    resolve_ok = auto_ticket == ticket_id

    results: list[dict] = []
    hits = 0
    total = 0
    for question in CONTROL_QUESTIONS:
        answer = assistant.ask(question["query"], top_k=top_k)
        facts = evaluate_answer_facts(answer.content, question["expected_facts"])
        found_ids = {src.recipe_id for src in answer.faq_sources}
        expected = set(question["expected_faq_ids"])
        recall = (not expected) or bool(expected & found_ids)
        passed = recall and facts["coverage"] >= 1.0
        hits += passed
        total += 1
        results.append(
            {
                "id": question["id"],
                "query": question["query"],
                "ticket_id": answer.ticket_id,
                "answer_kind": answer.answer_kind,
                "escalate": answer.escalate,
                "found_faq_ids": sorted(found_ids),
                "recall": recall,
                "facts": facts,
                "passed": passed,
                "answer": answer.content,
            }
        )
    coverage = round(hits / total, 3) if total else 0.0

    # 3.2. Якорный кейс из задания: ответ с учётом данных тикета.
    anchor = assistant.ask("Почему не работает авторизация?", ticket_id=ticket_id, top_k=top_k)
    status = (anchor.ticket_context or {}).get("status", "")
    version = (anchor.ticket_context or {}).get("version", "")
    anchor_ok = (
        anchor.ticket_id == ticket_id
        and bool(anchor.faq_sources)
        and bool(status)
        and bool(version)
        and status in anchor.content
    )

    # 3.3. Отрицательный контроль: ответа в FAQ нет → эскалация, без выдумок.
    negative = assistant.ask(NEGATIVE_QUESTION)
    negative_ok = negative.answer_kind == "unknown" and negative.escalate and not negative.faq_sources

    ok = resolve_ok and coverage >= 0.9 and anchor_ok and negative_ok
    print(f"Автопривязка тикета по вопросу: {auto_ticket} ({'OK' if resolve_ok else 'FAIL'})")
    print(f"Контрольные вопросы: {hits}/{total} = {coverage:.0%}")
    for item in results:
        if not item["passed"]:
            print(f"  MISS {item['id']}: {item['query']} (facts {item['facts']['coverage']:.0%})")
    print(f"Якорный кейс (тикет {ticket_id}): статус={status}, версия={version} ({'OK' if anchor_ok else 'FAIL'})")
    print(f"Отрицательный контроль: kind={negative.answer_kind}, escalate={negative.escalate} ({'OK' if negative_ok else 'FAIL'})")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "resolve_ticket": auto_ticket,
        "resolve_ok": resolve_ok,
        "coverage": coverage,
        "hits": hits,
        "total": total,
        "anchor_ok": anchor_ok,
        "anchor_status": status,
        "anchor_version": version,
        "negative_ok": negative_ok,
        "negative_kind": negative.answer_kind,
        "results": results,
    }


def render_report(index_info: dict, mcp_info: dict, qa: dict, all_ok: bool) -> str:
    lines = [
        "# День 33: ассистент поддержки — отчёт офлайн-проверки",
        "",
        f"Сформировано: {now_iso()}",
        "",
        "## 1. RAG по FAQ",
        "",
        f"- Карточек FAQ: {index_info.get('items')}, чанков: {index_info.get('chunks')}",
        f"- Полнота поиска: {index_info.get('hits')}/{index_info.get('total')} "
        f"({(index_info.get('recall') or 0) * 100:.0f}%)",
        f"- Итог: {'OK' if index_info.get('ok') else 'FAIL'}",
        "",
        "## 2. MCP-сервер поддержки (пользователи и тикеты)",
        "",
        f"- Пользователь: {mcp_info.get('user')}",
        f"- Тикет {mcp_info.get('ticket_id', 'T-1024')}: сообщений {mcp_info.get('messages')}",
        f"- Тикетов у пользователя: {mcp_info.get('list_total')}",
        f"- Поиск 'авторизация': {mcp_info.get('search_ids') or '—'}",
        f"- Итог: {'OK' if mcp_info.get('ok') else 'FAIL'}",
        "",
        "## 3. Ответы ассистента",
        "",
        f"- Автопривязка тикета по вопросу: `{qa.get('resolve_ticket')}`",
        f"- Контрольные вопросы: {qa.get('hits')}/{qa.get('total')} "
        f"({(qa.get('coverage') or 0) * 100:.0f}%)",
        f"- Якорный кейс: статус `{qa.get('anchor_status')}`, версия `{qa.get('anchor_version')}`",
        f"- Отрицательный контроль: вид ответа `{qa.get('negative_kind')}` (эскалация без выдумок)",
        f"- Итог: {'OK' if qa.get('ok') else 'FAIL'}",
        "",
        "### Ответы по контрольным вопросам",
        "",
    ]
    for item in qa.get("results", []):
        mark = "✅" if item["passed"] else "❌"
        lines.append(
            f"- {mark} **{item['query']}** — вид `{item['answer_kind']}`, "
            f"тикет `{item['ticket_id'] or '—'}`, источники {', '.join(item['found_faq_ids']) or '—'}"
        )
    lines += [
        "",
        f"**Общий итог: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 33: сквозная проверка ассистента поддержки (офлайн)"
    )
    parser.add_argument("--offline", action="store_true", help="офлайн-режим (по умолчанию)")
    parser.add_argument("--ticket", default="T-1024", help="якорный тикет для проверки")
    parser.add_argument("--top-k", type=int, default=5, help="сколько FAQ-источников брать")
    parser.add_argument("--faq-path", default=DEFAULT_FAQ_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--report-dir", default="history/support")
    parser.add_argument("--skip-save", action="store_true", help="не сохранять отчёт")
    args = parser.parse_args()

    print("=" * 78)
    print("🎧 ДЕНЬ 33: АССИСТЕНТ ПОДДЕРЖКИ (FAQ RAG + MCP-ТИКЕТЫ)")
    print("=" * 78)
    print(f"Индекс: {args.index_dir} (dataset: {DEFAULT_DATASET})")

    config: dict = {}
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # noqa: BLE001 — конфиг опционален
        print(f"ℹ️  Конфиг не загружен ({type(exc).__name__}); используются значения по умолчанию.")

    index_info = check_faq_index(args.index_dir, args.strategy, args.top_k)

    runtime = build_support_runtime(config)
    mcp_info = check_mcp(runtime, args.ticket)
    mcp_info["ticket_id"] = args.ticket

    assistant = build_support_assistant(
        config=config,
        offline=True,
        faq_path=args.faq_path,
        index_dir=args.index_dir,
        dataset=DEFAULT_DATASET,
        strategy=args.strategy,
        top_k=args.top_k,
        with_support_mcp=True,
        build_index_if_missing=False,
    )
    qa = check_qa(assistant, args.ticket, args.top_k)

    all_ok = bool(index_info["ok"] and mcp_info["ok"] and qa["ok"])

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "scenario_report.md").write_text(
            render_report(index_info, mcp_info, qa, all_ok), encoding="utf-8"
        )
        (report_dir / "support_results.json").write_text(
            json.dumps(
                {
                    "generated_at": now_iso(),
                    "index": index_info,
                    "mcp": mcp_info,
                    "qa": qa,
                    "all_ok": all_ok,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Отчёты сохранены: {report_dir}\n")

    print("=" * 78)
    print(f"ИТОГ: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
