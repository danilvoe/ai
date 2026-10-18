#!/usr/bin/env python3
"""День 31: сквозной сценарий ассистента разработчика (офлайн-проверка).

Сценарий проверяет все три требования Дня 31 без модели и сети:

1. **Документация в RAG** — собирает индекс из ``README`` + ``docs/`` и убеждается,
   что в индекс попали нужные документы;
2. **MCP проекта** — через ``mcp/project_server.py`` получает текущую git-ветку,
   список файлов, коммиты и diff (минимум — ветка);
3. **``/help`` отвечает на вопросы о проекте** — задаёт вопросы о структуре и
   проверяет, что ответы опираются на документацию и содержат git-контекст.

Отчёт сохраняется в ``history/dev_assistant/`` (Markdown + JSON).

Запуск::

    python3 -m agent.dev_assistant_scenarios --offline
    python3 -m agent.dev_assistant_scenarios --offline --question "где MCP-сервер?"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.config import load_config
from agent.dev_assistant import (
    DevAssistant,
    build_dev_assistant,
    build_project_runtime,
)
from agent.project_docs import (
    DEFAULT_DATASET,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY,
    build_documents_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Вопросы о структуре проекта и файлы, где ожидается ответ.
CONTROL_QUESTIONS: list[dict] = [
    {
        "id": "q1",
        "query": "Как устроена архитектура проекта и какие в нём подсистемы?",
        "expect_files": ["docs/architecture.md"],
    },
    {
        "id": "q2",
        "query": "Какие MCP-серверы есть в проекте и какие у них инструменты?",
        "expect_files": ["docs/api.md", "docs/architecture.md"],
    },
    {
        "id": "q3",
        "query": "Как устроена подсистема RAG и где хранится индекс?",
        "expect_files": ["docs/architecture.md", "docs/data-schemas.md"],
    },
    {
        "id": "q4",
        "query": "Как запустить офлайн-проверки без затрат на API?",
        "expect_files": ["docs/development.md"],
    },
    {
        "id": "q5",
        "query": "Какие схемы данных используются в проекте: конфиг, рецепты, сессии?",
        "expect_files": ["docs/data-schemas.md"],
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _preview(text: str, limit: int = 200) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def check_index(index_dir: str, dataset: str, strategy: str) -> dict:
    """Собирает индекс документации и проверяет состав источников."""
    print("=" * 78)
    print("1. Документация проекта в RAG (README + docs/)")
    print("=" * 78)
    index, docs = build_documents_index(
        index_dir=index_dir, dataset=dataset, strategy=strategy
    )
    sources = sorted({chunk.metadata.get("source", "?") for chunk in index.chunks})
    has_readme = "README.md" in sources
    has_docs = any(name.startswith("docs/") for name in sources)
    ok = bool(index.chunks) and has_readme and has_docs

    print(f"Документов: {len(docs)}; чанков: {len(index.chunks)}")
    print(f"README в индексе: {'да' if has_readme else 'нет'}")
    print(f"docs/ в индексе: {'да' if has_docs else 'нет'}")
    for name in sources:
        print(f"  - {name}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "documents": [doc.path for doc in docs],
        "sources": sources,
        "chunks": len(index.chunks),
        "has_readme": has_readme,
        "has_docs": has_docs,
    }


def check_mcp(runtime, limit: int = 10) -> dict:
    """Проверяет MCP-инструменты проекта (минимум — get_current_branch)."""
    print("=" * 78)
    print("2. MCP-сервер проекта: ветка, файлы, коммиты, diff")
    print("=" * 78)
    if runtime is None:
        print("MCP-сервер проекта не подключён — проверка не пройдена.\n")
        return {"ok": False, "error": "runtime is None"}

    branch = runtime.call("get_current_branch").structured or {}
    files = runtime.call("list_project_files", {"limit": limit}).structured or {}
    commits = runtime.call("get_recent_commits", {"limit": 3}).structured or {}
    diff = runtime.call("get_diff", {"max_chars": 500}).structured or {}

    branch_ok = bool(branch.get("branch")) and not branch.get("error")
    files_ok = bool(files.get("files"))
    commits_ok = bool(commits.get("commits"))
    ok = branch_ok and files_ok and commits_ok

    print(f"Ветка: {branch.get('branch')} (коммит {branch.get('commit')})")
    print(f"Файлов: {files.get('total')} (показаны {files.get('returned')})")
    print(f"Коммитов: {commits.get('count')}")
    print(f"Diff: {diff.get('chars', 0)} символов, base={diff.get('base')}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "branch": branch,
        "files_total": files.get("total"),
        "commits": commits.get("count"),
        "diff_chars": diff.get("chars", 0),
        "branch_ok": branch_ok,
        "files_ok": files_ok,
        "commits_ok": commits_ok,
    }


def check_help(assistant: DevAssistant, questions: list[dict]) -> list[dict]:
    """Проверяет ответы ``/help`` на вопросы о структуре проекта."""
    print("=" * 78)
    print("3. /help отвечает на вопросы о проекте")
    print("=" * 78)
    results: list[dict] = []
    covered: set[str] = set()

    for item in questions:
        answer = assistant.answer_help(item["query"])
        source_files = [src.url for src in answer.sources]
        covered.update(source_files)
        branch = (answer.project_context.get("branch") or {}).get("branch", "")
        checks = {
            "non_empty": bool(answer.content),
            "has_sources": bool(answer.sources),
            "cites_sources": "[Источник" in answer.content,
            "has_branch": bool(branch),
            "mentions_branch": bool(branch) and branch in answer.content,
        }
        ok = all(checks.values())
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "ok": ok,
                "checks": checks,
                "branch": branch,
                "sources": source_files,
                "answer": _preview(answer.content, 300),
            }
        )
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {item['id']}: {item['query']}")
        print(f"      источники: {', '.join(source_files[:3]) or '—'}")
        print(f"      ветка: {branch or '—'}")
        print(f"      ответ: {_preview(answer.content, 160)}")

    expected_docs = {"docs/architecture.md", "docs/api.md", "docs/data-schemas.md", "docs/development.md"}
    missing = sorted(expected_docs - covered)
    coverage_ok = not missing
    print(f"\nПокрытие ключевых документов: {'OK' if coverage_ok else 'FAIL — нет ' + ', '.join(missing)}\n")
    return results


def render_report(index_info: dict, mcp_info: dict, help_results: list[dict], coverage_ok: bool) -> str:
    lines = [
        "# День 31: ассистент разработчика — отчёт офлайн-проверки",
        "",
        f"Сформировано: {now_iso()}",
        "",
        "## 1. Документация в RAG",
        "",
        f"- Документов: {len(index_info.get('documents', []))}",
        f"- Чанков: {index_info.get('chunks', 0)}",
        f"- README в индексе: {'да' if index_info.get('has_readme') else 'нет'}",
        f"- docs/ в индексе: {'да' if index_info.get('has_docs') else 'нет'}",
        f"- Итог: {'OK' if index_info.get('ok') else 'FAIL'}",
        "",
        "## 2. MCP-сервер проекта",
        "",
        f"- Ветка: `{mcp_info.get('branch', {}).get('branch', '—')}`",
        f"- Файлов под контролем версий: {mcp_info.get('files_total', '—')}",
        f"- Последних коммитов: {mcp_info.get('commits', '—')}",
        f"- Diff: {mcp_info.get('diff_chars', 0)} символов",
        f"- Итог: {'OK' if mcp_info.get('ok') else 'FAIL'}",
        "",
        "## 3. /help: вопросы о проекте",
        "",
        "| ID | Вопрос | Источники | Ветка | Итог |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in help_results:
        sources = ", ".join(item["sources"][:3]) or "—"
        lines.append(
            f"| {item['id']} | {item['query']} | {sources} | "
            f"{item['branch'] or '—'} | {'OK' if item['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Покрытие ключевых документов: {'OK' if coverage_ok else 'FAIL'}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    config = load_config()
    section = config.get("dev_assistant", {}) or {}

    parser = argparse.ArgumentParser(
        description="День 31: сквозная проверка ассистента разработчика (офлайн)"
    )
    parser.add_argument("--offline", action="store_true", help="стаб без модели и сети")
    parser.add_argument("--no-llm", dest="offline", action="store_true", help="синоним --offline")
    parser.add_argument("--question", "-q", default=None, help="одиночный вопрос и выход")
    parser.add_argument("--top-k", type=int, default=int(section.get("top_k", 5)))
    parser.add_argument("--index-dir", default=section.get("index_dir", DEFAULT_INDEX_DIR))
    parser.add_argument("--dataset", default=section.get("dataset", DEFAULT_DATASET))
    parser.add_argument("--strategy", default=section.get("strategy", DEFAULT_STRATEGY))
    parser.add_argument("--no-mcp", action="store_true", help="не подключать MCP-сервер проекта")
    parser.add_argument(
        "--report-dir", default=section.get("report_dir", "history/dev_assistant")
    )
    parser.add_argument("--skip-save", action="store_true", help="не сохранять отчёт")
    args = parser.parse_args()

    print("=" * 78)
    print("🛠️  ДЕНЬ 31: АССИСТЕНТ РАЗРАБОТЧИКА (ДОКУМЕНТАЦИЯ + GIT-КОНТЕКСТ)")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой LLM API'}")
    print(f"Индекс: {args.index_dir}/{args.dataset}/{args.strategy}\n")

    # 1. Индекс документации.
    index_info = check_index(args.index_dir, args.dataset, args.strategy)

    # 2. MCP проекта.
    runtime = None if args.no_mcp else build_project_runtime(config)
    mcp_info = check_mcp(runtime)

    # Ассистент (использует собранный индекс).
    assistant = build_dev_assistant(
        config=config,
        offline=args.offline,
        index_dir=args.index_dir,
        dataset=args.dataset,
        strategy=args.strategy,
        top_k=args.top_k,
        build_index_if_missing=False,
        with_project_mcp=not args.no_mcp,
    )

    if args.question:
        answer = assistant.answer_help(args.question)
        print("=" * 78)
        print(f"Вопрос: {args.question}")
        print("=" * 78)
        print(answer.content)
        print("\nИсточники:")
        for src in answer.sources:
            print(f"  - {src.url} — «{src.section}» ({src.score:.3f})")
        branch = (answer.project_context.get("branch") or {}).get("branch")
        print(f"Ветка: {branch}")
        return 0 if answer.content else 1

    # 3. Вопросы /help.
    help_results = check_help(assistant, CONTROL_QUESTIONS)
    expected_docs = {"docs/architecture.md", "docs/api.md", "docs/data-schemas.md", "docs/development.md"}
    covered = {name for item in help_results for name in item["sources"]}
    coverage_ok = not (expected_docs - covered)

    all_ok = index_info["ok"] and mcp_info["ok"] and all(r["ok"] for r in help_results) and coverage_ok

    # 4. Отчёт.
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": now_iso(),
            "mode": "offline" if args.offline else "live",
            "index": index_info,
            "mcp": mcp_info,
            "help": help_results,
            "coverage_ok": coverage_ok,
            "all_ok": all_ok,
        }
        (report_dir / "dev_results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (report_dir / "dev_report.md").write_text(
            render_report(index_info, mcp_info, help_results, coverage_ok), encoding="utf-8"
        )
        print(f"Отчёт сохранён: {report_dir / 'dev_report.md'}\n")

    print("=" * 78)
    print(f"ИТОГ: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
