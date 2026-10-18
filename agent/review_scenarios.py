#!/usr/bin/env python3
"""День 32: сквозной сценарий автоматического ревью кода (офлайн-проверка).

Сценарий проверяет все три требования Дня 32 без модели и сети:

1. **Индексы RAG** — собираются индексы документации (``project_docs``) и кода
   (``project_code``), в кодовом индексе есть символы ключевых модулей;
2. **MCP-сервер ревью** — по локальному git-репозиторию отдаёт diff и изменённые
   файлы (``get_pr_diff`` / ``list_changed_files``);
3. **Текст ревью** — на временном репозитории с «посаженным» патчем ассистент
   (офлайн-эвристики) выдаёт находки во всех трёх категориях: баги, архитектура,
   рекомендации, а markdown содержит нужные разделы и ссылки на источники.

Отчёт сохраняется в ``history/code_review/`` (Markdown + JSON).

Запуск::

    python3 -m agent.review_scenarios --offline
    python3 -m agent.review_scenarios --offline --base HEAD~1 --head HEAD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent.code_index import (
    DEFAULT_CODE_DATASET,
    DEFAULT_CODE_STRATEGY,
    build_code_index,
    load_code_retriever,
)
from agent.code_review import (
    CATEGORY_ARCHITECTURE,
    CATEGORY_BUG,
    CATEGORY_RECOMMENDATION,
    CATEGORY_TITLES,
    analyze_diff,
    build_review_assistant,
    build_review_runtime,
    parse_findings,
)
from agent.config import load_config
from agent.project_docs import (
    DEFAULT_DATASET as DEFAULT_DOCS_DATASET,
    DEFAULT_INDEX_DIR,
    DEFAULT_STRATEGY as DEFAULT_DOCS_STRATEGY,
    build_documents_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Базовый и «пропатченный» модуль для временного репозитория: патч намеренно
# содержит проблемы каждой категории, чтобы проверить весь конвейер ревью.
BASE_MODULE = '''"""Демонстрационный модуль для сценария ревью."""


def add(a, b):
    """Складывает два числа."""
    return a + b
'''

PATCHED_MODULE = '''"""Демонстрационный модуль для сценария ревью."""

import subprocess

API_KEY = "sk-test-1234567890abcdef"


def add(a, b):
    """Складывает два числа."""
    return a + b


def run(cmd):
    return subprocess.run(cmd, shell=True)


def parse(text):
    try:
        return eval(text)
    except:
        pass
'''

# Синтетический diff для прямой проверки эвристик.
FIXTURE_DIFF = """diff --git a/agent/example.py b/agent/example.py
--- a/agent/example.py
+++ b/agent/example.py
@@ -1,5 +1,20 @@
 import subprocess
+
+API_KEY = "sk-test-1234567890abcdef"
+
+
+def run(cmd):
+    return subprocess.run(cmd, shell=True)
+
+
+def parse(text):
+    try:
+        return eval(text)
+    except:
+        pass
+
+# TODO: добавить тесты
"""

MD_MARKDOWN_SAMPLE = """### Резюме
Найдены проблемы.

### Потенциальные баги
- [critical] Захардкоженный секрет
- [major] `shell=True`

### Архитектурные проблемы
- [major] subprocess вне MCP-слоя

### Рекомендации
- [minor] Добавить тесты

```json
{"summary": "Итог", "findings": [{"category": "bug", "severity": "critical", "title": "Секрет", "file": "a.py", "line": 3}]}
```
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _temp_dir() -> str:
    """Временный каталог (предпочитает одобренный /tmp/.private/...)."""
    preferred = "/tmp/.private/lifeteo/opencode"
    base = preferred if Path(preferred).is_dir() else tempfile.gettempdir()
    return tempfile.mkdtemp(prefix="review_scenario_", dir=base)


def _git(args: list[str], cwd: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def check_indexes(index_dir: str, strategy: str) -> dict:
    """Собирает индексы документации и кода, проверяет состав."""
    print("=" * 78)
    print("1. RAG-индексы: документация + код")
    print("=" * 78)
    docs_index, docs = build_documents_index(
        index_dir=index_dir, dataset=DEFAULT_DOCS_DATASET, strategy=strategy
    )
    code_index, modules = build_code_index(
        index_dir=index_dir, dataset=DEFAULT_CODE_DATASET, strategy=DEFAULT_CODE_STRATEGY
    )
    code_sources = {chunk.metadata.get("source", "?") for chunk in code_index.chunks}
    has_rag = "agent/rag.py" in code_sources
    has_assistant = "agent/dev_assistant.py" in code_sources
    ok = bool(docs_index.chunks) and bool(code_index.chunks) and has_rag and has_assistant

    print(f"Документов: {len(docs)}; чанков документации: {len(docs_index.chunks)}")
    print(f"Модулей кода: {len(modules)}; чанков кода: {len(code_index.chunks)}")
    print(f"agent/rag.py в индексе кода: {'да' if has_rag else 'нет'}")
    print(f"agent/dev_assistant.py в индексе кода: {'да' if has_assistant else 'нет'}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "documents": len(docs),
        "docs_chunks": len(docs_index.chunks),
        "modules": len(modules),
        "code_chunks": len(code_index.chunks),
        "has_rag": has_rag,
        "has_assistant": has_assistant,
    }


def check_mcp(runtime, base: str | None, head: str | None) -> dict:
    """Проверяет MCP-сервер ревью: diff и изменённые файлы по реальному репозиторию."""
    print("=" * 78)
    print("2. MCP-сервер ревью: diff и изменённые файлы")
    print("=" * 78)
    if runtime is None:
        print("MCP-сервер ревью не подключён — проверка не пройдена.\n")
        return {"ok": False, "error": "runtime is None"}

    candidates = []
    if base is not None:
        candidates.append((base, head or ""))
    candidates += [("HEAD~1", "HEAD"), ("day30", "day31"), ("HEAD", "")]

    chosen = None
    files = {}
    for cand_base, cand_head in candidates:
        result = runtime.call("list_changed_files", {"base": cand_base, "head": cand_head})
        data = result.structured or {}
        if data.get("files"):
            chosen = (cand_base, cand_head, data)
            files = data
            break

    if chosen is None:
        print("Не удалось подобрать диапазон с непустым diff.\n")
        return {"ok": False, "error": "нет диапазона с изменениями"}

    cand_base, cand_head, files = chosen
    diff = runtime.call("get_pr_diff", {"base": cand_base, "head": cand_head, "max_chars": 2000}).structured or {}
    first_path = files["files"][0]["path"]
    content = runtime.call("get_file_content", {"path": first_path, "ref": cand_head or cand_base}).structured or {}

    diff_ok = bool(diff.get("diff")) and not diff.get("error")
    content_ok = bool(content.get("content")) and not content.get("error")
    ok = bool(files.get("files")) and diff_ok

    print(f"Диапазон: {cand_base} → {cand_head or 'working tree'}")
    print(f"Изменённых файлов: {files.get('count')} (первый: {first_path})")
    print(f"Diff: {diff.get('chars', 0)} символов, усечён={diff.get('truncated')}")
    print(f"Файл на ревизии прочитан: {'да' if content_ok else 'нет'}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "base": cand_base,
        "head": cand_head,
        "files": files.get("count"),
        "first_file": first_path,
        "diff_chars": diff.get("chars", 0),
        "content_ok": content_ok,
    }


def check_heuristics() -> dict:
    """Прямая проверка эвристик и парсера находок."""
    print("=" * 78)
    print("3. Эвристики по diff и разбор ответа модели")
    print("=" * 78)
    findings = analyze_diff(FIXTURE_DIFF)
    categories = {finding.category for finding in findings}
    coverage_ok = categories == {CATEGORY_BUG, CATEGORY_ARCHITECTURE, CATEGORY_RECOMMENDATION}
    print(f"Находок на фикстуре: {len(findings)} (категории: {sorted(categories)})")

    summary, parsed = parse_findings(MD_MARKDOWN_SAMPLE)
    parse_ok = bool(summary) and len(parsed) >= 1 and parsed[0].category == CATEGORY_BUG
    print(f"Разбор markdown: резюме={'да' if summary else 'нет'}, находок={len(parsed)}")

    ok = coverage_ok and parse_ok
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "findings": len(findings),
        "categories": sorted(categories),
        "coverage_ok": coverage_ok,
        "parse_ok": parse_ok,
    }


def build_fixture_repo() -> tuple[str, str]:
    """Создаёт временный git-репозиторий с «посаженным» патчем.

    Возвращает (каталог, SHA базового коммита).
    """
    directory = _temp_dir()
    _git(["init", "-q"], directory)
    module_dir = Path(directory) / "agent"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "example.py").write_text(BASE_MODULE, encoding="utf-8")
    docs_dir = Path(directory) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "README.md").write_text("# Демо-репозиторий\n\nПравила проекта.\n", encoding="utf-8")

    identity = ["-c", "user.email=review@example.com", "-c", "user.name=Review Bot"]
    _git([*identity, "add", "-A"], directory)
    _git([*identity, "commit", "-q", "-m", "base"], directory)
    base_sha = _git(["rev-parse", "HEAD"], directory)

    (module_dir / "example.py").write_text(PATCHED_MODULE, encoding="utf-8")
    _git([*identity, "add", "-A"], directory)
    _git([*identity, "commit", "-q", "-m", "patch with issues"], directory)
    return directory, base_sha


def check_review_e2e(config: dict, index_dir: str, strategy: str, top_k: int) -> dict:
    """Полный прогон ассистента на временном репозитории с патчем."""
    print("=" * 78)
    print("4. Текст ревью: ассистент на PR с посаженными проблемами")
    print("=" * 78)
    repo_dir, base_sha = build_fixture_repo()
    print(f"Временный репозиторий: {repo_dir}")

    fixture_config = dict(config)
    review_section = dict(fixture_config.get("code_review", {}) or {})
    review_section.setdefault("review_server", {})
    review_section["review_server"] = {
        **dict(review_section["review_server"]),
        "repo_root": repo_dir,
    }
    fixture_config["code_review"] = review_section

    assistant = build_review_assistant(
        config=fixture_config,
        offline=True,
        docs_index_dir=index_dir,
        docs_dataset=DEFAULT_DOCS_DATASET,
        docs_strategy=strategy,
        code_index_dir=index_dir,
        code_dataset=DEFAULT_CODE_DATASET,
        code_strategy=DEFAULT_CODE_STRATEGY,
        top_k=top_k,
        with_review_mcp=True,
        build_index_if_missing=False,
    )
    review = assistant.review(base=base_sha, head="")

    counts = review.counts()
    markdown = review.to_markdown()
    sections_ok = all(title in markdown for title in CATEGORY_TITLES.values())
    categories_ok = all(counts[category] >= 1 for category in (CATEGORY_BUG, CATEGORY_ARCHITECTURE, CATEGORY_RECOMMENDATION))
    changed_ok = any(
        item.get("path") == "agent/example.py" for item in review.changed_files
    )
    diff_ok = bool(review.diff_stats.get("count"))
    sources_ok = review.docs_used >= 1 and review.code_used >= 1
    ok = sections_ok and categories_ok and changed_ok and diff_ok and sources_ok

    print(f"Изменённых файлов: {review.diff_stats.get('count')} (agent/example.py: {'да' if changed_ok else 'нет'})")
    print(f"Находки: баги={counts[CATEGORY_BUG]}, архитектура={counts[CATEGORY_ARCHITECTURE]}, рекомендации={counts[CATEGORY_RECOMMENDATION]}")
    print(f"Источников: документация={review.docs_used}, код={review.code_used}")
    print("Примеры находок:")
    for finding in review.findings[:5]:
        print(f"  [{finding.severity}] {finding.category}: {_preview(finding.title)}")
    print(f"Разделы в markdown: {'OK' if sections_ok else 'FAIL'}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")

    return {
        "ok": ok,
        "repo": repo_dir,
        "base": base_sha,
        "counts": counts,
        "changed_ok": changed_ok,
        "sections_ok": sections_ok,
        "sources_ok": sources_ok,
        "docs_used": review.docs_used,
        "code_used": review.code_used,
        "findings": [finding.to_dict() for finding in review.findings[:10]],
        "markdown": markdown,
    }


def render_report(index_info: dict, mcp_info: dict, heuristics: dict, e2e: dict, all_ok: bool) -> str:
    lines = [
        "# День 32: автоматическое ревью PR — отчёт офлайн-проверки",
        "",
        f"Сформировано: {now_iso()}",
        "",
        "## 1. RAG-индексы",
        "",
        f"- Документов: {index_info.get('documents')}, чанков: {index_info.get('docs_chunks')}",
        f"- Модулей кода: {index_info.get('modules')}, чанков: {index_info.get('code_chunks')}",
        f"- Итог: {'OK' if index_info.get('ok') else 'FAIL'}",
        "",
        "## 2. MCP-сервер ревью",
        "",
        f"- Диапазон: `{mcp_info.get('base')}` → `{mcp_info.get('head') or 'working tree'}`",
        f"- Изменённых файлов: {mcp_info.get('files')}",
        f"- Diff: {mcp_info.get('diff_chars')} символов",
        f"- Итог: {'OK' if mcp_info.get('ok') else 'FAIL'}",
        "",
        "## 3. Эвристики и разбор ответа",
        "",
        f"- Находок на фикстуре: {heuristics.get('findings')} (категории: {', '.join(heuristics.get('categories', []))})",
        f"- Итог: {'OK' if heuristics.get('ok') else 'FAIL'}",
        "",
        "## 4. Сквозное ревью на PR с проблемами",
        "",
        f"- Баги: {e2e.get('counts', {}).get(CATEGORY_BUG, 0)}",
        f"- Архитектура: {e2e.get('counts', {}).get(CATEGORY_ARCHITECTURE, 0)}",
        f"- Рекомендации: {e2e.get('counts', {}).get(CATEGORY_RECOMMENDATION, 0)}",
        f"- Источников RAG: документация {e2e.get('docs_used', 0)}, код {e2e.get('code_used', 0)}",
        f"- Итог: {'OK' if e2e.get('ok') else 'FAIL'}",
        "",
        f"**Общий итог: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 32: сквозная проверка автоматического ревью PR (офлайн)"
    )
    parser.add_argument("--offline", action="store_true", help="офлайн-режим (по умолчанию)")
    parser.add_argument("--base", default=None, help="база для MCP-проверки (по умолчанию авто)")
    parser.add_argument("--head", default=None, help="верхняя ревизия для MCP-проверки")
    parser.add_argument("--top-k", type=int, default=5, help="сколько RAG-источников брать")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--strategy", default=DEFAULT_DOCS_STRATEGY)
    parser.add_argument("--report-dir", default="history/code_review")
    parser.add_argument("--skip-save", action="store_true", help="не сохранять отчёт")
    args = parser.parse_args()

    print("=" * 78)
    print("🔍 ДЕНЬ 32: АВТОМАТИЧЕСКОЕ РЕВЬЮ PR (RAG: ДОКУМЕНТАЦИЯ + КОД)")
    print("=" * 78)
    print(f"Индекс: {args.index_dir} (docs: {DEFAULT_DOCS_DATASET}, code: {DEFAULT_CODE_DATASET})\n")

    config: dict = {}
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # noqa: BLE001 — конфиг опционален
        print(f"ℹ️  Конфиг не загружен ({type(exc).__name__}); используются значения по умолчанию.")

    index_info = check_indexes(args.index_dir, args.strategy)

    # Ретривер кода нужен, чтобы гарантировать, что индекс читается с диска.
    try:
        load_code_retriever(index_dir=args.index_dir, dataset=DEFAULT_CODE_DATASET, strategy=DEFAULT_CODE_STRATEGY)
    except FileNotFoundError as exc:
        print(f"⚠️  {exc}")

    runtime = build_review_runtime(config)
    mcp_info = check_mcp(runtime, args.base, args.head)
    heuristics = check_heuristics()
    e2e = check_review_e2e(config, args.index_dir, args.strategy, args.top_k)

    all_ok = bool(index_info["ok"] and mcp_info["ok"] and heuristics["ok"] and e2e["ok"])

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "review_report.md").write_text(
            e2e.get("markdown", ""), encoding="utf-8"
        )
        (report_dir / "review_results.json").write_text(
            json.dumps(
                {
                    "generated_at": now_iso(),
                    "index": index_info,
                    "mcp": mcp_info,
                    "heuristics": heuristics,
                    "e2e": {key: value for key, value in e2e.items() if key != "markdown"},
                    "all_ok": all_ok,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (report_dir / "scenario_report.md").write_text(
            render_report(index_info, mcp_info, heuristics, e2e, all_ok), encoding="utf-8"
        )
        print(f"Отчёты сохранены: {report_dir}\n")

    print("=" * 78)
    print(f"ИТОГ: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
