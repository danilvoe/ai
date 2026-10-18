#!/usr/bin/env python3
"""День 34: сквозной сценарий ассистента файловых операций (офлайн-проверка).

Сценарий проверяет требования Дня 34 без модели и сети:

1. **Поиск использований** — по цели «найди, где используется X» ассистент сам
   ищет символ по многим файлам и читает контекст (≥2–3 файла);
2. **Генерация файла** — по цели «сгенерируй каталог MCP-серверов» ассистент
   читает исходники серверов и создаёт новый файл, показывая diff; повторный
   прогон даёт идентичный результат (воспроизводимость);
3. **Проверка инвариантов** — ассистент обходит файлы проекта и находит нарушения
   правил (на подготовленной песочнице с «посаженными» нарушениями);
4. **Обновление документации по изменениям в коде** — ассистент читает код и
   документ, обновляет документ и выдаёт diff по 2 файлам.

Все операции выполняются в песочнице ``history/day34/`` (артефакты и sandbox),
поэтому репозиторий не меняется. Отчёт сохраняется в ``history/file_assistant/``.

Запуск::

    python3 -m agent.file_scenarios --offline
    python3 -m agent.file_scenarios --offline --symbol build_support_runtime
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent.config import load_config
from agent.file_assistant import build_file_assistant

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WORK_DIR = "history/day34"
DEFAULT_REPORT_DIR = "history/file_assistant"
DEFAULT_SYMBOL = "build_support_runtime"

# --- Содержимое песочницы ----------------------------------------------------

BAD_MODULE = '''import os


def risky(value):
    try:
        return int(value)
    except:
        return 0


def debug():
    breakpoint()
    return 1
'''

GOOD_MODULE = '''"""Хороший модуль без нарушений правил."""


def safe(value: int) -> int:
    """Удваивает значение."""
    return value * 2
'''

SAMPLE_MODULE_BASE = '''"""Демонстрационный модуль для сценария обновления документации."""


def greet(name: str) -> str:
    """Приветствие."""
    return f"Привет, {name}!"
'''

SAMPLE_MODULE_CHANGED = SAMPLE_MODULE_BASE + '''

def farewell(name: str) -> str:
    """Прощание."""
    return f"Пока, {name}!"
'''

SAMPLE_DOC = '''# Пример документации

Здесь описаны не все функции модуля.
'''


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def prepare_sandbox(work_dir: Path) -> dict[str, Path]:
    """Создаёт (пересоздаёт) песочницу и возвращает её пути."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    invariants = work_dir / "sandbox" / "invariants"
    (invariants / "agent").mkdir(parents=True, exist_ok=True)
    (invariants / "agent" / "bad_module.py").write_text(BAD_MODULE, encoding="utf-8")
    (invariants / "agent" / "good_module.py").write_text(GOOD_MODULE, encoding="utf-8")

    update = work_dir / "sandbox" / "update_docs"
    baseline = update / "baseline"
    work = update / "work"
    for target in (baseline, work):
        (target / "agent").mkdir(parents=True, exist_ok=True)
        (target / "docs").mkdir(parents=True, exist_ok=True)
        (target / "agent" / "sample_module.py").write_text(SAMPLE_MODULE_BASE, encoding="utf-8")
        (target / "docs" / "sample.md").write_text(SAMPLE_DOC, encoding="utf-8")
    # В рабочей копии код «изменился» — добавлена новая функция.
    (work / "agent" / "sample_module.py").write_text(SAMPLE_MODULE_CHANGED, encoding="utf-8")

    artifacts = work_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return {"root": work_dir, "invariants": invariants, "baseline": baseline,
            "work": work, "artifacts": artifacts}


def check_find_usages(symbol: str, config: dict) -> dict:
    """Сценарий 1: поиск использований символа по многим файлам (чтение)."""
    print("=" * 78)
    print("1. Поиск использований: ассистент сам ищет символ по файлам")
    print("=" * 78)
    assistant = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=False)
    result = assistant.run(f"найди, где используется {symbol}", symbol=symbol)
    files = result.files_touched
    ok = (
        not result.error
        and result.task_type == "find_usages"
        and len(files) >= 2
        and len(result.findings) >= 2
    )
    print(f"Символ: {symbol}; файлов затронуто: {len(files)}; находок: {len(result.findings)}")
    for path in files[:5]:
        print(f"  - {path}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "task_type": result.task_type, "symbol": symbol,
        "files": files, "findings": len(result.findings),
        "content": result.content,
    }


def check_generate(artifacts: Path, config: dict) -> dict:
    """Сценарий 2: генерация нового файла из исходников (запись + diff)."""
    print("=" * 78)
    print("2. Генерация файла: ассистент читает исходники и создаёт новый файл")
    print("=" * 78)
    target = (artifacts / "mcp-catalog.md").relative_to(PROJECT_ROOT).as_posix()
    assistant = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=True)
    first = assistant.run("сгенерируй каталог MCP-серверов", kind="catalog", target=target)
    target_path = PROJECT_ROOT / target
    first_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    # Повторный прогон — проверка воспроизводимости.
    assistant.run("сгенерируй каталог MCP-серверов", kind="catalog", target=target)
    second_content = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    reproducible = bool(first_content) and first_content == second_content
    sources = [path for path in first.files_touched if path != target]
    ok = (
        not first.error
        and target_path.exists()
        and len(sources) >= 3
        and bool(first.artifacts and first.artifacts[0]["diff"])
        and reproducible
    )
    print(f"Файл создан: {target_path.exists()}; исходников прочитано: {len(sources)}")
    print(f"Воспроизводимость (два прогона идентичны): {reproducible}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "target": target, "sources": sources,
        "diff_chars": len(first.artifacts[0]["diff"]) if first.artifacts else 0,
        "reproducible": reproducible,
    }


def check_invariants(invariants_dir: Path, config: dict) -> dict:
    """Сценарий 3: проверка соответствия файлов правилам (инварианты)."""
    print("=" * 78)
    print("3. Проверка инвариантов: ассистент обходит файлы и находит нарушения")
    print("=" * 78)
    # 3.1. Реальный проект — ожидаем чистый результат.
    project = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=False)
    project_result = project.run("проверь соответствие правилам проекта")
    # 3.2. Песочница с «посаженными» нарушениями.
    sandbox = build_file_assistant(config=config, offline=True, workspace_root=invariants_dir, apply=False)
    sandbox_result = sandbox.run("проверь соответствие правилам проекта")
    kinds = {finding.message for finding in sandbox_result.findings}
    detected = {
        "bare_except": any("голый except" in message for message in kinds),
        "breakpoint": any("breakpoint" in message for message in kinds),
        "docstring": any("докстринг" in message for message in kinds),
    }
    ok = (
        not project_result.error
        and not sandbox_result.error
        and len(sandbox_result.findings) >= 3
        and all(detected.values())
    )
    print(f"Реальный проект: проверено {len(project_result.files_touched)} файлов, "
          f"нарушений {len(project_result.findings)}")
    print(f"Песочница с нарушениями: найдено {len(sandbox_result.findings)} нарушений "
          f"(bare except={detected['bare_except']}, breakpoint={detected['breakpoint']}, "
          f"docstring={detected['docstring']})")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "project_checked": len(project_result.files_touched),
        "project_findings": len(project_result.findings),
        "sandbox_findings": len(sandbox_result.findings),
        "detected": detected,
        "findings": [finding.to_dict() for finding in sandbox_result.findings],
    }


def check_update_docs(baseline: Path, work: Path, config: dict) -> dict:
    """Сценарий 4: обновление документации по изменениям в коде + diff."""
    print("=" * 78)
    print("4. Обновление документации по коду: чтение кода + правка документа + diff")
    print("=" * 78)
    assistant = build_file_assistant(config=config, offline=True, workspace_root=work, apply=True)
    result = assistant.run(
        "обнови документацию по коду",
        doc_path="docs/sample.md",
        source_paths=["agent/sample_module.py"],
    )
    doc_path = work / "docs" / "sample.md"
    doc_text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    doc_updated = "farewell" in doc_text and "greet" in doc_text
    diff_chars = len(result.artifacts[0]["diff"]) if result.artifacts else 0

    # Ассистент сам готовит diff изменений относительно baseline (код + док).
    diff_result = assistant.run(
        "подготовь diff изменений", task_type="prepare_diff", baseline_root=str(baseline),
    )
    changed = diff_result.artifacts[0]["diff"] if diff_result.artifacts else ""
    changed_files = set(diff_result.files_touched)
    ok = (
        not result.error
        and not diff_result.error
        and doc_updated
        and len(result.files_touched) >= 2
        and len(changed_files) >= 2
        and diff_chars > 0
    )
    print(f"Документ обновлён: {doc_updated}; файлов затронуто: {len(result.files_touched)}")
    print(f"Diff по baseline: изменённых файлов {len(changed_files)} "
          f"({', '.join(sorted(changed_files)) or '—'})")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "doc_updated": doc_updated,
        "files_touched": result.files_touched,
        "changed_files": sorted(changed_files),
        "diff_chars": diff_chars,
        "diff": changed[:2000],
    }


def render_report(scenarios: dict, all_ok: bool) -> str:
    lines = [
        "# День 34: ассистент файловых операций — отчёт офлайн-проверки",
        "",
        f"Сформировано: {now_iso()}",
        "",
        "## 1. Поиск использований",
        "",
        f"- Символ: `{scenarios['find_usages'].get('symbol')}`",
        f"- Файлов затронуто: {len(scenarios['find_usages'].get('files') or [])}",
        f"- Находок: {scenarios['find_usages'].get('findings')}",
        f"- Итог: {'OK' if scenarios['find_usages'].get('ok') else 'FAIL'}",
        "",
        "## 2. Генерация файла",
        "",
        f"- Файл: `{scenarios['generate'].get('target')}`",
        f"- Исходников прочитано: {len(scenarios['generate'].get('sources') or [])}",
        f"- Воспроизводимость: {scenarios['generate'].get('reproducible')}",
        f"- Итог: {'OK' if scenarios['generate'].get('ok') else 'FAIL'}",
        "",
        "## 3. Проверка инвариантов",
        "",
        f"- Реальный проект: файлов {scenarios['invariants'].get('project_checked')}, "
        f"нарушений {scenarios['invariants'].get('project_findings')}",
        f"- Песочница: найдено {scenarios['invariants'].get('sandbox_findings')} нарушений",
        f"- Итог: {'OK' if scenarios['invariants'].get('ok') else 'FAIL'}",
        "",
        "## 4. Обновление документации по коду",
        "",
        f"- Файлов затронуто: {len(scenarios['update_docs'].get('files_touched') or [])}",
        f"- Изменённых файлов в diff: {', '.join(scenarios['update_docs'].get('changed_files') or []) or '—'}",
        f"- Итог: {'OK' if scenarios['update_docs'].get('ok') else 'FAIL'}",
        "",
        f"**Общий итог: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 34: сквозная проверка ассистента файловых операций (офлайн)"
    )
    parser.add_argument("--offline", action="store_true", help="офлайн-режим (по умолчанию)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="символ для поиска использований")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="каталог песочницы/артефактов")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--skip-save", action="store_true", help="не сохранять отчёт")
    args = parser.parse_args()

    print("=" * 78)
    print("🗂️  ДЕНЬ 34: АССИСТЕНТ ФАЙЛОВЫХ ОПЕРАЦИЙ (MCP: чтение, поиск, запись)")
    print("=" * 78)

    config: dict = {}
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # noqa: BLE001 — конфиг опционален
        print(f"ℹ️  Конфиг не загружен ({type(exc).__name__}); значения по умолчанию.")

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = PROJECT_ROOT / work_dir
    paths = prepare_sandbox(work_dir)
    print(f"Песочница: {paths['root']}\n")

    scenarios = {
        "find_usages": check_find_usages(args.symbol, config),
        "generate": check_generate(paths["artifacts"], config),
        "invariants": check_invariants(paths["invariants"], config),
        "update_docs": check_update_docs(paths["baseline"], paths["work"], config),
    }
    all_ok = all(item["ok"] for item in scenarios.values())

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "scenario_report.md").write_text(
            render_report(scenarios, all_ok), encoding="utf-8"
        )
        (report_dir / "file_results.json").write_text(
            json.dumps({"generated_at": now_iso(), "scenarios": scenarios, "all_ok": all_ok},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Отчёты сохранены: {report_dir}\n")

    print("=" * 78)
    print(f"ИТОГ: {'все проверки пройдены ✅' if all_ok else 'есть ошибки ❌'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
