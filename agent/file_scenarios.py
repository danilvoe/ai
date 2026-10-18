#!/usr/bin/env python3
"""День 34: сквозной сценарий ассистента файловых операций (офлайн-проверка).

Сценарий проверяет требования Дня 34 без модели и сети:

1. **Поиск использований** — по цели «найди, где используется X» ассистент сам
   ищет символ по многим файлам и читает контекст (≥2–3 файла);
2. **Генерация файла** — по цели «сгенерируй каталог MCP-серверов» ассистент
   читает исходники серверов, показывает diff (dry-run) и сохраняет файл
   (apply); повторный прогон даёт идентичный результат (воспроизводимость);
3. **Проверка инвариантов** — ассистент обходит файлы проекта и находит нарушения
   правил (на подготовленной песочнице с «посаженными» нарушениями);
4. **Обновление документации по изменениям в коде** — ассистент читает код и
   документ, обновляет документ и выдаёт diff по 2 файлам.

Каждый сценарий дополнительно измеряет «минимальный результат»:
``files_count`` (сколько файлов затронуто), ``change_mode`` (``saved``/``diff``/
``read-only``) и ``reproducible`` (два независимых прогона дают одинаковый
результат). Итоговый блок «Минимальный результат» проверяет все три требования.

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


def _signature(result) -> str:
    """Стабильная подпись результата для проверки воспроизводимости."""
    return json.dumps(
        {
            "task": result.task_type,
            "files": sorted(result.files_touched),
            "findings": [
                [finding.kind, finding.path, finding.line, finding.message]
                for finding in result.findings
            ],
            "content": result.content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


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
    (baseline / "agent").mkdir(parents=True, exist_ok=True)
    (baseline / "docs").mkdir(parents=True, exist_ok=True)
    (baseline / "agent" / "sample_module.py").write_text(SAMPLE_MODULE_BASE, encoding="utf-8")
    (baseline / "docs" / "sample.md").write_text(SAMPLE_DOC, encoding="utf-8")
    work = update / "work"
    reset_work(baseline, work)

    artifacts = work_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return {"root": work_dir, "invariants": invariants, "baseline": baseline,
            "work": work, "artifacts": artifacts}


def reset_work(baseline: Path, work: Path) -> None:
    """Восстанавливает рабочую копию из baseline и вносит «изменение в коде»."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(baseline, work)
    (work / "agent" / "sample_module.py").write_text(SAMPLE_MODULE_CHANGED, encoding="utf-8")


def check_find_usages(symbol: str, config: dict) -> dict:
    """Сценарий 1: поиск использований символа по многим файлам (чтение)."""
    print("=" * 78)
    print("1. Поиск использований: ассистент сам ищет символ по файлам")
    print("=" * 78)
    assistant = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=False)
    goal = f"найди, где используется {symbol}"
    first = assistant.run(goal, symbol=symbol)
    second = assistant.run(goal, symbol=symbol)
    reproducible = _signature(first) == _signature(second)
    files = first.files_touched
    ok = (
        not first.error
        and first.task_type == "find_usages"
        and len(files) >= 2
        and len(first.findings) >= 2
        and reproducible
    )
    print(f"Символ: {symbol}; файлов затронуто: {len(files)}; находок: {len(first.findings)}")
    for path in files[:6]:
        print(f"  - {path}")
    print(f"Воспроизводимость: {reproducible}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "task_type": first.task_type, "symbol": symbol,
        "files": files, "files_count": len(files), "findings": len(first.findings),
        "change_mode": "read-only", "reproducible": reproducible,
        "content": first.content,
    }


def check_generate(artifacts: Path, config: dict) -> dict:
    """Сценарий 2: генерация файла — diff (dry-run) + сохранение (apply)."""
    print("=" * 78)
    print("2. Генерация файла: diff при dry-run и сохранение при apply")
    print("=" * 78)
    target = (artifacts / "mcp-catalog.md").relative_to(PROJECT_ROOT).as_posix()
    goal = "сгенерируй каталог MCP-серверов"
    target_path = PROJECT_ROOT / target
    if target_path.exists():
        target_path.unlink()

    dry_assistant = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=False)
    dry = dry_assistant.run(goal, kind="catalog", target=target)
    dry_diff = dry.artifacts[0]["diff"] if dry.artifacts else ""
    dry_again = dry_assistant.run(goal, kind="catalog", target=target)
    dry_diff_again = dry_again.artifacts[0]["diff"] if dry_again.artifacts else ""
    reproducible = bool(dry_diff) and dry_diff == dry_diff_again

    apply_assistant = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=True)
    saved = apply_assistant.run(goal, kind="catalog", target=target)
    saved_ok = target_path.exists()
    sources = [path for path in saved.files_touched if path != target]
    ok = (
        not dry.error and not saved.error
        and bool(dry_diff)
        and saved_ok
        and len(sources) >= 3
        and reproducible
    )
    print(f"Diff (dry-run): {len(dry_diff)} символов; файл сохранён: {saved_ok}")
    print(f"Исходников прочитано: {len(sources)}; воспроизводимость: {reproducible}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "target": target, "sources": sources, "files_count": len(sources),
        "change_mode": "saved+diff", "reproducible": reproducible,
        "saved": saved_ok, "diff_chars": len(dry_diff),
    }


def check_invariants(invariants_dir: Path, config: dict) -> dict:
    """Сценарий 3: проверка соответствия файлов правилам (инварианты)."""
    print("=" * 78)
    print("3. Проверка инвариантов: ассистент обходит файлы и находит нарушения")
    print("=" * 78)
    goal = "проверь соответствие правилам проекта"
    # 3.1. Реальный проект — один полный проход (много файлов).
    project = build_file_assistant(config=config, offline=True, workspace_root=PROJECT_ROOT, apply=False)
    project_result = project.run(goal)
    # 3.2. Песочница с «посаженными» нарушениями — два прогона (воспроизводимость).
    sandbox = build_file_assistant(config=config, offline=True, workspace_root=invariants_dir, apply=False)
    sandbox_first = sandbox.run(goal)
    sandbox_second = sandbox.run(goal)
    reproducible = _signature(sandbox_first) == _signature(sandbox_second)
    kinds = {finding.message for finding in sandbox_first.findings}
    detected = {
        "bare_except": any("голый except" in message for message in kinds),
        "breakpoint": any("breakpoint" in message for message in kinds),
        "docstring": any("докстринг" in message for message in kinds),
    }
    ok = (
        not project_result.error
        and not sandbox_first.error
        and len(project_result.files_touched) >= 2
        and len(sandbox_first.findings) >= 3
        and all(detected.values())
        and reproducible
    )
    print(f"Реальный проект: проверено {len(project_result.files_touched)} файлов, "
          f"нарушений {len(project_result.findings)}")
    print(f"Песочница с нарушениями: найдено {len(sandbox_first.findings)} нарушений "
          f"(bare except={detected['bare_except']}, breakpoint={detected['breakpoint']}, "
          f"docstring={detected['docstring']})")
    print(f"Воспроизводимость: {reproducible}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok,
        "files_count": len(project_result.files_touched),
        "change_mode": "read-only",
        "reproducible": reproducible,
        "project_checked": len(project_result.files_touched),
        "project_findings": len(project_result.findings),
        "sandbox_findings": len(sandbox_first.findings),
        "detected": detected,
        "findings": [finding.to_dict() for finding in sandbox_first.findings],
    }


def check_update_docs(baseline: Path, work: Path, config: dict) -> dict:
    """Сценарий 4: обновление документации по коду + diff по 2 файлам."""
    print("=" * 78)
    print("4. Обновление документации по коду: чтение кода + правка документа + diff")
    print("=" * 78)
    assistant = build_file_assistant(config=config, offline=True, workspace_root=work, apply=True)
    doc_path = work / "docs" / "sample.md"
    goal = "обнови документацию по коду"

    reset_work(baseline, work)
    assistant.run(goal, doc_path="docs/sample.md", source_paths=["agent/sample_module.py"])
    doc_first = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

    reset_work(baseline, work)
    second = assistant.run(goal, doc_path="docs/sample.md", source_paths=["agent/sample_module.py"])
    doc_second = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

    reproducible = bool(doc_first) and doc_first == doc_second
    doc_updated = "farewell" in doc_second and "greet" in doc_second
    diff_chars = len(second.artifacts[0]["diff"]) if second.artifacts else 0

    # Ассистент сам готовит diff изменений относительно baseline (код + документ).
    diff_result = assistant.run("подготовь diff изменений", task_type="prepare_diff", baseline_root=str(baseline))
    changed_files = set(diff_result.files_touched)
    ok = (
        not second.error
        and not diff_result.error
        and doc_updated
        and len(second.files_touched) >= 2
        and len(changed_files) >= 2
        and diff_chars > 0
        and reproducible
    )
    print(f"Документ обновлён: {doc_updated}; файлов затронуто: {len(second.files_touched)}")
    print(f"Diff по baseline: изменённых файлов {len(changed_files)} "
          f"({', '.join(sorted(changed_files)) or '—'})")
    print(f"Воспроизводимость: {reproducible}")
    print(f"Результат: {'OK' if ok else 'FAIL'}\n")
    return {
        "ok": ok, "doc_updated": doc_updated,
        "files_touched": second.files_touched, "files_count": len(second.files_touched),
        "change_mode": "saved+diff", "reproducible": reproducible,
        "changed_files": sorted(changed_files), "diff_chars": diff_chars,
        "diff": diff_result.artifacts[0]["diff"][:2000] if diff_result.artifacts else "",
    }


def summarize(scenarios: dict) -> dict:
    """Проверяет «минимальный результат» задания по всем сценариям."""
    files_ok = all(item.get("files_count", 0) >= 2 for item in scenarios.values())
    modes = {item.get("change_mode", "") for item in scenarios.values()}
    saved_ok = any("saved" in mode for mode in modes)
    diff_ok = any("diff" in mode for mode in modes)
    reproducible_ok = all(item.get("reproducible") for item in scenarios.values())
    return {
        "files_ok": files_ok,
        "saved_ok": saved_ok,
        "diff_ok": diff_ok,
        "reproducible_ok": reproducible_ok,
        "ok": files_ok and saved_ok and diff_ok and reproducible_ok,
    }


def render_report(scenarios: dict, minimal: dict, all_ok: bool) -> str:
    lines = [
        "# День 34: ассистент файловых операций — отчёт офлайн-проверки",
        "",
        f"Сформировано: {now_iso()}",
        "",
        "## Минимальный результат задания",
        "",
        "| Требование | Проверка | Итог |",
        "| --- | --- | --- |",
        f"| Ассистент работает минимум с 2–3 файлами | по каждому сценарию `files_count ≥ 2` | "
        f"{'OK' if minimal['files_ok'] else 'FAIL'} |",
        f"| Изменения сохраняются или выводятся как diff | есть `saved` и `diff` | "
        f"{'OK' if minimal['saved_ok'] and minimal['diff_ok'] else 'FAIL'} |",
        f"| Результат можно воспроизвести повторно | все сценарии `reproducible` | "
        f"{'OK' if minimal['reproducible_ok'] else 'FAIL'} |",
        "",
        "### Метрики по сценариям",
        "",
        "| Сценарий | Файлов | Режим изменений | Воспроизводимо | Итог |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, key in (
        ("1. Поиск использований", "find_usages"),
        ("2. Генерация файла", "generate"),
        ("3. Проверка инвариантов", "invariants"),
        ("4. Обновление документации", "update_docs"),
    ):
        item = scenarios[key]
        lines.append(
            f"| {name} | {item.get('files_count')} | {item.get('change_mode')} | "
            f"{item.get('reproducible')} | {'OK' if item.get('ok') else 'FAIL'} |"
        )
    lines += [
        "",
        "## Детали сценариев",
        "",
        f"- **1. Поиск использований.** Символ `{scenarios['find_usages'].get('symbol')}`, "
        f"файлов {scenarios['find_usages'].get('files_count')}, "
        f"находок {scenarios['find_usages'].get('findings')}.",
        f"- **2. Генерация файла.** `{scenarios['generate'].get('target')}`, "
        f"исходников {scenarios['generate'].get('files_count')}, "
        f"diff {scenarios['generate'].get('diff_chars')} символов, "
        f"сохранён {scenarios['generate'].get('saved')}.",
        f"- **3. Проверка инвариантов.** Проверено "
        f"{scenarios['invariants'].get('project_checked')} файлов; в песочнице найдено "
        f"{scenarios['invariants'].get('sandbox_findings')} нарушений.",
        f"- **4. Обновление документации.** Файлов затронуто "
        f"{scenarios['update_docs'].get('files_count')}, изменённых в diff "
        f"{', '.join(scenarios['update_docs'].get('changed_files') or []) or '—'}.",
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
    minimal = summarize(scenarios)
    all_ok = all(item["ok"] for item in scenarios.values()) and minimal["ok"]

    print("=" * 78)
    print("МИНИМАЛЬНЫЙ РЕЗУЛЬТАТ ЗАДАНИЯ")
    print("=" * 78)
    print(f"  2–3 файла на сценарий : {'OK' if minimal['files_ok'] else 'FAIL'}")
    print(f"  изменения saved/diff  : {'OK' if minimal['saved_ok'] and minimal['diff_ok'] else 'FAIL'}")
    print(f"  воспроизводимость     : {'OK' if minimal['reproducible_ok'] else 'FAIL'}")
    print()

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "scenario_report.md").write_text(
            render_report(scenarios, minimal, all_ok), encoding="utf-8"
        )
        (report_dir / "file_results.json").write_text(
            json.dumps({"generated_at": now_iso(), "scenarios": scenarios,
                        "minimal": minimal, "all_ok": all_ok},
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
