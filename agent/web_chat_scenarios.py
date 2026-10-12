#!/usr/bin/env python3
"""Day 27: сквозная проверка веб-чата с локальной LLM.

Проверяет критерии задачи:
1. веб-приложение отдаёт страницу чата (``GET /``);
2. запрос уходит в локальную модель и ответ возвращается (``POST /api/chat``);
3. диалог сохраняется между ходами, а история очищается (``/api/reset``);
4. приложение работает **без облачных моделей** (по умолчанию — офлайн-стаб).

Запуск::

    # детерминированная офлайн-проверка (без модели и сети)
    python3 -m agent.web_chat_scenarios --offline

    # проверка на живом локальном llama-server (нужна сборка Дня 26)
    python3 -m agent.web_chat_scenarios --live
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

from agent.config import load_config
from agent.local_llm import REPO_ROOT, LocalLLMError
from agent.web_chat import WebChatBackend, create_app

MATH_PROMPT = "Сколько будет 2+2? Ответь одним числом."
GREETING_PROMPT = "Привет!"


@dataclass
class CheckResult:
    """Результат одной проверки сценария."""

    name: str
    passed: bool
    detail: str
    latency_ms: float = 0.0


def _print_check(result: CheckResult) -> None:
    mark = "✓" if result.passed else "✗"
    latency = f" ({result.latency_ms:.0f} мс)" if result.latency_ms else ""
    print(f"   {mark} {result.name}{latency}")
    print(f"     {result.detail}")


def _preview(text: str, limit: int = 200) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _run_checks(backend: WebChatBackend) -> list[CheckResult]:
    """Прогоняет HTTP-проверки через тестовый клиент Flask (без сети)."""
    app = create_app(backend)
    client = app.test_client()
    results: list[CheckResult] = []

    started = time.perf_counter()
    page = client.get("/")
    elapsed = (time.perf_counter() - started) * 1000
    html = page.get_data(as_text=True)
    results.append(
        CheckResult(
            "Веб-приложение отдаёт страницу чата",
            page.status_code == 200 and "Локальный LLM-чат" in html,
            f"GET / → HTTP {page.status_code}, {len(html)} байт HTML",
            elapsed,
        )
    )

    status = client.get("/api/status").get_json()
    results.append(
        CheckResult(
            "Статус бэкенда: модель готова",
            bool(status.get("ready")),
            f"mode={status.get('mode')}, model={status.get('model')}, "
            f"endpoint={status.get('endpoint')}",
        )
    )

    started = time.perf_counter()
    first = client.post("/api/chat", json={"message": MATH_PROMPT}).get_json()
    elapsed = (time.perf_counter() - started) * 1000
    session_id = first.get("session_id")
    results.append(
        CheckResult(
            "Запрос уходит в локальную модель и возвращается ответ",
            bool(session_id) and "4" in (first.get("reply") or ""),
            f"POST /api/chat → session={session_id}, reply={_preview(first.get('reply'))}",
            elapsed,
        )
    )

    greeting = client.post(
        "/api/chat", json={"session_id": session_id, "message": GREETING_PROMPT}
    ).get_json()
    results.append(
        CheckResult(
            "Второй запрос в ту же сессию",
            bool(greeting.get("reply")),
            f"reply={_preview(greeting.get('reply'))}",
        )
    )

    history = client.get(f"/api/history?session_id={session_id}").get_json()
    messages = history.get("messages", [])
    results.append(
        CheckResult(
            "История диалога сохраняется между ходами",
            len(messages) == 4,
            f"сообщений в сессии: {len(messages)} "
            f"(user/assistant × 2)",
        )
    )

    reset = client.post("/api/reset", json={"session_id": session_id}).get_json()
    after = client.get(f"/api/history?session_id={session_id}").get_json()
    results.append(
        CheckResult(
            "История очищается командой /api/reset",
            reset.get("ok") is True and after.get("messages") == [],
            f"reset={reset.get('ok')}, сообщений после сброса: {len(after.get('messages', []))}",
        )
    )

    empty = client.post("/api/chat", json={"message": "   "})
    results.append(
        CheckResult(
            "Пустое сообщение отклоняется",
            empty.status_code == 400,
            f"HTTP {empty.status_code}: {_preview(empty.get_json().get('error'))}",
        )
    )

    return results


def _render_markdown(
    backend: WebChatBackend, results: list[CheckResult]
) -> str:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Day 27 — Веб-чат с локальной LLM: отчёт проверки",
        "",
        f"- Режим: {'офлайн (стаб)' if backend.offline else 'живой (llama.cpp)'}",
        f"- Модель: `{backend.model_label}`",
        f"- Endpoint модели: `{backend.endpoint}`",
        "",
        f"**Итог: {passed}/{len(results)} проверок пройдено.**",
        "",
        "| Проверка | Результат | Детали | Время, мс |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        mark = "✅" if result.passed else "❌"
        detail = _preview(result.detail, 1000).replace("|", "\\|")
        lines.append(f"| {result.name} | {mark} | {detail} | {result.latency_ms:.0f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 27: веб-чат с локальной LLM")
    parser.add_argument("--offline", action="store_true", help="Проверка на стабе (по умолчанию)")
    parser.add_argument("--live", action="store_true", help="Проверка на живом llama-server")
    parser.add_argument("--model", default=None, help="Путь к GGUF-модели")
    parser.add_argument("--llm-port", type=int, default=None, help="Порт llama-server")
    parser.add_argument("--report-dir", default="history/web_chat", help="Каталог для отчётов")
    args = parser.parse_args()

    offline = not args.live

    print("=" * 78)
    print("🌐 ДЕНЬ 27: ВЕБ-ЧАТ С ЛОКАЛЬНОЙ LLM")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if offline else 'живой llama.cpp'}\n")

    with tempfile.TemporaryDirectory(prefix="web_chat_") as history_dir:
        backend = WebChatBackend(
            load_config(),
            offline=offline,
            model_path=args.model,
            port=args.llm_port,
            history_dir=history_dir,
        )
        try:
            backend.start()
        except (LocalLLMError, OSError) as exc:
            print(f"✗ Ошибка запуска модели: {exc}", file=sys.stderr)
            return 1
        try:
            results = _run_checks(backend)
        finally:
            backend.stop()

    print("Проверки:")
    for result in results:
        _print_check(result)

    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(backend, results)
    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    payload = {
        "mode": "offline" if offline else "live",
        "endpoint": backend.endpoint,
        "checks": [asdict(r) for r in results],
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
    }
    (report_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    passed = sum(1 for r in results if r.passed)
    print(f"\nИтог: {passed}/{len(results)} проверок пройдено.")
    print(f"Отчёт: {report_dir / 'report.md'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
