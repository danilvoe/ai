#!/usr/bin/env python3
"""Day 26: сквозной сценарий запуска локальной LLM через llama.cpp.

Проверяет критерии задачи:
1. модель запускается локально (llama-server поднимается и отвечает на /health);
2. к ней можно обратиться через CLI (llama-cli отвечает на запрос);
3. модель отвечает на простой запрос (через OpenAI-совместимый /v1 API).

Запуск::

    # детерминированная офлайн-проверка (без сборки, модели и сети)
    python3 -m agent.local_llm_scenarios --offline

    # реальная проверка локальной модели
    python3 -m agent.local_llm_scenarios

    # свой запрос, порт и путь к модели
    python3 -m agent.local_llm_scenarios --prompt "Назови столицу Франции" --port 8090
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

from agent.config import load_config
from agent.local_llm import (
    REPO_ROOT,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMServer,
    StubLocalLLM,
)

DEFAULT_PROMPT = "Сколько будет 2+2? Ответь одним числом."


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


def _run_offline(config: LocalLLMConfig, prompt: str) -> list[CheckResult]:
    """Детерминированная проверка конвейера на стабе (без llama.cpp)."""
    stub = StubLocalLLM(config.model_label)
    results: list[CheckResult] = []

    started = time.perf_counter()
    answer = stub.answer(prompt)
    elapsed = (time.perf_counter() - started) * 1000
    results.append(CheckResult(
        "Модель запускается локально (стаб)",
        True,
        f"StubLocalLLM инициализирован, endpoint {stub.base_url}",
    ))
    results.append(CheckResult(
        "Обращение через CLI (стаб)",
        True,
        "llama-cli имитируется детерминированным ответом",
        elapsed,
    ))
    results.append(CheckResult(
        "Модель отвечает на простой запрос (стаб)",
        bool(answer.strip()),
        f"Запрос: {prompt!r} → ответ: {answer!r}",
        elapsed,
    ))
    return results


def _run_live(config: LocalLLMConfig, prompt: str, with_server: bool) -> list[CheckResult]:
    """Реальная проверка: бинарники, модель, CLI и HTTP-сервер."""
    results: list[CheckResult] = []

    cli = config.cli_binary()
    server_bin = config.server_binary()
    results.append(CheckResult(
        "Бинарники llama.cpp найдены",
        cli is not None and server_bin is not None,
        f"llama-cli={cli or '—'}, llama-server={server_bin or '—'}",
    ))
    results.append(CheckResult(
        "Файл модели найден",
        config.model_file.exists(),
        f"{config.model_file}",
    ))
    if not config.model_file.exists():
        return results

    server = LocalLLMServer(config)

    # 1. CLI-проверка.
    if cli is not None:
        started = time.perf_counter()
        try:
            cli_answer = server.run_cli_prompt(prompt)
            elapsed = (time.perf_counter() - started) * 1000
            results.append(CheckResult(
                "К модели можно обратиться через CLI (llama-cli)",
                bool(cli_answer.strip()),
                f"Ответ CLI: {_preview(cli_answer, 600)}",
                elapsed,
            ))
        except LocalLLMError as exc:
            results.append(CheckResult(
                "К модели можно обратиться через CLI (llama-cli)",
                False,
                str(exc),
            ))

    # 2. Сервер + HTTP.
    if with_server:
        started = time.perf_counter()
        try:
            server.start()
            startup_ms = (time.perf_counter() - started) * 1000
            models = server.list_models()
            results.append(CheckResult(
                "Модель запускается локально (llama-server)",
                bool(models),
                f"{server.base_url}, модели: {models}",
                startup_ms,
            ))
            started = time.perf_counter()
            completion = server.complete(prompt, max_tokens=64)
            elapsed = (time.perf_counter() - started) * 1000
            results.append(CheckResult(
                "Модель отвечает на простой запрос (OpenAI API)",
                bool(completion.content.strip()),
                f"Запрос: {prompt!r} → ответ: {_preview(completion.content, 600)} "
                f"[{completion.total_tokens or '?'} токенов]",
                elapsed,
            ))
        except (LocalLLMError, OSError, ValueError) as exc:
            results.append(CheckResult(
                "Модель запускается локально (llama-server)",
                False,
                str(exc),
            ))
        finally:
            server.stop()

    return results


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _render_markdown(
    config: LocalLLMConfig,
    prompt: str,
    offline: bool,
    results: list[CheckResult],
) -> str:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# Day 26 — Локальная LLM через llama.cpp: отчёт проверки",
        "",
        f"- Режим: {'офлайн (стаб)' if offline else 'живой (llama.cpp)'}",
        f"- Модель: `{config.model_file}`",
        f"- llama.cpp: `{config.llama_dir_path}`",
        f"- Endpoint: `http://{config.host}:{config.port}/v1`",
        f"- Потоки: {config.threads}, ctx: {config.ctx_size}, n_gpu_layers: {config.n_gpu_layers}",
        f"- Тестовый запрос: {prompt!r}",
        "",
        f"**Итог: {passed}/{len(results)} проверок пройдено.**",
        "",
        "| Проверка | Результат | Детали | Время, мс |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        mark = "✅" if result.passed else "❌"
        detail = _preview(result.detail, 1000).replace("|", "\\|")
        lines.append(
            f"| {result.name} | {mark} | {detail} | {result.latency_ms:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 26: запуск локальной LLM (llama.cpp)")
    parser.add_argument("--offline", action="store_true", help="Проверка на стабе без llama.cpp")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Простой тестовый запрос")
    parser.add_argument("--llama-dir", default=None, help="Каталог сборки llama.cpp")
    parser.add_argument("--model", default=None, help="Путь к GGUF-модели")
    parser.add_argument("--port", type=int, default=None, help="Порт llama-server")
    parser.add_argument("--no-server", action="store_true", help="Не поднимать llama-server (только CLI)")
    parser.add_argument("--report-dir", default="history/local_llm", help="Каталог для отчётов")
    args = parser.parse_args()

    config = LocalLLMConfig.from_config(load_config())
    if args.llama_dir:
        config.llama_dir = args.llama_dir
    if args.model:
        config.model_path = args.model
    if args.port:
        config.port = args.port

    print("=" * 78)
    print("🔌 ДЕНЬ 26: ЛОКАЛЬНАЯ LLM ЧЕРЕЗ LLAMA.CPP")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой llama.cpp'}")
    print(f"Модель: {config.model_file}")
    print(f"Запрос: {args.prompt!r}\n")

    try:
        if args.offline:
            results = _run_offline(config, args.prompt)
        else:
            results = _run_live(config, args.prompt, with_server=not args.no_server)
    except LocalLLMError as exc:
        print(f"✗ Ошибка: {exc}", file=sys.stderr)
        return 1

    print("Проверки:")
    for result in results:
        _print_check(result)

    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(config, args.prompt, args.offline, results)
    (report_dir / "report.md").write_text(markdown, encoding="utf-8")
    payload = {
        "mode": "offline" if args.offline else "live",
        "prompt": args.prompt,
        "model": str(config.model_file),
        "endpoint": f"http://{config.host}:{config.port}/v1",
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
