#!/usr/bin/env python3
"""Day 29: сквозной сценарий оптимизации локальной LLM под задачу (рецепты).

Что проверяет и показывает:

1. **Оптимизацию по трём осям**: параметры генерации (``temperature``,
   ``max_tokens``, окно контекста), квантование (Q4_K_M против Q2_K) и
   prompt-шаблон под кейс (базовый против структурированного рецептурного).
2. **Сравнение «до/после»** на контрольных вопросах по качеству (покрытие
   фактов, доля ответов со ссылками), скорости (задержка, токены/с) и ресурсам
   (размер модели, ориентир RAM, пик RSS).
3. **Настройки в веб-интерфейсе** — наличие контролов параметров/модели и
   применение ``ctx_size``/варианта через ``POST /api/settings``.

Запуск::

    # детерминированная офлайн-проверка (без модели и сети)
    python3 -m agent.optimization_scenarios --offline

    # живой прогон по скачанным вариантам (нужна сборка Дня 26)
    python3 -m agent.optimization_scenarios
    python3 -m agent.optimization_scenarios --variant Q2_K --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.config import load_config
from agent.indexing import HashingEmbedder
from agent.local_llm import REPO_ROOT, LocalLLMConfig, LocalLLMError, LocalLLMServer
from agent.local_rag import LocalRagStubClient
from agent.optimization import (
    ConfigurationSpec,
    GenerationParams,
    OptimizationConfiguration,
    OptimizationStubClient,
    compare_optimizations,
    find_template,
    find_variant,
    measure_peak_rss_mb,
    parse_configuration_specs,
    parse_presets,
    parse_variants,
    render_optimization_markdown,
)
from agent.rag import CONTROL_QUESTIONS, RagRetriever, load_retriever
from agent.web_chat import WebChatBackend, create_app


@dataclass
class CheckResult:
    """Результат одной HTTP-проверки веб-интерфейса."""

    name: str
    passed: bool
    detail: str
    latency_ms: float = 0.0


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _print_check(result: CheckResult) -> None:
    mark = "✓" if result.passed else "✗"
    latency = f" ({result.latency_ms:.0f} мс)" if result.latency_ms else ""
    print(f"   {mark} {result.name}{latency}")
    print(f"     {result.detail}")


def _load_retriever(config: dict, args) -> RagRetriever:
    idx_cfg = config.get("indexing", {}) or {}
    rag_cfg = config.get("local_rag", {}) or {}
    embedder = HashingEmbedder(dim=int(idx_cfg.get("embedding_dim", 256)))
    return load_retriever(
        index_dir=args.index_dir or idx_cfg.get("index_dir", "history/index"),
        dataset=args.dataset or rag_cfg.get("dataset", "recipt_all"),
        strategy=args.strategy or rag_cfg.get("strategy", "structural"),
        embedder=embedder,
        top_k=args.top_k,
    )


def _spec_to_config(
    spec: ConfigurationSpec,
    presets: dict[str, GenerationParams],
    variants,
    client,
    model_label: str,
) -> OptimizationConfiguration:
    params = presets.get(spec.preset, presets["baseline"])
    template = find_template(spec.prompt_template)
    variant = find_variant(variants, spec.variant)
    return OptimizationConfiguration(
        name=spec.name,
        label=spec.label or spec.name,
        params=params,
        template=template,
        variant=variant,
        client=client,
        model_label=model_label,
    )


def _offline_configurations(
    config: dict, args
) -> list[OptimizationConfiguration]:
    """Детерминированные конфигурации на офлайн-стабах."""
    presets = parse_presets(config)
    variants = parse_variants(config)
    specs = parse_configuration_specs(config)
    if args.variant:
        specs = [
            ConfigurationSpec(
                name=s.name, label=s.label, preset=s.preset,
                prompt_template=s.prompt_template, variant=args.variant,
            )
            for s in specs
        ]
    result: list[OptimizationConfiguration] = []
    for spec in specs:
        template = find_template(spec.prompt_template)
        variant = find_variant(variants, spec.variant)
        params = presets.get(spec.preset, presets["baseline"])
        client = OptimizationStubClient(
            f"{variant.label if variant else 'local'}-stub",
            variant=variant,
            params=params,
            optimized=template.name != "baseline",
        )
        result.append(
            _spec_to_config(
                spec, presets, variants, client,
                variant.label if variant else "local",
            )
        )
    return result


class _LiveServers:
    """Менеджер живых llama-server'ов: один процесс на вариант (квантование).

    Разные варианты поднимаются на разных портах, чтобы работали одновременно;
    конфигурации с одним и тем же вариантом переиспользуют уже запущенный сервер.
    """

    def __init__(self) -> None:
        self.servers: list[LocalLLMServer] = []
        self._by_variant: dict[str, LocalLLMServer] = {}

    def build(
        self,
        spec: ConfigurationSpec,
        presets: dict[str, GenerationParams],
        variants,
        config: dict,
    ) -> OptimizationConfiguration | None:
        variant = find_variant(variants, spec.variant)
        if variant is None:
            return None
        if not variant.available():
            print(
                f"   ⚠ Вариант «{variant.label}» пропущен: нет файла {variant.path}"
            )
            return None
        server = self._by_variant.get(variant.label)
        if server is None:
            local = LocalLLMConfig.from_config(config)
            local.model_path = variant.model_path
            local.model_label = variant.label
            if variant.ctx_size:
                local.ctx_size = int(variant.ctx_size)
            # Каждому варианту — свой порт, чтобы серверы не конфликтовали.
            local.port = int(local.port) + len(self._by_variant)
            server = LocalLLMServer(local)
            try:
                server.start()
            except (LocalLLMError, OSError) as exc:
                print(f"   ⚠ Вариант «{variant.label}» не запустился: {exc}")
                return None
            self._by_variant[variant.label] = server
            self.servers.append(server)
        return _spec_to_config(
            spec, presets, variants, server.build_client(), variant.label
        )

    def stop(self) -> None:
        for server in self.servers:
            server.stop()
        self.servers = []
        self._by_variant = {}


def _run_web_checks(retriever: RagRetriever, config: dict) -> list[CheckResult]:
    """HTTP-проверки контролов оптимизации в веб-чате."""
    results: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="optimization_web_") as history_dir:
        backend = WebChatBackend(
            config,
            offline=True,
            retriever=retriever,
            client=OptimizationStubClient("local-opt-stub"),
            cloud_client=LocalRagStubClient("cloud-opt-stub"),
            history_dir=history_dir,
        )
        backend.start()
        try:
            app = create_app(backend)
            client = app.test_client()
            html = client.get("/").get_data(as_text=True)
            controls = [
                'id="temperature"',
                'id="max_tokens"',
                'id="context_window"',
                'id="variant"',
                'id="preset"',
                'id="template"',
                'id="ctx_size"',
            ]
            missing = [c for c in controls if c not in html]
            results.append(
                CheckResult(
                    "Веб-страница содержит настройки параметров и модели",
                    not missing and "Параметры оптимизации" in html,
                    f"GET / → контролов найдено "
                    f"{len(controls) - len(missing)}/{len(controls)}, "
                    f"нет: {missing or '—'}",
                )
            )

            status = client.get("/api/status").get_json()
            results.append(
                CheckResult(
                    "Статус содержит пресеты и варианты моделей",
                    bool(status.get("presets"))
                    and len(status.get("variants") or []) >= 2
                    and bool(status.get("prompt_templates")),
                    f"presets={list((status.get('presets') or {}).keys())}, "
                    f"variants={[v.get('label') for v in status.get('variants') or []]}, "
                    f"templates={list((status.get('prompt_templates') or {}).keys())}",
                )
            )

            started = time.perf_counter()
            reply = client.post(
                "/api/chat",
                json={
                    "message": "Сколько соли на 1 кг мяса для прайм риб?",
                    "use_rag": True,
                    "temperature": 0.1,
                    "max_tokens": 128,
                    "context_window": 4,
                    "prompt_template": "recipe_structured",
                },
            )
            elapsed = (time.perf_counter() - started) * 1000
            data = reply.get_json()
            gen = data.get("generation") or {}
            results.append(
                CheckResult(
                    "Параметры из UI применяются к запросу",
                    reply.status_code == 200
                    and gen.get("temperature") == 0.1
                    and gen.get("max_tokens") == 128
                    and gen.get("context_window") == 4
                    and data.get("prompt_template") == "recipe_structured",
                    f"generation={gen}, prompt_template={data.get('prompt_template')}, "
                    f"reply={_preview(data.get('reply'))}",
                    elapsed,
                )
            )

            settings = client.post(
                "/api/settings", json={"ctx_size": 4096, "variant": "Q2_K"}
            ).get_json()
            results.append(
                CheckResult(
                    "POST /api/settings меняет ctx_size и вариант",
                    settings.get("ctx_size") == 4096
                    and settings.get("local_variant") == "Q2_K",
                    f"ctx_size={settings.get('ctx_size')}, "
                    f"local_variant={settings.get('local_variant')}",
                )
            )
        finally:
            backend.stop()
    return results


def _render_web_markdown(results: list[CheckResult]) -> str:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "## 9. Проверка настроек оптимизации в веб-интерфейсе",
        "",
        f"**Итог: {passed}/{len(results)} проверок пройдено.**",
        "",
        "| Проверка | Результат | Детали | Время, мс |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        mark = "✅" if result.passed else "❌"
        detail = _preview(result.detail, 400).replace("|", "\\|")
        lines.append(f"| {result.name} | {mark} | {detail} | {result.latency_ms:.0f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    config = load_config()
    opt_cfg = config.get("optimization", {}) or {}
    rag_cfg = config.get("local_rag", {}) or {}

    parser = argparse.ArgumentParser(
        description="Day 29: оптимизация локальной LLM под задачу (рецепты)."
    )
    parser.add_argument("--offline", action="store_true", help="Стаб без модели и сети")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Ограничить вопросы")
    parser.add_argument(
        "--repeats", type=int, default=int(opt_cfg.get("repeats", 2)),
        help="Повторов на вопрос",
    )
    parser.add_argument("--variant", default=None, help="Оставить один вариант модели")
    parser.add_argument("--top-k", type=int, default=int(rag_cfg.get("top_k", 5)))
    parser.add_argument(
        "--max-context-chars", type=int, default=int(rag_cfg.get("max_context_chars", 6000))
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=float(rag_cfg.get("relevance_threshold", 0.33)),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", default=None, choices=["structural", "fixed"])
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--report-dir", default=opt_cfg.get("report_dir", "history/optimization"))
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("⚙️  ДЕНЬ 29: ОПТИМИЗАЦИЯ ЛОКАЛЬНОЙ LLM ПОД ЗАДАЧУ (РЕЦЕПТЫ)")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой llama.cpp'}\n")

    # 1. Локальный retrieval.
    try:
        retriever = _load_retriever(config, args)
    except Exception as exc:  # noqa: BLE001 — понятная ошибка загрузки индекса
        print(f"✗ Ошибка загрузки индекса: {exc}", file=sys.stderr)
        return 1
    print(f"✓ Локальный индекс загружен: {retriever.total_vectors} векторов, "
          f"top-{args.top_k}")

    # 2. Конфигурации.
    servers = _LiveServers()
    try:
        if args.offline:
            configurations = _offline_configurations(config, args)
        else:
            presets = parse_presets(config)
            variants = parse_variants(config)
            specs = parse_configuration_specs(config)
            configurations = []
            for spec in specs:
                built = servers.build(spec, presets, variants, config)
                if built is not None:
                    configurations.append(built)
            if not configurations:
                print(
                    "✗ Ни один вариант локальной модели не доступен. "
                    "Скачайте модели: bash scripts/setup_llama_cpp.sh",
                    file=sys.stderr,
                )
                return 1
        print(f"✓ Конфигураций: {len(configurations)} — "
              + ", ".join(c.label for c in configurations))

        questions = CONTROL_QUESTIONS
        if args.limit:
            questions = questions[: args.limit]

        def _progress(name: str, current: int, total: int, query: str) -> None:
            print(f"   [{name} {current:>2}/{total}] {_preview(query, 56)}")

        print(f"\n[Сравнение] {len(questions)} вопросов, повторов: {args.repeats}")
        report = compare_optimizations(
            retriever,
            configurations,
            questions=questions,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            relevance_threshold=args.relevance_threshold,
            repeats=max(1, args.repeats),
            progress_callback=_progress,
        )

        # 3. Ресурсы живых серверов (пик RSS дочерних процессов).
        if not args.offline:
            peak = measure_peak_rss_mb()
            for config_entry in report["configurations"]:
                config_entry["resources"]["peak_rss_mb"] = peak

        # 4. HTTP-проверки веб-настроек.
        print("\n[Веб-чат] Проверка настроек оптимизации (HTTP)…")
        web_checks = _run_web_checks(retriever, config)
        for check in web_checks:
            _print_check(check)

        # 5. Сводка.
        print("\n" + "=" * 78)
        print("Сводка:")
        for config_entry in report["configurations"]:
            quality = config_entry["quality"]
            speed = config_entry["speed"]
            resources = config_entry["resources"]
            print(
                f"   {config_entry['label']:<28} "
                f"факты {quality['avg_facts'] * 100:5.1f}% · "
                f"ссылки {quality['citation_rate'] * 100:5.1f}% · "
                f"{speed['latency_ms']:7.0f} мс · "
                f"{speed['tokens_per_sec']:5.1f} ток/с · "
                f"{resources['model_size_mb'] or 0:.0f} МБ"
            )
        deltas = report.get("deltas")
        if deltas:
            print(
                f"   Дельта «после-до»: факты {deltas['avg_facts'] * 100:+.1f} п.п., "
                f"ссылки {deltas['citation_rate'] * 100:+.1f} п.п., "
                f"задержка {deltas['latency_ms']:+.0f} мс, "
                f"модель {deltas['model_size_mb']:+.0f} МБ"
            )
        print("=" * 78)

        if not args.skip_save:
            out_dir = Path(args.report_dir)
            if not out_dir.is_absolute():
                out_dir = REPO_ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            markdown = (
                render_optimization_markdown(report)
                + "\n"
                + _render_web_markdown(web_checks)
            )
            (out_dir / "report.md").write_text(markdown, encoding="utf-8")
            payload = {
                "mode": "offline" if args.offline else "live",
                "report": report,
                "web_checks": [asdict(c) for c in web_checks],
            }
            (out_dir / "results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nОтчёт: {out_dir / 'report.md'}")
            print(f"Результаты: {out_dir / 'results.json'}")

        return 0 if all(c.passed for c in web_checks) else 1
    finally:
        servers.stop()


if __name__ == "__main__":
    raise SystemExit(main())
