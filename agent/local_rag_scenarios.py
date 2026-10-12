#!/usr/bin/env python3
"""Day 28: сквозной сценарий RAG-системы, полностью работающей локально.

Что проверяет и показывает:

1. **Локальный retrieval** — FAISS-индекс рецептов (Дни 21–22) и офлайн-эмбеддер,
   без обращения к сети.
2. **Локальная генерация** — ответ строит ``llama-server`` из Дня 26 (или
   детерминированный стаб в офлайн-режиме).
3. **Сравнение** локальной и облачной модели по **качеству, скорости и
   стабильности** на 10 контрольных вопросах.
4. **Веб-чат с RAG** (День 27 + RAG) — ответ и источники через HTTP.

Запуск::

    # детерминированная офлайн-проверка (без модели и сети)
    python3 -m agent.local_rag_scenarios --offline

    # живой прогон: локальная модель + сравнение с облачной
    python3 -m agent.local_rag_scenarios

    # живой прогон только локальной модели
    python3 -m agent.local_rag_scenarios --no-cloud

    # одиночный вопрос
    python3 -m agent.local_rag_scenarios --question "Сколько соли на 1 кг мяса для прайм риб?"
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
from agent.llm_client import LLMClient
from agent.local_llm import REPO_ROOT, LocalLLMConfig, LocalLLMError
from agent.local_rag import (
    LocalRagStubClient,
    build_local_rag_client,
    compare_local_vs_cloud,
    detect_category,
    is_generic_query,
    render_local_rag_markdown,
    strip_meta_leak,
    timed_rag_query,
)
from agent.rag import CONTROL_QUESTIONS, RagRetriever, load_retriever
from agent.web_chat import WebChatBackend, create_app

DEMO_QUESTION = CONTROL_QUESTIONS[0]["query"]
OFF_TOPIC_QUESTION = "Какая типичная погода в Москве в августе?"
SECOND_QUESTION = CONTROL_QUESTIONS[1]["query"]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").split())


@dataclass
class CheckResult:
    """Результат одной HTTP-проверки веб-чата с RAG."""

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


def _load_retriever(args, config: dict) -> RagRetriever:
    idx_cfg = config.get("indexing", {}) or {}
    embedder = HashingEmbedder(dim=int(idx_cfg.get("embedding_dim", 256)))
    return load_retriever(
        index_dir=args.index_dir or idx_cfg.get("index_dir", "history/index"),
        dataset=args.dataset,
        strategy=args.strategy,
        embedder=embedder,
        top_k=args.top_k,
    )


def _run_web_checks(backend: WebChatBackend, question: str) -> list[CheckResult]:
    """HTTP-проверки локального RAG-веб-чата через тестовый клиент Flask."""
    app = create_app(backend)
    client = app.test_client()
    results: list[CheckResult] = []

    page = client.get("/")
    html = page.get_data(as_text=True)
    has_selector = 'id="mode"' in html and "не использовать RAG" in html and "рецепты" in html
    has_source = 'id="source"' in html and "локальная" in html and "облачная" in html
    results.append(
        CheckResult(
            "Веб-чат отдаёт страницу с выбором модели и режима",
            page.status_code == 200
            and "локальная и облачная модель" in html
            and has_selector
            and has_source,
            f"GET / → HTTP {page.status_code}, {len(html)} байт, "
            f"выбор режима: {'да' if has_selector else 'нет'}, "
            f"выбор модели: {'да' if has_source else 'нет'}",
        )
    )

    status = client.get("/api/status").get_json()
    results.append(
        CheckResult(
            "RAG (рецепты) доступен в статусе бэкенда",
            bool(status.get("rag")),
            f"rag={status.get('rag')}, dataset={status.get('dataset')}, "
            f"vectors={status.get('vectors')}",
        )
    )

    if backend.cloud_enabled:
        results.append(
            CheckResult(
                "Облачная модель доступна в статусе бэкенда",
                bool(status.get("cloud")) and "cloud" in (status.get("sources") or []),
                f"cloud={status.get('cloud')}, cloud_model={status.get('cloud_model')}",
            )
        )
        cloud = client.post(
            "/api/chat",
            json={"message": question, "use_rag": True, "model_source": "cloud"},
        ).get_json()
        cloudsrc = cloud.get("sources") or []
        results.append(
            CheckResult(
                "Облачная модель отвечает по локальному RAG-индексу",
                cloud.get("model_source") == "cloud"
                and bool(cloud.get("reply"))
                and cloud.get("used_rag") is True
                and bool(cloudsrc),
                f"model_source={cloud.get('model_source')}, источников: {len(cloudsrc)}, "
                f"reply={_preview(cloud.get('reply'))}",
            )
        )

    # 1. Режим «рецепты» (use_rag=true) — поиск + источники.
    started = time.perf_counter()
    response = client.post("/api/chat", json={"message": question, "use_rag": True})
    elapsed = (time.perf_counter() - started) * 1000
    data = response.get_json()
    sources = data.get("sources") or []
    results.append(
        CheckResult(
            "Режим «рецепты»: ответ с источниками из локального индекса",
            response.status_code == 200
            and bool(data.get("reply"))
            and data.get("used_rag") is True
            and bool(sources),
            f"reply={_preview(data.get('reply'))}; источников: {len(sources)}; "
            f"used_rag={data.get('used_rag')}",
            elapsed,
        )
    )

    first = sources[0] if sources else {}
    results.append(
        CheckResult(
            "Источник содержит метаданные рецепта",
            bool(first.get("title")) and first.get("recipe_id") is not None,
            f"«{first.get('title')}» (ID {first.get('recipe_id')}, "
            f"раздел {first.get('section')}, score={first.get('score')})",
        )
    )

    # 2. Режим «не использовать RAG» (use_rag=false) — без retrieval и источников.
    plain = client.post("/api/chat", json={"message": question, "use_rag": False}).get_json()
    results.append(
        CheckResult(
            "Режим «не использовать RAG»: без источников",
            bool(plain.get("reply"))
            and plain.get("used_rag") is False
            and not (plain.get("sources") or []),
            f"reply={_preview(plain.get('reply'))}; "
            f"used_rag={plain.get('used_rag')}; источников: {len(plain.get('sources') or [])}",
        )
    )

    # 3. Нерелевантный вопрос в режиме «рецепты» — отказ и НЕТ источников.
    off_topic = client.post(
        "/api/chat", json={"message": OFF_TOPIC_QUESTION, "use_rag": True}
    ).get_json()
    off_sources = off_topic.get("sources") or []
    reply_norm = _normalize(off_topic.get("reply", ""))
    refused = "нет информации" in reply_norm or "отсутств" in reply_norm or "не знаю" in reply_norm
    results.append(
        CheckResult(
            "Нерелевантный вопрос: отказ и НЕТ источников",
            bool(off_topic.get("reply")) and refused and not off_sources,
            f"reply={_preview(off_topic.get('reply'))}; источников: {len(off_sources)}",
        )
    )

    # 4. История диалога в RAG-режиме сохраняется между ходами.
    first_turn = client.post(
        "/api/chat", json={"message": question, "use_rag": True}
    ).get_json()
    sid = first_turn.get("session_id")
    client.post(
        "/api/chat", json={"session_id": sid, "message": SECOND_QUESTION, "use_rag": True}
    )
    history = client.get(f"/api/history?session_id={sid}").get_json()
    messages = history.get("messages", [])
    roles = [m.get("role") for m in messages]
    assistant_metas = [m.get("meta") or {} for m in messages if m.get("role") == "assistant"]
    meta_ok = bool(assistant_metas) and all(
        meta.get("mode") == "rag" and meta.get("sources") for meta in assistant_metas
    )
    results.append(
        CheckResult(
            "История диалога в RAG-режиме сохраняется (с режимом и источниками)",
            roles == ["user", "assistant", "user", "assistant"] and meta_ok,
            f"session={sid}, сообщений: {len(messages)}, роли: {roles}, "
            f"meta.mode у ассистентов: {[m.get('mode') for m in assistant_metas]}",
        )
    )

    # 5. Общий запрос: несколько рецептов-подсказок в контексте.
    generic = client.post(
        "/api/chat", json={"message": "Что можно приготовить?", "use_rag": True}
    ).get_json()
    generic_sources = generic.get("sources") or []
    results.append(
        CheckResult(
            "Общий вопрос: несколько источников для выбора",
            bool(generic.get("reply")) and len(generic_sources) >= 3,
            f"источников: {len(generic_sources)}; reply={_preview(generic.get('reply'))}",
        )
    )

    return results


def _run_routing_checks(
    retriever: RagRetriever, client, category_suggestions: int
) -> list[CheckResult]:
    """Проверки маршрутизации: категорийная подборка, конкретное блюдо, «топ N»."""
    results: list[CheckResult] = []
    by_recipe: dict = {}
    for chunk in retriever.index.chunks:
        by_recipe.setdefault(
            (chunk.metadata or {}).get("recipe_id"),
            str((chunk.metadata or {}).get("category", "")).lower(),
        )

    # 1. Категорийная подборка «рецепт из говядины».
    cat_res = timed_rag_query(
        client, retriever, "рецепт из говядины", category_suggestions=category_suggestions
    )
    cat_sources = cat_res["answer"].sources
    bad = [
        s.recipe_id for s in cat_sources if "говядин" not in by_recipe.get(s.recipe_id, "")
    ]
    results.append(
        CheckResult(
            "Категорийная подборка: «рецепт из говядины» → разные рецепты говядины",
            cat_res["route"] == "category" and len(cat_sources) >= 3 and not bad,
            f"route={cat_res['route']}, рецептов: {len(cat_sources)}, "
            f"не-говядина: {bad[:3]}",
        )
    )

    # 2. Конкретное блюдо не должно превращаться в подборку по категории.
    dish_res = timed_rag_query(client, retriever, "рецепт шницеля из говядины")
    dish_sources = dish_res["answer"].sources
    top_title = (dish_sources[0].title if dish_sources else "").lower()
    results.append(
        CheckResult(
            "Конкретное блюдо остаётся конкретным (не подборка)",
            dish_res["route"] == "specific" and "шницел" in top_title,
            f"route={dish_res['route']}, top=«{top_title}»",
        )
    )

    # 2b. «Покажи рецепт X» → в источниках весь рецепт, а не только описание.
    full_res = timed_rag_query(client, retriever, "покажи рецепт шницеля из говядины")
    full_sections = {s.section for s in full_res["answer"].sources}
    results.append(
        CheckResult(
            "«Покажи рецепт X» → ингредиенты и шаги попадают в источники",
            full_res["route"] == "specific"
            and "ingredients" in full_sections
            and "step" in full_sections,
            f"route={full_res['route']}, секции: {sorted(full_sections)}",
        )
    )

    # 3. «топ 5» без категории — общий маршрут с несколькими источниками.
    top_res = timed_rag_query(client, retriever, "топ 5")
    results.append(
        CheckResult(
            "Запрос без категории («топ 5») → общий маршрут с источниками",
            top_res["route"] == "generic" and len(top_res["answer"].sources) >= 3,
            f"route={top_res['route']}, источников: {len(top_res['answer'].sources)}",
        )
    )

    # 4. Пост-очистка «утечки» служебным контекстом (не только в отказе).
    leaked = (
        "В предоставленном контексте указано, что для шницеля из говядины "
        "требуется обсушить 3 шницеля бумажным полотенцем."
    )
    cleaned = strip_meta_leak(leaked)
    results.append(
        CheckResult(
            "Мета-утечка удаляется из ответа (не только при отказе)",
            "предоставленном контексте" not in cleaned.lower(),
            f"cleaned={_preview(cleaned)}",
        )
    )

    # 5. Уточнение-отсылка: «подробней про 2 позицию» → 2-й рецепт из списка.
    prior_titles = [s.title for s in cat_sources]
    history = [
        {"role": "user", "content": "рецепт из говядины"},
        {
            "role": "assistant",
            "content": "Вот несколько рецептов:\n"
            + "\n".join(
                f"{i}. {t} [Источник {i}]" for i, t in enumerate(prior_titles, 1)
            ),
            "meta": {
                "mode": "rag",
                "sources": [
                    {"rank": i, "title": t, "recipe_id": s.recipe_id}
                    for i, (t, s) in enumerate(zip(prior_titles, cat_sources), 1)
                ],
            },
        },
    ]
    follow_res = timed_rag_query(
        client, retriever, "подробней про 2 позицию", history=history
    )
    follow_sources = follow_res["answer"].sources
    expected = prior_titles[1] if len(prior_titles) > 1 else ""
    follow_top = follow_sources[0].title if follow_sources else ""
    results.append(
        CheckResult(
            "Уточнение «подробней про 2 позицию» → конкретный 2-й рецепт",
            follow_res["route"] == "followup" and follow_top == expected,
            f"route={follow_res['route']}, top=«{follow_top}», ожидали «{expected}»",
        )
    )

    # 6. Уточнение-местоимение → последний упомянутый рецепт.
    pronoun_res = timed_rag_query(
        client, retriever, "расскажи про него поподробнее", history=history
    )
    pronoun_top = pronoun_res["answer"].sources[0].title if pronoun_res["answer"].sources else ""
    results.append(
        CheckResult(
            "Уточнение-местоимение «про него» → последний рецепт из списка",
            pronoun_res["route"] == "followup" and pronoun_top == prior_titles[-1],
            f"route={pronoun_res['route']}, top=«{pronoun_top}»",
        )
    )

    # 7. Уточнение без предыдущего RAG-списка → просьба уточнить.
    clarify_res = timed_rag_query(
        client,
        retriever,
        "подробней про 2 позицию",
        history=[
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "Здравствуйте!"},
        ],
    )
    results.append(
        CheckResult(
            "Уточнение без списка → просьба уточнить (не «нет информации»)",
            clarify_res["route"] == "clarify",
            f"route={clarify_res['route']}, reply={_preview(clarify_res['answer'].content)}",
        )
    )
    return results


def _render_web_markdown(results: list[CheckResult]) -> str:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "## 7. Проверка локального RAG-веб-чата (HTTP)",
        "",
        f"**Итог: {passed}/{len(results)} проверок пройдено.**",
        "",
        "| Проверка | Результат | Детали | Время, мс |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        mark = "✅" if result.passed else "❌"
        detail = _preview(result.detail, 400).replace("|", "\\|")
        lines.append(
            f"| {result.name} | {mark} | {detail} | {result.latency_ms:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_single_question(
    retriever: RagRetriever,
    local_client,
    cloud_client,
    question: str,
    args,
) -> None:
    """Демонстрация одного вопроса: локальный и облачный ответ + источники."""
    local = timed_rag_query(
        local_client,
        retriever,
        question,
        top_k=args.top_k,
        temperature=args.temperature,
        max_context_chars=args.max_context_chars,
        relevance_threshold=args.relevance_threshold,
        generic_threshold=args.generic_threshold,
        generic_min_sources=args.generic_min_sources,
        category_suggestions=args.category_suggestions,
    )
    print("\n" + "-" * 72)
    print(f"Вопрос: {question}")
    route = local["route"]
    category = detect_category(question)
    suffix = f" (категория: {category})" if category else ""
    print(f"Маршрут: {route}{suffix}")
    print("-" * 72)
    if route == "category":
        sources = local["answer"].sources
        print(f"Источники категории ({len(sources)}):")
        for src in sources:
            print(f"  #{src.rank} [score={src.score:.4f}] «{src.title}» (ID {src.recipe_id})")
    else:
        raw = retriever.retrieve(question, top_k=args.top_k)
        generic = is_generic_query(question)
        threshold = args.generic_threshold if generic else args.relevance_threshold
        passed = [s for s in raw if s.score >= threshold]
        label = f"порог {threshold}" + (" — общий вопрос" if generic else "")
        print(f"Кандидаты локального retrieval ({label}):")
        if not raw:
            print("  (ничего не найдено)")
        for src in raw:
            mark = "✓" if src.score >= threshold else "×"
            print(f"  {mark} #{src.rank} [score={src.score:.4f}] «{src.title}» (ID {src.recipe_id})")
        print(f"  прошло порог: {len(passed)} из {len(raw)}")
    print(f"\n[Локальная модель] ({local['retrieval_ms']:.0f} мс поиск + "
          f"{local['generation_ms']:.0f} мс генерация):")
    print(local["answer"].content or f"Ошибка: {local['answer'].error}")
    if not local["answer"].sources and not local["low_relevance"]:
        print("(источники скрыты: модель ответила отказом)")

    if cloud_client is not None:
        cloud = timed_rag_query(
            cloud_client,
            retriever,
            question,
            top_k=args.top_k,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            relevance_threshold=args.relevance_threshold,
            category_suggestions=args.category_suggestions,
        )
        print(f"\n[Облачная модель] ({cloud['generation_ms']:.0f} мс генерация):")
        print(cloud["answer"].content or f"Ошибка: {cloud['answer'].error}")
    print()


def main() -> int:
    config = load_config()
    rag_cfg = config.get("local_rag", {}) or {}
    idx_cfg = config.get("indexing", {}) or {}

    parser = argparse.ArgumentParser(
        description="Day 28: RAG-система, полностью работающая локально."
    )
    parser.add_argument("--offline", action="store_true", help="Стаб без модели и сети")
    parser.add_argument("--no-cloud", action="store_true", help="Не сравнивать с облачной моделью")
    parser.add_argument("--question", "-q", default=None, help="Одиночный вопрос")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Ограничить число вопросов")
    parser.add_argument(
        "--repeats", type=int, default=int(rag_cfg.get("repeats", 2)),
        help="Повторов на вопрос (для оценки стабильности)",
    )
    parser.add_argument("--top-k", type=int, default=int(rag_cfg.get("top_k", 5)))
    parser.add_argument("--temperature", type=float, default=float(rag_cfg.get("temperature", 0.2)))
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=float(rag_cfg.get("relevance_threshold", 0.33)),
        help="Порог сходства: чанки ниже отбрасываются, при пустом списке модель не вызывается",
    )
    parser.add_argument(
        "--generic-threshold",
        type=float,
        default=float(rag_cfg.get("generic_threshold", 0.25)),
        help="Смягчённый порог для общих вопросов («что можно приготовить?»)",
    )
    parser.add_argument(
        "--generic-min-sources",
        type=int,
        default=int(rag_cfg.get("generic_min_sources", 3)),
        help="Минимум источников для общего вопроса",
    )
    parser.add_argument(
        "--category-suggestions",
        type=int,
        default=int(rag_cfg.get("category_suggestions", 5)),
        help="Сколько рецептов предлагать в категорийной подборке («рецепт из говядины»)",
    )
    parser.add_argument(
        "--max-context-chars", type=int, default=int(rag_cfg.get("max_context_chars", 6000))
    )
    parser.add_argument("--dataset", default=rag_cfg.get("dataset", "recipt_all"))
    parser.add_argument(
        "--strategy", default=rag_cfg.get("strategy", "structural"),
        choices=["structural", "fixed"],
    )
    parser.add_argument("--index-dir", default=idx_cfg.get("index_dir", "history/index"))
    parser.add_argument("--llama-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--report-dir", default=rag_cfg.get("report_dir", "history/local_rag"))
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    local_cfg = LocalLLMConfig.from_config(config)
    if args.llama_dir:
        local_cfg.llama_dir = args.llama_dir
    if args.model:
        local_cfg.model_path = args.model
    if args.port:
        local_cfg.port = args.port
    # Пробрасываем порог релевантности в конфиг, чтобы его подхватил и веб-бэкенд.
    config.setdefault("local_rag", {})["relevance_threshold"] = args.relevance_threshold
    config.setdefault("local_rag", {})["category_suggestions"] = args.category_suggestions

    print("=" * 78)
    print("🏠 ДЕНЬ 28: RAG-СИСТЕМА, ПОЛНОСТЬЮ РАБОТАЮЩАЯ ЛОКАЛЬНО")
    print("=" * 78)
    print(f"Режим: {'офлайн-стаб' if args.offline else 'живой llama.cpp'}")
    print(f"Локальная модель: {local_cfg.model_file}")
    print(f"Индекс: {args.index_dir}/{args.dataset}/{args.strategy}")
    print(f"Порог релевантности: {args.relevance_threshold}\n")

    # 1. Локальный retrieval.
    try:
        retriever = _load_retriever(args, config)
    except Exception as exc:  # noqa: BLE001 — понятная ошибка загрузки индекса
        print(f"✗ Ошибка загрузки индекса: {exc}", file=sys.stderr)
        return 1
    print(f"✓ Локальный индекс загружен: {retriever.total_vectors} векторов, top-{args.top_k}")

    # 2. Локальная генерация (сервер или стаб).
    server = None
    try:
        server, local_client = build_local_rag_client(
            config,
            offline=args.offline,
            model_path=args.model,
            port=args.port,
        )
    except (LocalLLMError, OSError) as exc:
        print(f"✗ Ошибка локальной модели: {exc}", file=sys.stderr)
        return 1
    print(f"✓ Локальная генерация готова: {getattr(local_client, 'config', {}).get('model', '?')}")

    cloud_client = None
    if not args.offline and not args.no_cloud and config.get("model"):
        cloud_client = LLMClient(config)
        print(f"✓ Облачная модель для сравнения: {config.get('model')}")

    try:
        # 3. Одиночный вопрос.
        if args.question:
            _run_single_question(
                retriever, local_client, cloud_client, args.question, args
            )
            return 0

        # 4. Пакетное сравнение.
        questions = CONTROL_QUESTIONS
        if args.limit:
            questions = questions[: args.limit]

        def _progress(current: int, total: int, query: str) -> None:
            print(f"   [{current:>2}/{total}] {_preview(query, 66)}")

        print(f"\n[Сравнение] {len(questions)} вопросов, повторов: {args.repeats}")
        report = compare_local_vs_cloud(
            retriever=retriever,
            local_client=local_client,
            cloud_client=cloud_client,
            questions=questions,
            top_k=args.top_k,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            repeats=max(1, args.repeats),
            relevance_threshold=args.relevance_threshold,
            local_label=getattr(local_client, "config", {}).get("model", "local-llm"),
            cloud_label=config.get("model", "cloud") if cloud_client else "cloud",
            progress_callback=_progress,
        )

        # 5. HTTP-проверка локального RAG-веб-чата.
        print("\n[Веб-чат] Проверка локального RAG-чата (HTTP)…")
        with tempfile.TemporaryDirectory(prefix="local_rag_web_") as history_dir:
            backend = WebChatBackend(
                config,
                offline=args.offline,
                retriever=retriever,
                top_k=args.top_k,
                temperature=args.temperature,
                client=local_client,
                history_dir=history_dir,
                # В офлайн-режиме облако недоступно; подменяем детерминированным
                # стабом, чтобы проверить маршрут local/cloud по тому же индексу.
                cloud_client=LocalRagStubClient("cloud-rag-stub")
                if args.offline
                else None,
            )
            backend.start()
            try:
                web_checks = _run_web_checks(backend, questions[0]["query"])
            finally:
                backend.stop()
        for check in web_checks:
            _print_check(check)

        print("\n[Маршрутизация] Проверки категорий/подборок и уточнений…")
        routing_checks = _run_routing_checks(retriever, local_client, args.category_suggestions)
        web_checks = web_checks + routing_checks
        for check in routing_checks:
            _print_check(check)

        # 6. Сводка и отчёты.
        quality = report["summary"]["quality"]
        speed = report["summary"]["speed"]
        stability = report["summary"]["stability"]
        print("\n" + "=" * 78)
        print("Сводка:")
        print(f"   Качество: покрытие фактов (локал) {quality['local_avg_facts'] * 100:.1f}%"
              + (f", облако {quality['cloud_avg_facts'] * 100:.1f}%"
                 if quality.get("cloud_avg_facts") is not None else ""))
        print(f"   Скорость: задержка (локал) {speed['local_latency_ms']:.0f} мс "
              f"(поиск {speed['local_retrieval_ms']:.0f} мс + генерация "
              f"{speed['local_generation_ms']:.0f} мс)")
        print(f"   Стабильность: ошибки {stability['local_error_rate'] * 100:.0f}%, "
              f"самосогласованность {stability['local_self_consistency']:.2f}")
        print("=" * 78)

        if not args.skip_save:
            out_dir = Path(args.report_dir)
            if not out_dir.is_absolute():
                out_dir = REPO_ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            markdown = render_local_rag_markdown(report) + "\n" + _render_web_markdown(web_checks)
            (out_dir / "report.md").write_text(markdown, encoding="utf-8")
            payload = {
                "mode": "offline" if args.offline else "live",
                "model": str(local_cfg.model_file),
                "report": report,
                "web_checks": [asdict(c) for c in web_checks],
            }
            (out_dir / "results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nОтчёт: {out_dir / 'report.md'}")
            print(f"Результаты: {out_dir / 'results.json'}")

        web_ok = all(c.passed for c in web_checks)
        return 0 if web_ok else 1
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
