#!/usr/bin/env python3
"""Офлайн end-to-end проверки сервиса рецептов.

Поднимает Flask-приложение на детерминированном RAG-стабе (индекс FAISS
обязателен, сеть и модель не нужны) и проверяет полный сценарий:

1. страницы/API требуют авторизации (302 на странице, 401 в API);
2. регистрация → выход → вход;
3. создание чата и RAG-вопрос → ответ с источниками;
4. сохранение истории в SQLite и режим «без RAG»;
5. изоляция чатов между пользователями (403 / пустой список);
6. очистка истории и выход.

Запуск::

    python3 -m scenarios.service_scenarios --offline
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from app import create_app
from app.config import SERVICE_ROOT, ServiceConfig
from app.engine import RagEngine
from core.local_rag import (
    NO_RELEVANT_INFO_MESSAGE,
    LocalRagStubClient,
    _category_menu,
    _recipe_fallback,
    extract_requested_count,
    retrieve_recipe_sections,
    timed_rag_query,
)
from core.llm_client import Completion
from core.query_plan import _llm_plan, normalize_markers, plan_query
from core.rag import RAG_SYSTEM_PROMPT, RECIPE_FORMAT_SYSTEM_PROMPT

PASS = "✓"
FAIL = "✗"


class _RecordingClient:
    """Обёртка над стабом, фиксирующая system-промпт каждого запроса."""

    def __init__(self, inner: LocalRagStubClient) -> None:
        self.inner = inner
        self.last_system: str | None = None

    @property
    def config(self) -> dict:
        return self.inner.config

    def reset(self) -> None:
        self.last_system = None

    def complete(self, messages, temperature=None, max_tokens=None):
        for message in messages:
            if message.get("role") == "system":
                self.last_system = message.get("content")
                break
        return self.inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )


class _JsonPlannerClient:
    """Фейковый LLM-планировщик: возвращает заданный JSON-план без сети."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.config = {"model": "fake-planner"}

    def complete(self, messages, temperature=None, max_tokens=None) -> Completion:
        return Completion(content=self.payload, model=self.config["model"])


class Checker:
    """Простейший раннер проверок с накоплением ошибок."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"  {PASS} {name}")
        else:
            self.failures.append(name)
            print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))

    def section(self, title: str) -> None:
        print(f"\n{title}")


def _build_app(tmp: Path):
    config = ServiceConfig.from_dict(
        {
            "database": {"path": str(tmp / "service.db")},
            "index": {"dir": str(SERVICE_ROOT / "index"), "dataset": "recipt_all",
                      "strategy": "structural", "top_k": 5},
            "local_rag": {"relevance_threshold": 0.33},
            "web": {"allow_reindex": False},
            "security": {"secret_path": str(tmp / "secret.key")},
        },
        service_root=SERVICE_ROOT,
    )
    engine = RagEngine(config, offline=True)
    app = create_app(config, engine=engine)
    return app, engine


def run() -> int:
    checker = Checker()
    with tempfile.TemporaryDirectory(prefix="recipe_service_") as tmp_dir:
        app, engine = _build_app(Path(tmp_dir))
        try:
            client = app.test_client()

            checker.section("1. Доступ без авторизации")
            resp = client.get("/chat")
            checker.check("GET /chat → редирект на /login", resp.status_code == 302)
            resp = client.get("/api/status")
            checker.check("GET /api/status → 401", resp.status_code == 401)
            resp = client.get("/api/chats")
            checker.check("GET /api/chats → 401", resp.status_code == 401)

            checker.section("2. Регистрация и вход")
            resp = client.post(
                "/api/register", json={"username": "alice", "password": "secret1"}
            )
            checker.check("POST /api/register → 201", resp.status_code == 201, resp.get_data(as_text=True))
            resp = client.get("/api/me")
            checker.check("GET /api/me → alice", resp.get_json().get("user", {}).get("username") == "alice")
            resp = client.post("/api/logout")
            checker.check("POST /api/logout → ok", resp.status_code == 200)
            resp = client.get("/api/me")
            checker.check("GET /api/me после logout → 401", resp.status_code == 401)
            resp = client.post(
                "/api/login", json={"username": "alice", "password": "wrong"}
            )
            checker.check("Неверный пароль → 401", resp.status_code == 401)
            resp = client.post(
                "/api/login", json={"username": "alice", "password": "secret1"}
            )
            checker.check("POST /api/login → 200", resp.status_code == 200)

            checker.section("3. Чат и RAG-вопрос")
            resp = client.post("/api/chats", json={})
            chat_id = resp.get_json().get("chat", {}).get("id")
            checker.check("POST /api/chats → 201", resp.status_code == 201 and chat_id)

            resp = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "рецепт шашлыка из говядины", "use_rag": True},
            )
            payload = resp.get_json() or {}
            checker.check("RAG-ответ → 200", resp.status_code == 200, str(payload))
            checker.check("Ответ непустой", bool(payload.get("reply", "").strip()))
            checker.check("Есть источники", len(payload.get("sources", [])) > 0)
            checker.check("Режим = rag", payload.get("mode") == "rag")

            resp = client.get(f"/api/chats/{chat_id}/messages")
            messages = resp.get_json().get("messages", [])
            checker.check("История: 2 сообщения", len(messages) == 2)
            assistant = next((m for m in messages if m["role"] == "assistant"), {})
            meta = assistant.get("meta") or {}
            checker.check("meta.mode = rag", meta.get("mode") == "rag")
            checker.check("meta.sources сохранены", len(meta.get("sources", [])) > 0)

            checker.section("4. Режим без RAG")
            resp = client.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "Привет!", "use_rag": False},
            )
            payload = resp.get_json() or {}
            checker.check("Plain-ответ → 200", resp.status_code == 200)
            checker.check("Режим = plain", payload.get("mode") == "plain")
            checker.check("Без источников", payload.get("sources") == [])

            checker.section("4c. Авто-форматирование рецепта")
            recorder = _RecordingClient(LocalRagStubClient())
            timed_rag_query(
                client=recorder,
                retriever=engine.retriever,
                question="покажи рецепт стейк рибай",
                top_k=5,
                relevance_threshold=0.33,
            )
            checker.check(
                "Конкретное блюдо → recipe-промпт",
                recorder.last_system == RECIPE_FORMAT_SYSTEM_PROMPT,
            )
            recorder.reset()
            timed_rag_query(
                client=recorder,
                retriever=engine.retriever,
                question="что можно приготовить?",
                top_k=5,
                relevance_threshold=0.33,
            )
            checker.check(
                "Общий вопрос → базовый промпт",
                recorder.last_system == RAG_SYSTEM_PROMPT,
            )
            raw_sources = engine.retriever.retrieve("стейк рибай", top_k=5)
            recipe_id = raw_sources[0].recipe_id if raw_sources else None
            recipe_sections = (
                retrieve_recipe_sections(
                    engine.retriever, recipe_id, question="стейк рибай"
                )
                if recipe_id is not None
                else []
            )
            fallback = _recipe_fallback(recipe_sections)
            checker.check(
                "Fallback: структура рецепта",
                fallback.startswith("### ")
                and "**Ингредиенты**" in fallback
                and "**Приготовление**" in fallback,
            )

            checker.section("4d. Планировщик запроса: подборки и категории")
            taxonomy = engine.taxonomy
            checker.check(
                "normalize_markers снимает «бы»",
                "что приготовить" in normalize_markers("что бы приготовить"),
            )
            category_cases = [
                ("что бы приготовить из морепродуктов?", "дары моря"),
                ("что можно сделать из рыбы", "рыба"),
                ("что-нибудь с говядиной", "говядина"),
                ("дай рецепт из курицы", "птица"),
            ]
            for question, expected_category in category_cases:
                result = timed_rag_query(
                    client=LocalRagStubClient(),
                    retriever=engine.retriever,
                    question=question,
                    top_k=5,
                    relevance_threshold=0.33,
                    taxonomy=taxonomy,
                )
                answer = result["answer"]
                checker.check(
                    f"«{question}» → маршрут category",
                    result["route"] == "category",
                    f"получен {result['route']}",
                )
                checker.check(
                    f"«{question}» → источники категории",
                    len(answer.sources) > 0,
                )
                detected = plan_query(
                    LocalRagStubClient(),
                    question,
                    [],
                    taxonomy,
                    engine.retriever.retrieve(question, top_k=5),
                )
                checker.check(
                    f"«{question}» → категория {expected_category}",
                    detected.category == expected_category,
                    f"получена {detected.category}",
                )

            specific_plan = plan_query(
                LocalRagStubClient(),
                "покажи рецепт стейк рибай",
                [],
                taxonomy,
                engine.retriever.retrieve("покажи рецепт стейк рибай", top_k=5),
            )
            checker.check(
                "Конкретное блюдо → specific_dish", specific_plan.intent == "specific_dish"
            )
            browse_plan = plan_query(
                LocalRagStubClient(),
                "что можно приготовить?",
                [],
                taxonomy,
                engine.retriever.retrieve("что можно приготовить?", top_k=5),
            )
            checker.check("Общий вопрос → browse", browse_plan.intent == "browse")

            menu = _category_menu(taxonomy, 8)
            checker.check(
                "Меню категорий вместо тупика",
                "говядина" in menu and "птица" in menu,
            )
            # Заведомо «пустая» подборка (высокие пороги) — вместо тупика меню.
            unknown = timed_rag_query(
                client=LocalRagStubClient(),
                retriever=engine.retriever,
                question="подбери что-нибудь необычное",
                top_k=5,
                relevance_threshold=0.99,
                generic_threshold=0.99,
                taxonomy=taxonomy,
            )
            checker.check(
                "Непонятная подборка → меню, не отказ",
                unknown["route"] == "menu"
                and unknown["answer"].content != NO_RELEVANT_INFO_MESSAGE
                and "категор" in unknown["answer"].content.lower(),
                f"route={unknown['route']}",
            )

            checker.section("4e. Запрос-список «собери N блюд»")
            checker.check(
                "Число блюд распознано («10 блюд»)",
                extract_requested_count("собери 10 блюд для банкета") == 10,
            )
            checker.check(
                "Число блюд распознано («топ 5 рецептов»)",
                extract_requested_count("топ 5 рецептов") == 5,
            )
            list_plan = plan_query(
                LocalRagStubClient(),
                "собери 10 блюд для банкета",
                [],
                taxonomy,
                engine.retriever.retrieve("собери 10 блюд для банкета", top_k=5),
            )
            checker.check(
                "«собери 10 блюд» → intent browse",
                list_plan.intent == "browse",
                f"получен {list_plan.intent}",
            )
            checker.check(
                "«собери 10 блюд» → limit 10", list_plan.limit == 10,
                f"получен {list_plan.limit}",
            )
            list_result = timed_rag_query(
                client=LocalRagStubClient(),
                retriever=engine.retriever,
                question="собери 10 блюд для банкета",
                top_k=5,
                relevance_threshold=0.33,
                taxonomy=taxonomy,
            )
            checker.check(
                "«собери 10 блюд» → маршрут list",
                list_result["route"] == "list",
                f"получен {list_result['route']}",
            )
            list_sources = list_result["answer"].sources
            checker.check(
                "Список → несколько разных рецептов",
                len({s.recipe_id for s in list_sources}) >= 3,
                f"рецептов {len({s.recipe_id for s in list_sources})}",
            )
            checker.check(
                "Ответ-список реально перечисляет блюда",
                list_result["answer"].content.count("\n- «") >= 3,
            )
            top_result = timed_rag_query(
                client=LocalRagStubClient(),
                retriever=engine.retriever,
                question="топ 5 рецептов",
                top_k=5,
                relevance_threshold=0.33,
                taxonomy=taxonomy,
            )
            checker.check(
                "«топ 5 рецептов» → list, не больше 5 источников",
                top_result["route"] == "list"
                and 0 < len(top_result["answer"].sources) <= 5,
                f"route={top_result['route']} sources={len(top_result['answer'].sources)}",
            )

            checker.section("4f. LLM-планировщик: число блюд (limit)")
            llm_plan = _llm_plan(
                _JsonPlannerClient(
                    '{"intent": "browse", "query": "блюда для банкета", '
                    '"category": null, "dish": null, "limit": 10}'
                ),
                "накрой стол на десять человек",
                [],
                taxonomy,
            )
            checker.check(
                "LLM-план → intent browse",
                llm_plan is not None and llm_plan.intent == "browse",
            )
            checker.check(
                "LLM-план → limit 10", llm_plan is not None and llm_plan.limit == 10
            )
            string_limit = _llm_plan(
                _JsonPlannerClient('{"intent": "browse", "query": "x", "limit": "8"}'),
                "q",
                [],
                taxonomy,
            )
            checker.check(
                "LLM-план: limit из строки «8»",
                string_limit is not None and string_limit.limit == 8,
            )
            bad_limit = _llm_plan(
                _JsonPlannerClient('{"intent": "browse", "query": "x", "limit": 0}'),
                "q",
                [],
                taxonomy,
            )
            checker.check(
                "LLM-план: невалидный limit → None",
                bad_limit is not None and bad_limit.limit is None,
            )
            specific_limit = _llm_plan(
                _JsonPlannerClient(
                    '{"intent": "specific_dish", "query": "стейк", '
                    '"dish": "стейк", "limit": 5}'
                ),
                "q",
                [],
                taxonomy,
            )
            checker.check(
                "LLM-план: limit игнорируется для specific_dish",
                specific_limit is not None and specific_limit.limit is None,
            )
            llm_routed = plan_query(
                _JsonPlannerClient(
                    '{"intent": "browse", "query": "блюда для застолья", "limit": 10}'
                ),
                "накрой стол на десять человек",
                [],
                taxonomy,
                [],
            )
            checker.check(
                "plan_query → LLM-ветка browse + limit",
                llm_routed.source == "llm" and llm_routed.limit == 10,
                f"source={llm_routed.source} limit={llm_routed.limit}",
            )

            checker.section("4b. Несколько чатов на пользователя")
            resp = client.post("/api/chats", json={"title": "Второй чат"})
            second_id = resp.get_json().get("chat", {}).get("id")
            checker.check("Второй чат создан", resp.status_code == 201 and second_id)
            resp = client.get("/api/chats")
            checker.check("Список: 2 чата", len(resp.get_json().get("chats", [])) == 2)
            resp = client.patch(f"/api/chats/{second_id}", json={"title": "Переименован"})
            checker.check("PATCH переименование → 200", resp.status_code == 200)
            checker.check(
                "Название обновилось",
                resp.get_json().get("chat", {}).get("title") == "Переименован",
            )
            resp = client.delete(f"/api/chats/{second_id}")
            checker.check("DELETE чата → 200", resp.status_code == 200)
            resp = client.get("/api/chats")
            checker.check("Список: снова 1 чат", len(resp.get_json().get("chats", [])) == 1)

            checker.section("5. Изоляция пользователей")
            bob = app.test_client()
            resp = bob.post(
                "/api/register", json={"username": "bob", "password": "secret2"}
            )
            checker.check("Регистрация bob → 201", resp.status_code == 201)
            resp = bob.get(f"/api/chats/{chat_id}/messages")
            checker.check("bob не видит чужой чат → 403", resp.status_code == 403)
            resp = bob.post(
                f"/api/chats/{chat_id}/messages",
                json={"message": "дай рецепт", "use_rag": True},
            )
            checker.check("bob не пишет в чужой чат → 403", resp.status_code == 403)
            resp = bob.get("/api/chats")
            checker.check("Список чатов bob пуст", resp.get_json().get("chats") == [])

            checker.section("6. Очистка и выход")
            resp = client.post(f"/api/chats/{chat_id}/reset")
            checker.check("POST reset → 200", resp.status_code == 200)
            resp = client.get(f"/api/chats/{chat_id}/messages")
            checker.check("История очищена", resp.get_json().get("messages") == [])
            resp = client.get("/api/chats/999999/messages")
            checker.check("Несуществующий чат → 404", resp.status_code == 404)
            resp = client.post("/api/logout")
            checker.check("Выход → 200", resp.status_code == 200)
            resp = client.get("/api/me")
            checker.check("После выхода → 401", resp.status_code == 401)
        finally:
            engine.close()

    print(f"\nИтог: {checker.total - len(checker.failures)}/{checker.total} проверок пройдено.")
    if checker.failures:
        print("Провалено: " + ", ".join(checker.failures))
        return 1
    print("Все офлайн-проверки сервиса пройдены.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Офлайн e2e проверки сервиса рецептов.")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="использовать детерминированный стаб (по умолчанию)",
    )
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
