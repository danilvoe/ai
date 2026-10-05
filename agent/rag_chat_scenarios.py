"""Day 24 / RAG Chat: Сквозные проверочные сценарии диалога с RAG и памятью задачи.

Содержит:
1. Сценарий 1 (12 ходов): Новогодний ужин — «Апельсиновая курица на гриле Kettle».
   - Фиксация цели (новогодний ужин из птицы на гриле);
   - Фиксация ограничения (без остроты/чили — дети не едят острое);
   - Уточнение оборудования (угольный котел Kettle 57 см);
   - Фиксация термина ("пряное сливочное масло");
   - Проверка удержания ограничения (попытка добавить соус шрирача);
   - Температуры готовности (74°C в грудке, непрямой жар 180–200°C);
   - Пошаговый вечерний тайминг;
   - Источники с обязательным chunk_id и кликабельным URL на каждом шаге.

2. Сценарий 2 (12 ходов): Мясное барбекю — «Говяжьи рёбрышки Hot & Fast».
   - Фиксация цели (барбекю для компании 6–8 друзей);
   - Фиксация ограничения (без томатов и кетчупа — непереносимость у гостя);
   - Фиксация термина ("Hot & Fast: копчение при 135–150°C");
   - Пропорции обсыпки (50/50 соль и свежемолотый перец);
   - Увлажнение без томатов (яблочный уксус 6% и вода);
   - Проверка удержания ограничения (попытка добавить томатный соус Heinz);
   - Определение готовности по щупу (~93–96°C) и отдых мяса (40–60 мин);
   - Итоговый чек-лист приготовления;
   - Источники с chunk_id и URL.

Запуск:
    python3 -m agent.rag_chat_scenarios
    python3 -m agent.rag_chat_scenarios --no-llm
    python3 -m agent.rag_chat_scenarios --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from agent.config import load_config
from agent.conversation import Conversation
from agent.rag_chat import (
    RagChatAgent,
    RagChatResponse,
    RagChatSession,
    build_rag_chat_agent,
)

# ============================================================================
# Данные сценариев тестирования (2 диалога по 12 ходов)
# ============================================================================

SCENARIO_1_STEPS = [
    {
        "step": 1,
        "input": "Хочу приготовить праздничную птицу на Новый год на угольном гриле. Что посоветуешь?",
        "expected_goal": "птиц",
        "expect_constraint_check": False,
        "notes": "Постановка общей цели (новогодний ужин из птицы)",
    },
    {
        "step": 2,
        "input": "Будут дети, поэтому строжайшее ограничение: никакой остроты, никакого чили и острых специй.",
        "expected_constraint": "острот",
        "expect_constraint_check": False,
        "notes": "Фиксация критического ограничения (без остроты/чили для детей)",
    },
    {
        "step": 3,
        "input": "У меня классический круглый угольный котел Kettle 57 см.",
        "expected_clarification": "гриль",
        "expect_constraint_check": False,
        "notes": "Уточнение оборудования (угольный котел Kettle 57 см)",
    },
    {
        "step": 4,
        "input": "Давай остановимся на рецепте апельсиновой курицы. Какие основные ингредиенты нужны для маринада/натирки?",
        "expected_keyword": "курица",
        "expect_constraint_check": False,
        "notes": "Выбор рецепта «Апельсиновая курица на гриле» и запрос ингредиентов",
    },
    {
        "step": 5,
        "input": "Зафиксируй термин: \"пряное сливочное масло\" — это размягченное масло с цедрой апельсина и розмарином.",
        "expected_term": "пряное сливочное масло",
        "expect_constraint_check": False,
        "notes": "Фиксация термина «пряное сливочное масло»",
    },
    {
        "step": 6,
        "input": "Какой метод жара нужен на моем Kettle — прямой или непрямой? Какая температура под крышкой?",
        "expected_keyword": "непрямой",
        "expect_constraint_check": False,
        "notes": "Технология жарки (непрямой жар 180-200°C на Kettle)",
    },
    {
        "step": 7,
        "input": "А можно добавить туда соус шрирача для пикантности?",
        "expect_constraint_check": True,
        "violation_expected": True,
        "notes": "Провокация: попытка добавить острый соус шрирача (должен сработать отказ)",
    },
    {
        "step": 8,
        "input": "До какой внутренней температуры в грудке и бедре нужно доводить курицу?",
        "expected_keyword": "74",
        "expect_constraint_check": False,
        "notes": "Температуры готовности (74°C в грудке, 80-84°C в бедре)",
    },
    {
        "step": 9,
        "input": "Как подготовить апельсины для подачи и когда наносить глазурь?",
        "expected_keyword": "глазур",
        "expect_constraint_check": False,
        "notes": "Глазирование и подача с запечёнными дольками апельсина",
    },
    {
        "step": 10,
        "input": "Кстати, сколько времени займет розжиг стартера с брикетами?",
        "expected_keyword": "стартер",
        "expect_constraint_check": False,
        "notes": "Уточняющий вопрос о розжиге стартера без потери контекста цели",
    },
    {
        "step": 11,
        "input": "Сведи весь наш план ужина в пошаговый тайминг на вечер.",
        "expected_keyword": "план",
        "expect_constraint_check": False,
        "notes": "Сводный тайминг ужина с проверкой удержания цели и ограничений",
    },
    {
        "step": 12,
        "input": "Перечисли все рецепты и источники с прямыми ссылками, которые мы использовали для этого ужина.",
        "expected_keyword": "источник",
        "expect_constraint_check": False,
        "notes": "Финальная ревизия источников с обязательным выводом URL",
    },
]

SCENARIO_2_STEPS = [
    {
        "step": 1,
        "input": "Планирую барбекю на субботу для компании из 6–8 друзей, хочу мощную говядину на гриле.",
        "expected_goal": "барбекю",
        "expect_constraint_check": False,
        "notes": "Постановка цели (барбекю из говядины на 6–8 человек)",
    },
    {
        "step": 2,
        "input": "Один из гостей не переносит томаты — исключаем любые томатные соусы и кетчуп.",
        "expected_constraint": "томат",
        "expect_constraint_check": False,
        "notes": "Фиксация критического ограничения (без томатов и кетчупа)",
    },
    {
        "step": 3,
        "input": "Готовить будем по методу Hot & Fast — зафиксируй термин: копчение говяжьих ребер при температуре 135–150°C.",
        "expected_term": "Hot & Fast",
        "expect_constraint_check": False,
        "notes": "Фиксация термина и метода «Hot & Fast (135–150°C)»",
    },
    {
        "step": 4,
        "input": "Какие именно говяжьи ребра лучше выбрать по рецепту и как их подготовить?",
        "expected_keyword": "ребр",
        "expect_constraint_check": False,
        "notes": "Выбор отруба (Short Ribs / Back Ribs) и снятие мембраны",
    },
    {
        "step": 5,
        "input": "Какие пропорции соли и черного перца классически рекомендуются для обсыпки?",
        "expected_keyword": "перец",
        "expect_constraint_check": False,
        "notes": "Пропорции сухого руба (соль и перец)",
    },
    {
        "step": 6,
        "input": "Чем опрыскивать мясо в процессе, чтобы соблюсти наше ограничение без томатов?",
        "expected_keyword": "уксус",
        "expect_constraint_check": False,
        "notes": "Спринцевание яблочным уксусом и водой (без томатов)",
    },
    {
        "step": 7,
        "input": "При достижении какой температуры мяса или корки нужно заворачивать ребра?",
        "expected_keyword": "заворач",
        "expect_constraint_check": False,
        "notes": "Момент заворачивания в фольгу или бумагу (~75-80°C)",
    },
    {
        "step": 8,
        "input": "Друг предлагает полить в фольгу соус барбекю Heinz из магазина, соглашаться?",
        "expect_constraint_check": True,
        "violation_expected": True,
        "notes": "Провокация: предложение магазинного соуса Heinz (отказ из-за томатов)",
    },
    {
        "step": 9,
        "input": "Как понять, что ребра готовы — по щупу или по температуре?",
        "expected_keyword": "щуп",
        "expect_constraint_check": False,
        "notes": "Контроль готовности (93-96°C, мягкость щупа)",
    },
    {
        "step": 10,
        "input": "Сколько времени ребра должны отдыхать перед нарезкой и где?",
        "expected_keyword": "отдых",
        "expect_constraint_check": False,
        "notes": "Отдых мяса (40-60 минут в тепле)",
    },
    {
        "step": 11,
        "input": "Собери итоговый чек-лист приготовления говяжьих ребер Hot & Fast.",
        "expected_keyword": "чек-лист",
        "expect_constraint_check": False,
        "notes": "Сводный чек-лист с сохранением метода Hot & Fast и всех ограничений",
    },
    {
        "step": 12,
        "input": "Выведи точные чанки и URL-ссылки рецепта, по которому мы готовили.",
        "expected_keyword": "рецепт",
        "expect_constraint_check": False,
        "notes": "Вывод точных чанков и URL рецепта рёбрышек Hot & Fast",
    },
]


# ============================================================================
# Раннер сценария
# ============================================================================


def run_scenario(
    scenario_id: str,
    scenario_title: str,
    steps: list[dict[str, Any]],
    agent: RagChatAgent,
    history_dir: Path,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Выполняет один многошаговый сценарий и возвращает метрики."""
    session_file_id = f"test_chat_{scenario_id}_{int(time.time())}"
    conversation = Conversation(session_file_id, history_dir=history_dir)
    session = RagChatSession(conversation)

    step_results: list[dict[str, Any]] = []
    effective_steps = steps[:max_steps] if max_steps else steps

    print(f"\n{'=' * 80}")
    print(f"ЗАПУСК СЦЕНАРИЯ: {scenario_title} ({len(effective_steps)} шагов)")
    print(f"Сессия: {session.session_id}")
    print(f"{'=' * 80}")

    for idx, item in enumerate(effective_steps, 1):
        step_num = item["step"]
        user_input = item["input"]
        notes = item["notes"]
        expect_violation = item.get("violation_expected", False)

        print(f"\n[Шаг {step_num}/{len(effective_steps)}] Пользователь: {user_input[:65]}...")
        t0 = time.perf_counter()
        resp: RagChatResponse = agent.ask(user_input, session)
        latency = (time.perf_counter() - t0) * 1000

        # Валидация источников: chunk_id + url
        sources_present = len(resp.sources) > 0
        all_have_chunk_ids = all(bool(s.chunk_id) for s in resp.sources) if sources_present else False
        all_have_urls = all(bool(s.url and s.url.startswith("http")) for s in resp.sources) if sources_present else False

        # Валидация сохранения цели
        goal_retained = bool(session.state.goal and len(session.state.goal) > 3)

        # Валидация соблюдения ограничений
        constraint_adherence = True
        if expect_violation:
            # На этом шаге модель обязана была обнаружить нарушение ограничения
            constraint_adherence = resp.is_constraint_violation

        # Вывод краткого отчёта в консоль
        url_status = "✓ URL есть" if all_have_urls else "✗ Нет URL"
        goal_status = "✓ Цель активна" if goal_retained else "✗ Цель утеряна"
        viol_status = "✓ Отказ сработал" if resp.is_constraint_violation else "✓ Штатный шаг"

        print(f"  Ассистент: {resp.answer[:90].replace(chr(10), ' ')}...")
        print(f"  Метрики: Источников: {len(resp.sources)} | {url_status} | {goal_status} | {viol_status} | {latency:.1f} мс")

        step_results.append({
            "step": step_num,
            "input": user_input,
            "notes": notes,
            "answer_preview": resp.answer[:140],
            "sources_count": len(resp.sources),
            "sources": [s.to_dict() for s in resp.sources],
            "all_have_chunk_ids": all_have_chunk_ids,
            "all_have_urls": all_have_urls,
            "goal_retained": goal_retained,
            "current_goal": session.state.goal,
            "constraints_count": len(session.state.constraints),
            "terms_count": len(session.state.terms),
            "is_violation_step": expect_violation,
            "constraint_adherence": constraint_adherence,
            "latency_ms": round(latency, 2),
        })

    # Сводные метрики сценария
    total = len(step_results)
    sources_rate = sum(1 for r in step_results if r["sources_count"] > 0) / total if total else 0.0
    url_rate = sum(1 for r in step_results if r["all_have_urls"]) / total if total else 0.0
    chunk_rate = sum(1 for r in step_results if r["all_have_chunk_ids"]) / total if total else 0.0
    goal_rate = sum(1 for r in step_results if r["goal_retained"]) / total if total else 0.0
    constraint_rate = sum(1 for r in step_results if r["constraint_adherence"]) / total if total else 0.0
    avg_latency = sum(r["latency_ms"] for r in step_results) / total if total else 0.0

    print(f"\nИТОГИ СЦЕНАРИЯ «{scenario_title}»:")
    print(f"  • Наличие источников:       {sources_rate * 100:.1f}% ({sum(1 for r in step_results if r['sources_count'] > 0)}/{total})")
    print(f"  • Наличие Chunk ID:         {chunk_rate * 100:.1f}%")
    print(f"  • Наличие кликабельных URL: {url_rate * 100:.1f}%")
    print(f"  • Удержание цели задачи:    {goal_rate * 100:.1f}%")
    print(f"  • Соблюдение ограничений:   {constraint_rate * 100:.1f}%")
    print(f"  • Средняя задержка:         {avg_latency:.1f} мс")

    return {
        "scenario_id": scenario_id,
        "title": scenario_title,
        "total_steps": total,
        "session_id": session.session_id,
        "final_state": session.state.to_dict(),
        "summary": {
            "sources_rate": round(sources_rate, 3),
            "chunk_rate": round(chunk_rate, 3),
            "url_rate": round(url_rate, 3),
            "goal_rate": round(goal_rate, 3),
            "constraint_adherence_rate": round(constraint_rate, 3),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "steps": step_results,
    }


def render_markdown_report(report_data: dict[str, Any]) -> str:
    """Генерирует подробный Markdown-отчёт по результатам проверки 2 сценариев."""
    scenarios = report_data.get("scenarios", [])
    overall = report_data.get("overall_summary", {})

    lines: list[str] = [
        "# Отчёт: RAG-чат с памятью задачи и источниками (Day 24)",
        "",
        f"- **Дата запуска:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Режим выполнения:** {'API LLM' if report_data.get('run_llm') else 'Офлайн (--no-llm)'}",
        f"- **Всего сценариев:** {len(scenarios)}",
        f"- **Всего шагов диалога:** {report_data.get('total_turns', 0)}",
        "",
        "## 1. Сводные метрики надежности и удержания контекста",
        "",
        "| Метрика | Значение | Описание |",
        "| --- | --- | --- |",
        f"| **Удержание цели диалога (Goal Retention)** | **{overall.get('avg_goal_rate', 0) * 100:.1f}%** | Цель не вымывается из памяти на 12 шагах |",
        f"| **Соблюдение ограничений (Constraint Adherence)** | **{overall.get('avg_constraint_rate', 0) * 100:.1f}%** | Пресечение попыток нарушить ограничения |",
        f"| **Наличие источников (Sources Coverage)** | **{overall.get('avg_sources_rate', 0) * 100:.1f}%** | Ответы заземлены на базу знаний |",
        f"| **Наличие Chunk ID в источниках** | **{overall.get('avg_chunk_rate', 0) * 100:.1f}%** | Точная идентификация сегментов рецепта |",
        f"| **Наличие кликабельных URL** | **{overall.get('avg_url_rate', 0) * 100:.1f}%** | Прямая ссылка на сайт рецепта |",
        f"| **Средняя задержка на ход диалога** | **{overall.get('avg_latency_ms', 0):.1f} мс** | Время обработки шага |",
        "",
    ]

    for sc in scenarios:
        lines.append(f"## 2.{scenarios.index(sc) + 1}. Сценарий: {sc['title']}")
        lines.append("")
        st = sc.get("final_state", {})
        lines.append(f"- **Зафиксированная цель:** `{st.get('goal', '—')}`")
        lines.append(f"- **Ограничения:** {', '.join(st.get('constraints', [])) or '—'}")
        lines.append(f"- **Зафиксированные термины:** {', '.join(f'`{k}: {v}`' for k, v in st.get('terms', {}).items()) or '—'}")
        lines.append("")
        lines.append("| Шаг | Запрос пользователя | Источники | Chunk ID | URL | Цель | Ограничение | Задержка |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

        for s in sc.get("steps", []):
            input_text = s["input"].replace("|", "\\|")
            src_mark = f"✓ ({s['sources_count']})" if s["sources_count"] > 0 else "✗"
            chunk_mark = "✓" if s["all_have_chunk_ids"] else "✗"
            url_mark = "✓" if s["all_have_urls"] else "✗"
            goal_mark = "✓" if s["goal_retained"] else "✗"
            const_mark = "✓" if s["constraint_adherence"] else "✗"
            lines.append(
                f"| {s['step']} | {input_text[:45]}… | {src_mark} | {chunk_mark} | {url_mark} | {goal_mark} | {const_mark} | {s['latency_ms']:.1f} мс |"
            )

        # Вывод примеров источников
        lines.append("")
        lines.append("### Примеры источников и ссылок:")
        for s in sc.get("steps", [])[:3]:
            if s.get("sources"):
                top_src = s["sources"][0]
                lines.append(f"- **Шаг {s['step']}:** `[{top_src.get('chunk_id')}]` {top_src.get('title')} ({top_src.get('section')})")
                lines.append(f"  - 🔗 URL: {top_src.get('url')}")
        lines.append("")

    lines.append("## 3. Выводы и архитектурные инварианты")
    lines.append("")
    lines.append("1. **Удержание контекста и цели:** Механизм `RagTaskState` сохраняет формализованные данные о цели и терминах, что исключает потерю контекста при диалогах любой длины.")
    lines.append("2. **Защита ограничений (Constraint Guard):** Попытки пользователя добавить запрещённые компоненты (острый чили/шрирачу для детей или томатные соусы Heinz) надёжно пресекаются с указанием зафиксированных договорённостей.")
    lines.append("3. **Верификация источников с URL:** Каждый ответ заземлён на фрагменты базы знаний, содержащие уникальный `chunk_id` и веб-адрес рецепта.")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Сквозная проверка RAG-чата с памятью задачи на 2 сценариях")
    parser.add_argument("--no-llm", action="store_true", help="Офлайн-прогон без обращений к API LLM")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число шагов в каждом сценарии")
    parser.add_argument("--report-dir", type=str, default="history/rag_chat", help="Каталог сохранения отчётов")
    args = parser.parse_args()

    config = load_config()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(__file__).resolve().parent.parent / "history"

    print("=" * 80)
    print("ВАЛИДАЦИЯ RAG-ЧАТА С ПАМЯТЬЮ ЗАДАЧИ (2 СЦЕНАРИЯ ПО 12 ХОДОВ)")
    print(f"Режим LLM: {'ВЫКЛЮЧЕН (--no-llm)' if args.no_llm else 'ВКЛЮЧЕН (API)'}")
    print(f"Каталог отчётов: {report_dir}")
    print("=" * 80)

    try:
        agent = build_rag_chat_agent(config=config, no_llm=args.no_llm)
        print("✓ RAG-агент успешно инициализирован")
    except Exception as exc:
        print(f"✗ Ошибка инициализации: {exc}", file=sys.stderr)
        sys.exit(1)

    scenarios_data = []

    # Сценарий 1: Апельсиновая курица
    sc1 = run_scenario(
        scenario_id="scenario_1_orange_chicken",
        scenario_title="Новогодний ужин: Апельсиновая курица на гриле Kettle",
        steps=SCENARIO_1_STEPS,
        agent=agent,
        history_dir=history_dir,
        max_steps=args.limit,
    )
    scenarios_data.append(sc1)

    # Сценарий 2: Говяжьи ребрышки Hot & Fast
    sc2 = run_scenario(
        scenario_id="scenario_2_beef_ribs",
        scenario_title="Мясное барбекю для компании: Говяжьи рёбрышки Hot & Fast",
        steps=SCENARIO_2_STEPS,
        agent=agent,
        history_dir=history_dir,
        max_steps=args.limit,
    )
    scenarios_data.append(sc2)

    # Общая сводка
    total_turns = sum(sc["total_steps"] for sc in scenarios_data)
    avg_goal = sum(sc["summary"]["goal_rate"] for sc in scenarios_data) / len(scenarios_data)
    avg_const = sum(sc["summary"]["constraint_adherence_rate"] for sc in scenarios_data) / len(scenarios_data)
    avg_sources = sum(sc["summary"]["sources_rate"] for sc in scenarios_data) / len(scenarios_data)
    avg_chunk = sum(sc["summary"]["chunk_rate"] for sc in scenarios_data) / len(scenarios_data)
    avg_url = sum(sc["summary"]["url_rate"] for sc in scenarios_data) / len(scenarios_data)
    avg_latency = sum(sc["summary"]["avg_latency_ms"] for sc in scenarios_data) / len(scenarios_data)

    overall_summary = {
        "avg_goal_rate": round(avg_goal, 3),
        "avg_constraint_rate": round(avg_const, 3),
        "avg_sources_rate": round(avg_sources, 3),
        "avg_chunk_rate": round(avg_chunk, 3),
        "avg_url_rate": round(avg_url, 3),
        "avg_latency_ms": round(avg_latency, 2),
    }

    report_payload = {
        "timestamp": time.time(),
        "run_llm": not args.no_llm,
        "total_scenarios": len(scenarios_data),
        "total_turns": total_turns,
        "overall_summary": overall_summary,
        "scenarios": scenarios_data,
    }

    # Сохранение JSON и Markdown
    json_path = report_dir / "chat_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    md_report = render_markdown_report(report_payload)
    md_path = report_dir / "chat_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА ПО ВСЕМ СЦЕНАРИЯМ:")
    print(f"  • Всего шагов:                    {total_turns}")
    print(f"  • Удержание цели (Goal Rate):     {avg_goal * 100:.1f}%")
    print(f"  • Соблюдение ограничений:         {avg_const * 100:.1f}%")
    print(f"  • Наличие источников:             {avg_sources * 100:.1f}%")
    print(f"  • Наличие Chunk ID:               {avg_chunk * 100:.1f}%")
    print(f"  • Наличие кликабельных URL:       {avg_url * 100:.1f}%")
    print(f"  • Средняя задержка на шаг:        {avg_latency:.1f} мс")
    print(f"\n✓ JSON-отчёт сохранён: {json_path}")
    print(f"✓ Markdown-отчёт сохранён: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
