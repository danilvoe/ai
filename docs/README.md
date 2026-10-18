# Документация проекта AI Advent

Учебный репозиторий по курсу **AI Advent (LLM & Agents)** — практические решения
для Дней 1–32. Каждая git-ветка (`day2`, `day3`, …, `day32`, `master`) фиксирует
состояние кода на соответствующем уроке, поэтому проект развивается накопительно:
`day32` содержит всё, что было в предыдущих днях.

## Разделы документации

| Документ | О чём |
| --- | --- |
| [architecture.md](architecture.md) | Архитектура, подсистемы и карта модулей |
| [data-schemas.md](data-schemas.md) | Схемы данных: конфиг, рецепты, сессии, индексы FAISS |
| [api.md](api.md) | HTTP API, MCP-серверы и инструменты, CLI-точки входа |
| [development.md](development.md) | Правила разработки, офлайн-проверки, линтинг |
| [modules.md](modules.md) | Автогенерируемый справочник по модулям `agent/` и `mcp/` |

## Как этим пользоваться

Документацию читает **ассистент разработчика** (День 31). Он индексирует README и
эту папку в локальный векторный индекс FAISS и отвечает на вопросы о проекте
командой `/help`:

```bash
python3 -m agent.project_docs --build      # собрать индекс документации (офлайн)
python3 chat_dev.py                        # запустить ассистента разработчика
python3 chat_dev.py --no-llm               # офлайн-режим без обращения к LLM API
```

Помимо документации ассистент подключается к самому проекту через MCP-сервер
[`mcp/project_server.py`](../mcp/project_server.py) и умеет сообщать текущую
git-ветку, список файлов, историю коммитов и diff.

## Автоматическое ревью PR (День 32)

Пайплайн GitHub Action [`.github/workflows/code-review.yml`](../.github/workflows/code-review.yml)
на каждый `pull_request` получает diff и изменённые файлы через MCP-сервер
[`mcp/review_server.py`](../mcp/review_server.py), ищет контекст в двух RAG-индексах
(документация + исходный код) и публикует ревью комментарием к PR: потенциальные
баги, архитектурные проблемы и рекомендации.

```bash
python3 -m agent.project_docs --build      # индекс документации
python3 -m agent.code_index --build        # индекс исходного кода
python3 review_pr.py --base HEAD~1 --head HEAD --offline   # локальный прогон
python3 -m agent.review_scenarios --offline                # офлайн-проверка
```

## Обзор репозитория

- **`agent/`** — ядро: агент и CLI, память, task state, MCP-клиент, RAG-подсистема
  (индексация, поиск, реранкинг, grounding), локальная LLM, веб-чат.
- **`mcp/`** — автономные FastMCP-серверы: `gitflic_server`, `pipeline_server`,
  `report_server`, `project_server`, `review_server`.
- **`recipe_chat_service/`** — итоговый проект Дня 30: многопользовательский
  сервис рецептов на Flask + SQLite с самодостаточной папкой `core/`.
- **`scripts/`** — вспомогательные скрипты: сборка `llama.cpp`, генерация
  документации.
- **`docs/`** — эта документация.
- **`history/`** — генерируемые артефакты (индексы FAISS, отчёты, сессии); в Git
  не коммитится.

## С чего начать

1. [architecture.md](architecture.md) — понять, как устроены подсистемы.
2. [api.md](api.md) — узнать про MCP-инструменты и HTTP API.
3. [development.md](development.md) — как запускать и проверять изменения офлайн.
