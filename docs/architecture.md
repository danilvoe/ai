# Архитектура проекта

## Назначение и принцип веток

Репозиторий — учебный: каждая git-ветка соответствует одному дню курса
(`day2` … `day31`). Код развивается **накопительно**, поэтому ветка `day31`
содержит решения всех предыдущих дней. Итоговая (`master`) ветка — самая полная.

## Структура каталогов

| Каталог / файл | Роль |
| --- | --- |
| `agent/` | Ядро на Python: агент, память, task state, MCP-клиент, RAG, локальная LLM, веб-чат, сценарии |
| `mcp/` | Автономные MCP-серверы на FastMCP (GitFlic, pipeline, report, project) |
| `recipe_chat_service/` | Итоговый веб-сервис Дня 30 (Flask + SQLite, самодостаточный `core/`) |
| `scripts/` | Сборка `llama.cpp`, генерация документации |
| `docs/` | Документация проекта (индексируется ассистентом разработчика) |
| `history/` | Генерируемые артефакты: индексы FAISS, отчёты, сессии (в Git не хранится) |
| `chat.py`, `chat_rag.py`, `chat_local.py`, `web_local.py`, `chat_dev.py` | CLI- и веб-точки входа |
| `config.json` / `config.one.json` | Локальная конфигурация (ключи API, модели); `config.example.json` — шаблон |

## Подсистемы

### A. Контекст и память
- **`agent/tokens.py`** — подсчёт токенов и учёт лимитов.
- **`agent/compression.py`** — сжатие истории (суммаризация).
- **`agent/context.py`** — стратегии контекста (Sliding Window, Sticky Facts, Branching).
- **`agent/memory.py`** — трёхуровневая память (`MemoryStore`, `MemoryLayer`).
- **`agent/personalization.py`** — профиль пользователя поверх памяти.

### B. Task State и guardrails
- **`agent/task_state.py`** — конечный автомат стадий задачи (`TaskStage`, `TaskState`).
- **`agent/invariants.py`** — проверка архитектурных и бизнес-инвариантов.
- **`agent/transition_scenarios.py`** — защита от некорректных переходов стадий.

### C. MCP (Model Context Protocol)
- **`agent/mcp_client.py`** — клиент Streamable HTTP и проверка соединения.
- **`agent/mcp_tools.py`** — stdio-клиент (`McpToolRuntime`), планирование вызова.
- **`agent/orchestrator.py`** — мульти-серверный оркестратор (`McpOrchestrator`, `RoutedTool`).
- **`mcp/gitflic_server.py`**, **`mcp/gitflic_scheduler.py`**, **`mcp/gitflic_cron.py`** — GitFlic и фоновые задачи.
- **`mcp/pipeline_server.py`**, **`mcp/pipeline_text.py`** — композиция search → summarize → saveToFile.
- **`mcp/report_server.py`** — журнал и индекс отчётов.
- **`mcp/project_server.py`** — доступ к проекту: git-ветка, файлы, коммиты, diff (День 31).

### D. RAG и поиск
- **`agent/indexing.py`** — чанкинг (`chunk_fixed_size`, `chunk_structural`), эмбеддеры
  (`HashingEmbedder`, `OpenAIEmbedder`), индекс `FaissIndex`.
- **`agent/rag.py`** — базовый RAG (`RagRetriever`, `RagAgent`, промпты).
- **`agent/reranking.py`** — двухстадийный RAG: переформулировка, реранкинг, фильтр релевантности.
- **`agent/grounding.py`** — цитаты, порог анти-галлюцинаций, выравнивание цитат.
- **`agent/rag_chat.py`**, **`agent/rag_chat_cli.py`** — мини-чат с RAG и памятью задачи.
- **`agent/project_docs.py`** — индексация документации проекта (День 31).

### E. Локальная LLM (llama.cpp)
- **`agent/local_llm.py`** — `LocalLLMConfig`, `LocalLLMServer`, `StubLocalLLM`, запуск `llama-server`.
- **`agent/local_rag.py`** — полностью локальный RAG (retrieval + локальная генерация).
- **`agent/optimization.py`** — параметры генерации, квантизация, prompt-шаблон под кейс.
- **`scripts/setup_llama_cpp.sh`** — сборка `llama.cpp` и загрузка GGUF-моделей.

### F. Веб-приложения
- **`agent/web_chat.py`** — Flask-бэкенд чата (`WebChatBackend`, `ChatSession`, `ChatChannel`).
- **`web_local.py`** — точка входа веб-чата (локальная/облачная модель, RAG-режим).
- **`recipe_chat_service/`** — многопользовательский сервис рецептов (День 30).

### G. Ассистент разработчика (День 31)
- **`agent/project_docs.py`** — загрузка и индексация README + `docs/`.
- **`agent/dev_assistant.py`** — ассистент: RAG по докам + MCP-контекст проекта.
- **`agent/dev_assistant_cli.py`** / **`chat_dev.py`** — CLI с командой `/help`.

## Поток данных RAG-запроса

```
вопрос → RagRetriever (FAISS-поиск) → реранкинг/фильтр релевантности
       → сборка контекста с источниками → промпт → LLM → ответ + цитаты
```

## Поток данных ассистента разработчика

```
/help <вопрос> → project_docs: поиск по документации (FAISS)
               → MCP project_server: git-ветка / файлы / diff
               → промпт (доки + контекст проекта) → LLM → ответ + источники
```

## Ключевые сквозные идеи курса

- **Детерминированный офлайн-режим.** Почти каждая подсистема имеет стаб
  (`HashingEmbedder`, `StubLocalLLM`, `*StubClient`), поэтому проверки
  выполняются без сети и без затрат на API.
- **Grounding и цитирование.** Ответы RAG-агентов всегда сопровождаются
  источниками (`chunk_id`, файл/раздел).
- **MCP как способ подключения инструментов.** Агент получает доступ к внешним
  данным (GitFlic, файлы проекта) через стандартизованные MCP-серверы.
