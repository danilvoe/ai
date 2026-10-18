# Схемы данных

В проекте используются четыре основных формата данных: **конфигурация**,
**датасет рецептов**, **файлы сессий диалога** и **артефакты векторного индекса**.
Ниже — их структура и назначение.

## 1. Конфигурация (`config.json` / `config.one.json` / `config.example.json`)

Загружается функцией `agent.config.load_config()`. `config.example.json` —
шаблон в Git; `config.json` / `config.one.json` — локальные файлы с ключами API
(в Git не коммитятся).

| Секция | Назначение |
| --- | --- |
| `base_url`, `api_key`, `model`, `temperature`, `timeout`, `max_retries` | Доступ к облачному OpenAI-совместимому API |
| `max_context_tokens` | Лимит контекста агента |
| `input_cost_per_million`, `output_cost_per_million`, `models` | Тарифы и список моделей для сравнения |
| `compression` | Параметры сжатия истории (`keep_recent`, `summarize_every`, `max_tokens`) |
| `context` | Стратегия управления контекстом (sliding window / sticky facts / branching) |
| `memory` | Трёхуровневая память (short-term / working / long-term) |
| `personalization` | Профиль пользователя |
| `task` | Task State FSM (`enabled`, `auto_advance`) |
| `invariants` | Инварианты и ограничения задачи |
| `mcp` | Параметры MCP-клиента (URL, timeout) |
| `tools` | MCP-сервер GitFlic (command, args, token, timeout) |
| `pipeline` | MCP-сервер pipeline и путь вывода |
| `orchestrator` | Список MCP-серверов для мульти-серверного флоу |
| `indexing` | Источник данных, каталог индекса, эмбеддер, размер чанка, top-k |
| `rag` | Датасет/стратегия, top-k, лимит контекста, каталог отчётов |
| `reranking` | Параметры двухстадийного RAG (retrieve_k, final_k, min_score, method, rewrite) |
| `grounding` | Порог релевантности, top-k, каталог отчётов |
| `local_llm` | Путь к `llama.cpp`, модель, порт, пресеты и варианты квантизации |
| `web_chat` | Параметры веб-чата (порт, каталог истории) |
| `local_rag` | Полностью локальный RAG (порог, подсказки по категориям, история) |
| `optimization` | Параметры оптимизации генерации (День 29) |
| `dev_assistant` | Ассистент разработчика (День 31): индекс доков, MCP-сервер проекта |
| `code_review` | Автоматическое ревью PR (День 32): индексы доков/кода, MCP-сервер ревью, публикация |

Пример секции ассистента разработчика:

```json
"dev_assistant": {
  "enabled": true,
  "index_dir": "history/index",
  "dataset": "project_docs",
  "strategy": "markdown",
  "top_k": 5,
  "max_context_chars": 6000,
  "temperature": 0.2,
  "min_score": 0.0,
  "report_dir": "history/dev_assistant",
  "project_server": {
    "command": "python3",
    "args": ["mcp/project_server.py"],
    "timeout": 30
  }
}
```

Пример секции автоматического ревью (День 32):

```json
"code_review": {
  "enabled": true,
  "index_dir": "history/index",
  "docs_dataset": "project_docs",
  "code_dataset": "project_code",
  "strategy": "markdown",
  "code_strategy": "symbols",
  "top_k": 5,
  "max_context_chars": 8000,
  "max_diff_chars": 20000,
  "fail_on": "none",
  "post_comment": true,
  "report_dir": "history/code_review",
  "review_server": {
    "command": "python3",
    "args": ["mcp/review_server.py"],
    "timeout": 30
  }
}
```

## 2. Датасет рецептов (`recipt_sample.json` / `recipt_all.json`)

Поддерживаются два формата: обёртка `{"total_recipes": N, "recipes": [...]}`
или плоский список `[...]`. Поля одного рецепта (см. `agent/indexing.py`):

| Поле | Тип | Описание |
| --- | --- | --- |
| `id` | int/str | Идентификатор рецепта |
| `title` | str | Название блюда |
| `category` | str | Категория (Птица, Говядина, …) |
| `meal_type` | str | Приём пищи |
| `difficulty` | str | Сложность |
| `preparation_time`, `cooking_time` | str | Время подготовки и приготовления |
| `servings` | int/str | Количество порций |
| `author`, `url` | str | Автор и ссылка на источник |
| `ingredients` | list[str] | Список ингредиентов |
| `intro_text` | str | Описание блюда |
| `steps` | list[dict] | Шаги: `{number, text, temperatures: [{type, value}]}` |
| `equipment` | list[dict] | Оборудование: `{name, ...}` |

## 3. Файлы сессий диалога (`history/<session_id>.json`)

Создаются и читаются классом `agent.conversation.Conversation`.

```json
{
  "summary": "краткое содержание старой части диалога",
  "summarized": 10,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "meta": {"mode": "rag", "sources": []}}
  ],
  "facts": {"ключ": "значение"},
  "working": {"ключ": "значение"},
  "task": {"stage": "...", "steps": []},
  "checkpoint": [],
  "branches": {"branch-id": {"messages": []}},
  "active_branch": null
}
```

`meta` сообщения (например, режим ответа и найденные источники) сохраняется в
файл, но отбрасывается перед отправкой в модель.

## 4. Артефакты векторного индекса FAISS

Каталог индекса имеет вид `<index_dir>/<dataset>/<strategy>/` и содержит:

| Файл | Содержимое |
| --- | --- |
| `faiss.index` | Бинарный индекс `faiss.IndexFlatIP` |
| `chunks.json` | Список чанков: `{id, text, metadata}` |
| `index_meta.json` | Метаданные: `dim`, `embedder`, `total_chunks`, `created_at`, `saved_at` |

Метаданные чанка рецепта включают `recipe_id`, `title`, `section`, `step_number`,
`url`, `category`, `n_tokens`, `source`. Для индекса документации
(`project_docs`) метаданные содержат `title` (имя документа), `section`
(заголовок раздела), `recipe_id` (идентификатор документа) и `url`
(относительный путь к файлу).

Для индекса кода (`code_index`, датасет `project_code`, стратегия `symbols`)
метаданные содержат `module` (путь к файлу), `symbol` (имя класса/функции),
`kind` (`module` / `constants` / `class` / `function`), `lineno` (номер строки),
`section` (имя символа) и `url` (путь к файлу). Чанк начинается служебной строкой
`Файл: … — символ: …`, чтобы заголовки участвовали в векторном поиске.

## 5. Результат вызова MCP-инструмента

`agent.mcp_tools.ToolResult`:

```json
{
  "name": "get_current_branch",
  "arguments": {},
  "content": "{\"branch\": \"day31\", ...}",
  "structured": {"branch": "day31", "detached": false, "commit": "fef93d3"},
  "is_error": false,
  "server": "project"
}
```

`structured` заполняется из `structuredContent` ответа MCP либо парсится из
текстового `content`, если это JSON.

## 6. Артефакты автоматического ревью (`history/code_review/`)

Создаются CLI ревью (`agent/review_cli.py`) и сценарием
(`agent/review_scenarios.py`):

| Файл | Содержимое |
| --- | --- |
| `review_report.md` | Человекочитаемый текст ревью (для комментария к PR) |
| `review_results.json` | Машиночитаемый результат (`CodeReview.to_dict()`) |
| `scenario_report.md` | Сводка офлайн-проверки Дня 32 |

Структура `CodeReview`:

```json
{
  "base": "HEAD~1",
  "head": "HEAD",
  "summary": "краткое резюме ревью",
  "findings": [
    {
      "category": "bug",
      "severity": "critical",
      "title": "Захардкоженный секрет",
      "detail": "…",
      "file": "agent/example.py",
      "line": 5,
      "sources": ["[Код 1]", "[Источник 1]"]
    }
  ],
  "counts": {"bug": 1, "architecture": 0, "recommendation": 2},
  "changed_files": [{"path": "agent/example.py", "status": "M", "additions": 15, "deletions": 0}],
  "diff_stats": {"count": 1, "addTotal": 15, "delTotal": 0},
  "sources": [],
  "docs_used": 5,
  "code_used": 5,
  "pr": {},
  "latency_ms": 42.0,
  "error": null
}
```

`category` — одно из `bug` / `architecture` / `recommendation`; `severity` — одно
из `critical` / `major` / `minor` / `info`.
