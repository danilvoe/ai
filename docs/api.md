# API проекта

Проект предоставляет три вида интерфейсов: **CLI-точки входа**, **HTTP API**
(веб-чат и сервис рецептов) и **MCP-серверы** с инструментами.

## 1. CLI-точки входа

| Команда | Модуль | Назначение |
| --- | --- | --- |
| `python3 chat.py` | `agent.cli` | Базовый чат-агент |
| `python3 chat_rag.py` | `agent.rag_chat_cli` | RAG-чат с памятью задачи и источниками |
| `python3 chat_local.py` | `agent.local_llm` | Чат с локальной LLM через `llama-cli` |
| `python3 chat_dev.py` | `agent.dev_assistant_cli` | **Ассистент разработчика** (День 31), команда `/help` |
| `python3 web_local.py` | `agent.web_chat` | Веб-чат (локальная/облачная модель, RAG-режим) |
| `python3 -m agent.project_docs --build` | `agent.project_docs` | Сборка индекса документации |
| `cd recipe_chat_service && python3 run.py` | `recipe_chat_service` | Сервис рецептов Дня 30 |

### Команды ассистента разработчика (`chat_dev.py`)

| Команда | Действие |
| --- | --- |
| `/help [вопрос]` | Ответить на вопрос о проекте по документации и контексту |
| `/branch` | Показать текущую git-ветку (MCP `get_current_branch`) |
| `/files [префикс]` | Список файлов проекта под контролем версий |
| `/diff [путь]` | Diff рабочего дерева |
| `/sources` | Источники последнего ответа (файл + раздел) |
| `/reindex` | Пересобрать индекс документации |
| `/exit` | Выйти |

## 2. HTTP API веб-чата (`agent/web_chat.py`)

| Метод и путь | Назначение |
| --- | --- |
| `GET /` | HTML-страница чата |
| `GET /api/status` | Статус: доступность RAG, источники моделей, текущая модель |
| `POST /api/chat` | Отправить сообщение. Тело: `message`, `session_id`, `use_rag`, `model_source` (`local`/`cloud`), `temperature`, `max_tokens`, `context_window`, `preset`, `prompt_template` |
| `POST /api/settings` | Серверные настройки Дня 29: `ctx_size`, `variant` (перезапуск `llama-server`) |
| `POST /api/reset` | Сбросить сессию: `session_id` |
| `GET /api/history` | История сообщений: `?session_id=...` |

## 3. HTTP API сервиса рецептов (`recipe_chat_service/`)

Аутентификация (`app/routes/auth.py`):

| Метод и путь | Назначение |
| --- | --- |
| `POST /api/register` | Регистрация: `username`, `password` |
| `POST /api/login` | Вход: `username`, `password` |
| `POST /api/logout` | Выход |
| `GET /api/me` | Текущий пользователь |

Чаты и сообщения (`app/routes/api.py`, требуют логина):

| Метод и путь | Назначение |
| --- | --- |
| `GET /api/chats` | Список чатов пользователя |
| `POST /api/chats` | Создать чат |
| `PATCH /api/chats/<id>` | Переименовать чат (`title`) |
| `DELETE /api/chats/<id>` | Удалить чат |
| `GET /api/chats/<id>/messages` | История сообщений чата |
| `POST /api/chats/<id>/messages` | Отправить сообщение (`message`, `use_rag`) |
| `POST /api/chats/<id>/reset` | Очистить чат |
| `GET /api/status` | Статус сервиса |
| `POST /api/admin/reindex` | Пересборка индекса (если разрешено конфигом) |

## 4. MCP-серверы и инструменты

### `mcp/gitflic_server.py` — публичные проекты GitFlic
`get_public_projects`, `count_public_projects`, `schedule_project_count`,
`list_count_jobs`, `cancel_count_job`, `get_project_count_report`.

### `mcp/pipeline_server.py` — композиция инструментов
`search` (GitFlic), `summarize` (локальная выжимка), `saveToFile` (запись в файл
внутри проекта), `run_pipeline` (цепочка search → summarize → saveToFile).

### `mcp/report_server.py` — журнал отчётов
`list_reports`, `record_run`, `build_report_index`.

### `mcp/project_server.py` — контекст проекта (День 31)
Инструменты доступа к самому репозиторию (все работают офлайн, через `git`):

| Инструмент | Возвращает |
| --- | --- |
| `get_current_branch` | `{branch, detached, commit, is_repo}` |
| `get_repo_status` | Ветку, upstream, ahead/behind, изменённые/новые файлы |
| `list_project_files` | Список файлов под контролем версий (`path` — префикс, `limit`) |
| `get_recent_commits` | Последние коммиты: `{hash, subject, author, date}` |
| `get_diff` | Diff (`base`, `staged`, `path`, `max_chars`) |

Проверка сервера из терминала без MCP-транспорта:

```bash
python3 mcp/project_server.py --check
python3 mcp/project_server.py --check --files
python3 mcp/project_server.py --check --commits
```

### Формат вызова MCP из агента

`agent/mcp_tools.McpToolRuntime` поднимает сервер по stdio, вызывает инструмент
(`runtime.call(name, arguments)`) и возвращает `ToolResult` с полями `content`
(текст) и `structured` (распарсенный JSON). Мульти-серверный режим —
`agent/orchestrator.McpOrchestrator` с маршрутизацией по имени инструмента.
