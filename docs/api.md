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
| `python3 review_pr.py` | `agent.review_cli` | **Автоматическое ревью PR** (День 32) |
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

### `mcp/review_server.py` — вход для ревью PR (День 32)
Инструменты доступа к локальному репозиторию для анализа pull request (офлайн,
через `git`; каталог задаётся `REVIEW_REPO_ROOT`):

| Инструмент | Возвращает |
| --- | --- |
| `get_pr_diff` | Diff `base...head` (или рабочего дерева): `{base, head, chars, truncated, diff}` |
| `list_changed_files` | Изменённые файлы: `{path, status, additions, deletions}` + счётчики |
| `get_file_content` | Содержимое файла на ревизии: `{path, ref, chars, content}` |
| `get_pr_metadata` | Метаданные PR из `GITHUB_EVENT_PATH` (номер, заголовок, автор, base/head) |

Проверка из терминала без MCP-транспорта:

```bash
python3 mcp/review_server.py --check --diff --files
python3 mcp/review_server.py --check --content agent/rag.py
python3 mcp/review_server.py --check --metadata
```

### `mcp/support_server.py` — CRM поддержки (День 33)
Локальные инструменты по JSON-CRM (`mcp/support_data/`, каталог задаётся
`SUPPORT_DATA_DIR`):

| Инструмент | Возвращает |
| --- | --- |
| `get_user` | Профиль пользователя (план, версия, статус) |
| `find_user` | Поиск пользователя по логину/e-mail/имени |
| `get_ticket` | Полный тикет с перепиской |
| `list_tickets` | Тикеты пользователя (фильтр по статусу) |
| `search_tickets` | Поиск тикетов по тексту (`scope=all|meta`) |
| `get_ticket_context` | Готовый контекст «пользователь + тикет + переписка» |

```bash
python3 mcp/support_server.py --check --ticket T-1024 --user 1 --search "авторизация"
```

### `mcp/fs_server.py` — файловые операции (День 34)
Локальные инструменты чтения и записи файлов проекта (jail по путям, корень
задаётся `FS_ROOT`):

| Инструмент | Возвращает |
| --- | --- |
| `list_files` | Список файлов (`path` — префикс, `pattern` — glob, `limit`) |
| `read_file` | Содержимое файла или диапазона строк (`start_line`/`end_line`) |
| `search_files` | Поиск по многим файлам: `{file, count, matches}` |
| `analyze_file` | Структура файла (для `.py` — `ast`: импорты/классы/функции) |
| `file_info` | Размер, mtime, число строк |
| `write_file` | Создание/перезапись файла (`overwrite`, `create_dirs`) |
| `edit_file` | Точечная замена подстроки (`old` → `new`, `count`) |
| `make_diff` | Diff относительно git или baseline-каталога (`baseline_root`) |

```bash
python3 mcp/fs_server.py --check --files --search "build_support_runtime" --path agent
python3 mcp/fs_server.py --check --analyze agent/rag.py
```

### Формат вызова MCP из агента

`agent/mcp_tools.McpToolRuntime` поднимает сервер по stdio, вызывает инструмент
(`runtime.call(name, arguments)`) и возвращает `ToolResult` с полями `content`
(текст) и `structured` (распарсенный JSON). Мульти-серверный режим —
`agent/orchestrator.McpOrchestrator` с маршрутизацией по имени инструмента.

## 5. Пайплайн ревью PR (GitHub Action)

`.github/workflows/code-review.yml` запускается на событиях `pull_request`
(opened/synchronize/reopened): ставит зависимости, собирает индексы RAG
(документация + код) и вызывает `review_pr.py`.

```bash
# локально, без модели и сети (офлайн-эвристики)
python3 review_pr.py --base HEAD~1 --head HEAD --offline

# как в CI: метаданные из события + публикация комментария к PR
python3 review_pr.py --event "$GITHUB_EVENT_PATH" \
  --base "$BASE_SHA" --head "$HEAD_SHA" --post-comment \
  --output review.md --json review.json
```

| Флаг | Назначение |
| --- | --- |
| `--base`, `--head` | Диапазон сравнения (SHA/ветка); без них — из события или `HEAD` |
| `--event` | Путь к payload GitHub-события (`$GITHUB_EVENT_PATH`) |
| `--offline` / `--no-llm` | Принудительно офлайн-эвристики без обращения к модели |
| `--output`, `--json` | Куда записать markdown-ревью и JSON-результат |
| `--post-comment` | Опубликовать комментарий к PR через GitHub REST API |
| `--fail-on {none,critical,major}` | Вернуть код 1 при находках заданной серьёзности (по умолчанию `none`) |

Переменные окружения: `GITHUB_TOKEN` (публикация комментария),
`GITHUB_REPOSITORY`, `GITHUB_EVENT_PATH`; для живой модели —
`OPENAI_API_KEY`/`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`. Если ключ не задан,
пайплайн автоматически работает в офлайн-режиме (эвристики по добавленным
строкам diff).
