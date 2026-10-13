# Final Project (Day 30): Recipe Chat Service — план (дорожная карта)

## 1. Цель и рамки

Полноценный многопользовательский веб-сервис для поиска рецептов: пользователи
регистрируются, ведут несколько чатов, а ответы строятся на локальном RAG по базе
рецептов (FAISS + локальная LLM). Код **самодостаточен**: нужные модули прошлых
дней копируются внутрь `recipe_chat_service/core/`, зависимостей на пакет `agent/` нет.

## 2. Зафиксированные решения

| Решение | Выбор |
| :--- | :--- |
| Каталог | `recipe_chat_service/` |
| Фреймворк | Flask (уже установлен) |
| Хранилище | SQLite через stdlib `sqlite3` |
| Аутентификация | Flask server-side session + `werkzeug.security` хеши |
| LLM-бэкенд | только локальный `llama-server` (llama.cpp) + офлайн-стаб для тестов |
| Функциональность | мультипользователь + несколько чатов на юзера + RAG + история + UI |
| Запуск | `python3 run.py` + офлайн-сценарий проверки |
| База рецептов | **копируем существующий индекс** `recipt_all/structural` + механика пересоздания |

## 3. Архитектура

```
Браузер ──HTTP──> Flask (app/) ──> ChatService (app/chat_service.py)
                    │                   │
                    │                   ├─> SQLite (data/service.db): users/chats/messages
                    │                   └─> RagEngine (app/engine.py)
                    │                          └─> LocalRagSystem (core/local_rag.py)
                    │                                 ├─> RagRetriever → FAISS index (index/)
                    │                                 └─> LLM client → llama-server (core/local_llm.py)
                    └─ login/register/session (app/security.py)
```

Один общий `llama-server` на процесс; генерация сериализуется через `threading.Semaphore`,
чтобы не перегружать слоты сервера. RAG-движок общий, история передаётся на каждый запрос.

## 4. Структура каталога

```
recipe_chat_service/
├── plan.md                    # этот файл
├── run.py                     # entrypoint: поднять llama-server/stub + Flask
├── rebuild_index.py           # CLI пересоздания индекса (см. §10)
├── requirements.txt           # flask, faiss-cpu, numpy
├── README.md
├── config.example.json        # шаблон конфигурации сервиса
├── .gitignore                 # data/, index/**/faiss.index, index/**/chunks.json, __pycache__
├── core/                      # САМОДОСТАТОЧНАЯ копия движка (без зависимостей на agent/)
│   ├── __init__.py
│   ├── llm_client.py          # копия: OpenAI-совместимый HTTP-клиент
│   ├── local_llm.py           # копия: LocalLLMConfig/Server, StubLocalLLM
│   ├── indexing.py            # копия: HashingEmbedder, FaissIndex, чанкинг, load_recipes
│   ├── rag.py                 # копия: RagRetriever/RagSource/RagAnswer, промпты, load_retriever
│   ├── refusal.py             # извлечено: _normalize_text + verify_refusal_and_clarification
│   ├── local_rag.py           # копия, импорт grounding → refusal
│   └── index_build.py         # сборка/пересоздание индекса (обёртка над indexing)
├── app/
│   ├── __init__.py            # create_app()
│   ├── config.py              # ServiceConfig (чтение config.json)
│   ├── db.py                  # схема + CRUD (users/chats/messages)
│   ├── security.py            # хеши, login_required, helpers сессии
│   ├── engine.py              # RagEngine: ретривер + LocalRagSystem + семафор
│   ├── chat_service.py        # оркестрация чата + персистентность
│   ├── routes/{__init__,auth,pages,api}.py
│   ├── templates/{base,login,register,chat}.html
│   └── static/{app.js,style.css}
├── index/                     # база рецептов
│   ├── dataset/recipt_all.json      # датасет (структура как раньше, см. §10)
│   └── recipt_all/structural/{faiss.index,chunks.json,index_meta.json}
├── data/                      # runtime: service.db, secret.key (gitignore)
└── scenarios/
    ├── __init__.py
    └── service_scenarios.py   # офлайн end-to-end проверки
```

## 5. Копируемый код (источник → назначение)

| Источник (day29) | Назначение | Правки |
| :--- | :--- | :--- |
| `agent/llm_client.py` | `core/llm_client.py` | без изменений |
| `agent/local_llm.py` | `core/local_llm.py` | убрать зависимость на `REPO_ROOT`/репо-пути |
| `agent/indexing.py` | `core/indexing.py` | без изменений |
| `agent/rag.py` | `core/rag.py` | без изменений |
| `agent/grounding.py` | `core/refusal.py` | извлечь только `_normalize_text` + `verify_refusal_and_clarification` (избегаем тянуть `reranking.py`, 113 КБ) |
| `agent/local_rag.py` | `core/local_rag.py` | `from .grounding import …` → `from .refusal import …` |
| `agent/optimization.py` | — | **не копируем** (панель Дня 29 вне рамок) |

Не копируем `agent.py`, `conversation.py`, `context.py`, `memory.py`, `tokens.py` —
сервис использует `LocalRagSystem` напрямую и хранит историю в SQLite.

## 6. Модель данных (SQLite)

```sql
users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
chats(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
messages(id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
         role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created_at TEXT NOT NULL);
-- индексы: chats(user_id), messages(chat_id)
```
`meta` — JSON с компактными источниками и режимом (`rag`/`plain`), как в `_compact_sources`.

## 7. HTTP API и страницы

Страницы: `GET /login`, `GET /register`, `GET /` (→ `/chat` или `/login`), `GET /chat`.
API (все, кроме auth, требуют логина; ресурсы проверяются на владельца):
`POST /api/register`, `POST /api/login`, `POST /api/logout`, `GET /api/me`,
`GET/POST /api/chats`, `PATCH/DELETE /api/chats/<id>`,
`GET /api/chats/<id>/messages`, `POST /api/chats/<id>/messages` (`{message, use_rag}`),
`POST /api/chats/<id>/reset`, `GET /api/status`.

## 8. Аутентификация и безопасность

- Пароли: `generate_password_hash` / `check_password_hash`; мин. длина 6.
- Сессия: подписанная cookie Flask, `SECRET_KEY` из `data/secret.key` (создаётся при первом запуске).
- Cookie: `HttpOnly`, `SameSite=Lax`; `login_required` для страниц и API.
- Изоляция: каждый запрос чата проверяет `chats.user_id == session.user_id`.
- Проверка входных данных и понятные JSON-ошибки (400/401/403/404/502).

## 9. RAG-движок и LLM-рантайм

- `RagEngine` при старте грузит FAISS-индекс (`core.rag.load_retriever`) и держит один
  `LocalRagSystem`; `threading.Semaphore(1)` вокруг генерации.
- Режимы сообщения: `use_rag=True` (рецепты) / `False` (обычный ответ через `plain_query`).
- Offline (`--offline`): `StubLocalLLM` → детерминированный клиент, индекс обязателен.
- История: последние N сообщений из SQLite передаются в `ask_with_rag(history=…)`.
- Конфиг `config.json`: `local_llm` (model_path/port/ctx), `index` (dir/dataset/strategy/source),
  `local_rag` (top_k, thresholds, category_suggestions), `web` (host/port), `database`.

## 10. База рецептов: копирование индекса и механика пересоздания

### 10.1 Копирование существующего индекса

При инициализации проекта копируем готовый индекс из репозитория:

```
history/index/recipt_all/structural/{faiss.index,chunks.json,index_meta.json}
        ──копия──>  recipe_chat_service/index/recipt_all/structural/
```

Копирование выполняет `rebuild_index.py --copy` (или setup-шаг в README). Так сервис
получает ту же базу (6403 чанка, `dim=256`, `hashing_256d`), что и Дни 22–29.

### 10.2 Формат данных остаётся прежним

**Ничего в структуре JSON не меняем.**

- Датасет (`index/dataset/recipt_all.json`) — тот же формат, что `recipt_sample.json`:
  `{"total_recipes": N, "recipes": [{"url", "title", "author", "publication_date",
  "preparation_time", "cooking_time", "difficulty", "servings", "category", "meal_type",
  "ingredients": [...], "equipment": [...], ...}, ...]}`. Читается `core.indexing.load_recipes`.
- Артефакты индекса — те же три файла в `index/<dataset>/<strategy>/`:
  - `faiss.index` — бинарный FAISS;
  - `chunks.json` — список чанков (текст + метаданные);
  - `index_meta.json` — `{strategy, source, created_at, dim, embedder, total_chunks,
    saved_at, index_file, chunks_file}`.
  Запись/чтение — существующими `FaissIndex.save()` / `FaissIndex.load()`.

### 10.3 Пересоздание индекса

CLI `rebuild_index.py`:

```bash
# скопировать готовый индекс из репозитория (по умолчанию)
python3 rebuild_index.py --copy

# пересоздать из датасета (та же структура JSON), стратегия structural
python3 rebuild_index.py --source index/dataset/recipt_all.json \
    --dataset recipt_all --strategy structural

# полный пересбор всех стратегий / другой датасет
python3 rebuild_index.py --source index/dataset/recipt_all.json --strategy all
```

Механика (через `core/index_build.py` → `core/indexing.py`):

1. `load_recipes(source)` → рецепты + сводка (формат не меняется).
2. `chunk_structural(recipes, source=dataset)` (или `chunk_fixed_size` для `--strategy fixed`).
3. `HashingEmbedder(dim=256)` → `FaissIndex.build(chunks, embedder)`.
4. `index.save(index/<dataset>/<strategy>/)` → пишет те же `faiss.index` / `chunks.json` /
   `index_meta.json`.
5. Атомарная замена: сборка во временный каталог + `os.replace`, чтобы работающий сервис
   не увидел полу-записанный индекс.

Дополнительно:

- **Обновление при старте**: `run.py --rebuild-index` пересоздаёт индекс перед запуском;
  `run.py --rebuild-if-missing` (по умолчанию) — только если индекса нет.
- **Проверка свежести** (опционально): сравнение `mtime`/размера датасета и `index_meta.json`;
  если датасет новее — предупреждение в логе.
- **Админ-эндпоинт** (опционально, под `login_required`+флаг конфига `web.allow_reindex`):
  `POST /api/admin/reindex` → запускает пересборку в фоне и перезагружает `RagEngine`.
  По умолчанию выключен.

## 11. Веб-интерфейс

`chat.html` + `app.js`: вход/регистрация, боковая панель чатов (создать/переименовать/удалить),
окно сообщений с бейджами режима и списком источников, переключатель «рецепты / без RAG»,
восстановление истории из API, индикатор статуса (`/api/status`). Тёмная тема как в Дне 27.

## 12. Этапы реализации

1. **Скелет проекта**: каталог, `plan.md`, `requirements.txt`, `config.example.json`, `.gitignore`, README.
2. **Копирование `core/`** и правка импортов; проверка импортируемости.
3. **Копирование индекса** `--copy` + `core/index_build.py` + `rebuild_index.py` (пересоздание, §10).
4. **`app/db.py`**: схема, инициализация, CRUD users/chats/messages.
5. **`app/security.py`** + `routes/auth.py`: регистрация/логин/логаут, `login_required`.
6. **`app/engine.py`**: загрузка индекса, `LocalRagSystem`, семафор, offline-стаб.
7. **`app/chat_service.py`**: оркестрация, персистентность, авто-заголовок чата.
8. **`routes/pages.py` + `routes/api.py`**: страницы и JSON API.
9. **UI**: шаблоны + `app.js` + `style.css`.
10. **`run.py`**: аргументы (`--offline`, `--host`, `--port`, `--model`, `--index-dir`,
    `--rebuild-index`, `--rebuild-if-missing`), запуск LLM-рантайма и Flask.
11. **`scenarios/service_scenarios.py`**: офлайн end-to-end (Flask test client):
    регистрация → логин → создание чата → RAG-вопрос → источники → изоляция второго юзера → 401 без логина.
12. **README** и финальная проверка (`ruff`, офлайн-сценарий, пересборка индекса).

## 13. Проверка (zero API cost)

```bash
cd recipe_chat_service
python3 rebuild_index.py --copy                    # индекс на месте
python3 rebuild_index.py --source index/dataset/recipt_all.json --strategy structural
python3 -m scenarios.service_scenarios --offline   # детерминированный e2e
ruff check core/ app/ scenarios/
python3 run.py --offline                           # ручная проверка UI на стабе
```

## 14. Риски и открытые вопросы

- **Исходный датасет `recipt_all.json`** отсутствует в репозитории: копируем готовый индекс,
  а для пересоздания из датасета кладём `index/dataset/recipt_all.json` (структура та же).
  Если датасета нет — пересоздание работает на `recipt_sample.json` (19 рецептов).
- **`core/refusal.py`**: извлечение функций вместо копии `grounding.py` — небольшая
  расходимость с оригиналом (функционально эквивалентно).
- **Конкурентность**: один llama-server; при многих пользователях запросы встают в очередь.

## 15. Definition of Done

- `python3 -m scenarios.service_scenarios --offline` проходит без сети/модели.
- `python3 rebuild_index.py --copy` кладёт рабочий индекс; `--source …` пересоздаёт его,
  сохраняя прежнюю структуру JSON (dataset / chunks.json / index_meta.json).
- Регистрация/логин/логаут, несколько чатов на пользователя, история в SQLite.
- RAG-ответы с источниками; данные строго изолированы по пользователям.
- `ruff check` без ошибок; проект не импортирует `agent/`/`mcp/`.
