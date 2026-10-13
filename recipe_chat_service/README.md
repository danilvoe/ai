# Recipe Chat Service (Day 30)

Полноценный многопользовательский веб-сервис для поиска рецептов: регистрация,
несколько чатов на пользователя, ответы на основе **локального RAG** (FAISS +
локальная LLM через `llama-server`), история диалога в SQLite.

Код **самодостаточен**: нужные модули прошлых дней скопированы в `core/`,
зависимостей на пакет `agent/` нет.

## Архитектура

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

Один общий `llama-server` на процесс; генерация сериализуется через
`threading.Semaphore(1)`, чтобы не перегружать слоты сервера.

## Установка

```bash
pip install -r requirements.txt
cp config.example.json config.json   # при необходимости поправьте пути
```

Зависимости: `flask`, `faiss-cpu`, `numpy`.

## Быстрый старт (офлайн, без модели и сети)

```bash
python3 rebuild_index.py --copy     # скопировать готовый индекс из репозитория
python3 run.py --offline            # UI на детерминированном стабе
# http://127.0.0.1:5000
```

## Запуск с локальной моделью

```bash
# соберите llama.cpp и скачайте GGUF (см. scripts/setup_llama_cpp.sh в корне репозитория)
python3 run.py
```

Опции `run.py`: `--offline`, `--host`, `--port`, `--model`, `--index-dir`,
`--dataset`, `--strategy`, `--source`, `--config`, `--rebuild-index`,
`--rebuild-if-missing` (по умолчанию; отключается `--no-rebuild-if-missing`).

## База рецептов: индекс

Готовый индекс (`6403` чанка, `dim=256`, `hashing_256d`) лежит в
`index/recipt_all/structural/` и состоит из трёх файлов:
`faiss.index`, `chunks.json`, `index_meta.json`.

```bash
# скопировать готовый индекс из репозитория
python3 rebuild_index.py --copy

# пересобрать из датасета (структура JSON не меняется)
python3 rebuild_index.py --source index/dataset/recipt_sample.json \
    --dataset recipt_all --strategy structural

# пересобрать все стратегии
python3 rebuild_index.py --source index/dataset/recipt_sample.json --strategy all
```

Датасет — тот же формат, что `recipt_sample.json`:
`{"total_recipes": N, "recipes": [...]}`. Запись индекса атомарна
(сборка во временный каталог + `os.replace`).

> **Внимание:** полноразмерного `recipt_all.json` в репозитории нет (он в
> `.gitignore`). Если указанный `--source` отсутствует, пересборка автоматически
> откатывается на `index/dataset/recipt_sample.json` (19 рецептов). Пересборка из
> него перезапишет полный индекс; восстановить его можно командой
> `python3 rebuild_index.py --copy`.

## Обслуживание

- **Конфиг** `config.json`: `index` (dir/dataset/strategy/top_k), `local_rag`
  (пороги/история), `local_llm` (model_path/port/ctx), `web` (host/port/allow_reindex).
- **Индекс**: `rebuild_index.py --copy` — взять готовый; `--source <json> --strategy
  structural|all` — пересобрать. При отсутствии источника `--copy` сам откатится
  на пересборку из `index/dataset/recipt_sample.json`.
- **Данные**: `data/service.db` (пользователи/чаты/сообщения) и `data/secret.key` —
  вне git. Делайте бэкап `service.db`; удаление `secret.key` разлогинит всех.
- **Логи модели**: `data/local_llm/server.log`.
- **Пересборка на ходу**: включите `web.allow_reindex=true` и вызовите
  `POST /api/admin/reindex` (пересборка в фоне + hot-reload движка).
- **Прод**: dev-сервер Flask не использовать — gunicorn/waitress. Один
  `llama-server` на процесс, генерация сериализована `Semaphore(1)` (запросы встают в очередь).

## HTTP API

Страницы: `GET /login`, `GET /register`, `GET /`, `GET /chat`.

API (кроме auth, требуют логина; ресурсы проверяются на владельца):

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/api/register` | регистрация |
| POST | `/api/login` | вход |
| POST | `/api/logout` | выход |
| GET | `/api/me` | текущий пользователь |
| GET/POST | `/api/chats` | список / создание чата |
| PATCH/DELETE | `/api/chats/<id>` | переименование / удаление |
| GET/POST | `/api/chats/<id>/messages` | история / новое сообщение (`{message, use_rag}`) |
| POST | `/api/chats/<id>/reset` | очистить историю |
| GET | `/api/status` | статус движка |

## Проверка (zero API cost)

```bash
cd recipe_chat_service
python3 rebuild_index.py --copy                    # индекс на месте
python3 -m scenarios.service_scenarios --offline   # детерминированный e2e
ruff check core/ app/ scenarios/
python3 run.py --offline                           # ручная проверка UI
```

## Структура

```
recipe_chat_service/
├── run.py                # entrypoint: индекс + Flask
├── rebuild_index.py      # CLI пересборки/копирования индекса
├── core/                 # самодостаточный RAG-движок (копии дней 21–28)
├── app/                  # Flask: config, db, security, engine, chat_service, routes, UI
├── index/                # датасет + FAISS-индекс
├── data/                 # runtime: service.db, secret.key (gitignore)
└── scenarios/            # офлайн end-to-end проверки
```
