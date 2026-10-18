# Разработка

## Ветки и структура

- **Ветка на урок.** Каждая git-ветка (`day2` … `day32`) — состояние кода на
  соответствующем дне курса. Код накопительный: `day32` содержит всё предыдущее.
- **Активную ветку** всегда проверяйте перед изменениями:
  ```bash
  git branch --show-current
  ```
- Итоговая ветка — `master`.

## Настройка окружения

```bash
cp config.example.json config.json      # или config.one.json
# укажите api_key и model в config.json
```

`config.json` и `config.one.json` игнорируются Git — ключи API не коммитятся.
Для локальной LLM (Дни 26–29) соберите `llama.cpp` и скачайте модель:

```bash
bash scripts/setup_llama_cpp.sh                 # сборка + загрузка Q4_K_M
bash scripts/setup_llama_cpp.sh --quant all     # обе квантизации (Q4_K_M + Q2_K)
```

## Офлайн-проверки (без затрат на API)

Почти каждая подсистема имеет детерминированный стаб. Запускайте проверки
офлайн, прежде чем делать живые вызовы LLM:

```bash
# Ассистент файловых операций (День 34) — MCP-файлы, стаб, без модели и сети
python3 mcp/fs_server.py --check --files --search "build_support_runtime" --path agent
python3 -m agent.file_scenarios --offline

# Ассистент поддержки (День 33) — FAQ RAG + MCP-тикеты, без модели и сети
python3 mcp/support_server.py --check --ticket T-1024 --user 1 --search "авторизация"
python3 -m agent.support_scenarios --offline

# Автоматическое ревью PR (День 32) — стаб/эвристики, без модели и сети
python3 -m agent.project_docs --build
python3 -m agent.code_index --build
python3 mcp/review_server.py --check --diff --files
python3 -m agent.review_scenarios --offline
python3 review_pr.py --base HEAD~1 --head HEAD --offline

# Ассистент разработчика (День 31) — стаб, без модели и сети
python3 -m agent.project_docs --build
python3 -m agent.dev_assistant_scenarios --offline

# Оптимизация локальной LLM (День 29)
python3 -m agent.optimization_scenarios --offline

# Сервис рецептов (День 30)
cd recipe_chat_service && python3 rebuild_index.py --copy && python3 -m scenarios.service_scenarios --offline

# Локальная LLM (Дни 26–28)
python3 -m agent.local_llm_scenarios --offline
python3 -m agent.web_chat_scenarios --offline
python3 -m agent.local_rag_scenarios --offline

# RAG (Дни 22–25)
python3 -m agent.rag_scenarios --no-llm
python3 -m agent.reranking_scenarios --no-llm
python3 -m agent.grounding_scenarios --no-llm
python3 -m agent.rag_chat_scenarios --no-llm

# Индексация и MCP (Дни 19–21)
python3 -m agent.indexing_scenarios
python3 -m agent.orchestration_scenarios --offline
```

## Линтинг

```bash
ruff check agent/ mcp/
```

## Правила экономии токенов (для агентов)

- **Не сканировать `history/`**: там лежат сгенерированные артефакты (индексы
  FAISS, JSON-дампы, логи сессий), которые быстро переполняют контекст.
- **Не запускать глобальные поиски** (`find .`, `ls -R`, неограниченный `grep`):
  ищите точечно внутри `agent/` или `mcp/`.
- **Не читать датасеты целиком** (`recipt_sample.json`, `recipt_all.json`).
- **Использовать офлайн-режимы** (`--no-llm` / `--offline`) при проверках.

## Генерация документации

Справочник по модулям собирается из исходников автоматически:

```bash
python3 scripts/generate_docs.py            # записать docs/modules.md
python3 scripts/generate_docs.py --check    # проверить актуальность
```

После изменения кода пересоберите справочник и обновите индекс документации:

```bash
python3 scripts/generate_docs.py
python3 -m agent.project_docs --build
```

## MCP-серверы вручную

```bash
python3 mcp/project_server.py --check
python3 mcp/review_server.py --check --diff --files
python3 mcp/support_server.py --check --ticket T-1024 --user 1
python3 mcp/fs_server.py --check --files --search "build_support_runtime" --path agent
python3 mcp/gitflic_server.py --check
python3 mcp/report_server.py --check --list
python3 -m agent.mcp_client --check
```

## Пайплайн ревью PR

Ядро ревью (Дни 32) тестируется локально теми же командами, что и в CI, а сам
пайплайн — GitHub Action `.github/workflows/code-review.yml` — запускается на
`pull_request`. Локальный прогон:

```bash
python3 -m agent.project_docs --build
python3 -m agent.code_index --build
python3 review_pr.py --base HEAD~1 --head HEAD --offline   # без модели и сети
python3 review_pr.py --base day31 --head day32             # живой LLM, если есть ключ
```

Режим выбирается автоматически: без ключа (`OPENAI_API_KEY`/`LLM_API_KEY`) ревью
идёт офлайн-эвристиками, с ключом — моделью. Публикация комментария к PR требует
`GITHUB_TOKEN` (в Action он выдаётся автоматически).
