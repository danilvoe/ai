# Day 31: Developer Assistant — Project Docs RAG + Project MCP

An assistant that understands the project itself: it answers questions about the
repository using **documentation retrieval (RAG)** over `README` + `docs/` and a
**project MCP server** that exposes live git context (branch, files, commits, diff).

**Flow:**
**`/help <question>`** → **Docs retrieval (FAISS over README + `docs/`)** → **Project context via MCP (`get_current_branch`, files, commits, diff)** → **Grounded answer with sources (`file` + `section`)**.

---

## Architecture & Components

| Component | Module | Responsibility |
| --- | --- | --- |
| **Docs indexer** | `agent/project_docs.py` | Loads `README.md`, `docs/*.md`, `AGENTS.md`; markdown-structural chunking; FAISS index at `history/index/project_docs/markdown/` |
| **Doc generator** | `scripts/generate_docs.py` | Auto-generates `docs/modules.md` (module docstrings + public classes/functions) from `agent/` and `mcp/` source via `ast` |
| **Project MCP server** | `mcp/project_server.py` | `get_current_branch` (required minimum), `get_repo_status`, `list_project_files`, `get_recent_commits`, `get_diff` — local `git`, no network |
| **Assistant** | `agent/dev_assistant.py` | Combines docs RAG + project MCP + LLM; light reranking (curated docs first); deterministic offline stub |
| **CLI** | `agent/dev_assistant_cli.py`, `chat_dev.py` | Commands `/help`, `/branch`, `/files`, `/diff`, `/sources`, `/reindex` |

---

## Documentation (`docs/`)

| Document | Content |
| --- | --- |
| `docs/architecture.md` | Subsystems, module map, data flows |
| `docs/data-schemas.md` | Config sections, recipe JSON, session JSON, FAISS index files |
| `docs/api.md` | HTTP API, MCP servers/tools, CLI entrypoints |
| `docs/development.md` | Branch-per-lesson rules, offline verification, linting |
| `docs/modules.md` | **Auto-generated** module reference (`python3 scripts/generate_docs.py`) |

---

## Run

```bash
# 1. Build the documentation index (offline, deterministic)
python3 -m agent.project_docs --build

# 2. Interactive developer assistant (LLM answers via docs + git context)
python3 chat_dev.py

# 3. Offline mode (no model / no network, deterministic stub)
python3 chat_dev.py --no-llm
python3 chat_dev.py --no-llm --question "где находится MCP-сервер проекта?"

# 4. Project MCP server check from the terminal (no MCP transport)
python3 mcp/project_server.py --check --status --files --commits

# 5. Full offline scenario: docs index + MCP + /help answers
python3 -m agent.dev_assistant_scenarios --offline

# 6. Regenerate the module reference and refresh the docs index
python3 scripts/generate_docs.py
python3 -m agent.project_docs --build
```

### `/help` and commands

```text
/help <вопрос>   — answer a question about the project (docs + git context)
/help            — show command help and example questions
/branch          — current git branch (MCP get_current_branch)
/files [prefix]  — project files under version control
/diff [path]     — working-tree diff
/sources         — sources of the last answer (file + section)
/reindex         — rebuild the documentation index
/exit            — quit
```

Any plain question (without `/`) is also sent to the assistant as a project question.

---

## What is evaluated

| Check | Requirement | Where |
| --- | --- | --- |
| **README + `docs/` in RAG** | Both are indexed into FAISS | `check_index` in `agent/dev_assistant_scenarios.py` |
| **MCP exposes git branch** | `get_current_branch` returns the active branch | `check_mcp` |
| **`/help` answers structure questions** | Non-empty, cited, includes the git branch | `check_help` (5 control questions) |

Reports are stored at `history/dev_assistant/dev_report.md` and `history/dev_assistant/dev_results.json`.

---

# Production-like Mini-Chat CLI with RAG & Task Memory (Day 24–26)

Production-like interactive mini-chat CLI combining **Retrieval-Augmented Generation (RAG)**, **Task State Memory (`RagTaskState`)**, and **Strict Attribution with Chunk IDs and Clickable Recipe URLs**.

**Dialogue Flow:**
**User Input** → **State Extractor & Constraint Guard** (if violation → **Explain & Uphold Restriction**) → **Contextual Query Enrichment** → **RAG Retrieval & Reranking** → **Grounded Prompting with Task State** → **Assistant Reply + Sources (`chunk_id` + `title` + `section` + `url`)** → **Persistent Session Storage (`history/<session_id>.json`)**.

---

## Architecture & Components

| Component | Module | Responsibility |
| --- | --- | --- |
| **`RagTaskState`** | `agent/rag_chat.py` | Explicit task memory structure maintaining `goal`, `clarifications`, `constraints`, and `terms` across lengthy conversations (10–15+ turns) |
| **`ChatSource`** | `agent/rag_chat.py` | Data model for attribution guaranteeing `chunk_id`, `title`, `section`, `recipe_id`, and direct clickable `url` |
| **`RagChatSession`** | `agent/rag_chat.py` | Session wrapper over `Conversation` with disk persistence, task state synchronization, and history logging |
| **`RagChatAgent`** | `agent/rag_chat.py` | Orchestrates query expansion, FAISS retrieval, reranking, constraint validation, and grounded LLM/offline answer generation |
| **Interactive CLI** | `agent/rag_chat_cli.py`, `chat_rag.py` | Full-featured CLI with terminal task banner and slash commands (`/state`, `/goal`, `/constraint`, `/clarify`, `/term`, `/sources`, `/history`, `/clear`, `/exit`) |
| **Scenario Validator** | `agent/rag_chat_scenarios.py` | Test runner executing two 12-turn end-to-end scenarios, validating goal retention, constraint adherence, chunk IDs, and clickable URLs |

---

## 2 End-to-End Scenarios Benchmark (24 Turns Total)

Validated across two complete 12-turn dialogue scenarios:

### Scenario 1: New Year Holiday Dinner — Orange Chicken on Charcoal Kettle (12 turns)
- **Goal:** Plan a holiday dinner around poultry on a charcoal grill.
- **Critical Constraint:** Zero spiciness, no chili, no hot peppers (children present).
- **Equipment Clarification:** Classic 57 cm charcoal Kettle grill.
- **Fixed Term:** *"Пряное сливочное масло"* (softened butter with orange zest and rosemary).
- **Constraint Provocation Check (Turn 7):** User asks to add sriracha sauce → Agent intercepts violation, refuses, reminds of no-chili constraint for children, and keeps poultry context.
- **Cooking Parameters:** Indirect heat (180–200°C), internal temperature 74°C in breast and 80–84°C in thigh, glaze 10–15 min before finish.
- **Full Evening Timeline:** Complete step-by-step evening timing generated without losing goal or constraints.

### Scenario 2: BBQ Party for Friends — Beef Ribs Hot & Fast (12 turns)
- **Goal:** BBQ for a group of 6–8 friends centered on beef ribs.
- **Critical Constraint:** Zero tomatoes, no tomato sauces, no ketchup (guest intolerance).
- **Fixed Term:** *"Hot & Fast"* (smoking/baking beef ribs at 135–150°C).
- **Rub & Spritzing:** 50/50 coarse salt and black pepper rub; spritzing with 50/50 apple cider vinegar and water (no tomato!).
- **Constraint Provocation Check (Turn 8):** Friend suggests Heinz BBQ sauce in foil → Agent intercepts violation, refuses because commercial BBQ sauces contain tomato paste, preserving the constraint.
- **Doneness & Resting:** Probe tenderness ("like warm butter", ~93–96°C) and mandatory 40–60 min resting in cooler/foil.
- **Final Summary Checklist:** Full technical checklist preserving all parameters and constraints.

---

## Validation Summary

| Metric | Target | Achieved | Notes |
| --- | --- | --- | --- |
| **Goal Retention Rate** | 100% | **100.0% (24/24)** | Goal preserved seamlessly across 12 consecutive turns |
| **Constraint Adherence Rate** | 100% | **100.0% (24/24)** | Provocations (sriracha, Heinz BBQ) 100% intercepted |
| **Sources Presence** | 100% | **100.0% (24/24)** | Every turn cites grounding chunks |
| **Chunk ID Presence** | 100% | **100.0% (24/24)** | Every source displays exact chunk ID (e.g. `struct_15454_ingredients`) |
| **Clickable URL Presence** | 100% | **100.0% (24/24)** | Every source displays full recipe URL (e.g. `https://grill-bbq.ru/recipe/...`) |
| **Average Offline Latency** | < 10 ms | **1.7 ms** | Instant local response for CI/CD |

Reports are stored at `history/rag_chat/chat_report.md` and `history/rag_chat/chat_results.json`.

---

## How to Run

```bash
# Launch interactive CLI RAG-Chat (creates or resumes session)
python3 chat_rag.py

# Launch interactive CLI in offline mode (no LLM API required)
python3 chat_rag.py --no-llm

# Launch with a brand-new clean session
python3 chat_rag.py --new

# Run the 2 benchmark scenarios (24 turns total) in fast offline verification mode
python3 -m agent.rag_chat_scenarios --no-llm

# Run benchmark scenarios with live LLM API
python3 -m agent.rag_chat_scenarios --limit 2
```

### In-Chat Slash Commands

```text
/state                     — Show current task card (goal, clarifications, constraints, terms)
/goal <description>        — Manually update current dialogue goal
/constraint <restriction>  — Register a new constraint (e.g., «без томатов»)
/clarify <key: value>      — Register a clarification (e.g., «гриль: Kettle 57 см»)
/term <term: definition>   — Register a culinary term or technique
/sources                   — Show detailed text excerpts of the last retrieved RAG chunks
/history                   — Show dialogue history for the active session
/clear                     — Reset dialogue history and task state
/help                      — Display help message
/exit                      — Exit chat and save session to disk
```

---

# Day 29: Local LLM Optimization for a Specific Task

Optimize the **local recipe-grilling assistant** (Days 26–28) along three axes and measure the
**before/after** trade-off across **quality, speed, and resources**:

1. **Generation parameters** — `temperature`, `max_tokens`, and the context window (client-side
   history/window **and** the `llama-server` `-c` context size), all editable **in the web UI**.
2. **Quantization** — next to the default **Qwen2.5-3B-Instruct Q4_K_M** another, deliberately
   weaker but lighter/faster **Q2_K** model is available as a second variant.
3. **Prompt template for the specific case** — instead of the generic `RAG_SYSTEM_PROMPT`, a compact
   structured recipe prompt (`Название / Ингредиенты / Приготовление`, strict citations, no meta-leak).

The result is an **optimized local LLM tuned for the recipe task**: shorter, more grounded answers,
faster CPU generation and smaller memory footprint, with a clear quality/speed/resource compromise.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`GenerationParams`** | `agent/optimization.py` | `temperature` / `max_tokens` / `context_window` of one configuration |
| **`ModelVariant`** | `agent/optimization.py` | One GGUF quantization (label, path, size, `tokens_per_sec`, RAM hint); measures the real file size |
| **`PromptTemplate`** | `agent/optimization.py` | Base prompt (`baseline`) vs case-tuned recipe prompt (`recipe_structured`) |
| **`OptimizationStubClient`** | `agent/optimization.py` | Deterministic offline client: baseline-style vs structured answers, token/speed estimates — enables zero-network checks |
| **`compare_optimizations`** | `agent/optimization.py` | Runs several configurations over the control questions and aggregates quality / speed / resources / stability + before/after deltas |
| **`measure_peak_rss_mb`** | `agent/optimization.py` | Peak RSS of the `llama-server` child processes (`resource.getrusage`) |
| **Scenario** | `agent/optimization_scenarios.py` | Offline/live run, web-settings HTTP checks, `history/optimization/report.md` + `results.json` |
| **Web settings** | `agent/web_chat.py`, `web_local.py` | UI panel for model/quantization, preset, prompt template, temperature, max tokens, context window, ctx size; `POST /api/settings` restarts `llama-server` on demand |

## Run

```bash
# 0. (once) build llama.cpp + download both models (Q4_K_M + Q2_K) — see Days 26/29
bash scripts/setup_llama_cpp.sh --quant all

# 1. Deterministic offline optimization check (stub, no model, no network)
python3 -m agent.optimization_scenarios --offline

# 2. Live before/after comparison on the real local models
python3 -m agent.optimization_scenarios
python3 -m agent.optimization_scenarios --variant Q2_K --limit 5

# 3. Web chat with the optimization panel (model + params + prompt template)
python3 web_local.py                       # live llama-server
python3 web_local.py --variant Q2_K        # start on the weaker quantization
python3 web_local.py --preset optimized --temperature 0.2 --max-tokens 384
```

In the web UI a collapsible **«Параметры оптимизации»** panel exposes the model/quantization, a
parameter preset, the prompt template, `temperature`, `max tokens`, the context window and the server
`ctx size`. The first four are applied **per message**; **model** and **ctx size** are applied by
restarting `llama-server` via **«Применить модель/ctx»** (`POST /api/settings`). Every requested and
effective setting is echoed in `GET /api/status` (`variants`, `presets`, `prompt_templates`,
`generation`, `local_variant`, `ctx_size`) and in the `/api/chat` response (`generation`,
`prompt_template`).

## What is evaluated

| Axis | Metrics |
| --- | --- |
| **Quality** | fact coverage, citation rate, source recall@1 |
| **Speed** | latency, generation-only latency, tokens/sec, average completion tokens |
| **Resources** | model file size, RAM hint, peak RSS of `llama-server` |
| **Stability** | error rate, self-consistency (Jaccard) |

Retrieval is shared and local, so the differences come purely from the parameters, the prompt
template and the quantization. In `--offline` mode the metrics come from the deterministic
`OptimizationStubClient`; the live run produces real measurements once both GGUF files are present.

Example offline run (deterministic stub):

```text
До оптимизации               факты   5.8% · ссылки  10.0% ·  10341 мс · 25.0 ток/с · 2007 МБ
После: параметры + промпт    факты  47.5% · ссылки 100.0% ·   7681 мс · 25.0 ток/с · 2007 МБ
После + квант Q2_K           факты  47.5% · ссылки 100.0% ·   4801 мс · 40.0 ток/с · 1150 МБ
Дельта «после-до»: факты +41.7 п.п., ссылки +90.0 п.п., задержка -2660 мс, модель +0 МБ
```

Reports are written to `history/optimization/report.md` and `history/optimization/results.json`.

## What was added

- `agent/optimization.py` — `GenerationParams`, `ModelVariant`, `PromptTemplate` (+ `RECIPE_STRUCTURED_SYSTEM_PROMPT`), presets, `OptimizationStubClient`, `measure_peak_rss_mb`, `compare_optimizations`, `render_optimization_markdown`.
- `agent/optimization_scenarios.py` — offline/live scenario, before/after comparison, web-settings HTTP checks.
- `agent/rag.py` — `build_rag_messages(..., system_prompt=...)`.
- `agent/local_rag.py` — `system_prompt`/`max_tokens`/`context_window` per-request overrides on `LocalRagSystem.ask_with_rag` and `timed_rag_query`.
- `agent/agent.py` — `Agent.ask(..., temperature, max_tokens, context_window)` per-request overrides.
- `agent/web_chat.py` — optimization state, per-message params, `POST /api/settings` (server restart), `GET /api/status`/`/api/chat` extended, UI panel.
- `web_local.py` — `--variant`, `--preset`, `--temperature`, `--max-tokens`, `--context-window`, `--ctx-size`.
- `scripts/setup_llama_cpp.sh` — `--quant q4_k_m|q2_k|all` and the Q2_K download.
- `config.one.json` / `config.example.json` — `local_llm.presets`, `local_llm.variants`, new `optimization` section.

---

# Day 28: Fully Local RAG — Local Retrieval + Local LLM

A **Retrieval-Augmented Generation system that runs entirely on the local machine**:
**question → local FAISS search over the recipe index → local LLM (`llama-server`) → grounded answer + sources**.
No cloud API, no keys, no network at inference time.

The retrieval layer reuses the **recipe index built in Week 6** (`history/index/recipt_all/structural`,
534 recipes → 6,403 vectors, deterministic offline `HashingEmbedder`), and the generation layer reuses the
**Day 26 `llama-server`**. The module also **compares the local model against a cloud model** on the same
retrieved context across three axes: **quality, speed, and stability**.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`LocalRagStubClient`** | `agent/local_rag.py` | Deterministic offline `LLMClient`-compatible client: answers extractively from the RAG context (plus a refusal for the anti-hallucination question). Enables zero-network verification |
| **`LocalRagSystem`** | `agent/local_rag.py` | Lightweight RAG session: local FAISS retrieval + grounded generation through any `.complete`-client (local server, stub, or cloud) |
| **`build_local_rag_client`** | `agent/local_rag.py` | Boots `llama-server` (Day 26) or returns the offline stub → `(server, client)` |
| **`timed_rag_query`** | `agent/local_rag.py` | Runs one RAG query and measures **retrieval** and **generation** time separately |
| **`compare_local_vs_cloud`** | `agent/local_rag.py` | Runs the 10 control questions with repeats; computes quality / speed / stability metrics |
| **`answer_self_consistency`** | `agent/local_rag.py` | Stability metric: exact-match rate + mean pairwise token-Jaccard across repeated answers |
| **Scenario** | `agent/local_rag_scenarios.py` | Offline stub or live `llama-server` run, local-vs-cloud comparison, HTTP check of the RAG web chat, report export |
| **Web RAG chat** | `agent/web_chat.py` (per-message RAG mode + local/cloud model), `web_local.py` | Flask chat over the Day 27 app; two UI dropdowns pick the model (**локальная** / **облачная**) and the mode (**не использовать RAG** / **рецепты**) per message, sources shown under the answer |

## Run

```bash
# 1. Deterministic offline check (stub, no model, no network)
python3 -m agent.local_rag_scenarios --offline

# 2. Live local RAG + comparison with the cloud model
python3 -m agent.local_rag_scenarios

# live local model only (no cloud calls)
python3 -m agent.local_rag_scenarios --no-cloud

# single question demo
python3 -m agent.local_rag_scenarios --question "Сколько соли на 1 кг мяса для прайм риб?"

# 3. Web chat: local + cloud model and a mode selector in the UI
python3 web_local.py

# offline demo of the web chat (deterministic stub)
python3 web_local.py --offline

# disable the cloud model (local model only)
python3 web_local.py --no-cloud

# disable the RAG option entirely (plain chat only)
python3 web_local.py --no-rag
```

In the web UI there are two dropdowns next to the input: the **model** — **локальная** (default) or
**облачная** — and the **mode** — **не использовать RAG** or **рецепты**. Every message is sent with the
chosen model and mode: with «рецепты» the answer is built from the local recipe index and sources are shown
under the reply; with «не использовать RAG» no retrieval is performed at all. The cloud option is available
only when the config has `api_key` and `model` (any OpenAI-compatible API) and is disabled in `--offline`;
local retrieval still applies, so the cloud model answers from the same recipe index. The RAG index is
loaded automatically at startup; if it is missing, the chat still works in the plain mode and selecting
«рецепты» returns a clear error ("RAG (рецепты) недоступен").

Reports are written to `history/local_rag/report.md` and `history/local_rag/results.json`.

Example offline verification run:

```text
✓ Локальный индекс загружен: 6403 векторов, top-5
✓ Локальная генерация готова: local-llm-rag-stub
✓ Веб-чат отдаёт страницу
✓ RAG включён в статусе бэкенда
✓ Ответ содержит и текст, и источники из локального индекса
✓ Источник содержит метаданные рецепта
```

## What is evaluated

| Axis | Local model | Cloud model |
| --- | --- | --- |
| **Quality** | fact coverage, citation rate, source recall@k (retrieval is shared) | fact coverage, citation rate |
| **Speed** | answer latency, generation-only latency, local retrieval latency, tokens/sec | answer latency, generation-only latency, tokens/sec |
| **Stability** | error rate, self-consistency (Jaccard), latency coefficient of variation | error rate, self-consistency, latency CV |

Retrieval is **identical and local** for both models (the same chunks are retrieved once per question), so the
retrieval metrics isolate search quality and the generation metrics isolate the model.

## Relevance guardrail

To avoid showing misleading sources (e.g. 5 low-similarity chunks next to a "don't know" answer), the local
RAG path applies the same relevance threshold as the Day 24 anti-hallucination gate (`0.33`):

- **Low relevance → no model call.** Chunks with `score < 0.33` are dropped before generation. If none
  remain, the system returns an honest refusal («нет информации, релевантной вопросу») with **zero sources**
  and never calls the LLM (also saves cloud tokens in the comparison).
- **Grounded refusal → sources hidden + clean wording.** If the model answers with a refusal ("не знаю" /
  "отсутствует" / "не указано", detected via `agent.grounding.verify_refusal_and_clarification`), the sources
  are cleared and a meta-leaking reply (e.g. «контекст описывает…») is replaced with a canonical refusal.
- **General questions ("что можно приготовить?")** soften the threshold to `local_rag.generic_threshold`
  (0.25) and gather at least `local_rag.generic_min_sources` (3) recipes so the model can suggest options
  instead of refusing.
- **Browse / category queries** (broadened `_BROWSE_MARKERS` + `detect_category`, stems matched by word prefix):
  - «рецепт из говядины» → **category browse**: sources are pulled straight from the index metadata
    (`retriever.index.chunks`, multi-valued `category`) and ranked by cosine similarity to the query
    (`retrieve_by_category`), yielding up to `local_rag.category_suggestions` **distinct** recipes of that
    category (no threshold cut-off needed — the category *is* the filter).
  - A query that names a **specific dish** is kept specific via a guard (`_matches_specific_dish`): significant
    query words (minus browse markers and category stems) are prefix-matched against the top result's title.
    So «рецепт шницеля из говядины» stays a single-dish query, while «рецепт из говядины» becomes a list.
  - «топ 5» / «список рецептов» (no category) → generic route (soft threshold + padding).
- **Follow-up references** ("подробней про 2 позицию", "второй", "расскажи про него"): the previous RAG turn's
  sources are carried in the conversation (`meta.sources`), and the anaphoric question is resolved to a dish
  name before retrieval — deterministically for an explicit ordinal (`_extract_ordinal` +
  `_heuristic_resolve_reference`), and by an **LLM query rewrite with history** for pronouns/pointers
  (`resolve_followup_query` + `FOLLOWUP_REWRITE_SYSTEM_PROMPT`), with the heuristic as fallback. The resolved
  query is re-retrieved and answered on route `followup` (`FOLLOWUP_EXTRA_INSTRUCTION`: one recipe, ingredients
  + steps). If the reference can't be bound, the reply asks to clarify (`CLARIFY_REFERENCE_MESSAGE`) instead of
  flatly refusing, and the generic padding no longer injects below-floor junk.
- **Meta-leak cleanup for every answer** (`strip_meta_leak`): sentences containing «контекст описывает…» /
  «в предоставленном контексте…» are removed (not only on refusals); if nothing useful remains, the canonical
  refusal is returned.
- Configurable via `local_rag.relevance_threshold` (or `--relevance-threshold` in the scenario).

## Dialogue history

The web chat keeps a real conversation, not single turns:

- **Both modes share the same `Conversation`** (Day 7): plain replies go through the `Agent`, and RAG replies
  are recorded explicitly with compact metadata (`meta.mode` + sources), so `/api/history` and the on-disk
  session include every turn. The mode metadata is **never sent to the LLM** (`Agent.context_messages` strips
  everything except `role`/`content`).
- **RAG uses the history**: the last `web_chat.window_size` messages (capped by
  `local_rag.history_max_chars`) are inserted into the prompt before the current question, so the local model
  remembers the dialogue. The same history (including `meta.sources`) is used to resolve follow-up references
  before retrieval.
- **The UI restores history on reload**: the session id and the selected mode are kept in `localStorage`, and
  on page load the messages are fetched from `GET /api/history` and rendered with a per-answer badge
  (**рецепты** / **без RAG**) and, for RAG turns, the sources. `/api/reset` clears the session.
- **Default mode is «рецепты»** when the index is available (falls back to «не использовать RAG» otherwise).
  The server-side API default stays `use_rag=false` for backward compatibility; the UI always sends the choice.

## What was added

- `agent/local_rag.py` — `LocalRagStubClient`, `LocalRagSystem`, `build_local_rag_client`, `timed_rag_query`, `answer_self_consistency`, `compare_local_vs_cloud`, `render_local_rag_markdown`.
- `agent/local_rag_scenarios.py` — offline/live scenario + local-vs-cloud comparison + RAG web-chat HTTP checks (including an off-topic question that must yield a refusal with no sources).
- `agent/web_chat.py` — per-message RAG mode and local/cloud model choice (UI dropdowns, `use_rag`/`used_rag`, `model_source`, `rag_available`/`rag`/`rag_error`, `cloud`/`cloud_model`) **plus dialogue history**: all channels share one `Conversation`, RAG replies are recorded with compact metadata (`meta.mode`, `meta.model_source` + sources), previous turns are fed into the prompt, and the UI restores the session + selected model/mode from `localStorage` with per-answer badges and sources; fully backward compatible with Day 27.
- `agent/conversation.py` — `append(role, content, meta=...)` to persist per-message metadata.
- `agent/agent.py` — `context_messages` strips everything except `role`/`content` before the LLM call.
- `agent/rag.py` — `build_rag_messages(..., history=..., extra_instruction=...)`.
- `agent/local_rag.py` — `select_history` (window + char budget), `is_generic_query`/`has_meta_leak`, generic-query softening, and refusal cleanup.
- `agent/local_rag.py` — category browse (`detect_category`, `retrieve_by_category`, `is_browse_query`), specific-dish guard (`_matches_specific_dish`), sentence-level meta-leak cleanup (`strip_meta_leak`, applied to every answer), and follow-up resolution (`is_followup_query`, `_extract_ordinal`, `_heuristic_resolve_reference`, `resolve_followup_query`, route `followup`, `CLARIFY_REFERENCE_MESSAGE`).
- `web_local.py` — RAG and the cloud model available by default, `--no-rag` / `--no-cloud` to disable them, plus `--dataset`, `--strategy`, `--top-k`, `--index-dir` flags.
- `config.one.json` / `config.example.json` — `local_rag` section now includes `relevance_threshold`, `history_max_chars`, `generic_threshold`, `generic_min_sources`, `category_suggestions`.

---

# Day 27: Local LLM in a Real Application — Web Chat

A small **Flask web application** that sends chat messages to the **local** LLM
from Day 26 (`llama-server`, OpenAI-compatible `/v1` endpoint) and renders the
replies in the browser. The local path needs no cloud models or API keys —
**browser → Flask → local `llama-server`** — and an optional **cloud** model
(any OpenAI-compatible API from the config) can be selected in the UI.

The web layer is a thin adapter over the existing `Agent` / `Conversation`
(unchanged): sessions persist to `history/<session_id>.json`, and a sliding
window keeps the local context bounded. A deterministic **offline stub** lets the
whole app be verified without building llama.cpp or downloading a model.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`WebChatBackend`** | `agent/web_chat.py` | Owns the local server / cloud client lifecycle and the in-memory session registry; `chat()`, `reset()`, `status()`, `history()`, `model_sources` |
| **`ChatSession`** | `agent/web_chat.py` | One dialogue: a `Conversation` + per-source `ChatChannel`s (`local`/`cloud`, each with an `Agent` + optional RAG system) under a per-session lock |
| **`StubLLMClient`** | `agent/web_chat.py` | Offline LLM client compatible with `Agent` (`.config` + `.complete()`), backed by `StubLocalLLM` |
| **`create_app`** | `agent/web_chat.py` | Flask app: `GET /` (HTML chat), `GET /api/status`, `POST /api/chat`, `POST /api/reset`, `GET /api/history` |
| **Entrypoint** | `web_local.py` | CLI: starts the backend and serves the web app (`--offline`, `--no-cloud`, `--no-rag`, `--host`, `--port`, `--llm-port`, `--model`, `--no-browser`) |
| **Scenario** | `agent/web_chat_scenarios.py` | Offline/live verification via Flask's test client; writes `history/web_chat/report.md` + `results.json` |

## Run

```bash
# 0. (once) install the web framework + build llama.cpp + download the model — see Day 26
pip install flask
bash scripts/setup_llama_cpp.sh

# 1. Deterministic offline check (no model, no network)
python3 -m agent.web_chat_scenarios --offline

# 2. Start the web chat on the real local LLM
python3 web_local.py

# offline demo of the web app (deterministic stub)
python3 web_local.py --offline

# custom web port / llama-server port / model
python3 web_local.py --port 8000 --llm-port 8090
```

Then open the printed URL (default `http://127.0.0.1:8000/`).

Example offline verification run:

```text
✓ Веб-приложение отдаёт страницу чата
✓ Статус бэкенда: модель готова
✓ Запрос уходит в локальную модель и возвращается ответ
✓ Второй запрос в ту же сессию
✓ История диалога сохраняется между ходами
✓ История очищается командой /api/reset
✓ Пустое сообщение отклоняется
Итог: 7/7 проверок пройдено.
```

## HTTP API

| Method | Path | Body / Query | Returns |
| --- | --- | --- | --- |
| `GET` | `/` | — | HTML chat UI |
| `GET` | `/api/status` | — | `{ready, mode, model, local_model, cloud, cloud_model, sources, endpoint, sessions, rag}` |
| `POST` | `/api/chat` | `{message, session_id?, use_rag?, model_source?}` | `{reply, session_id, model, model_source, sources, used_rag, rag_available}` |
| `POST` | `/api/reset` | `{session_id?}` | `{session_id, ok}` |
| `GET` | `/api/history` | `?session_id=` | `{session_id, messages}` |

## What was added

- `agent/web_chat.py` — `StubLLMClient`, `ChatChannel`, `ChatSession`, `WebChatBackend`, `create_app`, and the inline HTML chat page (model + RAG selectors).
- `web_local.py` — Flask entrypoint over the local `llama-server` with an optional cloud model.
- `agent/web_chat_scenarios.py` — offline/live end-to-end verification + report (local/cloud routing).
- `config.one.json` / `config.example.json` — new `web_chat` section (`host`, `port`, `window_size`, `max_sessions`).

---

# Day 26: Running a Local LLM with llama.cpp

Run an LLM **fully locally** on CPU with [llama.cpp](https://github.com/ggml-org/llama.cpp) — no cloud API, no keys, no network at inference time. The bootstrap script builds `llama-cli` and `llama-server` from source and downloads a small GGUF model; the scenario then proves all three acceptance criteria:

1. **the model starts locally** — `llama-server` boots and answers `/health` + `/v1/models`;
2. **it is reachable via CLI** — `llama-cli` answers a one-shot prompt;
3. **it answers a simple request** — a question is sent over the OpenAI-compatible `/v1/chat/completions` endpoint.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`LocalLLMConfig`** | `agent/local_llm.py` | Dataclass parsed from the `local_llm` config section: paths, host/port, context size, threads, GPU layers, timeouts |
| **`LocalLLMServer`** | `agent/local_llm.py` | Launches `llama-server`, polls `/health` until ready, lists `/v1/models`, runs one-shot `llama-cli` prompts, stops the process cleanly |
| **`build_local_client`** | `agent/local_llm.py` | Returns `(server, LLMClient)` so any existing agent/RAG flow can run on the local OpenAI-compatible endpoint unchanged |
| **`StubLocalLLM`** | `agent/local_llm.py` | Deterministic offline stub (no binary/model/network) for CI verification |
| **Scenario** | `agent/local_llm_scenarios.py` | Runs the checks, prints a report and writes `history/local_llm/report.md` + `results.json` |
| **Chat CLI** | `chat_local.py` | Starts the local server and runs the standard agent over it (`--prompt` for a single turn) |
| **Bootstrap** | `scripts/setup_llama_cpp.sh` | Idempotent: clones/builds llama.cpp into `history/llama.cpp` and downloads the GGUF into `history/models/` |

Everything the bootstrap produces lives under `history/` (gitignored), so no binaries or model weights ever leak into Git.

## Hardware profile

Tested on a CPU-only host (16 threads, 46 GB RAM, no GPU): `n_gpu_layers=0`, `threads=8`, `ctx_size=8192`, `max_response_tokens=1024`. The default model is **Qwen2.5-3B-Instruct (Q4_K_M, ~2.1 GB)** — a good quality/speed balance for CPU inference.

## Run

```bash
# 1. Build llama.cpp + download the model (idempotent; supports resume)
bash scripts/setup_llama_cpp.sh

# build only / download only
bash scripts/setup_llama_cpp.sh --skip-model
bash scripts/setup_llama_cpp.sh --skip-build

# 2. Verify without build/model/network (deterministic stub)
python3 -m agent.local_llm_scenarios --offline

# 3. Verify the real local model (CLI + server + HTTP)
python3 -m agent.local_llm_scenarios

# custom prompt / port / model
python3 -m agent.local_llm_scenarios --prompt "Назови столицу Франции" --port 8090

# 4. Interactive local chat (or a single prompt)
python3 chat_local.py
python3 chat_local.py --prompt "Сколько будет 2+2?"
```

Example live run:

```text
✓ Бинарники llama.cpp найдены
✓ Файл модели найден
✓ К модели можно обратиться через CLI (llama-cli) (1031 мс)
✓ Модель запускается локально (llama-server) (577 мс)
✓ Модель отвечает на простой запрос (OpenAI API) (184 мс)
  → 2+2=4 [51 токенов]
Итог: 5/5 проверок пройдено.
```

## What was added

- `agent/local_llm.py` — `LocalLLMConfig`, `LocalLLMServer` (start/wait_ready/stop, `list_models`, `complete`, `run_cli_prompt`, `client_config`/`build_client`), `build_local_client`, `StubLocalLLM`, and CLI-output cleaning.
- `agent/local_llm_scenarios.py` — offline/live scenario validating binary discovery, model presence, CLI access, server startup, and a simple HTTP prompt; writes `history/local_llm/report.md` and `results.json`.
- `chat_local.py` — interactive/single-turn local chat over `llama-server`.
- `scripts/setup_llama_cpp.sh` — CPU build of `llama-cli`/`llama-server` + resumable GGUF download.
- `config.one.json` / `config.example.json` — new `local_llm` section (`llama_dir`, `model_path`, `host`, `port`, `ctx_size`, `threads`, `n_gpu_layers`, `max_response_tokens`, `startup_timeout`, `model_label`). `chat_local.py` guards against context overflow (derives `max_context_tokens` from `ctx_size`, uses a sliding window, and supports `/clear`).

---

# Day 24: Citations, Sources, and Anti-Hallucination (Grounded RAG)

A grounded Retrieval-Augmented Generation system ensuring full factual attribution, verifiable citations, and anti-hallucination guardrails:
**user question** → **Vector Retrieval & Reranking** → **Relevance Guardrail Check** (if `< threshold` → **"Не знаю" + clarification request**) → **Grounded Prompting** → **Structured Response: Answer + Sources (`source` + `section`/`chunk_id`) + Quotes (verbatim chunk excerpts)** → **Faithfulness & Semantic Alignment Verification**.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`GroundedSource`** | `agent/grounding.py` | Structured source data model carrying `source` (recipe title), `section`, `chunk_id`, `recipe_id`, `url`, and similarity score |
| **`GroundedAnswer`** | `agent/grounding.py` | Complete grounded result model containing `answer`, `sources`, `quotes`, relevance score, refusal/clarification flags, and alignment evaluation |
| **`AlignmentEvaluation`** | `agent/grounding.py` | Evaluates semantic consistency between answer and quotes: Russian morphological stemming (`_stem_ru`), informational token overlap, number/measurement grounding, and verbatim quote verification |
| **Relevance Guardrail (Усиление)** | `agent/grounding.py` | Programmatic and prompt-level safeguard: if retrieved relevance `< threshold` (0.33) or factual data is absent, immediately returns «Не знаю» and asks for clarification, with 0 hallucinated quotes |
| **Structured Output Parser** | `agent/grounding.py` | Robust parser extracting strict JSON schemas with `answer`, `sources`, and `quotes`, with markdown fence stripping and fallback regex extraction |
| **`GroundedRagAgent`** | `agent/grounding.py` | High-level agent combining retriever, reranker, relevance guardrail, and verification suite |
| **Evaluation Suite** | `agent/grounding.py` | Benchmarks source presence rate, quote presence rate, verbatim quote faithfulness, semantic alignment, and low-relevance refusal behaviors |
| **CLI Scenario** | `agent/grounding_scenarios.py` | Interactive inspection (`--question`), offline verification (`--no-llm`), 10-question benchmark execution, and report export |

## Verification on 10 Control Questions Benchmark

Evaluated against the full recipe knowledge base (`recipt_all`, 534 recipes, 6,403 chunks) using `gemini-3.5-flash-lite`:

| № | Question | Sources Present (`source` + `chunk_id`) | Quotes Present (from chunks) | Meaning Matches Quotes (Semantic Alignment) | Relevance Score | Latency |
| - | --- | --- | --- | --- | --- | --- |
| 1 | Апельсиновая курица на гриле новогодний рецепт… | ✓ Yes (1) | ✓ Yes (1) | ✓ matched (88.4%) | 0.469 | 2640 ms |
| 2 | Прайм риб на гриле обсыпка солью пропорции… | ✓ Yes (3) | ✓ Yes (3) | ✓ matched (90.0%) | 0.440 | 3116 ms |
| 3 | Каре барашка с кашей на гриле гречневая каша… | ✓ Yes (2) | ✓ Yes (2) | ✓ matched (96.2%) | 0.458 | 1992 ms |
| 4 | Лондон бройл на гриле рецепт: сделайте маринад… | ✓ Yes (1) | ✓ Yes (7) | ✓ matched (100.0%) | 0.574 | 2164 ms |
| 5 | Пикантная смесь для курицы на гриле: паприка… | ✓ Yes (1) | ✓ Yes (1) | ✓ matched (97.5%) | 0.650 | 2133 ms |
| 6 | Крылышки 0-190 на пеллетном гриле: суть метода… | ✓ Yes (3) | ✓ Yes (3) | ✓ matched (90.7%) | 0.495 | 3057 ms |
| 7 | Пикантный горчичный соус для свинины на гриле… | ✓ Yes (1) | ✓ Yes (1) | ✓ matched (100.0%) | 0.576 | 2760 ms |
| 8 | Митболы из свинины на гриле сформируйте 40 г… | ✓ Yes (2) | ✓ Yes (2) | ✓ matched (95.0%) | 0.499 | 1921 ms |
| 9 | Рваная курица на гриле: сбрызгивание вода/уксус 50/50… | ✓ Yes (1) | ✓ Yes (1) | ✓ matched (98.0%) | 0.549 | 1968 ms |
| 10 | Говяжьи ребрышки hot & fast: время и вес пластин… | ✓ Yes (3) | ✓ Yes (3) | ✓ matched (94.0%) | 0.430 | 2773 ms |

### Stage 1 Summary:
- **Sources present in every answer:** **100.0% (10/10)** — each source includes recipe name, section, and exact `chunk_id`.
- **Quotes present in every answer:** **100.0% (10/10)** — all substantive quotes extracted verbatim from retrieved chunks.
- **Meaning matches quotes (Semantic Alignment):** **100.0% (10/10)** — facts, proportions, temperatures, and numbers are directly grounded in quoted excerpts.
- **Average latency:** 2452.5 ms.

## Anti-Hallucination Guardrail: Relevance Threshold & "Не знаю" (Усиление)

Rule: If relevance score `< threshold` (default `0.33`) or requested data is missing, the assistant **must** say «Не знаю» and request clarification, leaving sources and quotes empty:

| № | Question | Category | Relevance Score | «Не знаю» | Clarification Requested | Empty Quotes (Zero Fake Quotes) | Verdict |
| - | --- | --- | --- | --- | --- | --- | --- |
| 1 | Какая средняя температура на поверхности Марса? | Out-of-Domain (Астрономия) | 0.228 | ✓ Yes | ✓ Yes | ✓ Yes (0) | ✓ Pass |
| 2 | Как заменить тормозные колодки на автомобиле ВАЗ? | Out-of-Domain (Авторемонт) | 0.161 | ✓ Yes | ✓ Yes | ✓ Yes (0) | ✓ Pass |
| 3 | Рецепт классических суши Филадельфия с лососем | Absent Dish (Не в базе) | 0.158 | ✓ Yes | ✓ Yes | ✓ Yes (0) | ✓ Pass |
| 4 | Апельсиновая курица на гриле: калории и БЖУ? | Missing Data (Нет в рецепте) | 0.405 | ✓ Yes | ✓ Yes | ✓ Yes (0) | ✓ Pass |

### Stage 2 Summary:
- **"Не знаю" Refusal Rate:** **100.0% (4/4)**
- **Clarification Request Rate:** **100.0% (4/4)**
- **Zero Hallucinated Quotes Rate:** **100.0% (4/4)**

Full Markdown report is generated at `history/rag_grounding/grounding_report.md`, and raw JSON data at `history/rag_grounding/grounding_results.json`.

## Run

```bash
# Full benchmark run on 10 control questions + guardrail tests + report generation
python3 -m agent.grounding_scenarios

# Fast offline run (verifies structure, sources, quotes, and guardrails without LLM API calls)
python3 -m agent.grounding_scenarios --no-llm

# Single question demonstration with formatted answer, chunk_id, and quotes
python3 -m agent.grounding_scenarios --question "Апельсиновая курица на гриле новогодний рецепт: как подавать с фруктами розмарином и клюквой?"

# Test out-of-domain question to observe the low-relevance guardrail in action
python3 -m agent.grounding_scenarios --question "Какая средняя температура на поверхности планеты Марс?"

# Custom relevance threshold and top-k context
python3 -m agent.grounding_scenarios --min-relevance 0.35 --top-k 5
```

## What was added

- `agent/grounding.py` — core Grounded RAG module: `GroundedSource`, `GroundedAnswer`, `AlignmentEvaluation`, `GroundedRagAgent`, `GROUNDED_RAG_SYSTEM_PROMPT`, `build_grounded_context()`, `parse_grounded_response()`, `check_relevance_guardrail()`, `verify_sources_presence()`, `verify_quotes_presence()`, `verify_quote_verbatim_faithfulness()`, `verify_semantic_alignment()`, `verify_refusal_and_clarification()`, `render_grounding_markdown_report()`.
- `agent/grounding_scenarios.py` — CLI scenario supporting single query demo, offline verification (`--no-llm`), 10-question benchmark execution, guardrail testing, and report saving.
- `config.example.json` / `config.one.json` — added `grounding` configuration section (`relevance_threshold`, `top_k`, `max_context_chars`, `temperature`, `report_dir`).
- `history/rag_grounding/grounding_report.md` & `history/rag_grounding/grounding_results.json` — evaluation reports verifying sources, quotes, alignment, and threshold guardrails.

# Day 23: Reranking, Relevance Filtering, and Query Rewrite (Two-Stage RAG)

A second-stage enhancement pipeline for Retrieval-Augmented Generation that purifies context, prevents distractors, and boosts answer accuracy:
**user question** → **Query Rewrite** → **First-Stage Retrieval (retrieve_k=15)** → **Reranking (Heuristic/LLM)** → **Relevance Filter (min_score cutoff)** → **Filtered Context (final_k=5)** → **Grounded LLM Query**.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`QueryRewriter`** | `agent/reranking.py` | Transforms colloquial questions into clean search queries; includes `HeuristicQueryRewriter` (stop words stripping, keyword extraction) and `LLMQueryRewriter` with automatic heuristic fallback |
| **`RelevanceFilter`** | `agent/reranking.py` | Filters candidate chunks by similarity threshold (`min_score`), optional relative threshold (`score_ratio`), and protective minimum retention guardrail (`min_keep`) |
| **`HeuristicReranker`** | `agent/reranking.py` | Hybrid cross-feature scorer blending vector similarity, exact lexical token matching, recipe title overlap, and section intent priority |
| **`LLMReranker`** | `agent/reranking.py` | Cross-encoder style candidate scoring using a separate lightweight LLM model from configuration with automatic JSON parsing and heuristic fallback |
| **`RerankPipeline`** | `agent/reranking.py` | End-to-end two-stage orchestrator managing candidates before and after filtering (`retrieve_k` → rerank → filter → `final_k`) |
| **`RerankRagAnswer`** | `agent/reranking.py` | Response model carrying answer content, source attribution, timing breakdown, and full diagnostic details of kept vs dropped candidates |
| **Evaluation Suite** | `agent/reranking.py` | Computes Precision@K, Noise Filter Ratio, Context Reduction, Source Recall, and Fact Coverage across tested modes |
| **CLI Scenario** | `agent/reranking_scenarios.py` | Interactive demo, offline verification (`--no-llm`), 5-mode benchmark execution, and Markdown/JSON report export |

## Two-Stage Parameters: Top-K Before/After & Cutoff Threshold

| Parameter | Default Value | Role in Pipeline |
| --- | --- | --- |
| **`retrieve_k`** (Top-K before filtering) | **15** | Casts a wider net during initial dense FAISS search to capture all potentially relevant sections |
| **`final_k`** (Top-K after filtering) | **5** | Maximum number of purified chunks passed into the LLM context prompt |
| **`min_score`** (Cutoff Threshold) | **0.33** | Strict relevance cutoff discarding weak chunks (below cosine similarity / composite rerank score) |
| **`min_keep`** | **1** | Safety guard ensuring RAG context is never completely empty |
| **`rerank_weights`** | Vector: 0.50, Lexical: 0.30, Title: 0.15, Section: 0.05 | Weights for hybrid composite score in `HeuristicReranker` |

## Empirical Mode Comparison: 10 Control Questions Benchmark

Evaluated against the full recipe knowledge base (`recipt_all`, 534 recipes, 6,403 chunks) using `gemini-3.5-flash-lite`:

| Mode | Precision@K | Source Recall | Top-1 Hit | Fact Coverage | Citations | Avg Chunks | Noise Filtered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1. Baseline (Day 22: no filter, no rewrite)** | 56.0% | 100.0% | 100.0% | 87.5% | 100.0% | 5.0 | 0.0% |
| **2. Filter Only (threshold min_score=0.33)** | 85.2% | 100.0% | 100.0% | 85.0% | 90.0% | 3.3 | 94.8% |
| **3. Rerank Only (heuristic reranker)** | 62.0% | 100.0% | 100.0% | **90.8%** | 90.0% | 5.0 | 84.7% |
| **4. Query Rewrite Only (heuristic rewrite)** | 44.0% | 100.0% | 90.0% | 80.0% | 90.0% | 5.0 | 0.0% |
| **5. Full Improved RAG (rewrite + filter + rerank)** | **82.3%** | **100.0%** | **100.0%** | **90.8%** | 90.0% | **3.7** | **94.6%** |

### Key Impacts of the Two-Stage Enhancement (Full vs Baseline):
- **+26.3% higher context precision**: Precision@K increased from 56.0% to 82.3%, eliminating distractor chunks from unrelated recipes.
- **94.6% noise reduction**: 94.6% of irrelevant candidate chunks retrieved at stage 1 were successfully filtered out before prompt assembly.
- **+3.3% higher factual accuracy**: Fact coverage reached 90.8% due to cleaner context and higher ranking of critical recipe sections.
- **26% context compression**: Reduced average context from 5.0 to 3.7 chunks, lowering prompt token overhead and reducing hallucination surface.

Full Markdown report is generated at `history/rag_rerank/reranking_comparison.md`, and raw JSON data at `history/rag_rerank/reranking_results.json`.

## Run

```bash
# Full benchmark run across all modes with LLM responses + report generation
python3 -m agent.reranking_scenarios

# Fast offline run (verifies retrieval, reranking, and filter metrics with zero API calls)
python3 -m agent.reranking_scenarios --no-llm

# Step-by-step demonstration on a single question (shows rewrite, candidates, filter/rerank decisions)
python3 -m agent.reranking_scenarios --question "Сколько соли нужно на 1 кг мяса для прайм риб на гриле?"

# Quick run on first 3 questions
python3 -m agent.reranking_scenarios --limit 3

# Custom top-K and similarity cutoff threshold
python3 -m agent.reranking_scenarios --retrieve-k 20 --final-k 5 --min-score 0.35
```

## What was added

- `agent/reranking.py` — core two-stage module: `RankedSource`, `RerankOutcome`, `HeuristicQueryRewriter`, `LLMQueryRewriter`, `RelevanceFilter`, `HeuristicReranker`, `LLMReranker`, `RerankPipeline`, `rag_query_with_pipeline()`, evaluation metrics (`evaluate_precision_at_k`, `evaluate_noise_filter`), Markdown report generator (`render_rerank_comparison_markdown`).
- `agent/reranking_scenarios.py` — CLI scenario supporting single-question inspection, offline metric evaluation (`--no-llm`), 5-mode comparative benchmarking, and report saving.
- `config.example.json` / `config.one.json` — added `reranking` configuration section (`retrieve_k`, `final_k`, `min_score`, `score_ratio`, `method`, `rewrite`, `weights`, `report_dir`).
- `history/rag_rerank/reranking_comparison.md` & `history/rag_rerank/reranking_results.json` — evaluation reports with side-by-side mode metrics.

# Day 22: First RAG query — Retrieval-Augmented Generation, Agent with two modes, and comparative evaluation

The first complete RAG pipeline built on top of the recipe knowledge base indexed in Day 21:
**user question** → **vector retrieval (FAISS)** → **context synthesis & prompt combination** → **LLM query**.

The solution implements an **Agent with two operating modes**:
1. **Without RAG (Baseline)**: relies on the general parametric knowledge of the LLM.
2. **With RAG (Retrieval-Augmented)**: grounds responses in top-k relevant chunks from the database with explicit source citations (`[Источник N]`) and strict hallucination prevention.

## Architecture

| Component | Module | Responsibility |
| --- | --- | --- |
| **`RagRetriever`** | `agent/rag.py` | Loads FAISS index (`recipt_all/structural` or `recipt_sample`), performs vector search, formats numbered context blocks |
| **`RagAgent`** | `agent/rag.py` | Agent supporting two modes (`mode="rag"` and `mode="no_rag"`), methods `.ask()`, `.ask_with_rag()`, `.ask_without_rag()` |
| **Prompt Synthesis** | `agent/rag.py` | `build_rag_messages()` with grounded system prompt instructing citation and honest refusal if facts are absent |
| **Control Benchmark** | `agent/rag.py` | 10 control questions with explicit expectations, ground truth facts, and expected sources across 6 categories |
| **Evaluation Suite** | `agent/rag.py` | Automated fact coverage scoring (`evaluate_answer_facts`), source recall (`evaluate_source_recall`), citation checking |
| **CLI Scenario** | `agent/rag_scenarios.py` | Full evaluation workflow, interactive single-question demo, offline mode (`--no-llm`), Markdown report generation |

## 10 Control Questions Benchmark

A curated suite of 10 probe questions covering diverse recipe aspects:

| № | Query | Category | Expected Target Source | Key Ground Truth Facts |
| - | --- | --- | --- | --- |
| 1 | Апельсиновая курица на гриле новогодний рецепт: как подавать с фруктами розмарином и клюквой? | Птица | ID 15454 («Апельсиновая курица на гриле») | Апельсины (2 половинки), виноград, клюква (100 г), розмарин |
| 2 | Прайм риб на гриле обсыпка солью пропорции перец и температура копчения на пеллетном гриле | Говядина | ID 15452 («Прайм риб на гриле») | Соль 1 ч.л./кг, перец 1:1 к соли, копчение 76 °C |
| 3 | Каре барашка с кашей на гриле гречневая каша кедровые орехи брусника гарнир | Баранина | ID 15418 («Каре барашка с кашей») | Гречневая каша (300 г), кедровые орехи, брусника, лук |
| 4 | Лондон бройл на гриле рецепт: сделайте маринад красное вино бальзамический уксус соевый соус чеснок | Говядина | ID 15383 («Лондон бройл на гриле») | Сухое красное вино (100 мл), бальзамический уксус, соевый соус, чеснок |
| 5 | Пикантная смесь для курицы на гриле: за пикантность отвечают молотая паприка гранулированный чеснок порошок чили | Пряные смеси | ID 15381 («Пикантная смесь для курицы») | Паприка, чеснок, чили, перец, соль |
| 6 | Крылышки 0-190 на пеллетном гриле: суть метода выкладки и внутренняя температура готовности мяса | Птица | ID 15379 («Крылышки 0-190») | Выкладка до начала розжига (на холодный гриль), 190-200 °C, внутри 80 °C |
| 7 | Пикантный горчичный соус для свинины на гриле ингредиенты: горчица столовая мед шрирача уксус | Соусы | ID 15362 («Пикантный горчичный соус») | Столовая горчица (7 ч.л.), мёд (2 ч.л.), шрирача, яблочный уксус |
| 8 | Митболы из свинины на гриле шаг сформируйте митболы весом 40 г непрямой средний жар 170-200 | Свинина | ID 15360 («Митболы из свинины») | Вес 40 г, непрямой жар 170-200 °C, копчение 20 мин |
| 9 | Рваная курица на гриле рецепт: смесь для сбрызгивания вода и яблочный уксус 50/50 в пульверизатор | Птица | ID 14965 («Рваная курица на гриле») | Куриные бедрышки, вода и яблочный уксус 50/50 в пульверизатор |
| 10 | Апельсиновая курица на гриле: сколько калорий и белков в порции по рецепту? | Контроль галлюцинаций | ID 15454 («Апельсиновая курица») | **Отрицательный контроль**: данных о КБЖУ в базе нет, модель обязана отказать |

## Evaluation Results: Without RAG vs With RAG

Evaluated against the full dataset index (`recipt_all`, 534 recipes, 6,403 structural chunks) using `gemini-3.5-flash-lite`:

| Metric | Without RAG (Baseline) | With RAG (Retrieval) | Impact / Key Observation |
| --- | --- | --- | --- |
| **Fact Coverage (Полнота фактов)** | **78.3%** | **87.5%** | **+9.2%** higher factual accuracy |
| **Target Recipe in Top-5** | — | **100.0%** | All 10 queries retrieved target recipe in top-5 |
| **Target Recipe at Top-1** | — | **100.0%** | Exact target chunk ranked #1 for every query |
| **Source Citation Rate** | — | **100.0%** | Every RAG answer cites `[Источник N]` or recipe title |
| **Hallucination Control (Q10)** | **0%** (invented 190-220 kcal, 18-20g protein) | **100%** (honest refusal: explicitly stated no calories data in database) | Complete elimination of hallucinations when data is missing |

Full Markdown comparison report is generated at `history/rag/rag_comparison.md`, and raw JSON data at `history/rag/rag_results.json`.

## Run

```bash
# Run full 10-question evaluation scenario across both modes + generate Markdown report
python3 -m agent.rag_scenarios

# Run single question demonstration (shows sources, Baseline answer, and RAG answer)
python3 -m agent.rag_scenarios --question "Сколько соли нужно на 1 кг мяса для прайм риб на гриле?"

# Run with RAG mode only for a custom question
python3 -m agent.rag_scenarios --question "Как приготовить куриные крылышки 0-190 на гриле?" --mode rag

# Fast run on first 3 questions
python3 -m agent.rag_scenarios --limit 3

# Offline mode (verifies retrieval quality and source ranking without LLM calls)
python3 -m agent.rag_scenarios --no-llm

# Custom dataset or top-k
python3 -m agent.rag_scenarios --dataset recipt_all --strategy structural --top-k 5
```

## What was added

- `agent/rag.py` — core RAG module: `RagSource`, `RagAnswer`, `RagRetriever`, `rag_query()`, `plain_query()`, `RagAgent` with two switchable modes (`rag` / `no_rag`), 10 control questions (`CONTROL_QUESTIONS`), evaluation metrics (`evaluate_answer_facts`, `evaluate_source_recall`, `evaluate_citations`, `evaluate_rag_comparison`), Markdown report generator (`render_rag_comparison_markdown`).
- `agent/rag_scenarios.py` — complete CLI scenario for RAG query pipeline: single-question comparison, 10-question benchmark execution, summary metrics table, and automatic report saving.
- `config.example.json` / `config.one.json` — added `rag` configuration section (`dataset`, `strategy`, `top_k`, `max_context_chars`, `temperature`, `report_dir`).
- `history/rag/rag_comparison.md` & `history/rag/rag_results.json` — evaluation reports with side-by-side answers and metric breakdown.

# Day 21: Document indexing — chunking strategies, embeddings, FAISS

A complete document indexing pipeline built for structured recipes (or any domain documents):
**chunking** → **metadata enrichment** → **vector embeddings** → **FAISS index** → **comparative evaluation**.

The pipeline supports loading arbitrary recipe datasets: by default it indexes `recipt_sample.json`,
and the `--source` parameter allows running against any larger or full production dataset.

## Chunking strategies

Two chunking strategies are implemented and compared:

| Strategy | Principle | Metadata fields | Pros & Cons |
| --- | --- | --- | --- |
| **Fixed-size** (`fixed`) | Sliding window of $N$ characters (default 400) with overlap $M$ (default 80) and smart whitespace/sentence cutoffs | `recipe_id`, `title`, `section="document"`, `char_start`, `char_end`, `n_chars`, `n_tokens` | Simple, bounded chunk size, but splits cohesive sections and mixes unrelated text |
| **Structural** (`structural`) | Semantic document decomposition: `header`, `ingredients`, `intro`, `step` (with temperature modes), `equipment` | `recipe_id`, `title`, `section`, `step_number`, `ingredient_count`, `equipment_count`, `temperatures` | Atomic, highly relevant chunks, section-targeted filtering, higher retrieval precision (100% Recall@1) |

## Metadata enrichment

Every generated chunk contains structured metadata:
- `strategy`: name of the chunking strategy (`fixed` or `structural`);
- `recipe_id`, `title`, `category`, `meal_type`, `difficulty`, `author`, `url`;
- `section`: semantic section (`header`, `ingredients`, `intro`, `step`, `equipment`, or `document`);
- `step_number`: integer step index (for recipe steps);
- `char_start`, `char_end`, `n_chars`, `n_tokens`: character offsets and token estimates;
- `source`: dataset file name.

## Vector embeddings & FAISS index

- **`HashingEmbedder`** (default, offline): deterministic feature-hashing embedder combining word unigrams, bigrams, and character 3/4-grams. Blake2b hashing with random sign into 256-dimensional $L_2$-normalized vectors. Zero external network or model weights, reproducible across runs.
- **`OpenAIEmbedder`** (optional): embeddings via OpenAI-compatible `/embeddings` endpoint.
- **`FaissIndex`**: wrapper over `faiss.IndexFlatIP` (exact inner-product on normalized vectors = cosine similarity).
  Saves three artifacts under `history/index/<dataset>/<strategy>/`:
  - `faiss.index` — binary FAISS vector index;
  - `chunks.json` — all chunks with text and enriched metadata;
  - `index_meta.json` — index metadata, dimensions, timestamp, and statistics.
  Full index reload and integrity verification are checked automatically.

## Comparison: Fixed vs Structural

Evaluated on standard recipe probe queries (general queries, ingredient lookup, step/temperature instructions):

- **Recall@1**: Structural reaches **100.0%** vs Fixed **50.0%** (atomic chunks prevent unrelated text dilution).
- **MRR@5**: Structural achieves **1.000** vs Fixed **0.681**.
- **Average top score**: Structural yields higher cosine similarity (**0.3414** vs **0.2793**).
- Full Markdown report is written to `history/index/<dataset>/comparison.md`.

## Run

```bash
# Install dependency (if not installed)
pip install faiss-cpu

# Run with sample recipes (offline, default)
python3 -m agent.indexing_scenarios

# Run with full/production dataset
python3 -m agent.indexing_scenarios --source path/to/full_recipes.json

# Run with custom query against both saved indexes
python3 -m agent.indexing_scenarios --query "маринад для говядины с вином"

# Custom chunk window and top-k
python3 -m agent.indexing_scenarios --chunk-size 350 --overlap 70 --top-k 3
```

## What was added

- `agent/indexing.py` — document loader (`load_recipes`), canonical text converter (`recipe_to_text`), chunking strategies (`chunk_fixed_size`, `chunk_structural`), embedders (`HashingEmbedder`, `OpenAIEmbedder`), FAISS index (`FaissIndex`), evaluation (`evaluate_retrieval`, `compare_strategies`, `render_comparison_markdown`).
- `agent/indexing_scenarios.py` — complete CLI scenario loading datasets, chunking with both strategies, building and saving FAISS indexes, verifying reload, evaluating search quality, and writing comparison report.
- `config.one.json` / `config.example.json` — added `indexing` configuration section (`source_path`, `index_dir`, `embedder`, `embedding_dim`, `chunk_size`, `chunk_overlap`, `top_k`).

# Day 20: Orchestration MCP — several servers, routing and a long agent flow

Several MCP servers are registered at once, their tools are merged into a single
catalog, and the agent itself drives a **long interaction flow** across them:
on every step the LLM picks the next tool on any server, the call is **routed**
to the server that declares it, and its result feeds the next step.

## Registered servers

`agent/orchestrator.py` builds the servers from the `orchestrator.servers`
config section (each entry: `name`, `command`, `args`, `env`, `cwd`, `timeout`).
The GitFlic token/API URL are pulled from the `tools`/`pipeline` sections if a
server does not define them.

| Server | Server file | Tools |
| --- | --- | --- |
| `gitflic` | `mcp/gitflic_server.py` | `get_public_projects`, `count_public_projects`, `schedule_project_count`, `list_count_jobs`, `cancel_count_job`, `get_project_count_report` |
| `pipeline` | `mcp/pipeline_server.py` | `search`, `summarize`, `saveToFile`, `run_pipeline` |
| `report` | `mcp/report_server.py` (new) | `list_reports`, `record_run`, `build_report_index` |

`report` is a local server (no network): it lists saved reports, writes a run
journal (`history/report_runs.json`) and builds a combined Markdown report index.

## Routing

`McpOrchestrator` keeps one `McpToolRuntime` per server. Tools are exposed as
`server.tool`, and `resolve()` finds the target server by a qualified name or by
a short name (an ambiguous or unknown name is an error). `call()` then sends the
invocation to exactly that server and stamps the result with `server`, so it is
always visible which server answered. The agent sees the whole catalog
(`catalog()`) and the server of every tool while choosing the next call.

## Long agent flow

`Agent` accepts either one server (`McpToolRuntime`, Days 17–19) or the
orchestrator. `Agent._maybe_call_tool` is now a **loop** (`max_tool_steps`,
default 6): it asks the LLM for the next tool (the prompt includes the catalog
and all results already received), routes and calls it, appends the result to the
context, and repeats until the model answers `{"tool": null}`, an error occurs,
or a call repeats. The full ordered trace is available via
`agent.last_tool_calls` / `agent.last_tool_results` (`last_tool_call` /
`last_tool_result` still return the last one).

## Verification

`verify_agent_flow(trace, results, required_servers=..., min_servers=...)`
confirms the **choice of servers and the order of calls**:

* at least `min_servers` different servers were used (all `required_servers`);
* every call was routed to the server that declares that tool;
* the order is functionally correct: data is fetched before it is processed,
  processing/saving happens before the run journal, and aggregation happens last.

`describe_flow` prints the trace (`step. [server] tool [status] → result keys`).

## Run

```bash
# offline: agent loop on deterministic stub servers (no network/LLM)
python3 -m agent.orchestration_scenarios --offline

# live: real LLM selects tools, real MCP servers (GitFlic token needed)
GITFLIC_TOKEN=<token> python3 -m agent.orchestration_scenarios --query docs --size 5
GITFLIC_TOKEN=<token> python3 -m agent.orchestration_scenarios --max-steps 6

# routing/verification self-test without MCP
python3 -m agent.orchestrator

# report server without MCP
python3 mcp/report_server.py --check --list
python3 mcp/report_server.py --check --index
```

`agent/orchestration_scenarios.py` walks through: server registration and their
tools, the merged catalog, the long agent flow with the ordered trace, the
server/order check, and the saved report/index files.

In the CLI (`python3 -m agent.cli`) an enabled `orchestrator` section replaces the
single-server runtime; `/tools` lists servers with their tools and
`/tool pipeline.search {"query": "docs"}` calls a routed tool directly.

## What was added

- `mcp/report_server.py` — the third local MCP server (`list_reports`,
  `record_run`, `build_report_index`; `--check --list|--index`).
- `agent/orchestrator.py` — `ServerSpec`, `RoutedTool`, `ToolRoutingError`,
  `McpOrchestrator` (`list_routes`, `catalog`, `resolve`, `call`, `summarize`,
  `plan_messages`, `parse_plan`, `result_message`), `StaticToolRuntime` (offline
  stubs), `orchestrator_from_config`, `verify_agent_flow`, `describe_flow`, and an
  offline self-test (`python3 -m agent.orchestrator`).
- `agent/agent.py` — `tools` may be an orchestrator; `_maybe_call_tool` runs a
  bounded multi-step loop; `max_tool_steps`, `last_tool_calls`,
  `last_tool_results`.
- `agent/mcp_tools.py` — `ToolCall.server` / `ToolResult.server`;
  `plan_messages(user, results)`; server-aware `result_message`.
- `agent/orchestration_scenarios.py` — end-to-end scenario with a live and an
  offline (`--offline`, `ScriptedLLMClient`) mode.
- `agent/cli.py` — the orchestrator is preferred over the single server;
  `/tools` shows servers, `/tool <server>.<tool>` routes a call, the reply prints
  the full flow trace.

Config keys (in the `orchestrator` section): `enabled`, `max_steps`, `servers`
(each with `name`, `command`, `args`, `env`, `cwd`, `timeout`).

# Day 19: Composition of MCP tools — search → summarize → saveToFile

Several independent MCP tools are composed into an **automatic pipeline**: the
first tool **fetches** data, the second **processes** it, the third **saves** the
result. The chain is executed automatically and data is passed from tool to tool.

## Tools

`mcp/pipeline_server.py` (FastMCP, stdio) registers:

| Tool | Role in the pipeline | What it returns |
| --- | --- | --- |
| `search` | **fetches** public GitFlic projects (`mcp/gitflic_api.py`) | `{query, count, total, page, source, items: [{id,title,description,language,owner,topics,webUrl}]}` |
| `summarize` | **processes** the projects (local extractive summarization, no network/LLM) | `{summary, bullets, keywords, itemCount, sentenceCount, wordCount}` |
| `saveToFile` | **saves** the result atomically inside the project dir | `{path, relativePath, format, bytes, lines, savedAt}` |
| `run_pipeline` | composes all three **in one MCP call** | `{ok, query, savedPath, summary, steps:[{name, tool, arguments, result}]}` |

`summarize` uses `mcp/pipeline_text.py`: deterministic extractive summarization
(sentence scoring by word frequency, stop-words ru/en). The same input always
produces the same output, so data passing can be checked without a network or an
LLM. `saveToFile` rejects any path that escapes the project root.

## Composition and data passing

`agent/pipeline.py` is the composition engine. Steps are declared and the engine
runs them in order, resolving references of the form `${step.path.to.field}` in
the next step's arguments from the previous steps' results:

```python
PipelineStep("search",    "search",    {"query": q, "size": n})
PipelineStep("summarize", "summarize", {"items": "${search.items}"})
PipelineStep("saveToFile","saveToFile",{"content": "${summarize.summary}", "path": out})
```

A string that is exactly one reference is replaced by the **original object**
(list/dict), so structured data — not its string form — is passed between tools.
`gitflic_summary_pipeline(...)` builds the chain above.

`verify_pipeline_run(run)` checks the **correctness of data passing**: every step
succeeded; the `items` that `summarize` received are exactly the `items` that
`search` returned; the `content` that `saveToFile` received is exactly the
`summary` that `summarize` returned; and the file exists and matches `content`.

## Run

```bash
# offline self-test of the composition engine (no MCP, no network)
python3 -m agent.pipeline

# full scenario: discovery, automatic chain, data-passing check, composite call
GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --query docs --size 5
GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --skip-composite
GITFLIC_TOKEN=<token> python3 -m agent.pipeline_scenarios --agent "собери отчёт по проектам gitflic про docs"

# direct tool calls without MCP
GITFLIC_TOKEN=<token> python3 mcp/pipeline_server.py --check --query docs
GITFLIC_TOKEN=<token> python3 mcp/pipeline_server.py --check --pipeline --query docs
```

`agent/pipeline_scenarios.py` walks through: tool discovery, the automatically
executed `search → summarize → saveToFile` chain, the data-passing verification,
the server-side composite `run_pipeline`, the saved report, and an optional
agent-level call. The GitFlic token comes from `GITFLIC_TOKEN` or the `pipeline`
section, falling back to `tools.gitflic_token`.

## What was added

- `mcp/pipeline_text.py` — local extractive summarization (`tokenize`,
  `split_sentences`, `summarize_text`, `summarize_items`, `item_bullet`).
- `mcp/pipeline_server.py` — the four tools above; `search` reuses the GitFlic
  API layer, `saveToFile` writes atomically inside the project; `--check`
  (`--pipeline`, `--query`, `--size`, `--output`) runs tools without MCP.
- `agent/pipeline.py` — `PipelineStep`, `StepResult`, `PipelineRun`, `Pipeline`
  (automatic sequential execution), `resolve_refs` (`${...}` references),
  `gitflic_summary_pipeline`, `verify_pipeline_run`, `pipeline_runtime_from_config`
  and an offline self-test (`python3 -m agent.pipeline`).
- `agent/pipeline_scenarios.py` — end-to-end scenario with the chain, the
  data-passing check and the composite call.

Config keys (in the `pipeline` section): `enabled`, `command`, `args`,
`output_path`, `timeout`, `gitflic_token`, `gitflic_api_url`.

# Day 18: Planner and background tasks — counting GitFlic public projects

The MCP server from Day 17 now also ships a **tool with delayed/periodic
execution** around the public GitFlic API: it counts how many public projects
exist, aggregates the result (total, by language, by owner), and **saves the
data as JSON**. Periodic execution is driven by **cron**.

## Tools

`mcp/gitflic_server.py` registers, besides `get_public_projects`:

| Tool | What it does |
| --- | --- |
| `count_public_projects` | Counts public projects **now**, aggregates, saves a JSON snapshot, returns the aggregate. |
| `schedule_project_count` | Schedules the count **delayed** (`delay_seconds`) or **periodic** (`interval_seconds`); returns a `jobId`. |
| `list_count_jobs` | Lists scheduled jobs: status, next run, number of runs. |
| `cancel_count_job` | Cancels a periodic job by `jobId`. |
| `get_project_count_report` | Returns the aggregated result of the last count and the snapshot history. |

The exact number of public projects comes from the API pagination
(`totalElements`) on the first page, so the count is accurate and fast; the
by-language/by-owner breakdown is a **sample** over `max_pages` pages
(`countedProjects`, `sampled`).

## Data and scheduling

- **JSON storage** — `history/gitflic_counts.json` keeps snapshots
  (`{"createdAt", "updatedAt", "snapshots": [...]}`) with
  `totalPublicProjects`, `languages`, `topOwners`, `topTopics`, timestamps.
  Job schedules live in `history/gitflic_jobs.json`. Both are written
  atomically, and both paths can be overridden with `GITFLIC_COUNT_STORE` /
  `GITFLIC_JOBS_STORE`.
- **Schedule/execution** — because an MCP stdio session lives for a single tool
  call, schedules are **persisted** and executed by an external runner:
  `mcp/gitflic_cron.py`. `--run-due` executes every job whose `nextRunAt` is
  due (and re-schedules periodic ones); `--once` performs a plain count. The
  same module can run as a daemon (`--daemon`).

## Run

```bash
# tool discovery, delayed job, cron execution, aggregated report
GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios
GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios --interval 60
GITFLIC_TOKEN=<token> python3 -m agent.scheduler_scenarios --agent "сколько публичных проектов в gitflic"

# direct tool call (count + aggregate + JSON snapshot)
GITFLIC_TOKEN=<token> python3 mcp/gitflic_server.py --check --count
# the cron entrypoint (token falls back to tools.gitflic_token in config)
python3 mcp/gitflic_cron.py            # one count
python3 mcp/gitflic_cron.py --run-due  # run scheduled jobs
python3 mcp/gitflic_cron.py --print-crontab  # ready-to-paste lines
```

`gitflic_cron.py` takes the token from `GITFLIC_TOKEN`, and if it is not set,
from `tools.gitflic_token` in `config.one.json` / `config.json`. API/network
errors are reported as an aggregated `{"error": ...}` result with exit code 1.

Crontab (installed once):

```cron
# count public projects every hour
5 * * * * cd /path/to/ai_advent && GITFLIC_TOKEN=<token> python3 mcp/gitflic_cron.py >> history/gitflic_cron.log 2>&1
# run delayed/periodic jobs every minute
* * * * * cd /path/to/ai_advent && GITFLIC_TOKEN=<token> python3 mcp/gitflic_cron.py --run-due >> history/gitflic_cron.log 2>&1
```

## What was added

- `mcp/gitflic_api.py` — shared low-level GitFlic API access
  (`request_json`, `normalize_project`, `fetch_public_projects`), used by the
  server, the scheduler and the cron runner.
- `mcp/gitflic_scheduler.py` — `CountStore` (snapshots JSON), `JobStore`
  (schedules JSON), `aggregate_projects`, `ProjectCounter`, and
  `ProjectCountScheduler` (`schedule`, `due_jobs`, `run_due`, `start_daemon`).
- `mcp/gitflic_server.py` — the five new tools above; `--check --count`.
- `mcp/gitflic_cron.py` — cron entrypoint (`--once`, `--run-due`, `--daemon`,
  `--print-crontab`).
- `agent/scheduler_scenarios.py` — end-to-end scenario: discovery, delayed job,
  cron execution, aggregated report, optional agent call.

# Day 17: The first MCP tool — public GitFlic projects

`mcp/gitflic_server.py` is a **own MCP server** built on the official `mcp` SDK
(`FastMCP`) around the public GitFlic API. It registers one tool,
`get_public_projects`, which returns the list of public projects via
`GET https://api.gitflic.ru/project`.

The server does exactly what the task asks:

1. **Tool registration** — the `@mcp.tool()` decorator registers
   `get_public_projects`; clients discover it via `tools/list` (visible in the
   server info with name, description and schema).
2. **Input parameters** — the tool declares `query: str = ""`, `page: int = 0`,
   `size: int = 10`; FastMCP derives the JSON input schema from the types and
   the docstring `Args`, so the client sees each parameter and its meaning.
3. **Result** — the tool returns a result: `projects` (id, title, alias,
   description, language, owner, clone URLs, topics), `page`, `count` and
   `total`.

`agent/mcp_tools.py` is the **MCP runtime** that connects the server to the
agent (stdio transport: the client launches the server as a subprocess). The
`Agent` calls a tool in two steps (`Agent._maybe_call_tool`): a short LLM request
picks the tool and its arguments (`plan_messages` + `parse_plan`), then the call
goes through MCP (`call`), and the result is injected into the context as a
system message so the main model answers from real data.

## Authorization

The GitFlic API requires an access token (scope `PROJECT_READ`) in the header
`Authorization: token <token>`. Put it in the `tools.gitflic_token` config key
or in the `GITFLIC_TOKEN` environment variable. The API base can be overridden
with `gitflic_api_url` / `GITFLIC_API_URL` (default `https://api.gitflic.ru`).

## What was added

- `mcp/gitflic_server.py` — the MCP server: `get_public_projects` tool,
  request to `/project`, result normalization, and a `--check` mode that runs
  the tool directly and prints JSON.
- `agent/mcp_tools.py` — `McpToolRuntime` (stdio client: `list_tools`, `call`,
  `summarize`, `plan_messages`, `parse_plan`, `result_message`), `ToolSpec`,
  `ToolCall`, `ToolResult`, and `runtime_from_config`.
- `agent/agent.py` — accepts `tools`; `_maybe_call_tool()` selects and calls the
  MCP tool and injects the result into the request. `last_tool_call` /
  `last_tool_result` expose what happened.
- `agent/cli.py` — reads the `tools` section, shows a notice on start, and adds
  commands `/tools` (list tools + parameters) and `/tool <name> [<json>]`
  (call a tool directly and print the result).
- `agent/tool_scenarios.py` — scenario: tool registration/discovery, input
  parameter schema, a direct call with the result, and an agent-level call
  (`--agent`) where the agent chooses the tool and uses the result.

Config keys (in the `tools` section): `enabled`, `command`, `args`,
`gitflic_token`, `gitflic_api_url`, `timeout`.

## Run

```bash
cd ai_advent
GITFLIC_TOKEN=<token> python3 mcp/gitflic_server.py --check    # direct tool call
GITFLIC_TOKEN=<token> python3 -m agent.tool_scenarios          # discovery + call
GITFLIC_TOKEN=<token> python3 -m agent.tool_scenarios --agent "покажи публичные проекты gitflic"
python3 -m agent.cli                                           # /tools, /tool get_public_projects {"size": 5}
```

# Day 16: MCP connection and tool discovery

`agent/mcp_client.py` is a **minimal MCP client** built on the official
[`mcp`](https://pypi.org/project/mcp/) SDK (Streamable HTTP transport). It does
exactly what the task asks:

1. **Establishes an MCP connection** — opens a Streamable HTTP session and
   initializes it (`streamablehttp_client` + `ClientSession.initialize`).
2. **Fetches the list of available tools** — calls `tools/list` and prints every
   tool with its description and input parameters.
3. **Verifies the result** — the `--check` mode asserts that the server is
   reachable, the protocol is negotiated, and the tool list is non-empty and
   well-formed.

The default server is the Aurora OS developer portal
(`https://developer.auroraos.ru/api/mcp`, server `dev-aurora`). It only answers
browser clients, so the client sends a **browser `User-Agent`** header
(`BROWSER_USER_AGENT`, overridable with `--user-agent`). Session termination via
`DELETE` is disabled (`terminate_on_close=False`) because that server rejects it.

## Install

```bash
pip install mcp
```

## Run

```bash
cd ai_advent && python3 -m agent.mcp_client          # connect and print tools
cd ai_advent && python3 -m agent.mcp_client --check  # connect and self-check
cd ai_advent && python3 -m agent.mcp_client --json   # machine-readable output
```

Example (truncated):

```
==============================================================
MCP: подключение и список инструментов
==============================================================
URL сервера     : https://developer.auroraos.ru/api/mcp
Протокол        : 2025-03-26
Сервер          : dev-aurora 1.0.0
Соединение      : установлено
Инструментов    : 8

  1. get_doc_versions
  2. search           параметры: query*, index, version, limit  (* — обязательный)
  3. search_code      параметры: query*, limit
  ...
```

## What was added

- `agent/mcp_client.py` — `connect_and_list_tools()` (minimal connection +
  `tools/list`), `verify_connection()`, pretty/JSON printers and a CLI
  (`--url`, `--user-agent`, `--timeout`, `--json`, `--check`).

Config keys (optional `mcp` section): `url`, `user_agent`, `timeout`.

# Day 15: Controlled state transitions of a task

`transition_scenarios.py` makes the agent's **task lifecycle strictly controlled**.
A task moves through `planning → execution → validation → done`, and the assistant
**cannot jump a stage**. Transitions are controlled at two levels:

1. **Valid states and allowed transitions.** Each stage lists what is allowed
   (`STAGE_SCOPE` in `agent/task_state.py`) and what belongs to a later stage and
   is forbidden now. `TRANSITIONS` defines which stage may follow which: `planning`
   does not jump straight to `validation`, and `execution` cannot reach `done`
   without `validation`. While the current stage is not marked complete, a forward
   transition is rejected by `set_stage()`/`advance()`.
2. **The assistant cannot jump a stage.** The task's system message now tells the
   model what is allowed and what is forbidden on the current stage. If the user
   asks for work of a later stage (implementation before an approved plan, or a
   final deliverable without validation), the assistant **refuses** and holds the
   stage — this is a deterministic check (`guard_request`), no LLM call needed. If
   the assistant itself produces a result of a later stage, a short guard check
   (`transition_guard_messages` + `parse_transition_verdict`) **replaces** the
   answer with a refusal, so a jumped stage never lands in history.

The scenario walks through: valid states and allowed transitions, "cannot do
implementation before an approved plan", "cannot finalize without validation",
an attempt to move to an invalid state being rejected, the assistant's reaction
(refusal) when it would jump a stage, and a pause/resume that continues from the
same stage without repeated explanations.

## What was added

- `agent/task_state.py` — `STAGE_SCOPE` (per-stage allowed/forbidden work),
  `STAGE_JUMP_KEYWORDS` (deterministic jump detection), `TRANSITION_GUARD_PROMPT` /
  `parse_transition_verdict` / `TransitionVerdict` (LLM guard for the reply).
  `TaskState.system_message()` now includes the allowed/forbidden scope.
  `TaskStateMachine` gains `strict` (default on), `guard_request()`,
  `transition_guard_messages()` and `refusal_message()`.
- `agent/agent.py` — accepts the strict task; `_stage_guard_reason()` blocks a
  user's jump before calling the model, and `_enforce_stage_guard()` runs after
  each reply and replaces a jumping answer with a refusal.
  `last_transition_violation` exposes whether a refusal fired.
- `agent/cli.py` — reads `task.strict` from config, adds the `/task_strict`
  toggle, and prints a notice when a stage jump was refused.
- `agent/transition_scenarios.py` — live or demo (`--demo`) scenario that shows
  the controlled lifecycle, blocked jumps, and correct continuation after a pause.

Config keys (in the `task` section): `enabled`, `auto_advance`, `strict`.

## Run

```bash
cd ai_advent && python3 -m agent.transition_scenarios --demo   # fixed scenario, offline
cd ai_advent && python3 -m agent.transition_scenarios          # live input, offline
cd ai_advent && python3 -m agent.transition_scenarios --online # real API requests
```

Commands (not sent to the model): `task`, `task_new <описание>`,
`task_next`, `task_stage <этап>`, `task_step <N>`, `task_expected <действие>`,
`task_complete`, `task_auto`, `task_strict`, `task_pause`, `task_resume`,
`task_done`, `task_clear`, `context`, `exit`.

# Day 14: Invariants and state constraints

`invariant_scenarios.py` makes the assistant work within a set of **invariants** —
hard limits it must not violate: the chosen architecture, accepted technical
decisions, stack constraints, and business rules. Unlike the dialog, invariants
**live separately** in their own `invariants.json` and do not depend on what the
conversation is about.

Invariants are enforced in two complementary ways:

1. **Explicitly in reasoning.** The invariants' system message is prepended to
   every request (right after the profile, before the task), so the model sees
   the limits *before* proposing a solution and is instructed to check each
   candidate against every invariant.
2. **By a guard check.** After the assistant answers, the agent runs a short
   LLM check (`guard_check_messages` + `parse_guard_verdict`) — does the
   proposal violate any invariant? If yes, the answer is **replaced** by a
   refusal (`Invariants.refusal_message`) that names the violated invariant and
   explains why it cannot be broken.

So on a conflict between the request and an invariant the assistant **refuses**
and explains exactly which invariant is violated and why. This works even if the
model itself is "weak": the guard catches the violation and substitutes a
refusal, so a violating proposal never lands in the history. When an invariant
becomes obsolete, `/invariant_del <id>` removes it and the request becomes
allowed again.

## What was added

- `agent/invariants.py` — `Invariant` (id, category, description, rationale),
  `Invariants` (separate JSON store, `add`/`remove`/`get`/`clear`,
  `system_message()`, `guard_check_messages()`, `refusal_message()`,
  `summarize()`), `parse_guard_verdict`/`GuardVerdict`.
- `agent/agent.py` — accepts `invariants`; their system message is inserted
  after the profile and counted in `context_tokens`. After each answer
  `_enforce_invariants()` runs the guard check and, on a violation, replaces the
  reply with a refusal. `last_guard_violation` exposes whether a refusal fired.
- `agent/cli.py` — reads the `invariants` section and adds commands:
  `/invariant` (list), `/invariant_add <категория> <описание> [-- причина]`,
  `/invariant_del <id>`, `/invariant_clear`, `/invariant_enforce`, plus a
  refusal notice after replies.
- `agent/invariant_scenarios.py` — live or demo (`--demo`) scenario that shows
  invariants stored apart from the dialog, their system message in every
  request, a model-side refusal on a conflict, a guard-caught violation, and the
  request becoming allowed once the invariant is removed.

Config keys (in the `invariants` section): `enabled`, `path`, `enforce`.

## Run

```bash
cd ai_advent && python3 -m agent.invariant_scenarios --demo   # fixed scenario, offline
cd ai_advent && python3 -m agent.invariant_scenarios          # live input, offline
cd ai_advent && python3 -m agent.invariant_scenarios --online # real API requests
```

Commands (not sent to the model): `invariant`, `invariant_add <категория> <описание>`,
`invariant_del <id>`, `invariant_clear`, `invariant_enforce`, `context`, `exit`.

# Day 13: Task state as a finite state machine

`task_scenarios.py` makes the agent's **task state** formal instead of a free
string. The current task is described by a finite state machine with three
fields:

- **Stage** (`stage`) — where the task is right now: `planning → execution →
  validation → done`.
- **Current step** (`step`) — the step number within the stage.
- **Expected action** (`expected_action`) — what the agent must do next.

Stages are transitions of an automaton (`TRANSITIONS` in `agent/task_state.py`):
you cannot skip a stage or go back via an illegal transition (e.g. `planning →
validation` is rejected). If validation fails, the automaton allows a legal
rollback `validation → execution`, then `validation → done`.

**Pause is allowed at any stage**: `pause()` just flips the flag, it never
breaks the state machine. **Resume continues from the same place** — the task
description, stage and expected action stay in the state and are re-injected as
a system message, so the agent needs no repeated explanations.

**You cannot skip real work.** Every stage must be explicitly marked complete
(`complete()`) before a forward transition; `advance()`/`set_stage()` reject a
move forward while the current stage is not done. A corrective rollback
(`validation → execution` when validation fails) stays allowed without
completion. Steps move only forward too.

**Auto-advance** (`task.auto_advance`): when enabled, the agent does not wait
for a manual `/task_complete`. After each answer it makes a short completion
check to the LLM ("is the current stage's expected action fulfilled?") and, if
the model says yes, automatically completes the stage and moves the task on.
You can toggle it live with `/task_auto`.

## What was added

- `agent/task_state.py` — `TaskState` (stage/step/expected_action/description/
  paused/completed/log) and `TaskStateMachine` (wraps the state, validates
  transitions, auto-advance flag, completion check, `system_message()`,
  `summarize()`).
- `agent/agent.py` — accepts `task`; the state's system message is inserted
  after the profile and counted in `context_tokens`. `save_task()` persists the
  state into the session. `_maybe_auto_advance_task()` runs the completion check
  and advances automatically.
- `agent/conversation.py` — stores a `task` block in the session JSON, so the
  state survives a restart.
- `agent/cli.py` — reads the `task` section and adds commands: `/task`,
  `/task_new <описание>`, `/task_next`, `/task_stage <этап>`, `/task_step <N>`,
  `/task_expected <действие>`, `/task_complete`, `/task_auto`, `/task_pause`,
  `/task_resume`, `/task_done`, `/task_clear`.
- `agent/task_scenarios.py` — live or demo (`--demo`) scenario that walks a task
  through its full lifecycle with auto-advance, shows an illegal transition
  being rejected, a pause and a resume without repeated explanations.

Config keys (in the `task` section): `enabled`, `auto_advance`.

## Run

```bash
cd ai_advent && python3 -m agent.task_scenarios --demo   # fixed scenario, offline
cd ai_advent && python3 -m agent.task_scenarios          # live input, offline
cd ai_advent && python3 -m agent.task_scenarios --online # real API requests
```

Commands (not sent to the model): `task`, `task_new <описание>`,
`task_next`, `task_stage <этап>`, `task_step <N>`, `task_expected <действие>`,
`task_complete`, `task_auto`, `task_pause`, `task_resume`, `task_done`,
`task_clear`, `context`, `exit`.

# Day 12: Personalization of the assistant — user profile over the memory model

`personalization_scenarios.py` adds a **user profile** layer on top of the memory
model (see Day 11). Instead of just remembering facts, the agent now holds a
profile — who the user is and their preferences — and **automatically** injects
it into every request. The user doesn't have to repeat "answer briefly", "no
water", "use lists": these rules live in the profile and go to the model on
their own.

A profile has three preference groups plus identity:

- **Identity** — who the user is and their context (so the assistant speaks at
  the right level).
- **Style** (`style`) — how to answer: tone, formality, brevity, emoji.
- **Format** (`format`) — how to structure the answer: bullet lists, headings,
  tables, markdown, code, step-by-step.
- **Constraints** (`constraints`) — what to avoid: fluff, jargon, extra details,
  answer length.

Profiles are stored in a separate `profiles.json` and survive sessions. The
active profile's system message is **prepended to every request** and counted in
`context_tokens`, so the assistant adapts automatically without any extra
instructions.

The scenario runs the **same question** through three different profiles —
*Инженер* (terse, technical, with code), *Менеджер* (structured, with a
takeaway and a next step), *Новичок* (simple, step-by-step, no jargon) — and
shows each got its own profile block and a different, adapted answer.

## What was added

- `agent/personalization.py` — `UserProfile` (identity + style/format/constraints
  preferences), `Personalization` (manages named profiles + active profile,
  persisted to `profiles.json`), `system_message()` / `summarize()`.
- `agent/agent.py` — accepts `personalization`; the active profile's system
  message is inserted first into the request and counted in `context_tokens`.
- `agent/cli.py` — reads the `personalization` section and adds commands:
  `/profile` (show profiles), `/profile_new <имя>`, `/profile_use <имя>`,
  `/profile_del <имя>`, `/profile_off`, `/preference <группа> <ключ> = <значение>`,
  `/preference_del <группа> <ключ>`.
- `agent/personalization_scenarios.py` — live or demo (`--demo`) scenario that
  compares one question across several profiles.

Config keys (in the `personalization` section): `enabled`, `profiles_path`.

## Run

```bash
cd ai_advent && python3 -m agent.personalization_scenarios --demo   # fixed scenario, offline
cd ai_advent && python3 -m agent.personalization_scenarios          # live input, offline
cd ai_advent && python3 -m agent.personalization_scenarios --online # real API requests
```

Commands (not sent to the model): `profile`, `profile_new <имя>`,
`profile_use <имя>`, `profile_del <имя>`, `profile_off`,
`preference <style|format|constraints> <ключ> = <значение>`,
`preference_del <группа> <ключ>`, `context`, `exit`.

# Day 11: Agent memory model — three separate memory layers

`memory_scenarios.py` demonstrates an agent with an explicit memory model that
splits information into three layers, each stored separately and filled
explicitly (you choose what goes where):

- **Short-term** — the current dialog (messages exchanged in this session).
- **Working** — the current task's data (values the user entered, intermediate
  results, constraints of this task). Lives within the session and is cleared on
  a task switch (`clearworking`).
- **Long-term** — profile, decisions, and knowledge that the agent writes to a
  separate `profile.json` and remembers even in a brand-new session.

The layers are physically separate: the dialog lives in the session JSON, the
working memory in that session's `working` block, and the long-term memory in a
dedicated profile file. `build_memory_context()` assembles the layers into
system messages that are prepended to the request, so the agent actually sees
them.

## What was added

- `agent/memory.py` — `MemoryLayers` (the model), `WorkingMemory`,
  `LongTermMemory`, `LongTermKind` (profile/decision/knowledge). Explicit API:
  `remember_short_term`, `remember_working`, `remember_long_term`,
  `build_memory_context`, `summarize`.
- `agent/conversation.py` — a session now stores a `working` block
  (key-value) for the current task: `set_working`, `update_working`,
  `clear_working`, and the `working` property. Old formats still load.
- `agent/agent.py` — accepts `memory`; the memory layers' system messages are
  prepended to the request context and counted in `context_tokens`.
- `agent/cli.py` — reads the `memory` section and adds commands: `memory`
  (show all layers), `remember <short|working|profile|decision|knowledge>
  <ключ> = <значение>`, `forget <категория> <ключ>`, `clearworking`.
- `agent/memory_scenarios.py` — live or demo (`--demo`) scenario that shows what
  lands in each layer, how it affects the request, and how the agent answers
  after a task switch wipes working memory but keeps long-term memory.

Config keys (in the `memory` section): `enabled`, `profile_path`.

## Run

```bash
cd ai_advent && python3 -m agent.memory_scenarios --demo   # fixed scenario, offline
cd ai_advent && python3 -m agent.memory_scenarios          # live input, offline
cd ai_advent && python3 -m agent.memory_scenarios --online # real API requests
```

Commands (not sent to the model): `memory`, `remember <слой> <ключ> = <значение>`,
`forget <категория> <ключ>`, `clearworking`, `memory_reset`, `context`, `exit`.

# Day 10: Context management — strategies without summary

`context_scenarios.py` runs the same dialog under three different context-management
strategies (and a no-strategy baseline) and compares how the context grows, how many
tokens/cost the session uses, and whether the agent still retains early facts.

- **Sliding Window** — keeps only the last N messages; everything older is
  **discarded** (not just hidden from the request).
- **Sticky Facts (Key-Value Memory)** — important data (goal, constraints,
  preferences, decisions, agreements) is pulled into a `facts` block
  (key-value) that is sent together with the last N messages. Facts are updated
  after every user message.
- **Branching** — a `checkpoint` is saved, independent dialog branches are forked
  from it, the conversation continues separately in each branch, and you switch
  between branches. The common part (checkpoint) stays shared, so early facts
  survive.

## What was added

- `agent/context.py` — `ContextStrategy` (base), `SlidingWindowStrategy`,
  `StickyFactsStrategy`, `BranchingStrategy`, `extract_facts_from_text`
  (deterministic local fact extractor) and `strategy_from_config`.
- `agent/conversation.py` — a session now also stores `facts` (key-value) and the
  branching state: `checkpoint` (shared prefix), `branches` (id -> messages),
  `active_branch`. New methods: `set_fact`, `update_facts`, `trim_to`,
  `checkpoint`, `create_branch`, `switch_branch`. Old formats still load.
- `agent/agent.py` — accepts `context_strategy`; the request is built from the
  strategy's `build_messages`, and `on_user_message`/`on_reply` hooks run after
  each turn (fact extraction, window trimming).
- `agent/cli.py` — reads the `context` section, shows the active strategy, and
  adds commands: `context` (show facts/branches), `checkpoint`,
  `branch <name>`, `switch <name>`.

Config keys (in the `context` section): `strategy`
(`sliding_window` | `sticky_facts` | `branching`), `size` / `window_size`,
`max_facts`.

## Run

```bash
cd ai_advent && python3 -m agent.context_scenarios          # offline (responses simulated)
cd ai_advent && python3 -m agent.context_scenarios --online # real API requests
```

Run it from the project root with `python3 -m agent.context_scenarios`. It starts
four agents (one per strategy plus a no-strategy baseline) and feeds every line
you type through all of them at once, so you can watch how the context, per-turn
tokens, and cost diverge. There is no hardcoded dialog — you drive it live.

Commands (not sent to the model): `checkpoint`, `branch <name>`,
`switch <name>` (Branching only), `facts` (show the Sticky Facts block),
`status` (summary across all strategies), `exit`/`quit`.

# Day 9: Context management — history compression

`compression_scenarios.py` runs the same long dialog twice and compares:

- **Without compression**: the full history goes into every request.
- **With compression**: the full history is **kept on disk** (so the user can
  still browse it), but old messages are folded into a `summary` that is
  substituted into the request instead of the full history.

It prints how the context, per-session tokens, and cost differ, and then asks a
control question about a fact from the start of the dialog to compare answer
quality (did the agent retain the gist after compression?).

The run also shows a per-turn log: on each turn you see how many fresh (not yet
summarized) messages remain and how many are folded into the summary, with a
`[СЖАТИЕ]` marker on the exact turns where compression fires. The summary
calls themselves are counted separately (they are auxiliary LLM requests, so
they are not part of the dialog's token accounting).

## What was added

- `agent/compression.py` — `CompressionConfig`, `build_context_messages`
  (summary + fresh window), `build_summary_prompt` / `summarize_messages`
  (LLM call to compress), `compress` (fold fresh messages into summary, keep
  them on disk), and `compressed_context_tokens`.
- `agent/conversation.py` — a session stores `summary`, a `summarized` counter
  (how many leading messages are folded), and the full `messages` history. Old
  array format and old `{summary, messages}` format still load.
- `agent/agent.py` — the agent builds the request from `summary + fresh window`
  and triggers compression after each turn when the fresh window overgrows.
- `agent/cli.py` — reads the `compression` section, shows its status, and the
  `history` command prints the **full** history, marking messages that were
  folded into the summary.
- `agent/llm_client.py` — `complete()` accepts `max_tokens` (for the summary).

Config keys (in the `compression` section): `keep_recent` (how many fresh
messages stay uncompressed), `summarize_every` (compress when the fresh window
exceeds this), `max_tokens`.

## Run

```bash
python3 compression_scenarios.py          # offline: responses are simulated locally
python3 compression_scenarios.py --online # real API requests, real quality check
python3 compression_scenarios.py --turns 48   # longer dialog, bigger savings
```

In offline mode a control question checks that the fact survives in the summary
(a realistic summarizer is simulated). With `--online` quality is judged by the
real model's answer.

# Day 8: Token accounting

`token_scenarios.py` counts tokens for the current request, the whole dialog
history, and the model's answer, and shows how cost grows as the conversation
progresses and what breaks when the context exceeds the model limit.

## What was added

- `agent/tokens.py` — local token estimation (`estimate_tokens`), per-turn and
  cumulative usage/cost accounting, and `ContextOverflowError`.
- `agent/agent.py` — records tokens/cost on every turn and raises
  `ContextOverflowError` before sending if the dialog would exceed
  `max_context_tokens`.
- `agent/llm_client.py` — `Completion` now exposes `total_tokens`.
- `agent/cli.py` — prints per-turn and per-session token/cost stats.

Config keys: `max_context_tokens`, `input_cost_per_million`,
`output_cost_per_million`.

## Run

```bash
python3 token_scenarios.py          # offline: responses are simulated locally
python3 token_scenarios.py --online # real API requests
```

The script compares a short dialog, a long dialog, and a dialog that overflows
the context limit, printing how tokens and cost grow and what breaks on overflow.

# Day 5: Model versions

`model_comparison.py` sends one identical query to weak, medium, and strong models via Bothub. For every successful request it prints:

1. Full response time.
2. Prompt, completion, and total token counts returned by the API.
3. Request cost, calculated from the configured per-million-token tariffs.

## Setup

```bash
cp config.example.json config.json
```

Set `api_key` in `config.json`. In `models`, replace each `model-id-from-bothub` with an available model identifier from [Bothub Models](https://bothub.ru/models). Select one weak, one medium, and one strong model.

Set `input_cost_per_million` and `output_cost_per_million` to the dollar prices from the model page. Set both to `0` for a free model. Omit either tariff to leave the cost as `н/д`.

`config.json` is ignored by Git, so the API key will not be committed.

## Run

```bash
python3 model_comparison.py
```

The output groups each answer and its measurements by model:

```text
=== Слабая: model-id ===
...
Время ответа: 1.24 с
Токены: 88 всего (25 входных, 63 выходных)
Стоимость: $0.000123
```
