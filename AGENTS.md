# AI Advent: Instructions and Architecture Map for Coding Agents

## 1. Project Context & Branch Architecture

This repository contains practical solutions for the **AI Advent (LLM & Agents)** educational curriculum, spanning **Day 1 through Day 34**.

- **Branch-per-Lesson Structure**: Each Git branch (`day2`, `day3`, ..., `day27`, `master`) represents the code state at that specific lesson.
- **Cumulative Codebase**: The project evolves incrementally. For example, `day25` contains RAG chat with task state memory, while `day10` only contains early context management.
- **Active Context**: Always check the active branch via `git branch --show-current` before making assumptions about available modules.

---

## 2. STRICT TOKEN-SAVING DIRECTIVES (CRITICAL)

To prevent massive token consumption and context window exhaustion, any coding agent operating in this repository **MUST** adhere to these rules:

1. **NEVER scan, glob, find, or grep inside `history/`**:
   - `history/` contains generated runtime artifacts, benchmark dumps, dialogue session logs, raw chunk files (`chunks.json` ~7.2 MB), and binary FAISS indexes (`faiss.index` ~6.3 MB).
   - Scanning `history/` will load megabytes of unstructured text/JSON into the context window and exhaust token limits.
   - You may read an artifact inside `history/` **ONLY** if the user explicitly provides an exact path and asks to view that specific report or run.

2. **NEVER run unconstrained global searches**:
   - Avoid `find .`, `ls -R`, or unfiltered recursive `grep` across the whole repository root.
   - Search specifically inside `agent/` or `mcp/` using targeted file patterns (e.g. `agent/*.py`).

3. **NEVER load full dataset files into context**:
   - `recipt_sample.json` (~122 KB) and `recipt_all.json` (large dataset) should not be read in their entirety. Use targeted reads with line offsets/limits if structure inspection is needed.

4. **Always use Offline Verification (`--no-llm` / `--offline`)**:
   - All scenario runners and agents support deterministic offline/stub modes.
   - When verifying code changes, always run tests with `--no-llm` or `--offline` first. Do NOT make live LLM API calls unless explicitly requested by the user.

---

## 3. Directory Layout & Roles

| Directory / File | Role | Agent Action Rule |
| :--- | :--- | :--- |
| `agent/` | Core modular Python package (agents, memory, task FSM, RAG, MCP runtime, scenarios) | **Target for edits & inspection** |
| `mcp/` | Standalone FastMCP servers (GitFlic API, Pipeline, Reports, Cron runner) | **Target for MCP tool edits** |
| `./` (Root) | CLI entrypoints (`chat_rag.py`, `chat.py`), config templates, early scripts | **Read/edit only relevant entrypoints** |
| `history/` | Generated test runs, benchmark JSONs, FAISS index binaries, markdown reports | **STRICTLY IGNORE / DO NOT SCAN** |
| `config.json` / `config.one.json` | Active local configuration (API keys, models, endpoints) | **Do not expose secrets** |

---

## 4. Complete Lesson & File Navigation Matrix (Day 1 – Day 34)

Use this table to jump directly to the relevant file instead of traversing the filesystem:

| Day | Topic | Key Modules | Scenarios & Tests (Offline Mode) |
| :--- | :--- | :--- | :--- |
| **Day 2** | Generation parameters & response control | `response_control.py` | `python3 response_control.py` |
| **Day 3** | Reasoning modes & prompts | `reasoning_modes.py` | `python3 reasoning_modes.py` |
| **Day 4** | Temperature exploration & determinism | `temperature.py` | `python3 temperature.py` |
| **Day 5** | Multi-model comparison | `model_comparison.py` | `python3 model_comparison.py` |
| **Day 6** | Agent foundation & CLI | `agent/agent.py`, `agent/cli.py`, `agent/llm_client.py`, `agent/config.py` | `python3 chat.py` |
| **Day 7** | Persistent conversation sessions | `agent/conversation.py` | Tested via `agent/cli.py` |
| **Day 8** | Token counting & limits accounting | `agent/tokens.py`, `token_scenarios.py` | `python3 token_scenarios.py` |
| **Day 9** | Context compression (history summarization) | `agent/compression.py`, `compression_scenarios.py` | `python3 -m compression_scenarios` |
| **Day 10** | Context strategies (Sliding Window, Sticky Facts, Branching) | `agent/context.py`, `agent/context_scenarios.py` | `python3 -m agent.context_scenarios` |
| **Day 11** | 3-layer memory model (Short-term, Working, Long-term) | `agent/memory.py`, `agent/memory_scenarios.py` | `python3 -m agent.memory_scenarios` |
| **Day 12** | Personalization & user profile over memory | `agent/personalization.py`, `agent/personalization_scenarios.py` | `python3 -m agent.personalization_scenarios` |
| **Day 13** | Task State Machine (finite stages, steps, auto-advance) | `agent/task_state.py`, `agent/task_scenarios.py` | `python3 -m agent.task_scenarios` |
| **Day 14** | Invariant enforcement (architecture & business constraints) | `agent/invariants.py`, `agent/invariant_scenarios.py` | `python3 -m agent.invariant_scenarios` |
| **Day 15** | Controlled task transitions (prevent stage jumping) | `agent/transition_scenarios.py` | `python3 -m agent.transition_scenarios` |
| **Day 16** | MCP client & tool discovery via official SDK | `agent/mcp_client.py` | `python3 -m agent.mcp_client --check` |
| **Day 17** | MCP server: GitFlic public projects tool | `mcp/gitflic_server.py`, `mcp/gitflic_api.py`, `agent/mcp_tools.py`, `agent/tool_scenarios.py` | `python3 mcp/gitflic_server.py --check` |
| **Day 18** | Planner & background scheduled jobs (Cron) | `mcp/gitflic_scheduler.py`, `mcp/gitflic_cron.py`, `agent/scheduler_scenarios.py` | `python3 mcp/gitflic_cron.py --run-due` |
| **Day 19** | MCP tool composition (Search → Summarize → SaveToFile) | `mcp/pipeline_server.py`, `mcp/pipeline_text.py`, `agent/pipeline.py`, `agent/pipeline_scenarios.py` | `python3 -m agent.pipeline` |
| **Day 20** | Multi-server MCP orchestrator & long agent flow | `mcp/report_server.py`, `agent/orchestrator.py`, `agent/orchestration_scenarios.py` | `python3 -m agent.orchestration_scenarios --offline` |
| **Day 21** | Document indexing (fixed/structural chunking, FAISS, embeddings) | `agent/indexing.py`, `agent/indexing_scenarios.py` | `python3 -m agent.indexing_scenarios` |
| **Day 22** | First RAG query (RagRetriever, baseline vs RAG evaluation) | `agent/rag.py`, `agent/rag_scenarios.py` | `python3 -m agent.rag_scenarios --no-llm` |
| **Day 23** | Two-stage RAG (Query rewrite, reranking, relevance filter) | `agent/reranking.py`, `agent/reranking_scenarios.py` | `python3 -m agent.reranking_scenarios --no-llm` |
| **Day 24** | Grounded RAG (Citations, anti-hallucination threshold, quote alignment) | `agent/grounding.py`, `agent/grounding_scenarios.py` | `python3 -m agent.grounding_scenarios --no-llm` |
| **Day 25** | Mini-Chat CLI with RAG, Task State memory & verified sources | `agent/rag_chat.py`, `agent/rag_chat_cli.py`, `agent/rag_chat_scenarios.py`, `chat_rag.py` | `python3 -m agent.rag_chat_scenarios --no-llm` |
| **Day 26** | Local LLM via llama.cpp (build, CLI + OpenAI-compatible server) | `agent/local_llm.py`, `agent/local_llm_scenarios.py`, `chat_local.py`, `scripts/setup_llama_cpp.sh` | `python3 -m agent.local_llm_scenarios --offline` |
| **Day 27** | Web chat app (Flask) over the local LLM, with optional cloud model | `agent/web_chat.py`, `agent/web_chat_scenarios.py`, `web_local.py` | `python3 -m agent.web_chat_scenarios --offline` |
| **Day 28** | Fully local RAG: local retrieval (FAISS) + local LLM generation, with local-vs-cloud comparison | `agent/local_rag.py`, `agent/local_rag_scenarios.py`, `agent/web_chat.py` (per-message RAG + local/cloud model selectors in UI), `web_local.py` | `python3 -m agent.local_rag_scenarios --offline` |
| **Day 29** | Local LLM optimization for the recipe task: params (temperature/max tokens/context window), quantization (Q4_K_M vs weaker Q2_K), case-tuned prompt template; before/after quality/speed/resources comparison + web settings panel | `agent/optimization.py`, `agent/optimization_scenarios.py`, `agent/web_chat.py` (optimization panel + `/api/settings`), `agent/rag.py`, `agent/local_rag.py`, `agent/agent.py`, `scripts/setup_llama_cpp.sh` (`--quant all`) | `python3 -m agent.optimization_scenarios --offline` |
| **Day 30** | Final project: multi-user Recipe Chat Service (Flask + SQLite, auth, many chats/user, local RAG + history, self-contained `core/`, index copy/rebuild) | `recipe_chat_service/run.py`, `recipe_chat_service/rebuild_index.py`, `recipe_chat_service/core/` (`llm_client`, `local_llm`, `indexing`, `rag`, `refusal`, `local_rag`, `index_build`), `recipe_chat_service/app/` (`config`, `db`, `security`, `engine`, `chat_service`, `routes/`, templates/static), `recipe_chat_service/scenarios/service_scenarios.py` | `cd recipe_chat_service && python3 -m scenarios.service_scenarios --offline` |
| **Day 31** | Developer assistant: RAG over project docs (README + `docs/`) + MCP project context (git branch/files/commits/diff) + `/help` command; docs auto-generation | `docs/*.md` (architecture, data-schemas, api, development, generated modules), `scripts/generate_docs.py`, `agent/project_docs.py`, `mcp/project_server.py`, `agent/dev_assistant.py`, `agent/dev_assistant_cli.py`, `chat_dev.py`, `agent/dev_assistant_scenarios.py` | `python3 -m agent.dev_assistant_scenarios --offline` |
| **Day 32** | Automated PR code review: pipeline gets PR diff + changed files (MCP), uses RAG over docs **and code**, returns review (bugs / architecture issues / recommendations); GitHub Action on `pull_request` + PR comment | `agent/code_index.py`, `mcp/review_server.py`, `agent/code_review.py`, `agent/review_cli.py`, `review_pr.py`, `agent/review_scenarios.py`, `.github/workflows/code-review.yml`, `requirements.txt` | `python3 -m agent.review_scenarios --offline` |
| **Day 33** | User support assistant: FAQ knowledge base + RAG over it, user/ticket context from a JSON CRM via MCP, answers product questions with ticket data («Почему не работает авторизация?») | `mcp/support_server.py`, `mcp/support_data/` (`faq.json`, `users.json`, `tickets.json`), `agent/support_faq.py`, `agent/support_assistant.py`, `agent/support_cli.py`, `chat_support.py`, `agent/support_scenarios.py` | `python3 -m agent.support_scenarios --offline` |
| **Day 34** | File-operations assistant: goal-level tasks over project files via MCP — find usages, search/analyze across files, generate/update files (catalog/ADR/changelog), invariant checks, prepare diff; dry-run by default, sandbox for reproducible runs | `mcp/fs_server.py`, `agent/file_assistant.py`, `agent/file_cli.py`, `chat_files.py`, `agent/file_scenarios.py` | `python3 -m agent.file_scenarios --offline` |

---

## 5. Architectural Subsystems Quick Lookup

When working on a specific subsystem, target these exact modules:

### A. Context & Memory Subsystem
- **Token Accounting**: `agent/tokens.py`
- **History Summarization**: `agent/compression.py`
- **Context Strategies**: `agent/context.py` (SlidingWindow, StickyFacts, Branching)
- **Hierarchical Memory**: `agent/memory.py` (`MemoryStore`, `MemoryLayer`)
- **User Profiling**: `agent/personalization.py` (`UserProfile`, `PersonalizedAgent`)

### B. Task State & Guardrails Subsystem
- **Task State FSM**: `agent/task_state.py` (`TaskStage`, `TaskState`, `CompletionChecker`)
- **Invariants Enforcement**: `agent/invariants.py` (`InvariantValidator`, `ArchitecturalInvariants`)
- **Transition Guardrails**: `agent/transition_scenarios.py`

### C. MCP (Model Context Protocol) Subsystem
- **Client & Transport**: `agent/mcp_client.py` (Streamable HTTP), `agent/mcp_tools.py` (Stdio transport)
- **GitFlic Server & Scheduler**: `mcp/gitflic_server.py`, `mcp/gitflic_scheduler.py`, `mcp/gitflic_cron.py`
- **Pipeline Composition**: `mcp/pipeline_server.py`, `mcp/pipeline_text.py`, `agent/pipeline.py`
- **Multi-Server Orchestrator**: `mcp/report_server.py`, `agent/orchestrator.py` (`McpOrchestrator`, `RoutedTool`)

### D. RAG & Retrieval Subsystem
- **Chunking & Vector Indexing**: `agent/indexing.py` (`HashingEmbedder`, `FaissIndex`, `chunk_structural`, `chunk_fixed_size`)
- **Basic RAG**: `agent/rag.py` (`RagRetriever`, `RagAgent`, prompt synthesis)
- **Reranker & Relevance Filter**: `agent/reranking.py` (`HeuristicReranker`, `LLMReranker`, `RelevanceFilter`, `HeuristicQueryRewriter`)
- **Grounding & Guardrails**: `agent/grounding.py` (`GroundedRagAgent`, `check_relevance_guardrail`, `AlignmentEvaluation`)
- **RAG Chat CLI with Task State**: `agent/rag_chat.py` (`RagTaskState`, `RagChatAgent`, `RagChatSession`), `chat_rag.py`

### E. Local LLM Subsystem (llama.cpp)
- **Local runtime**: `agent/local_llm.py` (`LocalLLMConfig`, `LocalLLMServer`, `StubLocalLLM`, `build_local_client`, `run_cli_prompt`)
- **Scenario**: `agent/local_llm_scenarios.py` (offline stub + live llama-cli/llama-server checks)
- **Bootstrap**: `scripts/setup_llama_cpp.sh` (build llama.cpp + download GGUF into `history/`)
- **Chat entrypoint**: `chat_local.py`
- **Web chat (Day 27)**: `agent/web_chat.py` (`WebChatBackend`, `ChatChannel`, `ChatSession`, `StubLLMClient`, `create_app`), `web_local.py`; **local/cloud model selection** — `ChatSession` holds `local`/`cloud` `ChatChannel`-s sharing one `Conversation`; cloud channel is built by `_build_cloud_client` only when config has `api_key`+`model` (never in `--offline`); UI `<select id="source">` (локальная/облачная) next to `<select id="mode">`, `/api/chat` takes `model_source`, `GET /api/status` reports `cloud`/`cloud_model`/`sources`, `--no-cloud` disables it.
- **Local LLM optimization (Day 29)**: `agent/optimization.py` (`GenerationParams`, `ModelVariant`, `PromptTemplate` + `RECIPE_STRUCTURED_SYSTEM_PROMPT`, presets, `OptimizationStubClient`, `measure_peak_rss_mb`, `compare_optimizations`, `render_optimization_markdown`), `agent/optimization_scenarios.py` (offline/live before/after comparison + web-settings HTTP checks). Web settings panel in `agent/web_chat.py` (UI + `POST /api/settings` to restart `llama-server` with a new `-c`/variant); per-request `temperature`/`max_tokens`/`context_window` overrides on `Agent.ask` and `LocalRagSystem.ask_with_rag`; `build_rag_messages(..., system_prompt=...)`; `--quant q4_k_m|q2_k|all` in `scripts/setup_llama_cpp.sh`; `local_llm.presets`/`local_llm.variants` + `optimization` config section.
- **Fully local RAG (Day 28)**: `agent/local_rag.py` (`LocalRagStubClient`, `LocalRagSystem`, `build_local_rag_client`, `compare_local_vs_cloud`; relevance threshold 0.33 — low-score chunks dropped, refusal hides sources; `is_generic_query` softens threshold for "что можно приготовить?"; `select_history` window+budget). **Routing (Day 28+)**: browse/category queries — `detect_category` (word-prefix stem match over multi-valued `category` metadata) + `retrieve_by_category` (up to `local_rag.category_suggestions` distinct recipes ranked by cosine, category as filter), guarded by `_matches_specific_dish` so a named dish stays specific; «топ N»/lists → generic route. `strip_meta_leak` removes meta-leak sentences from **every** answer. **Follow-ups (Day 28+)**: `is_followup_query` + `_extract_ordinal` + `_heuristic_resolve_reference` resolve "подробней про 2 позицию"/"второй" against the previous turn's `meta.sources`; `resolve_followup_query` does an LLM query-rewrite with history (`FOLLOWUP_REWRITE_SYSTEM_PROMPT`) for pronouns/pointers with the heuristic fallback; resolved query is re-retrieved and answered on route `followup`; unresolvable references ask to clarify (`CLARIFY_REFERENCE_MESSAGE`) and generic padding no longer injects below-floor junk. `agent/local_rag_scenarios.py`, web RAG via a UI mode selector ("не использовать RAG" / "рецепты", default **рецепты** when the index is available) in `web_local.py`. **Dialogue history**: web chat shares one `Conversation` across both modes; RAG replies are stored with compact `meta` (mode + sources, stripped before the LLM call in `Agent.context_messages`); previous turns fed into the prompt (`agent/rag.py` `build_rag_messages(history=..., extra_instruction=...)`); UI keeps session+mode in `localStorage` and restores history via `GET /api/history` with per-answer badges (`Conversation.append(role, content, meta=...)`).

### F. Developer Assistant Subsystem (Day 31)
- **Docs indexing**: `agent/project_docs.py` (`ProjectDocument`, `discover_documents`, `chunk_markdown`, `build_documents_index`, `load_project_retriever`, `build_docs_context`); index at `history/index/project_docs/markdown/`, reuses `HashingEmbedder`/`FaissIndex`/`RagRetriever`; chunk text is prefixed with a `Документ: … — раздел: …` context line so headings are searchable.
- **Docs generation**: `scripts/generate_docs.py` (`ast`-based, no imports) → `docs/modules.md`; curated docs in `docs/` (`README`, `architecture`, `data-schemas`, `api`, `development`).
- **Project MCP server**: `mcp/project_server.py` (`get_current_branch` (required), `get_repo_status`, `list_project_files`, `get_recent_commits`, `get_diff`; local `git` via `subprocess` list args, `--check` mode, `log_level="WARNING"`).
- **Assistant**: `agent/dev_assistant.py` (`DevAssistant`, `DevAnswer`, `DevAssistantStubClient`, `build_dev_messages`, `render_project_context`, `build_project_runtime`, `build_dev_assistant`); light reranking via `SOURCE_PRIOR` + lexical overlap; `DEV_ASSISTANT_SYSTEM_PROMPT`.
- **CLI + `/help`**: `agent/dev_assistant_cli.py`, `chat_dev.py` (`/help`, `/branch`, `/files`, `/diff`, `/sources`, `/reindex`, `/exit`; `--no-llm`, `--question`, `--no-mcp`).
- **Scenario**: `agent/dev_assistant_scenarios.py` (`--offline`) — checks docs index, MCP branch/files/commits/diff, and 5 `/help` structure questions; report in `history/dev_assistant/`.
- **Config**: `dev_assistant` section (`index_dir`, `dataset`, `strategy`, `top_k`, `min_score`, `temperature`, `max_context_chars`, `chunk_size`, `overlap`, `report_dir`, `project_server`).

### G. Code Review Automation (Day 32)
- **Code indexing (RAG source "code")**: `agent/code_index.py` (`CodeSymbol`, `CodeModule`, `discover_code_files`, `parse_module_symbols` via `ast`, `chunk_code_module`, `build_code_index`, `load_code_retriever`, `build_code_context`); index at `history/index/project_code/symbols/`; one chunk per top-level class/function plus module header/constants chunks; metadata `module`/`symbol`/`kind`/`lineno`.
- **Review MCP server**: `mcp/review_server.py` (`get_pr_diff` (`base...head`), `list_changed_files` (`--numstat`+`--name-status`), `get_file_content`, `get_pr_metadata` (from `GITHUB_EVENT_PATH`); repo root via `REVIEW_REPO_ROOT`; `--check` mode, `log_level="WARNING"`).
- **Review assistant**: `agent/code_review.py` (`ReviewFinding`, `CodeReview` with `to_markdown()`, `CodeReviewAssistant`, `build_review_messages`, `CODE_REVIEW_SYSTEM_PROMPT`, `analyze_diff`, `ReviewStubClient`, `parse_findings`, `build_review_config`/`has_llm_credentials`/`build_review_runtime`/`build_review_assistant`); categories `bug`/`architecture`/`recommendation`, severities `critical`/`major`/`minor`/`info`; retrieves docs (`project_docs`) + code (`code_index`) and reranks (docs reuse `SOURCE_PRIOR`).
- **Offline heuristics** (`analyze_diff`, only added diff lines): bare `except`, `eval`/`exec`, `shell=True`, `os.system`, hardcoded secrets, mutable defaults; architecture: LLM SDK / `faiss` / `subprocess` outside their layers, `recipe_chat_service` imported from `agent/`, `print` in library code, missing return annotations; recommendations: TODO/FIXME, docs+reindex reminder.
- **CLI**: `agent/review_cli.py`, `review_pr.py` (`--base`/`--head`/`--event`, `--offline`, `--output`/`--json`, `--post-comment` (GitHub REST via `urllib`), `--fail-on {none,critical,major}`, `--no-mcp`); auto-selects live LLM when `OPENAI_API_KEY`/`LLM_API_KEY`+`LLM_BASE_URL`+`LLM_MODEL` present, else offline heuristics; artifacts in `history/code_review/`.
- **Pipeline**: `.github/workflows/code-review.yml` — `on: pull_request` (opened/synchronize/reopened), `fetch-depth: 0`, builds both indexes, runs `review_pr.py --event "$GITHUB_EVENT_PATH" --post-comment`, uploads artifacts; `requirements.txt` added for CI.
- **Scenario**: `agent/review_scenarios.py` (`--offline`) — index check, MCP diff/changed-files check, heuristic fixture (all 3 categories), full e2e on a temp repo with a planted patch; report in `history/code_review/`.
- **Config**: `code_review` section (`index_dir`, `docs_dataset`, `code_dataset`, `strategy`, `code_strategy`, `top_k`, `min_score`, `temperature`, `max_context_chars`, `max_diff_chars`, `fail_on`, `post_comment`, `report_dir`, `review_server`).

### H. User Support Assistant (Day 33)
- **Data / FAQ KB**: `mcp/support_data/faq.json` (26 Q&A cards for Recipe Chat Service, fields `id/question/answer/category/product/tags/related`), `mcp/support_data/users.json`, `mcp/support_data/tickets.json` (synthetic CRM; anchored ticket `T-1024` about login).
- **Support MCP (CRM)**: `mcp/support_server.py` (`FastMCP` "support", local JSON, data dir via `SUPPORT_DATA_DIR`); tools `get_user`, `find_user`, `get_ticket`, `list_tickets`, `search_tickets` (`scope="all"|"meta"`), `get_ticket_context`; `--check` mode.
- **FAQ RAG**: `agent/support_faq.py` (`SupportFaq`, `load_faq`, `chunk_faq`, `build_faq_index`, `load_faq_retriever`, `build_faq_context`); index at `history/index/support_faq/faq/`; metadata `faq_id`/`category`/`tags`; `CONTROL_QUESTIONS` (16 verification questions incl. negative control).
- **Assistant**: `agent/support_assistant.py` (`SupportAnswer`, `SupportAssistant.ask(question, ticket_id, user_id)`, `resolve_support_ticket`, `build_support_messages`, `SUPPORT_SYSTEM_PROMPT`, `SupportStubClient`, `parse_answer_meta`, `build_support_runtime`, `build_support_assistant`); relevance guardrail (`relevance_threshold=0.30`) — weak FAQ matches → escalate; ticket auto-resolution: explicit id / `T-\d+` regex / user's tickets / `search_tickets` (meta) with confidence rule; answer JSON block `answer_kind/ticket_id/escalate/steps`.
- **CLI + entrypoint**: `agent/support_cli.py`, `chat_support.py` (`/ticket`, `/user`, `/tickets`, `/sources`, `/new`, `/reindex`, `/help`; `--offline`, `--ticket`, `--user`, `--question`, `--no-mcp`).
- **Scenario**: `agent/support_scenarios.py` (`--offline`) — FAQ recall (16/16), MCP user/ticket/list/search/context checks, anchor case (ticket status/version in answer), negative control; report in `history/support/`.
- **Config**: `support` section (`faq_path`, `index_dir`, `dataset`, `strategy`, `top_k`, `min_score`, `relevance_threshold`, `temperature`, `max_context_chars`, `chunk_size`, `overlap`, `report_dir`, `support_server`).

### I. File-Operations Assistant (Day 34)
- **FS MCP**: `mcp/fs_server.py` (`FastMCP` "filesystem", local, path-jail via `FS_ROOT`, ignores `.git`/`history`/`__pycache__` on traversal); tools `list_files`, `read_file` (`start_line`/`end_line`), `search_files` (multi-file, grouped `{file, count, matches}`), `analyze_file` (`ast` for `.py`, headings for `.md`), `file_info`, `write_file` (`overwrite`/`create_dirs`), `edit_file` (`old`→`new`, `count`), `make_diff` (git or `baseline_root` via `difflib`); `--check` mode.
- **Assistant**: `agent/file_assistant.py` (`classify_goal` → `find_usages`/`generate_file`/`update_docs`/`check_invariants`/`prepare_diff`; `FileAction`, `FileFinding`, `FileTaskResult`; `FileAssistant.run(goal, apply=...)`; `FileAssistantStubClient`; `render_document`; `_python_invariants` via `ast`; `build_fs_runtime`, `build_file_assistant`). The assistant itself chooses files: `search_files`+`read_file` for usages, `list_files`+`analyze_file`+`read_file` for generation, `list_files`+`read_file`+`analyze_file` for invariants.
- **Safety**: writes are dry-run by default (only unified diff); real writes only when `apply=True`/`--apply`. `_stage_write` reads the original first and always emits a diff artifact.
- **CLI + entrypoint**: `agent/file_cli.py`, `chat_files.py` (`/find`, `/search`, `/read`, `/analyze`, `/files`, `/generate`, `/update`, `/check`, `/diff`, `/apply on|off`, `/help`; `--offline`, `--goal`, `--workspace`, `--apply`, `--no-mcp`).
- **Scenario**: `agent/file_scenarios.py` (`--offline`) — 4 sandboxed scenarios (find usages over ≥2–3 files; generate catalog from 7 `mcp/*_server.py` + dry-run diff + apply-save + reproducibility check; invariant detection on a planted sandbox; update docs from code change with 2-file baseline diff); each scenario reports `files_count`/`change_mode`/`reproducible`, and a **minimal-result** block verifies ≥2 files, saved-or-diff, and reproducibility; sandbox `history/day34/`, report in `history/file_assistant/`.
- **Config**: `file_assistant` section (`enabled`, `workspace_root`, `apply`, `max_chars`, `report_dir`, `fs_server{enabled, command, args, timeout}`).

---

## 6. Verification Cheat Sheet (Fast & Zero API Token Cost)

Run these exact commands to verify your changes without incurring API costs or waiting for network roundtrips:

```bash
# Day 34 File-operations assistant (MCP read/search/analyze/write) verification - offline stub, no model/network
python3 mcp/fs_server.py --check --files --search "build_support_runtime" --path agent
python3 -m agent.file_scenarios --offline
python3 chat_files.py --offline --goal "найди, где используется McpToolRuntime"

# Day 33 User support assistant (FAQ RAG + MCP ticket context) verification - offline stub, no model/network
python3 mcp/support_server.py --check --ticket T-1024 --user 1 --search "авторизация"
python3 -m agent.support_faq --build
python3 -m agent.support_scenarios --offline
python3 chat_support.py --offline --question "Почему не работает авторизация?" --ticket T-1024

# Day 32 Automated PR code review (docs+code RAG, GitHub Action) verification - offline heuristics, no model/network
python3 -m agent.project_docs --build
python3 -m agent.code_index --build
python3 mcp/review_server.py --check --diff --files
python3 -m agent.review_scenarios --offline
python3 review_pr.py --base HEAD~1 --head HEAD --offline

# Day 31 Developer assistant (docs RAG + project MCP) verification - offline stub, no model/network
python3 -m agent.project_docs --build
python3 mcp/project_server.py --check
python3 -m agent.dev_assistant_scenarios --offline

# Day 29 Local LLM optimization verification - offline stub, no model/network
python3 -m agent.optimization_scenarios --offline

# Day 30 Recipe Chat Service verification - offline stub e2e, no model/network
cd recipe_chat_service && python3 rebuild_index.py --copy && python3 -m scenarios.service_scenarios --offline

# Day 26 Local LLM (llama.cpp) verification - offline stub, no build/network
python3 -m agent.local_llm_scenarios --offline

# Day 27 Web chat with local LLM verification - offline stub, no model/network
python3 -m agent.web_chat_scenarios --offline

# Day 28 Fully local RAG (local retrieval + local LLM) verification - offline stub
python3 -m agent.local_rag_scenarios --offline

# Day 25 RAG Chat verification
python3 -m agent.rag_chat_scenarios --no-llm

# Day 24 Grounded RAG verification
python3 -m agent.grounding_scenarios --no-llm

# Day 23 Two-Stage Reranking verification
python3 -m agent.reranking_scenarios --no-llm

# Day 22 Basic RAG verification
python3 -m agent.rag_scenarios --no-llm

# Day 21 FAISS Indexing verification
python3 -m agent.indexing_scenarios

# Day 20 MCP Orchestrator offline test
python3 -m agent.orchestrator
python3 -m agent.orchestration_scenarios --offline

# Day 19 Pipeline composition test
python3 -m agent.pipeline

# Day 13 Task State FSM test
python3 -m agent.task_scenarios

# Day 11 Memory model test
python3 -m agent.memory_scenarios

# Linting check
ruff check agent/ mcp/
```
