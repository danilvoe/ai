# AI Advent: Instructions and Architecture Map for Coding Agents

## 1. Project Context & Branch Architecture

This repository contains practical solutions for the **AI Advent (LLM & Agents)** educational curriculum, spanning **Day 1 through Day 25**.

- **Branch-per-Lesson Structure**: Each Git branch (`day2`, `day3`, ..., `day25`, `master`) represents the code state at that specific lesson.
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

## 4. Complete Lesson & File Navigation Matrix (Day 1 – Day 25)

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

---

## 6. Verification Cheat Sheet (Fast & Zero API Token Cost)

Run these exact commands to verify your changes without incurring API costs or waiting for network roundtrips:

```bash
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
