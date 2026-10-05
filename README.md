# Production-like Mini-Chat CLI with RAG & Task Memory (Day 24+)

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
