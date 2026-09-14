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
python3 context_scenarios.py          # offline: responses are simulated locally
python3 context_scenarios.py --online # real API requests, real quality check
python3 context_scenarios.py --turns 32 --window 6   # longer dialog
```

The offline run compares token usage, final context size, and a control question
about an early fact (does it survive under each strategy?): Sliding Window drops
it, Sticky Facts keeps it in the `facts` block, Branching keeps it in the shared
checkpoint. It then demonstrates branching live: save a checkpoint, fork two
branches, continue each independently, and switch between them.

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
