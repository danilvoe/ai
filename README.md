# Day 9: Context management — history compression

`compression_scenarios.py` runs the same long dialog twice and compares:

- **Without compression**: the full history goes into every request.
- **With compression**: the last `keep_recent` messages are kept as-is, and the
  rest is replaced by a `summary` that is substituted into the request instead of
  the full history.

It prints how the context, per-session tokens, and cost differ, and then asks a
control question about a fact from the start of the dialog to compare answer
quality (did the agent retain the gist after compression?).

## What was added

- `agent/compression.py` — `CompressionConfig`, `build_context_messages`
  (summary + recent window), `build_summary_prompt` / `summarize_messages`
  (LLM call to compress), `compress` (drop old messages, keep summary), and
  `compressed_context_tokens`.
- `agent/conversation.py` — a session now stores a separate `summary` field
  next to `messages` (old array format still loads).
- `agent/agent.py` — the agent builds the request from `summary + recent window`
  and triggers compression after each turn when the window overgrows.
- `agent/cli.py` — reads the `compression` section and shows its status.
- `agent/llm_client.py` — `complete()` accepts `max_tokens` (for the summary).

Config keys (in the `compression` section): `keep_recent` (last N messages kept
as-is), `summarize_every` (compress when the window exceeds this), `max_tokens`.

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
