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
