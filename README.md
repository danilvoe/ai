# Day 2: Control the API Response

The script sends the same user query to a Chat Completions-compatible API twice and prints both responses:

1. Without constraints.
2. With response controls: a strict JSON format, a 25-word instruction, `max_tokens`, and an explicit completion instruction.

## Setup

```bash
cp config.example.json config.json
```

Set `api_key`, `base_url`, and `model` in `config.json`. The configuration file is ignored by Git.

## Run

```bash
python3 response_control.py
```

Example output sections make the comparison suitable for a screen recording:

```text
=== Without constraints ===
...

=== With JSON format, length, and completion controls ===
{"answer":"...","status":"complete"}
```

## Controls used

The controlled request keeps the user query unchanged and adds a system instruction:

```text
Ответь строго одним JSON-объектом формата
`{"answer":"<одно предложение>","status":"complete"}`.
Значение answer не должно превышать 25 слов. Заверши ответ полем status
со значением complete и не добавляй текст после }.
```

It also sends `max_tokens: 150`. The final `status: "complete"` field and the instruction to add nothing after `}` are the completion condition. The larger technical limit leaves room for models that use internal reasoning; the 25-word instruction controls the visible answer length.
