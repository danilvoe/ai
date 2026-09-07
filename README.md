# Day 4: Temperature

The script sends the same user query to a Chat Completions-compatible API three times. Only the `temperature` parameter changes:

1. `0`
2. `0.7`
3. `1.2`

## Setup

```bash
cp config.example.json config.json
```

Set `api_key`, `base_url`, and `model` in `config.json`. The configuration file is ignored by Git.

## Run

```bash
python3 temperature.py
```

The output groups each answer by its temperature value:

```text
=== temperature = 0 ===
...

=== temperature = 0.7 ===
...

=== temperature = 1.2 ===
...
```

`temperature.py` explicitly sets the parameter for every request, so the optional `temperature` setting in `config.json` does not affect the comparison.
