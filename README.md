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
