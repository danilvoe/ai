#!/usr/bin/env python3
"""Compare latency, token usage, and price for several chat models."""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
QUERY = "Объясни простыми словами, почему небо голубое, не более чем в трёх предложениях."


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def request_completion(config: dict, model: str) -> tuple[str, dict, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": QUERY}],
        "temperature": config.get("temperature", 0.7),
    }
    request = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
            "Authorization": f"Bearer {config['api_key']}",
        },
        method="POST",
    )
    started_at = time.perf_counter()
    with urllib.request.urlopen(request, timeout=config.get("timeout", 120)) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"], data.get("usage", {}), time.perf_counter() - started_at


def calculate_cost(model: dict, usage: dict) -> float | None:
    input_price = model.get("input_cost_per_million")
    output_price = model.get("output_cost_per_million")
    if input_price is None or output_price is None:
        return None
    return (
        usage.get("prompt_tokens", 0) * input_price
        + usage.get("completion_tokens", 0) * output_price
    ) / 1_000_000


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
        models = config["models"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Config error: {exc}. Add the 'models' list from config.example.json.", file=sys.stderr)
        raise SystemExit(1) from exc

    results = []
    for index, model_config in enumerate(models, start=1):
        label = model_config["label"]
        model = model_config["model"]
        try:
            print(f"{index}/{len(models)}: Запрос к {label.lower()} модели ({model})...", flush=True)
            answer, usage, elapsed = request_completion(config, model)
            results.append((label, model, answer, usage, elapsed, calculate_cost(model_config, usage), None))
        except (KeyError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            results.append((label, model, "", {}, 0, None, str(exc)))

    print("\nЗапрос:", QUERY)
    for label, model, answer, usage, elapsed, cost, error in results:
        print(f"\n=== {label}: {model} ===")
        if error:
            print("Ошибка:", error)
            continue
        print(answer)
        print(f"Время ответа: {elapsed:.2f} с")
        print(f"Токены: {usage.get('total_tokens', 'н/д')} всего "
              f"({usage.get('prompt_tokens', 'н/д')} входных, "
              f"{usage.get('completion_tokens', 'н/д')} выходных)")
        print("Стоимость: н/д (укажите тарифы в config.json)" if cost is None else f"Стоимость: ${cost:.6f}")


if __name__ == "__main__":
    main()
