#!/usr/bin/env python3
"""Compare responses to one query at different temperature values."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
QUERY = "Придумай короткий слоган для приложения, которое помогает планировать день."
TEMPERATURES = (0, 0.7, 1.2)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def request_completion(config: dict, temperature: float) -> str:
    body = {
        "model": config["model"],
        "messages": [{"role": "user", "content": QUERY}],
        "temperature": temperature,
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
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> None:
    config = load_config(CONFIG_PATH)
    responses: list[tuple[float, str]] = []

    try:
        for index, temperature in enumerate(TEMPERATURES, start=1):
            print(f"{index}/{len(TEMPERATURES)}: Запрос с temperature = {temperature}...", flush=True)
            responses.append((temperature, request_completion(config, temperature)))
    except (OSError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        print(f"API error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nЗапрос:", QUERY)
    for temperature, response in responses:
        print(f"\n=== temperature = {temperature} ===\n{response}")


if __name__ == "__main__":
    main()
