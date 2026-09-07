#!/usr/bin/env python3
"""Compare an unrestricted model response with a controlled one."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
QUERY = "Объясни, почему регулярный сон важен для продуктивности."


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def request_completion(config: dict, messages: list[dict], **options: object) -> str:
    body = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("temperature", 0.7),
        **options,
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
    unrestricted_messages = [{"role": "user", "content": QUERY}]
    controlled_messages = [
        {
            "role": "system",
            "content": (
                "Ответь строго одним JSON-объектом формата "
                '`{"answer":"<одно предложение>","status":"complete"}`. '
                "Значение answer не должно превышать 25 слов. "
                "Заверши ответ полем status со значением complete и не добавляй текст после }."
            ),
        },
        {"role": "user", "content": QUERY},
    ]

    try:
        print("Sending request without constraints...", flush=True)
        unrestricted = request_completion(config, unrestricted_messages)
        print("Received unrestricted response.", flush=True)
        print("Sending request with response controls...", flush=True)
        controlled = request_completion(
            config,
            controlled_messages,
            max_tokens=150,
        )
        print("Received controlled response.", flush=True)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"API error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Query:", QUERY)
    print("\n=== Without constraints ===\n", unrestricted)
    print("\n=== With JSON format, length, and completion controls ===\n", controlled)


if __name__ == "__main__":
    main()
