#!/usr/bin/env python3
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chat_completion(config: dict, messages: list[dict]) -> str:
    url = config["base_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("temperature", 0.7),
    }
    request = urllib.request.Request(
        url,
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
    messages: list[dict[str, str]] = []
    print("Чат запущен. Введите 'exit' или Ctrl+C для выхода.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("До свидания!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            reply = chat_completion(config, messages)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"\nАссистент: {reply}\n")


if __name__ == "__main__":
    main()