#!/usr/bin/env python3
"""Compare four prompting approaches for the same algorithmic task."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
TASK = (
    "Дан массив целых чисел: все числа встречаются ровно два раза, кроме одного. "
    "Найдите это число за O(n) времени и O(1) дополнительной памяти. "
    "Объясните алгоритм и приведите реализацию на Python."
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def request_completion(config: dict, messages: list[dict[str, str]]) -> str:
    body = {
        "model": config["model"],
        "messages": messages,
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
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def solve(config: dict, instruction: str = "") -> str:
    content = f"{instruction}\n\nЗадача:\n{TASK}".strip()
    return request_completion(config, [{"role": "user", "content": content}])


def main() -> None:
    config = load_config(CONFIG_PATH)
    try:
        print("1/5: Получаю прямой ответ...", flush=True)
        direct_answer = solve(config)

        print("2/5: Получаю пошаговое решение...", flush=True)
        step_by_step_answer = solve(config, "Решай пошагово.")

        print("3/5: Прошу составить промпт для решения...", flush=True)
        generated_prompt = request_completion(
            config,
            [
                {
                    "role": "user",
                    "content": (
                        "Составь один подробный промпт на русском для другой модели, "
                        "чтобы она качественно решила следующую задачу. Верни только текст промпта.\n\n"
                        f"Задача:\n{TASK}"
                    ),
                }
            ],
        )
        print("4/5: Решаю задачу с созданным промптом...", flush=True)
        prompt_engineered_answer = solve(config, generated_prompt)

        print("5/5: Получаю мнения группы экспертов...", flush=True)
        expert_answer = solve(
            config,
            (
                "Сымитируй группу экспертов. Аналитик объясняет идею и доказательство, "
                "инженер пишет надёжный код Python и оценивает сложность, критик проверяет "
                "допущения и крайние случаи. Дай отдельный ответ каждого под его заголовком, "
                "затем общий итог."
            ),
        )
    except (OSError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        print(f"API error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("\nЗадача:\n", TASK)
    print("\n=== 1. Прямой ответ ===\n", direct_answer)
    print("\n=== 2. Инструкция 'решай пошагово' ===\n", step_by_step_answer)
    print("\n=== 3. Сначала сгенерированный промпт ===\n", generated_prompt)
    print("\n=== 3. Ответ по сгенерированному промпту ===\n", prompt_engineered_answer)
    print("\n=== 4. Группа экспертов ===\n", expert_answer)


if __name__ == "__main__":
    main()
