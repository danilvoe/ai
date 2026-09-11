"""Загрузка и доступ к конфигурации агента."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.one.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)
