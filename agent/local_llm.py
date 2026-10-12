"""Day 26: локальный запуск LLM через llama.cpp.

Модуль инкапсулирует запуск и проверку локальной модели, собранной llama.cpp:

* поиск бинарников ``llama-cli`` и ``llama-server`` (сборка, PATH или конфиг);
* запуск ``llama-server`` с OpenAI-совместимым endpoint ``/v1``;
* проверка готовности через ``/health`` и ``/v1/models``;
* одиночный запрос через ``llama-cli`` (CLI-проверка);
* drop-in конфигурация для существующего :class:`agent.llm_client.LLMClient`.

Плюс детерминированный офлайн-стаб (:class:`StubLocalLLM`) для проверок без
сборки, модели и сети.

Запуск полного сценария::

    python3 -m agent.local_llm_scenarios --offline
    python3 -m agent.local_llm_scenarios
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .llm_client import Completion, LLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LLAMA_DIR = "history/llama.cpp"
DEFAULT_MODEL_PATH = "history/models/qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX_SIZE = 8192
DEFAULT_MAX_RESPONSE_TOKENS = 1024
DEFAULT_STARTUP_TIMEOUT = 120.0


def _default_threads() -> int:
    """Разумное число потоков для CPU-инференса."""
    return max(1, (os.cpu_count() or 4) // 2)


def _resolve(path: str) -> Path:
    """Разрешает путь относительно корня репозитория, если он не абсолютный."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


class LocalLLMError(RuntimeError):
    """Ошибка локального запуска LLM (нет бинарника, модели, сервер не поднялся)."""


@dataclass
class LocalLLMConfig:
    """Настройки локальной LLM из секции ``local_llm`` конфигурации."""

    enabled: bool = True
    llama_dir: str = DEFAULT_LLAMA_DIR
    model_path: str = DEFAULT_MODEL_PATH
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    ctx_size: int = DEFAULT_CTX_SIZE
    threads: int = _default_threads()
    n_gpu_layers: int = 0
    max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS
    server_bin: str = ""
    cli_bin: str = ""
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    model_label: str = "local-llm"

    @classmethod
    def from_config(cls, config: dict | None) -> LocalLLMConfig:
        section = (config or {}).get("local_llm", {}) or {}
        return cls(
            enabled=bool(section.get("enabled", True)),
            llama_dir=section.get("llama_dir", DEFAULT_LLAMA_DIR),
            model_path=section.get("model_path", DEFAULT_MODEL_PATH),
            host=section.get("host", DEFAULT_HOST),
            port=int(section.get("port", DEFAULT_PORT)),
            ctx_size=int(section.get("ctx_size", DEFAULT_CTX_SIZE)),
            threads=int(section.get("threads", _default_threads())),
            n_gpu_layers=int(section.get("n_gpu_layers", 0)),
            max_response_tokens=int(
                section.get("max_response_tokens", DEFAULT_MAX_RESPONSE_TOKENS)
            ),
            server_bin=section.get("server_bin", ""),
            cli_bin=section.get("cli_bin", ""),
            startup_timeout=float(section.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)),
            model_label=section.get("model_label", "local-llm"),
        )

    @property
    def llama_dir_path(self) -> Path:
        return _resolve(self.llama_dir)

    @property
    def model_file(self) -> Path:
        return _resolve(self.model_path)

    def binary(self, name: str, explicit: str = "") -> Path | None:
        """Ищет исполняемый бинарник: явный путь → сборка llama.cpp → PATH."""
        if explicit:
            path = _resolve(explicit)
            return path if path.exists() and os.access(path, os.X_OK) else None
        for candidate in (
            self.llama_dir_path / "build" / "bin" / name,
            self.llama_dir_path / "bin" / name,
        ):
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
        found = shutil.which(name)
        return Path(found) if found else None

    def server_binary(self) -> Path | None:
        return self.binary("llama-server", self.server_bin)

    def cli_binary(self) -> Path | None:
        return self.binary("llama-cli", self.cli_bin)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_cli_output(raw: str, prompt: str) -> str:
    """Убирает из вывода ``llama-cli`` баннер, эхо запроса и служебные строки."""
    text = _ANSI_RE.sub("", raw).replace("\r", "")
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == prompt or stripped == f"> {prompt}":
            start = index + 1
            break
    cleaned: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped == "Exiting...":
            continue
        if stripped.startswith("[") and stripped.endswith("]") and "t/s" in stripped:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


class LocalLLMServer:
    """Обёртка над процессом ``llama-server`` (OpenAI-совместимый endpoint)."""

    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self._log_path = REPO_ROOT / "history" / "local_llm" / "server.log"
        self.last_command: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/v1"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def build_command(self) -> list[str]:
        binary = self.config.server_binary()
        if binary is None:
            raise LocalLLMError(
                "llama-server не найден. Соберите llama.cpp: "
                "bash scripts/setup_llama_cpp.sh"
            )
        if not self.config.model_file.exists():
            raise LocalLLMError(
                f"Файл модели не найден: {self.config.model_file}. "
                "Скачайте модель: bash scripts/setup_llama_cpp.sh"
            )
        return [
            str(binary),
            "-m", str(self.config.model_file),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "-c", str(self.config.ctx_size),
            "-t", str(self.config.threads),
            "-ngl", str(self.config.n_gpu_layers),
            "-a", self.config.model_label,
            "--no-webui",
        ]

    def start(self, wait: bool = True) -> LocalLLMServer:
        """Запускает сервер и (по умолчанию) ждёт готовности."""
        if not self.config.enabled:
            raise LocalLLMError(
                "Локальная LLM отключена (local_llm.enabled = false в конфиге)."
            )
        if self.is_running:
            return self
        command = self.build_command()
        self.last_command = command
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self._log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        if wait:
            self.wait_ready()
        return self

    def wait_ready(self, timeout: float | None = None) -> None:
        """Опрашивает ``/health`` до готовности или таймаута."""
        timeout = timeout if timeout is not None else self.config.startup_timeout
        deadline = time.monotonic() + timeout
        url = f"http://{self.config.host}:{self.config.port}/health"
        last_error = "нет ответа"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise LocalLLMError(
                    f"llama-server завершился с кодом {self._process.returncode}. "
                    f"Смотрите лог: {self._log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    if response.status == 200 and body.get("status") in ("ok", "ready"):
                        return
                    last_error = f"status={body.get('status')}"
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self.stop()
        raise LocalLLMError(
            f"llama-server не поднялся за {timeout:.0f} с ({last_error}). "
            f"Смотрите лог: {self._log_path}"
        )

    def stop(self) -> None:
        """Останавливает сервер, при необходимости эскалируя до kill."""
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._process = None

    def __enter__(self) -> LocalLLMServer:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def client_config(self) -> dict:
        """Конфигурация для :class:`agent.llm_client.LLMClient` (drop-in)."""
        return {
            "base_url": self.base_url,
            "api_key": "local",
            "model": self.config.model_label,
            "temperature": 0.7,
            "timeout": 120,
            "max_retries": 0,
            "max_tokens": self.config.max_response_tokens,
        }

    @property
    def max_context_tokens(self) -> int:
        """Безопасный лимит контекста клиента (ctx_size минус запас на ответ)."""
        return max(256, self.config.ctx_size - self.config.max_response_tokens - 256)

    def build_client(self) -> LLMClient:
        """LLM-клиент, направленный на локальный llama-server."""
        return LLMClient(self.client_config())

    def list_models(self) -> list[str]:
        """Возвращает имена моделей из ``/v1/models``."""
        url = f"{self.base_url}/models"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [item.get("id", "") for item in data.get("data", [])]

    def complete(self, prompt: str, max_tokens: int = 128) -> Completion:
        """Одиночный запрос к локальному серверу через OpenAI-совместимый API."""
        client = self.build_client()
        return client.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

    def run_cli_prompt(self, prompt: str, max_tokens: int = 64, timeout: float = 180.0) -> str:
        """Разовый запрос через ``llama-cli`` (проверка «обращения через CLI»)."""
        binary = self.config.cli_binary()
        if binary is None:
            raise LocalLLMError(
                "llama-cli не найден. Соберите llama.cpp: "
                "bash scripts/setup_llama_cpp.sh"
            )
        if not self.config.model_file.exists():
            raise LocalLLMError(f"Файл модели не найден: {self.config.model_file}")
        command = [
            str(binary),
            "-m", str(self.config.model_file),
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", str(self.config.threads),
            "-ngl", str(self.config.n_gpu_layers),
            "-st",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-disable",
            "--simple-io",
        ]
        self.last_command = command
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalLLMError(f"llama-cli не ответил за {timeout:.0f} с") from exc
        if proc.returncode != 0:
            raise LocalLLMError(
                f"llama-cli завершился с кодом {proc.returncode}: "
                f"{proc.stderr.strip()[-500:]}"
            )
        return _clean_cli_output(proc.stdout, prompt)


def build_local_client(config: dict) -> tuple[LocalLLMServer, LLMClient]:
    """Поднимает локальный сервер и возвращает (сервер, клиент) для агентов."""
    local_config = LocalLLMConfig.from_config(config)
    server = LocalLLMServer(local_config).start()
    return server, server.build_client()


class StubLocalLLM:
    """Детерминированный офлайн-стаб локальной LLM (без бинарников и модели)."""

    def __init__(self, model_label: str = "local-llm-stub") -> None:
        self.config = LocalLLMConfig(model_label=model_label)

    @property
    def base_url(self) -> str:
        return "stub://local-llm"

    def answer(self, prompt: str) -> str:
        text = prompt.lower()
        if "2+2" in text.replace(" ", "") or "2 + 2" in text:
            return "4"
        if "столиц" in text and "франц" in text:
            return "Столица Франции — Париж."
        if "привет" in text or "hello" in text:
            return "Привет! Я локальная модель, работающая через llama.cpp."
        return f"Локальный ответ на запрос: {prompt.strip()[:120]}"

    def complete(self, prompt: str, max_tokens: int = 128) -> Completion:
        content = self.answer(prompt)
        return Completion(content=content, model=self.config.model_label)

    def client_config(self) -> dict:
        return {
            "base_url": self.base_url,
            "api_key": "local",
            "model": self.config.model_label,
            "temperature": 0.7,
        }
