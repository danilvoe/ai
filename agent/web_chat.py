"""Day 27: интеграция локальной LLM в веб-приложение (базовый веб-чат).

Модуль поднимает Flask-приложение, которое отправляет запросы в **локальную**
модель из Дня 26 (``llama-server``, OpenAI-совместимый endpoint ``/v1``) и
отображает её ответы в браузере. Облачные модели и API-ключи не используются.

Компоненты:

* :class:`StubLLMClient` — детерминированный офлайн-клиент, совместимый с
  :class:`agent.agent.Agent` (нужен для проверок без сборки модели и сети);
* :class:`ChatSession` — сессия диалога (``Conversation`` + ``Agent``);
* :class:`WebChatBackend` — жизненный цикл локального сервера и реестр сессий;
* :func:`create_app` — Flask-приложение с JSON API и встроенным HTML-чатом.

Запуск (через точку входа)::

    python3 web_local.py            # живой локальный llama-server
    python3 web_local.py --offline  # веб-чат на детерминированном стабе
"""

from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request

from .agent import Agent
from .context import SlidingWindowStrategy
from .conversation import Conversation, SessionStore
from .llm_client import Completion
from .local_llm import LocalLLMConfig, LocalLLMServer, StubLocalLLM
from .tokens import ContextOverflowError

DEFAULT_WINDOW_SIZE = 8
DEFAULT_MAX_SESSIONS = 64


class StubLLMClient:
    """Офлайн-клиент локальной LLM, совместимый с :class:`agent.agent.Agent`.

    Повторяет интерфейс :class:`agent.llm_client.LLMClient` (``config`` dict и
    ``complete(messages, ...)``), но отвечает детерминированно через
    :class:`agent.local_llm.StubLocalLLM` — без бинарников, модели и сети.
    """

    def __init__(self, model_label: str = "local-llm-stub") -> None:
        self.config = {
            "model": model_label,
            "temperature": 0.7,
            "input_cost_per_million": 0,
            "output_cost_per_million": 0,
        }
        self._stub = StubLocalLLM(model_label)

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break
        content = self._stub.answer(prompt)
        return Completion(content=content, model=self.config["model"])


class ChatSession:
    """Одна сессия веб-чата: диалог с моделью и его история на диске."""

    def __init__(self, conversation: Conversation, agent: Agent) -> None:
        self._conversation = conversation
        self._agent = agent
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._conversation.session_id

    @property
    def messages(self) -> list[dict]:
        """История сообщений сессии (user/assistant)."""
        return self._conversation.messages

    def ask(self, message: str) -> Completion:
        """Отправляет сообщение в локальную модель и возвращает ответ."""
        with self._lock:
            return self._agent.ask(message)

    def reset(self) -> None:
        """Очищает историю диалога сессии."""
        with self._lock:
            self._conversation.clear()


class WebChatBackend:
    """Бэкенд веб-чата: локальный сервер + реестр сессий диалога.

    В живом режиме поднимает ``llama-server`` (День 26) и направляет на него
    агентов. В офлайн-режиме использует :class:`StubLLMClient`, поэтому
    приложение и его проверки работают без сборки и модели.
    """

    def __init__(
        self,
        config: dict,
        offline: bool = False,
        model_path: str | None = None,
        port: int | None = None,
        window_size: int | None = None,
        max_sessions: int | None = None,
        history_dir=None,
    ) -> None:
        self.config = config
        self.offline = offline
        self.local = LocalLLMConfig.from_config(config)
        if model_path:
            self.local.model_path = model_path
        if port:
            self.local.port = port
        section = (config or {}).get("web_chat", {}) or {}
        window = window_size if window_size is not None else section.get(
            "window_size", DEFAULT_WINDOW_SIZE
        )
        sessions = max_sessions if max_sessions is not None else section.get(
            "max_sessions", DEFAULT_MAX_SESSIONS
        )
        self._window_size = max(1, int(window))
        self._max_sessions = max(1, int(sessions))
        self._store = (
            SessionStore(Path(history_dir)) if history_dir else SessionStore()
        )
        self._server: LocalLLMServer | None = None
        self._client = None
        self._max_context_tokens: int | None = None
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        """Режим работы: ``offline`` (стаб) или ``live`` (llama-server)."""
        return "offline" if self.offline else "live"

    @property
    def ready(self) -> bool:
        """Готов ли бэкенд принимать запросы."""
        return self._client is not None

    @property
    def model_label(self) -> str:
        return self.local.model_label

    @property
    def endpoint(self) -> str:
        """Адрес модели, к которому обращается веб-чат."""
        if self.offline:
            return "stub://local-llm"
        return f"http://{self.local.host}:{self.local.port}/v1"

    def start(self) -> WebChatBackend:
        """Готовит модель: поднимает llama-server или включает офлайн-стаб."""
        if self.offline:
            self._client = StubLLMClient(f"{self.local.model_label}-stub")
            self._max_context_tokens = None
            return self
        self._server = LocalLLMServer(self.local)
        self._server.start()
        self._client = self._server.build_client()
        self._max_context_tokens = self._server.max_context_tokens
        return self

    def stop(self) -> None:
        """Останавливает локальный сервер, если он был запущен."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def status(self) -> dict:
        """Сводка состояния бэкенда для ``GET /api/status``."""
        return {
            "ready": self.ready,
            "mode": self.mode,
            "model": self.model_label,
            "endpoint": self.endpoint,
            "sessions": len(self._sessions),
        }

    def _build_session(self, session_id: str | None) -> ChatSession:
        conversation = (
            self._store.get(session_id) if session_id else self._store.create()
        )
        agent = Agent(
            self._client,
            conversation,
            max_context_tokens=self._max_context_tokens,
            context_strategy=SlidingWindowStrategy(size=self._window_size),
        )
        return ChatSession(conversation, agent)

    def _get_session(self, session_id: str | None) -> ChatSession:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            session = self._build_session(session_id)
            if (
                session.session_id not in self._sessions
                and len(self._sessions) >= self._max_sessions
            ):
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            self._sessions[session.session_id] = session
            return session

    def chat(self, message: str, session_id: str | None = None) -> dict:
        """Отправляет сообщение в локальную модель и возвращает ответ.

        Возвращает словарь ``{session_id, reply, model}``. Если ``session_id``
        не передан — создаётся новая сессия, её id возвращается клиенту.
        """
        if not self.ready:
            raise RuntimeError("Бэкенд не запущен: вызовите start() перед запросами.")
        session = self._get_session(session_id)
        reply = session.ask(message)
        return {
            "session_id": session.session_id,
            "reply": reply.content,
            "model": reply.model,
        }

    def reset(self, session_id: str | None) -> str:
        """Сбрасывает историю сессии и возвращает её id."""
        session = self._get_session(session_id)
        session.reset()
        return session.session_id

    def history(self, session_id: str | None) -> list[dict]:
        """История сообщений сессии."""
        return self._get_session(session_id).messages


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Локальный LLM-чат</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #0f1117; color: #e6e8ee;
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  header {
    padding: 12px 18px; border-bottom: 1px solid #232734;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .status { font-size: 12px; color: #8b93a7; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #f0a020; margin-right: 6px; vertical-align: middle; }
  .dot.ok { background: #38b26a; }
  main { flex: 1; overflow-y: auto; padding: 18px; }
  .msg { max-width: 760px; margin: 0 auto 14px; display: flex; }
  .msg .bubble {
    padding: 10px 14px; border-radius: 12px; white-space: pre-wrap;
    word-wrap: break-word; max-width: 85%;
  }
  .msg.user { justify-content: flex-end; }
  .msg.user .bubble { background: #2b5cff; color: #fff; border-bottom-right-radius: 3px; }
  .msg.assistant .bubble { background: #1b1f2a; border: 1px solid #232734;
                           border-bottom-left-radius: 3px; }
  .msg.system .bubble { background: #2a2010; border: 1px solid #4a3a12; color: #f0c060; }
  footer { border-top: 1px solid #232734; padding: 12px 18px; }
  form { max-width: 760px; margin: 0 auto; display: flex; gap: 10px; }
  textarea {
    flex: 1; resize: none; height: 46px; padding: 12px; border-radius: 10px;
    border: 1px solid #2c3140; background: #151924; color: inherit; font: inherit;
  }
  textarea:focus { outline: none; border-color: #2b5cff; }
  button {
    padding: 0 18px; border: 0; border-radius: 10px; cursor: pointer;
    background: #2b5cff; color: #fff; font: inherit; font-weight: 600;
  }
  button.ghost { background: #1b1f2a; border: 1px solid #2c3140; font-weight: 400; }
  button:disabled { opacity: .5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>Локальный LLM-чат</h1>
  <span class="status"><span id="dot" class="dot"></span><span id="status">подключение…</span></span>
  <span style="flex:1"></span>
  <button class="ghost" id="reset" type="button">Очистить</button>
</header>
<main id="log"></main>
<footer>
  <form id="form">
    <textarea id="input" placeholder="Спросите локальную модель…" autofocus></textarea>
    <button id="send" type="submit">Отправить</button>
  </form>
</footer>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const resetBtn = document.getElementById('reset');
  const statusEl = document.getElementById('status');
  const dot = document.getElementById('dot');
  let sessionId = null;

  function add(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  async function loadStatus() {
    try {
      const r = await fetch('api/status');
      const s = await r.json();
      statusEl.textContent = s.mode + ' · ' + s.model;
      dot.className = 'dot' + (s.ready ? ' ok' : '');
    } catch (e) {
      statusEl.textContent = 'нет связи';
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    add('user', text);
    input.value = '';
    send.disabled = true;
    const pending = add('assistant', '…');
    try {
      const r = await fetch('api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId, message: text}),
      });
      const data = await r.json();
      if (!r.ok) {
        pending.parentElement.className = 'msg system';
        pending.textContent = '⚠️ ' + (data.error || ('HTTP ' + r.status));
      } else {
        sessionId = data.session_id;
        pending.textContent = data.reply;
      }
    } catch (err) {
      pending.parentElement.className = 'msg system';
      pending.textContent = '⚠️ Ошибка соединения: ' + err;
    } finally {
      send.disabled = false;
      input.focus();
    }
  });

  resetBtn.addEventListener('click', async () => {
    await fetch('api/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId}),
    });
    log.innerHTML = '';
    sessionId = null;
    add('system', 'История очищена.');
  });

  loadStatus();
</script>
</body>
</html>
"""


def create_app(backend: WebChatBackend) -> Flask:
    """Создаёт Flask-приложение поверх готового :class:`WebChatBackend`."""
    app = Flask(__name__)
    # Кириллица в JSON-ответах должна быть читаемой, а не \uXXXX.
    app.json.ensure_ascii = False

    @app.get("/")
    def index() -> Response:
        return Response(HTML_PAGE, mimetype="text/html; charset=utf-8")

    @app.get("/api/status")
    def status():
        return jsonify(backend.status())

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Пустое сообщение"}), 400
        session_id = data.get("session_id")
        try:
            result = backend.chat(message, session_id)
        except ContextOverflowError:
            return (
                jsonify(
                    {
                        "error": "Контекст переполнен. Очистите историю.",
                        "session_id": session_id,
                    }
                ),
                409,
            )
        except Exception as exc:  # noqa: BLE001 — понятная ошибка локальной модели
            return jsonify({"error": f"Ошибка локальной модели: {exc}"}), 502
        return jsonify(result)

    @app.post("/api/reset")
    def reset():
        data = request.get_json(silent=True) or {}
        session_id = backend.reset(data.get("session_id"))
        return jsonify({"session_id": session_id, "ok": True})

    @app.get("/api/history")
    def history():
        session_id = request.args.get("session_id")
        return jsonify({"session_id": session_id, "messages": backend.history(session_id)})

    return app
