"""Самодостаточная копия RAG-движка сервиса (без зависимостей на ``agent/``).

Модули скопированы из уроков Дней 21–28:

* :mod:`core.llm_client` — OpenAI-совместимый HTTP-клиент;
* :mod:`core.local_llm` — запуск ``llama-server`` + офлайн-стаб;
* :mod:`core.indexing` — чанкинг, эмбеддер и FAISS-индекс;
* :mod:`core.rag` — ретривер, промпты и контрольные вопросы;
* :mod:`core.refusal` — правило отказа/уточнения;
* :mod:`core.local_rag` — полностью локальный RAG-сеанс;
* :mod:`core.index_build` — сборка/пересоздание индекса.
"""

from __future__ import annotations

__all__ = [
    "index_build",
    "indexing",
    "llm_client",
    "local_llm",
    "local_rag",
    "rag",
    "refusal",
    "tokens",
]
