"""День 33: RAG по FAQ ассистента поддержки.

Модуль готовит базу знаний поддержки: читает ``faq.json`` (вопросы и ответы о
продукте), превращает каждую карточку в чанк с метаданными (вопрос, категория,
теги) и строит локальный векторный индекс FAISS тем же офлайн-эмбеддером
(``HashingEmbedder``), что и в Днях 21–22.

Компоненты:

* :class:`SupportFaq` — одна карточка FAQ;
* :func:`load_faq` — загрузка ``faq.json``;
* :func:`chunk_faq` — чанкинг карточки (с заголовком «FAQ: … — раздел: …»);
* :func:`build_faq_index` — сборка и сохранение индекса;
* :func:`load_faq_retriever` — загрузка индекса как :class:`RagRetriever`;
* :func:`build_faq_context` — контекст с указанием вопроса и раздела;
* :data:`CONTROL_QUESTIONS` — перечень вопросов для проверки.

Запуск::

    python3 -m agent.support_faq --build
    python3 -m agent.support_faq --query "почему не работает авторизация" --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .indexing import Chunk, FaissIndex, HashingEmbedder, estimate_tokens
from .rag import RagRetriever, RagSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FAQ_PATH = "mcp/support_data/faq.json"
DEFAULT_INDEX_DIR = "history/index"
DEFAULT_DATASET = "support_faq"
DEFAULT_STRATEGY = "faq"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


@dataclass
class SupportFaq:
    """Одна карточка FAQ, попадающая в RAG-индекс."""

    id: str
    question: str
    answer: str
    category: str = "Прочее"
    product: str = ""
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    url: str = DEFAULT_FAQ_PATH


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    return target


def load_faq(path: str | Path = DEFAULT_FAQ_PATH, root: Path = PROJECT_ROOT) -> list[SupportFaq]:
    """Загружает карточки FAQ из JSON-файла."""
    target = _resolve(path, root)
    if not target.exists():
        raise FileNotFoundError(f"Файл FAQ не найден: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать FAQ {target}: {exc}") from exc

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Ожидался список карточек в поле 'items': {target}")

    product = str(data.get("product", "")) if isinstance(data, dict) else ""
    relative = target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)
    faq: list[SupportFaq] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        faq.append(
            SupportFaq(
                id=str(item.get("id") or f"faq-{index:03d}"),
                question=question,
                answer=answer,
                category=str(item.get("category") or "Прочее"),
                product=str(item.get("product") or product),
                tags=[str(tag) for tag in item.get("tags") or []],
                related=[str(rel) for rel in item.get("related") or []],
                url=relative,
            )
        )
    return faq


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет длинный текст на окна с перекрытием по границам строк/слов."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть строго меньше chunk_size")

    step = chunk_size - overlap
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            cut = window.rfind("\n")
            if cut < step // 2:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(start + 1, end - overlap)
    return pieces


def chunk_faq(
    item: SupportFaq,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Превращает карточку FAQ в один или несколько чанков.

    Заголовок («FAQ: <вопрос> — раздел: <категория>») включается в текст, чтобы
    эмбеддер учитывал вопрос и категорию при поиске. Слишком длинный ответ режется
    на окна с перекрытием.
    """
    header = f"FAQ: {item.question} — раздел: {item.category}"
    body = f"Вопрос: {item.question}\nОтвет: {item.answer}"
    pieces = _split_long_text(body, chunk_size, overlap) if len(header) + len(body) > chunk_size else [body]

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        chunk_text = f"{header}\n{piece}"
        metadata: dict[str, Any] = {
            "strategy": DEFAULT_STRATEGY,
            "doc_type": "faq",
            "faq_id": item.id,
            # recipe_id/url/title/section — совместимость с RagSource.
            "recipe_id": item.id,
            "title": item.question,
            "section": item.category,
            "url": item.url,
            "source": item.url,
            "category": item.category,
            "tags": list(item.tags),
            "related": list(item.related),
            "product": item.product,
            "step_number": None,
            "chunk_index": index,
            "n_chars": len(chunk_text),
            "n_tokens": estimate_tokens(chunk_text),
        }
        chunks.append(Chunk(id=f"faq_{item.id}_{index:02d}", text=chunk_text, metadata=metadata))
    return chunks


def build_faq_index(
    root: Path = PROJECT_ROOT,
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: HashingEmbedder | None = None,
    items: list[SupportFaq] | None = None,
    save: bool = True,
) -> tuple[FaissIndex, list[SupportFaq]]:
    """Собирает (и при ``save=True`` сохраняет) индекс FAQ."""
    faq = items if items is not None else load_faq(faq_path, root)
    chunks: list[Chunk] = []
    for item in faq:
        chunks.extend(chunk_faq(item, chunk_size=chunk_size, overlap=overlap))

    embedder = embedder or HashingEmbedder()
    metadata = {
        "dataset": dataset,
        "strategy": strategy,
        "kind": "support_faq",
        "source": str(faq_path),
        "items": [item.id for item in faq],
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    index = FaissIndex.build(chunks, embedder=embedder, metadata=metadata)
    if save:
        target = _resolve(index_dir, root) / dataset / strategy
        index.save(target)
    return index, faq


def load_faq_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    root: Path = PROJECT_ROOT,
) -> RagRetriever:
    """Загружает индекс FAQ с диска как :class:`RagRetriever`."""
    target_dir = _resolve(index_dir, root) / dataset / strategy
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Индекс FAQ не найден: {target_dir}. "
            f"Соберите его: python3 -m agent.support_faq --build"
        )

    dim = 256
    meta_file = target_dir / "index_meta.json"
    if meta_file.exists():
        try:
            dim = int(json.loads(meta_file.read_text(encoding="utf-8")).get("dim", 256))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    embedder = HashingEmbedder(dim=dim)
    index = FaissIndex.load(target_dir, embedder=embedder)
    return RagRetriever(index=index, top_k=top_k, min_score=min_score)


def build_faq_context(sources: list[RagSource], max_chars: int = 6000) -> str:
    """Собирает контекст FAQ с указанием вопроса и раздела."""
    if not sources:
        return "В базе знаний (FAQ) не найдено подходящих карточек по запросу."

    blocks: list[str] = []
    current_len = 0
    for idx, src in enumerate(sources, start=1):
        meta_line = (
            f"[Источник {idx}] Вопрос FAQ: «{src.title}», раздел: «{src.section}» "
            f"(сходство: {src.score:.3f})"
        )
        body = src.text.strip()
        if body.startswith("FAQ: ") and "\n" in body:
            body = body.split("\n", 1)[1].strip()
        block = f"{meta_line}\n{body}"
        block_len = len(block) + 2
        if current_len + block_len > max_chars and blocks:
            break
        blocks.append(block)
        current_len += block_len
    return "\n\n".join(blocks)


# ============================================================================
# Перечень вопросов для проверки
# ============================================================================
#
# ``expected_faq_ids`` — целевые карточки FAQ (для оценки полноты поиска).
# ``expected_facts``   — OR-группы синонимов: группа засчитана, если хотя бы один
#                        синоним встречается в ответе (как в agent/rag.py).
# ``negative``         — контроль галлюцинаций: ответа в FAQ быть не должно.

CONTROL_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q1",
        "query": "Почему не работает авторизация?",
        "category": "Авторизация",
        "expected_faq_ids": ["faq-auth-001"],
        "expected_facts": [
            ["сесси", "cookie", "secret.key"],
            ["парол", "логин", "3 символ", "6 символ"],
        ],
        "expectation": "Ответ объясняет причины сбоя входа (учётные данные, сессия/secret.key) и что данные тикета учтены.",
    },
    {
        "id": "q2",
        "query": "Вход пишет «Неверное имя пользователя или пароль»",
        "category": "Авторизация",
        "expected_faq_ids": ["faq-auth-002"],
        "expected_facts": [["401", "неверн"], ["хеш", "парол", "регистр"]],
        "expectation": "Указано, что это 401 и пара логин+пароль не совпала; пароль хранится как хеш.",
    },
    {
        "id": "q3",
        "query": "Какие требования к паролю и логину при регистрации?",
        "category": "Регистрация",
        "expected_faq_ids": ["faq-reg-001"],
        "expected_facts": [["6"], ["3 символ", "3 символ"]],
        "expectation": "Пароль минимум 6 символов, имя пользователя минимум 3 символа.",
    },
    {
        "id": "q4",
        "query": "Сколько чатов можно создать одному пользователю?",
        "category": "Чаты",
        "expected_faq_ids": ["faq-chat-001"],
        "expected_facts": [["не огранич", "много", "любое"], ["чат"]],
        "expectation": "Количество чатов не ограничено на уровне сервиса.",
    },
    {
        "id": "q5",
        "query": "Как переименовать или удалить чат?",
        "category": "Чаты",
        "expected_faq_ids": ["faq-chat-002"],
        "expected_facts": [["PATCH", "DELETE", "api/chats"]],
        "expectation": "Переименование — PATCH /api/chats/<id>, удаление — DELETE /api/chats/<id>.",
    },
    {
        "id": "q6",
        "query": "Как очистить историю сообщений в чате?",
        "category": "Чаты",
        "expected_faq_ids": ["faq-chat-003"],
        "expected_facts": [["reset"], ["истори", "сообщени"]],
        "expectation": "Очистка истории — POST /api/chats/<id>/reset.",
    },
    {
        "id": "q7",
        "query": "Почему ассистент не находит нужный рецепт?",
        "category": "RAG и поиск",
        "expected_faq_ids": ["faq-rag-001"],
        "expected_facts": [["индекс", "index", "датасет"], ["порог", "релевант", "top_k", "формулиров"]],
        "expectation": "Названы причины: нет в индексе, низкое сходство/порог, формулировка, укороченный датасет.",
    },
    {
        "id": "q8",
        "query": "Чем ответ с RAG отличается от ответа без RAG?",
        "category": "RAG и поиск",
        "expected_faq_ids": ["faq-rag-002"],
        "expected_facts": [["источник"], ["галлюцин", "общ", "выдум"]],
        "expectation": "С RAG — ответ по источникам [Источник N]; без RAG — по общим знаниям, риск галлюцинаций.",
    },
    {
        "id": "q9",
        "query": "Как пересобрать индекс рецептов?",
        "category": "Индекс и пересборка",
        "expected_faq_ids": ["faq-index-001"],
        "expected_facts": [["rebuild_index"], ["structural", "copy", "source"]],
        "expectation": "Команда rebuild_index.py --source ... --strategy structural (или --copy).",
    },
    {
        "id": "q10",
        "query": "Как включить пересборку индекса на ходу без перезапуска?",
        "category": "Индекс и пересборка",
        "expected_faq_ids": ["faq-index-002"],
        "expected_facts": [["allow_reindex"], ["reindex"]],
        "expectation": "web.allow_reindex=true и POST /api/admin/reindex.",
    },
    {
        "id": "q11",
        "query": "Модель отвечает очень долго, что делать?",
        "category": "Локальная модель",
        "expected_faq_ids": ["faq-llm-001"],
        "expected_facts": [["очеред", "semaphore", "по одному"], ["max_tokens", "квант", "threads", "контекст"]],
        "expectation": "Генерация сериализована (Semaphore(1)), помогают параметры/квантование/потоки.",
    },
    {
        "id": "q12",
        "query": "Что означает офлайн-режим сервиса?",
        "category": "Офлайн-режим",
        "expected_faq_ids": ["faq-offline-001"],
        "expected_facts": [["стаб", "детерминиров"], ["модел", "сеть"]],
        "expectation": "run.py --offline поднимает стаб вместо модели, без сети.",
    },
    {
        "id": "q13",
        "query": "Где хранятся пользователи и сообщения и как сделать бэкап?",
        "category": "Данные и бэкап",
        "expected_faq_ids": ["faq-data-001"],
        "expected_facts": [["service.db"], ["копир", "бэкап", "скопир"]],
        "expectation": "Данные в data/service.db, бэкап — копирование файла базы.",
    },
    {
        "id": "q14",
        "query": "Почему после перезапуска разлогинило всех пользователей?",
        "category": "Данные и бэкап",
        "expected_faq_ids": ["faq-data-002"],
        "expected_facts": [["secret.key"], ["сесси", "cookie", "разлог"]],
        "expectation": "Смена/удаление ключа подписи делает cookie-сессии недействительными.",
    },
    {
        "id": "q15",
        "query": "Где смотреть логи локальной модели?",
        "category": "Ошибки и логи",
        "expected_faq_ids": ["faq-errors-001"],
        "expected_facts": [["server.log"]],
        "expectation": "Логи модели в data/local_llm/server.log.",
    },
    {
        "id": "q16",
        "query": "Сколько стоит подписка и можно ли оплатить картой?",
        "category": "Контроль галлюцинаций",
        "expected_faq_ids": [],
        "expected_facts": [
            [
                "не указан",
                "отсутств",
                "нет данн",
                "не содержит",
                "не найдено",
                "нет информац",
                "не найдена",
                "не указано",
                "нет ответ",
                "эскалац",
            ],
        ],
        "negative": True,
        "expectation": "В FAQ нет данных о тарифах; ассистент не выдумывает и предлагает эскалацию.",
    },
]


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _print_build_stats(items: list[SupportFaq], index: FaissIndex) -> None:
    by_category: dict[str, int] = {}
    for chunk in index.chunks:
        category = chunk.metadata.get("category", "?")
        by_category[category] = by_category.get(category, 0) + 1
    print(f"Карточек FAQ: {len(items)}; чанков: {len(index.chunks)}; dim: {index.embedder.dim}")
    for category in sorted(by_category):
        print(f"  - {category}: {by_category[category]} чанков")


def _run_query(retriever: RagRetriever, query: str, top_k: int, max_chars: int) -> None:
    sources = retriever.retrieve(query, top_k=top_k)
    print(f"Запрос: {query}\nНайдено источников: {len(sources)}\n")
    for idx, src in enumerate(sources, start=1):
        print(f"{idx}. [{(src.recipe_id)}] «{src.title}» — {src.section} (сходство {src.score:.3f})")
        print(f"   {_preview(src.text)}\n")
    print("Контекст:\n" + build_faq_context(sources, max_chars=max_chars))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 33: индексация FAQ поддержки (mcp/support_data/faq.json) в FAISS"
    )
    parser.add_argument("--build", action="store_true", help="собрать индекс (по умолчанию)")
    parser.add_argument("--query", default=None, help="поисковый запрос по FAQ")
    parser.add_argument("--top-k", type=int, default=5, help="сколько чанков вернуть")
    parser.add_argument("--faq-path", default=DEFAULT_FAQ_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--max-context-chars", type=int, default=6000, help="лимит контекста для --query"
    )
    args = parser.parse_args()

    if args.query:
        try:
            retriever = load_faq_retriever(
                index_dir=args.index_dir,
                dataset=args.dataset,
                strategy=args.strategy,
                top_k=args.top_k,
            )
        except FileNotFoundError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 1
        _run_query(retriever, args.query, args.top_k, args.max_context_chars)
        return 0

    index, items = build_faq_index(
        faq_path=args.faq_path,
        index_dir=args.index_dir,
        dataset=args.dataset,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    target = (PROJECT_ROOT / args.index_dir / args.dataset / args.strategy).resolve()
    print(f"✓ Индекс FAQ сохранён: {target}")
    _print_build_stats(items, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
