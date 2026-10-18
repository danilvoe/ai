"""День 31: индексация документации проекта для ассистента разработчика.

Модуль готовит RAG-базу знаний о самом проекте: загружает README и папку
``docs/``, разбивает markdown по заголовкам на чанки с метаданными (документ,
раздел, путь к файлу) и строит локальный векторный индекс FAISS тем же
офлайн-эмбеддером (``HashingEmbedder``), что и в Днях 21–22.

Компоненты:

* :func:`discover_documents` — находит markdown-файлы документации;
* :func:`chunk_markdown` — структурный чанкинг по заголовкам;
* :func:`build_documents_index` — собирает и сохраняет индекс;
* :func:`load_project_retriever` — загружает индекс как :class:`RagRetriever`;
* :func:`build_docs_context` — собирает контекст с указанием файла и раздела.

Запуск::

    python3 -m agent.project_docs --build
    python3 -m agent.project_docs --query "как устроен RAG?" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .indexing import Chunk, FaissIndex, HashingEmbedder, estimate_tokens
from .rag import RagRetriever, RagSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INDEX_DIR = "history/index"
DEFAULT_DATASET = "project_docs"
DEFAULT_STRATEGY = "markdown"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120

# Markdown-файлы, из которых состоит база знаний о проекте.
DEFAULT_DOC_GLOBS = (
    "README.md",
    "AGENTS.md",
    "docs/*.md",
    "recipe_chat_service/README.md",
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class ProjectDocument:
    """Один документ проекта, попадающий в RAG-индекс."""

    id: str
    title: str
    path: str  # относительный путь от корня проекта
    text: str
    doc_type: str = "markdown"


def _slug(value: str) -> str:
    """Преобразует путь в безопасный идентификатор документа."""
    slug = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", value).strip("_").lower()
    return slug or "document"


def _document_title(text: str, path: Path) -> str:
    """Берёт первый заголовок H1 как название документа, иначе — имя файла."""
    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return path.stem


def discover_documents(
    root: Path = PROJECT_ROOT,
    globs: tuple[str, ...] = DEFAULT_DOC_GLOBS,
) -> list[Path]:
    """Находит существующие markdown-файлы документации (отсортировано)."""
    found: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                found[path.relative_to(root).as_posix()] = path
    return [found[key] for key in sorted(found)]


def load_document(path: Path, root: Path = PROJECT_ROOT) -> ProjectDocument:
    """Загружает один markdown-документ в :class:`ProjectDocument`."""
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(root).as_posix()
    return ProjectDocument(
        id=_slug(relative),
        title=_document_title(text, path),
        path=relative,
        text=text,
    )


def load_documents(
    root: Path = PROJECT_ROOT,
    globs: tuple[str, ...] = DEFAULT_DOC_GLOBS,
) -> list[ProjectDocument]:
    """Загружает всю документацию проекта."""
    return [load_document(path, root) for path in discover_documents(root, globs)]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет длинный раздел на окна с перекрытием по границам строк/слов."""
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


def _section_chunks(
    doc: ProjectDocument,
    heading: str,
    heading_path: str,
    body: str,
    chunk_index: int,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], int]:
    """Создаёт чанки для одного раздела (с разбиением, если он длинный)."""
    chunks: list[Chunk] = []
    for piece in _split_long_text(body, chunk_size, overlap) if body else []:
        chunk_id = f"doc_{doc.id}_{chunk_index:03d}"
        # Заголовок документа и раздела включаем в текст чанка, чтобы эмбеддер
        # учитывал их при поиске (вектор строится по тексту, а не по метаданным).
        context_line = f"Документ: {doc.title} — раздел: {heading_path}"
        chunk_text = f"{context_line}\n{piece}"
        metadata: dict[str, Any] = {
            "strategy": "markdown",
            "doc_type": doc.doc_type,
            # recipe_id/url/title/section — совместимость с RagSource.
            "recipe_id": doc.id,
            "title": doc.title,
            "section": heading,
            "heading_path": heading_path,
            "url": doc.path,
            "source": doc.path,
            "step_number": None,
            "chunk_index": chunk_index,
            "n_chars": len(chunk_text),
            "n_tokens": estimate_tokens(chunk_text),
        }
        chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=metadata))
        chunk_index += 1
    return chunks, chunk_index


def chunk_markdown(
    doc: ProjectDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Структурный чанкинг markdown по заголовкам.

    Каждый раздел становится одним чанком; слишком длинные разделы режутся на
    окна с перекрытием. Метаданные содержат документ, заголовок раздела, путь к
    файлу и токенную оценку.
    """
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    heading = "введение"
    heading_path = doc.title
    buffer: list[str] = []
    chunk_index = 0
    fence: str | None = None

    def flush() -> None:
        nonlocal buffer, chunk_index
        body = "\n".join(buffer).strip()
        if body:
            new_chunks, chunk_index = _section_chunks(
                doc, heading, heading_path, body, chunk_index, chunk_size, overlap
            )
            chunks.extend(new_chunks)
        buffer = []

    for raw_line in doc.text.splitlines():
        stripped = raw_line.strip()
        fence_match = re.match(r"^(```+|~~~+)", stripped)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            buffer.append(raw_line)
            continue
        # Заголовки внутри блоков кода (``` … ```) не считаем заголовками
        # документа: иначе bash-комментарии «# …» ломают структуру разделов.
        if fence is not None:
            buffer.append(raw_line)
            continue
        match = _HEADING_RE.match(stripped)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading = title
            heading_path = " > ".join(item[1] for item in heading_stack)
            # Не дублируем заголовок документа (H1) в оглавлении раздела.
            if len(heading_stack) > 1 and heading_stack[0][1] == doc.title:
                heading_path = " > ".join(item[1] for item in heading_stack[1:])
            continue
        buffer.append(raw_line)

    flush()
    return chunks


def build_documents_index(
    root: Path = PROJECT_ROOT,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: HashingEmbedder | None = None,
    documents: list[ProjectDocument] | None = None,
    save: bool = True,
) -> tuple[FaissIndex, list[ProjectDocument]]:
    """Собирает (и при ``save=True`` сохраняет) индекс документации проекта."""
    docs = documents if documents is not None else load_documents(root)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_markdown(doc, chunk_size=chunk_size, overlap=overlap))

    embedder = embedder or HashingEmbedder()
    metadata = {
        "dataset": dataset,
        "strategy": strategy,
        "kind": "project_docs",
        "documents": [doc.path for doc in docs],
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    index = FaissIndex.build(chunks, embedder=embedder, metadata=metadata)
    if save:
        target = Path(index_dir)
        if not target.is_absolute():
            target = root / target
        index.save(target / dataset / strategy)
    return index, docs


def load_project_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    root: Path = PROJECT_ROOT,
) -> RagRetriever:
    """Загружает индекс документации с диска как :class:`RagRetriever`."""
    target = Path(index_dir)
    if not target.is_absolute():
        target = root / target
    target_dir = target / dataset / strategy
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Индекс документации не найден: {target_dir}. "
            f"Соберите его: python3 -m agent.project_docs --build"
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


def build_docs_context(sources: list[RagSource], max_chars: int = 6000) -> str:
    """Собирает контекст документации с указанием файла и раздела."""
    if not sources:
        return "В документации проекта не найдено подходящих разделов по запросу."

    blocks: list[str] = []
    current_len = 0
    for idx, src in enumerate(sources, start=1):
        path = src.url or "документация"
        meta_line = (
            f"[Источник {idx}] Документ: «{src.title}», файл: {path}, "
            f"раздел: «{src.section}» (сходство: {src.score:.3f})"
        )
        block = f"{meta_line}\n{src.text.strip()}"
        block_len = len(block) + 2
        if current_len + block_len > max_chars and blocks:
            break
        blocks.append(block)
        current_len += block_len
    return "\n\n".join(blocks)


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _print_build_stats(docs: list[ProjectDocument], index: FaissIndex) -> None:
    by_doc: dict[str, int] = {}
    for chunk in index.chunks:
        source = chunk.metadata.get("source", "?")
        by_doc[source] = by_doc.get(source, 0) + 1
    print(f"Документов: {len(docs)}; чанков: {len(index.chunks)}; dim: {index.embedder.dim}")
    for source in sorted(by_doc):
        print(f"  - {source}: {by_doc[source]} чанков")


def _run_query(
    retriever: RagRetriever, query: str, top_k: int, max_chars: int
) -> None:
    sources = retriever.retrieve(query, top_k=top_k)
    print(f"Запрос: {query}\nНайдено источников: {len(sources)}\n")
    for idx, src in enumerate(sources, start=1):
        print(f"{idx}. {src.url} — «{src.section}» (сходство {src.score:.3f})")
        print(f"   {_preview(src.text)}\n")
    print("Контекст:\n" + build_docs_context(sources, max_chars=max_chars))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 31: индексация документации проекта (README + docs) в FAISS"
    )
    parser.add_argument("--build", action="store_true", help="собрать индекс (по умолчанию)")
    parser.add_argument("--query", default=None, help="поисковый запрос по документации")
    parser.add_argument("--top-k", type=int, default=5, help="сколько чанков вернуть")
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
            retriever = load_project_retriever(
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

    index, docs = build_documents_index(
        index_dir=args.index_dir,
        dataset=args.dataset,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    target = (PROJECT_ROOT / args.index_dir / args.dataset / args.strategy).resolve()
    print(f"✓ Индекс документации сохранён: {target}")
    _print_build_stats(docs, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
