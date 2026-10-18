"""День 32: индексация исходного кода проекта для RAG-ревью.

Дополняет :mod:`agent.project_docs` (документация) вторым источником знаний —
**кодом**. Модуль разбирает Python-файлы через :mod:`ast`, нарезает их на чанки
по top-level символам (классы и функции) с метаданными (модуль, символ, вид,
строка) и строит локальный индекс FAISS тем же офлайн-эмбеддером
(``HashingEmbedder``), что и в Днях 21–22.

Компоненты:

* :func:`discover_code_files` — находит исходники по glob-шаблонам;
* :func:`parse_module_symbols` — извлекает символы модуля через ``ast``;
* :func:`chunk_code_module` — превращает модуль в список чанков;
* :func:`build_code_index` — собирает и сохраняет индекс;
* :func:`load_code_retriever` — загружает индекс как :class:`RagRetriever`;
* :func:`build_code_context` — собирает контекст с указанием файла и символа.

Запуск::

    python3 -m agent.code_index --build
    python3 -m agent.code_index --query "как устроен RagRetriever" --top-k 5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .indexing import Chunk, FaissIndex, HashingEmbedder, estimate_tokens
from .rag import RagRetriever, RagSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CODE_INDEX_DIR = "history/index"
DEFAULT_CODE_DATASET = "project_code"
DEFAULT_CODE_STRATEGY = "symbols"
DEFAULT_MAX_SYMBOL_CHARS = 3000

# Исходники проекта, которые попадают в RAG-индекс кода.
DEFAULT_CODE_GLOBS = (
    "agent/*.py",
    "mcp/*.py",
    "scripts/*.py",
    "recipe_chat_service/core/*.py",
    "recipe_chat_service/app/**/*.py",
)

# Тестовые/сгенерированные файлы, которые не нужно индексировать.
_SKIP_SUFFIX = ("__init__.py",)


@dataclass
class CodeSymbol:
    """Один top-level символ модуля (класс, функция, константы, шапка)."""

    name: str
    kind: str  # "module" | "constants" | "class" | "function"
    lineno: int
    end_lineno: int
    text: str


@dataclass
class CodeModule:
    """Python-модуль, попадающий в RAG-индекс кода."""

    path: str  # относительный путь от корня проекта
    title: str
    text: str
    symbols: list[CodeSymbol] = field(default_factory=list)


def _slug(value: str) -> str:
    """Преобразует путь в безопасный идентификатор документа."""
    slug = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", value).strip("_").lower()
    return slug or "module"


def discover_code_files(
    root: Path = PROJECT_ROOT,
    globs: tuple[str, ...] = DEFAULT_CODE_GLOBS,
) -> list[Path]:
    """Находит Python-файлы проекта по glob-шаблонам (отсортировано)."""
    found: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path.name in _SKIP_SUFFIX:
                continue
            found[path.relative_to(root).as_posix()] = path
    return [found[key] for key in sorted(found)]


def _module_title(tree: ast.Module, path: Path) -> str:
    """Берёт первую строку docstring модуля как название, иначе — имя файла."""
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return first or path.stem


def parse_module_symbols(text: str, max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS) -> list[CodeSymbol]:
    """Извлекает top-level символы модуля через ``ast``.

    Возвращает шапку модуля (docstring + импорты), блок констант и по одному
    символу на класс/функцию. Длинные символы режутся на окна с перекрытием.
    При ``SyntaxError`` возвращает пустой список (вызывающий сделает fallback).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    symbols: list[CodeSymbol] = []

    def segment(node: ast.AST) -> str:
        return ast.get_source_segment(text, node) or ""

    # 1. Шапка модуля: docstring + импорты (полезный контекст для ревью).
    header_parts: list[str] = []
    docstring = ast.get_docstring(tree)
    if docstring:
        header_parts.append('"""' + docstring + '"""')
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_parts.append(segment(node))
    if header_parts:
        symbols.append(
            CodeSymbol(
                name="<module>",
                kind="module",
                lineno=1,
                end_lineno=1,
                text="\n".join(part for part in header_parts if part),
            )
        )

    # 2. Блок модульных констант (Assignment/AnnAssign верхнего уровня).
    constant_nodes = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    if constant_nodes:
        constants_text = "\n".join(segment(node) for node in constant_nodes)
        symbols.append(
            CodeSymbol(
                name="<constants>",
                kind="constants",
                lineno=min(node.lineno for node in constant_nodes),
                end_lineno=max(node.end_lineno or node.lineno for node in constant_nodes),
                text=constants_text,
            )
        )

    # 3. Классы и функции верхнего уровня.
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end = node.end_lineno or node.lineno
        body = "\n".join(lines[start - 1 : end])
        for piece in _split_long_text(body, max_symbol_chars, max_symbol_chars // 5):
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                    lineno=start,
                    end_lineno=end,
                    text=piece,
                )
            )
    return symbols


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет длинный текст на окна с перекрытием по границам строк/слов."""
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

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


def chunk_code_module(
    module: CodeModule,
    max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS,
) -> list[Chunk]:
    """Превращает модуль в чанки с метаданными о файле и символе."""
    chunks: list[Chunk] = []
    for index, symbol in enumerate(module.symbols):
        chunk_id = f"code_{_slug(module.path)}_{index:03d}"
        # Заголовок файла и символа включаем в текст чанка, чтобы эмбеддер
        # учитывал их при поиске (вектор строится по тексту, а не по метаданным).
        context_line = (
            f"Файл: {module.path} — символ: {symbol.name} "
            f"({symbol.kind}, строка {symbol.lineno})"
        )
        chunk_text = f"{context_line}\n{symbol.text}"
        metadata: dict[str, Any] = {
            "strategy": "symbols",
            "doc_type": "code",
            "module": module.path,
            # recipe_id/url/title/section — совместимость с RagSource.
            "recipe_id": module.path,
            "title": module.title,
            "section": symbol.name,
            "heading_path": f"{module.path} > {symbol.name}",
            "url": module.path,
            "source": module.path,
            "symbol": symbol.name,
            "kind": symbol.kind,
            "lineno": symbol.lineno,
            "step_number": None,
            "chunk_index": index,
            "n_chars": len(chunk_text),
            "n_tokens": estimate_tokens(chunk_text),
        }
        chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=metadata))
    return chunks


def load_code_module(path: Path, root: Path = PROJECT_ROOT) -> CodeModule:
    """Загружает один Python-модуль и разбирает его символы."""
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(root).as_posix()
    try:
        tree: ast.Module | None = ast.parse(text)
    except SyntaxError:
        tree = None
    title = _module_title(tree, path) if tree is not None else path.stem
    symbols = parse_module_symbols(text)
    if not symbols:
        # Fallback: файл не парсится (сгенерирован/битый) — режем целиком.
        symbols = [
            CodeSymbol(
                name="<file>",
                kind="file",
                lineno=1,
                end_lineno=len(text.splitlines()),
                text=text,
            )
        ]
    return CodeModule(path=relative, title=title, text=text, symbols=symbols)


def load_code_modules(
    root: Path = PROJECT_ROOT,
    globs: tuple[str, ...] = DEFAULT_CODE_GLOBS,
) -> list[CodeModule]:
    """Загружает все исходники проекта."""
    return [load_code_module(path, root) for path in discover_code_files(root, globs)]


def build_code_index(
    root: Path = PROJECT_ROOT,
    index_dir: str | Path = DEFAULT_CODE_INDEX_DIR,
    dataset: str = DEFAULT_CODE_DATASET,
    strategy: str = DEFAULT_CODE_STRATEGY,
    max_symbol_chars: int = DEFAULT_MAX_SYMBOL_CHARS,
    embedder: HashingEmbedder | None = None,
    modules: list[CodeModule] | None = None,
    save: bool = True,
) -> tuple[FaissIndex, list[CodeModule]]:
    """Собирает (и при ``save=True`` сохраняет) индекс исходного кода проекта."""
    loaded = modules if modules is not None else load_code_modules(root)
    chunks: list[Chunk] = []
    for module in loaded:
        chunks.extend(chunk_code_module(module, max_symbol_chars=max_symbol_chars))

    embedder = embedder or HashingEmbedder()
    metadata = {
        "dataset": dataset,
        "strategy": strategy,
        "kind": "project_code",
        "modules": [module.path for module in loaded],
        "max_symbol_chars": max_symbol_chars,
    }
    index = FaissIndex.build(chunks, embedder=embedder, metadata=metadata)
    if save:
        target = Path(index_dir)
        if not target.is_absolute():
            target = root / target
        index.save(target / dataset / strategy)
    return index, loaded


def load_code_retriever(
    index_dir: str | Path = DEFAULT_CODE_INDEX_DIR,
    dataset: str = DEFAULT_CODE_DATASET,
    strategy: str = DEFAULT_CODE_STRATEGY,
    top_k: int = 5,
    min_score: float = 0.0,
    root: Path = PROJECT_ROOT,
) -> RagRetriever:
    """Загружает индекс кода с диска как :class:`RagRetriever`."""
    target = Path(index_dir)
    if not target.is_absolute():
        target = root / target
    target_dir = target / dataset / strategy
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Индекс кода не найден: {target_dir}. "
            f"Соберите его: python3 -m agent.code_index --build"
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


def build_code_context(sources: list[RagSource], max_chars: int = 6000) -> str:
    """Собирает контекст кода с указанием файла и символа."""
    if not sources:
        return "В кодовой базе проекта не найдено подходящих фрагментов по запросу."

    blocks: list[str] = []
    current_len = 0
    for idx, src in enumerate(sources, start=1):
        path = src.url or "код"
        meta_line = (
            f"[Код {idx}] Файл: {path}, символ: {src.section} "
            f"(сходство: {src.score:.3f})"
        )
        body = _body_without_header(src.text)
        block = f"{meta_line}\n{body}"
        block_len = len(block) + 2
        if current_len + block_len > max_chars and blocks:
            break
        blocks.append(block)
        current_len += block_len
    return "\n\n".join(blocks)


def _body_without_header(text: str) -> str:
    """Убирает служебную строку «Файл: … — символ: …» из начала чанка."""
    stripped = text.strip()
    if stripped.startswith("Файл: ") and "\n" in stripped:
        return stripped.split("\n", 1)[1].strip()
    return stripped


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _print_build_stats(modules: list[CodeModule], index: FaissIndex) -> None:
    by_file: dict[str, int] = {}
    for chunk in index.chunks:
        source = chunk.metadata.get("source", "?")
        by_file[source] = by_file.get(source, 0) + 1
    print(
        f"Модулей: {len(modules)}; чанков: {len(index.chunks)}; "
        f"dim: {index.embedder.dim}"
    )
    for source in sorted(by_file):
        print(f"  - {source}: {by_file[source]} чанков")


def _run_query(retriever: RagRetriever, query: str, top_k: int, max_chars: int) -> None:
    sources = retriever.retrieve(query, top_k=top_k)
    print(f"Запрос: {query}\nНайдено источников: {len(sources)}\n")
    for idx, src in enumerate(sources, start=1):
        print(f"{idx}. {src.url} — «{src.section}» (сходство {src.score:.3f})")
        print(f"   {_preview(src.text)}\n")
    print("Контекст:\n" + build_code_context(sources, max_chars=max_chars))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="День 32: индексация исходного кода проекта (symbols) в FAISS"
    )
    parser.add_argument("--build", action="store_true", help="собрать индекс (по умолчанию)")
    parser.add_argument("--query", default=None, help="поисковый запрос по коду")
    parser.add_argument("--top-k", type=int, default=5, help="сколько чанков вернуть")
    parser.add_argument("--index-dir", default=DEFAULT_CODE_INDEX_DIR)
    parser.add_argument("--dataset", default=DEFAULT_CODE_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_CODE_STRATEGY)
    parser.add_argument("--max-symbol-chars", type=int, default=DEFAULT_MAX_SYMBOL_CHARS)
    parser.add_argument(
        "--max-context-chars", type=int, default=6000, help="лимит контекста для --query"
    )
    args = parser.parse_args()

    if args.query:
        try:
            retriever = load_code_retriever(
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

    index, modules = build_code_index(
        index_dir=args.index_dir,
        dataset=args.dataset,
        strategy=args.strategy,
        max_symbol_chars=args.max_symbol_chars,
    )
    target = (PROJECT_ROOT / args.index_dir / args.dataset / args.strategy).resolve()
    print(f"✓ Индекс кода сохранён: {target}")
    _print_build_stats(modules, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
