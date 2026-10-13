"""Сборка и пересоздание FAISS-индекса рецептов для сервиса.

Обёртка над :mod:`core.indexing`:

* :func:`copy_index` — копирует готовый индекс из репозитория;
* :func:`build_index` — пересобирает индекс из датасета (структура JSON не меняется);
* :func:`build_all_strategies` — пересобирает все стратегии (``structural``/``fixed``).

Запись атомарна: сборка идёт во временный каталог, затем файлы переносятся
через :func:`os.replace`, чтобы работающий сервис не увидел полу-записанный индекс.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .indexing import Chunk, FaissIndex, HashingEmbedder, chunk_fixed_size, chunk_structural, load_recipes

SERVICE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = SERVICE_ROOT / "index"
DEFAULT_DATASET = "recipt_all"
DEFAULT_STRATEGY = "structural"
DEFAULT_DIM = 256

INDEX_FILES = ("faiss.index", "chunks.json", "index_meta.json")
STRATEGIES = ("structural", "fixed")


class IndexBuildError(RuntimeError):
    """Ошибка сборки/копирования индекса."""


def _chunk(recipes: list[dict], strategy: str, dataset: str) -> list[Chunk]:
    if strategy == "structural":
        return chunk_structural(recipes, source=dataset)
    if strategy == "fixed":
        return chunk_fixed_size(recipes, source=dataset)
    raise IndexBuildError(
        f"Неизвестная стратегия '{strategy}'. Допустимо: {', '.join(STRATEGIES)}."
    )


def dataset_path(dataset: str, index_dir: str | Path = DEFAULT_INDEX_DIR) -> Path:
    """Путь к JSON-датасету: ``<index_dir>/dataset/<dataset>.json``."""
    return Path(index_dir).resolve() / "dataset" / f"{dataset}.json"


def copy_index(
    src_dir: str | Path,
    dst_dir: str | Path,
) -> Path:
    """Копирует готовый индекс (три файла) из ``src_dir`` в ``dst_dir``.

    Возвращает каталог назначения.
    """
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if not src.is_dir():
        raise IndexBuildError(f"Исходный индекс не найден: {src}")
    missing = [name for name in INDEX_FILES if not (src / name).exists()]
    if missing:
        raise IndexBuildError(
            f"В {src} отсутствуют файлы индекса: {', '.join(missing)}"
        )
    dst.mkdir(parents=True, exist_ok=True)
    for name in INDEX_FILES:
        shutil.copy2(src / name, dst / name)
    return dst


def build_index(
    source: str | Path,
    dataset: str = DEFAULT_DATASET,
    strategy: str = DEFAULT_STRATEGY,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dim: int = DEFAULT_DIM,
) -> Path:
    """Пересобирает индекс одной стратегии из датасета и сохраняет его атомарно.

    Структура JSON не меняется: пишутся те же ``faiss.index`` / ``chunks.json`` /
    ``index_meta.json``. Возвращает каталог ``<index_dir>/<dataset>/<strategy>``.
    """
    recipes, file_meta = load_recipes(source)
    if not recipes:
        raise IndexBuildError(f"Датасет пуст: {source}")

    chunks = _chunk(recipes, strategy, dataset)
    if not chunks:
        raise IndexBuildError(f"Не удалось построить чанки для {dataset}/{strategy}")

    embedder = HashingEmbedder(dim=dim)
    metadata = {
        "strategy": strategy,
        "source": dataset,
        "source_file": file_meta.get("source_name", Path(source).name),
    }
    index = FaissIndex.build(chunks, embedder, metadata=metadata)

    target = Path(index_dir).resolve() / dataset / strategy
    target.mkdir(parents=True, exist_ok=True)
    # Временный каталог создаём рядом с целью: os.replace атомарен только
    # в пределах одной файловой системы.
    with tempfile.TemporaryDirectory(prefix=".index_build_", dir=target.parent) as tmp:
        tmp_dir = Path(tmp)
        index.save(tmp_dir)
        for name in INDEX_FILES:
            os.replace(tmp_dir / name, target / name)
    return target


def build_all_strategies(
    source: str | Path,
    dataset: str = DEFAULT_DATASET,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    dim: int = DEFAULT_DIM,
) -> list[Path]:
    """Пересобирает все стратегии (``structural`` и ``fixed``)."""
    return [
        build_index(source, dataset=dataset, strategy=strategy, index_dir=index_dir, dim=dim)
        for strategy in STRATEGIES
    ]


def resolve_dataset_source(
    dataset: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> Path:
    """Возвращает путь к датасету, откатываясь на ``recipt_sample.json``.

    Приоритет: ``<index_dir>/dataset/<dataset>.json`` → ``<index_dir>/dataset/recipt_sample.json``
    → ``<SERVICE_ROOT>/../recipt_sample.json``.
    """
    primary = dataset_path(dataset, index_dir)
    if primary.exists():
        return primary
    fallback = Path(index_dir).resolve() / "dataset" / "recipt_sample.json"
    if fallback.exists():
        return fallback
    repo_sample = SERVICE_ROOT.parent / "recipt_sample.json"
    if repo_sample.exists():
        return repo_sample
    raise IndexBuildError(
        f"Датасет не найден: {primary}. Положите JSON в {primary.parent}."
    )
