"""Day 21: Пайплайн индексации документов (рецептов).

Включает:
1. Загрузка документов из JSON (с поддержкой произвольного источника).
2. Две стратегии разбиения на чанки (chunking):
   - по фиксированному размеру (fixed-size с перекрытием и smart-границами слов);
   - по структуре документа (structural: шапка, ингредиенты, вступление, шаги, оборудование).
3. Обогащение каждого чанка структурированными метаданными (усиление).
4. Генерация векторных эмбеддингов (детерминированный HashingEmbedder офлайн
   и опциональный OpenAIEmbedder через API).
5. Локальный векторный индекс FAISS (IndexFlatIP) с сохранением, загрузкой и поиском.
6. Сравнение стратегий по структурным характеристикам и качеству поиска (Recall@k, MRR@k).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import faiss
import numpy as np

from agent.tokens import estimate_tokens


# ============================================================================
# 1. Модель чанка и метаданных
# ============================================================================


@dataclass
class Chunk:
    """Единица индексации: текст и сопутствующие структурированные метаданные."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata", {})),
        )


# ============================================================================
# 2. Загрузка документов
# ============================================================================


def load_recipes(source: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Загружает рецепты из JSON-файла.

    Поддерживает:
    - формат с обёрткой: ``{"total_recipes": N, "recipes": [...]}``
    - плоский список: ``[...]``

    Возвращает список словарей рецептов и метаданные источника.
    """
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл рецептов не найден: {path}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        recipes = data.get("recipes")
        if not isinstance(recipes, list):
            raise ValueError(
                f"Некорректная структура JSON в {path.name}: ожидался ключ 'recipes' со списком."
            )
        total_declared = data.get("total_recipes", len(recipes))
    elif isinstance(data, list):
        recipes = data
        total_declared = len(recipes)
    else:
        raise ValueError(
            f"Неподдерживаемый корневой тип в {path.name}: {type(data).__name__}"
        )

    # Базовая валидация элементов
    valid_recipes: list[dict[str, Any]] = []
    for idx, item in enumerate(recipes, start=1):
        if not isinstance(item, dict):
            continue
        # Гарантируем строковый или целочисленный id
        recipe_id = item.get("id")
        if recipe_id is None:
            item["id"] = idx
        valid_recipes.append(item)

    file_meta = {
        "source_path": str(path),
        "source_name": path.name,
        "source_size_bytes": path.stat().st_size,
        "total_declared": total_declared,
        "total_loaded": len(valid_recipes),
    }
    return valid_recipes, file_meta


def recipe_to_text(recipe: dict[str, Any]) -> str:
    """Формирует полный канонический текст рецепта для сплошного чанкинга."""
    parts: list[str] = []

    title = recipe.get("title", "").strip()
    if title:
        parts.append(title)

    meta_parts: list[str] = []
    if recipe.get("category"):
        meta_parts.append(f"Категория: {recipe['category']}")
    if recipe.get("meal_type"):
        meta_parts.append(f"Приём пищи: {recipe['meal_type']}")
    if recipe.get("difficulty"):
        meta_parts.append(f"Сложность: {recipe['difficulty']}")
    if recipe.get("preparation_time"):
        meta_parts.append(f"Подготовка: {recipe['preparation_time']}")
    if recipe.get("cooking_time"):
        meta_parts.append(f"Приготовление: {recipe['cooking_time']}")
    if recipe.get("servings"):
        meta_parts.append(f"Порций: {recipe['servings']}")
    if meta_parts:
        parts.append(" | ".join(meta_parts))

    ingredients = recipe.get("ingredients") or []
    if ingredients:
        parts.append("Ингредиенты:\n" + "\n".join(f"- {ing}" for ing in ingredients))

    intro = recipe.get("intro_text", "").strip()
    if intro:
        parts.append(f"Описание:\n{intro}")

    steps = recipe.get("steps") or []
    if steps:
        step_lines: list[str] = []
        for s in steps:
            num = s.get("number", "")
            txt = s.get("text", "").strip()
            temps = s.get("temperatures") or []
            t_str = ""
            if temps:
                t_str = " (" + ", ".join(
                    f"{t.get('type')}: {t.get('value')}" for t in temps if t.get("value")
                ) + ")"
            step_lines.append(f"Шаг {num}: {txt}{t_str}")
        parts.append("Пошаговый рецепт:\n" + "\n".join(step_lines))

    equipment = recipe.get("equipment") or []
    if equipment:
        eq_names = [e.get("name") for e in equipment if isinstance(e, dict) and e.get("name")]
        if eq_names:
            parts.append("Оборудование:\n" + ", ".join(eq_names))

    return "\n\n".join(parts)


# ============================================================================
# 3. Стратегии чанкинга
# ============================================================================


def _find_smart_cut(text: str, target: int, min_cut: int) -> int:
    """Находит ближайшую границу предложения или слова перед target."""
    if target >= len(text):
        return len(text)

    # Приоритет 1: конец предложения/абзаца
    for delim in ("\n\n", ".\n", ". ", "? ", "! ", "\n"):
        pos = text.rfind(delim, min_cut, target)
        if pos != -1:
            return pos + len(delim)

    # Приоритет 2: граница слова (пробел)
    pos = text.rfind(" ", min_cut, target)
    if pos != -1:
        return pos + 1

    return target


def chunk_fixed_size(
    recipes: list[dict[str, Any]],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    source: str = "",
) -> list[Chunk]:
    """Стратегия 1: разбиение по фиксированному размеру окна со сдвигом.

    Разрезает канонический текст рецепта скользящим окном с шагом
    ``(chunk_size - chunk_overlap)``, сохраняя границы слов и предложений.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть строго меньше chunk_size")

    step = chunk_size - chunk_overlap
    min_cut = max(1, step // 2)
    chunks: list[Chunk] = []

    for recipe in recipes:
        recipe_id = recipe.get("id", "unknown")
        title = recipe.get("title", "")
        doc_text = recipe_to_text(recipe)
        if not doc_text:
            continue

        start = 0
        chunk_idx = 0
        text_len = len(doc_text)

        while start < text_len:
            desired_end = min(start + chunk_size, text_len)
            if desired_end == text_len:
                actual_end = text_len
            else:
                actual_end = _find_smart_cut(doc_text, desired_end, start + min_cut)
                if actual_end <= start:
                    actual_end = desired_end

            chunk_text = doc_text[start:actual_end].strip()
            if chunk_text:
                chunk_id = f"fix_{recipe_id}_{chunk_idx:03d}"
                meta = {
                    "strategy": "fixed",
                    "recipe_id": recipe_id,
                    "title": title,
                    "category": recipe.get("category", ""),
                    "meal_type": recipe.get("meal_type", ""),
                    "difficulty": recipe.get("difficulty", ""),
                    "author": recipe.get("author", ""),
                    "url": recipe.get("url", ""),
                    "section": "document",
                    "step_number": None,
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": actual_end,
                    "n_chars": len(chunk_text),
                    "n_tokens": estimate_tokens(chunk_text),
                    "source": source,
                }
                chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=meta))
                chunk_idx += 1

            if actual_end >= text_len:
                break

            # Сдвиг с учётом перекрытия
            start = max(start + 1, actual_end - chunk_overlap)

    return chunks


def chunk_structural(
    recipes: list[dict[str, Any]],
    source: str = "",
) -> list[Chunk]:
    """Стратегия 2: разбиение по логической структуре рецепта.

    Разделы:
    - ``header``: общие параметры рецепта (время, сложность, порции, категория);
    - ``ingredients``: полный список ингредиентов с заголовком;
    - ``intro``: описание блюда и кулинарные особенности;
    - ``step``: каждый пошаговый этап (с номерами и температурным режимом);
    - ``equipment``: перечень необходимого оборудования и грилей.
    """
    chunks: list[Chunk] = []

    for recipe in recipes:
        recipe_id = recipe.get("id", "unknown")
        title = recipe.get("title", "")
        base_meta = {
            "strategy": "structural",
            "recipe_id": recipe_id,
            "title": title,
            "category": recipe.get("category", ""),
            "meal_type": recipe.get("meal_type", ""),
            "difficulty": recipe.get("difficulty", ""),
            "author": recipe.get("author", ""),
            "url": recipe.get("url", ""),
            "source": source,
        }
        chunk_idx = 0

        # 1. Шапка (header)
        header_lines = [f"Рецепт: {title}"]
        if recipe.get("category"):
            header_lines.append(f"Категория: {recipe['category']}")
        if recipe.get("meal_type"):
            header_lines.append(f"Приём пищи: {recipe['meal_type']}")
        if recipe.get("difficulty"):
            header_lines.append(f"Сложность: {recipe['difficulty']}")
        if recipe.get("preparation_time"):
            header_lines.append(f"Время подготовки: {recipe['preparation_time']}")
        if recipe.get("cooking_time"):
            header_lines.append(f"Время приготовления: {recipe['cooking_time']}")
        if recipe.get("servings"):
            header_lines.append(f"Количество порций: {recipe['servings']}")
        if recipe.get("author"):
            header_lines.append(f"Автор: {recipe['author']}")

        header_text = "\n".join(header_lines).strip()
        if header_text:
            meta = dict(base_meta)
            meta.update({
                "section": "header",
                "step_number": None,
                "chunk_index": chunk_idx,
                "char_start": 0,
                "char_end": len(header_text),
                "n_chars": len(header_text),
                "n_tokens": estimate_tokens(header_text),
            })
            chunks.append(
                Chunk(id=f"struct_{recipe_id}_header", text=header_text, metadata=meta)
            )
            chunk_idx += 1

        # 2. Ингредиенты (ingredients)
        ingredients = recipe.get("ingredients") or []
        if ingredients:
            ing_text = (
                f"Ингредиенты для «{title}»:\n"
                + "\n".join(f"- {ing}" for ing in ingredients)
            ).strip()
            meta = dict(base_meta)
            meta.update({
                "section": "ingredients",
                "step_number": None,
                "chunk_index": chunk_idx,
                "ingredient_count": len(ingredients),
                "char_start": 0,
                "char_end": len(ing_text),
                "n_chars": len(ing_text),
                "n_tokens": estimate_tokens(ing_text),
            })
            chunks.append(
                Chunk(id=f"struct_{recipe_id}_ingredients", text=ing_text, metadata=meta)
            )
            chunk_idx += 1

        # 3. Вводный текст (intro)
        intro = recipe.get("intro_text", "").strip()
        if intro:
            intro_text = f"Описание «{title}»:\n{intro}"
            meta = dict(base_meta)
            meta.update({
                "section": "intro",
                "step_number": None,
                "chunk_index": chunk_idx,
                "char_start": 0,
                "char_end": len(intro_text),
                "n_chars": len(intro_text),
                "n_tokens": estimate_tokens(intro_text),
            })
            chunks.append(
                Chunk(id=f"struct_{recipe_id}_intro", text=intro_text, metadata=meta)
            )
            chunk_idx += 1

        # 4. Пошаговые инструкции (steps)
        steps = recipe.get("steps") or []
        for s in steps:
            num = s.get("number", 0)
            txt = s.get("text", "").strip()
            if not txt:
                continue

            temps = s.get("temperatures") or []
            temp_notes: list[str] = []
            for t in temps:
                t_type = t.get("type", "")
                t_val = t.get("value", "")
                if t_type and t_val:
                    temp_notes.append(f"{t_type}: {t_val}")

            temp_suffix = f" [Температура: {'; '.join(temp_notes)}]" if temp_notes else ""
            step_text = f"«{title}», Шаг {num}: {txt}{temp_suffix}"

            meta = dict(base_meta)
            meta.update({
                "section": "step",
                "step_number": num,
                "chunk_index": chunk_idx,
                "has_temperatures": bool(temp_notes),
                "temperatures": temp_notes,
                "char_start": 0,
                "char_end": len(step_text),
                "n_chars": len(step_text),
                "n_tokens": estimate_tokens(step_text),
            })
            chunks.append(
                Chunk(id=f"struct_{recipe_id}_step_{num:02d}", text=step_text, metadata=meta)
            )
            chunk_idx += 1

        # 5. Оборудование (equipment)
        equipment = recipe.get("equipment") or []
        if equipment:
            eq_names = [e.get("name") for e in equipment if isinstance(e, dict) and e.get("name")]
            if eq_names:
                eq_text = f"Оборудование для «{title}»:\n" + ", ".join(eq_names)
                meta = dict(base_meta)
                meta.update({
                    "section": "equipment",
                    "step_number": None,
                    "chunk_index": chunk_idx,
                    "equipment_count": len(eq_names),
                    "char_start": 0,
                    "char_end": len(eq_text),
                    "n_chars": len(eq_text),
                    "n_tokens": estimate_tokens(eq_text),
                })
                chunks.append(
                    Chunk(id=f"struct_{recipe_id}_equipment", text=eq_text, metadata=meta)
                )
                chunk_idx += 1

    return chunks


# ============================================================================
# 4. Генерация эмбеддингов
# ============================================================================


@runtime_checkable
class Embedder(Protocol):
    """Интерфейс генератора эмбеддингов."""

    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Возвращает массив векторов размера (len(texts), dim) с L2-нормой 1."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Возвращает один вектор размера (dim,) с L2-нормой 1."""
        ...


class HashingEmbedder:
    """Детерминированный офлайн-эмбеддер на основе хеширования признаков.

    Извлекает:
    - униграммы слов (вес 1.0)
    - биграммы слов (вес 1.5)
    - символьные 3-граммы (вес 0.5) и 4-граммы (вес 0.3) для учёта окончаний кириллицы

    Хеширует каждый признак через blake2b в диапазон [0, dim-1] с псевдослучайным
    знаком (+1/-1), затем нормализует вектор по L2 (inner product = косинусное сходство).

    Работает полностью офлайн, мгновенно, без внешних весов и с нулевой вероятностью
    дрейфа между запусками.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim должен быть положительным")
        self.dim = dim
        self.name = f"hashing_{dim}d"

    def _extract_features(self, text: str) -> list[tuple[str, float]]:
        text_norm = text.lower()
        words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text_norm)
        features: list[tuple[str, float]] = []

        # Униграммы слов и n-граммы символов
        for w in words:
            features.append((f"w:{w}", 1.0))
            w_len = len(w)
            if w_len >= 3:
                for i in range(w_len - 2):
                    features.append((f"c3:{w[i:i + 3]}", 0.5))
            if w_len >= 4:
                for i in range(w_len - 3):
                    features.append((f"c4:{w[i:i + 4]}", 0.3))

        # Биграммы слов
        for i in range(len(words) - 1):
            features.append((f"bg:{words[i]}_{words[i + 1]}", 1.5))

        return features

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        features = self._extract_features(text)
        if not features:
            return vec

        for feat, weight in features:
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            matrix[i] = self.embed_one(text)
        return matrix


class OpenAIEmbedder:
    """Эмбеддер через OpenAI-совместимый API (/embeddings)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openai.bothub.chat/v1",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.name = f"openai_{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        body = {
            "model": self.model,
            "input": texts,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        data = payload.get("data", [])
        data_sorted = sorted(data, key=lambda x: x.get("index", 0))
        matrix = np.array([item["embedding"] for item in data_sorted], dtype=np.float32)

        # L2-нормализация
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ============================================================================
# 5. Индекс FAISS с хранением метаданных
# ============================================================================


class FaissIndex:
    """Локальный векторный индекс FAISS (IndexFlatIP) с сохранением метаданных."""

    def __init__(
        self,
        index: faiss.IndexFlatIP,
        chunks: list[Chunk],
        embedder: Embedder,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.metadata = dict(metadata or {})

    @property
    def total(self) -> int:
        return self.index.ntotal

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        metadata: dict[str, Any] | None = None,
    ) -> FaissIndex:
        """Строит FAISS-индекс по списку чанков."""
        if not chunks:
            index = faiss.IndexFlatIP(embedder.dim)
            return cls(index=index, chunks=[], embedder=embedder, metadata=metadata)

        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        index = faiss.IndexFlatIP(embedder.dim)
        index.add(vectors)

        meta = dict(metadata or {})
        meta.setdefault("created_at", time.time())
        meta.setdefault("dim", embedder.dim)
        meta.setdefault("embedder", embedder.name)
        meta.setdefault("total_chunks", len(chunks))

        return cls(index=index, chunks=list(chunks), embedder=embedder, metadata=meta)

    def save(self, directory: str | Path) -> Path:
        """Сохраняет FAISS-индекс и сопутствующие файлы в директорию:

        - ``faiss.index`` — бинарный файл FAISS;
        - ``chunks.json`` — полный список чанков с текстом и метаданными;
        - ``index_meta.json`` — метаданные индекса (размерность, эмбеддер, статистика).
        """
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "faiss.index"
        chunks_file = dir_path / "chunks.json"
        meta_file = dir_path / "index_meta.json"

        faiss.write_index(self.index, str(index_file))

        with chunks_file.open("w", encoding="utf-8") as file:
            json.dump([c.to_dict() for c in self.chunks], file, ensure_ascii=False, indent=2)

        meta_data = dict(self.metadata)
        meta_data.update({
            "saved_at": time.time(),
            "total_chunks": len(self.chunks),
            "dim": self.embedder.dim,
            "embedder": self.embedder.name,
            "index_file": str(index_file.name),
            "chunks_file": str(chunks_file.name),
        })
        with meta_file.open("w", encoding="utf-8") as file:
            json.dump(meta_data, file, ensure_ascii=False, indent=2)

        return dir_path

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> FaissIndex:
        """Загружает индекс и метаданные из директории."""
        dir_path = Path(directory).resolve()
        index_file = dir_path / "faiss.index"
        chunks_file = dir_path / "chunks.json"
        meta_file = dir_path / "index_meta.json"

        if not index_file.exists():
            raise FileNotFoundError(f"Файл индекса FAISS не найден: {index_file}")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Файл чанков не найден: {chunks_file}")

        index = faiss.read_index(str(index_file))

        with chunks_file.open(encoding="utf-8") as file:
            chunks_raw = json.load(file)
        chunks = [Chunk.from_dict(item) for item in chunks_raw]

        meta: dict[str, Any] = {}
        if meta_file.exists():
            with meta_file.open(encoding="utf-8") as file:
                meta = json.load(file)

        return cls(index=index, chunks=chunks, embedder=embedder, metadata=meta)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Ищет ближайшие чанки по запросу.

        Возвращает список результатов с рангом, косинусным сходством (score),
        текстом чанка и его полными метаданными.
        """
        if self.total == 0:
            return []

        k = min(top_k, self.total)
        query_vec = self.embedder.embed_one(query).reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        distances, indices = self.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
            })
        return results


# ============================================================================
# 6. Сравнение стратегий chunking
# ============================================================================


def compute_chunking_stats(chunks: list[Chunk]) -> dict[str, Any]:
    """Вычисляет статистику распределения размеров и разделов чанков."""
    if not chunks:
        return {
            "count": 0,
            "total_chars": 0,
            "total_tokens": 0,
            "avg_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
            "std_chars": 0.0,
            "avg_tokens": 0.0,
            "sections": {},
        }

    char_lens = [len(c.text) for c in chunks]
    token_lens = [c.metadata.get("n_tokens", estimate_tokens(c.text)) for c in chunks]

    n = len(char_lens)
    total_chars = sum(char_lens)
    avg_chars = total_chars / n
    variance = sum((x - avg_chars) ** 2 for x in char_lens) / n
    std_chars = math.sqrt(variance)

    sections: dict[str, int] = {}
    for c in chunks:
        sec = str(c.metadata.get("section", "unknown"))
        sections[sec] = sections.get(sec, 0) + 1

    return {
        "count": n,
        "total_chars": total_chars,
        "total_tokens": sum(token_lens),
        "avg_chars": round(avg_chars, 1),
        "min_chars": min(char_lens),
        "max_chars": max(char_lens),
        "std_chars": round(std_chars, 1),
        "avg_tokens": round(sum(token_lens) / n, 1),
        "sections": sections,
    }


# Набор контрольных запросов для оценки качества поиска
DEFAULT_PROBE_QUERIES: list[dict[str, Any]] = [
    {
        "query": "Апельсиновая курица на гриле рецепт для праздника с розмарином",
        "target_recipe_ids": [15454],
        "expected_section": "intro",
        "description": "Поиск по общему описанию праздничного блюда из птицы",
    },
    {
        "query": "Сколько соли перца и горчицы нужно для обсыпки прайм риб",
        "target_recipe_ids": [15452],
        "expected_section": "ingredients",
        "description": "Поиск точного состава ингредиентов крупного мясного блюда",
    },
    {
        "query": "Каре барашка гарнир гречневая каша кедровые орехи брусника",
        "target_recipe_ids": [15418],
        "expected_section": "ingredients",
        "description": "Поиск блюда по специфическим ингредиентам гарнира",
    },
    {
        "query": "Маринад из красного вина бальзамического уксуса и чеснока для Лондон бройл",
        "target_recipe_ids": [15383],
        "expected_section": "step",
        "description": "Поиск конкретного кулинарного процесса (маринование стейка)",
    },
    {
        "query": "Митболы из свиного фарша температура приготовления на гриле",
        "target_recipe_ids": [15360],
        "expected_section": "step",
        "description": "Поиск параметров температурного режима и времени копчения",
    },
    {
        "query": "Пикантный горчичный соус для свинины шрирача мед яблочный уксус",
        "target_recipe_ids": [15362],
        "expected_section": "ingredients",
        "description": "Поиск соуса по полному перечню приправ",
    },
]


def evaluate_retrieval(
    index: FaissIndex,
    queries: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Оценивает качество поиска по тестовым запросам.

    Метрики:
    - ``recall_at_1``: доля запросов, где целевой рецепт оказался на 1 месте;
    - ``recall_at_k``: доля запросов, где целевой рецепт попал в top-k;
    - ``mrr_at_k``: Mean Reciprocal Rank (средний обратный ранг);
    - ``avg_top_score``: средний косинусный скор первого результата;
    - ``section_accuracy``: доля попаданий в ожидаемый раздел (для structural).
    """
    if not queries or index.total == 0:
        return {
            "total_queries": 0,
            "recall_at_1": 0.0,
            f"recall_at_{top_k}": 0.0,
            f"mrr_at_{top_k}": 0.0,
            "avg_top_score": 0.0,
            "section_accuracy": 0.0,
            "details": [],
        }

    hits_top1 = 0
    hits_topk = 0
    reciprocal_ranks: list[float] = []
    top_scores: list[float] = []
    section_hits = 0
    details: list[dict[str, Any]] = []

    for item in queries:
        q = item["query"]
        target_ids = set(item.get("target_recipe_ids", []))
        expected_sec = item.get("expected_section")

        results = index.search(q, top_k=top_k)
        if not results:
            reciprocal_ranks.append(0.0)
            details.append({
                "query": q,
                "found": False,
                "rank": None,
                "score": 0.0,
                "top_title": "",
            })
            continue

        top_scores.append(results[0]["score"])

        found_rank: int | None = None
        for res in results:
            res_recipe_id = res["metadata"].get("recipe_id")
            if res_recipe_id in target_ids:
                found_rank = res["rank"]
                break

        if found_rank == 1:
            hits_top1 += 1
        if found_rank is not None:
            hits_topk += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Проверка соответствия раздела для structural
        top_sec = results[0]["metadata"].get("section")
        sec_matched = (expected_sec is None) or (top_sec == expected_sec)
        if sec_matched:
            section_hits += 1

        details.append({
            "query": q,
            "target_ids": list(target_ids),
            "found_rank": found_rank,
            "top_score": round(results[0]["score"], 4),
            "top_title": results[0]["metadata"].get("title", ""),
            "top_section": top_sec,
            "expected_section": expected_sec,
        })

    n = len(queries)
    return {
        "total_queries": n,
        "recall_at_1": round(hits_top1 / n, 3),
        f"recall_at_{top_k}": round(hits_topk / n, 3),
        f"mrr_at_{top_k}": round(sum(reciprocal_ranks) / n, 3),
        "avg_top_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0,
        "section_accuracy": round(section_hits / n, 3),
        "details": details,
    }


def compare_strategies(
    recipes: list[dict[str, Any]],
    source_name: str = "recipes",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    embedder: Embedder | None = None,
    top_k: int = 5,
    queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Сравнивает две стратегии chunking (fixed vs structural).

    Возвращает словарь с:
    - списком чанков обеих стратегий;
    - построенными FAISS-индексами;
    - структурными метриками (размеры, распределения);
    - метриками качества поиска (Recall@1, Recall@k, MRR@k);
    - сводной сравнительной таблицей.
    """
    if embedder is None:
        embedder = HashingEmbedder(dim=256)
    if queries is None:
        queries = DEFAULT_PROBE_QUERIES

    t0 = time.time()
    fixed_chunks = chunk_fixed_size(
        recipes, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=source_name
    )
    t_fixed_chunk = time.time() - t0

    t0 = time.time()
    struct_chunks = chunk_structural(recipes, source=source_name)
    t_struct_chunk = time.time() - t0

    fixed_stats = compute_chunking_stats(fixed_chunks)
    struct_stats = compute_chunking_stats(struct_chunks)

    # Построение индексов FAISS
    t0 = time.time()
    fixed_index = FaissIndex.build(
        fixed_chunks,
        embedder=embedder,
        metadata={"strategy": "fixed", "source": source_name, "chunk_size": chunk_size},
    )
    t_fixed_index = time.time() - t0

    t0 = time.time()
    struct_index = FaissIndex.build(
        struct_chunks,
        embedder=embedder,
        metadata={"strategy": "structural", "source": source_name},
    )
    t_struct_index = time.time() - t0

    # Оценка качества поиска
    fixed_retrieval = evaluate_retrieval(fixed_index, queries, top_k=top_k)
    struct_retrieval = evaluate_retrieval(struct_index, queries, top_k=top_k)

    return {
        "source_name": source_name,
        "total_recipes": len(recipes),
        "embedder": embedder.name,
        "dim": embedder.dim,
        "top_k": top_k,
        "fixed": {
            "chunks": fixed_chunks,
            "index": fixed_index,
            "stats": fixed_stats,
            "retrieval": fixed_retrieval,
            "chunking_time_ms": round(t_fixed_chunk * 1000, 2),
            "indexing_time_ms": round(t_fixed_index * 1000, 2),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "structural": {
            "chunks": struct_chunks,
            "index": struct_index,
            "stats": struct_stats,
            "retrieval": struct_retrieval,
            "chunking_time_ms": round(t_struct_chunk * 1000, 2),
            "indexing_time_ms": round(t_struct_index * 1000, 2),
        },
    }


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Формирует понятный Markdown-отчёт со сравнением двух стратегий."""
    fixed = report["fixed"]
    struct = report["structural"]
    f_stats = fixed["stats"]
    s_stats = struct["stats"]
    f_ret = fixed["retrieval"]
    s_ret = struct["retrieval"]
    top_k = report.get("top_k", 5)

    lines: list[str] = [
        f"# Сравнение стратегий чанкинга для «{report.get('source_name')}»",
        "",
        f"- **Всего рецептов в источнике:** {report.get('total_recipes')}",
        f"- **Эмбеддер:** `{report.get('embedder')}` (размерность {report.get('dim')})",
        f"- **Тестовых запросов:** {f_ret.get('total_queries')}",
        "",
        "## 1. Сводная таблица метрик",
        "",
        "| Метрика | Фиксированный размер (fixed) | По структуре (structural) | Разница / Вывод |",
        "| --- | --- | --- | --- |",
        f"| Количество чанков | {f_stats['count']} | {s_stats['count']} | Структурных на {s_stats['count'] - f_stats['count']} больше |",
        f"| Средняя длина чанка (симв.) | {f_stats['avg_chars']} | {s_stats['avg_chars']} | Структурные компактнее и атомарнее |",
        f"| Мин / Макс длина (симв.) | {f_stats['min_chars']} / {f_stats['max_chars']} | {s_stats['min_chars']} / {s_stats['max_chars']} | Fixed строго ограничен окном |",
        f"| Стандартное отклонение длины | {f_stats['std_chars']} | {s_stats['std_chars']} | Структурные варьируются по типу раздела |",
        f"| Всего токенов (оценка) | {f_stats['total_tokens']} | {s_stats['total_tokens']} | За счёт перекрытий и префиксов контекста |",
        f"| Recall@1 (первый результат) | **{f_ret.get('recall_at_1')}** | **{s_ret.get('recall_at_1')}** | {'Структурный точнее' if s_ret.get('recall_at_1', 0) >= f_ret.get('recall_at_1', 0) else 'Fixed выше'} |",
        f"| Recall@{top_k} (попадание в top-{top_k}) | **{f_ret.get(f'recall_at_{top_k}')}** | **{s_ret.get(f'recall_at_{top_k}')}** | Полнота выборки |",
        f"| MRR@{top_k} (Mean Reciprocal Rank) | **{f_ret.get(f'mrr_at_{top_k}')}** | **{s_ret.get(f'mrr_at_{top_k}')}** | Ранг правильного рецепта |",
        f"| Средний score топ-1 | {f_ret.get('avg_top_score')} | {s_ret.get('avg_top_score')} | Косинусное сходство первого кандидата |",
        "",
        "## 2. Распределение чанков по разделам (structural)",
        "",
    ]

    for sec, cnt in sorted(s_stats.get("sections", {}).items(), key=lambda x: -x[1]):
        lines.append(f"- **{sec}**: {cnt} чанков")

    lines.extend([
        "",
        "## 3. Анализ тестовых запросов",
        "",
        "| Запрос | Ожидался рецепт | Ранг (Fixed) | Ранг (Structural) | Раздел топа (Structural) |",
        "| --- | --- | --- | --- | --- |",
    ])

    f_details = {d["query"]: d for d in f_ret.get("details", [])}
    for s_item in s_ret.get("details", []):
        q = s_item["query"]
        f_item = f_details.get(q, {})
        target = ",".join(map(str, s_item.get("target_ids", [])))
        f_rank = f_item.get("found_rank") or "—"
        s_rank = s_item.get("found_rank") or "—"
        s_sec = s_item.get("top_section", "—")
        lines.append(
            f"| {q[:45]}… | {target} | {f_rank} | {s_rank} | `{s_sec}` |"
        )

    lines.extend([
        "",
        "## 4. Выводы",
        "",
        "1. **Атомарность информации:** Структурный чанкинг выделяет каждый шаг, список ингредиентов "
        "и вступление в самостоятельный чанк. В отличие от фиксированного окна, здесь исключены "
        "«склеивания» разнородных блоков и разрезание предложений.",
        "2. **Точность метаданных:** Метаданные структурного чанка содержат поле `section` "
        "(`ingredients`, `step`, `intro`, `header`, `equipment`), что позволяет фильтровать поиск "
        "по типу сущности (например, искать только по ингредиентам или только по шагам с температурой).",
        "3. **Практическая рекомендация:** Для структурированных документов (рецепты, спецификации, "
        "инструкции) стратегия `structural` предпочтительнее фиксированного окна.",
    ])

    return "\n".join(lines)
