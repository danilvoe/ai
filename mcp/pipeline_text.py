#!/usr/bin/env python3
"""Day 19: локальная экстрактивная суммаризация для MCP-инструмента ``summarize``.

Модуль выделяет обработку текста из MCP-сервера (``mcp/pipeline_server.py``) в
отдельное место, чтобы её можно было проверять детерминированно, без сети и
без вызова LLM.

Алгоритм простой и воспроизводимый (экстрактивный):

1. Каждый элемент (проект GitFlic) превращается в текст: название, описание,
   язык, владелец, топики.
2. Текст режется на предложения; слова приводятся к нижнему регистру, из них
   убираются стоп-слова (русские и английские).
3. Предложения оцениваются суммой частот входящих слов; выбираются лучшие
   ``max_sentences`` и возвращаются в исходном порядке.
4. Отдельно собираются ключевые слова (топ по частоте) и краткие пункты по
   каждому элементу.

Так ``summarize`` не зависит от внешних сервисов: один и тот же вход всегда
даёт один и тот же выход, что удобно для проверки передачи данных в пайплайне.
"""

import re
from collections import Counter

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]{3,}")

STOPWORDS = {
    # русские
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    # английские
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "may", "new", "now", "old", "see", "two", "way", "who",
    "boy", "did", "use", "that", "with", "this", "from", "they", "will",
    "have", "been", "were", "said", "each", "which", "their", "them", "than",
    "some", "into", "more", "very", "when", "what", "your", "about", "would",
    "there", "could", "other", "these", "then", "also", "using", "used",
}


def tokenize(text: str) -> list[str]:
    """Слова текста в нижнем регистре без стоп-слов и коротких токенов."""
    words = WORD_RE.findall((text or "").lower())
    return [word for word in words if word not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения и отбрасывает пустые."""
    parts = [part.strip() for part in SENTENCE_SPLIT.split(text or "")]
    return [part for part in parts if part]


def item_to_text(item: dict) -> str:
    """Текст проекта для суммаризации: название, описание, язык, владелец, топики."""
    parts: list[str] = []
    title = (item.get("title") or "").strip()
    description = (item.get("description") or "").strip()
    language = (item.get("language") or "").strip()
    owner = ((item.get("owner") or {}).get("alias") or "").strip()
    topics = [str(topic).strip() for topic in (item.get("topics") or []) if str(topic).strip()]
    if title:
        parts.append(title + ".")
    if description:
        parts.append(description)
    meta = []
    if language:
        meta.append(f"язык: {language}")
    if owner:
        meta.append(f"владелец: {owner}")
    if topics:
        meta.append("топики: " + ", ".join(topics))
    if meta:
        parts.append("(" + "; ".join(meta) + ").")
    return " ".join(parts)


def item_bullet(item: dict, max_topics: int = 4) -> str:
    """Краткий пункт по проекту: название, язык, владелец, топики."""
    title = (item.get("title") or item.get("alias") or f"project#{item.get('id')}").strip()
    bits = []
    language = (item.get("language") or "").strip()
    owner = ((item.get("owner") or {}).get("alias") or "").strip()
    if language:
        bits.append(language)
    if owner:
        bits.append(f"@{owner}")
    topics = [str(topic).strip() for topic in (item.get("topics") or []) if str(topic).strip()]
    if topics:
        bits.append(", ".join(topics[:max_topics]))
    suffix = f" — {'; '.join(bits)}" if bits else ""
    return f"{title}{suffix}"


def top_keywords(text: str, limit: int = 8) -> list[str]:
    """Ключевые слова: самые частые значимые слова текста."""
    counts = Counter(tokenize(text))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ordered[: max(0, limit)]]


def summarize_text(
    text: str, max_sentences: int = 3, max_keywords: int = 8
) -> dict:
    """Экстрактивная выжимка из текста: лучшие предложения и ключевые слова."""
    sentences = split_sentences(text)
    words = tokenize(text)
    frequencies = Counter(words)
    if not sentences:
        return {
            "summary": "",
            "keywords": top_keywords(text, max_keywords),
            "sentenceCount": 0,
            "wordCount": len(words),
        }
    if not frequencies:
        # Нет значимых слов — берём первые предложения как есть.
        selected = sentences[: max(1, max_sentences)]
        return {
            "summary": " ".join(selected),
            "keywords": [],
            "sentenceCount": len(sentences),
            "wordCount": len(words),
        }

    scored = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        score = sum(frequencies[token] for token in tokens) / (len(tokens) ** 0.5)
        scored.append((score, index, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen_indices = sorted(index for _, index, _ in scored[: max(1, max_sentences)])
    summary = " ".join(sentences[index] for index in chosen_indices)
    return {
        "summary": summary,
        "keywords": top_keywords(text, max_keywords),
        "sentenceCount": len(sentences),
        "wordCount": len(words),
    }


def summarize_items(
    items: list[dict],
    title: str = "",
    max_sentences: int = 3,
    max_keywords: int = 8,
    max_bullets: int = 10,
) -> dict:
    """Суммаризует список проектов GitFlic: выжимка, пункты, ключевые слова."""
    items = [item for item in (items or []) if isinstance(item, dict)]
    combined = "\n".join(item_to_text(item) for item in items)
    if title:
        combined = f"{title}.\n{combined}"
    base = summarize_text(combined, max_sentences=max_sentences, max_keywords=max_keywords)
    bullets = [item_bullet(item) for item in items[:max_bullets]]
    return {
        "summary": base["summary"],
        "bullets": bullets,
        "keywords": base["keywords"],
        "itemCount": len(items),
        "sentenceCount": base["sentenceCount"],
        "wordCount": base["wordCount"],
    }
