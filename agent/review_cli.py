"""День 32: CLI автоматического ревью pull request.

Точка входа для пайплайна (GitHub Action) и локального запуска: собирает
контекст PR через MCP, прогоняет ассистента ревью (RAG по документации и коду),
печатает и сохраняет текст ревью, при необходимости публикует комментарий к PR.

Запуск::

    python3 review_pr.py --event "$GITHUB_EVENT_PATH" --post-comment
    python3 review_pr.py --base HEAD~1 --head HEAD --offline
    python3 review_pr.py --base day31 --head day32 --offline --output review.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .code_index import (
    DEFAULT_CODE_DATASET,
    DEFAULT_CODE_STRATEGY,
)
from .code_review import (
    CATEGORIES,
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    CodeReview,
    build_review_assistant,
    build_review_config,
    has_llm_credentials,
)
from .config import load_config
from .project_docs import (
    DEFAULT_DATASET as DEFAULT_DOCS_DATASET,
    DEFAULT_INDEX_DIR as DEFAULT_DOCS_INDEX_DIR,
    DEFAULT_STRATEGY as DEFAULT_DOCS_STRATEGY,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = "history/code_review"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_event(path: str | None) -> dict:
    """Загружает payload GitHub-события (если путь задан и файл существует)."""
    if not path:
        return {}
    event_path = Path(path)
    if not event_path.exists():
        return {}
    try:
        return json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def event_pr_info(event: dict) -> dict:
    """Извлекает из события PR нужные для запуска и публикации поля."""
    pr = (event or {}).get("pull_request") or {}
    repo = (event or {}).get("repository") or {}
    if not pr:
        return {}
    base = pr.get("base") or {}
    head = pr.get("head") or {}
    user = pr.get("user") or {}
    return {
        "number": pr.get("number"),
        "title": pr.get("title") or "",
        "author": user.get("login") or "",
        "url": pr.get("html_url") or "",
        "repository": repo.get("full_name") or "",
        "baseSha": base.get("sha") or "",
        "headSha": head.get("sha") or "",
    }


def post_pr_comment(body: str, repo: str, number: int, token: str) -> tuple[bool, str]:
    """Публикует комментарий к PR через GitHub REST API (без внешних зависимостей)."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}/comments"
    payload = json.dumps({"body": body}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "ai-advent-review-bot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return False, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def should_fail(review: CodeReview, fail_on: str) -> bool:
    """Решает, должен ли пайплайн упасть по итогам ревью."""
    if fail_on == "none":
        return False
    severities = {finding.severity for finding in review.findings}
    if fail_on == "critical":
        return SEVERITY_CRITICAL in severities
    if fail_on == "major":
        return bool(severities & {SEVERITY_CRITICAL, SEVERITY_MAJOR})
    return False


def build_review_header(review: CodeReview, mode: str) -> str:
    counts = review.counts()
    parts = ", ".join(f"{category}: {counts[category]}" for category in CATEGORIES)
    return (
        f"Режим: {mode}; диапазон: {review.base} → {review.head}; "
        f"файлов: {len(review.changed_files)}; находки: {parts}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="День 32: автоматическое ревью PR (RAG по документации и коду)"
    )
    parser.add_argument("--base", default=None, help="база сравнения (SHA/ветка)")
    parser.add_argument("--head", default=None, help="верхняя ревизия (SHA/ветка)")
    parser.add_argument("--event", default=None, help="путь к payload GitHub-события")
    parser.add_argument("--pr-number", type=int, default=None, help="номер PR для публикации")
    parser.add_argument("--offline", "--no-llm", dest="offline", action="store_true",
                        help="принудительно использовать офлайн-эвристики")
    parser.add_argument("--output", default=None, help="записать markdown-ревью в файл")
    parser.add_argument("--json", default=None, help="записать результат ревью в JSON")
    parser.add_argument("--post-comment", action="store_true", help="опубликовать комментарий к PR")
    parser.add_argument("--fail-on", choices=("none", "critical", "major"), default="none",
                        help="при какой серьёзности находок пайплайн падает")
    parser.add_argument("--top-k", type=int, default=5, help="сколько RAG-источников брать")
    parser.add_argument("--min-score", type=float, default=0.0, help="минимальное сходство чанка")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--max-diff-chars", type=int, default=20000)
    parser.add_argument("--index-dir", default=DEFAULT_DOCS_INDEX_DIR, help="каталог индексов")
    parser.add_argument("--docs-dataset", default=DEFAULT_DOCS_DATASET)
    parser.add_argument("--code-dataset", default=DEFAULT_CODE_DATASET)
    parser.add_argument("--strategy", default=DEFAULT_DOCS_STRATEGY)
    parser.add_argument("--code-strategy", default=DEFAULT_CODE_STRATEGY)
    parser.add_argument("--no-mcp", action="store_true", help="не подключать MCP-сервер ревью")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--skip-save", action="store_true", help="не сохранять отчёт")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = load_config()
    except (OSError, ValueError, json.JSONDecodeError):
        # В CI файла config.one.json нет — конфиг берётся из переменных окружения.
        config = {}

    event = load_event(args.event)
    pr_info = event_pr_info(event)

    base = args.base or pr_info.get("baseSha") or "HEAD"
    head = args.head
    if head is None:
        head = pr_info.get("headSha") or ""

    llm_config = build_review_config(config)
    if args.offline:
        offline = True
    elif has_llm_credentials(llm_config):
        offline = False
    else:
        offline = True
        print("⚠️  LLM-ключ не найден — использую офлайн-эвристики.", file=sys.stderr)

    mode = "офлайн-эвристики" if offline else f"живой LLM ({llm_config.get('model')})"
    print("=" * 78)
    print("🔍 ДЕНЬ 32: АВТОМАТИЧЕСКОЕ РЕВЬЮ PR (RAG: ДОКУМЕНТАЦИЯ + КОД)")
    print("=" * 78)
    print(f"Диапазон: {base} → {head or 'working tree'}")
    print(f"Режим: {mode}")
    if pr_info.get("number"):
        print(f"PR #{pr_info['number']}: {pr_info.get('title', '')}")
    print()

    try:
        assistant = build_review_assistant(
            config=config,
            offline=offline,
            docs_index_dir=args.index_dir,
            docs_dataset=args.docs_dataset,
            docs_strategy=args.strategy,
            code_index_dir=args.index_dir,
            code_dataset=args.code_dataset,
            code_strategy=args.code_strategy,
            top_k=args.top_k,
            min_score=args.min_score,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            max_diff_chars=args.max_diff_chars,
            with_review_mcp=not args.no_mcp,
        )
    except Exception as exc:  # noqa: BLE001 — понятная ошибка инициализации
        print(f"✗ Ошибка инициализации ревью: {exc}", file=sys.stderr)
        return 1

    review = assistant.review(base=base, head=head, pr=pr_info or None)
    markdown = review.to_markdown()
    print(markdown)
    print("\n" + build_review_header(review, mode))

    if review.error:
        print(f"⚠️  Ревью завершилось с ошибкой: {review.error}", file=sys.stderr)

    # Артефакты: явные пути + отчёт в history/.
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"✓ Markdown-ревью сохранено: {output_path}")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": now_iso(),
            "mode": "offline" if offline else "live",
            "base": base,
            "head": head,
            "pr": pr_info,
            "review": review.to_dict(),
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ JSON-результат сохранён: {json_path}")

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    if not args.skip_save:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "review_report.md").write_text(markdown, encoding="utf-8")
        (report_dir / "review_results.json").write_text(
            json.dumps(
                {
                    "generated_at": now_iso(),
                    "mode": "offline" if offline else "live",
                    "base": base,
                    "head": head,
                    "pr": pr_info,
                    "review": review.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"✓ Отчёт сохранён: {report_dir / 'review_report.md'}")

    # Публикация комментария к PR.
    if args.post_comment:
        repo = pr_info.get("repository") or os.environ.get("GITHUB_REPOSITORY", "")
        number = args.pr_number or pr_info.get("number")
        token = os.environ.get("GITHUB_TOKEN", "")
        if not (repo and number and token):
            print(
                "⚠️  Публикация пропущена: нужны GITHUB_TOKEN, репозиторий и номер PR.",
                file=sys.stderr,
            )
        else:
            ok, detail = post_pr_comment(markdown, repo, int(number), token)
            if ok:
                print(f"✓ Комментарий опубликован в PR #{number} ({detail})")
            else:
                print(f"⚠️  Не удалось опубликовать комментарий: {detail}", file=sys.stderr)

    fail = should_fail(review, args.fail_on)
    print("=" * 78)
    print(f"ИТОГ: {'есть блокирующие находки ❌' if fail else 'ревью завершено ✅'}")
    print("=" * 78)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
