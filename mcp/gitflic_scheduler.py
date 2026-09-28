#!/usr/bin/env python3
"""Day 18: планировщик и фоновые задачи вокруг публичного API GitFlic.

Модуль — ядро MCP-инструмента с отложенным и периодическим выполнением.
Он считает количество публичных проектов GitFlic, агрегирует результат
(всего, по языкам, по владельцам) и сохраняет данные в JSON.

Основные части:

* :class:`CountStore` — JSON-хранилище снимков (история подсчётов);
* :class:`JobStore` — JSON-хранилище расписаний (задачи планировщика);
* :func:`aggregate_projects` — агрегация списка проектов;
* :class:`ProjectCounter` — один подсчёт ``всего → по страницам → агрегат``
  с сохранением снимка;
* :class:`ProjectCountScheduler` — отложенные и периодические задачи.

Сами задачи хранятся в JSON, поэтому переживают перезапуск процесса. Их
выполняет либо внешний ``cron`` (``python3 mcp/gitflic_cron.py --run-due``),
либо поток-демон внутри долгоживущего процесса (``start_daemon``). Так
MCP-сервер, у которого сессия живёт ровно один вызов инструмента, может
поставить задачу «подсчитай через час», а cron — выполнить её.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from gitflic_api import fetch_public_projects

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = PROJECT_ROOT / "history" / "gitflic_counts.json"
DEFAULT_JOBS = PROJECT_ROOT / "history" / "gitflic_jobs.json"
DEFAULT_PAGE_SIZE = 100
# По умолчанию обходим несколько страниц — этого достаточно для выборки, а
# точное число публичных проектов берём из пагинации API на первой странице.
DEFAULT_MAX_PAGES = 5


def now_iso() -> str:
    """Текущее время в ISO-8601 с локальным часовым поясом."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _future_iso(seconds: float) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=seconds))
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_path(path: str | Path | None, env_var: str, default: Path) -> Path:
    if path:
        return Path(path)
    return Path(os.environ.get(env_var, str(default)))


class _JsonStore:
    """Небольшое JSON-хранилище с атомарной записью (общая база для хранилищ)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {"createdAt": now_iso()}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"createdAt": now_iso(), "recovered": True}
        if not isinstance(data, dict):
            return {"createdAt": now_iso()}
        data.setdefault("createdAt", now_iso())
        return data

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


class CountStore(_JsonStore):
    """JSON-хранилище снимков подсчёта публичных проектов.

    Формат файла: ``{"createdAt": ..., "updatedAt": ..., "snapshots": [...]}``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(_resolve_path(path, "GITFLIC_COUNT_STORE", DEFAULT_STORE))

    def load(self) -> dict:
        data = super().load()
        data.setdefault("snapshots", [])
        return data

    def append(self, snapshot: dict) -> dict:
        data = self.load()
        data["snapshots"].append(snapshot)
        data["updatedAt"] = snapshot.get("countedAt") or now_iso()
        data["snapshotCount"] = len(data["snapshots"])
        self.save(data)
        return data

    def latest(self) -> dict | None:
        snapshots = self.load().get("snapshots") or []
        return snapshots[-1] if snapshots else None

    def history(self, limit: int = 20) -> list[dict]:
        snapshots = self.load().get("snapshots") or []
        if limit and limit > 0:
            snapshots = snapshots[-limit:]
        return [
            {
                "countedAt": snap.get("countedAt"),
                "totalPublicProjects": snap.get("totalPublicProjects"),
                "query": snap.get("query", ""),
                "jobId": snap.get("jobId"),
                "trigger": snap.get("trigger"),
            }
            for snap in snapshots
        ]


class JobStore(_JsonStore):
    """JSON-хранилище расписаний планировщика (задач подсчёта)."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(_resolve_path(path, "GITFLIC_JOBS_STORE", DEFAULT_JOBS))

    def load(self) -> dict:
        data = super().load()
        data.setdefault("jobs", {})
        return data

    def put(self, job: dict) -> None:
        data = self.load()
        data["jobs"][job["jobId"]] = job
        data["updatedAt"] = now_iso()
        self.save(data)

    def get(self, job_id: str) -> dict | None:
        return self.load().get("jobs", {}).get(job_id)

    def all(self) -> list[dict]:
        return list((self.load().get("jobs") or {}).values())


def aggregate_projects(projects: list[dict]) -> dict:
    """Агрегирует список проектов: итог, языки, владельцы, топики."""
    languages: dict[str, int] = {}
    owners: dict[str, int] = {}
    topics: dict[str, int] = {}
    with_description = 0
    for project in projects:
        language = (project.get("language") or "").strip() or "не указан"
        languages[language] = languages.get(language, 0) + 1
        owner = ((project.get("owner") or {}).get("alias") or "").strip() or "не указан"
        owners[owner] = owners.get(owner, 0) + 1
        for topic in project.get("topics") or []:
            topics[topic] = topics.get(topic, 0) + 1
        if (project.get("description") or "").strip():
            with_description += 1

    def _top(mapping: dict[str, int], limit: int) -> list[dict]:
        ordered = sorted(mapping.items(), key=lambda item: (-item[1], item[0]))
        return [{"name": name, "count": count} for name, count in ordered[:limit]]

    return {
        "totalPublicProjects": len(projects),
        "languages": dict(
            sorted(languages.items(), key=lambda item: (-item[1], item[0]))
        ),
        "topOwners": _top(owners, 10),
        "topTopics": _top(topics, 10),
        "withDescription": with_description,
        "uniqueOwners": len(owners),
        "uniqueLanguages": len(languages),
    }


class ProjectCounter:
    """Считает публичные проекты GitFlic и сохраняет снимок в JSON."""

    def __init__(
        self,
        fetch: Callable[..., dict] = fetch_public_projects,
        store: CountStore | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> None:
        self._fetch = fetch
        self.store = store or CountStore()
        self.page_size = max(1, min(int(page_size), 100))
        self.max_pages = max(1, int(max_pages))

    def count_now(
        self, query: str = "", job_id: str | None = None, trigger: str = "manual"
    ) -> dict:
        """Делает подсчёт, сохраняет снимок и возвращает агрегированный результат.

        Точное количество публичных проектов берётся из пагинации API
        (``totalElements``) на первой странице, а по страницам (до
        ``max_pages``) собирается выборка для агрегации по языкам и владельцам.
        """
        started = time.monotonic()
        seen: set = set()
        projects: list[dict] = []
        page_number = 0
        total_pages = 1
        total: int | None = None
        truncated = False

        while page_number < total_pages:
            if page_number >= self.max_pages:
                truncated = True
                break
            page = self._fetch(query=query, page=page_number, size=self.page_size)
            if total is None:
                total = int(page.get("total") or 0)
                total_pages = int(page.get("page", {}).get("totalPages", 1)) or 1
            for project in page.get("projects") or []:
                key = project.get("id") or project.get("alias") or id(project)
                if key in seen:
                    continue
                seen.add(key)
                projects.append(project)
            page_number += 1

        aggregated = aggregate_projects(projects)
        if total is None:
            total = len(projects)
        # Точное число из API; выборка — сколько проектов реально разобрали.
        aggregated["totalPublicProjects"] = total
        aggregated["countedProjects"] = len(projects)
        aggregated["sampled"] = truncated or len(projects) < total
        snapshot = {
            "countedAt": now_iso(),
            "query": query,
            "jobId": job_id,
            "trigger": trigger,
            "pagesFetched": page_number,
            "totalPages": total_pages,
            "truncated": truncated,
            "durationMs": int((time.monotonic() - started) * 1000),
            "storePath": str(self.store.path),
            **aggregated,
        }
        self.store.append(snapshot)
        return snapshot


class ProjectCountScheduler:
    """Планировщик отложенных и периодических подсчётов публичных проектов.

    Задачи хранятся в JSON (:class:`JobStore`), поэтому не теряются между
    вызовами MCP-инструментов. Поддерживаются:

    * отложенный запуск — ``delay_seconds > 0`` и без интервала;
    * периодический запуск — ``interval_seconds`` (задача повторяется);
    * немедленный запуск — ``run_immediately=True`` (сразу выполняет подсчёт).

    Выполнение запускается либо извне — ``run_due()`` (в cron), либо
    потоком-демоном — ``start_daemon()``.
    """

    def __init__(
        self,
        counter: ProjectCounter | None = None,
        jobs_store: JobStore | None = None,
    ) -> None:
        self._counter = counter or ProjectCounter()
        self.jobs = jobs_store or JobStore()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def counter(self) -> ProjectCounter:
        return self._counter

    def schedule(
        self,
        query: str = "",
        delay_seconds: float = 0.0,
        interval_seconds: float | None = None,
        run_immediately: bool = False,
    ) -> dict:
        """Ставит задачу подсчёта, сохраняет её и возвращает описание."""
        delay = max(0.0, float(delay_seconds))
        if run_immediately:
            delay = 0.0
        interval = None if interval_seconds is None else max(1.0, float(interval_seconds))
        job = {
            "jobId": uuid.uuid4().hex[:12],
            "query": query or "",
            "delaySeconds": delay,
            "intervalSeconds": interval,
            "periodic": interval is not None,
            "createdAt": now_iso(),
            "nextRunAt": _future_iso(delay),
            "runs": 0,
            "cancelled": False,
            "lastResult": None,
        }
        self.jobs.put(job)
        if run_immediately:
            self._run_job(job["jobId"])
            job = self.jobs.get(job["jobId"]) or job
        return _job_view(job)

    def _run_job(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None or job.get("cancelled"):
            return None
        query = job.get("query", "")
        interval = job.get("intervalSeconds")
        try:
            result = self._counter.count_now(
                query=query, job_id=job_id, trigger="schedule"
            )
        except Exception as exc:  # noqa: BLE001 — фоновую ошибку сохраняем в задаче
            result = {
                "countedAt": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "totalPublicProjects": None,
            }
        job["runs"] = int(job.get("runs", 0)) + 1
        job["lastRunAt"] = result.get("countedAt")
        job["lastResult"] = result
        if interval is not None:
            job["nextRunAt"] = _future_iso(float(interval))
        else:
            job["cancelled"] = True
        self.jobs.put(job)
        return _job_view(job)

    def due_jobs(self, at: datetime | None = None) -> list[dict]:
        """Задачи, готовые к выполнению (``nextRunAt <= now`` и не отменённые)."""
        moment = at or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        due = []
        for job in self.jobs.all():
            if job.get("cancelled"):
                continue
            next_run = _parse_iso(job.get("nextRunAt"))
            if next_run is None or next_run <= moment:
                due.append(job)
        return due

    def run_due(self, at: datetime | None = None) -> list[dict]:
        """Выполняет все готовые задачи и возвращает результаты их запусков."""
        executed = []
        for job in self.due_jobs(at):
            view = self._run_job(job["jobId"])
            if view is not None:
                executed.append(view)
        return executed

    def list_jobs(self) -> list[dict]:
        jobs = [_job_view(job) for job in self.jobs.all()]
        return sorted(jobs, key=lambda item: item.get("createdAt") or "")

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job["cancelled"] = True
        self.jobs.put(job)
        self._wake.set()
        return True

    def start_daemon(self) -> None:
        """Запускает поток, который сам выполняет готовые задачи раз в секунду."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="gitflic-count-scheduler", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_due()
            self._wake.wait(timeout=1.0)
            self._wake.clear()

    def stop_daemon(self) -> None:
        self._stop.set()
        self._wake.set()


def _job_view(job: dict) -> dict:
    """Компактное представление задачи для инструмента/CLI."""
    last = job.get("lastResult") or {}
    return {
        "jobId": job.get("jobId"),
        "query": job.get("query", ""),
        "delaySeconds": job.get("delaySeconds"),
        "intervalSeconds": job.get("intervalSeconds"),
        "periodic": job.get("periodic", False),
        "createdAt": job.get("createdAt"),
        "nextRunAt": job.get("nextRunAt"),
        "lastRunAt": job.get("lastRunAt"),
        "runs": job.get("runs", 0),
        "cancelled": job.get("cancelled", False),
        "lastTotal": last.get("totalPublicProjects") if last else None,
        "lastError": last.get("error") if last else None,
    }
