"""Background job queue: track / solve / smooth / render / export / proxy.

PLAN.md asks for "job queue (track/solve/render as background jobs), WebSocket
progress". :class:`JobManager` is a single background worker thread draining a
FIFO queue — one job runs at a time. That is a deliberate simplification for a
single-user local process: it removes GPU-contention and project-file-race
concerns entirely, at the cost of parallelism nobody asked for. A job is just a
callable that receives a progress callback and returns a JSON-safe result.

The WebSocket route polls :meth:`JobManager.snapshot` rather than the worker
thread pushing into the asyncio loop directly — bridging a plain
:class:`threading.Thread` into ``asyncio`` correctly needs
``call_soon_threadsafe`` plumbing that buys nothing here, since polling a
few times a second is imperceptible for a local progress bar.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

__all__ = [
    "JobStatus",
    "Job",
    "JobManager",
    "get_job_manager",
    "reset_job_manager",
    "MODELS_JOB_SCOPE",
]

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]

JobFn = Callable[[Callable[[float, str], None]], dict[str, Any]]
"""A job body: takes a ``progress(fraction, message)`` callback, returns a result dict."""

_MAX_FINISHED_PER_PROJECT = 100
"""Finished jobs kept per project before the oldest are evicted."""

MODELS_JOB_SCOPE = "_models"
"""Reserved pseudo-project id for model-checkpoint download jobs.

Model weights (:mod:`stable.trackers.weights`) are process-wide, not tied to
any project, but downloading one is exactly the same "background work with
live progress" shape :class:`JobManager` already gives track/solve/render
jobs. Rather than inventing a second progress mechanism, downloads are
submitted under this reserved id (see
:mod:`stable.server.routes.models` and the ``/ws/models`` route in
:mod:`stable.server.routes.ws`). No :class:`~stable.server.storage.Workspace`
directory ever exists for it — it is a key into this module's dicts only.
"""


@dataclass
class Job:
    """One unit of background work and its live status.

    Args:
        id: Short opaque identifier.
        project_id: The project this job belongs to.
        kind: ``"proxy"``, ``"track"``, ``"solve"``, ``"smooth"``, ``"render"``
            or ``"export"``.
        target: Optional id the job concerns (e.g. a point id), for display.
    """

    id: str
    project_id: str
    kind: str
    target: str | None = None
    status: JobStatus = "queued"
    fraction: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_event(self) -> dict[str, Any]:
        """Return the JSON-safe dict sent to WebSocket subscribers and REST callers."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "fraction": self.fraction,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class JobManager:
    """A single-worker FIFO background job queue with live progress tracking."""

    def __init__(self) -> None:
        """Start the background worker thread."""
        self._jobs: dict[str, Job] = {}
        self._by_project: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._fns: dict[str, JobFn] = {}
        self._thread = threading.Thread(target=self._worker, name="stable-jobs", daemon=True)
        self._thread.start()

    def submit(self, project_id: str, kind: str, fn: JobFn, *, target: str | None = None) -> Job:
        """Enqueue a job and return it immediately (still ``"queued"``).

        Args:
            project_id: The project the job concerns.
            kind: A short label for the kind of work (see :class:`Job`).
            fn: The job body; called from the worker thread.
            target: Optional id the job concerns, e.g. a point id.

        Returns:
            The newly created :class:`Job`, in ``"queued"`` status.
        """
        job = Job(id=uuid4().hex[:12], project_id=project_id, kind=kind, target=target)
        with self._lock:
            self._jobs[job.id] = job
            self._fns[job.id] = fn
            bucket = self._by_project.setdefault(project_id, [])
            bucket.append(job.id)
            self._evict_old(bucket)
        self._queue.put(job.id)
        return job

    def _evict_old(self, bucket: list[str]) -> None:
        """Drop the oldest finished jobs for a project past the retention cap."""
        finished_statuses = ("done", "error", "cancelled")
        finished = [jid for jid in bucket if self._jobs[jid].status in finished_statuses]
        overflow = len(finished) - _MAX_FINISHED_PER_PROJECT
        for jid in finished[: max(overflow, 0)]:
            bucket.remove(jid)
            self._jobs.pop(jid, None)
            self._fns.pop(jid, None)

    def get(self, job_id: str) -> Job | None:
        """Return a job by id, or ``None`` if it is unknown (or was evicted)."""
        with self._lock:
            return self._jobs.get(job_id)

    def jobs_for_project(self, project_id: str) -> list[Job]:
        """Return every tracked job for a project, oldest first."""
        with self._lock:
            ids = self._by_project.get(project_id, [])
            return [self._jobs[jid] for jid in ids if jid in self._jobs]

    def snapshot(self, project_id: str) -> list[dict[str, Any]]:
        """Return :meth:`Job.to_event` for every tracked job of a project."""
        return [job.to_event() for job in self.jobs_for_project(project_id)]

    # -- worker --------------------------------------------------------------

    def _worker(self) -> None:
        """Drain the queue forever, running one job at a time."""
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                fn = self._fns.get(job_id)
            if job is None or fn is None:  # pragma: no cover - evicted before running
                continue
            job.status = "running"
            job.updated_at = time.time()

            def progress(fraction: float, message: str = "", _job: Job = job) -> None:
                _job.fraction = max(0.0, min(1.0, float(fraction)))
                _job.message = message
                _job.updated_at = time.time()

            try:
                result = fn(progress)
                job.status = "done"
                job.fraction = 1.0
                job.result = result
            except Exception as exc:  # noqa: BLE001 - a job's failure must not kill the worker
                logger.warning("job %s (%s) failed: %s", job.id, job.kind, exc)
                logger.debug("job %s traceback:\n%s", job.id, traceback.format_exc())
                job.status = "error"
                job.error = str(exc)
            finally:
                job.updated_at = time.time()
                with self._lock:
                    self._fns.pop(job_id, None)


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return the process-wide :class:`JobManager` singleton, creating it lazily."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_job_manager() -> JobManager:
    """Replace the process-wide job manager with a fresh one (tests use this)."""
    global _manager
    with _manager_lock:
        _manager = JobManager()
        return _manager
