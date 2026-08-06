"""Background job queue: track / solve / smooth / render / export / proxy.

PLAN.md asks for "job queue (track/solve/render as background jobs), WebSocket
progress". :class:`JobManager` drains a FIFO queue with a small pool of worker
threads. A job is just a callable that receives a progress callback and returns
a JSON-safe result.

Why a pool, and why that is still safe
--------------------------------------
Phase 4 ran exactly one worker, to keep GPU contention and project-file races
out of the picture. The races never needed it — every write goes through
:meth:`stable.server.storage.Workspace.update_project`, which holds a global
lock — and the cost was severe: tracking three points meant three tracking jobs
executed strictly one after another on a machine with 32 idle cores. Jobs now
run ``STABLE_JOB_WORKERS`` at a time (default: a quarter of the CPUs, at least
two, at most four).

GPU contention is handled where it actually lives instead. Deep trackers
already fall back to the CPU on out-of-memory (see
:class:`stable.trackers.deep.DeepTrackerPlugin`), and any job that must not
overlap another of its kind can say so with ``exclusive=True`` at submit time —
video export uses that, because two FFmpeg encodes fight over the same cores to
nobody's benefit.

Live partial results
--------------------
The progress callback handed to a job is a :class:`JobProgress`, which is still
just ``progress(fraction, message)`` — but it also carries
:meth:`JobProgress.publish`, for streaming a partial result *before* the job
finishes. Whatever a job publishes rides out on the same WebSocket event as its
progress (:attr:`Job.partial`), so the UI can show work in flight rather than
only its outcome.

Tracking gets that for free: the worker installs a
:mod:`stable.video.telemetry` watcher for the duration of each job, so the frame
the tracker is reading right now is published as it moves. That is the tracker's
own cursor — no tracker or tracking-pipeline code had to change to expose it —
and it is what lets an operator watch a run progress frame by frame instead of
waiting for the whole clip (todos 4 and 16).

The WebSocket route polls :meth:`JobManager.snapshot` rather than the worker
threads pushing into the asyncio loop directly — bridging a plain
:class:`threading.Thread` into ``asyncio`` correctly needs
``call_soon_threadsafe`` plumbing that buys nothing here, since polling a
few times a second is imperceptible for a local progress bar.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from ..video.telemetry import watching_frames

__all__ = [
    "JobStatus",
    "Job",
    "JobProgress",
    "JobManager",
    "get_job_manager",
    "reset_job_manager",
    "default_job_workers",
    "MODELS_JOB_SCOPE",
    "ENV_JOB_WORKERS",
]

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]

JobFn = Callable[[Any], dict[str, Any]]
"""A job body: takes a :class:`JobProgress`, returns a JSON-safe result dict.

Typed loosely on purpose — a job body only needs to call ``progress(f, msg)``,
so every existing one written against a plain callback keeps working.
"""

_MAX_FINISHED_PER_PROJECT = 100
"""Finished jobs kept per project before the oldest are evicted."""

ENV_JOB_WORKERS = "STABLE_JOB_WORKERS"
"""Environment variable overriding how many jobs may run at once."""

_MAX_DEFAULT_WORKERS = 4
"""Ceiling on the automatic worker count.

Jobs are heavyweight (a whole tracking run, an FFmpeg encode), and each already
uses several threads inside itself — OpenCV, FFmpeg's decoder and the frame
read-ahead all scale out on their own. Four concurrent jobs is enough to keep a
workstation busy without oversubscribing it into cache thrash.
"""

_LIVE_FRAME_INTERVAL = 0.1
"""Minimum seconds between two live frame-cursor updates for one job."""

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


def default_job_workers() -> int:
    """Return how many jobs may run concurrently on this machine.

    Returns:
        ``$STABLE_JOB_WORKERS`` when it is set to a positive integer, else a
        quarter of the CPU count clamped to ``[2, 4]``.
    """
    raw = os.environ.get(ENV_JOB_WORKERS)
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
        if requested > 0:
            return requested
    cpus = os.cpu_count() or 1
    return max(2, min(_MAX_DEFAULT_WORKERS, cpus // 4 or 1))


@dataclass
class Job:
    """One unit of background work and its live status.

    Args:
        id: Short opaque identifier.
        project_id: The project this job belongs to.
        kind: ``"proxy"``, ``"track"``, ``"solve"``, ``"smooth"``, ``"render"``
            or ``"export"``.
        target: Optional id the job concerns (e.g. a point id), for display.
        exclusive: Whether no other job of the same ``kind`` may run alongside it.
    """

    id: str
    project_id: str
    kind: str
    target: str | None = None
    status: JobStatus = "queued"
    fraction: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    partial: dict[str, Any] | None = None
    error: str | None = None
    exclusive: bool = False
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
            "partial": self.partial,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class JobProgress:
    """The progress callback a job body receives.

    Callable as ``progress(fraction, message)`` — the whole of the older
    contract — plus :meth:`publish` for streaming a partial result while the job
    is still running.

    Args:
        job: The job being reported on.
    """

    def __init__(self, job: Job) -> None:
        """Bind this reporter to one job."""
        self._job = job

    def __call__(self, fraction: float, message: str = "") -> None:
        """Report overall completion.

        Args:
            fraction: Completion in ``[0, 1]``; values outside are clamped.
            message: Optional short human-readable status.
        """
        self._job.fraction = max(0.0, min(1.0, float(fraction)))
        self._job.message = message
        self._job.updated_at = time.time()

    def publish(self, partial: dict[str, Any] | None) -> None:
        """Attach a partial result to the job, visible to subscribers immediately.

        Args:
            partial: Any JSON-safe dict describing work done so far, or ``None``
                to clear it. Merged into the job event as ``"partial"``.
        """
        self._job.partial = partial
        self._job.updated_at = time.time()

    def merge(self, **fields: Any) -> None:
        """Update individual keys of the partial result, keeping the rest.

        Args:
            **fields: Keys to set on the partial payload.
        """
        current = dict(self._job.partial or {})
        current.update(fields)
        self.publish(current)


class _FrameCursor:
    """Publishes the frame a job's decoder just served, at a bounded rate.

    Installed by the worker for the whole of a job (see
    :func:`stable.video.telemetry.watching_frames`). Frame reads happen
    thousands of times per run and every publish wakes a WebSocket poll, so
    updates are throttled; the *range* covered is accumulated between them, so
    nothing is lost, only smoothed.
    """

    def __init__(self, progress: JobProgress) -> None:
        """Bind a cursor to one job's progress reporter."""
        self._progress = progress
        self._last_publish = 0.0
        self._low: int | None = None
        self._high: int | None = None
        self._seen = 0

    def __call__(self, index: int) -> None:
        """Record that frame ``index`` was read; publish if the interval has passed."""
        self._seen += 1
        self._low = index if self._low is None else min(self._low, index)
        self._high = index if self._high is None else max(self._high, index)
        now = time.time()
        if now - self._last_publish < _LIVE_FRAME_INTERVAL:
            return
        self._last_publish = now
        self._progress.merge(
            frame=index,
            frame_min=self._low,
            frame_max=self._high,
            frames_read=self._seen,
        )


class JobManager:
    """A FIFO background job queue with a worker pool and live progress.

    Args:
        max_workers: How many jobs may run at once; ``None`` uses
            :func:`default_job_workers`.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """Start the background worker threads."""
        self.max_workers = max(1, int(max_workers or default_job_workers()))
        self._jobs: dict[str, Job] = {}
        self._by_project: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._fns: dict[str, JobFn] = {}
        self._exclusive_kinds: set[str] = set()
        self._exclusive_lock = threading.Condition(threading.Lock())
        self._threads = [
            threading.Thread(target=self._worker, name=f"stable-jobs-{i}", daemon=True)
            for i in range(self.max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        project_id: str,
        kind: str,
        fn: JobFn,
        *,
        target: str | None = None,
        exclusive: bool = False,
    ) -> Job:
        """Enqueue a job and return it immediately (still ``"queued"``).

        Args:
            project_id: The project the job concerns.
            kind: A short label for the kind of work (see :class:`Job`).
            fn: The job body; called from a worker thread with a
                :class:`JobProgress`.
            target: Optional id the job concerns, e.g. a point id.
            exclusive: Serialise this job against every other job of the same
                ``kind``. Use it for work that saturates a shared resource (an
                FFmpeg encode); leave it off for work that parallelises.

        Returns:
            The newly created :class:`Job`, in ``"queued"`` status.
        """
        job = Job(
            id=uuid4().hex[:12],
            project_id=project_id,
            kind=kind,
            target=target,
            exclusive=exclusive,
        )
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

    def cancel(self, job_id: str) -> Job:
        """Drop a job that has not started yet.

        Only a *queued* job can be cancelled. Interrupting a running one would
        mean abandoning an encode part-way through its file or a tracking run
        part-way through its merge, and there is no safe point to do that at —
        so this refuses rather than pretending. The queue entry stays in the
        list, marked ``cancelled``, because "I cancelled that" is something the
        operator should still be able to see (todo 8, fourth round).

        Args:
            job_id: The job to drop.

        Returns:
            The job, now ``cancelled``.

        Raises:
            KeyError: If no such job is tracked.
            RuntimeError: If the job is already running or finished.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "queued":
                raise RuntimeError(f"job {job_id} is already {job.status}")
            job.status = "cancelled"
            job.message = "cancelled before it started"
            job.updated_at = time.time()
            self._fns.pop(job_id, None)
            return job

    def jobs_for_project(self, project_id: str) -> list[Job]:
        """Return every tracked job for a project, oldest first."""
        with self._lock:
            ids = self._by_project.get(project_id, [])
            return [self._jobs[jid] for jid in ids if jid in self._jobs]

    def snapshot(self, project_id: str) -> list[dict[str, Any]]:
        """Return :meth:`Job.to_event` for every tracked job of a project."""
        return [job.to_event() for job in self.jobs_for_project(project_id)]

    @property
    def running(self) -> list[Job]:
        """Every job currently executing, for diagnostics and tests."""
        with self._lock:
            return [job for job in self._jobs.values() if job.status == "running"]

    # -- worker --------------------------------------------------------------

    def _worker(self) -> None:
        """Drain the queue forever, running one job at a time on this thread."""
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                fn = self._fns.get(job_id)
            # `fn` is dropped by both eviction and `cancel`, which is what makes
            # a cancelled job a no-op here rather than a special case.
            if job is None or fn is None or job.status != "queued":
                continue
            self._run_job(job, fn)

    def _run_job(self, job: Job, fn: JobFn) -> None:
        """Execute one job, recording its outcome and never letting it kill the worker."""
        if job.exclusive:
            # Waiting for the slot happens *before* the job is claimed, so an
            # export sitting behind three others still reads as "queued" — and
            # is still cancellable, which is the whole point of a queue.
            self._enter_exclusive(job.kind)
        try:
            # Claim it, atomically against `cancel`. Between being taken off the
            # queue and reaching here the job may have been cancelled; whichever
            # of the two takes the lock first wins, and the other sees it.
            with self._lock:
                if job.status != "queued":
                    return
                job.status = "running"
                job.updated_at = time.time()

            progress = JobProgress(job)
            try:
                with watching_frames(_FrameCursor(progress)):
                    result = fn(progress)
                job.status = "done"
                job.fraction = 1.0
                job.result = result
            except Exception as exc:  # noqa: BLE001 - a failure must not kill the worker
                logger.warning("job %s (%s) failed: %s", job.id, job.kind, exc)
                logger.debug("job %s traceback:\n%s", job.id, traceback.format_exc())
                job.status = "error"
                job.error = str(exc)
            finally:
                job.partial = None
                job.updated_at = time.time()
                with self._lock:
                    self._fns.pop(job.id, None)
        finally:
            if job.exclusive:
                self._leave_exclusive(job.kind)

    def _enter_exclusive(self, kind: str) -> None:
        """Block until no other exclusive job of ``kind`` is running."""
        with self._exclusive_lock:
            while kind in self._exclusive_kinds:
                self._exclusive_lock.wait()
            self._exclusive_kinds.add(kind)

    def _leave_exclusive(self, kind: str) -> None:
        """Release this kind's exclusive slot."""
        with self._exclusive_lock:
            self._exclusive_kinds.discard(kind)
            self._exclusive_lock.notify_all()


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return the process-wide :class:`JobManager` singleton, creating it lazily."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_job_manager(max_workers: int | None = None) -> JobManager:
    """Replace the process-wide job manager with a fresh one (tests use this).

    Args:
        max_workers: Worker count for the new manager; ``None`` uses the default.

    Returns:
        The new manager.
    """
    global _manager
    with _manager_lock:
        _manager = JobManager(max_workers)
        return _manager
