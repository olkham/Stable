"""The render queue: exports that wait to be asked for (todo 6, tenth round).

"When a render is queued don't auto render it. Create a processing queue list so
the user can queue up many different runs, then add a button to process them.
Let the user delete items from the queue, re-order them, and see the estimated
time for each render."

Every one of those five needs the same missing thing: a queued render has to be
a *stored object* rather than a job that has already been handed to a worker.
``POST .../export/video`` plans and submits in one breath, so its "queue" was
only ever a list of jobs already running or already finished — nothing to
reorder, nothing to estimate, and removable only in the sliver of time before a
worker picked it up (:meth:`stable.server.jobs.JobManager.cancel` refuses a job
that has started, correctly: an encode abandoned half way through its file is
worse than one that finishes).

What a queued render carries
----------------------------
Everything the render depends on, captured when it was queued: the filename, the
codec and CRF, the border policy, the whole :class:`~stable.render.options.RenderOptions`
block, the smoothing filter list, and the resolved frame range. That capture is
the entire reason a queue is worth having — line up "this crop" and then "that
crop", or "1 Hz" and then "0.5 Hz", and each entry renders what *it* was queued
with rather than whatever happened to be saved last.

The plan itself is not stored: a :class:`~stable.render.stabilize.StabilizationPlan`
is an in-memory O(T) fit over the whole clip. It is rebuilt from the snapshot
when the entry runs. What *is* done at queue time is building one and checking
it — an entry that would render nothing at all is refused while the operator is
still looking at the settings that caused it, rather than minutes later.

Why a runner thread rather than submitting them all
---------------------------------------------------
Two reasons, both about :class:`~stable.server.jobs.JobManager`. Exclusive jobs
wait on a condition variable woken with ``notify_all``, so several exports
submitted together do not necessarily start in the order they were submitted —
and a reorder the server will not honour is a lie. And a waiting exclusive job
holds a worker thread: the pool is 2–4 wide, so six queued exports would occupy
all of it while a tracking job sat behind them.

So one daemon thread per project submits the *next* pending entry only once the
previous one has reached a terminal state. Each job is still ``exclusive``,
which now guards only against a manual ``POST .../export/video`` landing
alongside the queue.

Where it is kept
----------------
``render_queue.json`` and ``render_stats.json``, sidecars in the project
directory beside ``clip_range.json`` and ``render_options.json``. The locked v1
project schema stays untouched: a queue is working state, not part of the
document a migration would have to carry forward.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = [
    "RENDER_QUEUE_CACHE",
    "RENDER_STATS_CACHE",
    "PENDING",
    "TERMINAL",
    "QueuedRender",
    "load_queue",
    "save_queue",
    "append_entry",
    "remove_entry",
    "reorder_queue",
    "mark_entry",
    "next_pending",
    "record_render",
    "seconds_per_frame",
    "estimate_seconds",
    "start_runner",
    "runner_active",
    "drain_runners",
]

RENDER_QUEUE_CACHE = "render_queue.json"
"""File name of a project's render-queue sidecar."""

RENDER_STATS_CACHE = "render_stats.json"
"""File name of the measured render-speed sidecar the estimates come from."""

PENDING = "pending"
"""Status of an entry that has been queued and not yet handed to a worker."""

TERMINAL: tuple[str, ...] = ("done", "error", "cancelled")
"""Statuses an entry cannot leave. Mirrors the job statuses it copies."""

DEFAULT_SECONDS_PER_MEGAPIXEL_FRAME = 0.05
"""What one output megapixel-frame is assumed to cost before anything is measured.

A warp plus an encode, at roughly ten 1080p frames a second on a workstation
CPU. It is a seed, not a claim: the first render that finishes replaces it with
what this machine actually did, and until then every estimate built from it is
flagged ``measured: false`` so the panel can say "rough" rather than quoting a
guess as a measurement.
"""

_STATS_SAMPLES = 5
"""How many finished renders the estimate averages over.

Long enough to survive one unusually fast or slow encode, short enough that
switching codec or resolution is reflected within a couple of renders rather
than being averaged away for the rest of the project's life.
"""

_LOCK = threading.Lock()
"""Serialises every read-modify-write of both sidecars.

One lock for every project, like ``routes/solve.py``'s ``_OPTIONS_LOCK`` and for
the same reason: these are sub-millisecond file writes, and a per-project map
would be more bookkeeping than the contention it saves.
"""


@dataclass(frozen=True)
class QueuedRender:
    """One render waiting to be asked for, and everything it will need.

    Args:
        id: Short identifier, unique within the queue.
        filename: The ``.mp4`` this will be written to. Reserved when the entry
            is queued, so two queued renders cannot collide on a name.
        codec: FFmpeg codec name.
        crf: Constant rate factor.
        border: The border policy this entry was queued with.
        options: The whole :class:`~stable.render.options.RenderOptions` dict.
        smoothing: The project's smoothing filter list, as project-file dicts.
        frame_range: The ``[start, end)`` frames this will write.
        width: Output width in pixels, for the estimate.
        height: Output height in pixels, for the estimate.
        status: ``pending``, or whatever its job last reported.
        job_id: The job rendering it, once one has been submitted.
        created_at: When it was queued.
        error: Why it failed, when it did.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    filename: str = ""
    codec: str = "h264"
    crf: int = 18
    border: str = "auto_zoom"
    options: dict[str, Any] = field(default_factory=dict)
    smoothing: list[dict[str, Any]] = field(default_factory=list)
    frame_range: tuple[int, int] = (0, 0)
    width: int = 0
    height: int = 0
    status: str = PENDING
    job_id: str | None = None
    created_at: float = field(default_factory=time.time)
    error: str | None = None

    @property
    def frames(self) -> int:
        """How many frames this entry will write."""
        return max(0, int(self.frame_range[1]) - int(self.frame_range[0]))

    @property
    def is_pending(self) -> bool:
        """Whether this entry may still be reordered, dropped or started."""
        return self.status == PENDING

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe sidecar form."""
        return {
            "id": self.id,
            "filename": self.filename,
            "codec": self.codec,
            "crf": self.crf,
            "border": self.border,
            "options": dict(self.options),
            "smoothing": [dict(spec) for spec in self.smoothing],
            "frame_range": [int(self.frame_range[0]), int(self.frame_range[1])],
            "width": int(self.width),
            "height": int(self.height),
            "status": self.status,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueuedRender:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        window = data.get("frame_range") or [0, 0]
        return cls(
            id=str(data.get("id") or uuid4().hex[:12]),
            filename=str(data.get("filename", "")),
            codec=str(data.get("codec", "h264")),
            crf=int(data.get("crf", 18)),
            border=str(data.get("border", "auto_zoom")),
            options=dict(data.get("options") or {}),
            smoothing=[dict(spec) for spec in data.get("smoothing") or []],
            frame_range=(int(window[0]), int(window[1])),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            status=str(data.get("status", PENDING)),
            job_id=data.get("job_id"),
            created_at=float(data.get("created_at", 0.0)),
            error=data.get("error"),
        )


def queue_path(project_dir: Path) -> Path:
    """Return where a project's render queue lives."""
    return Path(project_dir) / RENDER_QUEUE_CACHE


def stats_path(project_dir: Path) -> Path:
    """Return where a project's measured render speeds live."""
    return Path(project_dir) / RENDER_STATS_CACHE


def _read(path: Path) -> Any:
    """Return a sidecar's parsed contents, or ``None`` if it cannot be read."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar only
        return None


def load_queue(project_dir: Path) -> list[QueuedRender]:
    """Return the project's queue, in order. An unreadable sidecar is an empty one."""
    payload = _read(queue_path(project_dir))
    if not isinstance(payload, list):
        return []
    return [QueuedRender.from_dict(dict(entry)) for entry in payload if isinstance(entry, dict)]


def save_queue(project_dir: Path, entries: Sequence[QueuedRender]) -> list[QueuedRender]:
    """Write the queue out, in the order given."""
    path = queue_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry.to_dict() for entry in entries], indent=2), encoding="utf-8")
    return list(entries)


def _update(
    project_dir: Path, mutate: Callable[[list[QueuedRender]], list[QueuedRender]]
) -> list[QueuedRender]:
    """Read, mutate and write the queue under the module lock."""
    with _LOCK:
        return save_queue(project_dir, mutate(load_queue(project_dir)))


def append_entry(project_dir: Path, entry: QueuedRender) -> list[QueuedRender]:
    """Add one entry to the end of the queue."""
    return _update(project_dir, lambda entries: [*entries, entry])


def remove_entry(project_dir: Path, entry_id: str) -> list[QueuedRender]:
    """Drop one entry, whatever state it is in.

    A finished or failed entry is dropped as readily as a pending one: once it
    has stopped being work it is a record, and dismissing a record is the only
    way the list stays a queue rather than a log. Removing a *running* entry
    only forgets it here — the job keeps going, because there is no safe point
    to interrupt an encode at, which is the same answer
    :meth:`stable.server.jobs.JobManager.cancel` gives.
    """
    return _update(project_dir, lambda entries: [e for e in entries if e.id != entry_id])


def reorder_queue(project_dir: Path, ids: Iterable[str]) -> list[QueuedRender]:
    """Put the pending entries in the order named.

    Only the pending ones move. An entry already handed to a worker has a place
    in the world — a job, and soon a file — and pretending it could be shuffled
    would be the same lie as a reorder the runner ignores. Pending entries the
    caller did not name keep their relative order, at the end.
    """
    wanted = list(ids)

    def _mutate(entries: list[QueuedRender]) -> list[QueuedRender]:
        pending = {e.id: e for e in entries if e.is_pending}
        ordered = [pending.pop(entry_id) for entry_id in wanted if entry_id in pending]
        ordered.extend(e for e in entries if e.is_pending and e.id in pending)
        moved = iter(ordered)
        return [next(moved) if e.is_pending else e for e in entries]

    return _update(project_dir, _mutate)


def mark_entry(
    project_dir: Path,
    entry_id: str,
    *,
    status: str,
    job_id: str | None = None,
    error: str | None = None,
) -> list[QueuedRender]:
    """Record what has happened to one entry."""

    def _mutate(entries: list[QueuedRender]) -> list[QueuedRender]:
        return [
            replace(
                entry,
                status=status,
                job_id=job_id if job_id is not None else entry.job_id,
                error=error,
            )
            if entry.id == entry_id
            else entry
            for entry in entries
        ]

    return _update(project_dir, _mutate)


def next_pending(project_dir: Path) -> QueuedRender | None:
    """Return the entry the runner should start next, or ``None`` if the queue is done."""
    with _LOCK:
        for entry in load_queue(project_dir):
            if entry.is_pending:
                return entry
    return None


def reserved_names(project_dir: Path) -> set[str]:
    """Filenames already claimed by queued entries.

    A queued render has not written its file yet, so the "does this name exist
    on disk?" check that keeps two exports from overwriting each other cannot
    see it. Queueing three renders in a row with no name would otherwise give
    all three the same timestamped one.
    """
    return {entry.filename for entry in load_queue(project_dir) if entry.filename}


# -- how long a render takes ----------------------------------------------------


def record_render(
    project_dir: Path, *, seconds: float, frames: int, width: int, height: int
) -> None:
    """Record what a finished render actually cost, for the next estimate.

    Stored per output megapixel-frame rather than per frame, so a 4K render
    teaches the estimator something usable about a 1080p one.
    """
    megapixels = (max(1, int(width)) * max(1, int(height))) / 1_000_000.0
    work = megapixels * max(1, int(frames))
    if seconds <= 0.0 or work <= 0.0:  # pragma: no cover - a zero-length render
        return
    with _LOCK:
        samples = _read(stats_path(project_dir))
        values = [float(v) for v in samples] if isinstance(samples, list) else []
        values.append(seconds / work)
        path = stats_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values[-_STATS_SAMPLES:]), encoding="utf-8")


def seconds_per_frame(project_dir: Path, width: int, height: int) -> tuple[float, bool]:
    """Return the cost of one output frame at this size, and whether it was measured.

    Returns:
        ``(seconds, measured)``. ``measured`` is False when nothing has finished
        on this machine yet and the answer comes from
        :data:`DEFAULT_SECONDS_PER_MEGAPIXEL_FRAME` — which callers pass on, so
        the difference between a measurement and a seed is never lost on the way
        to the operator.
    """
    samples = _read(stats_path(project_dir))
    values = [float(v) for v in samples] if isinstance(samples, list) else []
    measured = bool(values)
    per_megapixel = sum(values) / len(values) if measured else DEFAULT_SECONDS_PER_MEGAPIXEL_FRAME
    megapixels = (max(1, int(width)) * max(1, int(height))) / 1_000_000.0
    return (per_megapixel * megapixels, measured)


def estimate_seconds(project_dir: Path, entry: QueuedRender) -> tuple[float, bool]:
    """Return how long this entry should take, and whether that is measured."""
    per_frame, measured = seconds_per_frame(project_dir, entry.width, entry.height)
    return (per_frame * entry.frames, measured)


# -- the runner -----------------------------------------------------------------

_RUNNERS: set[str] = set()
_RUNNER_LOCK = threading.Lock()

_POLL_INTERVAL = 0.2
"""Seconds between checks on the job the runner is waiting for.

The same trade the WebSocket route makes for job progress: polling a handful of
times a second is imperceptible next to a render measured in minutes, and it
avoids bridging worker threads into anybody's event loop.
"""


def runner_active(project_id: str) -> bool:
    """Whether a queue runner is already working through this project."""
    with _RUNNER_LOCK:
        return project_id in _RUNNERS


def start_runner(
    project_id: str,
    project_dir: Path,
    submit: Callable[[QueuedRender], str],
    status_of: Callable[[str], tuple[str, str | None]],
) -> bool:
    """Start working through a project's pending entries, one at a time.

    Args:
        project_id: The project whose queue to drain.
        project_dir: Where its sidecars live.
        submit: Turns one entry into a running job and returns the job id. Any
            exception it raises marks that entry failed and the runner moves on
            — one unrenderable entry must not strand the ones behind it.
        status_of: Returns ``(status, error)`` for a job id, so the runner can
            wait for it. Injected rather than imported to keep this module free
            of the route layer that owns the plan-building.

    Returns:
        Whether a runner was started. ``False`` means one is already going,
        which is not a failure: the entries just queued will be picked up by the
        run already in progress.
    """
    with _RUNNER_LOCK:
        if project_id in _RUNNERS:
            return False
        _RUNNERS.add(project_id)

    def _drive() -> None:
        try:
            while True:
                entry = next_pending(project_dir)
                if entry is None:
                    return
                try:
                    job_id = submit(entry)
                except Exception as exc:  # noqa: BLE001 - one bad entry, not the whole queue
                    mark_entry(project_dir, entry.id, status="error", error=str(exc))
                    continue
                mark_entry(project_dir, entry.id, status="running", job_id=job_id)
                while True:
                    status, error = status_of(job_id)
                    if status in TERMINAL:
                        mark_entry(project_dir, entry.id, status=status, job_id=job_id, error=error)
                        break
                    time.sleep(_POLL_INTERVAL)
        finally:
            with _RUNNER_LOCK:
                _RUNNERS.discard(project_id)

    threading.Thread(target=_drive, name=f"render-queue-{project_id}", daemon=True).start()
    return True


def drain_runners(timeout: float = 60.0) -> bool:
    """Wait until no queue runner is working. For tests, which cannot guess.

    Returns:
        Whether every runner finished within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _RUNNER_LOCK:
            if not _RUNNERS:
                return True
        time.sleep(0.02)
    return False
