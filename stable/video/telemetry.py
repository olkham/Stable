"""Which frame is being read, right now — the seam live progress is built on.

Every frame-stepping tracker in this project reads frames strictly in run order,
so "which frame did the decoder just hand out on this thread" *is* the tracker's
cursor. That single number is what turns a tracking job from an opaque wait into
something the operator can watch: a progress bar that moves per frame instead of
once per run, and a live "the tracker is here" marker on the timeline (todos 4
and 16).

Getting at it needed no change to any tracker, because the video source is the
one object every tracker is handed. Sources call :func:`note_frame`; whoever
wants to listen installs a watcher on the thread doing the work, with
:func:`watching_frames`.

Deliberately **thread-local and dependency-free**: this module knows nothing
about jobs, HTTP or the server, so ``stable.video`` keeps its "no upward
dependencies" property, and two jobs running in parallel on two worker threads
cannot see each other's cursor.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

__all__ = ["FrameWatcher", "note_frame", "set_frame_watcher", "watching_frames"]


class FrameWatcher(Protocol):
    """Callable notified each time a frame is served on this thread.

    Implementations must be cheap and must never raise — they run inside the
    decode path.
    """

    def __call__(self, index: int) -> None:
        """Record that frame ``index`` was just read."""
        ...


_local = threading.local()


def set_frame_watcher(watcher: FrameWatcher | None) -> FrameWatcher | None:
    """Install (or clear) the frame watcher for the calling thread.

    Args:
        watcher: The watcher to install, or ``None`` to remove it.

    Returns:
        The watcher that was previously installed, so callers can restore it.
    """
    previous: FrameWatcher | None = getattr(_local, "watcher", None)
    _local.watcher = watcher
    return previous


@contextmanager
def watching_frames(watcher: FrameWatcher | None) -> Iterator[None]:
    """Install ``watcher`` for the duration of the block, then restore the old one.

    Args:
        watcher: The watcher to install; ``None`` disables watching in the block.

    Yields:
        Nothing; the watcher is active for the body of the ``with`` statement.
    """
    previous = set_frame_watcher(watcher)
    try:
        yield
    finally:
        set_frame_watcher(previous)


def note_frame(index: int) -> None:
    """Report that frame ``index`` was served on this thread.

    A no-op (one thread-local attribute lookup) when nothing is watching, which
    is the case for the CLI, tests, and every request thread — so the decode
    path pays effectively nothing for this.

    Args:
        index: Absolute frame index that was just handed to a caller.
    """
    watcher: FrameWatcher | None = getattr(_local, "watcher", None)
    if watcher is None:
        return
    try:
        watcher(index)
    except Exception:  # noqa: BLE001 - telemetry must never break a decode
        set_frame_watcher(None)
