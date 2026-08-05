"""Per-frame progress for the frame-at-a-time trackers.

Every classic-CV tracker here walks a run one frame at a time, and every one of
them used to report progress **once per query point**, after the walk::

    progress((qi + 1) / len(queries), f"point {qi + 1}/{len(queries)}")

Server-side tracking gives each run exactly one query
(:func:`stable.server.tracking.run_point_tracking` calls
``plugin.track(video, [spec.anchor], ...)``), so that line fires once — when
the run is already over. Everything in between reported nothing at all, which
is why a track over a long clip sat at::

    running 0% frame 1483

The frame number moved because it comes from somewhere else entirely: the
decoder telemetry cursor in :class:`stable.server.jobs._FrameCursor`. The
percentage did not, because nothing had told it anything. The deep trackers were
never affected — :class:`stable.trackers.deep.DeepTrackerPlugin` reports once per
inference chunk — which is exactly why the symptom looked tracker-specific.

How this fixes it
-----------------
A run's unit of work is a frame, so a frame is what progress should be counted
in. :func:`planned_steps` says how many frames a whole ``track()`` call will
step through, and :class:`RunProgress` reports one frame's worth as each is
stepped.

The trackers all share one shape — a ``_run_direction`` helper taking
``indices: range``, which is only ever *iterated* — so
:meth:`RunProgress.steps` wraps the range at the call site and no helper's
signature changes. One definition of "how far through is this run", used seven
times, rather than seven chances to get the arithmetic wrong differently.

Fractions are reported with an **empty message** on purpose. A plugin knows
where it is in its own run; it does not know that the run is the second of four
planned by the merge, and :func:`stable.server.tracking.run_point_tracking`'s
wrapper is the thing that does. Sending a message from here would overwrite the
one that has that context.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from ..core.types import Query
from .base import Direction, ProgressCallback, noop_progress

__all__ = ["planned_steps", "RunProgress"]


def planned_steps(
    queries: Sequence[Query], frame_range: tuple[int, int], direction: Direction | str
) -> int:
    """Return how many frames one ``track()`` call will step through.

    Args:
        queries: The run's anchor positions.
        frame_range: The half-open ``(start, end)`` range being covered.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.

    Returns:
        The total number of frame steps, counting each direction of each query.
        The query frames themselves are not steps — they are known already — so
        a single-frame range is ``0``, which :class:`RunProgress` treats as
        "nothing to report but done".
    """
    start, end = int(frame_range[0]), int(frame_range[1])
    total = 0
    for query in queries:
        frame = int(query.frame)
        if direction in ("fwd", "bidir"):
            total += max(0, end - frame - 1)
        if direction in ("bwd", "bidir"):
            total += max(0, frame - start)
    return total


class RunProgress:
    """Reports the fraction of a run's frames that have been stepped.

    Args:
        progress: The plugin's progress callback.
        total: How many frame steps the whole run will take; see
            :func:`planned_steps`.

    The fraction is monotonic across every query and both directions of a
    bidirectional run, because the count is over the run as a whole rather than
    per direction — a bar that resets halfway is worse than one that never
    moves.
    """

    def __init__(self, progress: ProgressCallback = noop_progress, *, total: int = 0) -> None:
        """Bind a reporter to one ``track()`` call."""
        self._progress = progress
        self._total = max(0, int(total))
        self._done = 0

    @property
    def fraction(self) -> float:
        """Completion so far, in ``[0, 1]``."""
        if self._total <= 0:
            return 1.0
        return min(1.0, self._done / self._total)

    def step(self, count: int = 1) -> None:
        """Record ``count`` frames as stepped and report the new fraction."""
        self._done += int(count)
        self._progress(self.fraction, "")

    def steps(self, indices: Iterable[int]) -> Iterator[int]:
        """Yield ``indices``, counting one frame per item.

        This is how a tracker adopts per-frame progress without changing its
        stepping loop: wrap the range where it is passed in, and the loop that
        consumes it reports as it goes.
        """
        for index in indices:
            yield index
            self.step()
