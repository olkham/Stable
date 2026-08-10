"""Building one track out of several others (todo 2, tenth round).

"Create path3 from path1 and path2, where p3 = mean(p1, p2)." The mean is taken
**frame by frame**: on frame ``f``, draw the straight line joining where the
sources are on frame ``f``, and the derived path is at the midpoint of it (the
centroid, for more than two)::

    p3[f] = mean(p1[f], p2[f])

Why not by normalised progress
------------------------------
This module first read each source at matching *progress* along its own path —
``p3[0.25] = mean(p1[0.25], p2[0.25])`` — to sidestep contributors that cover
different frames. That is wrong, and visibly so: it time-warps any source whose
covered span is not the longest. A track covering frames 0–180 stretched across
0–200 replays its motion at nine tenths speed, so the derived path leads it early
and lags it late, wandering off the line between the two features it is supposed
to sit between. Two features in one shot share a clock. Averaging them has to
respect it, and a frame is the clock.

What differing lengths mean instead
-----------------------------------
Not a resampling, but a question about which frames the answer covers
(:data:`SPAN_MODES`):

``overlap``
    Only the frames *every* source has a position on. Every sample of the result
    is then a true mean of all of them. This is the default, because a path that
    is sometimes the midpoint of two features and sometimes just one of them is
    not one thing.
``union``
    Every frame *any* source has a position on, averaging whichever are present.
    More coverage, at the cost of a step wherever a source starts or stops
    contributing — the derived path jumps toward the ones still there.

Everything here is pure array work on :class:`~stable.core.types.PointTrack`, on
the clip's own timeline: the result spans the whole clip with ``NaN`` outside the
frames it covers, exactly like every other track (see :mod:`stable.core.types`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import ContractError
from .types import PointTrack

__all__ = ["SPAN_MODES", "DEFAULT_SPAN_MODE", "covered_span", "mean_track"]

SPAN_MODES: tuple[str, ...] = ("overlap", "union")
"""Which frames a derived path covers when its sources do not agree."""

DEFAULT_SPAN_MODE = "overlap"
"""Only the frames every source covers — where the result is a true mean."""


def covered_span(track: PointTrack) -> tuple[int, int] | None:
    """Return the inclusive ``(first, last)`` frames this track has a position on.

    A track's array always spans the whole clip with ``NaN`` where nothing was
    tracked (the timeline convention in :mod:`stable.core.types`), so "where is
    this path" is a question about its *covered* frames, not about ``n_frames``
    — which is the same number for every point in the project.

    Args:
        track: The track to measure.

    Returns:
        The first and last frames with a finite position, or ``None`` if the
        track has none at all. Frames *inside* the span with no position (gaps)
        are still part of it: a path with a hole in the middle is one path.
    """
    valid = np.nonzero(np.asarray(track.valid))[0]
    if valid.size == 0:
        return None
    return (int(valid[0]), int(valid[-1]))


def mean_track(tracks: Sequence[PointTrack], mode: str = DEFAULT_SPAN_MODE) -> PointTrack:
    """Return the per-frame mean of several tracks, on the clip's own timeline.

    On each frame the answer is the centroid of wherever the contributing
    sources are on *that frame* — for two tracks, the midpoint of the straight
    line joining them. No source is ever read at a frame other than its own, so
    the derived path cannot lead or lag the features it sits between.

    Args:
        tracks: The contributing tracks. At least one, and all the same length —
            they are slices of one clip's timeline, so a mismatch means they are
            not from the same project.
        mode: One of :data:`SPAN_MODES`. ``overlap`` keeps only the frames every
            source covers; ``union`` keeps every frame any of them covers and
            averages whichever are present there.

    Returns:
        A clip-length :class:`~stable.core.types.PointTrack`: the mean position
        and the mean confidence of the contributing sources on the frames the
        mode keeps, and ``NaN`` / zero / not-visible everywhere else — the same
        shape and the same conventions as a tracked path.

    Raises:
        ContractError: If ``tracks`` is empty, the lengths disagree, or ``mode``
            is not a span mode. A result covering *no* frames is not an error
            here — two tracks that never overlap are a fact about them, and the
            caller is better placed to say so than an array function is.
    """
    if not tracks:
        raise ContractError("cannot average an empty list of tracks")
    if mode not in SPAN_MODES:
        raise ContractError(f"unknown span mode {mode!r}; expected {SPAN_MODES}")
    lengths = {track.n_frames for track in tracks}
    if len(lengths) != 1:
        raise ContractError(
            f"all tracks must share the same clip timeline, got lengths {sorted(lengths)}"
        )
    n_frames = lengths.pop()

    # [N, T] where each source has a position, and [N, T, 2] of where.
    valid = np.stack([np.asarray(track.valid) for track in tracks])
    positions = np.stack([np.asarray(track.points, dtype=np.float64) for track in tracks])
    confidence = np.stack([np.asarray(track.confidence, dtype=np.float64) for track in tracks])

    contributors = valid.sum(axis=0)
    keep = contributors == len(tracks) if mode == "overlap" else contributors > 0
    # A frame nothing covers would divide by zero; it is dropped by `keep`
    # anyway, so the denominator there only has to be something finite.
    divisor = np.where(contributors > 0, contributors, 1).astype(np.float64)

    # The sources that *have* a position on a frame are the ones averaged there.
    # `np.where` selects rather than multiplies, so a NaN in a frame a source did
    # not cover cannot leak into the sum.
    summed = np.where(valid[..., None], positions, 0.0).sum(axis=0)
    mean_position = summed / divisor[:, None]
    mean_confidence = np.where(valid, confidence, 0.0).sum(axis=0) / divisor

    points = np.full((n_frames, 2), np.nan, dtype=np.float64)
    points[keep] = mean_position[keep]
    result_confidence = np.zeros(n_frames, dtype=np.float64)
    result_confidence[keep] = mean_confidence[keep]
    return PointTrack(
        points=points.astype(np.float32),
        confidence=np.clip(result_confidence, 0.0, 1.0).astype(np.float32),
        visible=keep,
    )
