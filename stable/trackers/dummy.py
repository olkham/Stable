"""A deterministic, dependency-free tracker used by tests and demos.

``dummy`` exists so the whole pipeline — registry, merge, project file, later the
solver and the server's job queue — can be exercised without PyTorch, without a
GPU and without a video file. It is registered as a real entry point so that
entry-point discovery itself is covered end to end.

Behaviour (fully deterministic, no RNG):

* If the source exposes ``true_position`` (see
  :class:`~stable.core.video.SyntheticVideoSource`) the tracker follows that
  ground truth, offset so it passes exactly through the query position. This
  makes it a "perfect tracker" for synthetic clips.
* Otherwise it moves the query position at a constant ``drift`` velocity.
* Confidence decays linearly with distance from the query frame, which is what
  real trackers do and what the merge falloff is designed around.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import TrackerError
from ..core.types import Query, TrackResult
from ..core.video import VideoSource
from .base import Direction, ProgressCallback, TrackerPlugin, noop_progress

__all__ = ["DummyTracker"]


class DummyTracker(TrackerPlugin):
    """A synthetic tracker with no dependencies and no randomness.

    Args:
        drift: Per-frame ``(dx, dy)`` velocity applied away from the query
            frame when the source has no ground truth. Handy for simulating a
            drifting track in multi-anchor tests.
        confidence_falloff: Confidence lost per frame of distance from the query.
        min_confidence: Floor the decaying confidence cannot go below.
        follow_truth: Follow the source's ``true_position`` when it has one.
    """

    name = "dummy"
    display_name = "Dummy (testing)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        drift: tuple[float, float] = (0.0, 0.0),
        confidence_falloff: float = 0.01,
        min_confidence: float = 0.05,
        follow_truth: bool = True,
    ) -> None:
        """Create a dummy tracker; see the class docstring for the arguments."""
        super().__init__()
        self.drift = (float(drift[0]), float(drift[1]))
        self.confidence_falloff = float(confidence_falloff)
        self.min_confidence = float(min_confidence)
        self.follow_truth = bool(follow_truth)

    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Produce a deterministic track for every query.

        Args:
            video: Any :class:`~stable.core.video.VideoSource`; only
                ``frame_count`` (and optionally ``true_position``) is read.
            queries: One anchor per point.
            frame_range: Half-open ``(start, end)`` absolute frame range.
            direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
            progress: Progress reporter, called once per point plus a final 1.0.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``.

        Raises:
            TrackerError: If the range is empty or falls outside the clip, the
                direction is unknown, or a query sits outside the clip.
        """
        if not self.is_prepared:
            self.prepare("cpu")
        n_frames = int(video.frame_count)
        start, end = int(frame_range[0]), int(frame_range[1])
        if direction not in ("fwd", "bwd", "bidir"):
            raise TrackerError(f"unknown direction {direction!r}")
        if end <= start:
            raise TrackerError(f"frame_range must be non-empty, got [{start}, {end})")
        if start < 0 or end > n_frames:
            raise TrackerError(f"frame_range [{start}, {end}) is outside the clip (T={n_frames})")
        if not queries:
            return TrackResult.empty(0, n_frames)

        truth = getattr(video, "true_position", None) if self.follow_truth else None
        points = np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        visible = np.zeros((len(queries), n_frames), dtype=bool)

        for index, query in enumerate(queries):
            if not 0 <= query.frame < n_frames:
                raise TrackerError(
                    f"query frame {query.frame} is outside the clip (T={n_frames})"
                )
            if direction == "fwd":
                lo, hi = max(start, query.frame), end
            elif direction == "bwd":
                lo, hi = start, min(end, query.frame + 1)
            else:
                lo, hi = start, end
            if hi <= lo:
                progress((index + 1) / len(queries))
                continue

            frames = np.arange(lo, hi)
            if truth is not None:
                anchor_truth = np.asarray(truth(query.frame), dtype=np.float64)
                path = np.asarray([truth(int(f)) for f in frames], dtype=np.float64)
                positions = path + (np.asarray(query.position, dtype=np.float64) - anchor_truth)
            else:
                offsets = (frames - query.frame).astype(np.float64)[:, None]
                positions = np.asarray(query.position, dtype=np.float64) + offsets * np.asarray(
                    self.drift, dtype=np.float64
                )

            distance = np.abs(frames - query.frame).astype(np.float64)
            scores = np.clip(1.0 - self.confidence_falloff * distance, self.min_confidence, 1.0)
            points[index, lo:hi] = positions
            confidence[index, lo:hi] = scores
            visible[index, lo:hi] = True
            progress((index + 1) / len(queries), f"point {index + 1}/{len(queries)}")

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)
