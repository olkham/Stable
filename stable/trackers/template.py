"""Adaptive template matching (NCC) — the other no-GPU classic-CV baseline.

Tracks each query point by normalized cross-correlation (``cv2.TM_CCOEFF_NORMED``)
of a small patch centred on the last known position against a search window in
the next frame, independently per direction from the query's anchor frame. The
template adapts over time — blended a little toward the freshly matched patch
after every high-confidence match — so gradual appearance change (lighting,
slight rotation) does not starve the match; the blend is skipped on
low-confidence frames so a bad match (occlusion, drift) cannot poison the
template with what it just failed to find.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.errors import TrackerError
from ..core.types import Query, TrackResult
from ..core.video import VideoSource
from .base import Direction, ProgressCallback, TrackerPlugin, noop_progress
from .frames import converted_frames
from .progress import RunProgress, planned_steps

__all__ = ["TemplateTracker"]


def _cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Returns:
        The imported ``cv2`` module.

    Raises:
        TrackerError: If OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise TrackerError(
            "TemplateTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class TemplateTracker(TrackerPlugin):
    """NCC template matcher with an adaptively updated template.

    Args:
        template_radius: Half-size of the square template patch, in pixels
            (patch side is ``2 * template_radius + 1``).
        search_radius: How far, in pixels, beyond the template size to search
            around the last known position each frame.
        update_rate: Blend factor toward the newly matched patch after a
            high-confidence match, in ``[0, 1]``; ``0`` disables adaptation.
        update_confidence: Minimum match score required before the template is
            updated at all, in ``[0, 1]`` — guards against learning an
            occluded/wrong patch.
    """

    name = "template"
    display_name = "Template Matching (NCC)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        template_radius: int = 12,
        search_radius: int = 24,
        update_rate: float = 0.1,
        update_confidence: float = 0.6,
    ) -> None:
        """Create a template tracker; see the class docstring for the arguments."""
        super().__init__()
        self.template_radius = int(template_radius)
        self.search_radius = int(search_radius)
        self.update_rate = float(update_rate)
        self.update_confidence = float(update_confidence)

    @classmethod
    def is_available(cls) -> bool:
        """Whether OpenCV is importable in this environment."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Track every query point with NCC template matching, one run, no merging.

        Each query must lie inside ``frame_range`` and far enough from the
        frame edge to extract a full template patch.

        Args:
            video: Clip to read frames from.
            queries: Anchor positions to track, one per point.
            frame_range: ``(start, end)`` half-open absolute frame range.
            direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
            progress: Progress reporter; called once per frame stepped,
                plus a final 1.0.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``, in query order.

        Raises:
            TrackerError: If OpenCV is unavailable, the range is invalid, a
                query frame lies outside ``frame_range``, or a query sits too
                close to the frame edge to build a template.
        """
        cv2 = _cv2()
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

        points = np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        visible = np.zeros((len(queries), n_frames), dtype=bool)

        # Bounded, LRU: an unbounded dict here held the whole clip's grayscale
        # copy for the length of the run (see stable/trackers/frames.py).
        gray = converted_frames(video, lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

        # Progress is counted in frames, not in queries: a server-side run has
        # exactly one query, so reporting per query reported nothing until the
        # run was over (see stable/trackers/progress.py).
        reporter = RunProgress(progress, total=planned_steps(queries, (start, end), direction))

        for qi, query in enumerate(queries):
            if not start <= query.frame < end:
                raise TrackerError(
                    f"query frame {query.frame} must lie inside frame_range [{start}, {end})"
                )
            anchor_gray = gray(query.frame)
            anchor_patch = self._extract_patch(anchor_gray, query.position)
            if anchor_patch is None:
                raise TrackerError(
                    f"query at frame {query.frame} position {query.position} is too close to the "
                    f"frame edge for template_radius={self.template_radius}"
                )
            points[qi, query.frame] = query.position
            confidence[qi, query.frame] = 1.0
            visible[qi, query.frame] = True

            if direction in ("fwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query.position,
                    anchor_patch,
                    reporter.steps(range(query.frame + 1, end)),
                    points,
                    confidence,
                    visible,
                )
            if direction in ("bwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query.position,
                    anchor_patch,
                    reporter.steps(range(query.frame - 1, start - 1, -1)),
                    points,
                    confidence,
                    visible,
                )

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)

    def _extract_patch(
        self, gray_frame: np.ndarray, position: tuple[float, float]
    ) -> np.ndarray | None:
        """Return the ``float32`` patch centred at ``position``, or ``None`` if out of bounds."""
        radius = self.template_radius
        height, width = gray_frame.shape[:2]
        cx, cy = int(round(position[0])), int(round(position[1]))
        x0, x1 = cx - radius, cx + radius + 1
        y0, y1 = cy - radius, cy + radius + 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        return gray_frame[y0:y1, x0:x1].astype(np.float32)

    def _run_direction(
        self,
        cv2: Any,
        gray: Any,
        qi: int,
        anchor_position: tuple[float, float],
        anchor_patch: np.ndarray,
        indices: Iterable[int],
        points: np.ndarray,
        confidence: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        """Search frame by frame from ``anchor_position`` across ``indices``.

        Stops early (leaving the remaining frames ``NaN``) once the search
        window can no longer be built near the frame edge; a low match score
        drops confidence and ``visible`` without stopping, since occlusion can
        end and the point can be found again.
        """
        template = anchor_patch
        position = anchor_position
        radius = self.template_radius
        margin = self.search_radius + radius

        for index in indices:
            frame_gray = gray(index)
            height, width = frame_gray.shape[:2]
            cx, cy = int(round(position[0])), int(round(position[1]))
            x0, y0 = max(cx - margin, 0), max(cy - margin, 0)
            x1, y1 = min(cx + margin + 1, width), min(cy + margin + 1, height)
            window = frame_gray[y0:y1, x0:x1]
            th, tw = template.shape[:2]
            if window.shape[0] < th or window.shape[1] < tw:
                break

            result = cv2.matchTemplate(window.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            conf = 0.0 if math.isnan(max_val) else max(0.0, min(1.0, float(max_val)))

            if conf > 0.0:
                position = (float(x0 + max_loc[0] + radius), float(y0 + max_loc[1] + radius))
            points[qi, index] = position
            confidence[qi, index] = conf
            visible[qi, index] = conf > 0.0

            if conf >= self.update_confidence:
                new_patch = self._extract_patch(frame_gray, position)
                if new_patch is not None:
                    template = (1.0 - self.update_rate) * template + self.update_rate * new_patch
