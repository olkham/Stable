"""FFT phase-correlation tracker — a classic-CV baseline distinct from ``template``.

:mod:`stable.trackers.template` already covers normalized cross-correlation (NCC):
``cv2.matchTemplate`` with an **explicit spatial search** — the template patch is
compared against every offset in a search window around the last known position,
an ``O(search_area x template_area)`` scan.

This tracker uses a different, older algorithm for the same job: Kuglin & Hines'
*phase correlation*. Two equally-sized patches at the **same** location in
consecutive frames are Fourier-transformed; the phase of their cross-power
spectrum encodes the translational shift between them directly, recovered by one
inverse FFT and a sub-pixel peak fit — ``O(n log n)``, with **no spatial search
at all**. Two properties fall out of that which NCC does not share:

* It is **contrast/brightness invariant by construction** — only the phase of the
  spectrum is used, the magnitude (which is where uniform lighting changes show
  up) is discarded entirely, rather than merely normalised the way NCC's own
  score is.
* The result is a *displacement*, not a match score at discrete offsets, so
  sub-pixel accuracy falls out of the peak's parabolic interpolation rather than
  needing the template search grid to be dense.

OpenCV's ``cv2.phaseCorrelate`` implements the whole thing (FFT, cross-power
spectrum, sub-pixel peak) and additionally reports ``response`` — the signal
power concentrated at that peak, in ``[0, 1]`` — which stands in for NCC's peak
score as this plugin's confidence signal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.errors import TrackerError
from ..core.types import Query, TrackResult
from ..core.video import VideoSource
from .base import Direction, ProgressCallback, TrackerPlugin, noop_progress
from .confidence import normalize_scores, visibility_from_confidence

__all__ = ["CorrelationTracker"]


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
            "CorrelationTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class CorrelationTracker(TrackerPlugin):
    """FFT phase-correlation tracker with an adaptively updated reference patch.

    Args:
        patch_radius: Half-size of the square patch correlated between frames
            (patch side is ``2 * patch_radius + 1``). Must comfortably exceed the
            largest expected per-frame displacement — unlike ``template``, this
            tracker never searches a wider window, it only measures the shift
            between two same-sized, same-location patches.
        update_rate: Blend factor toward the newly matched patch after a
            high-confidence match, in ``[0, 1]``; ``0`` disables adaptation.
        update_confidence: Minimum *raw* response required before the reference
            patch is updated at all — guards against learning an occluded or
            drifted patch, exactly as ``template``'s ``update_confidence`` does.
        confidence_range: ``(low, high)`` calibration applied to
            ``cv2.phaseCorrelate``'s own ``response`` (already in ``[0, 1]``, but
            still needs the shared dead-zone/saturation treatment — see
            CONTRIBUTING.md's "Calibrating confidence").
    """

    name = "correlation"
    display_name = "Phase Correlation (FFT)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        patch_radius: int = 14,
        update_rate: float = 0.05,
        update_confidence: float = 0.5,
        confidence_range: tuple[float, float] = (0.05, 0.5),
    ) -> None:
        """Create a correlation tracker; see the class docstring for the arguments."""
        super().__init__()
        self.patch_radius = int(patch_radius)
        self.update_rate = float(update_rate)
        self.update_confidence = float(update_confidence)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))

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
        """Track every query point by FFT phase correlation, one run, no merging.

        Each query must lie inside ``frame_range`` and far enough from the frame
        edge to extract a full patch.

        Args:
            video: Clip to read frames from.
            queries: Anchor positions to track, one per point.
            frame_range: ``(start, end)`` half-open absolute frame range.
            direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
            progress: Progress reporter; called once per point plus a final 1.0.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``, in query order.

        Raises:
            TrackerError: If OpenCV is unavailable, the range is invalid, a query
                frame lies outside ``frame_range``, or a query sits too close to
                the frame edge to build a patch.
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
        raw_confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        touched = np.zeros((len(queries), n_frames), dtype=bool)

        size = 2 * self.patch_radius + 1
        window = cv2.createHanningWindow((size, size), cv2.CV_32F)

        gray_cache: dict[int, np.ndarray] = {}

        def gray(index: int) -> np.ndarray:
            cached = gray_cache.get(index)
            if cached is None:
                cached = cv2.cvtColor(video.get_frame(index), cv2.COLOR_RGB2GRAY)
                gray_cache[index] = cached
            return cached

        for qi, query in enumerate(queries):
            if not start <= query.frame < end:
                raise TrackerError(
                    f"query frame {query.frame} must lie inside frame_range [{start}, {end})"
                )
            anchor_gray = gray(query.frame)
            anchor_patch = self._extract_patch(anchor_gray, query.position)
            if anchor_patch is None:
                raise TrackerError(
                    f"query at frame {query.frame} position {query.position} is too close to "
                    f"the frame edge for patch_radius={self.patch_radius}"
                )
            points[qi, query.frame] = query.position
            raw_confidence[qi, query.frame] = 1.0
            touched[qi, query.frame] = True

            if direction in ("fwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    window,
                    qi,
                    query.position,
                    anchor_patch,
                    range(query.frame + 1, end),
                    points,
                    raw_confidence,
                    touched,
                )
            if direction in ("bwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    window,
                    qi,
                    query.position,
                    anchor_patch,
                    range(query.frame - 1, start - 1, -1),
                    points,
                    raw_confidence,
                    touched,
                )
            progress((qi + 1) / len(queries), f"point {qi + 1}/{len(queries)}")

        low, high = self.confidence_range
        confidence = normalize_scores(raw_confidence, low=low, high=high)
        confidence = np.where(touched, confidence, 0.0).astype(np.float32)
        visible = visibility_from_confidence(confidence, 0.5) & touched

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)

    def _extract_patch(
        self, gray_frame: np.ndarray, position: tuple[float, float]
    ) -> np.ndarray | None:
        """Return the ``float32`` patch centred at ``position``, or ``None`` if out of bounds."""
        radius = self.patch_radius
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
        window: np.ndarray,
        qi: int,
        anchor_position: tuple[float, float],
        anchor_patch: np.ndarray,
        indices: range,
        points: np.ndarray,
        raw_confidence: np.ndarray,
        touched: np.ndarray,
    ) -> None:
        """Step phase correlation one frame at a time from ``anchor_position``.

        Stops early (leaving the remaining frames untouched) once the patch can
        no longer be built near the frame edge; a low response drops confidence
        without stopping, since occlusion can end and the point can be found
        again — the same policy ``lk`` and ``template`` use.
        """
        template = anchor_patch
        position = anchor_position

        for index in indices:
            frame_gray = gray(index)
            patch = self._extract_patch(frame_gray, position)
            if patch is None:
                break

            (dx, dy), response = cv2.phaseCorrelate(template, patch, window)
            if not (math.isfinite(dx) and math.isfinite(dy)):
                dx = dy = 0.0
                response = 0.0
            response = 0.0 if math.isnan(response) else max(0.0, min(1.0, float(response)))

            position = (position[0] + dx, position[1] + dy)
            points[qi, index] = position
            raw_confidence[qi, index] = response
            touched[qi, index] = True

            if response >= self.update_confidence:
                new_patch = self._extract_patch(frame_gray, position)
                if new_patch is not None:
                    template = (1.0 - self.update_rate) * template + self.update_rate * new_patch
