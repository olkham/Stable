"""Mean-shift colour-histogram tracker — mode-seeking, not correlation.

Todo 26 asks for "a mean-shift tracker". Every other classic-CV plugin in this
package (``template``, ``correlation``, and the measurement source inside
``kalman``/``particle``) is fundamentally the same operation: compare a
reference patch against candidate patches and pick (or weight by) the best
match. Mean-shift (Comaniciu & Meer, 2000; the algorithm behind OpenCV's
``camshift`` demo) is a genuinely different idea:

#. Build a **colour-distribution model** of the target — a histogram over a
   colour feature (here, Hue-Saturation), not a pixel template. Two patches
   with identical colour statistics but different internal layouts (a
   rearranged texture, a rotated non-symmetric marker) look *identical* to
   this model, which is precisely the point: mean-shift trades layout
   sensitivity for robustness to in-plane rotation and non-rigid deformation
   that a fixed pixel template cannot tolerate at all.
#. Back-project that histogram onto the next frame: replace every pixel with
   how well its colour matches the model, producing a probability-like
   density map.
#. **Climb the density** from the last known window by iterated mean-shift —
   repeatedly move the window to the centroid of the density within it —
   until it converges on a mode (a local maximum), rather than *searching* a
   grid of candidate offsets the way NCC does.

This plugin builds the model histogram from a **small kernel-weighted region
at the centre of the search window**, not the whole window — a hard-radius
mask standing in for the smooth (e.g. Epanechnikov) kernel the original
algorithm weights the histogram by. Without it, a search window sized for a
comfortable capture range also bakes a lot of *background* colour into the
model, which measurably hurts convergence once the object has moved; giving
the histogram only the object's own footprint is what makes mode-seeking
reliable. Confidence comes from the Bhattacharyya similarity between the
model histogram and the converged window's own histogram — a standard
distributional-distance measure for exactly this kind of histogram, not a
per-pixel error.
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
from .confidence import normalize_scores, visibility_from_confidence
from .frames import converted_frames
from .progress import RunProgress, planned_steps

__all__ = ["MeanShiftTracker"]


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
            "MeanShiftTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class MeanShiftTracker(TrackerPlugin):
    """Hue-Saturation histogram mean-shift tracker.

    Args:
        window_radius: Half-size of the square search window mean-shift
            iterates within (window side is ``2 * window_radius + 1``). Large
            enough to comfortably contain the object at its new position each
            frame — mean-shift climbs the density *within* this window, it does
            not search beyond it.
        histogram_radius: Half-size of the small, centred region the model
            histogram is built from — the hard-radius stand-in for mean-shift's
            usual smooth kernel weighting (see the module docstring). Size this
            to the object's own visible footprint, not the search window: too
            large and the model absorbs background colour, which is what
            actually breaks convergence, not an insufficient search radius.
        hue_bins: Number of histogram bins over Hue (``0..180`` in OpenCV's
            8-bit HSV).
        sat_bins: Number of histogram bins over Saturation (``0..256``).
        max_iterations: Cap on mean-shift iterations per frame.
        min_shift: Convergence threshold, in pixels; iteration stops once a
            step moves the window centre less than this.
        confidence_range: ``(low, high)`` calibration applied to the
            Bhattacharyya similarity between the model and converged window
            histograms.
    """

    name = "meanshift"
    display_name = "Mean-Shift (colour histogram)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        window_radius: int = 10,
        histogram_radius: int = 2,
        hue_bins: int = 24,
        sat_bins: int = 32,
        max_iterations: int = 20,
        min_shift: float = 0.3,
        confidence_range: tuple[float, float] = (0.1, 0.6),
    ) -> None:
        """Create a mean-shift tracker; see the class docstring for the arguments."""
        super().__init__()
        self.window_radius = int(window_radius)
        self.histogram_radius = int(histogram_radius)
        self.hue_bins = int(hue_bins)
        self.sat_bins = int(sat_bins)
        self.max_iterations = int(max_iterations)
        self.min_shift = float(min_shift)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))
        if self.histogram_radius > self.window_radius:
            raise TrackerError(
                f"histogram_radius ({self.histogram_radius}) must not exceed "
                f"window_radius ({self.window_radius})"
            )

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
        """Track every query point by mean-shift, one run, no merging.

        Each query must lie inside ``frame_range`` and far enough from the
        frame edge to build a search window.

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
                close to the frame edge to build a search window.
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

        # Bounded, LRU. An HSV frame is three times the size of a grayscale one,
        # so the unbounded dict this replaces exhausted memory on an ordinary HD
        # shot long before the other trackers did (see stable/trackers/frames.py).
        hsv = converted_frames(video, lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))

        mask_template = self._kernel_mask()
        ranges = [0, 180, 0, 256]
        hist_bins = [self.hue_bins, self.sat_bins]

        # Progress is counted in frames, not in queries: a server-side run has
        # exactly one query, so reporting per query reported nothing until the
        # run was over (see stable/trackers/progress.py).
        reporter = RunProgress(progress, total=planned_steps(queries, (start, end), direction))

        for qi, query in enumerate(queries):
            if not start <= query.frame < end:
                raise TrackerError(
                    f"query frame {query.frame} must lie inside frame_range [{start}, {end})"
                )
            anchor_hsv = hsv(query.frame)
            model_hist = self._extract_histogram(
                cv2, anchor_hsv, query.position, mask_template, hist_bins, ranges
            )
            if model_hist is None:
                raise TrackerError(
                    f"query at frame {query.frame} position {query.position} is too close to "
                    f"the frame edge for window_radius={self.window_radius}"
                )
            points[qi, query.frame] = query.position
            raw_confidence[qi, query.frame] = 1.0
            touched[qi, query.frame] = True

            if direction in ("fwd", "bidir"):
                self._run_direction(
                    cv2,
                    hsv,
                    qi,
                    query.position,
                    model_hist,
                    mask_template,
                    hist_bins,
                    ranges,
                    reporter.steps(range(query.frame + 1, end)),
                    points,
                    raw_confidence,
                    touched,
                )
            if direction in ("bwd", "bidir"):
                self._run_direction(
                    cv2,
                    hsv,
                    qi,
                    query.position,
                    model_hist,
                    mask_template,
                    hist_bins,
                    ranges,
                    reporter.steps(range(query.frame - 1, start - 1, -1)),
                    points,
                    raw_confidence,
                    touched,
                )

        low, high = self.confidence_range
        confidence = normalize_scores(raw_confidence, low=low, high=high)
        confidence = np.where(touched, confidence, 0.0).astype(np.float32)
        visible = visibility_from_confidence(confidence, 0.5) & touched

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)

    def _kernel_mask(self) -> np.ndarray:
        """Return the fixed hard-radius mask used to build every model histogram.

        A circular mask of :attr:`histogram_radius` centred in a
        :attr:`window_radius`-sized square — see the module docstring for why
        this stands in for mean-shift's usual smooth kernel weighting.
        """
        size = 2 * self.window_radius + 1
        centre = self.window_radius
        ys, xs = np.mgrid[0:size, 0:size]
        distance = np.sqrt((xs - centre) ** 2 + (ys - centre) ** 2)
        return (distance <= self.histogram_radius).astype(np.uint8) * 255

    def _extract_histogram(
        self,
        cv2: Any,
        hsv_frame: np.ndarray,
        position: tuple[float, float],
        mask: np.ndarray,
        hist_bins: list[int],
        ranges: list[int],
    ) -> np.ndarray | None:
        """Return the normalised Hue-Saturation histogram of the window at ``position``.

        Returns:
            The histogram, or ``None`` if the window does not fit inside the frame.
        """
        window = self._extract_window(hsv_frame, position)
        if window is None:
            return None
        hist = cv2.calcHist([window], [0, 1], mask, hist_bins, ranges)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def _extract_window(
        self, hsv_frame: np.ndarray, position: tuple[float, float]
    ) -> np.ndarray | None:
        """Return the ``window_radius``-sized HSV crop centred at ``position``."""
        radius = self.window_radius
        height, width = hsv_frame.shape[:2]
        cx, cy = int(round(position[0])), int(round(position[1]))
        x0, x1 = cx - radius, cx + radius + 1
        y0, y1 = cy - radius, cy + radius + 1
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        return hsv_frame[y0:y1, x0:x1]

    def _run_direction(
        self,
        cv2: Any,
        hsv: Any,
        qi: int,
        anchor_position: tuple[float, float],
        model_hist: np.ndarray,
        mask_template: np.ndarray,
        hist_bins: list[int],
        ranges: list[int],
        indices: Iterable[int],
        points: np.ndarray,
        raw_confidence: np.ndarray,
        touched: np.ndarray,
    ) -> None:
        """Step mean-shift one frame at a time from ``anchor_position``.

        Stops early (leaving the remaining frames untouched) once the search
        window can no longer be built near the frame edge; a low histogram
        similarity drops confidence without stopping, matching every other
        plugin's policy of letting the estimator carry the point through a
        short occlusion.
        """
        radius = self.window_radius
        position = anchor_position

        for index in indices:
            frame_hsv = hsv(index)
            height, width = frame_hsv.shape[:2]
            window = self._extract_window(frame_hsv, position)
            if window is None:
                break

            back_projection = cv2.calcBackProject([window], [0, 1], model_hist, ranges, 1)
            centre = self._mean_shift_converge(back_projection)
            cx, cy = int(round(position[0])), int(round(position[1]))
            new_position = (
                float(np.clip(cx - radius + centre[0], 0.0, width - 1.0)),
                float(np.clip(cy - radius + centre[1], 0.0, height - 1.0)),
            )

            similarity = self._similarity(cv2, frame_hsv, new_position, mask_template, model_hist)

            points[qi, index] = new_position
            raw_confidence[qi, index] = similarity
            touched[qi, index] = True
            position = new_position

    def _mean_shift_converge(self, back_projection: np.ndarray) -> tuple[float, float]:
        """Iterate mean-shift within ``back_projection`` and return the converged centre.

        Operates in the local window's own pixel coordinates (``[0, side)``),
        starting from the window's centre — the last known position, since the
        window is always re-cropped centred there each frame.

        Args:
            back_projection: ``[side, side]`` probability-like density map.

        Returns:
            ``(x, y)`` converged centre, in the window's local coordinates.
        """
        side = back_projection.shape[0]
        centre = float(side // 2)
        cx, cy = centre, centre
        for _ in range(self.max_iterations):
            x0, y0 = max(int(cx) - self.window_radius, 0), max(int(cy) - self.window_radius, 0)
            x1 = min(int(cx) + self.window_radius + 1, side)
            y1 = min(int(cy) + self.window_radius + 1, side)
            patch = back_projection[y0:y1, x0:x1].astype(np.float64)
            total = patch.sum()
            if total <= 1e-6:
                break
            ys, xs = np.mgrid[y0:y1, x0:x1]
            new_cx = float((xs * patch).sum() / total)
            new_cy = float((ys * patch).sum() / total)
            shift = math.hypot(new_cx - cx, new_cy - cy)
            cx, cy = new_cx, new_cy
            if shift < self.min_shift:
                break
        return (cx, cy)

    def _similarity(
        self,
        cv2: Any,
        frame_hsv: np.ndarray,
        position: tuple[float, float],
        mask_template: np.ndarray,
        model_hist: np.ndarray,
    ) -> float:
        """Return the Bhattacharyya similarity in ``[0, 1]`` between the model and this window.

        ``1`` means the converged window's colour distribution is identical to
        the model's; ``0`` means they share nothing — the standard
        distributional confidence measure for a histogram-based tracker (see
        the module docstring), used here in place of a per-pixel match score.
        """
        hist_bins = [self.hue_bins, self.sat_bins]
        current_hist = self._extract_histogram(
            cv2, frame_hsv, position, mask_template, hist_bins, [0, 180, 0, 256]
        )
        if current_hist is None:
            return 0.0
        distance = cv2.compareHist(model_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
        if math.isnan(distance):
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distance)))
