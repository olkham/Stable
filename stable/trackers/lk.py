"""Pyramidal Lucas-Kanade optical flow — the no-GPU classic-CV baseline.

Tracks each query point frame-to-frame with OpenCV's pyramidal LK
(``cv2.calcOpticalFlowPyrLK``), independently per direction, starting from the
query's anchor frame. Confidence comes from the forward-backward (FB) check
(Kalal et al.): track a point forward one frame, then track the result
backward to the previous frame; the distance between where it lands and where
it started is the FB error. A small FB error means the flow is self-consistent
(reliable); a large one — the classic signature of occlusion or the point
leaving frame — drives confidence toward zero and ``visible`` to ``False``,
without needing OpenCV to report outright failure.

Optional feature (re)selection (see :mod:`stable.trackers.klt`)
-----------------------------------------------------------------
Plain LK, as implemented here by default, blindly follows the optical-flow
solution from wherever the previous frame put it — it has no notion of
whether that pixel is actually a *good* feature to track (Shi & Tomasi's
minimum-eigenvalue criterion, the "T" in "KLT"), so small numerical drift
compounds silently on a feature that is not a strong corner. Setting
:attr:`LKTracker.redetect_interval` above ``0`` turns on periodic re-detection:
every ``redetect_interval`` frames, ``cv2.goodFeaturesToTrack`` looks for the
strongest corner within :attr:`LKTracker.redetect_search_radius` pixels of the
current tracked position, and — only if one exists within
:attr:`LKTracker.redetect_max_shift` pixels — the tracked position (and the
seed LK continues from) snaps to it. This never touches a query's own anchor
frame (the contract's exact-passthrough guarantee is unaffected), only the
frames in between. It is off by default (``redetect_interval=0``) so plain
``lk`` is unchanged; :class:`~stable.trackers.klt.KLTTracker` is this same
class with it turned on, registered under its own name.
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

__all__ = ["LKTracker"]


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
            "LKTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class LKTracker(TrackerPlugin):
    """Pyramidal Lucas-Kanade tracker with forward-backward confidence.

    Args:
        win_size: Side length in pixels of the LK search window.
        max_level: Number of pyramid levels (0 = no pyramid).
        max_iterations: Iteration cap for the LK termination criteria.
        epsilon: Convergence threshold for the LK termination criteria.
        fb_threshold: Forward-backward error, in pixels, at which confidence
            reaches zero. Confidence falls off linearly from ``1.0`` at
            ``fb_error == 0`` to ``0.0`` at ``fb_error >= fb_threshold``.
        redetect_interval: Frames between Shi-Tomasi corner re-detection
            attempts; ``0`` (the default) disables it entirely, so plain
            ``lk`` behaves exactly as before. See the module docstring and
            :mod:`stable.trackers.klt`, which enables this by default.
        redetect_search_radius: Half-size, in pixels, of the neighbourhood
            searched for a stronger corner around the current position.
        redetect_quality_level: ``cv2.goodFeaturesToTrack``'s ``qualityLevel``
            — corners weaker than this fraction of the neighbourhood's best
            corner response are not considered at all.
        redetect_min_distance: ``cv2.goodFeaturesToTrack``'s minimum
            separation between reported corners, in pixels.
        redetect_max_shift: How far, in pixels, a re-detected corner may be
            from the current position before it is trusted. Keeps
            re-detection a *drift correction* — snapping to a nearby stronger
            corner — rather than letting it jump to an unrelated feature.
    """

    name = "lk"
    display_name = "Lucas-Kanade (optical flow)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        win_size: int = 21,
        max_level: int = 3,
        max_iterations: int = 30,
        epsilon: float = 0.01,
        fb_threshold: float = 2.0,
        redetect_interval: int = 0,
        redetect_search_radius: int = 15,
        redetect_quality_level: float = 0.1,
        redetect_min_distance: float = 3.0,
        redetect_max_shift: float = 4.0,
    ) -> None:
        """Create an LK tracker; see the class docstring for the arguments."""
        super().__init__()
        self.win_size = int(win_size)
        self.max_level = int(max_level)
        self.max_iterations = int(max_iterations)
        self.epsilon = float(epsilon)
        self.fb_threshold = float(fb_threshold)
        self.redetect_interval = int(redetect_interval)
        self.redetect_search_radius = int(redetect_search_radius)
        self.redetect_quality_level = float(redetect_quality_level)
        self.redetect_min_distance = float(redetect_min_distance)
        self.redetect_max_shift = float(redetect_max_shift)

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
        """Track every query point with pyramidal LK, one run, no merging.

        Each query must lie inside ``frame_range``: a frame-stepping tracker
        has no way to "arrive" at a query frame outside the range it is asked
        to cover, unlike a closed-form generator.

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
            TrackerError: If OpenCV is unavailable, the range is invalid, or a
                query frame lies outside ``frame_range``.
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

        lk_params: dict[str, Any] = {
            "winSize": (self.win_size, self.win_size),
            "maxLevel": self.max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.max_iterations,
                self.epsilon,
            ),
        }

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
            points[qi, query.frame] = query.position
            confidence[qi, query.frame] = 1.0
            visible[qi, query.frame] = True

            if direction in ("fwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query,
                    reporter.steps(range(query.frame + 1, end)),
                    points,
                    confidence,
                    visible,
                    lk_params,
                )
            if direction in ("bwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query,
                    reporter.steps(range(query.frame - 1, start - 1, -1)),
                    points,
                    confidence,
                    visible,
                    lk_params,
                )

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)

    def _run_direction(
        self,
        cv2: Any,
        gray: Any,
        qi: int,
        query: Query,
        indices: Iterable[int],
        points: np.ndarray,
        confidence: np.ndarray,
        visible: np.ndarray,
        lk_params: dict[str, Any],
    ) -> None:
        """Step LK one frame at a time from ``query`` across ``indices``.

        Stops early (leaving the remaining frames ``NaN``) if OpenCV reports
        the point lost or it leaves the frame; a large forward-backward error
        instead drops confidence and ``visible`` without stopping, since a
        temporarily occluded point can still recover once the occlusion ends.
        """
        prev_gray = gray(query.frame)
        prev_pt = np.array([[list(query.position)]], dtype=np.float32)
        step = 0

        for next_index in indices:
            next_gray = gray(next_index)
            next_pt, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, next_gray, prev_pt, None, **lk_params
            )
            if next_pt is None or status is None or int(status[0, 0]) == 0:
                break

            back_pt, back_status, _ = cv2.calcOpticalFlowPyrLK(
                next_gray, prev_gray, next_pt, None, **lk_params
            )
            if back_pt is None or back_status is None or int(back_status[0, 0]) == 0:
                fb_error = math.inf
            else:
                delta = back_pt[0, 0] - prev_pt[0, 0]
                fb_error = float(np.hypot(float(delta[0]), float(delta[1])))

            x, y = (float(v) for v in next_pt[0, 0])
            if not (math.isfinite(x) and math.isfinite(y)):
                break
            height, width = next_gray.shape[:2]
            if not (0.0 <= x < width and 0.0 <= y < height):
                break

            conf = (
                0.0
                if math.isinf(fb_error)
                else max(0.0, min(1.0, 1.0 - fb_error / self.fb_threshold))
            )
            step += 1
            if self.redetect_interval > 0 and step % self.redetect_interval == 0 and conf > 0.3:
                redetected = self._redetect_corner(cv2, next_gray, (x, y))
                if redetected is not None:
                    x, y = redetected

            points[qi, next_index] = (x, y)
            confidence[qi, next_index] = conf
            visible[qi, next_index] = conf > 0.0

            prev_gray, prev_pt = next_gray, np.array([[[x, y]]], dtype=np.float32)

    def _redetect_corner(
        self, cv2: Any, gray_frame: np.ndarray, position: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Return the strongest Shi-Tomasi corner near ``position``, if one is close enough.

        Searches a :attr:`redetect_search_radius`-sized neighbourhood of
        ``position`` with ``cv2.goodFeaturesToTrack`` (the minimum-eigenvalue
        criterion Tomasi added to Lucas-Kanade, hence "KLT") and returns its
        best corner in absolute frame coordinates, but only if it lies within
        :attr:`redetect_max_shift` pixels — far enough away and it is almost
        certainly a different feature, not this one having drifted.

        Args:
            cv2: The imported ``cv2`` module (passed in, like every other
                private helper here, rather than importing it again).
            gray_frame: Full grayscale frame to search within.
            position: Current tracked position, in absolute frame coordinates.

        Returns:
            The corrected ``(x, y)``, or ``None`` if no corner qualifies (the
            neighbourhood has nothing better, or is too close to the frame
            edge to search at all).
        """
        radius = self.redetect_search_radius
        height, width = gray_frame.shape[:2]
        cx, cy = int(round(position[0])), int(round(position[1]))
        x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
        x1, y1 = min(cx + radius + 1, width), min(cy + radius + 1, height)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        roi = gray_frame[y0:y1, x0:x1]
        corners = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=5,
            qualityLevel=self.redetect_quality_level,
            minDistance=self.redetect_min_distance,
            blockSize=7,
        )
        if corners is None or len(corners) == 0:
            return None
        absolute = corners.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
        distances = np.hypot(absolute[:, 0] - position[0], absolute[:, 1] - position[1])
        best = int(np.argmin(distances))
        if distances[best] > self.redetect_max_shift:
            return None
        return (float(absolute[best, 0]), float(absolute[best, 1]))
