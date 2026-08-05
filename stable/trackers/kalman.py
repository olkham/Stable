"""Constant-velocity Kalman filter tracker — an *estimator*, not an appearance tracker.

Todo 24 asks for "a Kalman filter tracker". A Kalman filter on its own has
nothing to look at: it is a recursive estimator that fuses a motion model with
noisy *measurements* of position, it does not itself know where anything is in
an image. So, as CONTRIBUTING.md's guidance for this todo says, this plugin
pairs the filter with a measurement source — normalized cross-correlation
(NCC) patch matching, the same primitive :mod:`stable.trackers.template` uses —
and the interesting part is how the two combine, not either one alone:

#. **Predict.** A constant-velocity model (state ``[x, y, vx, vy]``) projects
   the *previous* filtered position and velocity forward one frame. This
   prediction, not the last raw measurement, is where the NCC search is
   centred — on a fast-moving point this materially narrows the search
   compared to searching around the last known position the way ``template``
   does, because the filter already has a velocity estimate.
#. **Measure.** NCC match a small patch against a small window around that
   predicted position, exactly as ``template`` does over a wider window.
#. **Update.** The measurement is fused into the state with the Kalman gain,
   weighted by the *measurement noise* — which this plugin scales inversely
   with the NCC score, so a confident match is trusted almost fully and a weak
   one barely nudges the filter's own motion-model prediction. This is the
   payoff of pairing the two: the filter smooths through a bad measurement
   instead of jumping to it the way a pure per-frame matcher must.

Confidence combines *two* independent signals, in the sense
:mod:`stable.trackers.confidence` means it: the raw NCC score (is this actually
the right patch?) and the filter's own post-update positional uncertainty (does
the estimator itself trust this state?). Both must hold for a sample to be
usable, so they are multiplied, then rescaled onto the shared ``[0, 1]`` scale.
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
from .confidence import combine_scores, normalize_scores, visibility_from_confidence
from .frames import converted_frames
from .progress import RunProgress, planned_steps

__all__ = ["KalmanTracker"]


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
            "KalmanTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class _ConstantVelocityKalman:
    """A textbook 2D constant-velocity Kalman filter, position-only measurements.

    State ``x = [px, py, vx, vy]``; one time step is exactly one video frame
    (``dt = 1``), which keeps the transition matrix fixed and dependency-free.

    Args:
        position: Initial ``(x, y)`` position (velocity starts at zero — the
            first frame carries no motion information yet).
        process_noise: Diagonal process-noise variance ``Q``. Larger values let
            the filter follow a manoeuvring target more readily, at the cost of
            trusting noisy measurements more between manoeuvres.
        measurement_noise: Base measurement-noise variance at NCC score ``1.0``;
            see :meth:`update` for how a lower score inflates this.
    """

    def __init__(
        self,
        position: tuple[float, float],
        *,
        process_noise: float,
        measurement_noise: float,
    ) -> None:
        """Initialise the filter at ``position`` with zero velocity."""
        self.x = np.array([position[0], position[1], 0.0, 0.0], dtype=np.float64)
        # Velocity starts unknown; the large initial variance there (not on
        # position, which the anchor gives exactly) lets the first couple of
        # measurements establish it quickly rather than fighting a confident
        # prior about a velocity nobody has observed yet.
        self.P = np.diag([1e-6, 1e-6, 25.0, 25.0])
        self.transition = np.array(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        self.observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.process_noise = np.eye(4) * float(process_noise)
        self.measurement_noise = float(measurement_noise)

    def predict(self) -> tuple[float, float]:
        """Advance the state one frame and return the predicted position.

        Returns:
            The predicted ``(x, y)`` — where the measurement search should be
            centred.
        """
        self.x = self.transition @ self.x
        self.P = self.transition @ self.P @ self.transition.T + self.process_noise
        return (float(self.x[0]), float(self.x[1]))

    def update(self, measurement: tuple[float, float], score: float) -> tuple[float, float, float]:
        """Fuse a position measurement, weighted by how much ``score`` trusts it.

        Args:
            measurement: The ``(x, y)`` position NCC matched.
            score: NCC peak in ``[0, 1]``; the measurement-noise covariance used
                here is ``measurement_noise / max(score, eps)``, so a weak match
                is treated as a noisy measurement (barely moves the state) and a
                strong one as a near-exact one (moves it almost fully there).

        Returns:
            ``(x, y, uncertainty)`` — the filtered position and the trace of the
            post-update positional covariance (lower is more certain).
        """
        r = self.measurement_noise / max(score, 1e-3)
        residual = np.asarray(measurement, dtype=np.float64) - self.observation @ self.x
        innovation_cov = self.observation @ self.P @ self.observation.T + np.eye(2) * r
        gain = self.P @ self.observation.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ residual
        self.P = (np.eye(4) - gain @ self.observation) @ self.P
        uncertainty = float(np.trace(self.P[:2, :2]))
        return (float(self.x[0]), float(self.x[1]), uncertainty)


class KalmanTracker(TrackerPlugin):
    """NCC measurement fused with a constant-velocity Kalman filter.

    See the module docstring — this is an *estimator*, not an appearance
    matcher; the NCC patch match supplies the "where is it" signal a Kalman
    filter otherwise has no way to obtain.

    Args:
        template_radius: Half-size of the square template patch, in pixels.
        search_radius: How far, in pixels, beyond the template size to search
            around the filter's *predicted* position each frame.
        process_noise: Forwarded to :class:`_ConstantVelocityKalman`.
        measurement_noise: Forwarded to :class:`_ConstantVelocityKalman`.
        confidence_range: ``(low, high)`` calibration applied to the combined
            NCC-score x filter-certainty signal.
    """

    name = "kalman"
    display_name = "Kalman Filter (NCC measurement)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        template_radius: int = 12,
        search_radius: int = 16,
        process_noise: float = 0.5,
        measurement_noise: float = 2.0,
        confidence_range: tuple[float, float] = (0.1, 0.7),
    ) -> None:
        """Create a Kalman tracker; see the class docstring for the arguments."""
        super().__init__()
        self.template_radius = int(template_radius)
        self.search_radius = int(search_radius)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
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
        """Track every query point with a Kalman filter over NCC measurements.

        One run, no merging; each query must lie inside ``frame_range`` and far
        enough from the frame edge to build a template patch.

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
        raw_confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        touched = np.zeros((len(queries), n_frames), dtype=bool)

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
            template = self._extract_patch(anchor_gray, query.position)
            if template is None:
                raise TrackerError(
                    f"query at frame {query.frame} position {query.position} is too close to "
                    f"the frame edge for template_radius={self.template_radius}"
                )
            points[qi, query.frame] = query.position
            raw_confidence[qi, query.frame] = 1.0
            touched[qi, query.frame] = True

            if direction in ("fwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query.position,
                    template,
                    reporter.steps(range(query.frame + 1, end)),
                    points,
                    raw_confidence,
                    touched,
                )
            if direction in ("bwd", "bidir"):
                self._run_direction(
                    cv2,
                    gray,
                    qi,
                    query.position,
                    template,
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
        template: np.ndarray,
        indices: Iterable[int],
        points: np.ndarray,
        raw_confidence: np.ndarray,
        touched: np.ndarray,
    ) -> None:
        """Step predict/measure/update one frame at a time from ``anchor_position``.

        Stops early (leaving the remaining frames untouched) once the search
        window can no longer be built near the frame edge; a low NCC score
        drops confidence without stopping (the prediction alone carries the
        filter through a short occlusion), matching ``template``'s policy.
        """
        kalman = _ConstantVelocityKalman(
            anchor_position,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        radius = self.template_radius
        margin = self.search_radius + radius
        template_h, template_w = template.shape[:2]
        # The covariance trace settles at a steady-state floor that depends on
        # process_noise/measurement_noise, not on anything meaningful about this
        # particular clip, so "certain" is calibrated *relative to the best this
        # run has achieved* rather than against an absolute pixel^2 threshold —
        # scale-invariant to those two constructor constants, and it still
        # collapses correctly under occlusion (a run of poor measurements
        # inflates the covariance well above its floor).
        best_uncertainty: float | None = None

        for index in indices:
            frame_gray = gray(index)
            height, width = frame_gray.shape[:2]
            predicted = kalman.predict()

            cx, cy = int(round(predicted[0])), int(round(predicted[1]))
            x0, y0 = max(cx - margin, 0), max(cy - margin, 0)
            x1, y1 = min(cx + margin + 1, width), min(cy + margin + 1, height)
            window = frame_gray[y0:y1, x0:x1]
            if window.shape[0] < template_h or window.shape[1] < template_w:
                break

            result = cv2.matchTemplate(window.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            score = 0.0 if math.isnan(max_val) else max(0.0, min(1.0, float(max_val)))
            measurement = (float(x0 + max_loc[0] + radius), float(y0 + max_loc[1] + radius))

            filtered_x, filtered_y, uncertainty = kalman.update(measurement, max(score, 1e-3))
            best_uncertainty = (
                uncertainty if best_uncertainty is None else min(best_uncertainty, uncertainty)
            )
            # Certainty in (0, 1]: 1.0 at (or below) the best this run has seen,
            # falling as the post-update covariance inflates above that floor —
            # so a filter that is losing confidence in its own state pulls the
            # reported confidence down even if this one frame's NCC score looked
            # fine on its own.
            certainty = min(1.0, best_uncertainty / uncertainty) if uncertainty > 0 else 1.0

            points[qi, index] = (filtered_x, filtered_y)
            raw_confidence[qi, index] = float(combine_scores(np.array(score), np.array(certainty)))
            touched[qi, index] = True
