"""Bootstrap particle filter tracker — an *estimator*, not an appearance tracker.

Todo 25 asks for "a particle filter tracker". Like the Kalman filter in
:mod:`stable.trackers.kalman`, a particle filter is a recursive state
estimator with nothing to look at on its own; it needs a measurement source to
weight its hypotheses against. This plugin pairs the classic bootstrap /
CONDENSATION particle filter (Isard & Blake, 1998) with normalized
cross-correlation (NCC) patch matching — the same measurement primitive
:mod:`stable.trackers.template` and :mod:`stable.trackers.kalman` use, just
consumed differently:

#. **Predict.** Every particle (a candidate ``(x, y)`` hypothesis) is moved by
   the run's estimated mean velocity plus independent Gaussian diffusion —
   unlike the Kalman filter's single Gaussian belief, a particle *cloud* can
   represent an arbitrarily shaped, even multi-modal, distribution over where
   the point might be, which is the classic reason to reach for a particle
   filter over a Kalman filter: it needs no linear-Gaussian assumption.
#. **Weight.** Each particle's patch is compared against the reference
   template by NCC (not a search — each particle already *is* a position
   hypothesis, so this evaluates the match at exactly that point). The score
   becomes the particle's importance weight.
#. **Resample.** Particles are resampled proportional to weight (systematic
   resampling, which is lower-variance than the naive multinomial scheme),
   so on the next frame effort concentrates where the previous frame's
   evidence said the point actually was — this is what makes a particle
   filter track through a rougher motion or a briefly ambiguous appearance
   without needing the target to sit inside one, small, Gaussian-shaped
   region of uncertainty the way a Kalman filter must.

The reported position is each frame's weight-mean particle. Confidence
combines *two* independent signals: the mean particle's own NCC score (is
this actually the right patch?) and how *concentrated* the cloud is around
that mean (does the filter itself agree on where the point is, or is belief
smeared across the frame?) — both must hold, so they are multiplied, exactly
as :mod:`stable.trackers.kalman` combines score with its own covariance.
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

__all__ = ["ParticleFilterTracker"]


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
            "ParticleFilterTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return particle indices resampled proportional to ``weights``.

    Systematic resampling draws one random offset and takes evenly spaced
    strides through the cumulative weight distribution, which has much lower
    variance than drawing ``n`` independent multinomial samples — the standard
    choice for a bootstrap particle filter.

    Args:
        weights: ``[n]`` non-negative weights, need not be pre-normalised.
        rng: Random generator (owned by the tracker run, not global state, so
            results are reproducible given a seed).

    Returns:
        ``[n]`` indices into ``weights`` to resample.
    """
    n = weights.shape[0]
    total = weights.sum()
    normalized = np.full(n, 1.0 / n) if total <= 0 else weights / total
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0  # guard against floating-point short-fall
    start = rng.uniform(0.0, 1.0 / n)
    positions = start + np.arange(n) / n
    return np.searchsorted(cumulative, positions)


class ParticleFilterTracker(TrackerPlugin):
    """NCC measurement fused with a bootstrap particle filter.

    See the module docstring — this is an *estimator*, not an appearance
    matcher; NCC patch matching supplies the "how well does this hypothesis
    fit" weight a particle filter otherwise has no way to obtain.

    Args:
        n_particles: Number of particle hypotheses tracked per point. More
            particles cover a larger area per frame at proportional cost;
            :attr:`n_particles` is a straightforward speed/robustness trade.
        template_radius: Half-size of the square template patch, in pixels.
        motion_std: Standard deviation, in pixels, of the per-frame Gaussian
            diffusion added to each particle on top of the estimated mean
            velocity — the filter's tolerance for how wrong a constant-velocity
            guess can be before particles stop covering the true position.
        confidence_range: ``(low, high)`` calibration applied to the combined
            NCC-score x cloud-concentration signal.
        seed: Seed for the run's random generator, so results are
            deterministic (the shared test suite requires this of every
            plugin; a particle filter is the one algorithm here that would
            otherwise be stochastic by nature).
    """

    name = "particle"
    display_name = "Particle Filter (NCC measurement)"
    requires_gpu = False
    license = "MIT"

    def __init__(
        self,
        *,
        n_particles: int = 400,
        template_radius: int = 12,
        motion_std: float = 3.0,
        confidence_range: tuple[float, float] = (0.05, 0.4),
        seed: int = 0,
    ) -> None:
        """Create a particle filter tracker; see the class docstring for the arguments."""
        super().__init__()
        self.n_particles = int(n_particles)
        self.template_radius = int(template_radius)
        self.motion_std = float(motion_std)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))
        self.seed = int(seed)
        if self.n_particles < 4:
            raise TrackerError(f"n_particles must be >= 4, got {self.n_particles}")

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
        """Track every query point with a particle filter over NCC measurements.

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
                    seed=self.seed + 2 * qi,
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
                    seed=self.seed + 2 * qi + 1,
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

    def _ncc_scores(
        self, cv2: Any, frame_gray: np.ndarray, template: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Return each particle's NCC score against ``template``, ``0`` where out of bounds.

        Args:
            cv2: The imported ``cv2`` module.
            frame_gray: Full grayscale frame to sample patches from.
            template: Reference patch, ``float32``.
            positions: ``[n, 2]`` particle ``(x, y)`` positions (need not be
                integer; rounded to the nearest pixel to extract a patch).

        Returns:
            ``[n]`` scores in ``[0, 1]`` (NCC's ``[-1, 1]`` clamped at ``0``,
            the same convention :mod:`stable.trackers.template` uses).
        """
        radius = self.template_radius
        height, width = frame_gray.shape[:2]
        scores = np.zeros(positions.shape[0], dtype=np.float32)
        for i, (px, py) in enumerate(positions):
            cx, cy = int(round(float(px))), int(round(float(py)))
            x0, x1 = cx - radius, cx + radius + 1
            y0, y1 = cy - radius, cy + radius + 1
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue
            patch = frame_gray[y0:y1, x0:x1].astype(np.float32)
            result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            value = float(result[0, 0])
            scores[i] = 0.0 if math.isnan(value) else max(0.0, min(1.0, value))
        return scores

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
        *,
        seed: int,
    ) -> None:
        """Step predict/weight/resample one frame at a time from ``anchor_position``.

        Stops early (leaving the remaining frames untouched) once every
        particle falls outside the frame (nothing left to weight); a low
        cloud-mean score drops confidence without stopping, matching
        ``template``/``kalman``'s policy of letting the estimator carry the
        point through a short occlusion.
        """
        rng = np.random.default_rng(seed)
        particles = np.tile(np.asarray(anchor_position, dtype=np.float64), (self.n_particles, 1))
        velocity = np.zeros(2, dtype=np.float64)
        previous_estimate = np.asarray(anchor_position, dtype=np.float64)
        radius = self.template_radius

        for index in indices:
            frame_gray = gray(index)
            height, width = frame_gray.shape[:2]

            # Predict: mean-velocity drift plus independent Gaussian diffusion
            # per particle, so the cloud covers *around* the constant-velocity
            # guess rather than assuming it is exact the way the Kalman
            # tracker's single Gaussian belief effectively must.
            noise = rng.normal(scale=self.motion_std, size=particles.shape)
            particles = particles + velocity + noise
            particles[:, 0] = np.clip(particles[:, 0], 0.0, width - 1.0)
            particles[:, 1] = np.clip(particles[:, 1], 0.0, height - 1.0)

            scores = self._ncc_scores(cv2, frame_gray, template, particles)
            if scores.sum() <= 0.0:
                break

            mean_position = np.average(particles, axis=0, weights=scores)
            # Concentration: 1 at a point cloud (all particles agree), falling
            # toward 0 as the weighted spread grows relative to the template
            # size — the particle-filter analogue of the Kalman tracker's
            # posterior-covariance certainty term.
            spread = math.sqrt(
                float(np.average(np.sum((particles - mean_position) ** 2, axis=1), weights=scores))
            )
            concentration = 1.0 / (1.0 + spread / max(radius, 1))
            # The match quality signal is evaluated *at the weighted estimate*,
            # not averaged over the whole cloud — most particles sit away from
            # the true position by construction (that is what the spread means),
            # so their individual scores are low even when the cloud's own
            # weighted estimate sits exactly on the target.
            point_scores = self._ncc_scores(cv2, frame_gray, template, mean_position[None, :])
            point_score = float(point_scores[0])

            points[qi, index] = (float(mean_position[0]), float(mean_position[1]))
            raw_confidence[qi, index] = float(
                combine_scores(np.array(point_score), np.array(concentration))
            )
            touched[qi, index] = True

            # Velocity is the change in the *weighted estimate* frame to frame,
            # not the raw particle mean (which is dominated by diffusion noise,
            # not evidence) — this is what lets the cloud drift with a moving
            # target instead of only ever diffusing outward from a fixed spot.
            velocity = mean_position - previous_estimate
            previous_estimate = mean_position
            indices_resampled = _systematic_resample(scores, rng)
            particles = particles[indices_resampled]
