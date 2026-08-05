"""Local feature matching (SIFT / ORB) — a tracker that can survive a gap.

Every other classic-CV plugin here follows a point *incrementally*: LK integrates
optical flow frame to frame, template and correlation match the previous
neighbourhood into the next one. That is accurate and cheap, and it has one
structural weakness — there is no way back. Once a step is wrong (an occlusion, a
cut, a fast pan that leaves the search window behind), every later frame inherits
the error, because the only thing anchoring the track is the frame before it.

Feature matching answers a different question. Each frame is described
independently — keypoints with descriptors that are invariant to rotation and
scale — and the tracked position is recovered by matching the *anchor frame's*
neighbourhood against the current frame's keypoints and fitting the local
transform that best explains the matches. Nothing is integrated, so nothing
compounds: a frame that fails to match leaves a gap, and the next frame is
matched against the anchor exactly as if the failure had not happened.

Detectors
---------
``sift``
    Lowe's Scale-Invariant Feature Transform. In mainline OpenCV since 4.4 (its
    patent expired in March 2020), so it needs no contrib build. Slower than ORB
    and distinctly better on scale change and low-contrast texture.
``orb``
    Oriented FAST and Rotated BRIEF — free, binary descriptors, several times
    faster than SIFT, weaker under large scale change.

**Not SURF.** It lives in ``opencv-contrib-python-nonfree``, which is not
installable from PyPI for most users and is patent-encumbered in several
jurisdictions. Shipping a plugin that almost nobody can import would be worse
than not shipping it; the two above cover the same ground.

Confidence
----------
The raw signal is the number of inlier matches supporting the fitted local
transform, normalised onto the shared [0, 1] scale by
:mod:`stable.trackers.confidence`. That is honest in the way this tracker's
failure mode needs: an occluded point does not drift, it stops having inliers.
"""

from __future__ import annotations

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

__all__ = ["FeatureTracker", "SiftTracker", "OrbTracker"]


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
            "FeatureTracker requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


class FeatureTracker(TrackerPlugin):
    """Track a point by matching descriptors around it, frame by frame.

    Args:
        detector: ``"sift"`` or ``"orb"``.
        patch_radius: Half-size of the neighbourhood around the point whose
            keypoints describe it. Big enough to contain several features, small
            enough that they all move together — this is the setting the size
            slider edits.
        search_radius: How far from the last known position to look for the
            match. Keypoints outside it are ignored, which is what stops a
            repeated texture elsewhere in the frame from capturing the track.
        n_features: Keypoints the detector may return per frame.
        ratio: Lowe's ratio test threshold; a match is kept only if the best
            descriptor distance is this much better than the second best.
        min_matches: Fewest kept matches that can still place the point. Two
            is the useful floor: one match is a single keypoint's opinion,
            with nothing to check it against.
        sensitivity: How readily the detector calls something a feature, in
            ``(0, 1]``. ``1.0`` is OpenCV's own default; lower it on
            low-contrast material where the defaults find nothing, at the cost
            of describing noise. It maps onto ``contrastThreshold`` for SIFT and
            ``fastThreshold`` for ORB — one control rather than two names for
            the same idea.
        confidence_range: ``(low, high)`` inlier counts mapped onto ``[0, 1]``.
    """

    name = "features"
    display_name = "Feature Matching (SIFT)"
    requires_gpu = False
    license = "MIT"

    #: Which OpenCV detector this plugin builds. Subclasses override.
    detector_name = "sift"

    def __init__(
        self,
        *,
        detector: str | None = None,
        patch_radius: int = 32,
        search_radius: int = 64,
        n_features: int = 400,
        ratio: float = 0.8,
        min_matches: int = 2,
        sensitivity: float = 1.0,
        confidence_range: tuple[float, float] = (3.0, 25.0),
    ) -> None:
        """Create a feature tracker; see the class docstring for the arguments."""
        super().__init__()
        self.detector = str(detector or self.detector_name).lower()
        if self.detector not in ("sift", "orb"):
            raise TrackerError(f"unknown detector {self.detector!r}; expected 'sift' or 'orb'")
        self.patch_radius = int(patch_radius)
        self.search_radius = int(search_radius)
        self.n_features = int(n_features)
        self.ratio = float(ratio)
        self.min_matches = int(min_matches)
        self.sensitivity = float(sensitivity)
        if not 0.0 < self.sensitivity <= 1.0:
            raise TrackerError(f"sensitivity must lie in (0, 1], got {self.sensitivity}")
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))

    @classmethod
    def is_available(cls) -> bool:
        """Whether OpenCV is importable and offers this detector."""
        try:
            import cv2
        except ImportError:
            return False
        return hasattr(cv2, "SIFT_create") or hasattr(cv2, "ORB_create")

    @classmethod
    def unavailable_reason(cls) -> str | None:
        """Why the plugin cannot run, and what fixes it."""
        if cls.is_available():
            return None
        return (
            "needs opencv-python with SIFT/ORB: pip install 'stable-tracker[video]' "
            "(SIFT is in mainline OpenCV from 4.4)"
        )

    # -- detector ---------------------------------------------------------

    def _build_detector(self, cv2: Any) -> Any:
        """Create the OpenCV detector this plugin is configured for.

        Raises:
            TrackerError: If this OpenCV build does not offer it.
        """
        if self.detector == "sift":
            factory = getattr(cv2, "SIFT_create", None)
            if factory is None:  # pragma: no cover - only on an ancient OpenCV
                raise TrackerError(
                    "this OpenCV build has no SIFT; it is in mainline OpenCV from 4.4, "
                    "or use the 'orb' detector"
                )
            return factory(
                nfeatures=self.n_features,
                # OpenCV's defaults are 0.04 / 10; scaling the contrast
                # threshold down and the edge threshold up is what "more
                # sensitive" means for a DoG detector.
                contrastThreshold=0.04 * self.sensitivity,
                edgeThreshold=10.0 / self.sensitivity,
            )
        factory = getattr(cv2, "ORB_create", None)
        if factory is None:  # pragma: no cover - ORB is always present in practice
            raise TrackerError("this OpenCV build has no ORB")
        # FAST's threshold is an intensity difference (default 20), and its
        # patch/edge sizes have to shrink with it or a small feature is refused
        # for being too close to the border of its own descriptor patch.
        fast = max(1, int(round(20 * self.sensitivity)))
        patch = 31 if self.sensitivity > 0.5 else 7
        return factory(
            nfeatures=self.n_features,
            fastThreshold=fast,
            patchSize=patch,
            edgeThreshold=patch,
        )

    def _matcher(self, cv2: Any) -> Any:
        """Return a brute-force matcher with the right metric for the descriptors.

        SIFT descriptors are float vectors compared with L2; ORB's are binary
        strings compared with Hamming distance. Using the wrong one does not
        error, it just matches badly — which is the kind of bug that looks like
        "the tracker is not very good".
        """
        norm = cv2.NORM_L2 if self.detector == "sift" else cv2.NORM_HAMMING
        return cv2.BFMatcher(norm)

    # -- the run ----------------------------------------------------------

    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Track every query by matching its anchor neighbourhood into each frame.

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
                query frame lies outside it, or the anchor neighbourhood has too
                little texture to describe.
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

        detector = self._build_detector(cv2)
        matcher = self._matcher(cv2)
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
            anchor = self._near(
                self._detect(detector, gray(query.frame)), query.position, self.patch_radius
            )
            if anchor is None:
                raise TrackerError(
                    f"no features to track at frame {query.frame} position {query.position}; "
                    f"this patch has too little texture for {self.detector.upper()} — "
                    "try a larger pattern size, or a different tracker"
                )
            points[qi, query.frame] = query.position
            raw_confidence[qi, query.frame] = float(self.confidence_range[1])
            touched[qi, query.frame] = True

            spans = []
            if direction in ("fwd", "bidir"):
                spans.append(reporter.steps(range(query.frame + 1, end)))
            if direction in ("bwd", "bidir"):
                spans.append(reporter.steps(range(query.frame - 1, start - 1, -1)))
            for span in spans:
                self._run_direction(
                    cv2,
                    detector,
                    matcher,
                    gray,
                    qi,
                    query,
                    anchor,
                    span,
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

    def _detect(self, detector: Any, frame_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect and describe every keypoint in a **whole** frame.

        Deliberately not on a crop. Detecting inside a cut-out window changes
        the answer: SIFT builds a scale space, and a window's own edges create
        responses that the same pixels in context do not have, while its size
        caps the scales that can be examined at all. Describing the frame once
        and *then* selecting by position is both the correct scale space and
        cheaper when several points share a frame.

        Returns:
            ``(positions[K, 2], descriptors[K, D])``; both empty when the frame
            has no describable features.
        """
        keypoints, descriptors = detector.detectAndCompute(frame_gray, None)
        if descriptors is None or not keypoints:
            return np.zeros((0, 2)), np.zeros((0, 1), dtype=np.float32)
        positions = np.asarray([kp.pt for kp in keypoints], dtype=np.float64)
        return positions, np.asarray(descriptors)

    def _near(
        self,
        features: tuple[np.ndarray, np.ndarray],
        centre: tuple[float, float],
        radius: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the subset of ``features`` within ``radius`` of ``centre``.

        This is what keeps the tracker local: a repeated texture on the other
        side of the frame is an excellent descriptor match and completely the
        wrong answer, so it never gets to compete.

        Returns:
            ``(positions, descriptors)``, or ``None`` when nothing is near
            enough to describe the point.
        """
        positions, descriptors = features
        if len(positions) == 0:
            return None
        within = np.hypot(positions[:, 0] - centre[0], positions[:, 1] - centre[1]) <= radius
        if not within.any():
            return None
        return positions[within], descriptors[within]

    def _run_direction(
        self,
        cv2: Any,
        detector: Any,
        matcher: Any,
        gray: Any,
        qi: int,
        query: Query,
        anchor: tuple[np.ndarray, np.ndarray],
        indices: Iterable[int],
        points: np.ndarray,
        raw_confidence: np.ndarray,
        touched: np.ndarray,
    ) -> None:
        """Match the anchor's descriptors into each frame of ``indices``.

        The anchor is never replaced. That is the point of this tracker: an
        unmatched frame is a **gap**, not a new reference, so a bad frame cannot
        become the thing every later frame is measured against. The search
        window follows the last *successful* position, so the point can be
        picked up again after an occlusion where an incremental tracker would
        have drifted away.
        """
        anchor_positions, anchor_descriptors = anchor
        position = np.asarray(query.position, dtype=np.float64)

        for index in indices:
            found = self._near(
                self._detect(detector, gray(index)), tuple(position), self.search_radius
            )
            if found is None:
                continue
            frame_positions, frame_descriptors = found
            if len(frame_descriptors) < 2 or len(anchor_descriptors) < 2:
                continue

            pairs = matcher.knnMatch(anchor_descriptors, frame_descriptors, k=2)
            offsets = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                best, second = pair
                # Lowe's ratio test: a descriptor that matches two frame
                # keypoints almost equally well has not identified either.
                if best.distance < self.ratio * second.distance:
                    offsets.append(frame_positions[best.trainIdx] - anchor_positions[best.queryIdx])
            if len(offsets) < self.min_matches:
                continue

            # The local motion is the median offset of the matched features:
            # robust to a few wrong matches without needing a full RANSAC, and
            # the count of agreeing matches is what confidence is built from.
            shifts = np.asarray(offsets, dtype=np.float64)
            median = np.median(shifts, axis=0)
            agreeing = shifts[np.hypot(*(shifts - median).T) <= max(2.0, self.patch_radius / 4.0)]
            if len(agreeing) < self.min_matches:
                continue
            offset = agreeing.mean(axis=0)

            position = np.asarray(query.position, dtype=np.float64) + offset
            points[qi, index] = position
            raw_confidence[qi, index] = float(len(agreeing))
            touched[qi, index] = True


class SiftTracker(FeatureTracker):
    """SIFT feature matching — the accurate, scale-tolerant option."""

    name = "sift"
    display_name = "Feature Matching (SIFT)"
    detector_name = "sift"


class OrbTracker(FeatureTracker):
    """ORB feature matching — the fast, binary-descriptor option.

    Same algorithm as :class:`SiftTracker` with a different descriptor: several
    times quicker per frame, and less tolerant of large scale change. Worth
    reaching for on long clips where SIFT's cost per frame adds up.
    """

    name = "orb"
    display_name = "Feature Matching (ORB)"
    detector_name = "orb"
