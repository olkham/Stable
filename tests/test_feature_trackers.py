"""SIFT/ORB feature matching (todo 30), on footage that actually has features.

Why these two are not in ``test_tracker_suite.py``'s parametrisation
-------------------------------------------------------------------
``orb`` is (see that file). ``sift`` is not, and the reason is a property of the
algorithm rather than a gap in the plugin: the shared suite's clip is a smooth
gradient with one bright square, and SIFT's difference-of-Gaussians finds **one**
keypoint in the whole frame even with its thresholds wound down. A tracker that
needs texture cannot be tested on a clip that has none.

So the same contract is asserted here against ``WarpedClipSource`` — a noise
texture warped by a *known* similarity per frame, which gives ground truth for
every pixel, not just for a marker. ``test_sift_says_why_rather_than_guessing``
covers the low-texture case directly: the plugin must refuse with an explanation
instead of returning a track nobody should trust.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import WarpedClipSource, handheld_transforms

from stable.core.errors import TrackerError
from stable.core.types import Query
from stable.core.video import SyntheticVideoSource
from stable.trackers.features import FeatureTracker, OrbTracker, SiftTracker

DETECTORS = ["sift", "orb"]


@pytest.fixture
def textured() -> WarpedClipSource:
    """40 frames of texture moved by a known handheld path."""
    return WarpedClipSource(handheld_transforms(40, pivot=(160.0, 120.0)))


def strongest_feature(clip: WarpedClipSource, detector: str) -> tuple[float, float]:
    """A position on frame 0 this detector can actually see.

    Anchoring on a real keypoint is what an operator does when they place a
    point on a corner; the ground truth is the clip's imposed warp, which is
    valid for any pixel, so nothing is being made easy here except the choice of
    a trackable subject.
    """
    frame = cv2.cvtColor(clip.get_frame(0), cv2.COLOR_RGB2GRAY)
    factory = (
        cv2.SIFT_create(nfeatures=400) if detector == "sift" else cv2.ORB_create(nfeatures=400)
    )
    keypoints, _ = factory.detectAndCompute(frame, None)
    inside = [
        kp.pt
        for kp in keypoints
        if 80 < kp.pt[0] < 240 and 60 < kp.pt[1] < 180  # away from the frame edge
    ]
    assert inside, f"{detector} found no keypoints to anchor on"
    return inside[0]


def truth_at(clip: WarpedClipSource, reference: np.ndarray, frame: int) -> np.ndarray:
    """Where a reference-frame position sits on ``frame``, from the imposed warp."""
    return np.asarray(clip.transform_at(frame).apply(reference))


@pytest.mark.parametrize("detector", DETECTORS)
def test_it_follows_a_known_camera_move_to_sub_pixel_accuracy(textured, detector):
    plugin = FeatureTracker(detector=detector)
    start = strongest_feature(textured, detector)
    reference = np.asarray(textured.transform_at(0).inverse().apply(np.asarray(start)))

    result = plugin.track(
        textured, [Query(frame=0, x=start[0], y=start[1])], (0, textured.frame_count), "fwd"
    )
    path = result.points[0]
    covered = np.isfinite(path[:, 0])
    assert covered.sum() >= textured.frame_count - 2

    errors = [
        float(np.hypot(*(path[f] - truth_at(textured, reference, f))))
        for f in range(textured.frame_count)
        if covered[f]
    ]
    assert max(errors) < 2.0, f"worst error {max(errors):.2f}px"


@pytest.mark.parametrize("detector", DETECTORS)
def test_the_anchor_frame_holds_the_users_position_exactly(textured, detector):
    """The Phase 0 contract every plugin owes: a query frame is not estimated."""
    plugin = FeatureTracker(detector=detector)
    start = strongest_feature(textured, detector)
    result = plugin.track(textured, [Query(frame=5, x=start[0], y=start[1])], (0, 20), "bidir")
    assert result.points[0, 5] == pytest.approx(start, abs=1e-5)
    assert result.confidence[0, 5] == pytest.approx(1.0)


class Occluded:
    """``textured``, with a stretch of frames covered by a flat card.

    A featureless region is the cleanest possible occlusion for a feature
    tracker: there is nothing to match, so the honest answer is "I don't know",
    which is exactly the behaviour under test.
    """

    def __init__(self, clip: WarpedClipSource, span: tuple[int, int]) -> None:
        self._clip = clip
        self._span = span

    def __getattr__(self, name: str):  # pragma: no cover - plain delegation
        return getattr(self._clip, name)

    @property
    def frame_count(self) -> int:
        return self._clip.frame_count

    def get_frame(self, index: int) -> np.ndarray:
        frame = self._clip.get_frame(index).copy()
        low, high = self._span
        if low <= index < high:
            frame[:] = 128
        return frame


@pytest.mark.parametrize("detector", DETECTORS)
def test_an_occlusion_is_a_gap_and_the_track_recovers_after_it(textured, detector):
    """The reason this tracker exists: failure does not compound.

    Frames 12-18 have nothing to match, so they must come back as gaps. What
    matters is frame 18 onwards: because every frame is matched against the
    *anchor* rather than against its predecessor, the track picks up exactly
    where the subject is, with none of the error an incremental tracker would
    have accumulated while it was guessing through the occlusion.
    """
    start = strongest_feature(textured, detector)
    reference = np.asarray(textured.transform_at(0).inverse().apply(np.asarray(start)))
    clip = Occluded(textured, (12, 18))

    result = FeatureTracker(detector=detector).track(
        clip, [Query(frame=0, x=start[0], y=start[1])], (0, textured.frame_count), "fwd"
    )
    path = result.points[0]
    covered = np.isfinite(path[:, 0])

    assert not covered[12:18].any(), "an occluded frame must be a gap, not a guess"
    assert covered[18:].sum() >= (textured.frame_count - 18) - 2, "it must come back"
    after = [
        float(np.hypot(*(path[f] - truth_at(textured, reference, f))))
        for f in range(18, textured.frame_count)
        if covered[f]
    ]
    assert max(after) < 2.0, f"drifted to {max(after):.2f}px after the occlusion"


@pytest.mark.parametrize("detector", DETECTORS)
def test_confidence_is_normalised_and_zero_where_nothing_was_found(textured, detector):
    start = strongest_feature(textured, detector)
    clip = Occluded(textured, (12, 18))
    result = FeatureTracker(detector=detector).track(
        clip, [Query(frame=0, x=start[0], y=start[1])], (0, textured.frame_count), "fwd"
    )
    confidence = result.confidence[0]
    assert ((confidence >= 0.0) & (confidence <= 1.0)).all()
    missing = ~np.isfinite(result.points[0, :, 0])
    assert missing.any(), "the occlusion should have produced gaps to check"
    assert (confidence[missing] == 0.0).all()
    assert (~result.visible[0][missing]).all()


def test_sift_says_why_rather_than_guessing():
    """On texture-free material the honest answer is a refusal with a reason."""
    plain = SyntheticVideoSource(frame_count=8, width=64, height=48)
    x, y = plain.true_position(0)
    with pytest.raises(TrackerError, match="too little texture"):
        SiftTracker().track(plain, [Query(frame=0, x=x, y=y)], (0, 8), "fwd")


def test_orb_can_work_where_sift_cannot():
    """Not interchangeable: ORB's corner detector fires where SIFT's DoG does not."""
    plain = SyntheticVideoSource(frame_count=24, width=64, height=48)
    x, y = plain.true_position(2)
    result = OrbTracker(sensitivity=0.2, patch_radius=16, search_radius=20).track(
        plain, [Query(frame=2, x=x, y=y)], (0, 24), "bidir"
    )
    path = result.points[0]
    covered = np.isfinite(path[:, 0])
    assert covered.sum() >= 20
    worst = max(
        float(np.hypot(*(path[f] - np.asarray(plain.true_position(f)))))
        for f in range(24)
        if covered[f]
    )
    assert worst < 2.0


def test_the_detector_choice_is_validated():
    with pytest.raises(TrackerError, match="unknown detector"):
        FeatureTracker(detector="surf")


def test_sensitivity_is_bounded():
    with pytest.raises(TrackerError, match="sensitivity"):
        FeatureTracker(sensitivity=0.0)
    with pytest.raises(TrackerError, match="sensitivity"):
        FeatureTracker(sensitivity=2.0)


def test_the_two_registrations_differ_only_in_their_detector():
    assert SiftTracker().detector == "sift"
    assert OrbTracker().detector == "orb"
    assert SiftTracker.name == "sift" and OrbTracker.name == "orb"
    # Both are the same code path, so a fix to one is a fix to both.
    assert isinstance(OrbTracker(), FeatureTracker)


def test_the_window_settings_are_the_ones_the_ui_can_set():
    """`patch_radius`/`search_radius` are what the size sliders edit (todo 30)."""
    from stable.server.tracker_options import tracker_settings

    rows = {row["attribute"]: row for row in tracker_settings("sift")}
    assert rows["patch_radius"]["role"] == "window"
    assert rows["search_radius"]["role"] == "search"
    assert rows["patch_radius"]["value"] == SiftTracker().patch_radius
