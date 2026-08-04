"""Contract and behaviour tests for the pyramidal Lucas-Kanade tracker."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import anchor_at

from stable.core.errors import TrackerError
from stable.core.types import Query, TrackResult
from stable.trackers.lk import LKTracker
from stable.trackers.registry import available_trackers, create_tracker, describe_trackers


def test_metadata_and_lifecycle():
    assert LKTracker.name == "lk"
    assert LKTracker.display_name
    assert LKTracker.requires_gpu is False
    assert LKTracker.license == "MIT"
    assert LKTracker.is_available() is True

    tracker = LKTracker()
    assert not tracker.is_prepared
    tracker.prepare("cpu")
    assert tracker.is_prepared and tracker.device == "cpu"
    tracker.release()
    assert not tracker.is_prepared
    assert "lk" in repr(LKTracker())


def test_discoverable_through_the_registry():
    assert "lk" in available_trackers()
    rows = {row["name"]: row for row in describe_trackers()}
    assert rows["lk"]["license"] == "MIT"
    assert rows["lk"]["available"] is True
    tracker = create_tracker("lk")
    assert isinstance(tracker, LKTracker)
    assert tracker.is_prepared


def test_result_spans_the_whole_clip_timeline(video):
    tracker = LKTracker()
    anchor = anchor_at(video, 4)
    result = tracker.track(video, [anchor], (2, 8))

    assert isinstance(result, TrackResult)
    assert result.n_points == 1
    assert result.n_frames == video.frame_count
    assert np.isnan(result.points[0, :2]).all()
    assert np.isnan(result.points[0, 8:]).all()
    assert np.isfinite(result.points[0, 2:8]).all()


def test_points_come_back_in_query_order(video):
    queries = [anchor_at(video, 0), Query(0, 1.0, 1.0), Query(0, 5.0, 5.0)]
    result = LKTracker().track(video, queries, (0, video.frame_count))
    assert result.n_points == 3
    for index, query in enumerate(queries):
        assert result.points[index, 0] == pytest.approx(query.position, abs=1.5)


def test_tracks_the_synthetic_marker_accurately(video):
    """Pure algorithmic accuracy, no real video encode/decode involved."""
    anchor_frame = 6
    query = anchor_at(video, anchor_frame)
    result = LKTracker().track(video, [query], (0, video.frame_count), "bidir")

    for frame in range(video.frame_count):
        px, py = result.points[0, frame]
        assert np.isfinite(px)
        tx, ty = video.true_position(frame)
        assert (px - tx) ** 2 + (py - ty) ** 2 < 1.5**2
        assert result.confidence[0, frame] > 0.9
        assert result.visible[0, frame]


def test_directions_cover_the_expected_frames(video):
    tracker = LKTracker()
    query = anchor_at(video, 10)

    fwd = tracker.track(video, [query], (0, video.frame_count), "fwd")
    assert np.isnan(fwd.points[0, :10]).all()
    assert np.isfinite(fwd.points[0, 10:]).all()

    bwd = tracker.track(video, [query], (0, video.frame_count), "bwd")
    assert np.isfinite(bwd.points[0, : 10 + 1]).all()
    assert np.isnan(bwd.points[0, 11:]).all()

    bidir = tracker.track(video, [query], (0, video.frame_count), "bidir")
    assert np.isfinite(bidir.points[0]).all()


def test_forward_backward_error_drops_confidence_on_occlusion(occluded_video):
    """The headline Phase 1 acceptance criterion for the LK plugin."""
    anchor_frame = 5
    query = Query(anchor_frame, *occluded_video.true_position(anchor_frame))
    result = LKTracker().track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")

    lo, hi = occluded_video.occlude_range
    clean_confidence = result.confidence[0, anchor_frame:lo]
    assert clean_confidence.min() > 0.9  # rock solid while the marker is visible

    # The very first occluded frame is where the FB check must catch the problem:
    # forward flow still "succeeds" (settles on something), but back-tracking it
    # does not return to the anchor, so FB error - and hence confidence - spikes down.
    assert result.confidence[0, lo] < 0.5


def test_query_must_lie_inside_frame_range(video):
    tracker = LKTracker()
    with pytest.raises(TrackerError, match="frame_range"):
        tracker.track(video, [Query(2, 1.0, 1.0)], (5, 10))


def test_invalid_arguments_raise_tracker_errors(video):
    tracker = LKTracker()
    query = Query(0, 1.0, 1.0)
    with pytest.raises(TrackerError, match="non-empty"):
        tracker.track(video, [query], (5, 5))
    with pytest.raises(TrackerError, match="outside the clip"):
        tracker.track(video, [query], (0, 999))
    with pytest.raises(TrackerError, match="unknown direction"):
        tracker.track(video, [query], (0, 5), "sideways")


def test_empty_query_list_yields_an_empty_result(video):
    result = LKTracker().track(video, [], (0, 5))
    assert result.n_points == 0
    assert result.n_frames == video.frame_count


def test_progress_is_reported_and_ends_at_one(video):
    seen: list[float] = []
    LKTracker().track(
        video,
        [anchor_at(video, 0), anchor_at(video, 1)],
        (0, video.frame_count),
        progress=lambda fraction, message="": seen.append(fraction),
    )
    assert seen
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_output_is_deterministic(video):
    query = anchor_at(video, 3)
    a = LKTracker().track(video, [query], (0, video.frame_count))
    b = LKTracker().track(video, [query], (0, video.frame_count))
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.confidence, b.confidence)
