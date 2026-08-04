"""Contract tests for the dummy tracker — and thus for the plugin contract."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import TrackerError
from stable.core.types import Query, TrackResult
from stable.core.video import SyntheticVideoSource, VideoSource
from stable.trackers.dummy import DummyTracker


def test_synthetic_source_satisfies_the_protocol(video):
    assert isinstance(video, VideoSource)
    frame = video.get_frame(0)
    assert frame.shape == (video.height, video.width, 3)
    assert frame.dtype == np.uint8
    assert [i for i, _ in video.iter_frames(2, 5)] == [2, 3, 4]
    with pytest.raises(IndexError):
        video.get_frame(video.frame_count)


def test_result_spans_the_whole_clip_timeline(video):
    """T is always the clip length; frames outside the range are NaN."""
    tracker = DummyTracker()
    result = tracker.track(video, [Query(4, 10.0, 10.0)], (2, 8))

    assert isinstance(result, TrackResult)
    assert result.n_points == 1
    assert result.n_frames == video.frame_count
    assert np.isnan(result.points[0, :2]).all()
    assert np.isfinite(result.points[0, 2:8]).all()
    assert np.isnan(result.points[0, 8:]).all()
    assert not result.confidence[0, 8:].any()
    assert not result.visible[0, 8:].any()


def test_points_come_back_in_query_order(video):
    queries = [Query(0, 1.0, 1.0), Query(0, 50.0, 20.0), Query(0, 5.0, 40.0)]
    result = DummyTracker().track(video, queries, (0, video.frame_count))
    assert result.n_points == 3
    for index, query in enumerate(queries):
        assert result.points[index, 0] == pytest.approx(query.position, abs=1e-4)


def test_it_follows_ground_truth_through_the_anchor(video):
    anchor_frame = 6
    truth = video.true_position(anchor_frame)
    result = DummyTracker().track(video, [Query(anchor_frame, *truth)], (0, video.frame_count))
    for frame in range(video.frame_count):
        assert result.points[0, frame] == pytest.approx(video.true_position(frame), abs=1e-3)


def test_directions_cover_the_expected_frames(video):
    tracker = DummyTracker()
    query = Query(10, 5.0, 5.0)

    fwd = tracker.track(video, [query], (0, 20), "fwd")
    assert np.isnan(fwd.points[0, :10]).all()
    assert np.isfinite(fwd.points[0, 10:20]).all()

    bwd = tracker.track(video, [query], (0, 20), "bwd")
    assert np.isfinite(bwd.points[0, :11]).all()
    assert np.isnan(bwd.points[0, 11:]).all()

    bidir = tracker.track(video, [query], (0, 20), "bidir")
    assert np.isfinite(bidir.points[0, 0:20]).all()


def test_confidence_decays_away_from_the_anchor(plain_video):
    tracker = DummyTracker(confidence_falloff=0.05, min_confidence=0.1)
    result = tracker.track(plain_video, [Query(0, 4.0, 4.0)], (0, 20), "fwd")
    confidence = result.confidence[0, :20]
    assert confidence[0] == pytest.approx(1.0)
    assert np.all(np.diff(confidence) <= 0)
    assert confidence.min() >= 0.1
    assert 0.0 <= confidence.min() <= confidence.max() <= 1.0


def test_drift_mode_without_ground_truth(plain_video):
    tracker = DummyTracker(drift=(2.0, -1.0))
    result = tracker.track(plain_video, [Query(5, 10.0, 10.0)], (0, 10), "bidir")
    assert result.points[0, 5] == pytest.approx((10.0, 10.0))
    assert result.points[0, 7] == pytest.approx((14.0, 8.0))
    assert result.points[0, 3] == pytest.approx((6.0, 12.0))


def test_output_is_deterministic(video):
    a = DummyTracker().track(video, [Query(3, 7.0, 8.0)], (0, 24))
    b = DummyTracker().track(video, [Query(3, 7.0, 8.0)], (0, 24))
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.confidence, b.confidence)


def test_progress_is_reported_and_ends_at_one(video):
    seen: list[float] = []
    DummyTracker().track(
        video,
        [Query(0, 1.0, 1.0), Query(0, 2.0, 2.0)],
        (0, 24),
        progress=lambda fraction, message="": seen.append(fraction),
    )
    assert seen
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen == sorted(seen)


def test_metadata_and_lifecycle():
    assert DummyTracker.name == "dummy"
    assert DummyTracker.display_name
    assert DummyTracker.requires_gpu is False
    assert DummyTracker.license == "MIT"
    assert DummyTracker.is_available() is True

    tracker = DummyTracker()
    assert not tracker.is_prepared
    tracker.prepare("cpu")
    assert tracker.is_prepared and tracker.device == "cpu"
    assert "dummy" in repr(tracker)


def test_empty_query_list_yields_an_empty_result(video):
    result = DummyTracker().track(video, [], (0, 5))
    assert result.n_points == 0
    assert result.n_frames == video.frame_count


def test_invalid_arguments_raise_tracker_errors(video):
    tracker = DummyTracker()
    query = Query(0, 0.0, 0.0)
    with pytest.raises(TrackerError, match="non-empty"):
        tracker.track(video, [query], (5, 5))
    with pytest.raises(TrackerError, match="outside the clip"):
        tracker.track(video, [query], (0, 999))
    with pytest.raises(TrackerError, match="unknown direction"):
        tracker.track(video, [query], (0, 5), "sideways")
    with pytest.raises(TrackerError, match="query frame"):
        tracker.track(video, [Query(999, 0.0, 0.0)], (0, 5))


def test_single_frame_clip_edge_case():
    tiny = SyntheticVideoSource(frame_count=1, width=8, height=8)
    result = DummyTracker().track(tiny, [Query(0, 4.0, 4.0)], (0, 1))
    assert result.n_frames == 1
    assert np.isfinite(result.points[0, 0]).all()
