"""Contract and behaviour tests for the NCC template-matching tracker."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import anchor_at

from stable.core.errors import TrackerError
from stable.core.types import Query, TrackResult
from stable.trackers.registry import available_trackers, create_tracker, describe_trackers
from stable.trackers.template import TemplateTracker


def test_metadata_and_lifecycle():
    assert TemplateTracker.name == "template"
    assert TemplateTracker.display_name
    assert TemplateTracker.requires_gpu is False
    assert TemplateTracker.license == "MIT"
    assert TemplateTracker.is_available() is True

    tracker = TemplateTracker()
    assert not tracker.is_prepared
    tracker.prepare("cpu")
    assert tracker.is_prepared and tracker.device == "cpu"
    tracker.release()
    assert not tracker.is_prepared
    assert "template" in repr(TemplateTracker())


def test_discoverable_through_the_registry():
    assert "template" in available_trackers()
    rows = {row["name"]: row for row in describe_trackers()}
    assert rows["template"]["license"] == "MIT"
    assert rows["template"]["available"] is True
    tracker = create_tracker("template")
    assert isinstance(tracker, TemplateTracker)
    assert tracker.is_prepared


def test_result_spans_the_whole_clip_timeline(video):
    tracker = TemplateTracker()
    anchor = anchor_at(video, 4)
    result = tracker.track(video, [anchor], (2, 8))

    assert isinstance(result, TrackResult)
    assert result.n_points == 1
    assert result.n_frames == video.frame_count
    assert np.isnan(result.points[0, :2]).all()
    assert np.isnan(result.points[0, 8:]).all()
    assert np.isfinite(result.points[0, 2:8]).all()


def test_points_come_back_in_query_order(video):
    # Positions kept well clear of the frame edge: the default template_radius
    # (12) needs a 13px margin on every side to extract a patch.
    queries = [anchor_at(video, 0), Query(0, 20.0, 20.0), Query(0, 30.0, 15.0)]
    result = TemplateTracker().track(video, queries, (0, video.frame_count))
    assert result.n_points == 3
    for index, query in enumerate(queries):
        assert result.points[index, 0] == pytest.approx(query.position, abs=1.5)


def test_tracks_the_synthetic_marker_accurately(video):
    anchor_frame = 6
    query = anchor_at(video, anchor_frame)
    result = TemplateTracker().track(video, [query], (0, video.frame_count), "bidir")

    for frame in range(video.frame_count):
        px, py = result.points[0, frame]
        assert np.isfinite(px)
        tx, ty = video.true_position(frame)
        assert (px - tx) ** 2 + (py - ty) ** 2 < 1.5**2
        assert result.confidence[0, frame] > 0.9


def test_directions_cover_the_expected_frames(video):
    tracker = TemplateTracker()
    query = anchor_at(video, 10)

    fwd = tracker.track(video, [query], (0, video.frame_count), "fwd")
    assert np.isnan(fwd.points[0, :10]).all()
    assert np.isfinite(fwd.points[0, 10:]).all()

    bwd = tracker.track(video, [query], (0, video.frame_count), "bwd")
    assert np.isfinite(bwd.points[0, : 10 + 1]).all()
    assert np.isnan(bwd.points[0, 11:]).all()

    bidir = tracker.track(video, [query], (0, video.frame_count), "bidir")
    assert np.isfinite(bidir.points[0]).all()


def test_adaptive_template_update_beats_a_frozen_template(growing_marker_video):
    """The Phase 1 feature: adaptive template update should measurably help
    when the tracked patch's appearance drifts (here: the marker grows).
    """
    query = Query(0, *growing_marker_video.true_position(0))
    frame_range = (0, growing_marker_video.frame_count)

    frozen = TemplateTracker(update_rate=0.0, template_radius=6)
    adaptive = TemplateTracker(update_rate=0.2, template_radius=6)

    frozen_result = frozen.track(growing_marker_video, [query], frame_range, "fwd")
    adaptive_result = adaptive.track(growing_marker_video, [query], frame_range, "fwd")

    late_frame = growing_marker_video.frame_count - 1
    assert adaptive_result.confidence[0, late_frame] > frozen_result.confidence[0, late_frame]
    assert adaptive_result.confidence[0, late_frame] > 0.9
    assert adaptive_result.confidence[0].mean() > frozen_result.confidence[0].mean()


def test_confidence_drops_when_the_marker_is_occluded(occluded_video):
    anchor_frame = 5
    query = Query(anchor_frame, *occluded_video.true_position(anchor_frame))
    frame_range = (0, occluded_video.frame_count)
    result = TemplateTracker().track(occluded_video, [query], frame_range, "fwd")

    lo, hi = occluded_video.occlude_range
    clean_confidence = result.confidence[0, anchor_frame:lo]
    occluded_confidence = result.confidence[0, lo:hi]
    assert clean_confidence.min() > 0.9
    assert occluded_confidence.max() < clean_confidence.min()


def test_query_must_lie_inside_frame_range(video):
    tracker = TemplateTracker()
    with pytest.raises(TrackerError, match="frame_range"):
        tracker.track(video, [Query(2, 1.0, 1.0)], (5, 10))


def test_query_too_close_to_the_edge_raises():
    from stable.core.video import SyntheticVideoSource

    tiny = SyntheticVideoSource(frame_count=5, width=16, height=16)
    tracker = TemplateTracker(template_radius=12)
    with pytest.raises(TrackerError, match="edge"):
        tracker.track(tiny, [Query(0, 1.0, 1.0)], (0, 5))


def test_invalid_arguments_raise_tracker_errors(video):
    tracker = TemplateTracker()
    query = Query(0, 1.0, 1.0)
    with pytest.raises(TrackerError, match="non-empty"):
        tracker.track(video, [query], (5, 5))
    with pytest.raises(TrackerError, match="outside the clip"):
        tracker.track(video, [query], (0, 999))
    with pytest.raises(TrackerError, match="unknown direction"):
        tracker.track(video, [query], (0, 5), "sideways")


def test_empty_query_list_yields_an_empty_result(video):
    result = TemplateTracker().track(video, [], (0, 5))
    assert result.n_points == 0
    assert result.n_frames == video.frame_count


def test_progress_is_reported_and_ends_at_one(video):
    seen: list[float] = []
    TemplateTracker().track(
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
    a = TemplateTracker().track(video, [query], (0, video.frame_count))
    b = TemplateTracker().track(video, [query], (0, video.frame_count))
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.confidence, b.confidence)
