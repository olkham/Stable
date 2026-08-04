"""The Phase 2 multi-anchor blend: falloff, disagreement, and fixing a drift.

``test_merge.py`` covers the API and the unambiguous cases fixed in Phase 0.
This file covers what Phase 2 added — the confidence-weighted blend with
distance-to-anchor falloff and the disagreement penalty — plus the headline
acceptance criterion: *a deliberately drifting track is fixed by adding one
mid-clip anchor; the merged path passes through all anchors and confidence dips
where the runs disagreed.*

The blend is tracker-agnostic by design, so it is exercised here with a
deterministic drifting tracker and then, at the end, with the real ``lk`` plugin
to prove the same code applies uniformly to Phase 1's trackers.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DriftingTracker, anchor_at, execute_run

from stable.core.types import PointTrack, Query
from stable.core.video import SyntheticVideoSource
from stable.trackers.lk import LKTracker
from stable.trackers.merge import (
    MergeConfig,
    TrackRun,
    merge_anchor_runs,
    plan_anchor_runs,
)


def merge_with(tracker, video, anchors, frame_range=None, config=None):
    """Plan, execute and merge every run for one point."""
    frame_range = frame_range or (0, video.frame_count)
    runs = [execute_run(tracker, video, spec) for spec in plan_anchor_runs(anchors, frame_range)]
    return merge_anchor_runs(runs, anchors, video.frame_count, config)


def truth_error(track, video):
    """Return the per-frame distance between a merged track and ground truth."""
    truth = np.array([video.true_position(f) for f in range(video.frame_count)])
    return np.hypot(track.points[:, 0] - truth[:, 0], track.points[:, 1] - truth[:, 1])


@pytest.fixture
def long_video() -> SyntheticVideoSource:
    """A 41-frame synthetic clip — long enough for two anchor spans."""
    return SyntheticVideoSource(frame_count=41, width=128, height=96, radius=20.0, period=80.0)


# -- the acceptance criterion --------------------------------------------


def test_one_mid_clip_anchor_fixes_a_drifting_track(long_video):
    """PLAN.md Phase 2: the drift is repaired by adding a single anchor."""
    tracker = DriftingTracker(drift=(0.6, 0.0))
    start_anchor = anchor_at(long_video, 0)
    mid_anchor = anchor_at(long_video, 20)

    before = merge_with(tracker, long_video, [start_anchor])
    after = merge_with(tracker, long_video, [start_anchor, mid_anchor])

    before_error = truth_error(before.track, long_video)
    after_error = truth_error(after.track, long_video)

    # The single-anchor run drifts further and further from the marker...
    assert before_error.max() > 20.0
    # ...and one extra anchor at least halves the worst error: no frame is now
    # more than 20 frames from an anchor, instead of 40.
    assert after_error.max() <= before_error.max() / 2.0 + 1e-6
    # Inside the newly bracketed span the two runs cancel almost exactly.
    assert after_error[1:20].max() < 1.0
    assert before_error[1:20].max() > 5.0


def test_the_merged_path_passes_through_every_anchor_exactly(long_video):
    tracker = DriftingTracker(drift=(0.9, -0.4))
    anchors = [Query(0, 30.0, 30.0), Query(18, 44.5, 51.25), Query(40, 62.0, 20.0)]

    result = merge_with(tracker, long_video, anchors)

    for anchor in anchors:
        assert result.track.points[anchor.frame] == pytest.approx(anchor.position, abs=1e-4)
        assert result.track.confidence[anchor.frame] == pytest.approx(1.0)
        assert bool(result.track.visible[anchor.frame])
    assert result.report.anchor_frames == (0, 18, 40)
    assert np.isfinite(result.track.points).all()


def test_confidence_dips_where_the_runs_disagreed(long_video):
    """The "go fix this" signal: only the troubled span loses confidence."""
    # Drift accumulates only between frames 20 and 40, so the first span merges
    # cleanly and the second does not.
    tracker = DriftingTracker(drift=(0.7, 0.0), drift_range=(20, 40))
    anchors = [anchor_at(long_video, 0), anchor_at(long_video, 20), anchor_at(long_video, 40)]

    result = merge_with(tracker, long_video, anchors)

    calm = result.track.confidence[1:20]
    troubled = result.track.confidence[21:40]
    assert calm.min() > 0.9, "a span where both runs agree must keep its confidence"
    assert troubled.max() < calm.min(), "the disagreeing span must be visibly worse"

    disagreement = result.report.disagreement
    assert np.nanmax(disagreement[1:20]) < 1e-6
    assert np.nanmax(disagreement[21:40]) > 8.0
    assert all(20 <= frame <= 40 for frame in result.report.worst_frames())


def test_disagreement_penalty_scales_with_the_configured_tolerance(long_video):
    tracker = DriftingTracker(drift=(0.5, 0.0), drift_range=(20, 40))
    anchors = [anchor_at(long_video, 0), anchor_at(long_video, 20), anchor_at(long_video, 40)]

    strict = merge_with(tracker, long_video, anchors, config=MergeConfig(disagreement_px=2.0))
    lenient = merge_with(tracker, long_video, anchors, config=MergeConfig(disagreement_px=40.0))

    assert strict.track.confidence[30] == 0.0
    assert 0.0 < lenient.track.confidence[30] < 1.0


# -- the blend itself -----------------------------------------------------


def two_runs(left_position, right_position, n_frames=11, confidence=(1.0, 1.0)):
    """Build two full-length runs anchored at either end of a short clip."""
    left = TrackRun(
        Query(0, *left_position[0]),
        "fwd",
        PointTrack(
            points=np.asarray(left_position, dtype=np.float32),
            confidence=np.full(n_frames, confidence[0], dtype=np.float32),
            visible=np.ones(n_frames, dtype=bool),
        ),
    )
    right = TrackRun(
        Query(n_frames - 1, *right_position[-1]),
        "bwd",
        PointTrack(
            points=np.asarray(right_position, dtype=np.float32),
            confidence=np.full(n_frames, confidence[1], dtype=np.float32),
            visible=np.ones(n_frames, dtype=bool),
        ),
    )
    return [left, right]


def test_the_blend_crossfades_between_the_two_anchors():
    left = np.stack([np.zeros(11), np.zeros(11)], axis=1)
    right = np.stack([np.full(11, 10.0), np.zeros(11)], axis=1)
    runs = two_runs(left, right)
    anchors = [Query(0, 0.0, 0.0), Query(10, 10.0, 0.0)]

    merged = merge_anchor_runs(runs, anchors, 11).track

    xs = merged.points[:, 0]
    assert xs[0] == pytest.approx(0.0) and xs[10] == pytest.approx(10.0)
    assert xs[5] == pytest.approx(5.0, abs=1e-6), "equal distance means equal say"
    assert np.all(np.diff(xs) > 0), "the handover is monotone, never a jump back"
    # Near an anchor, that anchor's own run dominates.
    assert xs[1] < 2.0 and xs[9] > 8.0


def test_a_confident_run_outweighs_an_unsure_one_at_the_same_distance():
    left = np.stack([np.zeros(11), np.zeros(11)], axis=1)
    right = np.stack([np.full(11, 10.0), np.zeros(11)], axis=1)
    anchors = [Query(0, 0.0, 0.0), Query(10, 10.0, 0.0)]

    balanced = merge_anchor_runs(two_runs(left, right), anchors, 11).track
    tilted = merge_anchor_runs(two_runs(left, right, confidence=(0.9, 0.1)), anchors, 11).track

    assert balanced.points[5, 0] == pytest.approx(5.0, abs=1e-6)
    assert tilted.points[5, 0] < 2.0  # the confident run wins the middle


def test_falloff_frames_can_be_pinned_explicitly():
    left = np.stack([np.zeros(11), np.zeros(11)], axis=1)
    right = np.stack([np.full(11, 10.0), np.zeros(11)], axis=1)
    anchors = [Query(0, 0.0, 0.0), Query(10, 10.0, 0.0)]

    # A very short falloff means both runs sit on their floor weight away from
    # their anchors, so the middle becomes a plain average instead of a ramp.
    pinned = merge_anchor_runs(
        two_runs(left, right), anchors, 11, MergeConfig(falloff_frames=1.0)
    ).track
    assert pinned.points[3, 0] == pytest.approx(5.0, abs=1e-6)
    assert pinned.points[7, 0] == pytest.approx(5.0, abs=1e-6)


def test_the_floor_keeps_a_distant_run_usable():
    left = np.stack([np.zeros(11), np.zeros(11)], axis=1)
    right = np.stack([np.full(11, 10.0), np.zeros(11)], axis=1)
    anchors = [Query(0, 0.0, 0.0), Query(10, 10.0, 0.0)]

    high_floor = merge_anchor_runs(
        two_runs(left, right), anchors, 11, MergeConfig(falloff_floor=1.0)
    ).track
    assert high_floor.points[1, 0] == pytest.approx(5.0, abs=1e-6)  # no falloff at all


def test_zero_confidence_everywhere_still_merges_a_position():
    left = np.stack([np.zeros(11), np.zeros(11)], axis=1)
    right = np.stack([np.full(11, 10.0), np.zeros(11)], axis=1)
    runs = two_runs(left, right, confidence=(0.0, 0.0))
    anchors = [Query(0, 0.0, 0.0), Query(10, 10.0, 0.0)]

    merged = merge_anchor_runs(runs, anchors, 11).track

    assert np.isfinite(merged.points).all()
    assert merged.confidence[5] == 0.0  # ...but nobody is asked to trust it


def test_visibility_is_a_weighted_vote_of_the_contributing_runs():
    positions = np.zeros((5, 2), dtype=np.float32)
    seen = PointTrack(
        points=positions,
        confidence=np.full(5, 0.9, dtype=np.float32),
        visible=np.ones(5, dtype=bool),
    )
    unseen = PointTrack(
        points=positions,
        confidence=np.full(5, 0.9, dtype=np.float32),
        visible=np.zeros(5, dtype=bool),
    )
    anchors = [Query(0, 0.0, 0.0), Query(4, 0.0, 0.0)]
    runs = [TrackRun(anchors[0], "fwd", seen), TrackRun(anchors[1], "bwd", unseen)]

    merged = merge_anchor_runs(runs, anchors, 5).track
    # Equally confident runs, so the nearer anchor's run carries the vote.
    assert bool(merged.visible[1])
    assert not bool(merged.visible[3])


def test_merged_confidence_never_leaves_the_contract(long_video):
    tracker = DriftingTracker(drift=(2.0, 2.0))
    anchors = [anchor_at(long_video, 0), anchor_at(long_video, 15), anchor_at(long_video, 40)]

    result = merge_with(tracker, long_video, anchors)

    assert np.isfinite(result.track.confidence).all()
    assert result.track.confidence.min() >= 0.0
    assert result.track.confidence.max() <= 1.0


def test_report_counts_contributions_per_frame(long_video):
    tracker = DriftingTracker(drift=(0.1, 0.0))
    anchors = [anchor_at(long_video, 0), anchor_at(long_video, 20)]

    report = merge_with(tracker, long_video, anchors).report

    assert report.contributions[:21].min() == 2  # bracketed by two runs
    assert report.contributions[25:].max() == 1  # only the trailing run
    assert np.isnan(report.disagreement[25:]).all()


# -- the blend is tracker-agnostic ---------------------------------------


def test_the_same_blend_applies_to_a_phase_1_plugin(video):
    """Merging is plugin-independent: `lk` gets the identical treatment."""
    anchors = [anchor_at(video, 0), anchor_at(video, 12), anchor_at(video, 23)]

    result = merge_with(LKTracker(), video, anchors)

    for anchor in anchors:
        assert result.track.points[anchor.frame] == pytest.approx(anchor.position, abs=1e-4)
    error = truth_error(result.track, video)
    assert error.max() < 1.5
    assert result.report.contributions.min() >= 1
    assert result.track.confidence.min() > 0.5
