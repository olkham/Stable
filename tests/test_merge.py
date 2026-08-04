"""Contract tests for the multi-anchor merge API (blend itself is Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import anchor_at, execute_run

from stable.core.errors import ContractError
from stable.core.types import PointTrack, Query
from stable.trackers.dummy import DummyTracker
from stable.trackers.merge import (
    MergeConfig,
    MergeReport,
    MergeResult,
    RunSpec,
    TrackRun,
    merge_anchor_runs,
    plan_anchor_runs,
)

# -- planning ------------------------------------------------------------


def test_single_anchor_at_the_start_needs_one_run():
    runs = plan_anchor_runs([Query(0, 1.0, 1.0)], (0, 10))
    assert [(r.direction, r.frame_range) for r in runs] == [("fwd", (0, 10))]


def test_single_anchor_in_the_middle_runs_both_ways():
    runs = plan_anchor_runs([Query(5, 1.0, 1.0)], (0, 10))
    assert [(r.direction, r.frame_range) for r in runs] == [("bwd", (0, 6)), ("fwd", (5, 10))]
    assert all(r.anchor.frame == 5 for r in runs)


def test_adjacent_anchors_produce_exactly_two_overlapping_runs():
    anchors = [Query(0, 0.0, 0.0), Query(5, 0.0, 0.0), Query(9, 0.0, 0.0)]
    runs = plan_anchor_runs(anchors, (0, 12))
    # Ordered by (range start, direction) — deterministic, one span at a time.
    assert [(r.anchor.frame, r.direction, r.frame_range) for r in runs] == [
        (5, "bwd", (0, 6)),
        (0, "fwd", (0, 6)),
        (9, "bwd", (5, 10)),
        (5, "fwd", (5, 10)),
        (9, "fwd", (9, 12)),
    ]
    # Cost stays ~2x the clip length however many anchors the user adds.
    assert sum(r.n_frames for r in runs) <= 2 * 12 + len(anchors)


def test_planning_ignores_anchors_outside_the_range():
    runs = plan_anchor_runs([Query(3, 0.0, 0.0), Query(99, 0.0, 0.0)], (0, 10))
    assert {r.anchor.frame for r in runs} == {3}


def test_planning_rejects_bad_input():
    with pytest.raises(ContractError, match="distinct frames"):
        plan_anchor_runs([Query(4, 0.0, 0.0), Query(4, 1.0, 1.0)], (0, 10))
    with pytest.raises(ContractError, match="no anchor"):
        plan_anchor_runs([Query(50, 0.0, 0.0)], (0, 10))
    with pytest.raises(ContractError, match="no anchor"):
        plan_anchor_runs([], (0, 10))
    with pytest.raises(ContractError, match="non-empty"):
        plan_anchor_runs([Query(0, 0.0, 0.0)], (5, 5))


def test_degenerate_single_frame_range():
    runs = plan_anchor_runs([Query(3, 1.0, 2.0)], (3, 4))
    assert [(r.direction, r.frame_range) for r in runs] == [("fwd", (3, 4))]


def test_run_spec_validation():
    with pytest.raises(ContractError, match="outside frame_range"):
        RunSpec(Query(20, 0.0, 0.0), "fwd", (0, 10))
    with pytest.raises(ContractError, match="non-empty"):
        RunSpec(Query(0, 0.0, 0.0), "fwd", (5, 5))
    with pytest.raises(ContractError, match="direction"):
        RunSpec(Query(0, 0.0, 0.0), "diagonal", (0, 5))


# -- merging -------------------------------------------------------------


def test_end_to_end_single_anchor_merge(video):
    """Plan -> run with a plugin -> merge, for a point with one anchor."""
    tracker = DummyTracker()
    anchor = anchor_at(video, 6)
    runs = [execute_run(tracker, video, spec) for spec in plan_anchor_runs([anchor], (0, 24))]

    result = merge_anchor_runs(runs, [anchor], video.frame_count)

    assert isinstance(result, MergeResult)
    assert result.track.n_frames == 24
    assert np.isfinite(result.track.points).all()
    for frame in range(24):
        assert result.track.points[frame] == pytest.approx(video.true_position(frame), abs=1e-3)
    assert result.report.contributions.min() >= 1
    assert result.report.anchor_frames == (6,)


def test_merged_track_passes_through_every_anchor_exactly(video):
    """Hard constraint: anchors are ground truth, not just heavily weighted."""
    tracker = DummyTracker(follow_truth=False, drift=(3.0, 3.0))
    anchor = Query(frame=6, x=123.5, y=45.25)
    runs = [execute_run(tracker, video, spec) for spec in plan_anchor_runs([anchor], (0, 24))]

    result = merge_anchor_runs(runs, [anchor], video.frame_count)

    assert result.track.points[6] == pytest.approx((123.5, 45.25))
    assert result.track.confidence[6] == pytest.approx(1.0)
    assert bool(result.track.visible[6])
    # ...even though the run itself drifted away from the anchor elsewhere.
    assert result.track.points[10] == pytest.approx((123.5 + 12.0, 45.25 + 12.0))


def test_anchor_passthrough_overrides_a_wrong_sample():
    track = PointTrack(
        points=np.array([[0.0, 0.0], [50.0, 50.0], [2.0, 2.0]], dtype=np.float32),
        confidence=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        visible=np.ones(3, dtype=bool),
    )
    anchor = Query(1, 1.0, 1.0)
    result = merge_anchor_runs([TrackRun(anchor, "fwd", track)], [anchor], 3)
    assert result.track.points[1] == pytest.approx((1.0, 1.0))


def test_anchor_confidence_is_configurable():
    anchor = Query(0, 4.0, 4.0)
    result = merge_anchor_runs([], [anchor], 3, MergeConfig(anchor_confidence=0.75))
    assert result.track.confidence[0] == pytest.approx(0.75)


def test_merging_no_runs_leaves_only_the_anchors():
    anchor = Query(1, 8.0, 9.0)
    result = merge_anchor_runs([], [anchor], 4)
    assert result.track.points[1] == pytest.approx((8.0, 9.0))
    assert np.isnan(result.track.points[[0, 2, 3]]).all()
    assert result.report.contributions.tolist() == [0, 0, 0, 0]
    assert np.isnan(result.report.disagreement).all()


def test_overlapping_runs_blend_instead_of_raising(video):
    """Two anchors overlap between them; Phase 2 blends rather than refusing.

    The detailed behaviour of that blend lives in ``test_merge_blend.py``; this
    only pins that the Phase 0 ``NotImplementedError`` is gone for good.
    """
    tracker = DummyTracker()
    anchors = [anchor_at(video, 0), anchor_at(video, 12)]
    runs = [execute_run(tracker, video, spec) for spec in plan_anchor_runs(anchors, (0, 24))]

    result = merge_anchor_runs(runs, anchors, video.frame_count)

    assert np.isfinite(result.track.points).all()
    assert result.report.contributions[1:12].min() >= 2
    for frame in range(video.frame_count):
        assert result.track.points[frame] == pytest.approx(video.true_position(frame), abs=1e-3)


def test_overlap_on_an_anchor_frame_is_not_an_overlap(video):
    """The forward and backward runs of one anchor meet on that anchor frame."""
    tracker = DummyTracker()
    anchor = anchor_at(video, 6)
    specs = plan_anchor_runs([anchor], (0, 24))
    assert len(specs) == 2
    runs = [execute_run(tracker, video, spec) for spec in specs]
    assert runs[0].track.valid[6] and runs[1].track.valid[6]

    merge_anchor_runs(runs, [anchor], video.frame_count)  # must not raise


def test_min_confidence_gate_drops_weak_samples():
    weak = PointTrack(
        points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        confidence=np.array([0.9, 0.1], dtype=np.float32),
        visible=np.ones(2, dtype=bool),
    )
    strong = PointTrack(
        points=np.array([[np.nan, np.nan], [7.0, 7.0]], dtype=np.float32),
        confidence=np.array([0.0, 0.8], dtype=np.float32),
        visible=np.array([False, True]),
    )
    anchor = Query(0, 0.0, 0.0)
    runs = [TrackRun(anchor, "fwd", weak), TrackRun(anchor, "fwd", strong)]

    ungated = merge_anchor_runs(runs, [anchor], 2)
    assert ungated.report.contributions.tolist() == [1, 2]  # frame 1 has two contributors

    gated = merge_anchor_runs(runs, [anchor], 2, MergeConfig(min_confidence=0.5))
    assert gated.track.points[1] == pytest.approx((7.0, 7.0))
    assert gated.report.contributions.tolist() == [1, 1]


def test_merge_validates_its_inputs():
    anchor = Query(0, 0.0, 0.0)
    short = TrackRun(anchor, "fwd", PointTrack.empty(3))
    with pytest.raises(ContractError, match="expected 5"):
        merge_anchor_runs([short], [anchor], 5)
    with pytest.raises(ContractError, match="outside the clip"):
        merge_anchor_runs([], [Query(9, 0.0, 0.0)], 5)
    with pytest.raises(ContractError):
        merge_anchor_runs([], [anchor], -1)


def test_merge_config_validation():
    with pytest.raises(ContractError):
        MergeConfig(falloff_frames=0)
    with pytest.raises(ContractError):
        MergeConfig(falloff_floor=0.0)
    with pytest.raises(ContractError):
        MergeConfig(disagreement_px=-1)
    with pytest.raises(ContractError):
        MergeConfig(min_confidence=2.0)
    with pytest.raises(ContractError):
        MergeConfig(anchor_confidence=-0.5)
    assert MergeConfig().falloff_frames is None


def test_report_surfaces_the_worst_frames():
    report = MergeReport(
        contributions=np.array([1, 2, 2, 2]),
        disagreement=np.array([np.nan, 1.0, 9.0, 4.0]),
        anchor_frames=(0,),
    )
    assert report.n_frames == 4
    assert report.worst_frames() == [2, 3, 1]
    assert report.worst_frames(limit=1) == [2]
    empty = MergeReport(np.zeros(3, dtype=int), np.full(3, np.nan))
    assert empty.worst_frames() == []


def test_report_arrays_are_read_only():
    result = merge_anchor_runs([], [Query(0, 0.0, 0.0)], 3)
    with pytest.raises(ValueError):
        result.report.contributions[0] = 5
    with pytest.raises(ValueError):
        result.report.disagreement[0] = 5.0
