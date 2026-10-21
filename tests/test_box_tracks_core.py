"""Box tracks at the data-model layer: anchors with boxes, sizes, corrections, merge."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.corrections import (
    DeleteRangeCorrection,
    GapFillCorrection,
    OffsetCorrection,
    ReplaceCorrection,
    RetrackCorrection,
    apply_corrections,
    correction_from_dict,
    ground_truth_anchors,
)
from stable.core.errors import ContractError
from stable.core.project import PointSpec
from stable.core.types import PointTrack, Query, TrackResult
from stable.trackers.base import TrackerPlugin
from stable.trackers.merge import (
    MergeConfig,
    TrackRun,
    enforce_anchor_agreement,
    merge_anchor_runs,
)
from stable.trackers.registry import TrackerRegistry

T = 10


def _track(sizes: np.ndarray | None = None, x0: float = 0.0, conf: float = 1.0) -> PointTrack:
    xs = np.arange(T, dtype=np.float32) + x0
    return PointTrack(
        points=np.stack([xs, xs * 0.0], axis=1),
        confidence=np.full(T, conf, dtype=np.float32),
        visible=np.ones(T, dtype=bool),
        sizes=sizes,
    )


def _sizes(w: float, h: float) -> np.ndarray:
    return np.tile(np.array([w, h], dtype=np.float32), (T, 1))


# -- Query ---------------------------------------------------------------


def test_point_query_has_no_box_and_serialises_without_w_h():
    q = Query(3, 1.0, 2.0)
    assert not q.has_box
    assert q.size is None
    assert q.to_dict() == {"frame": 3, "x": 1.0, "y": 2.0}


def test_box_query_reports_its_size_and_round_trips():
    q = Query(3, 10.0, 20.0, 8, 6)
    assert q.has_box
    assert q.size == (8.0, 6.0)
    assert q.to_dict() == {"frame": 3, "x": 10.0, "y": 20.0, "w": 8.0, "h": 6.0}
    assert Query.from_dict(q.to_dict()) == q


@pytest.mark.parametrize(
    ("w", "h"), [(5.0, None), (None, 5.0), (0.0, 5.0), (5.0, -1.0), (np.nan, 5.0), (5.0, np.inf)]
)
def test_query_rejects_an_incomplete_or_unusable_box(w, h):
    with pytest.raises(ContractError):
        Query(0, 1.0, 1.0, w, h)


# -- TrackResult / PointTrack sizes --------------------------------------


def test_sizes_must_match_the_track_shape():
    with pytest.raises(ContractError):
        _track(sizes=np.ones((T - 1, 2)))
    with pytest.raises(ContractError):
        TrackResult(
            points=np.zeros((2, T, 2)),
            confidence=np.zeros((2, T)),
            visible=np.zeros((2, T), dtype=bool),
            sizes=np.ones((T, 2)),
        )


def test_sizes_reject_inf_and_non_positive_values_but_allow_nan():
    bad = _sizes(4, 4)
    bad[2, 0] = np.inf
    with pytest.raises(ContractError):
        _track(sizes=bad)
    bad = _sizes(4, 4)
    bad[2, 1] = 0.0
    with pytest.raises(ContractError):
        _track(sizes=bad)
    ok = _sizes(4, 4)
    ok[2] = np.nan
    assert _track(sizes=ok).size_at(2) is None


def test_sizes_are_frozen_like_the_other_arrays():
    track = _track(sizes=_sizes(4, 3))
    assert track.sizes is not None
    with pytest.raises(ValueError):
        track.sizes[0, 0] = 1.0


def test_size_at_reports_box_or_none_and_bounds_checks():
    assert _track().size_at(0) is None
    track = _track(sizes=_sizes(4, 3))
    assert track.size_at(5) == (4.0, 3.0)
    with pytest.raises(IndexError):
        track.size_at(T)


def test_point_select_and_stacking_carry_sizes():
    boxed = _track(sizes=_sizes(4, 3))
    plain = _track()
    result = TrackResult.from_tracks([plain, boxed])
    assert result.has_boxes
    assert result.sizes is not None
    assert np.isnan(result.sizes[0]).all()
    assert result.point(1).size_at(0) == (4.0, 3.0)
    assert result.point(0).sizes is not None  # NaN rows, but the field is carried
    assert result.select([1]).sizes is not None
    assert not TrackResult.from_tracks([plain, plain]).has_boxes


def test_replace_keeps_sizes_unless_given_new_ones():
    track = _track(sizes=_sizes(4, 3))
    assert track.replace().size_at(0) == (4.0, 3.0)
    assert track.replace(sizes=_sizes(9, 9)).size_at(0) == (9.0, 9.0)
    assert track.as_result().has_boxes


def test_npz_round_trips_sizes_and_older_files_load_without_them(tmp_path):
    boxed = _track(sizes=_sizes(4, 3))
    boxed.save_npz(tmp_path / "b.npz")
    assert PointTrack.load_npz(tmp_path / "b.npz").size_at(1) == (4.0, 3.0)

    _track().save_npz(tmp_path / "p.npz")
    with np.load(tmp_path / "p.npz") as data:
        assert "sizes" not in data.files
    assert PointTrack.load_npz(tmp_path / "p.npz").sizes is None
    assert TrackResult.empty(1, T).sizes is None


# -- project -------------------------------------------------------------


def test_point_spec_is_a_box_track_when_an_anchor_states_a_box():
    assert not PointSpec(id="p1", queries=[Query(0, 1, 1)]).is_box
    assert PointSpec(id="p1", queries=[Query(0, 1, 1), Query(4, 2, 2, 5, 5)]).is_box


# -- corrections ---------------------------------------------------------


def test_replace_with_a_box_sets_the_size_even_on_a_point_track():
    out = apply_corrections(_track(), [ReplaceCorrection(frame=3, x=7.0, y=8.0, w=6.0, h=4.0)])
    assert out.size_at(3) == (6.0, 4.0)
    assert out.size_at(2) is None
    assert out.position_at(3) == (7.0, 8.0)


def test_replace_without_a_box_leaves_a_point_track_without_sizes():
    out = apply_corrections(_track(), [ReplaceCorrection(frame=3, x=7.0, y=8.0)])
    assert out.sizes is None


def test_gap_fill_interpolates_sizes_across_the_gap():
    sizes = np.stack([np.linspace(10, 19, T)] * 2, axis=1)
    out = apply_corrections(
        _track(sizes=sizes),
        [
            DeleteRangeCorrection(start=4, end=5),
            GapFillCorrection(start=4, end=5, method="linear"),
        ],
    )
    # Delete cleared sizes; the fill restores them from the knots outside the gap.
    w, h = out.size_at(4) or (0.0, 0.0)
    assert w == pytest.approx(14.0)
    assert h == pytest.approx(14.0)
    assert (out.size_at(5) or (0.0,))[0] == pytest.approx(15.0)


def test_delete_range_clears_sizes():
    out = apply_corrections(_track(sizes=_sizes(4, 3)), [DeleteRangeCorrection(start=2, end=3)])
    assert out.size_at(2) is None and out.size_at(3) is None
    assert out.size_at(4) == (4.0, 3.0)


def test_offset_keeps_sizes():
    out = apply_corrections(_track(sizes=_sizes(4, 3)), [OffsetCorrection(frame=3, dx=2.0, dy=1.0)])
    assert out.size_at(3) == (4.0, 3.0)


def test_retrack_splices_the_run_sizes_and_reasserts_the_stated_box():
    run = _track(sizes=_sizes(9, 7), x0=100.0)
    correction = RetrackCorrection(frame=5, x=50.0, y=60.0, direction="fwd", w=2.0, h=2.0)
    out = apply_corrections(
        _track(sizes=_sizes(4, 3)), [correction], resolve_retrack=lambda _c: run
    )
    assert out.size_at(4) == (4.0, 3.0)  # before the span: untouched
    assert out.size_at(7) == (9.0, 7.0)  # spliced from the run
    assert out.size_at(5) == (2.0, 2.0)  # the operator's box outranks the run
    assert out.position_at(5) == (50.0, 60.0)


def test_ground_truth_anchors_carry_the_boxes_corrections_state():
    anchors = ground_truth_anchors(
        [
            ReplaceCorrection(frame=2, x=1.0, y=1.0, w=5.0, h=6.0),
            RetrackCorrection(frame=6, x=2.0, y=2.0),
        ]
    )
    assert anchors[0].size == (5.0, 6.0)
    assert anchors[1].size is None


def test_corrections_omit_an_unset_box_from_the_project_file_and_round_trip_a_set_one():
    plain = ReplaceCorrection(frame=1, x=1.0, y=2.0).to_dict()
    assert "w" not in plain and "h" not in plain
    assert "w" not in RetrackCorrection(frame=1, x=1.0, y=2.0).to_dict()
    boxed = RetrackCorrection(frame=1, x=1.0, y=2.0, w=3.0, h=4.0)
    again = correction_from_dict(boxed.to_dict())
    assert isinstance(again, RetrackCorrection) and (again.w, again.h) == (3.0, 4.0)


@pytest.mark.parametrize("kwargs", [{"w": 3.0}, {"w": -1.0, "h": 2.0}, {"w": np.nan, "h": 2.0}])
def test_a_correction_box_must_be_complete_positive_and_finite(kwargs):
    with pytest.raises(ContractError):
        ReplaceCorrection(frame=1, x=1.0, y=2.0, **kwargs)
    with pytest.raises(ContractError):
        RetrackCorrection(frame=1, x=1.0, y=2.0, **kwargs)


# -- merge ---------------------------------------------------------------


def _run(anchor: Query, direction: str, track: PointTrack) -> TrackRun:
    return TrackRun(anchor=anchor, direction=direction, track=track)  # type: ignore[arg-type]


def test_merge_blends_sizes_of_overlapping_runs_with_the_position_weights():
    a, b = Query(0, 0.0, 0.0), Query(9, 9.0, 0.0)
    config = MergeConfig(falloff_frames=9.0, falloff_floor=0.05)
    runs = [
        _run(a, "fwd", _track(sizes=_sizes(10, 10))),
        _run(b, "bwd", _track(sizes=_sizes(20, 20))),
    ]
    merged = merge_anchor_runs(runs, [a, b], T, config).track
    near_a, mid, near_b = (merged.size_at(f) for f in (1, 4, 8))
    assert near_a and mid and near_b
    assert 10.0 < near_a[0] < mid[0] < near_b[0] < 20.0


def test_merge_renormalises_over_the_runs_that_know_a_size():
    a, b = Query(0, 0.0, 0.0), Query(9, 9.0, 0.0)
    runs = [_run(a, "fwd", _track(sizes=_sizes(10, 12))), _run(b, "bwd", _track())]
    merged = merge_anchor_runs(runs, [a, b], T).track
    assert merged.size_at(4) == (10.0, 12.0)


def test_merge_passes_a_single_runs_sizes_straight_through():
    a = Query(0, 0.0, 0.0)
    merged = merge_anchor_runs([_run(a, "fwd", _track(sizes=_sizes(7, 5)))], [a], T).track
    assert merged.size_at(6) == (7.0, 5.0)


def test_merge_enforces_an_anchors_box_exactly():
    a = Query(0, 0.0, 0.0, 30.0, 20.0)
    merged = merge_anchor_runs([_run(a, "fwd", _track(sizes=_sizes(7, 5)))], [a], T).track
    assert merged.size_at(0) == (30.0, 20.0)
    assert merged.size_at(1) == (7.0, 5.0)


def test_merge_of_point_runs_has_no_sizes():
    a = Query(0, 0.0, 0.0)
    assert merge_anchor_runs([_run(a, "fwd", _track())], [a], T).track.sizes is None


def test_merge_with_only_an_anchor_box_still_yields_sizes():
    a = Query(0, 0.0, 0.0, 30.0, 20.0)
    merged = merge_anchor_runs([_run(a, "fwd", _track())], [a], T).track
    assert merged.size_at(0) == (30.0, 20.0)
    assert merged.size_at(1) is None


def test_enforce_anchor_agreement_trims_sizes_with_the_positions():
    far = Query(5, 500.0, 0.0)
    trimmed, violations = enforce_anchor_agreement(
        _track(sizes=_sizes(4, 3)), [Query(0, 0.0, 0.0), far], origin=0, direction="fwd"
    )
    assert len(violations) == 1
    assert trimmed.size_at(4) == (4.0, 3.0)
    assert trimmed.size_at(5) is None and trimmed.size_at(9) is None


# -- registry ------------------------------------------------------------


def test_registry_describe_reports_whether_a_tracker_tracks_boxes():
    class Boxy(TrackerPlugin):
        name = "boxy"
        tracks_boxes = True

        def track(self, *args, **kwargs):  # pragma: no cover - never run
            raise NotImplementedError

    class Plain(Boxy):
        name = "plain"
        tracks_boxes = False

    reg = TrackerRegistry()
    reg.register(Boxy)
    reg.register(Plain)
    rows = {row["name"]: row for row in reg.describe()}
    assert rows["boxy"]["tracks_boxes"] is True
    assert rows["plain"]["tracks_boxes"] is False
    assert TrackerPlugin.tracks_boxes is False
