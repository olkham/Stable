"""Phase 5: what each correction type actually does to a track.

The contract tests for the *types* live in ``test_corrections.py``; these are
about :func:`stable.core.corrections.apply_corrections` — the single funnel every
edited track comes out of.
"""

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
)
from stable.core.errors import ContractError
from stable.core.types import PointTrack


def straight_track(n_frames: int = 40, *, confidence: float = 0.9) -> PointTrack:
    """A point moving diagonally at one pixel per frame, fully confident."""
    frames = np.arange(n_frames, dtype=np.float32)
    return PointTrack(
        points=np.stack([10.0 + frames, 20.0 + 2.0 * frames], axis=1),
        confidence=np.full(n_frames, confidence, dtype=np.float32),
        visible=np.ones(n_frames, dtype=bool),
    )


def punch_hole(track: PointTrack, start: int, end: int) -> PointTrack:
    """Return ``track`` with ``[start, end]`` blanked, as a lost tracker would."""
    points = np.array(track.points, copy=True)
    confidence = np.array(track.confidence, copy=True)
    visible = np.array(track.visible, copy=True)
    points[start : end + 1] = np.nan
    confidence[start : end + 1] = 0.0
    visible[start : end + 1] = False
    return PointTrack(points=points, confidence=confidence, visible=visible)


# -- purity and immutability ---------------------------------------------------


def test_the_raw_track_is_never_modified():
    raw = straight_track()
    before = np.array(raw.points, copy=True)
    out = apply_corrections(raw, [ReplaceCorrection(frame=5, x=0.0, y=0.0)])
    assert np.array_equal(raw.points, before)
    assert not out.points.flags.writeable
    assert out is not raw


def test_applying_twice_gives_the_same_answer():
    raw = straight_track()
    edits = [
        OffsetCorrection(frame=10, dx=3.0, dy=-2.0, falloff_before=4, falloff_after=4),
        GapFillCorrection(start=20, end=22),
    ]
    first = apply_corrections(raw, edits)
    second = apply_corrections(raw, edits)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.confidence, second.confidence)


def test_disabled_corrections_change_nothing():
    raw = straight_track()
    out = apply_corrections(raw, [ReplaceCorrection(frame=5, x=0.0, y=0.0, enabled=False)])
    assert np.array_equal(out.points, raw.points)


# -- offset --------------------------------------------------------------------


def test_offset_is_full_at_its_own_frame_and_zero_beyond_the_falloff():
    raw = straight_track()
    out = apply_corrections(
        raw, [OffsetCorrection(frame=20, dx=5.0, dy=-3.0, falloff_before=6, falloff_after=6)]
    )
    assert out.points[20] == pytest.approx(np.array(raw.points[20]) + (5.0, -3.0), abs=1e-4)
    # Exactly at the falloff edge the weight has decayed to zero.
    assert out.points[14] == pytest.approx(raw.points[14], abs=1e-4)
    assert out.points[26] == pytest.approx(raw.points[26], abs=1e-4)
    assert out.points[5] == pytest.approx(raw.points[5], abs=1e-4)


def test_offset_falloff_is_monotone_and_partial_in_between():
    raw = straight_track()
    out = apply_corrections(
        raw, [OffsetCorrection(frame=20, dx=10.0, dy=0.0, falloff_before=8, falloff_after=8)]
    )
    applied = out.points[:, 0] - raw.points[:, 0]
    rising = applied[12:21]
    assert np.all(np.diff(rising) >= -1e-6)
    assert 0.0 < applied[16] < 10.0
    assert applied[20] == pytest.approx(10.0, abs=1e-4)


def test_zero_falloff_touches_exactly_one_frame():
    raw = straight_track()
    out = apply_corrections(raw, [OffsetCorrection(frame=15, dx=4.0, dy=4.0)])
    moved = np.linalg.norm(np.asarray(out.points) - np.asarray(raw.points), axis=1)
    assert moved[15] == pytest.approx(np.hypot(4.0, 4.0), abs=1e-4)
    assert np.count_nonzero(moved > 1e-6) == 1


def test_linear_and_smooth_shapes_differ_but_agree_at_the_ends():
    raw = straight_track()
    kwargs = {"frame": 20, "dx": 10.0, "dy": 0.0, "falloff_before": 10, "falloff_after": 10}
    linear = apply_corrections(raw, [OffsetCorrection(shape="linear", **kwargs)])
    smooth = apply_corrections(raw, [OffsetCorrection(shape="smooth", **kwargs)])
    assert linear.points[20] == pytest.approx(smooth.points[20], abs=1e-4)
    assert linear.points[10] == pytest.approx(smooth.points[10], abs=1e-4)
    # They agree at the ends and at the exact midpoint of the ramp (smoothstep
    # is symmetric), so compare somewhere the eased curve is genuinely behind.
    assert abs(float(linear.points[13, 0]) - float(smooth.points[13, 0])) > 0.5


def test_offset_cannot_invent_a_position_where_none_exists():
    raw = punch_hole(straight_track(), 18, 22)
    out = apply_corrections(
        raw, [OffsetCorrection(frame=20, dx=5.0, dy=5.0, falloff_before=10, falloff_after=10)]
    )
    assert np.isnan(out.points[18:23]).all()
    assert out.points[15] != pytest.approx(raw.points[15], abs=1e-6)


# -- replace -------------------------------------------------------------------


def test_replace_hard_sets_position_confidence_and_visibility():
    raw = punch_hole(straight_track(), 10, 12)
    out = apply_corrections(raw, [ReplaceCorrection(frame=11, x=123.5, y=45.25)])
    assert out.points[11] == pytest.approx((123.5, 45.25), abs=1e-4)
    assert out.confidence[11] == pytest.approx(1.0)
    assert bool(out.visible[11])
    assert np.isnan(out.points[10]).all()


def test_replace_outside_the_clip_is_ignored_rather_than_fatal():
    raw = straight_track(10)
    out = apply_corrections(raw, [ReplaceCorrection(frame=99, x=1.0, y=1.0)])
    assert np.array_equal(out.points, raw.points)


# -- gap fill ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "spline"])
def test_gap_fill_recovers_a_straight_line_exactly(method):
    truth = straight_track()
    raw = punch_hole(truth, 15, 21)
    out = apply_corrections(raw, [GapFillCorrection(start=15, end=21, method=method)])
    assert np.asarray(out.points[15:22]) == pytest.approx(np.asarray(truth.points[15:22]), abs=1e-3)
    assert bool(out.visible[15:22].all())


def test_gap_fill_uses_its_own_confidence_so_the_solver_can_down_weight_it():
    raw = punch_hole(straight_track(), 15, 18)
    out = apply_corrections(raw, [GapFillCorrection(start=15, end=18, confidence=0.25)])
    assert out.confidence[15:19] == pytest.approx(0.25, abs=1e-6)
    assert out.confidence[14] == pytest.approx(0.9, abs=1e-6)


def test_spline_gap_fill_beats_linear_on_a_curved_path():
    n = 60
    frames = np.arange(n, dtype=np.float64)
    curved = PointTrack(
        points=np.stack([frames, 40.0 * np.sin(frames / 9.0)], axis=1).astype(np.float32),
        confidence=np.ones(n, dtype=np.float32),
        visible=np.ones(n, dtype=bool),
    )
    raw = punch_hole(curved, 26, 33)
    spline = apply_corrections(raw, [GapFillCorrection(start=26, end=33, method="spline")])
    linear = apply_corrections(raw, [GapFillCorrection(start=26, end=33, method="linear")])
    truth = np.asarray(curved.points[26:34], dtype=np.float64)
    spline_error = np.abs(np.asarray(spline.points[26:34], np.float64) - truth).max()
    linear_error = np.abs(np.asarray(linear.points[26:34], np.float64) - truth).max()
    assert spline_error < linear_error


def test_gap_fill_at_the_clip_start_holds_the_nearest_known_sample():
    raw = punch_hole(straight_track(), 0, 4)
    out = apply_corrections(raw, [GapFillCorrection(start=0, end=4, method="linear")])
    assert np.isfinite(out.points[0:5]).all()
    assert out.points[0] == pytest.approx(raw.points[5], abs=1e-4)


def test_gap_fill_with_nothing_to_interpolate_from_leaves_the_hole():
    raw = PointTrack.empty(10)
    out = apply_corrections(raw, [GapFillCorrection(start=2, end=5)])
    assert np.isnan(out.points).all()


# -- delete range --------------------------------------------------------------


def test_delete_range_blanks_position_confidence_and_visibility():
    raw = straight_track()
    out = apply_corrections(raw, [DeleteRangeCorrection(start=12, end=17)])
    assert np.isnan(out.points[12:18]).all()
    assert out.confidence[12:18] == pytest.approx(0.0)
    assert not out.visible[12:18].any()
    assert np.isfinite(out.points[11]).all()
    assert np.isfinite(out.points[18]).all()


def test_delete_then_fill_is_the_repair_workflow():
    truth = straight_track()
    out = apply_corrections(
        truth,
        [
            DeleteRangeCorrection(start=20, end=25),
            GapFillCorrection(start=20, end=25, method="spline", confidence=0.4),
        ],
    )
    assert np.asarray(out.points[20:26]) == pytest.approx(np.asarray(truth.points[20:26]), abs=1e-3)
    assert out.confidence[20:26] == pytest.approx(0.4, abs=1e-6)


def test_order_matters_and_is_list_order():
    raw = straight_track()
    fill_then_delete = apply_corrections(
        raw,
        [GapFillCorrection(start=20, end=25), DeleteRangeCorrection(start=20, end=25)],
    )
    assert np.isnan(fill_then_delete.points[20:26]).all()


# -- retrack -------------------------------------------------------------------


def shifted_run(n_frames: int = 40, *, dx: float = 40.0) -> PointTrack:
    """A full-clip run sitting a long way from :func:`straight_track`."""
    frames = np.arange(n_frames, dtype=np.float32)
    return PointTrack(
        points=np.stack([10.0 + dx + frames, 20.0 + 2.0 * frames], axis=1),
        confidence=np.full(n_frames, 0.7, dtype=np.float32),
        visible=np.ones(n_frames, dtype=bool),
    )


def test_retrack_splices_only_its_own_direction():
    raw = straight_track()
    run = shifted_run()
    correction = RetrackCorrection(frame=20, x=99.0, y=98.0, direction="fwd")
    out = apply_corrections(raw, [correction], resolve_retrack=lambda _c: run)
    assert out.points[25] == pytest.approx(run.points[25], abs=1e-4)
    assert out.points[15] == pytest.approx(raw.points[15], abs=1e-4)
    # The user's corrected position outranks the run at its own frame.
    assert out.points[20] == pytest.approx((99.0, 98.0), abs=1e-4)
    assert out.confidence[20] == pytest.approx(1.0)


def test_backward_retrack_covers_the_frames_before_the_query():
    raw = straight_track()
    run = shifted_run()
    out = apply_corrections(
        raw,
        [RetrackCorrection(frame=20, x=99.0, y=98.0, direction="bwd")],
        resolve_retrack=lambda _c: run,
    )
    assert out.points[10] == pytest.approx(run.points[10], abs=1e-4)
    assert out.points[30] == pytest.approx(raw.points[30], abs=1e-4)


def test_unresolved_retrack_still_contributes_its_corrected_position():
    raw = straight_track()
    out = apply_corrections(raw, [RetrackCorrection(frame=20, x=99.0, y=98.0)])
    assert out.points[20] == pytest.approx((99.0, 98.0), abs=1e-4)
    assert out.points[25] == pytest.approx(raw.points[25], abs=1e-4)


def test_retrack_run_must_span_the_same_timeline():
    raw = straight_track(40)
    with pytest.raises(ContractError, match="whole clip timeline"):
        apply_corrections(
            raw,
            [RetrackCorrection(frame=5, x=0.0, y=0.0)],
            resolve_retrack=lambda _c: straight_track(12),
        )


def test_retrack_does_not_overwrite_frames_its_run_never_covered():
    raw = straight_track()
    partial = punch_hole(shifted_run(), 30, 39)
    out = apply_corrections(
        raw,
        [RetrackCorrection(frame=20, x=99.0, y=98.0, direction="fwd")],
        resolve_retrack=lambda _c: partial,
    )
    assert out.points[25] == pytest.approx(partial.points[25], abs=1e-4)
    assert out.points[35] == pytest.approx(raw.points[35], abs=1e-4)
