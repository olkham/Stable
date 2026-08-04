"""Contract tests for the non-destructive correction layer."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.corrections import (
    CORRECTION_TYPES,
    Correction,
    DeleteRangeCorrection,
    GapFillCorrection,
    OffsetCorrection,
    ReplaceCorrection,
    RetrackCorrection,
    apply_corrections,
    correction_from_dict,
)
from stable.core.errors import ContractError
from stable.core.types import PointTrack


def sample_corrections() -> list[Correction]:
    return [
        OffsetCorrection(frame=10, dx=1.5, dy=-2.0, falloff_before=4, falloff_after=6),
        ReplaceCorrection(frame=12, x=100.0, y=200.0),
        GapFillCorrection(start=20, end=25, method="spline", confidence=0.4),
        DeleteRangeCorrection(start=30, end=33),
        RetrackCorrection(frame=40, x=5.0, y=6.0, direction="bidir", tracker="dummy"),
    ]


def test_every_type_round_trips_through_the_project_form():
    for correction in sample_corrections():
        data = correction.to_dict()
        assert data["kind"] in CORRECTION_TYPES
        assert correction_from_dict(data) == correction
        assert Correction.from_dict(data) == correction


def test_ids_are_generated_and_preserved():
    a = OffsetCorrection(frame=1, dx=0.0, dy=0.0)
    b = OffsetCorrection(frame=1, dx=0.0, dy=0.0)
    assert a.id and a.id != b.id
    assert correction_from_dict(a.to_dict()).id == a.id


def test_unknown_keys_are_ignored_on_load():
    data = OffsetCorrection(frame=2, dx=1.0, dy=1.0).to_dict()
    data["from_a_newer_build"] = {"anything": True}
    assert correction_from_dict(data).frame == 2


def test_bad_discriminators_are_rejected():
    with pytest.raises(ContractError):
        correction_from_dict({"frame": 1})
    with pytest.raises(ContractError):
        correction_from_dict({"kind": "teleport", "frame": 1})
    with pytest.raises(ContractError):
        correction_from_dict({"kind": "offset"})  # missing required fields


def test_validation_rules():
    with pytest.raises(ContractError):
        OffsetCorrection(frame=-1, dx=0.0, dy=0.0)
    with pytest.raises(ContractError):
        OffsetCorrection(frame=1, dx=0.0, dy=0.0, falloff_after=-3)
    with pytest.raises(ContractError):
        OffsetCorrection(frame=1, dx=0.0, dy=0.0, shape="bouncy")
    with pytest.raises(ContractError):
        ReplaceCorrection(frame=-2, x=0.0, y=0.0)
    with pytest.raises(ContractError):
        GapFillCorrection(start=10, end=4)
    with pytest.raises(ContractError):
        GapFillCorrection(start=1, end=2, method="magic")
    with pytest.raises(ContractError):
        GapFillCorrection(start=1, end=2, confidence=1.4)
    with pytest.raises(ContractError):
        DeleteRangeCorrection(start=5, end=1)
    with pytest.raises(ContractError):
        RetrackCorrection(frame=1, x=0.0, y=0.0, direction="sideways")


def test_offset_influence_window():
    correction = OffsetCorrection(frame=10, dx=0.0, dy=0.0, falloff_before=15, falloff_after=5)
    assert correction.frame_range == (0, 15)  # clamped at the start of the clip


def test_apply_with_no_corrections_returns_an_untouched_copy():
    raw = PointTrack(
        points=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        confidence=np.array([1.0, 0.5], dtype=np.float32),
        visible=np.array([True, True]),
    )
    out = apply_corrections(raw, [])
    assert out is not raw
    assert np.array_equal(out.points, raw.points)
    assert not out.points.flags.writeable


def test_disabled_corrections_are_skipped():
    raw = PointTrack.empty(4)
    disabled = OffsetCorrection(frame=1, dx=5.0, dy=5.0, enabled=False)
    out = apply_corrections(raw, [disabled])
    assert np.array_equal(np.isnan(out.points), np.isnan(raw.points))


def test_enabled_corrections_are_applied():
    # Phase 0 pinned this signature and left the body raising NotImplementedError
    # until Phase 5; the semantics of each type are covered in
    # test_corrections_apply.py. This only asserts the contract is now honoured.
    raw = PointTrack(
        points=np.zeros((4, 2), dtype=np.float32),
        confidence=np.ones(4, dtype=np.float32),
        visible=np.ones(4, dtype=bool),
    )
    out = apply_corrections(raw, [OffsetCorrection(frame=1, dx=1.0, dy=0.0)])
    assert out.points[1, 0] == pytest.approx(1.0)
    assert out.points[0, 0] == pytest.approx(0.0)
