"""Contract tests for Query, TrackResult and PointTrack."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import ContractError
from stable.core.types import PointTrack, Query, TrackResult


def make_result(n: int = 2, t: int = 5) -> TrackResult:
    points = np.arange(n * t * 2, dtype=np.float32).reshape(n, t, 2)
    confidence = np.full((n, t), 0.5, dtype=np.float32)
    visible = np.ones((n, t), dtype=bool)
    return TrackResult(points=points, confidence=confidence, visible=visible)


# -- Query ---------------------------------------------------------------


def test_query_round_trip():
    q = Query(frame=3, x=1.5, y=-2.25)
    assert Query.from_dict(q.to_dict()) == q
    assert q.position == (1.5, -2.25)


def test_query_coerces_and_validates():
    q = Query(frame=np.int64(4), x=1, y=2)
    assert isinstance(q.frame, int) and isinstance(q.x, float)
    with pytest.raises(ContractError):
        Query(frame=-1, x=0, y=0)
    with pytest.raises(ContractError):
        Query(frame=0, x=float("nan"), y=0)
    with pytest.raises(ContractError):
        Query.from_dict({"frame": 0, "x": 0})


# -- TrackResult ---------------------------------------------------------


def test_track_result_shapes():
    r = make_result(3, 7)
    assert (r.n_points, r.n_frames) == (3, 7)
    assert len(r) == 3
    assert r.points.dtype == np.float32
    assert r.confidence.dtype == np.float32
    assert r.visible.dtype == bool


def test_track_result_rejects_contract_violations():
    with pytest.raises(ContractError):
        TrackResult(np.zeros((2, 5, 3)), np.zeros((2, 5)), np.zeros((2, 5), bool))
    with pytest.raises(ContractError):
        TrackResult(np.zeros((2, 5, 2)), np.zeros((2, 4)), np.zeros((2, 5), bool))
    with pytest.raises(ContractError):
        TrackResult(np.zeros((2, 5, 2)), np.zeros((2, 5)), np.zeros((3, 5), bool))
    with pytest.raises(ContractError):  # confidence must be in [0, 1]
        TrackResult(np.zeros((1, 2, 2)), np.array([[0.0, 1.5]]), np.ones((1, 2), bool))
    with pytest.raises(ContractError):  # confidence must be finite
        TrackResult(np.zeros((1, 2, 2)), np.array([[0.0, np.nan]]), np.ones((1, 2), bool))
    with pytest.raises(ContractError):  # positions may be NaN but never inf
        TrackResult(
            np.array([[[0.0, np.inf]]]), np.zeros((1, 1)), np.zeros((1, 1), bool)
        )


def test_nan_positions_are_allowed():
    r = TrackResult(np.full((1, 3, 2), np.nan), np.zeros((1, 3)), np.zeros((1, 3), bool))
    assert np.isnan(r.points).all()


def test_raw_output_is_immutable():
    """Hard requirement: raw tracker output can never be edited in place."""
    r = make_result()
    with pytest.raises(ValueError):
        r.points[0, 0, 0] = 99.0
    with pytest.raises(ValueError):
        r.confidence[0, 0] = 0.0
    with pytest.raises(ValueError):
        r.visible[0, 0] = False


def test_constructor_copies_its_inputs():
    points = np.zeros((1, 2, 2), dtype=np.float32)
    r = TrackResult(points, np.zeros((1, 2)), np.ones((1, 2), bool))
    points[0, 0, 0] = 5.0
    assert r.points[0, 0, 0] == 0.0


def test_empty_result_is_all_invalid():
    r = TrackResult.empty(2, 4)
    assert np.isnan(r.points).all()
    assert not r.confidence.any()
    assert not r.visible.any()
    with pytest.raises(ContractError):
        TrackResult.empty(-1, 3)


def test_point_select_and_stack():
    r = make_result(3, 4)
    track = r.point(1)
    assert isinstance(track, PointTrack)
    assert np.array_equal(track.points, r.points[1])
    assert np.array_equal(r.point(-1).points, r.points[2])
    with pytest.raises(IndexError):
        r.point(5)

    subset = r.select([2, 0])
    assert subset.n_points == 2
    assert np.array_equal(subset.points[0], r.points[2])

    restacked = TrackResult.from_tracks(r.tracks())
    assert np.array_equal(restacked.points, r.points)
    with pytest.raises(ContractError):
        TrackResult.from_tracks([])
    with pytest.raises(ContractError):
        TrackResult.from_tracks([PointTrack.empty(3), PointTrack.empty(4)])


def test_npz_round_trip(tmp_path):
    r = make_result(2, 6)
    path = tmp_path / "tracks" / "p1.npz"
    r.save_npz(path)
    assert path.exists()
    loaded = TrackResult.load_npz(path)
    assert np.array_equal(loaded.points, r.points)
    assert np.array_equal(loaded.confidence, r.confidence)
    assert np.array_equal(loaded.visible, r.visible)


def test_npz_rejects_missing_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, points=np.zeros((1, 2, 2)))
    with pytest.raises(ContractError):
        TrackResult.load_npz(path)


# -- PointTrack ----------------------------------------------------------


def test_point_track_helpers():
    points = np.array([[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]], dtype=np.float32)
    track = PointTrack(points, np.array([1.0, 0.0, 0.5]), np.array([True, False, True]))
    assert track.n_frames == 3 and len(track) == 3
    assert track.valid.tolist() == [True, False, True]
    assert track.position_at(0) == (1.0, 2.0)
    assert track.position_at(1) is None
    with pytest.raises(IndexError):
        track.position_at(3)


def test_point_track_is_immutable_and_replace_is_pure():
    track = PointTrack.empty(4)
    with pytest.raises(ValueError):
        track.points[0, 0] = 1.0
    replaced = track.replace(confidence=np.full(4, 0.25, dtype=np.float32))
    assert replaced is not track
    assert not track.confidence.any()
    assert replaced.confidence.tolist() == [0.25] * 4


def test_point_track_result_bridge(tmp_path):
    track = PointTrack.empty(3)
    result = track.as_result()
    assert result.n_points == 1
    path = tmp_path / "p1.npz"
    track.save_npz(path)
    assert PointTrack.load_npz(path).n_frames == 3
    make_result(2, 3).save_npz(tmp_path / "multi.npz")
    with pytest.raises(ContractError):
        PointTrack.load_npz(tmp_path / "multi.npz")


def test_point_track_validation():
    with pytest.raises(ContractError):
        PointTrack(np.zeros((3, 3)), np.zeros(3), np.zeros(3, bool))
    with pytest.raises(ContractError):
        PointTrack(np.zeros((3, 2)), np.zeros(2), np.zeros(3, bool))
    with pytest.raises(ContractError):
        PointTrack(np.zeros((3, 2)), np.zeros(3), np.zeros(2, bool))
    with pytest.raises(ContractError):
        PointTrack(np.zeros((2, 2)), np.array([0.0, 2.0]), np.zeros(2, bool))
    with pytest.raises(ContractError):
        PointTrack.empty(-1)
