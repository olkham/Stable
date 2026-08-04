"""Contract tests for SimilarityTransform (the X, Y, R, S motion model)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable.core.errors import ContractError
from stable.core.geometry import CHANNELS, SimilarityTransform


def test_identity_is_neutral():
    identity = SimilarityTransform.identity()
    assert identity.is_identity()
    points = np.array([[1.0, 2.0], [30.0, -4.0]])
    assert np.allclose(identity.apply(points), points)


def test_apply_matches_matrix():
    t = SimilarityTransform(tx=10.0, ty=-5.0, rotation=0.4, scale=1.7)
    points = np.array([[0.0, 0.0], [3.0, 4.0], [-2.5, 8.0]])
    m = t.to_matrix()
    expected = points @ m[:, :2].T + m[:, 2]
    assert np.allclose(t.apply(points), expected)


def test_matrix_round_trip():
    t = SimilarityTransform(tx=3.5, ty=-2.0, rotation=-0.9, scale=0.6)
    assert SimilarityTransform.from_matrix(t.to_matrix()).allclose(t)
    assert SimilarityTransform.from_matrix(t.to_matrix3()).allclose(t)


def test_from_matrix_rejects_non_similarity():
    shear = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ContractError):
        SimilarityTransform.from_matrix(shear)
    with pytest.raises(ContractError):
        SimilarityTransform.from_matrix(np.zeros((2, 2)))
    with pytest.raises(ContractError):
        SimilarityTransform.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=float))


def test_inverse_undoes_transform():
    t = SimilarityTransform(tx=-40.0, ty=12.0, rotation=1.2, scale=2.5)
    points = np.array([[0.0, 0.0], [100.0, 50.0]])
    assert np.allclose(t.inverse().apply(t.apply(points)), points, atol=1e-9)
    assert t.inverse().compose(t).is_identity(atol=1e-9)


def test_compose_applies_right_hand_side_first():
    a = SimilarityTransform(tx=5.0, ty=0.0, rotation=0.3, scale=1.5)
    b = SimilarityTransform(tx=-2.0, ty=7.0, rotation=-0.8, scale=0.4)
    points = np.array([[2.0, 3.0], [-1.0, 0.5]])
    assert np.allclose(a.compose(b).apply(points), a.apply(b.apply(points)))
    assert (a @ b).allclose(a.compose(b))


def test_about_pivot_leaves_pivot_fixed():
    pivot = (960.0, 540.0)
    t = SimilarityTransform.about_pivot(pivot, rotation=0.7, scale=1.3)
    assert np.allclose(t.apply(np.array(pivot)), pivot, atol=1e-9)


def test_about_pivot_with_translation():
    pivot = (10.0, 20.0)
    t = SimilarityTransform.about_pivot(pivot, rotation=0.2, scale=2.0, translation=(4.0, -6.0))
    assert np.allclose(t.apply(np.array(pivot)), (14.0, 14.0), atol=1e-9)


def test_channels_round_trip():
    t = SimilarityTransform(tx=1.0, ty=2.0, rotation=0.5, scale=1.1)
    channels = t.to_channels()
    assert set(channels) == set(CHANNELS)
    assert SimilarityTransform.from_channels(channels).allclose(t)
    assert SimilarityTransform.from_dict(t.to_dict()).allclose(t)


def test_rotation_is_radians_with_degree_view():
    t = SimilarityTransform(rotation=math.pi / 2)
    assert t.rotation_deg == pytest.approx(90.0)
    assert t.to_channels()["r"] == pytest.approx(math.pi / 2)


def test_lerp_is_geometric_in_scale_and_shortest_in_rotation():
    a = SimilarityTransform(tx=0.0, ty=0.0, rotation=-0.1, scale=1.0)
    b = SimilarityTransform(tx=10.0, ty=-4.0, rotation=0.1, scale=4.0)
    mid = SimilarityTransform.lerp(a, b, 0.5)
    assert mid.tx == pytest.approx(5.0)
    assert mid.ty == pytest.approx(-2.0)
    assert mid.scale == pytest.approx(2.0)
    assert mid.rotation == pytest.approx(0.0)

    wrap = SimilarityTransform.lerp(
        SimilarityTransform(rotation=3.0), SimilarityTransform(rotation=-3.0), 0.5
    )
    assert abs(wrap.rotation) > 3.0  # went the short way, through pi


def test_allclose_handles_full_turns():
    a = SimilarityTransform(rotation=0.25)
    b = SimilarityTransform(rotation=0.25 + 2 * math.pi)
    assert a.allclose(b)


def test_validation_rejects_bad_components():
    with pytest.raises(ContractError):
        SimilarityTransform(scale=0.0)
    with pytest.raises(ContractError):
        SimilarityTransform(tx=float("nan"))
    with pytest.raises(ContractError):
        SimilarityTransform(ty=float("inf"))
    with pytest.raises(ContractError):
        SimilarityTransform.from_channels({"x": 0.0, "y": 0.0, "r": 0.0})


def test_apply_rejects_bad_shape():
    t = SimilarityTransform.identity()
    with pytest.raises(ContractError):
        t.apply(np.zeros((4, 3)))


def test_transform_is_immutable():
    t = SimilarityTransform(tx=1.0)
    with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
        t.tx = 2.0  # type: ignore[misc]
