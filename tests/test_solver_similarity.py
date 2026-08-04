"""The weighted Umeyama fit: exactness, weighting, DOF locks and degeneracy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable.core.errors import SolverError
from stable.core.geometry import SimilarityTransform
from stable.solver.similarity import (
    SolveConstraints,
    solve_similarity,
    transform_residuals,
    weighted_rms,
)

RNG = np.random.default_rng(20240703)


def random_points(count: int, *, spread: float = 200.0) -> np.ndarray:
    """Return ``count`` well-separated random points."""
    return RNG.uniform(-spread, spread, size=(count, 2))


@pytest.mark.parametrize(
    "truth",
    [
        SimilarityTransform.identity(),
        SimilarityTransform(tx=12.5, ty=-3.25),
        SimilarityTransform(rotation=0.3),
        SimilarityTransform(scale=1.4),
        SimilarityTransform(tx=-40.0, ty=17.0, rotation=-1.1, scale=0.62),
        SimilarityTransform(tx=5.0, ty=5.0, rotation=math.pi - 0.01, scale=1.0),
    ],
)
def test_recovers_an_exact_similarity(truth):
    """A noise-free correspondence set recovers the transform that made it."""
    source = random_points(6)
    target = truth.apply(source)
    assert solve_similarity(source, target).allclose(truth, atol=1e-9)


def test_two_points_determine_the_full_similarity():
    """Four degrees of freedom need exactly two correspondences, and no more."""
    truth = SimilarityTransform(tx=9.0, ty=-4.0, rotation=0.42, scale=1.15)
    source = np.array([[0.0, 0.0], [100.0, 30.0]])
    assert solve_similarity(source, truth.apply(source)).allclose(truth, atol=1e-9)


def test_one_point_degrades_to_translation():
    """A single correspondence cannot observe rotation or scale, so it reports none."""
    result = solve_similarity(np.array([[10.0, 20.0]]), np.array([[13.0, 26.0]]))
    assert result.allclose(SimilarityTransform(tx=3.0, ty=6.0), atol=1e-12)


def test_coincident_points_degrade_to_translation():
    """Points stacked on one spot say nothing about rotation or scale either."""
    source = np.full((4, 2), 50.0)
    target = source + np.array([2.0, -7.0])
    result = solve_similarity(source, target)
    assert result.allclose(SimilarityTransform(tx=2.0, ty=-7.0), atol=1e-12)


def test_weights_decide_which_points_the_fit_believes():
    """A heavily weighted subset dominates the least-squares answer."""
    truth = SimilarityTransform(tx=6.0, ty=-2.0, rotation=0.2, scale=1.05)
    source = random_points(6)
    target = truth.apply(source)
    target[0] += (60.0, -80.0)  # one wildly wrong correspondence

    weights = np.ones(6)
    weights[0] = 1e-6
    trusted = solve_similarity(source, target, weights)
    naive = solve_similarity(source, target)

    assert trusted.allclose(truth, atol=1e-3)
    assert not naive.allclose(truth, atol=1.0)


def test_only_weight_ratios_matter():
    """Scaling every weight by the same factor cannot change the answer."""
    source = random_points(5)
    target = SimilarityTransform(tx=3.0, rotation=0.1, scale=1.2).apply(source)
    weights = RNG.uniform(0.1, 1.0, size=5)
    assert solve_similarity(source, target, weights).allclose(
        solve_similarity(source, target, weights * 1000.0), atol=1e-9
    )


def test_locking_rotation_gives_a_constrained_refit_not_a_clipped_one():
    """Rotation off re-optimises the rest; it does not just zero the angle."""
    truth = SimilarityTransform(tx=10.0, ty=4.0, rotation=0.35, scale=1.2)
    source = random_points(8)
    target = truth.apply(source)

    locked = solve_similarity(source, target, constraints=SolveConstraints(rotate=False))
    assert locked.rotation == 0.0
    # The unconstrained answer with its rotation crudely zeroed is a worse fit
    # than the constrained solve, which is the whole point of constraining.
    clipped = SimilarityTransform(truth.tx, truth.ty, 0.0, truth.scale)
    assert weighted_rms(transform_residuals(locked, source, target)) < weighted_rms(
        transform_residuals(clipped, source, target)
    )


def test_locking_scale_gives_a_rigid_fit():
    """Scale off pins the scale at 1 and still recovers the rotation."""
    truth = SimilarityTransform(tx=-3.0, ty=8.0, rotation=0.25, scale=1.0)
    source = random_points(7)
    target = truth.apply(source)
    result = solve_similarity(source, target, constraints=SolveConstraints(scale=False))
    assert result.scale == 1.0
    assert result.allclose(truth, atol=1e-9)


def test_locking_both_gives_a_pure_translation():
    """With rotation and scale off, the fit is the weighted mean displacement."""
    source = random_points(5)
    target = SimilarityTransform(tx=4.0, ty=-9.0, rotation=0.4, scale=1.3).apply(source)
    result = solve_similarity(
        source, target, constraints=SolveConstraints(rotate=False, scale=False)
    )
    assert result.rotation == 0.0
    assert result.scale == 1.0
    expected = (target - source).mean(axis=0)
    assert result.tx == pytest.approx(expected[0])
    assert result.ty == pytest.approx(expected[1])


def test_constraints_report_their_degrees_of_freedom():
    """The DOF and minimum-point counts follow from which locks are on."""
    assert SolveConstraints().dof == 4
    assert SolveConstraints().min_points == 2
    assert SolveConstraints(rotate=False, scale=False).dof == 2
    assert SolveConstraints(rotate=False, scale=False).min_points == 1


def test_constraints_from_project_channels():
    """``solve.channels`` maps onto the rotation and scale locks."""
    assert SolveConstraints.from_channels(None) == SolveConstraints()
    assert SolveConstraints.from_channels({"r": False}) == SolveConstraints(rotate=False)
    assert SolveConstraints.from_channels({"s": False, "x": False}) == SolveConstraints(scale=False)


def test_rotation_is_never_a_reflection():
    """A mirrored point set is not a camera move, so the fit stays a rotation."""
    source = random_points(6)
    target = source * np.array([1.0, -1.0])  # reflection about the x axis
    result = solve_similarity(source, target)
    assert result.scale > 0.0


@pytest.mark.parametrize(
    ("source", "target", "weights", "message"),
    [
        (np.zeros((0, 2)), np.zeros((0, 2)), None, "zero correspondences"),
        (np.zeros((3, 3)), np.zeros((3, 3)), None, "shape"),
        (np.zeros((3, 2)), np.zeros((4, 2)), None, "does not match"),
        (np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(3), "sum to zero"),
        (np.zeros((3, 2)), np.zeros((3, 2)), np.array([1.0, -1.0, 1.0]), "non-negative"),
        (np.zeros((3, 2)), np.zeros((3, 2)), np.array([1.0, np.nan, 1.0]), "finite"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2)), None, "finite"),
    ],
)
def test_malformed_input_is_rejected(source, target, weights, message):
    """Bad shapes, non-finite values and useless weights raise, never guess."""
    with pytest.raises(SolverError, match=message):
        solve_similarity(source, target, weights)


def test_residuals_and_rms():
    """Residuals are per-point distances; the RMS honours the weights."""
    source = np.array([[0.0, 0.0], [10.0, 0.0]])
    target = np.array([[3.0, 4.0], [10.0, 0.0]])
    errors = transform_residuals(SimilarityTransform.identity(), source, target)
    assert errors == pytest.approx([5.0, 0.0])
    assert weighted_rms(errors) == pytest.approx(math.sqrt(12.5))
    assert weighted_rms(errors, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert weighted_rms(np.zeros(0)) == 0.0
    # All-zero weights fall back to the unweighted RMS rather than dividing by zero.
    assert weighted_rms(errors, np.zeros(2)) == pytest.approx(math.sqrt(12.5))
