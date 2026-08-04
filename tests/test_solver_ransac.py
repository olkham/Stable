"""RANSAC: outlier rejection, determinism and the fall-backs around it."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import SolverError
from stable.core.geometry import SimilarityTransform
from stable.solver.ransac import RansacConfig, solve_similarity_ransac
from stable.solver.similarity import SolveConstraints, solve_similarity

TRUTH = SimilarityTransform(tx=14.0, ty=-6.0, rotation=0.18, scale=1.07)


def clean_correspondences(count: int = 10, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return ``count`` noise-free correspondences of :data:`TRUTH`."""
    rng = np.random.default_rng(seed)
    source = rng.uniform(-150.0, 150.0, size=(count, 2))
    return source, np.asarray(TRUTH.apply(source))


def test_a_single_gross_outlier_is_rejected():
    """One point that slid onto something else must not move the answer."""
    source, target = clean_correspondences()
    target[4] += (70.0, -55.0)

    result = solve_similarity_ransac(source, target)
    assert result.transform.allclose(TRUTH, atol=1e-6)
    assert not result.inliers[4]
    assert result.n_inliers == len(source) - 1

    # Without RANSAC the same data gives a visibly wrong transform.
    assert not solve_similarity(source, target).allclose(TRUTH, atol=1.0)


def test_several_outliers_are_rejected():
    """A minority of bad points loses to the consensus of the good ones."""
    source, target = clean_correspondences(count=14, seed=11)
    bad = [1, 6, 9]
    target[bad] += np.array([[80.0, 40.0], [-65.0, 90.0], [120.0, -30.0]])

    result = solve_similarity_ransac(source, target)
    assert result.transform.allclose(TRUTH, atol=1e-6)
    assert set(np.flatnonzero(~result.inliers)) == set(bad)


def test_confidence_breaks_ties_between_hypotheses():
    """Consensus sets are scored by weight, so a doubtful point counts for less."""
    source, target = clean_correspondences(count=6, seed=5)
    # Split the set in half: three points follow TRUTH, three follow a decoy.
    decoy = SimilarityTransform(tx=-30.0, ty=25.0, rotation=-0.3, scale=0.9)
    target[3:] = decoy.apply(source[3:])
    weights = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05])

    result = solve_similarity_ransac(source, target, weights)
    assert result.transform.allclose(TRUTH, atol=1e-6)
    assert result.inliers[:3].all()


def test_small_point_sets_are_enumerated_and_therefore_deterministic():
    """Below the trial budget every pair is tried, so there is no randomness at all."""
    source, target = clean_correspondences(count=8)
    first = solve_similarity_ransac(source, target)
    second = solve_similarity_ransac(source, target)
    assert first.exhaustive and second.exhaustive
    assert first.transform.allclose(second.transform, atol=0.0)
    assert np.array_equal(first.inliers, second.inliers)


def test_large_point_sets_sample_but_stay_reproducible():
    """Above the budget the sampler is seeded, so two runs still agree exactly."""
    source, target = clean_correspondences(count=40, seed=17)
    target[3] += (90.0, 90.0)
    config = RansacConfig(max_trials=25)
    first = solve_similarity_ransac(source, target, config=config)
    second = solve_similarity_ransac(source, target, config=config)
    assert not first.exhaustive
    assert first.transform.allclose(second.transform, atol=0.0)
    assert np.array_equal(first.inliers, second.inliers)
    assert first.transform.allclose(TRUTH, atol=1e-6)


def test_fewer_than_three_points_skip_ransac_entirely():
    """With no redundancy there is nothing to reject, so the plain fit is used."""
    source = np.array([[0.0, 0.0], [80.0, 20.0]])
    result = solve_similarity_ransac(source, TRUTH.apply(source))
    assert result.trials == 0
    assert result.exhaustive
    assert result.inliers.tolist() == [True, True]
    assert result.transform.allclose(TRUTH, atol=1e-9)


def test_near_coincident_pairs_are_not_used_as_hypotheses():
    """Two points a fraction of a pixel apart cannot pin down rotation or scale."""
    rng = np.random.default_rng(2)
    cluster = rng.normal(scale=0.05, size=(4, 2)) + np.array([10.0, 10.0])
    spread = np.array([[200.0, 0.0], [0.0, 150.0], [180.0, 160.0]])
    source = np.vstack([cluster, spread])
    result = solve_similarity_ransac(source, TRUTH.apply(source))
    assert result.transform.allclose(TRUTH, atol=1e-6)


def test_a_tight_cluster_still_fits_when_no_pair_is_separated_enough():
    """The separation filter is lifted rather than giving up on the frame."""
    rng = np.random.default_rng(4)
    source = rng.normal(scale=0.1, size=(5, 2))
    result = solve_similarity_ransac(
        source, TRUTH.apply(source), config=RansacConfig(min_separation_px=50.0)
    )
    assert result.n_inliers >= 2


def test_no_consensus_falls_back_to_the_plain_weighted_fit():
    """When nothing agrees, the best available answer is still returned."""
    rng = np.random.default_rng(8)
    source = rng.uniform(-100.0, 100.0, size=(5, 2))
    target = rng.uniform(-100.0, 100.0, size=(5, 2))  # unrelated to the source
    result = solve_similarity_ransac(
        source, target, config=RansacConfig(threshold_px=0.001)
    )
    assert result.inliers.all()
    assert result.transform.allclose(solve_similarity(source, target), atol=1e-9)


def test_constraints_are_honoured_by_the_robust_path():
    """Locking a channel constrains the RANSAC refit too, not just the plain one."""
    rigid = SimilarityTransform(tx=14.0, ty=-6.0, rotation=0.18)
    rng = np.random.default_rng(13)
    source = rng.uniform(-150.0, 150.0, size=(9, 2))
    target = np.asarray(rigid.apply(source))
    target[2] += (60.0, 60.0)

    result = solve_similarity_ransac(source, target, constraints=SolveConstraints(scale=False))
    assert result.transform.scale == 1.0
    assert result.transform.allclose(rigid, atol=1e-6)
    assert not result.inliers[2]


def test_a_misspecified_model_finds_no_consensus_and_says_so():
    """Locking scale on a shot that really scaled leaves every point an outlier.

    The honest answer is the constrained fit over everything, with a residual
    large enough that the frame is flagged — not a confident-looking fit built
    from two arbitrary points.
    """
    source, target = clean_correspondences(count=9, seed=13)  # TRUTH scales by 1.07
    result = solve_similarity_ransac(source, target, constraints=SolveConstraints(scale=False))
    assert result.inliers.all()
    assert result.transform.scale == 1.0


def test_zero_weight_points_never_become_the_consensus():
    """A consensus made only of points nobody trusts is not a consensus."""
    source, target = clean_correspondences(count=6, seed=21)
    result = solve_similarity_ransac(source, target, np.zeros(6))
    assert result.inliers.all()  # falls back to fitting everything


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold_px": 0.0}, "threshold_px"),
        ({"max_trials": 0}, "max_trials"),
        ({"min_separation_px": -1.0}, "min_separation_px"),
        ({"early_stop_confidence": 1.0}, "early_stop_confidence"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message):
    """Nonsense tuning fails loudly at construction, not silently at render time."""
    with pytest.raises(SolverError, match=message):
        RansacConfig(**kwargs)
