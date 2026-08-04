"""RANSAC outlier rejection around the weighted similarity fit.

A least-squares fit has no defence against a single bad track: one point that
slid onto a passing car drags the whole frame's transform with it, and the error
shows up in the render as a lurch. RANSAC fixes that by fitting from *minimal*
samples — a 2D similarity is pinned down by exactly two correspondences — and
keeping the sample whose consensus set (points landing within
:attr:`RansacConfig.threshold_px`) carries the most confidence weight. The final
transform is then re-fitted on that consensus set with full weights, so the
answer still uses every good point, not just the two that voted it in.

Determinism
-----------
Editorial tools must render the same frames twice. With few points the sampler
**enumerates every pair** (at most :attr:`RansacConfig.max_trials` of them), so
the result is exact and reproducible with no randomness at all; only above that
does it fall back to sampling, and then from a seeded generator. Either way the
same input gives the same output on every machine.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..core.errors import SolverError
from ..core.geometry import SimilarityTransform
from .similarity import SolveConstraints, solve_similarity, transform_residuals

__all__ = ["RansacConfig", "RansacResult", "solve_similarity_ransac"]


@dataclass(frozen=True)
class RansacConfig:
    """Tuning for :func:`solve_similarity_ransac`.

    Args:
        threshold_px: Reprojection error, in pixels, below which a
            correspondence is an inlier of a candidate fit.
        max_trials: Maximum number of minimal samples to try. When the number of
            distinct pairs is at or below this, every pair is enumerated and the
            search is exhaustive (and random-free).
        min_separation_px: Reference-frame distance two sampled points must be
            apart before their pair is used. Near-coincident pairs give wild
            rotation/scale estimates from sub-pixel noise.
        seed: Seed for the fallback random sampler; fixed so runs reproduce.
        early_stop_confidence: Probability of having seen an all-inlier sample
            at which random sampling may stop early. Ignored when enumerating.

    Raises:
        SolverError: If a value is out of range.
    """

    threshold_px: float = 2.0
    max_trials: int = 200
    min_separation_px: float = 1.0
    seed: int = 0
    early_stop_confidence: float = 0.999

    def __post_init__(self) -> None:
        """Validate the tuning values (dataclass hook)."""
        if self.threshold_px <= 0.0:
            raise SolverError("RansacConfig.threshold_px must be positive")
        if self.max_trials < 1:
            raise SolverError("RansacConfig.max_trials must be >= 1")
        if self.min_separation_px < 0.0:
            raise SolverError("RansacConfig.min_separation_px must be >= 0")
        if not 0.0 < self.early_stop_confidence < 1.0:
            raise SolverError("RansacConfig.early_stop_confidence must lie in (0, 1)")


@dataclass(frozen=True)
class RansacResult:
    """The outcome of one RANSAC fit.

    Args:
        transform: The refined similarity, fitted on the consensus set.
        inliers: ``[K]`` bool mask of the correspondences that agreed.
        trials: How many minimal samples were actually evaluated.
        exhaustive: Whether every distinct pair was enumerated (so the search
            was deterministic and complete).
    """

    transform: SimilarityTransform
    inliers: np.ndarray
    trials: int
    exhaustive: bool

    @property
    def n_inliers(self) -> int:
        """Number of correspondences in the consensus set."""
        return int(np.count_nonzero(self.inliers))


def _candidate_pairs(
    source: np.ndarray, config: RansacConfig
) -> tuple[list[tuple[int, int]], bool]:
    """Return the minimal samples to try, and whether the list is exhaustive.

    Pairs whose reference-frame separation is below
    :attr:`RansacConfig.min_separation_px` are dropped: they cannot pin down a
    rotation or a scale. If that leaves nothing, the filter is lifted rather
    than giving up — a tightly clustered point set is still worth fitting.
    """
    count = source.shape[0]
    total_pairs = count * (count - 1) // 2
    separation_ok: list[tuple[int, int]] = []
    if total_pairs <= config.max_trials:
        for i, j in itertools.combinations(range(count), 2):
            if math.dist(source[i], source[j]) >= config.min_separation_px:
                separation_ok.append((i, j))
        if separation_ok:
            return separation_ok, True
        return list(itertools.combinations(range(count), 2)), True

    rng = np.random.default_rng(config.seed)
    for _ in range(config.max_trials):
        i, j = (int(v) for v in rng.choice(count, size=2, replace=False))
        separation_ok.append((i, j))
    return separation_ok, False


def _required_trials(inlier_ratio: float, confidence: float) -> float:
    """Return how many random pairs are needed to see one all-inlier sample.

    Standard RANSAC stopping rule for a minimal sample of size 2:
    ``log(1 - confidence) / log(1 - ratio^2)``.
    """
    if inlier_ratio <= 0.0:
        return math.inf
    if inlier_ratio >= 1.0:
        return 1.0
    denominator = math.log(1.0 - inlier_ratio**2)
    if denominator >= 0.0:  # pragma: no cover - ratio == 1 handled above
        return 1.0
    return math.log(1.0 - confidence) / denominator


def solve_similarity_ransac(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    constraints: SolveConstraints | None = None,
    config: RansacConfig | None = None,
) -> RansacResult:
    """Fit a similarity robustly, rejecting correspondences that disagree.

    With fewer than three correspondences there is nothing to reject — the fit
    is exactly or under-determined — so this falls straight through to
    :func:`~stable.solver.similarity.solve_similarity` with every point marked
    an inlier.

    Args:
        source: ``[K, 2]`` reference-frame positions.
        target: ``[K, 2]`` positions in the frame being solved.
        weights: Optional ``[K]`` non-negative confidences. Consensus sets are
            scored by total inlier *weight*, so a confident point counts for
            more than a doubtful one when two hypotheses tie on raw count.
            All-zero weights are treated as uniform.
        constraints: Degrees of freedom the fit may use.
        config: RANSAC tuning; defaults to :class:`RansacConfig`.

    Returns:
        A :class:`RansacResult`. When no hypothesis wins support beyond its own
        minimal sample, the plain weighted fit over every point is returned with
        all points marked inliers — a frame nobody agrees about should look bad
        in the residual, not quietly adopt two arbitrary points as the truth.

    Raises:
        SolverError: If the correspondence arrays are malformed (see
            :func:`~stable.solver.similarity.solve_similarity`).
    """
    limits = constraints or SolveConstraints()
    tuning = config or RansacConfig()
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    count = src.shape[0] if src.ndim == 2 else 0

    if count < 3:
        transform = solve_similarity(src, dst, weights, constraints=limits)
        return RansacResult(
            transform=transform,
            inliers=np.ones(max(count, 0), dtype=bool),
            trials=0,
            exhaustive=True,
        )

    wts = (
        np.ones(count, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if wts.size == count and float(wts.sum()) <= 0.0:
        # Nobody is confident about anything in this frame. That is a reason to
        # weight the points equally, not a reason to refuse to solve: the frame
        # still gets an answer, and its residual still flags it for review.
        wts = np.ones(count, dtype=np.float64)
    pairs, exhaustive = _candidate_pairs(src, tuning)

    best_mask = np.zeros(count, dtype=bool)
    best_score = -1.0
    best_error = math.inf
    trials = 0
    for i, j in pairs:
        trials += 1
        pair = [i, j]
        try:
            candidate = solve_similarity(src[pair], dst[pair], wts[pair], constraints=limits)
        except SolverError:  # pragma: no cover - both weights zero for this pair
            continue
        errors = transform_residuals(candidate, src, dst)
        mask: np.ndarray = np.asarray(errors <= tuning.threshold_px, dtype=bool)
        # Weight the vote by confidence, but keep a floor so a consensus set of
        # zero-confidence points still beats no consensus at all.
        score = float(wts[mask].sum()) + 1e-6 * float(np.count_nonzero(mask))
        error = float(errors[mask].sum()) if mask.any() else math.inf
        if score > best_score or (score == best_score and error < best_error):
            best_score, best_error, best_mask = score, error, mask
        if not exhaustive:
            ratio = float(np.count_nonzero(best_mask)) / count
            if trials >= _required_trials(ratio, tuning.early_stop_confidence):
                break

    # A minimal sample always fits its own two points exactly, so "two inliers"
    # is no evidence at all. A hypothesis has to win *support beyond its own
    # sample* to be believed; otherwise the honest answer is to fit everything
    # and let the residual flag the frame.
    min_consensus = min(3, count)
    if np.count_nonzero(best_mask) < min_consensus or float(wts[best_mask].sum()) <= 0.0:
        transform = solve_similarity(src, dst, wts, constraints=limits)
        return RansacResult(
            transform=transform,
            inliers=np.ones(count, dtype=bool),
            trials=trials,
            exhaustive=exhaustive,
        )

    refined = solve_similarity(src[best_mask], dst[best_mask], wts[best_mask], constraints=limits)
    # One re-scoring pass with the refined transform: points that only just
    # missed the minimal-sample hypothesis usually agree with the refit.
    final_mask: np.ndarray = np.asarray(
        transform_residuals(refined, src, dst) <= tuning.threshold_px, dtype=bool
    )
    if np.count_nonzero(final_mask) >= min_consensus and float(wts[final_mask].sum()) > 0.0:
        refined = solve_similarity(
            src[final_mask], dst[final_mask], wts[final_mask], constraints=limits
        )
        best_mask = final_mask

    return RansacResult(
        transform=refined, inliers=best_mask, trials=trials, exhaustive=exhaustive
    )
