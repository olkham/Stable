"""Weighted Umeyama similarity fit — the closed-form heart of the motion solver.

Given the same physical points seen in a reference frame (``source``) and in the
frame being solved (``target``), find the 2D similarity ``s R(theta) p + t`` that
best explains the motion in the weighted least-squares sense::

    minimise  sum_i w_i * || s R a_i + t - b_i ||^2

Umeyama's closed-form solution specialised to 2D (and to rotations only — never
reflections, because a camera cannot mirror a scene) reduces to a handful of
weighted sums, so a frame solves in microseconds and the answer is exact rather
than iterative.

Weights are the tracker confidences. Because every plugin normalises confidence
to *P(this position is accurate to within a few pixels)* (see ``CONTRIBUTING.md``),
"weight by confidence" means the same thing whichever tracker produced the track
— which is what lets the solver treat all trackers uniformly.

Degrees of freedom
------------------
:class:`SolveConstraints` locks rotation and/or scale, and the fit is then
re-derived under that constraint rather than solved fully and clipped after the
fact — locking scale gives a genuine rigid (Procrustes) fit, not a similarity
fit with its scale thrown away. Translation is never constrained here: it is the
only channel whose meaning depends on where the origin is, so gating it is done
by :mod:`stable.solver.motion`, which knows the pivot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..core.errors import SolverError
from ..core.geometry import SimilarityTransform

__all__ = [
    "SolveConstraints",
    "solve_similarity",
    "transform_residuals",
    "weighted_rms",
]

_MIN_SCALE = 1e-6
"""Smallest scale the fit will report; below this the similarity is degenerate."""


@dataclass(frozen=True)
class SolveConstraints:
    """Which degrees of freedom :func:`solve_similarity` may use.

    Args:
        rotate: Allow rotation. When False the fit is constrained to
            ``theta = 0`` and the remaining parameters are re-optimised.
        scale: Allow uniform scaling. When False the fit is constrained to
            ``s = 1``, i.e. a rigid Procrustes fit.
    """

    rotate: bool = True
    scale: bool = True

    @classmethod
    def from_channels(cls, channels: Mapping[str, bool] | None) -> SolveConstraints:
        """Build constraints from a project's ``solve.channels`` mapping.

        Only ``r`` and ``s`` are read; ``x`` and ``y`` gate the *reported*
        translation curves, which is a pivot-dependent decision made in
        :mod:`stable.solver.motion`.

        Args:
            channels: Mapping with optional ``r`` and ``s`` keys; ``None`` means
                "everything enabled".

        Returns:
            The corresponding :class:`SolveConstraints`.
        """
        if channels is None:
            return cls()
        return cls(rotate=bool(channels.get("r", True)), scale=bool(channels.get("s", True)))

    @property
    def dof(self) -> int:
        """Number of free parameters: 2 for translation plus rotation and scale."""
        return 2 + int(self.rotate) + int(self.scale)

    @property
    def min_points(self) -> int:
        """Fewest correspondences that determine this fit exactly."""
        return 1 if self.dof <= 2 else 2


def _validate_pair(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce and validate a correspondence set.

    Returns:
        ``(source [K, 2], target [K, 2], weights [K])`` as float64 arrays with
        the weights normalised to sum to 1.

    Raises:
        SolverError: On a bad shape, a non-finite value, a negative weight, or
            weights that sum to zero.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2:
        raise SolverError(f"source must have shape [K, 2], got {src.shape}")
    if dst.shape != src.shape:
        raise SolverError(f"target shape {dst.shape} does not match source shape {src.shape}")
    count = src.shape[0]
    if count == 0:
        raise SolverError("cannot fit a similarity to zero correspondences")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise SolverError("source and target positions must all be finite")

    if weights is None:
        wts = np.ones(count, dtype=np.float64)
    else:
        wts = np.asarray(weights, dtype=np.float64).reshape(-1)
        if wts.shape != (count,):
            raise SolverError(f"weights must have shape [{count}], got {wts.shape}")
        if not np.isfinite(wts).all():
            raise SolverError("weights must be finite")
        if (wts < 0.0).any():
            raise SolverError("weights must be non-negative")
    total = float(wts.sum())
    if total <= 0.0:
        raise SolverError("weights sum to zero: every correspondence was given no say")
    return src, dst, wts / total


def solve_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    constraints: SolveConstraints | None = None,
) -> SimilarityTransform:
    """Fit the weighted least-squares similarity taking ``source`` onto ``target``.

    Args:
        source: ``[K, 2]`` positions in the reference frame.
        target: ``[K, 2]`` positions of the same points in the frame being
            solved, in the same order.
        weights: Optional ``[K]`` non-negative weights (tracker confidences).
            ``None`` weights every correspondence equally. Only their ratios
            matter; they are normalised internally.
        constraints: Which degrees of freedom to use; defaults to all four.

    Returns:
        The :class:`~stable.core.geometry.SimilarityTransform` mapping reference
        coordinates to this frame's coordinates. With a single correspondence
        (or several coincident ones) the result degrades to a pure translation
        rather than failing — rotation and scale are simply unobservable there.

    Raises:
        SolverError: If the arrays are malformed, hold non-finite values, or the
            weights sum to zero.
    """
    limits = constraints or SolveConstraints()
    src, dst, wts = _validate_pair(source, target, weights)

    mean_src = (wts[:, None] * src).sum(axis=0)
    mean_dst = (wts[:, None] * dst).sum(axis=0)
    centred_src = src - mean_src
    centred_dst = dst - mean_dst

    # Weighted 2D cross-covariance, expressed as the two rotation-invariant
    # sums that a 2x2 similarity actually depends on.
    dot = float((wts * (centred_src[:, 0] * centred_dst[:, 0]
                        + centred_src[:, 1] * centred_dst[:, 1])).sum())
    cross = float((wts * (centred_src[:, 0] * centred_dst[:, 1]
                          - centred_src[:, 1] * centred_dst[:, 0])).sum())
    spread = float((wts * (centred_src[:, 0] ** 2 + centred_src[:, 1] ** 2)).sum())

    if spread <= 0.0:
        # Every correspondence sits on the same reference position: rotation and
        # scale leave the fit unchanged, so the only observable motion is a shift.
        rotation, scale = 0.0, 1.0
    elif limits.rotate:
        rotation = math.atan2(cross, dot)
        scale = math.hypot(dot, cross) / spread if limits.scale else 1.0
    else:
        rotation = 0.0
        scale = dot / spread if limits.scale else 1.0

    if limits.scale and scale < _MIN_SCALE:
        # A negative or vanishing scale is a collapsed (or mirrored) point set,
        # not a camera move; fall back to the rigid answer for that frame.
        scale = 1.0
    cos_s = scale * math.cos(rotation)
    sin_s = scale * math.sin(rotation)
    return SimilarityTransform(
        tx=float(mean_dst[0] - (cos_s * mean_src[0] - sin_s * mean_src[1])),
        ty=float(mean_dst[1] - (sin_s * mean_src[0] + cos_s * mean_src[1])),
        rotation=float(rotation),
        scale=float(scale),
    )


def transform_residuals(
    transform: SimilarityTransform, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Return the per-correspondence reprojection error, in pixels.

    Args:
        transform: The fitted similarity.
        source: ``[K, 2]`` reference-frame positions.
        target: ``[K, 2]`` positions in the solved frame.

    Returns:
        ``[K]`` float64 distances ``|transform(source_i) - target_i|``.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    delta = transform.apply(src) - dst
    return np.hypot(delta[:, 0], delta[:, 1])


def weighted_rms(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Return the weighted root-mean-square of ``values``.

    Args:
        values: ``[K]`` residuals, in pixels.
        weights: Optional ``[K]`` non-negative weights; ``None`` weights equally.

    Returns:
        The weighted RMS, or ``0.0`` for an empty input or all-zero weights.
    """
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return 0.0
    if weights is None:
        return float(np.sqrt(np.mean(vals**2)))
    wts = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(wts.sum())
    if total <= 0.0:
        return float(np.sqrt(np.mean(vals**2)))
    return float(np.sqrt(float((wts * vals**2).sum()) / total))
