"""Turning per-model scores into one comparable ``[0, 1]`` confidence.

The solver weights every track by its confidence, and the UI colours the
timeline by it. Both would be nonsense if ``0.8`` meant "80% sure it is visible"
from one plugin and "normalised correlation peak" from another. So every plugin
maps its raw scores onto a single documented meaning:

    **confidence ~ the probability that this position is accurate to within a
    few pixels.**

``0`` means "no information" (occluded, lost, never tracked), ``1`` means "as
good as an anchor". That is what ``lk`` approximates with its forward-backward
error ramp, what ``template`` approximates with its NCC peak, what CoTracker3
gives directly as *visibility x confidence*, and what TAPIR gives as
*(1 - occlusion) x P(error < threshold)*.

Two knobs are enough to line those up:

* :func:`normalize_scores` — a linear rescale with dead zones at both ends.
  Below ``low`` a model is empirically guessing, so it is clamped to 0 rather
  than left as a small non-zero weight the solver would still trust; above
  ``high`` the differences are noise, so it saturates at 1.
* ``gamma`` — optional bias for models whose scores bunch up near 1.

Plugins expose their ``(low, high)`` as constructor arguments so a user can
retune a model against their own footage without touching the solver.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ContractError

__all__ = [
    "sigmoid",
    "sanitize_scores",
    "normalize_scores",
    "combine_scores",
    "visibility_from_confidence",
]


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid, computed without overflowing on big inputs.

    Args:
        values: Raw logits of any shape.

    Returns:
        ``float64`` probabilities in ``(0, 1)`` with the same shape.
    """
    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0
    out = np.empty_like(array)
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """Clamp arbitrary model output into the ``[0, 1]`` contract.

    ``NaN`` and ``-inf`` become ``0`` ("no information") and ``+inf`` becomes
    ``1``; anything else is clipped. Plugins call this last so a surprising
    model release can never make :class:`~stable.core.types.TrackResult` reject
    an otherwise fine track.

    Args:
        scores: Raw scores of any shape.

    Returns:
        ``float32`` values in ``[0, 1]``.
    """
    array = np.asarray(scores, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def normalize_scores(
    scores: np.ndarray,
    *,
    low: float = 0.0,
    high: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Rescale a model's raw scores onto the shared confidence meaning.

    Args:
        scores: Raw scores, nominally in ``[0, 1]``.
        low: Score at or below which the model carries no usable information;
            maps to ``0``.
        high: Score at or above which the model is fully trusted; maps to ``1``.
        gamma: Exponent applied after the rescale. ``> 1`` is stricter (pushes
            middling scores down), ``< 1`` more forgiving.

    Returns:
        ``float32`` confidences in ``[0, 1]``.

    Raises:
        ContractError: If ``high <= low`` or ``gamma <= 0``.
    """
    if not high > low:
        raise ContractError(f"normalize_scores needs high > low, got low={low}, high={high}")
    if gamma <= 0:
        raise ContractError(f"normalize_scores needs gamma > 0, got {gamma}")
    array = np.asarray(scores, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = (array - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, gamma)
    return scaled.astype(np.float32)


def combine_scores(*scores: np.ndarray | None) -> np.ndarray:
    """Combine independent reliability signals into one score by multiplication.

    Independent probabilities multiply: a point that is 0.9 likely visible and
    0.8 likely within tolerance is 0.72 likely to be a usable sample. ``None``
    arguments are skipped so a plugin can pass an optional head straight
    through.

    Args:
        *scores: Arrays broadcastable to a common shape, or ``None``.

    Returns:
        ``float64`` element-wise product, each factor clipped to ``[0, 1]``
        first.

    Raises:
        ContractError: If every argument is ``None``.
    """
    factors = [np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0) for s in scores if s is not None]
    if not factors:
        raise ContractError("combine_scores needs at least one non-None array")
    product = factors[0]
    for factor in factors[1:]:
        product = product * factor
    return product


def visibility_from_confidence(confidence: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Derive the boolean ``visible`` flag from normalised confidence.

    Args:
        confidence: ``[0, 1]`` confidences.
        threshold: Confidence at or above which the point counts as visible.

    Returns:
        A boolean array of the same shape.

    Raises:
        ContractError: If ``threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ContractError(f"threshold must lie in [0, 1], got {threshold}")
    return np.asarray(confidence, dtype=np.float64) >= threshold
