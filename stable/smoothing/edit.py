"""Curve-editor operations: smoothing a sub-range, and pinning keyframes.

Two things an operator asks of a smoothed curve that a filter alone cannot give:

**"Smooth only this part of the shot."**
    A whip pan in the middle of an otherwise locked-off take should not be
    filtered at all, while the handheld drift either side of it should.
    :func:`blend_subrange` restricts a filtered curve to a frame range and
    cross-fades back to raw over a few frames at each edge, because a hard edge
    would put a step in the curve — and a step in the stabilisation warp is a
    visible jump in the render.

**"Whatever else you do, hit this frame exactly."**
    The frame where an actor's eyeline matters, or the one the shot cuts on.
    :func:`pin_curve` re-seats the smoothed curve on the raw value at each
    pinned frame while leaving the rest of the smoothing intact.

Both are pure ``[T] -> [T]`` array functions with no notion of channels, filters
or projects, so the curve editor, the CLI and the tests all drive them the same
way.

How pinning stays exact
-----------------------
The residual ``d(t) = raw(t) - smoothed(t)`` is known at every pin. A correction
curve ``c`` is built that equals ``d`` at every pin and decays to zero away from
them, and the answer is ``smoothed + c`` — so the value at a pin is exactly
``raw``, by construction rather than by iteration.

Between two pins the shape of ``c`` depends on how far apart they are: pins
closer together than ``2 * radius`` blend directly into one another (there is no
room for the curve to return to the smoothed path and come back), while pins
further apart each decay to zero within ``radius`` frames and leave the
smoothing untouched in between. That is what makes a pin a *local* edit: pinning
frame 100 must not silently drag frame 400 around.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.errors import SmoothingError
from .filters import fill_gaps

__all__ = ["DEFAULT_PIN_RADIUS", "DEFAULT_RANGE_BLEND", "blend_subrange", "pin_curve"]

DEFAULT_PIN_RADIUS = 20
"""Frames a pin reaches either side before the smoothing is fully back in charge."""

DEFAULT_RANGE_BLEND = 5
"""Frames of cross-fade at each edge of a sub-range, so no step is introduced."""


def _smoothstep(values: np.ndarray) -> np.ndarray:
    """Return the C1 smoothstep of ``values`` (assumed already in ``[0, 1]``)."""
    return values * values * (3.0 - 2.0 * values)


def _ramp(distance: np.ndarray, width: float) -> np.ndarray:
    """Return a smooth ``1 -> 0`` ramp over ``width`` frames of ``distance``."""
    if width <= 0.0:
        return np.zeros_like(distance)
    return _smoothstep(np.clip(1.0 - distance / width, 0.0, 1.0))


def blend_subrange(
    raw: np.ndarray,
    smoothed: np.ndarray,
    start: int,
    end: int,
    *,
    blend: int = DEFAULT_RANGE_BLEND,
) -> np.ndarray:
    """Keep ``smoothed`` inside ``[start, end]`` and ``raw`` outside it.

    Args:
        raw: ``[T]`` unsmoothed curve.
        smoothed: ``[T]`` curve the filter chain produced, same length.
        start: First frame of the range, inclusive.
        end: Last frame of the range, inclusive.
        blend: Frames of cross-fade either side of the range. ``0`` gives a hard
            edge, which is almost never what you want on a motion curve.

    Returns:
        ``[T]`` float64 curve: ``smoothed`` in the range, ``raw`` well outside
        it, and a smoothstep cross-fade between the two.

    Raises:
        SmoothingError: If the two curves differ in length, the range is
            inverted, or ``blend`` is negative.
    """
    raw_curve = fill_gaps(np.asarray(raw, dtype=np.float64))
    smooth_curve = np.asarray(smoothed, dtype=np.float64)
    if raw_curve.shape != smooth_curve.shape:
        raise SmoothingError(
            f"raw and smoothed curves must be the same length, got "
            f"{raw_curve.shape} and {smooth_curve.shape}"
        )
    if end < start:
        raise SmoothingError(f"invalid smoothing range [{start}, {end}]: end < start")
    if blend < 0:
        raise SmoothingError(f"range blend must be >= 0, got {blend}")

    n_frames = raw_curve.shape[0]
    if n_frames == 0:
        return raw_curve
    frames = np.arange(n_frames, dtype=np.float64)
    weights = np.zeros(n_frames, dtype=np.float64)
    inside = (frames >= start) & (frames <= end)
    weights[inside] = 1.0
    if blend > 0:
        before = frames < start
        after = frames > end
        weights[before] = _ramp(float(start) - frames[before], float(blend))
        weights[after] = _ramp(frames[after] - float(end), float(blend))
    return raw_curve + weights * (smooth_curve - raw_curve)


def _pin_correction(
    residual: np.ndarray, pins: np.ndarray, n_frames: int, radius: float
) -> np.ndarray:
    """Return the ``[T]`` curve that is ``residual`` at each pin and decays away.

    See the module docstring: adjacent pins closer than ``2 * radius`` blend
    into each other, further-apart ones each fall to zero within ``radius``.
    """
    frames = np.arange(n_frames, dtype=np.float64)
    correction = np.zeros(n_frames, dtype=np.float64)

    head = frames < pins[0]
    correction[head] = residual[0] * _ramp(pins[0] - frames[head], radius)
    tail = frames > pins[-1]
    correction[tail] = residual[-1] * _ramp(frames[tail] - pins[-1], radius)

    for index in range(pins.shape[0] - 1):
        left, right = pins[index], pins[index + 1]
        segment = (frames >= left) & (frames <= right)
        if not segment.any():
            continue
        positions = frames[segment]
        gap = right - left
        if gap <= 2.0 * radius:
            # No room to return to the smoothed curve and come back: one blend.
            step = _smoothstep((positions - left) / gap)
            correction[segment] = residual[index] * (1.0 - step) + residual[index + 1] * step
        else:
            correction[segment] = residual[index] * _ramp(positions - left, radius) + residual[
                index + 1
            ] * _ramp(right - positions, radius)
    correction[pins.astype(np.int64)] = residual
    return correction


def pin_curve(
    raw: np.ndarray,
    smoothed: np.ndarray,
    pins: Sequence[int],
    *,
    radius: int = DEFAULT_PIN_RADIUS,
) -> np.ndarray:
    """Return ``smoothed`` re-seated so it passes exactly through every pin.

    Args:
        raw: ``[T]`` unsmoothed curve — the values the pins lock onto.
        smoothed: ``[T]`` curve the filter chain produced, same length.
        pins: Frames the result must equal ``raw`` at. Duplicates and frames
            outside the clip are ignored; order does not matter.
        radius: How many frames either side of a pin the correction reaches
            before the smoothing is fully back in charge. Must be positive.

    Returns:
        ``[T]`` float64 curve equal to ``raw`` at every pin, and equal to
        ``smoothed`` more than ``radius`` frames from any pin (when the pins are
        far enough apart for that to be possible).

    Raises:
        SmoothingError: If the two curves differ in length or ``radius`` is not
            positive.
    """
    raw_curve = fill_gaps(np.asarray(raw, dtype=np.float64))
    smooth_curve = np.asarray(smoothed, dtype=np.float64).copy()
    if raw_curve.shape != smooth_curve.shape:
        raise SmoothingError(
            f"raw and smoothed curves must be the same length, got "
            f"{raw_curve.shape} and {smooth_curve.shape}"
        )
    if radius <= 0:
        raise SmoothingError(f"pin radius must be positive, got {radius}")

    n_frames = raw_curve.shape[0]
    frames = sorted({int(p) for p in pins if 0 <= int(p) < n_frames})
    if not frames or n_frames == 0:
        return smooth_curve
    pinned = np.asarray(frames, dtype=np.float64)
    residual = raw_curve[np.asarray(frames, dtype=np.int64)] - smooth_curve[
        np.asarray(frames, dtype=np.int64)
    ]
    return smooth_curve + _pin_correction(residual, pinned, n_frames, float(radius))
