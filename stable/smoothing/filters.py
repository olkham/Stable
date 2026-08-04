"""The smoothing filter catalogue — one registry, six filters, uniform contract.

Every filter takes a raw ``[T]`` curve and returns a smoothed ``[T]`` curve of
the same length. Nothing here knows which channel it is working on beyond what
:class:`FilterContext` tells it, so the same catalogue serves ``x``, ``y``,
``r`` and ``s`` and any future channel.

The filters
-----------
``moving_average``
    Box filter. Cheap, obvious, and rings a little; the one users reach for first.
``gaussian``
    Smooth roll-off, no ringing. The safe default for organic-looking motion.
``butterworth``
    Maximally flat low-pass applied forward and backward (zero phase). Cuts by
    *frequency*, so "remove everything faster than 1 Hz" is a thing you can
    actually ask for. This is the filter for handheld judder.
``savgol``
    Savitzky-Golay: fits a local polynomial, so it removes wobble while
    preserving the shape and speed of a deliberate move (a whip pan survives it).
``spline``
    Penalised cubic smoothing spline — global, gives the smoothest curve for a
    given amount of deviation from the raw motion.
``tripod``
    Locks the channel to its value at the reference frame. Combined with a full
    solve this is the classic "nail the shot down" mode: the stabilisation warp
    becomes ``raw(t)^-1`` and the frame stops dead.

Strength
--------
Every filter accepts a universal ``strength`` in ``[0, 1]``, applied as
``raw + strength * (filtered - raw)``. ``0`` is a bypass, ``1`` is the filter's
full effect, and anything between is a partial application — which is what
"stabilise this shot 70%, keep some life in it" means to an editor. Because the
blend is linear it never introduces motion that neither the raw nor the filtered
curve had.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import SmoothingError
from . import dsp

__all__ = [
    "FilterContext",
    "FilterParam",
    "FilterInfo",
    "FILTERS",
    "STRENGTH",
    "get_filter",
    "filter_names",
    "describe_filters",
    "apply_filter",
    "fill_gaps",
]


@dataclass(frozen=True)
class FilterContext:
    """What a filter needs to know about the curve beyond its samples.

    Args:
        fps: Frame rate of the clip, so filters can work in Hz and seconds.
        reference_frame: The solve's reference frame — where ``tripod`` locks to.
        channel: Which of ``x``, ``y``, ``r``, ``s`` is being smoothed.
        n_frames: Clip length ``T``.
    """

    fps: float = 25.0
    reference_frame: int = 0
    channel: str = "x"
    n_frames: int = 0

    def __post_init__(self) -> None:
        """Validate the context (dataclass hook)."""
        if self.fps <= 0.0:
            raise SmoothingError(f"FilterContext.fps must be positive, got {self.fps}")

    @property
    def nyquist_hz(self) -> float:
        """Highest frequency the frame rate can represent, in Hz."""
        return self.fps / 2.0


@dataclass(frozen=True)
class FilterParam:
    """One tunable of one filter — the schema a UI builds a control from.

    Args:
        name: Parameter key inside ``SmoothingSpec.params``.
        kind: ``"int"`` or ``"float"``.
        default: Value used when the caller omits it.
        minimum: Inclusive lower bound, or ``None``.
        maximum: Inclusive upper bound, or ``None``.
        unit: Human unit label (``"frames"``, ``"Hz"``, ``""``).
        description: One-line explanation for a tooltip.
    """

    name: str
    kind: str
    default: float
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    description: str = ""

    def coerce(self, value: Any) -> float | int:
        """Validate and convert one supplied value.

        Args:
            value: The raw value from a project file or CLI argument.

        Returns:
            The value as an ``int`` or ``float``.

        Raises:
            SmoothingError: If it is not numeric or falls outside the bounds.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SmoothingError(
                f"parameter {self.name!r} must be a number, got {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise SmoothingError(f"parameter {self.name!r} must be finite, got {value!r}")
        if self.minimum is not None and number < self.minimum:
            raise SmoothingError(f"parameter {self.name!r} must be >= {self.minimum}, got {number}")
        if self.maximum is not None and number > self.maximum:
            raise SmoothingError(f"parameter {self.name!r} must be <= {self.maximum}, got {number}")
        return int(round(number)) if self.kind == "int" else number

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe schema dict for this parameter."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "description": self.description,
        }


STRENGTH = FilterParam(
    name="strength",
    kind="float",
    default=1.0,
    minimum=0.0,
    maximum=1.0,
    unit="",
    description="How much of the filter's effect to apply; 0 bypasses it, 1 applies it fully.",
)
"""The universal strength parameter, accepted by every filter."""

FilterFn = Callable[[np.ndarray, FilterContext, Mapping[str, Any]], np.ndarray]
"""Signature of a filter implementation: ``(values, context, params) -> values``."""


@dataclass(frozen=True)
class FilterInfo:
    """A registered filter: its schema and its implementation.

    Args:
        name: Registry key, as stored in ``SmoothingSpec.filter``.
        display_name: Human label for the UI.
        description: One-line explanation of what it is good for.
        params: Its tunables, excluding the universal ``strength``.
        fn: The implementation.
    """

    name: str
    display_name: str
    description: str
    params: tuple[FilterParam, ...]
    fn: FilterFn = field(repr=False)

    @property
    def all_params(self) -> tuple[FilterParam, ...]:
        """Its own parameters plus the universal :data:`STRENGTH`."""
        return (*self.params, STRENGTH)

    def coerce(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate a parameter mapping and fill in defaults.

        Args:
            params: Values from a project file or the CLI; ``None`` means "all
                defaults".

        Returns:
            A complete, validated parameter dict.

        Raises:
            SmoothingError: On an unknown key or an invalid value. Unknown keys
                are an error rather than being ignored, because a silently
                dropped ``cutoff_hz`` typo looks exactly like a filter that does
                not work.
        """
        supplied = dict(params or {})
        schema = {p.name: p for p in self.all_params}
        unknown = sorted(set(supplied) - set(schema))
        if unknown:
            raise SmoothingError(
                f"filter {self.name!r} has no parameter(s) {unknown}; "
                f"expected any of {sorted(schema)}"
            )
        return {
            name: param.coerce(supplied[name]) if name in supplied else param.default
            for name, param in schema.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe schema dict for this filter."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "params": [p.to_dict() for p in self.all_params],
        }


# -- implementations ---------------------------------------------------------


def _moving_average(
    values: np.ndarray, context: FilterContext, params: Mapping[str, Any]
) -> np.ndarray:
    """Box-filter the curve over an odd window, odd-reflected at the ends."""
    window = int(params["window"])
    if window % 2 == 0:
        window += 1
    if window <= 1 or values.shape[0] < 2:
        return values.copy()
    window = min(window, values.shape[0] * 2 - 1)
    half = window // 2
    padded = dsp.odd_extension(values, half)
    if padded.shape[0] < window:  # pragma: no cover - only for a 1-sample curve
        return values.copy()
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    smoothed = np.convolve(padded, kernel, mode="same")
    start = (padded.shape[0] - values.shape[0]) // 2
    return smoothed[start : start + values.shape[0]]


def _gaussian_sigma(context: FilterContext, params: Mapping[str, Any]) -> float:
    """Return the Gaussian width in frames, from ``cutoff_hz`` when given."""
    cutoff = float(params["cutoff_hz"])
    if cutoff > 0.0:
        # -3 dB point of exp(-(sigma w)^2 / 2) sits at sigma = sqrt(ln 2) / w_c.
        return math.sqrt(math.log(2.0)) * context.fps / (2.0 * math.pi * cutoff)
    return float(params["sigma"])


def _gaussian(values: np.ndarray, context: FilterContext, params: Mapping[str, Any]) -> np.ndarray:
    """Convolve the curve with a normalised Gaussian, odd-reflected at the ends."""
    sigma = _gaussian_sigma(context, params)
    if sigma <= 0.0 or values.shape[0] < 2:
        return values.copy()
    kernel = dsp.gaussian_kernel(sigma, max_radius=values.shape[0] - 1)
    padded = dsp.odd_extension(values, kernel.shape[0] // 2)
    smoothed = np.convolve(padded, kernel, mode="same")
    start = (padded.shape[0] - values.shape[0]) // 2
    return smoothed[start : start + values.shape[0]]


def _butterworth(
    values: np.ndarray, context: FilterContext, params: Mapping[str, Any]
) -> np.ndarray:
    """Zero-phase Butterworth low-pass at ``cutoff_hz``."""
    cutoff = float(params["cutoff_hz"])
    order = int(params["order"])
    if values.shape[0] < 3:
        return values.copy()
    if cutoff >= context.nyquist_hz:
        raise SmoothingError(
            f"butterworth cutoff_hz {cutoff:g} is at or above Nyquist "
            f"({context.nyquist_hz:g} Hz at {context.fps:g} fps); nothing would be filtered"
        )
    sos = dsp.butter_lowpass_sos(order, cutoff, context.fps)
    return dsp.sosfiltfilt(sos, values)


def _savgol(values: np.ndarray, context: FilterContext, params: Mapping[str, Any]) -> np.ndarray:
    """Fit a local polynomial over a sliding window (Savitzky-Golay)."""
    return dsp.savgol_filter(values, int(params["window"]), int(params["polyorder"]))


def _spline(values: np.ndarray, context: FilterContext, params: Mapping[str, Any]) -> np.ndarray:
    """Fit a penalised natural cubic smoothing spline."""
    cutoff = float(params["cutoff_hz"])
    if cutoff > 0.0:
        if cutoff >= context.nyquist_hz:
            raise SmoothingError(
                f"spline cutoff_hz {cutoff:g} is at or above Nyquist "
                f"({context.nyquist_hz:g} Hz at {context.fps:g} fps)"
            )
        # The second-difference penalty gives |H(w)| = 1 / (1 + lam (2 - 2cos w)^2);
        # solving for the half-power point fixes lam from a frequency.
        omega = 2.0 * math.pi * cutoff / context.fps
        lam = 1.0 / max((2.0 - 2.0 * math.cos(omega)) ** 2, 1e-18)
    else:
        lam = float(params["lam"])
    return dsp.smoothing_spline(values, lam)


def _tripod(values: np.ndarray, context: FilterContext, params: Mapping[str, Any]) -> np.ndarray:
    """Lock the curve to its value at the reference frame.

    With every channel on tripod the smoothed motion is the identity, so the
    stabilisation warp collapses to ``raw(t)^-1`` and the shot is nailed to the
    reference frame — the "locked-off camera" look.
    """
    if values.shape[0] == 0:
        return values.copy()
    frame = min(max(int(context.reference_frame), 0), values.shape[0] - 1)
    return np.full_like(values, float(values[frame]))


FILTERS: dict[str, FilterInfo] = {
    info.name: info
    for info in (
        FilterInfo(
            name="moving_average",
            display_name="Moving average",
            description="Box average over a window of frames. Simple and fast.",
            params=(
                FilterParam(
                    "window", "int", 15, minimum=1, unit="frames",
                    description="Window length; forced odd so the filter stays centred.",
                ),
            ),
            fn=_moving_average,
        ),
        FilterInfo(
            name="gaussian",
            display_name="Gaussian",
            description="Gaussian-weighted average: smooth roll-off, no ringing.",
            params=(
                FilterParam(
                    "sigma", "float", 8.0, minimum=0.0, unit="frames",
                    description="Standard deviation in frames; ignored when cutoff_hz is set.",
                ),
                FilterParam(
                    "cutoff_hz", "float", 0.0, minimum=0.0, unit="Hz",
                    description="Optional -3 dB cutoff; 0 means use sigma instead.",
                ),
            ),
            fn=_gaussian,
        ),
        FilterInfo(
            name="butterworth",
            display_name="Butterworth (zero phase)",
            description="Maximally flat low-pass run forward and backward, so no lag.",
            params=(
                FilterParam(
                    "cutoff_hz", "float", 1.0, minimum=1e-6, unit="Hz",
                    description="Motion faster than this is removed.",
                ),
                FilterParam(
                    "order", "int", 2, minimum=1, maximum=8, unit="",
                    description="Roll-off steepness; doubled again by the two-pass application.",
                ),
            ),
            fn=_butterworth,
        ),
        FilterInfo(
            name="savgol",
            display_name="Savitzky-Golay",
            description="Local polynomial fit: removes wobble but keeps deliberate moves sharp.",
            params=(
                FilterParam(
                    "window", "int", 21, minimum=3, unit="frames",
                    description="Window length; forced odd and clipped to the clip length.",
                ),
                FilterParam(
                    "polyorder", "int", 3, minimum=0, maximum=9, unit="",
                    description="Degree of the fitted polynomial; must stay below the window.",
                ),
            ),
            fn=_savgol,
        ),
        FilterInfo(
            name="spline",
            display_name="Smoothing spline",
            description="Penalised cubic spline: the smoothest curve for a given deviation.",
            params=(
                FilterParam(
                    "lam", "float", 1.0, minimum=0.0, unit="",
                    description="Roughness penalty; higher is smoother. 0 is a bypass.",
                ),
                FilterParam(
                    "cutoff_hz", "float", 0.0, minimum=0.0, unit="Hz",
                    description="Optional half-power cutoff to derive lam from; 0 means use lam.",
                ),
            ),
            fn=_spline,
        ),
        FilterInfo(
            name="tripod",
            display_name="Tripod (lock to reference)",
            description="Holds the channel at its reference-frame value: a locked-off camera.",
            params=(),
            fn=_tripod,
        ),
    )
}
"""Every registered filter, keyed by the name project files store."""


def filter_names() -> list[str]:
    """Return the registered filter names, sorted."""
    return sorted(FILTERS)


def get_filter(name: str) -> FilterInfo:
    """Return the registered filter called ``name``.

    Args:
        name: A key of :data:`FILTERS`.

    Returns:
        Its :class:`FilterInfo`.

    Raises:
        SmoothingError: If no such filter is registered.
    """
    try:
        return FILTERS[name]
    except KeyError as exc:
        raise SmoothingError(
            f"unknown smoothing filter {name!r}; expected one of {filter_names()}"
        ) from exc


def describe_filters() -> list[dict[str, Any]]:
    """Return the whole catalogue as JSON-safe schema dicts.

    This is what a UI's smoothing panel builds its controls from, so adding a
    filter to :data:`FILTERS` is enough to make it appear there.
    """
    return [FILTERS[name].to_dict() for name in filter_names()]


def apply_filter(
    name: str,
    values: np.ndarray,
    context: FilterContext,
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Run one filter over one curve, including the universal strength blend.

    Args:
        name: Registered filter name.
        values: ``[T]`` raw curve. Non-finite samples are linearly interpolated
            away first — a filter is a global operation and one NaN would
            otherwise poison the whole curve.
        context: Frame rate, reference frame and channel.
        params: Filter parameters; missing ones fall back to their defaults.

    Returns:
        ``[T]`` float64 smoothed curve.

    Raises:
        SmoothingError: If the filter or a parameter is unknown or invalid.
    """
    info = get_filter(name)
    settings = info.coerce(params)
    raw = fill_gaps(np.asarray(values, dtype=np.float64))
    if raw.shape[0] == 0:
        return raw
    strength = float(settings["strength"])
    if strength == 0.0:
        return raw
    smoothed = np.asarray(info.fn(raw, context, settings), dtype=np.float64)
    if smoothed.shape != raw.shape:  # pragma: no cover - guards third-party filters
        raise SmoothingError(
            f"filter {name!r} returned {smoothed.shape} for a {raw.shape} curve"
        )
    if strength >= 1.0:
        return smoothed
    return raw + strength * (smoothed - raw)


def fill_gaps(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with every non-finite sample linearly interpolated away.

    A filter is a global operation, so one ``NaN`` would otherwise poison the
    whole curve. Shared with :mod:`stable.smoothing.edit` so sub-range blending
    and keyframe pinning compare against exactly the curve the filters saw.
    """
    finite = np.isfinite(values)
    if finite.all():
        return values.astype(np.float64, copy=True)
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(values.shape[0], dtype=np.float64)
    filled = values.astype(np.float64, copy=True)
    filled[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return filled
