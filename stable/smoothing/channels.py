"""Per-channel smoothing configuration — enable, strength and a filter chain.

The project file (schema v1, locked in Phase 0) stores smoothing as a flat,
ordered list of :class:`~stable.core.project.SmoothingSpec` entries, each naming
one channel, one filter and its parameters. That is the wire format;
:class:`SmoothingSettings` is the in-memory view of it, grouped the way a user
thinks about the problem: *per channel*, with an enable and a strength.

The two forms round-trip exactly. A channel's strength lives in each of its
specs' ``params["strength"]`` — the universal parameter every filter accepts —
so nothing here invents a key the locked schema does not already carry.

What "channel enabled" means downstream
---------------------------------------
A disabled channel is not smoothed, so its smoothed curve *is* its raw curve,
so the stabilisation warp ``smoothed(t) . raw(t)^-1`` contributes nothing on
that axis. "Don't stabilise rotation" and "don't smooth rotation" are the same
statement, and there is deliberately only one switch for it.

Curve-editor directives
-----------------------
Phase 5's curve editor adds two per-channel settings that are not filters: the
sub-range a chain applies to, and the frames the smoothed curve must pass
through (see :mod:`stable.smoothing.edit`). They still have to survive a round
trip through the locked v1 schema, whose only smoothing container is that flat
``{channel, filter, params, enabled}`` list — so they are written as entries
whose ``filter`` is a reserved, ``@``-prefixed name (:data:`RANGE_DIRECTIVE`,
:data:`PIN_DIRECTIVE`). The ``@`` cannot collide with a registered filter, the
file stays readable, and no key was invented.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import SmoothingError
from ..core.geometry import CHANNELS
from ..core.project import SmoothingSpec
from .edit import DEFAULT_PIN_RADIUS, DEFAULT_RANGE_BLEND, blend_subrange, pin_curve
from .filters import FilterContext, apply_filter, get_filter

__all__ = [
    "ChannelSmoothing",
    "SmoothingSettings",
    "smooth_curve",
    "RANGE_DIRECTIVE",
    "PIN_DIRECTIVE",
    "DIRECTIVES",
]

RANGE_DIRECTIVE = "@range"
"""Reserved ``SmoothingSpec.filter`` marking a sub-range, not a filter."""

PIN_DIRECTIVE = "@pin"
"""Reserved ``SmoothingSpec.filter`` marking one pinned keyframe."""

DIRECTIVES: tuple[str, ...] = (RANGE_DIRECTIVE, PIN_DIRECTIVE)
"""Every reserved name :meth:`SmoothingSettings.from_specs` treats as a directive."""


def smooth_curve(
    values: np.ndarray,
    specs: Sequence[SmoothingSpec],
    context: FilterContext,
) -> np.ndarray:
    """Apply a chain of filters to one curve, in order.

    Args:
        values: ``[T]`` raw curve.
        specs: Filters to apply; disabled entries are skipped. Each one sees the
            output of the previous, so a chain composes (a Butterworth followed
            by a light moving average, say).
        context: Frame rate, reference frame and channel.

    Returns:
        ``[T]`` float64 smoothed curve. An empty chain returns a copy of the
        input, which is what makes "no smoothing" and "no stabilisation" the
        same thing.

    Raises:
        SmoothingError: If a filter name or parameter is invalid.
    """
    curve = np.asarray(values, dtype=np.float64).copy()
    for spec in specs:
        if not spec.enabled:
            continue
        curve = apply_filter(spec.filter, curve, context, spec.params)
    return curve


@dataclass
class ChannelSmoothing:
    """How one of ``x``, ``y``, ``r``, ``s`` is smoothed.

    Args:
        channel: The channel this configures.
        enabled: Master switch. When False the channel passes through raw and is
            therefore not stabilised at all.
        filters: Ordered filter chain for this channel.
        sub_range: Inclusive ``(start, end)`` the chain applies to, or ``None``
            for the whole clip. See :func:`~stable.smoothing.edit.blend_subrange`.
        range_blend: Frames of cross-fade at each edge of ``sub_range``.
        pins: Frames the smoothed curve must pass through exactly. See
            :func:`~stable.smoothing.edit.pin_curve`.
        pin_radius: How far a pin's influence reaches, in frames.

    Raises:
        SmoothingError: If ``channel`` is not a known channel name, a spec
            targets a different one, or a range/pin setting is out of range.
    """

    channel: str
    enabled: bool = True
    filters: list[SmoothingSpec] = field(default_factory=list)
    sub_range: tuple[int, int] | None = None
    range_blend: int = DEFAULT_RANGE_BLEND
    pins: list[int] = field(default_factory=list)
    pin_radius: int = DEFAULT_PIN_RADIUS

    def __post_init__(self) -> None:
        """Validate the channel, its specs and its curve-editor settings."""
        if self.channel not in CHANNELS:
            raise SmoothingError(f"unknown channel {self.channel!r}; expected {CHANNELS}")
        for spec in self.filters:
            if spec.channel != self.channel:
                raise SmoothingError(
                    f"filter {spec.filter!r} targets channel {spec.channel!r}, "
                    f"not {self.channel!r}"
                )
            get_filter(spec.filter).coerce(spec.params)
        if self.sub_range is not None:
            start, end = int(self.sub_range[0]), int(self.sub_range[1])
            if start < 0 or end < start:
                raise SmoothingError(
                    f"invalid smoothing range [{start}, {end}] on channel {self.channel!r}"
                )
            self.sub_range = (start, end)
        if self.range_blend < 0:
            raise SmoothingError(f"range_blend must be >= 0, got {self.range_blend}")
        if self.pin_radius <= 0:
            raise SmoothingError(f"pin_radius must be positive, got {self.pin_radius}")
        self.pins = sorted({int(frame) for frame in self.pins if int(frame) >= 0})

    @property
    def active(self) -> bool:
        """Whether this channel will actually change: enabled with a live filter."""
        return self.enabled and any(spec.enabled for spec in self.filters)

    @property
    def strength(self) -> float:
        """Strength of the chain's last enabled filter; ``0.0`` when inactive.

        Setting it writes the value to every filter in the chain, which is what
        a single per-channel strength slider means.
        """
        if not self.active:
            return 0.0
        for spec in reversed(self.filters):
            if spec.enabled:
                return float(spec.params.get("strength", 1.0))
        return 0.0  # pragma: no cover - `active` already guarantees one enabled spec

    @strength.setter
    def strength(self, value: float) -> None:
        """Set the strength of every filter in this channel's chain."""
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise SmoothingError(f"strength must lie in [0, 1], got {number}")
        for spec in self.filters:
            spec.params["strength"] = number

    def add(self, filter_name: str, params: Mapping[str, Any] | None = None) -> SmoothingSpec:
        """Append a filter to this channel's chain.

        Args:
            filter_name: A registered filter name.
            params: Its parameters; validated immediately so a typo fails here
                rather than mid-render.

        Returns:
            The appended :class:`~stable.core.project.SmoothingSpec`.

        Raises:
            SmoothingError: If the filter or a parameter is unknown or invalid.
        """
        get_filter(filter_name).coerce(params)
        spec = SmoothingSpec(channel=self.channel, filter=filter_name, params=dict(params or {}))
        self.filters.append(spec)
        return spec

    def apply(self, values: np.ndarray, context: FilterContext) -> np.ndarray:
        """Smooth one curve with this channel's chain, sub-range and pins.

        The order is deliberate: filter first, then restrict the result to the
        sub-range, then re-seat it on the pins. Pinning last is what makes "the
        smoothed curve passes through this frame" a guarantee — anything applied
        after it could move the curve off the pin again.

        Args:
            values: ``[T]`` raw curve.
            context: Frame rate and reference frame; its ``channel`` is replaced
                with this channel's name.

        Returns:
            ``[T]`` float64 curve — a copy of the input when the channel is
            disabled or has no live filters.
        """
        raw = np.asarray(values, dtype=np.float64)
        if not self.enabled:
            return raw.copy()
        local = FilterContext(
            fps=context.fps,
            reference_frame=context.reference_frame,
            channel=self.channel,
            n_frames=context.n_frames,
        )
        smoothed = smooth_curve(raw, self.filters, local)
        if self.sub_range is not None:
            smoothed = blend_subrange(
                raw, smoothed, self.sub_range[0], self.sub_range[1], blend=self.range_blend
            )
        if self.pins:
            smoothed = pin_curve(raw, smoothed, self.pins, radius=self.pin_radius)
        return smoothed

    def to_specs(self) -> list[SmoothingSpec]:
        """Return the project-file specs for this channel.

        A disabled channel emits its filters with ``enabled=False``, so the
        user's chain survives a round trip through the switch. The sub-range and
        pins follow as :data:`DIRECTIVES` entries (see the module docstring).
        """
        specs = [
            SmoothingSpec(
                channel=spec.channel,
                filter=spec.filter,
                params=dict(spec.params),
                enabled=bool(spec.enabled and self.enabled),
            )
            for spec in self.filters
        ]
        if self.sub_range is not None:
            specs.append(
                SmoothingSpec(
                    channel=self.channel,
                    filter=RANGE_DIRECTIVE,
                    params={
                        "start": int(self.sub_range[0]),
                        "end": int(self.sub_range[1]),
                        "blend": int(self.range_blend),
                    },
                )
            )
        specs.extend(
            SmoothingSpec(
                channel=self.channel,
                filter=PIN_DIRECTIVE,
                params={"frame": int(frame), "radius": int(self.pin_radius)},
            )
            for frame in self.pins
        )
        return specs


@dataclass
class SmoothingSettings:
    """The whole smoothing configuration: one :class:`ChannelSmoothing` each.

    Args:
        channels: Mapping from channel name to its configuration. Missing
            channels are filled in as empty (i.e. not smoothed).
    """

    channels: dict[str, ChannelSmoothing] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in any missing channel with an empty configuration."""
        unknown = sorted(set(self.channels) - set(CHANNELS))
        if unknown:
            raise SmoothingError(f"unknown channel(s) {unknown}; expected {CHANNELS}")
        for channel in CHANNELS:
            self.channels.setdefault(channel, ChannelSmoothing(channel=channel, enabled=False))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_specs(cls, specs: Iterable[SmoothingSpec]) -> SmoothingSettings:
        """Build settings from a project file's flat ``smoothing`` list.

        Args:
            specs: The project's :class:`~stable.core.project.SmoothingSpec`
                entries, in order. Entries whose ``filter`` is one of
                :data:`DIRECTIVES` configure the channel's sub-range or pins
                instead of joining its filter chain.

        Returns:
            The grouped :class:`SmoothingSettings`. A channel with no specs — or
            with only disabled ones — comes back disabled.

        Raises:
            SmoothingError: If a directive's parameters are missing or invalid.
        """
        grouped: dict[str, ChannelSmoothing] = {}
        for spec in specs:
            channel = grouped.setdefault(
                spec.channel, ChannelSmoothing(channel=spec.channel, enabled=False)
            )
            if spec.filter in DIRECTIVES:
                if spec.enabled:
                    _apply_directive(channel, spec)
                continue
            channel.filters.append(
                SmoothingSpec(
                    channel=spec.channel,
                    filter=spec.filter,
                    params=dict(spec.params),
                    enabled=spec.enabled,
                )
            )
            channel.enabled = channel.enabled or spec.enabled
        return cls(channels=grouped)

    @classmethod
    def uniform(
        cls,
        filter_name: str,
        params: Mapping[str, Any] | None = None,
        *,
        channels: Sequence[str] = CHANNELS,
    ) -> SmoothingSettings:
        """Apply one filter with one set of parameters to several channels.

        Args:
            filter_name: A registered filter name.
            params: Its parameters.
            channels: Which channels to configure; the rest stay disabled.

        Returns:
            The corresponding :class:`SmoothingSettings`.

        Raises:
            SmoothingError: If a channel name, the filter, or a parameter is
                invalid.
        """
        unknown = sorted(set(channels) - set(CHANNELS))
        if unknown:
            raise SmoothingError(f"unknown channel(s) {unknown}; expected {CHANNELS}")
        settings = cls()
        for channel in channels:
            settings.channels[channel] = ChannelSmoothing(channel=channel, enabled=True)
            settings.channels[channel].add(filter_name, params)
        return settings

    @classmethod
    def tripod(
        cls, *, channels: Sequence[str] = CHANNELS, strength: float = 1.0
    ) -> SmoothingSettings:
        """Return a locked-off ("tripod") configuration.

        Args:
            channels: Channels to lock; anything left out keeps moving.
            strength: How hard to lock, in ``[0, 1]``.

        Returns:
            Settings whose smoothed motion is the identity on every locked
            channel, so the stabilisation warp becomes ``raw(t)^-1``.
        """
        return cls.uniform("tripod", {"strength": strength}, channels=channels)

    # -- access ------------------------------------------------------------

    def channel(self, name: str) -> ChannelSmoothing:
        """Return the configuration for ``name``.

        Raises:
            SmoothingError: If ``name`` is not a known channel.
        """
        if name not in self.channels:
            raise SmoothingError(f"unknown channel {name!r}; expected {CHANNELS}")
        return self.channels[name]

    @property
    def active_channels(self) -> list[str]:
        """Channels that will actually be smoothed, in canonical order."""
        return [c for c in CHANNELS if self.channels[c].active]

    def set_enabled(self, channel: str, enabled: bool) -> None:
        """Turn one channel's smoothing (and therefore stabilisation) on or off."""
        self.channel(channel).enabled = bool(enabled)

    def set_strength(self, value: float, *, channels: Sequence[str] | None = None) -> None:
        """Set the strength of every filter on the given channels.

        Args:
            value: Strength in ``[0, 1]``.
            channels: Channels to change; ``None`` means all of them.
        """
        for name in channels if channels is not None else CHANNELS:
            self.channel(name).strength = value

    def to_specs(self) -> list[SmoothingSpec]:
        """Return the flat, ordered spec list a project file stores."""
        specs: list[SmoothingSpec] = []
        for channel in CHANNELS:
            specs.extend(self.channels[channel].to_specs())
        return specs

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary: per channel, its enable, strength and chain."""
        return {channel: self._channel_dict(channel) for channel in CHANNELS}

    def _channel_dict(self, channel: str) -> dict[str, Any]:
        """Return one channel's JSON-safe summary, including its editor settings."""
        settings = self.channels[channel]
        span = settings.sub_range
        return {
            "enabled": settings.enabled,
            "active": settings.active,
            "strength": settings.strength,
            "filters": [spec.to_dict() for spec in settings.filters],
            "range": None if span is None else [span[0], span[1]],
            "range_blend": settings.range_blend,
            "pins": list(settings.pins),
            "pin_radius": settings.pin_radius,
        }


def _int_param(spec: SmoothingSpec, name: str, default: int | None = None) -> int:
    """Read one integer parameter off a directive spec.

    Raises:
        SmoothingError: If it is absent with no default, or is not a number.
    """
    if name not in spec.params:
        if default is None:
            raise SmoothingError(
                f"{spec.filter!r} directive on channel {spec.channel!r} needs a {name!r} parameter"
            )
        return default
    try:
        return int(spec.params[name])
    except (TypeError, ValueError) as exc:
        raise SmoothingError(
            f"{spec.filter!r} directive parameter {name!r} must be an integer, "
            f"got {spec.params[name]!r}"
        ) from exc


def _apply_directive(channel: ChannelSmoothing, spec: SmoothingSpec) -> None:
    """Fold one :data:`DIRECTIVES` spec into a channel's curve-editor settings.

    Raises:
        SmoothingError: If the directive's parameters are missing or invalid.
    """
    if spec.filter == RANGE_DIRECTIVE:
        start = _int_param(spec, "start")
        end = _int_param(spec, "end")
        if start < 0 or end < start:
            raise SmoothingError(
                f"invalid smoothing range [{start}, {end}] on channel {spec.channel!r}"
            )
        blend = _int_param(spec, "blend", DEFAULT_RANGE_BLEND)
        if blend < 0:
            raise SmoothingError(f"range blend must be >= 0, got {blend}")
        channel.sub_range = (start, end)
        channel.range_blend = blend
        return
    frame = _int_param(spec, "frame")
    if frame < 0:
        raise SmoothingError(f"pinned frame must be >= 0, got {frame}")
    radius = _int_param(spec, "radius", channel.pin_radius)
    if radius <= 0:
        raise SmoothingError(f"pin radius must be positive, got {radius}")
    channel.pin_radius = radius
    if frame not in channel.pins:
        channel.pins = sorted([*channel.pins, frame])
