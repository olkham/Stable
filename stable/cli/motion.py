"""Shared plumbing for the ``solve`` and ``stabilize`` subcommands.

Both commands take the same road — load tracks, solve the motion, smooth the
curves — and differ only in what they do at the end, so the argument group, the
parsing of filter specifications and the solve/smooth pipeline live here once.

Filter specification syntax
---------------------------
A filter is written ``name`` or ``name:key=value,key=value``::

    --smooth butterworth:cutoff_hz=1.0,order=3
    --smooth-channel r:gaussian:sigma=12
    --smooth-channel s:tripod

Parameter names and ranges come straight from the filter catalogue
(:func:`stable.smoothing.describe_filters`), so ``stable solve --list-filters``
and the error message for a typo can never drift out of step with the filters
themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ..core.errors import StableError
from ..core.geometry import CHANNELS
from ..core.types import TrackResult
from ..smoothing import (
    ChannelSmoothing,
    SmoothedMotion,
    SmoothingSettings,
    describe_filters,
    get_filter,
    smooth_motion,
)
from ..solver import MotionSolve, RansacConfig, SolveConfig, solve_motion

__all__ = [
    "DEFAULT_SMOOTHING",
    "load_tracks",
    "parse_filter_spec",
    "parse_channel_filter_spec",
    "parse_channels",
    "add_motion_options",
    "build_solve_config",
    "build_smoothing_settings",
    "run_motion",
    "print_filters",
    "summarise_solve",
    "channel_report",
    "cli_progress",
    "finish_progress",
]

DEFAULT_SMOOTHING = "butterworth:cutoff_hz=1.0"
"""What ``stable stabilize`` smooths with when the user does not say."""


# -- argument parsing --------------------------------------------------------


def parse_filter_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse ``name`` or ``name:key=value,key=value`` into a filter and its params.

    Args:
        spec: The text form, e.g. ``"butterworth:cutoff_hz=1.0,order=3"``.

    Returns:
        ``(filter_name, params)`` with the parameters already validated against
        the filter's schema.

    Raises:
        argparse.ArgumentTypeError: If the syntax is wrong, the filter is
            unknown, or a parameter is unknown or out of range.
    """
    name, _, tail = spec.partition(":")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"invalid filter {spec!r}: no filter name")
    params: dict[str, Any] = {}
    for chunk in (piece for piece in tail.split(",") if piece.strip()):
        key, sep, value = chunk.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(
                f"invalid filter parameter {chunk!r} in {spec!r}: expected 'key=value'"
            )
        params[key.strip()] = value.strip()
    try:
        get_filter(name).coerce(params)
    except StableError as exc:
        raise argparse.ArgumentTypeError(f"invalid filter {spec!r}: {exc}") from exc
    return name, params


def parse_channel_filter_spec(spec: str) -> tuple[str, str, dict[str, Any]]:
    """Parse ``channel:name[:key=value,...]`` for ``--smooth-channel``.

    Args:
        spec: e.g. ``"r:gaussian:sigma=12"``.

    Returns:
        ``(channel, filter_name, params)``.

    Raises:
        argparse.ArgumentTypeError: If the channel or the filter part is invalid.
    """
    channel, sep, rest = spec.partition(":")
    channel = channel.strip()
    if not sep or channel not in CHANNELS:
        raise argparse.ArgumentTypeError(
            f"invalid --smooth-channel {spec!r}: expected 'channel:filter[:params]' "
            f"with channel in {list(CHANNELS)}"
        )
    name, params = parse_filter_spec(rest)
    return channel, name, params


def parse_channels(spec: str) -> list[str]:
    """Parse a channel set written as letters (``"xyr"``) or a list (``"x,y,r"``).

    Args:
        spec: The text form; ``"none"`` selects nothing.

    Returns:
        The channel names, in canonical order.

    Raises:
        argparse.ArgumentTypeError: If a letter is not a known channel.
    """
    text = spec.strip().lower()
    if text in ("none", ""):
        return []
    pieces = [p.strip() for p in text.split(",")] if "," in text else list(text)
    unknown = sorted({p for p in pieces if p not in CHANNELS})
    if unknown:
        raise argparse.ArgumentTypeError(
            f"invalid channels {spec!r}: {unknown} are not channels; expected {list(CHANNELS)}"
        )
    return [c for c in CHANNELS if c in set(pieces)]


def _parse_pivot(spec: str) -> tuple[float, float]:
    """Parse a ``--pivot`` argument of the form ``x,y``."""
    try:
        x_text, y_text = spec.split(",", 1)
        return (float(x_text), float(y_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --pivot {spec!r}: expected 'x,y', e.g. '960,540'"
        ) from exc


def add_motion_options(parser: argparse.ArgumentParser) -> None:
    """Attach the solve and smoothing options shared by both subcommands.

    Args:
        parser: The subcommand parser to extend.
    """
    solve_group = parser.add_argument_group("solve")
    solve_group.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        metavar="N",
        help="frame the solve is relative to (default: %(default)s)",
    )
    solve_group.add_argument(
        "--solve-channels",
        type=parse_channels,
        default=list(CHANNELS),
        metavar="xyrs",
        help="degrees of freedom the solve may use (default: xyrs)",
    )
    solve_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.05,
        metavar="F",
        help="ignore track samples below this confidence (default: %(default)s)",
    )
    solve_group.add_argument(
        "--ransac-threshold",
        type=float,
        default=2.0,
        metavar="PX",
        help="inlier distance for RANSAC outlier rejection (default: %(default)s)",
    )
    solve_group.add_argument(
        "--max-residual",
        type=float,
        default=3.0,
        metavar="PX",
        help="flag solved frames whose residual exceeds this (default: %(default)s)",
    )
    solve_group.add_argument(
        "--pivot",
        type=_parse_pivot,
        default=None,
        metavar="x,y",
        help="point the X/Y curves are measured about (default: the frame centre)",
    )

    smooth_group = parser.add_argument_group("smoothing")
    smooth_group.add_argument(
        "--smooth",
        action="append",
        type=parse_filter_spec,
        default=None,
        metavar="FILTER[:k=v,...]",
        help="smoothing filter applied to every enabled channel; repeat to chain filters",
    )
    smooth_group.add_argument(
        "--smooth-channel",
        action="append",
        type=parse_channel_filter_spec,
        default=None,
        metavar="CH:FILTER[:k=v,...]",
        help="smoothing filter for one channel only; overrides --smooth for that channel",
    )
    smooth_group.add_argument(
        "--channels",
        type=parse_channels,
        default=list(CHANNELS),
        metavar="xyrs",
        help="channels to smooth, and therefore to stabilise (default: xyrs)",
    )
    smooth_group.add_argument(
        "--strength",
        type=float,
        default=None,
        metavar="F",
        help="how much of the smoothing to apply, 0 to 1 (default: 1)",
    )
    smooth_group.add_argument(
        "--tripod",
        action="store_true",
        help="lock the shot to the reference frame (shorthand for --smooth tripod)",
    )
    smooth_group.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="F",
        help="frame rate the frequency-domain filters use (default: the clip's)",
    )
    smooth_group.add_argument(
        "--list-filters",
        action="store_true",
        help="print the smoothing filter catalogue and exit",
    )


# -- loading -----------------------------------------------------------------


def load_tracks(path: str | os.PathLike[str]) -> TrackResult:
    """Load a :class:`~stable.core.types.TrackResult` from ``.json`` or ``.npz``.

    Args:
        path: A file written by ``stable track`` (either format).

    Returns:
        The tracks.

    Raises:
        StableError: If the file cannot be read or does not hold tracks.
    """
    target = os.fspath(path)
    if target.lower().endswith(".npz"):
        return TrackResult.load_npz(target)
    from .track import track_result_from_json_dict

    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise StableError(f"cannot read tracks {target!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StableError(f"{target!r} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "points" not in data:
        raise StableError(
            f"{target!r} does not look like a tracks file written by `stable track`"
        )
    return track_result_from_json_dict(data)


# -- pipeline ----------------------------------------------------------------


def build_solve_config(args: argparse.Namespace) -> SolveConfig:
    """Build a :class:`~stable.solver.SolveConfig` from parsed arguments."""
    enabled = set(args.solve_channels)
    return SolveConfig(
        reference_frame=int(args.reference_frame),
        channels={channel: channel in enabled for channel in CHANNELS},
        pivot=args.pivot,
        min_confidence=float(args.min_confidence),
        max_residual_px=float(args.max_residual),
        ransac=RansacConfig(threshold_px=float(args.ransac_threshold)),
    )


def build_smoothing_settings(
    args: argparse.Namespace, *, default: str | None = None
) -> SmoothingSettings:
    """Build a :class:`~stable.smoothing.SmoothingSettings` from parsed arguments.

    Args:
        args: Parsed arguments carrying ``smooth``, ``smooth_channel``,
            ``channels``, ``strength`` and ``tripod``.
        default: Filter specification to use when the user asked for smoothing
            implicitly (``stable stabilize`` without any ``--smooth``). ``None``
            means "smooth nothing", which is what ``stable solve`` wants.

    Returns:
        The settings.

    Raises:
        StableError: If a filter or parameter is invalid.
    """
    wanted = list(args.channels)
    settings = SmoothingSettings()

    global_filters: list[tuple[str, dict[str, Any]]] = list(args.smooth or [])
    if args.tripod:
        global_filters.append(("tripod", {}))
    if not global_filters and not (args.smooth_channel or []) and default is not None:
        global_filters.append(parse_filter_spec(default))

    for channel in wanted:
        config = ChannelSmoothing(channel=channel, enabled=True)
        for name, params in global_filters:
            config.add(name, params)
        settings.channels[channel] = config

    # Per-channel filters replace the global chain for the channels they name,
    # so `--smooth gaussian --smooth-channel r:tripod` reads the way it looks.
    for channel, name, params in args.smooth_channel or []:
        config = ChannelSmoothing(channel=channel, enabled=True)
        config.add(name, params)
        settings.channels[channel] = config

    for channel in CHANNELS:
        if channel not in wanted and not any(
            entry[0] == channel for entry in (args.smooth_channel or [])
        ):
            settings.channels[channel] = ChannelSmoothing(channel=channel, enabled=False)

    if args.strength is not None:
        settings.set_strength(float(args.strength), channels=settings.active_channels)
    return settings


def run_motion(
    tracks: TrackResult,
    args: argparse.Namespace,
    *,
    frame_size: tuple[int, int] | None,
    fps: float,
    default_smoothing: str | None,
    progress: Any = None,
) -> tuple[MotionSolve, SmoothedMotion]:
    """Solve and smooth, the shared middle of both subcommands.

    Args:
        tracks: The loaded tracks.
        args: Parsed arguments.
        frame_size: ``(width, height)`` of the clip, for the default pivot.
        fps: Frame rate for the frequency-domain filters.
        default_smoothing: Filter specification to fall back on; see
            :func:`build_smoothing_settings`.
        progress: Optional progress reporter passed to the solver.

    Returns:
        ``(solve, smoothed)``.

    Raises:
        StableError: If the solve or the smoothing configuration is invalid.
    """
    from ..trackers.base import noop_progress

    solve = solve_motion(
        tracks,
        config=build_solve_config(args),
        frame_size=frame_size,
        progress=progress or noop_progress,
    )
    settings = build_smoothing_settings(args, default=default_smoothing)
    return solve, smooth_motion(solve, settings, fps=fps)


# -- reporting ---------------------------------------------------------------


def print_filters(as_json: bool = False) -> int:
    """Print the smoothing filter catalogue.

    Args:
        as_json: Emit the machine-readable schema instead of a table.

    Returns:
        Always ``0``.
    """
    catalogue = describe_filters()
    if as_json:
        print(json.dumps(catalogue, indent=2))
        return 0
    for entry in catalogue:
        print(f"{entry['name']}  -  {entry['display_name']}")
        print(f"    {entry['description']}")
        for param in entry["params"]:
            bounds = []
            if param["minimum"] is not None:
                bounds.append(f">= {param['minimum']:g}")
            if param["maximum"] is not None:
                bounds.append(f"<= {param['maximum']:g}")
            limits = f" ({', '.join(bounds)})" if bounds else ""
            unit = f" {param['unit']}" if param["unit"] else ""
            print(
                f"    - {param['name']}: {param['kind']}{unit}, "
                f"default {param['default']:g}{limits}"
            )
        print()
    return 0


def summarise_solve(solve: MotionSolve) -> str:
    """Return a one-paragraph human summary of a solve's quality."""
    summary = solve.summary()
    counts = summary["status_counts"]
    degraded = ", ".join(
        f"{name}={count}" for name, count in counts.items() if name != "solved" and count
    )
    lines = [
        f"solved {counts['solved']}/{summary['frames']} frames "
        f"(reference {summary['reference_frame']}, pivot "
        f"{summary['pivot'][0]:.1f},{summary['pivot'][1]:.1f})",
        f"residual: mean {summary['mean_residual_px']:.3f} px, "
        f"worst {summary['max_residual_px']:.3f} px",
    ]
    if degraded:
        lines.append(f"degraded frames: {degraded}")
    if summary["flagged"]:
        lines.append(f"{summary['flagged']} frame(s) flagged for review")
    return "\n".join(lines)


def cli_progress(quiet: bool, label: str = "working") -> Any:
    """Return a progress callback printing one overwriting line to stderr.

    Args:
        quiet: Suppress output entirely.
        label: Prefix shown before the percentage.

    Returns:
        A :class:`~stable.trackers.base.ProgressCallback`.
    """

    def _report(fraction: float, message: str = "") -> None:
        if quiet:
            return
        line = f"\r{label}... {fraction * 100:5.1f}%"
        if message:
            line += f"  {message}"
        print(line, end="", file=sys.stderr, flush=True)

    return _report


def finish_progress(quiet: bool) -> None:
    """End the overwriting progress line with a newline, unless quiet."""
    if not quiet:
        print(file=sys.stderr)


def channel_report(settings: SmoothingSettings) -> str:
    """Return a one-line description of which channels are being stabilised."""
    parts = []
    for channel in CHANNELS:
        config = settings.channels[channel]
        if not config.active:
            parts.append(f"{channel}=off")
            continue
        chain = "+".join(spec.filter for spec in config.filters if spec.enabled)
        parts.append(f"{channel}={chain}@{config.strength:g}")
    return "  ".join(parts)
