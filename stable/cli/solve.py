"""The ``stable solve`` subcommand: tracks in, X/Y/R/S curves out.

Headless entry point for::

    stable solve tracks.json --out curves.csv --video clip.mp4
    stable solve tracks.json --out curves.json --smooth butterworth:cutoff_hz=1.0

The exports named in ``PLAN.md``'s data-model section — CSV and JSON of the
solved (and optionally smoothed) ``X, Y, R, S`` curves — are exactly what this
writes, so a user can take Stable's solve into another application without going
anywhere near the stabilised render.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..core.errors import StableError
from .motion import (
    add_motion_options,
    channel_report,
    cli_progress,
    finish_progress,
    load_tracks,
    print_filters,
    run_motion,
    summarise_solve,
)

__all__ = ["add_solve_parser", "run_solve"]


def add_solve_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``solve`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "solve",
        help="solve X/Y/rotation/scale motion from tracks and export the curves",
        description=(
            "Fit a per-frame 2D similarity to tracked points and write the solved "
            "(and optionally smoothed) X, Y, R, S curves as CSV or JSON."
        ),
    )
    parser.add_argument("tracks", help="tracks file written by `stable track` (.json or .npz)")
    parser.add_argument(
        "--out",
        required=True,
        help="output path; '.csv' writes CSV, anything else JSON",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="the source clip, used for the default pivot and frame rate",
    )
    add_motion_options(parser)
    parser.add_argument("--quiet", action="store_true", help="suppress progress and the summary")
    return parser


def _clip_geometry(path: str | None) -> tuple[tuple[int, int] | None, float | None]:
    """Return ``(frame_size, fps)`` from a clip, or ``(None, None)`` without one."""
    if path is None:
        return None, None
    from ..video.source import PyAVVideoSource

    video = PyAVVideoSource(path)
    try:
        return (int(video.width), int(video.height)), float(video.fps)
    finally:
        video.close()


def run_solve(args: argparse.Namespace) -> int:
    """Execute the ``solve`` subcommand.

    Args:
        args: Parsed arguments from :func:`add_solve_parser`.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a recognised failure.
    """
    if args.list_filters:
        return print_filters()
    try:
        tracks = load_tracks(args.tracks)
        frame_size, clip_fps = _clip_geometry(args.video)
        fps = float(args.fps) if args.fps is not None else (clip_fps or 25.0)
        solve, smoothed = run_motion(
            tracks,
            args,
            frame_size=frame_size,
            fps=fps,
            default_smoothing=None,
            progress=cli_progress(args.quiet, "solving"),
        )
        finish_progress(args.quiet)
        # With nothing to smooth the two curve sets are identical, so writing the
        # plain solve keeps the file free of redundant columns.
        written = (
            smoothed.save(args.out)
            if smoothed.settings.active_channels
            else solve.save(args.out)
        )
    except StableError as exc:
        finish_progress(args.quiet)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(summarise_solve(solve))
        if smoothed.settings.active_channels:
            print(f"smoothing: {channel_report(smoothed.settings)}  (at {fps:g} fps)")
    print(f"wrote {written}")
    return 0
