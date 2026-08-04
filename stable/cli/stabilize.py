"""The ``stable stabilize`` subcommand: tracks in, stabilised video out.

Headless entry point for::

    stable stabilize clip.mp4 --tracks tracks.json --out stable.mp4
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 --border crop \
        --smooth butterworth:cutoff_hz=0.8 --strength 0.9
    stable stabilize clip.mp4 --tracks tracks.json --out frames/ --border transparent

It is the whole Phase 3 pipeline in one command — solve, smooth, plan the border
policy, warp, encode — and the audio of the source clip is passed through to the
output untouched unless ``--no-audio`` says otherwise.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..core.errors import StableError
from ..core.project import BORDER_POLICIES
from ..render.warp import INTERPOLATIONS
from .motion import (
    DEFAULT_SMOOTHING,
    add_motion_options,
    channel_report,
    cli_progress,
    finish_progress,
    load_tracks,
    print_filters,
    run_motion,
    summarise_solve,
)

__all__ = ["add_stabilize_parser", "run_stabilize"]


def _parse_frame_range(spec: str) -> tuple[int, int]:
    """Parse a ``--frame-range`` argument of the form ``start:end``."""
    try:
        start_text, end_text = spec.split(":", 1)
        return (int(start_text), int(end_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --frame-range {spec!r}: expected 'start:end', e.g. '0:100'"
        ) from exc


def add_stabilize_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``stabilize`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "stabilize",
        help="solve, smooth and render a stabilised video from tracked points",
        description=(
            "Solve X/Y/rotation/scale motion from tracks, smooth the curves, and render "
            "the clip with the inverse warp applied. Audio is passed through untouched."
        ),
    )
    parser.add_argument("video", help="path to the source video file")
    parser.add_argument(
        "--tracks", required=True, help="tracks file written by `stable track` (.json or .npz)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "output video file, or a directory to write a numbered PNG sequence "
            "(the only form that can carry the 'transparent' border's alpha)"
        ),
    )
    add_motion_options(parser)

    render_group = parser.add_argument_group("render")
    render_group.add_argument(
        "--border",
        choices=BORDER_POLICIES,
        default="auto_zoom",
        help=(
            "how to handle the uncovered edges: 'crop' outputs the common area at its own "
            "size, 'auto_zoom' scales that area back to full size, 'transparent' keeps the "
            "full frame and leaves the gaps as alpha (default: %(default)s)"
        ),
    )
    render_group.add_argument(
        "--max-zoom",
        type=float,
        default=None,
        metavar="F",
        help="cap the auto-zoom factor; gaps may remain if the shot needed more",
    )
    render_group.add_argument(
        "--interpolation",
        choices=INTERPOLATIONS,
        default="linear",
        help="resampling kernel for the warp (default: %(default)s)",
    )
    render_group.add_argument(
        "--codec", default="h264", help="output codec (default: %(default)s)"
    )
    render_group.add_argument(
        "--crf", type=int, default=18, metavar="N", help="encoder quality; lower is better"
    )
    render_group.add_argument(
        "--no-audio", action="store_true", help="do not copy the source clip's audio track"
    )
    render_group.add_argument(
        "--frame-range",
        type=_parse_frame_range,
        default=None,
        metavar="start:end",
        help="half-open range of frames to render (default: the whole clip)",
    )
    render_group.add_argument(
        "--gpu", action="store_true", help="try OpenCV's CUDA warp, falling back to the CPU"
    )
    render_group.add_argument(
        "--curves",
        default=None,
        metavar="PATH",
        help="also write the raw and smoothed curves here ('.csv' or '.json')",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress and the summary")
    return parser


def run_stabilize(args: argparse.Namespace) -> int:
    """Execute the ``stabilize`` subcommand.

    Args:
        args: Parsed arguments from :func:`add_stabilize_parser`.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a recognised failure.
    """
    if args.list_filters:
        return print_filters()

    from ..render.stabilize import StabilizationPlan, render_stabilized
    from ..video.source import PyAVVideoSource

    video = None
    try:
        video = PyAVVideoSource(args.video)
        tracks = load_tracks(args.tracks)
        if tracks.n_frames != video.frame_count:
            raise StableError(
                f"{args.tracks!r} covers {tracks.n_frames} frames but {args.video!r} has "
                f"{video.frame_count}; the tracks were made from a different clip"
            )
        fps = float(args.fps) if args.fps is not None else float(video.fps)
        solve, smoothed = run_motion(
            tracks,
            args,
            frame_size=(video.width, video.height),
            fps=fps,
            default_smoothing=DEFAULT_SMOOTHING,
            progress=cli_progress(args.quiet, "solving"),
        )
        finish_progress(args.quiet)

        if not args.quiet:
            print(summarise_solve(solve))
            print(f"smoothing: {channel_report(smoothed.settings)}  (at {fps:g} fps)")

        plan = StabilizationPlan.build(
            smoothed,
            source_size=(video.width, video.height),
            border=args.border,
            max_zoom=args.max_zoom,
            interpolation=args.interpolation,
        )
        if not args.quiet:
            geometry = plan.geometry
            print(
                f"border: {geometry.border}  output {geometry.width}x{geometry.height}  "
                f"zoom {geometry.zoom:.4f}  (covers {geometry.coverage * 100:.1f}% of frame)"
            )
            if geometry.clamped:
                print(
                    "warning: --max-zoom capped the zoom, so some frames will show gaps",
                    file=sys.stderr,
                )

        written = render_stabilized(
            video,
            plan,
            args.out,
            fps=fps,
            codec=args.codec,
            crf=args.crf,
            audio_source=None if args.no_audio else args.video,
            frame_range=args.frame_range,
            use_gpu=args.gpu,
            progress=cli_progress(args.quiet, "rendering"),
        )
        finish_progress(args.quiet)
        if args.curves:
            print(f"wrote {smoothed.save(args.curves)}")
    except StableError as exc:
        finish_progress(args.quiet)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if video is not None:
            video.close()

    print(f"wrote {written}")
    return 0
