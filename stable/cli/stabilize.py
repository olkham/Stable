"""The ``stable stabilize`` subcommand: tracks in, stabilised video out.

Headless entry point for::

    stable stabilize clip.mp4 --tracks tracks.json --out stable.mp4
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 --border crop \
        --smooth butterworth:cutoff_hz=0.8 --strength 0.9
    stable stabilize clip.mp4 --tracks tracks.json --out frames/ --border transparent
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 \
        --target point:2 --crop 0.85 --crop-centre target --fill blur --feather 8

It is the whole Phase 3 pipeline in one command — solve, smooth, plan the border
policy, warp, encode — and the audio of the source clip is passed through to the
output untouched unless ``--no-audio`` says otherwise.

The ``--target`` / ``--crop`` / ``--fill`` group is todo 17's: pin a tracked
position instead of (or as well as) filtering the curves, choose how much of the
frame survives, and decide what goes in the edge that is exposed. They are the
same options the export dialog offers, so a headless render and a browser one
produce the same file.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from ..core.errors import StableError
from ..core.project import BORDER_POLICIES
from ..render.fill import FILL_MODES, BorderFill
from ..render.options import CROP_CENTRES, CropSpec, average_positions
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


def _parse_target(spec: str) -> tuple[str, int | None]:
    """Parse a ``--target`` argument: ``frame``, ``centroid`` or ``point:INDEX``.

    The CLI addresses points by their index in the tracks file (the server
    addresses them by id), because a tracks file has no ids to offer.
    """
    mode, _, tail = spec.partition(":")
    mode = mode.strip().lower()
    if mode in ("frame", "none", "off"):
        return ("frame", None)
    if mode in ("centroid", "average", "all"):
        return ("centroid", None)
    if mode == "point":
        try:
            return ("point", int(tail))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --target {spec!r}: 'point' needs an index, e.g. 'point:0'"
            ) from exc
    raise argparse.ArgumentTypeError(
        f"invalid --target {spec!r}: expected 'frame', 'centroid' or 'point:INDEX'"
    )


def _parse_colour(spec: str) -> tuple[int, int, int]:
    """Parse a ``--fill-color`` argument: ``R,G,B`` or ``#rrggbb``."""
    text = spec.strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise argparse.ArgumentTypeError(f"invalid colour {spec!r}: expected '#rrggbb'")
        try:
            return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid colour {spec!r}") from exc
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"invalid colour {spec!r}: expected 'R,G,B' or '#rrggbb'")
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid colour {spec!r}: components must be 0-255"
                                         ) from exc
    if any(not 0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError(f"invalid colour {spec!r}: components must be 0-255")
    return (values[0], values[1], values[2])


def _target_positions(tracks: Any, mode: str, index: int | None) -> np.ndarray | None:
    """Resolve ``--target`` against a loaded tracks file.

    Returns:
        A ``[T, 2]`` path in source pixels, or ``None`` for ``frame`` mode.

    Raises:
        StableError: If the point index is out of range.
    """
    if mode == "frame":
        return None
    n_points = int(tracks.n_points)
    if mode == "point":
        if index is None or not 0 <= index < n_points:
            raise StableError(
                f"--target point:{index} is out of range; the tracks file holds "
                f"{n_points} point(s), so valid indices are 0..{n_points - 1}"
            )
        return np.asarray(tracks.point(index).points, dtype=np.float64)
    return average_positions(
        [np.asarray(tracks.point(i).points, dtype=np.float64) for i in range(n_points)]
    )


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

    target_group = parser.add_argument_group("stabilisation target, crop and border fill")
    target_group.add_argument(
        "--target",
        type=_parse_target,
        default=("frame", None),
        metavar="SPEC",
        help=(
            "what to hold still: 'frame' (only undo what the smoothing removed, the default), "
            "'centroid' (the average of every tracked point), or 'point:INDEX' (one point of "
            "the tracks file). A target pins that position to its anchor frame, so it "
            "stabilises even with no smoothing filters at all"
        ),
    )
    target_group.add_argument(
        "--target-frame",
        type=int,
        default=None,
        metavar="N",
        help="frame whose target position is held (default: the solve's reference frame)",
    )
    target_group.add_argument(
        "--crop",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "fraction of the frame to keep, in (0, 1]; the default computes the minimal crop "
            "that leaves no gaps. A larger value keeps more of the frame and exposes edges "
            "for --fill to handle"
        ),
    )
    target_group.add_argument(
        "--crop-centre",
        choices=CROP_CENTRES,
        default="frame",
        help="centre the output on the frame or on the stabilisation target (default: %(default)s)",
    )
    target_group.add_argument(
        "--fill",
        choices=FILL_MODES,
        default="solid",
        help=(
            "what goes in the exposed edge: a solid colour, the edge pixels stretched or "
            "mirrored outwards, or a blurred version of the frame (default: %(default)s)"
        ),
    )
    target_group.add_argument(
        "--fill-color",
        type=_parse_colour,
        default=(0, 0, 0),
        metavar="R,G,B",
        help="colour for --fill solid, and the colour --fill-tint blends toward",
    )
    target_group.add_argument(
        "--fill-blur",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian blur applied to the fill, in pixels; combines with any --fill mode",
    )
    target_group.add_argument(
        "--fill-tint",
        type=float,
        default=0.0,
        metavar="F",
        help="blend the fill toward --fill-color by this fraction (0-1); combines with any mode",
    )
    target_group.add_argument(
        "--feather",
        type=float,
        default=0.0,
        metavar="PX",
        help="soften the join between the picture and the fill over this many pixels",
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

        target_mode, target_index = args.target
        plan = StabilizationPlan.build(
            smoothed,
            source_size=(video.width, video.height),
            border=args.border,
            max_zoom=args.max_zoom,
            interpolation=args.interpolation,
            target_positions=_target_positions(tracks, target_mode, target_index),
            anchor_frame=args.target_frame,
            crop=CropSpec(scale=args.crop, centre=args.crop_centre),
            fill=BorderFill(
                mode=args.fill,
                color=args.fill_color,
                blur=args.fill_blur,
                tint=args.fill_tint,
                feather=args.feather,
            ),
        )
        if not args.quiet:
            geometry = plan.geometry
            print(
                f"border: {geometry.border}  output {geometry.width}x{geometry.height}  "
                f"zoom {geometry.zoom:.4f}  (covers {geometry.coverage * 100:.1f}% of frame)"
            )
            if target_mode != "frame":
                held = "every point" if target_mode == "centroid" else f"point {target_index}"
                print(f"target: holding {held} still (anchor frame {args.target_frame or 0})")
            if geometry.clamped:
                print(
                    "warning: --max-zoom capped the zoom, so some frames will show gaps",
                    file=sys.stderr,
                )
            if geometry.gaps:
                print(
                    f"note: the output keeps {geometry.crop_scale * 100:.1f}% of the frame but "
                    f"only {geometry.coverage * 100:.1f}% is always covered, so the "
                    f"'{args.fill}' fill will be visible at the edges"
                )
            if plan.is_no_op:
                print(
                    "warning: nothing asked the picture to move (no smoothing filter is "
                    "enabled and no --target is set), so the output will be a re-encode of "
                    "the input",
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
