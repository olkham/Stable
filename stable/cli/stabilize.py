"""The ``stable stabilize`` subcommand: tracks in, stabilised video out.

Headless entry point for::

    stable stabilize clip.mp4 --tracks tracks.json --out stable.mp4
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 --border crop \
        --smooth butterworth:cutoff_hz=0.8 --strength 0.9
    stable stabilize clip.mp4 --tracks tracks.json --out frames/ --border transparent
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 \
        --target track:2 --target-at top-left --crop 0.85 --width 1920 --fill blur
    stable stabilize clip.mp4 --tracks t.json --out s.mp4 \
        --frame-range 900:1600 --crop-to-range

It is the whole Phase 3 pipeline in one command — solve, smooth, plan the border
policy, warp, encode — and the audio of the source clip is passed through to the
output untouched unless ``--no-audio`` says otherwise.

The ``--target`` / ``--target-at`` / ``--crop`` / ``--fill`` group is the export
screen's, headless: name the path to hold, say where in the *output* frame it
goes, choose how much of the frame survives, and decide what goes in the edge
that is exposed. They are the same options the export panel offers, so a
headless render and a browser one produce the same file.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from ..core.errors import StableError
from ..core.project import BORDER_POLICIES
from ..render.fill import FILL_MODES, BorderFill
from ..render.options import CropSpec, OutputResolution, average_positions
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


TARGET_SPECS = ("none", "track:INDEX", "smoothed:INDEX", "average")
"""What ``--target`` accepts, for the help text and the error message."""


def _parse_target(spec: str) -> tuple[str, int | None]:
    """Parse a ``--target`` argument.

    ``none`` (or ``frame``/``off``) undoes only what the smoothing removed —
    still reachable here, unlike in the UI, because a headless render of a
    smoothing-only stabilisation is a perfectly good thing to ask for.
    ``track:INDEX`` nails one point, ``smoothed:INDEX`` frames it and lets the
    smoothing decide the rest, and ``average`` nails the mean of every point.

    The CLI addresses points by their index in the tracks file (the server
    addresses them by id), because a tracks file has no ids to offer.
    """
    mode, _, tail = spec.partition(":")
    mode = mode.strip().lower()
    if mode in ("none", "frame", "off"):
        return ("none", None)
    if mode in ("average", "centroid", "all"):
        return ("average", None)
    if mode in ("track", "point", "smoothed", "smoothed_track"):
        canonical = "smoothed_track" if mode.startswith("smoothed") else "track"
        try:
            return (canonical, int(tail))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --target {spec!r}: {mode!r} needs a point index, e.g. '{mode}:0'"
            ) from exc
    raise argparse.ArgumentTypeError(
        f"invalid --target {spec!r}: expected one of {', '.join(TARGET_SPECS)}"
    )


def _parse_placement(spec: str) -> tuple[float, float]:
    """Parse a ``--target-at`` argument: ``U,V`` fractions, or a named position.

    Named positions are the ones an operator actually reaches for — the centre
    and the four thirds intersections — spelled the way they would say them.
    """
    text = spec.strip().lower()
    named = {
        "centre": (0.5, 0.5),
        "center": (0.5, 0.5),
        "top-left": (1 / 3, 1 / 3),
        "top-right": (2 / 3, 1 / 3),
        "bottom-left": (1 / 3, 2 / 3),
        "bottom-right": (2 / 3, 2 / 3),
    }
    if text in named:
        return named[text]
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid --target-at {spec!r}: expected 'U,V' fractions or one of "
            f"{', '.join(sorted(named))}"
        )
    try:
        u, v = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --target-at {spec!r}: U and V must be numbers"
        ) from exc
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(
            f"invalid --target-at {spec!r}: U and V are fractions of the output frame, so both "
            "must be in [0, 1]"
        )
    return (u, v)


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


def _target_positions(
    tracks: Any, mode: str, index: int | None, motion: Any = None
) -> np.ndarray | None:
    """Resolve ``--target`` against a loaded tracks file.

    Args:
        tracks: The loaded tracks file.
        mode: The parsed ``--target`` mode.
        index: The point index, for the per-point modes.
        motion: The smoothed motion — what turns a raw path into the smoothed
            estimate ``smoothed:INDEX`` holds, exactly as the browser's dashed
            trail is produced.

    Returns:
        A ``[T, 2]`` path in source pixels, or ``None`` for ``none``.

    Raises:
        StableError: If the point index is out of range.
    """
    if mode == "none":
        return None
    n_points = int(tracks.n_points)
    if mode in ("track", "smoothed_track"):
        if index is None or not 0 <= index < n_points:
            raise StableError(
                f"--target {mode}:{index} is out of range; the tracks file holds "
                f"{n_points} point(s), so valid indices are 0..{n_points - 1}"
            )
        path = np.asarray(tracks.point(index).points, dtype=np.float64)
    else:
        path = average_positions(
            [np.asarray(tracks.point(i).points, dtype=np.float64) for i in range(n_points)]
        )
    if mode != "smoothed_track":
        return path
    warps = motion.stabilization_transforms()
    smoothed = np.full_like(path, np.nan)
    for frame in range(min(len(warps), path.shape[0])):
        if np.isfinite(path[frame]).all():
            smoothed[frame] = warps[frame].apply(path[frame])
    return smoothed


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
        default=("none", None),
        metavar="SPEC",
        help=(
            "which path to hold: 'none' (only undo what the smoothing removed, the default), "
            "'track:INDEX' (nail one point of the tracks file to --target-at on every frame), "
            "'smoothed:INDEX' (put it there at --target-frame and let the smoothing decide the "
            "rest), or 'average' (nail the mean of every tracked point). A nailed target "
            "stabilises even with no smoothing filters at all"
        ),
    )
    target_group.add_argument(
        "--target-at",
        type=_parse_placement,
        default=(0.5, 0.5),
        metavar="U,V",
        help=(
            "where the target sits in the *output* frame, as fractions of it, or one of "
            "centre / top-left / top-right / bottom-left / bottom-right (the thirds "
            "intersections). Default: centre"
        ),
    )
    target_group.add_argument(
        "--target-frame",
        type=int,
        default=None,
        metavar="N",
        help=(
            "frame whose target position sets the framing (default: the solve's reference "
            "frame). Only changes the result for 'smoothed:INDEX' — a nailed target agrees "
            "on every frame by construction"
        ),
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
        "--width",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "deliver the crop window at this width; the height follows unless --height is "
            "given too, in which case the crop takes that aspect ratio"
        ),
    )
    target_group.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="PX",
        help="deliver the crop window at this height (see --width)",
    )
    target_group.add_argument(
        "--crop-to-range",
        action="store_true",
        help=(
            "measure the automatic crop over --frame-range only, so frames that are not being "
            "rendered stop dictating how much of the ones that are survives"
        ),
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
        placement = tuple(args.target_at)
        held = _target_positions(tracks, target_mode, target_index, smoothed)
        plan = StabilizationPlan.build(
            smoothed,
            source_size=(video.width, video.height),
            border=args.border,
            max_zoom=args.max_zoom,
            interpolation=args.interpolation,
            sources=None if held is None else held[None, :, :],
            placements=None if held is None else np.asarray([placement], dtype=np.float64),
            crop=CropSpec(scale=args.crop, limit_to_range=args.crop_to_range),
            fill=BorderFill(
                mode=args.fill,
                color=args.fill_color,
                blur=args.fill_blur,
                tint=args.fill_tint,
                feather=args.feather,
            ),
            coverage_range=args.frame_range if args.crop_to_range else None,
            resolution=OutputResolution(width=args.width, height=args.height),
        )
        if not args.quiet:
            geometry = plan.geometry
            measured = (
                f", measured over {geometry.measured_frames} frames" if args.crop_to_range else ""
            )
            print(
                f"border: {geometry.border}  output {geometry.width}x{geometry.height}  "
                f"zoom {geometry.zoom:.4f}  (covers {geometry.coverage * 100:.1f}% of frame"
                f"{measured})"
            )
            if target_mode != "none":
                what = "every point" if target_mode == "average" else f"point {target_index}"
                if target_mode == "smoothed_track":
                    what = f"the smoothed estimate of {what}"
                at = f"({placement[0] * 100:.0f}%, {placement[1] * 100:.0f}%) of the output"
                print(f"target: holding {what} at {at}")
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
