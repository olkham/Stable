"""The ``stable track`` subcommand: run a tracker plugin from the command line.

Headless entry point for::

    stable track clip.mp4 --point 120.5,80@0 --tracker lk --out tracks.json

Every tracked point is independent (one query each) — this command calls a
:class:`~stable.trackers.base.TrackerPlugin` directly rather than going through
:mod:`stable.trackers.merge`, since multi-anchor merging is about combining
several runs *for the same point*, which is a project-file-level concept the
CLI does not (yet) expose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import numpy as np

from ..core.errors import ContractError, StableError
from ..core.types import Query, TrackResult
from ..trackers.registry import create_tracker, registry
from ..video.source import PyAVVideoSource

__all__ = [
    "add_track_parser",
    "run_track",
    "track_result_to_json_dict",
    "track_result_from_json_dict",
    "parse_point_spec",
]


def parse_point_spec(spec: str) -> Query:
    """Parse a ``--point`` argument of the form ``x,y@frame``.

    Args:
        spec: e.g. ``"120.5,80@0"`` — x, y and the anchor frame.

    Returns:
        The corresponding :class:`~stable.core.types.Query`.

    Raises:
        argparse.ArgumentTypeError: If ``spec`` is not in the expected form.
    """
    try:
        coords, frame_text = spec.split("@", 1)
        x_text, y_text = coords.split(",", 1)
        return Query(frame=int(frame_text), x=float(x_text), y=float(y_text))
    except (ValueError, ContractError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --point {spec!r}: expected 'x,y@frame', e.g. '120.5,80@0' ({exc})"
        ) from exc


def _parse_frame_range(spec: str) -> tuple[int, int]:
    """Parse a ``--frame-range`` argument of the form ``start:end``."""
    try:
        start_text, end_text = spec.split(":", 1)
        return (int(start_text), int(end_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --frame-range {spec!r}: expected 'start:end', e.g. '0:100'"
        ) from exc


def add_track_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``track`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "track",
        help="track one or more points through a video with a tracker plugin",
        description="Track points through a video end to end and write the result as JSON.",
    )
    parser.add_argument("video", help="path to the video file")
    parser.add_argument(
        "--point",
        dest="points",
        action="append",
        required=True,
        type=parse_point_spec,
        metavar="x,y@frame",
        help="a point to track, as 'x,y@frame'; repeat for multiple points",
    )
    parser.add_argument(
        "--tracker", default="lk", help="tracker plugin name to use (default: %(default)s)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output path; '.json' writes the CLI's JSON schema, '.npz' the raw TrackResult format",
    )
    parser.add_argument(
        "--frame-range",
        type=_parse_frame_range,
        default=None,
        metavar="start:end",
        help="half-open frame range to track over (default: the whole clip)",
    )
    parser.add_argument(
        "--direction",
        choices=("fwd", "bwd", "bidir"),
        default="bidir",
        help="tracking direction relative to each point's anchor frame (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default=None, help="compute device ('cpu', 'cuda', ...); default: automatic"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the progress line")
    return parser


def _points_row_to_json(row: Any) -> list[list[float] | None]:
    """Convert one point's ``[T, 2]`` position array to a JSON-safe list.

    ``NaN`` samples (frames the run did not cover, or lost after tracking)
    become ``null`` since JSON has no native NaN.
    """
    out: list[list[float] | None] = []
    for x, y in row.tolist():
        out.append([x, y] if math.isfinite(x) and math.isfinite(y) else None)
    return out


def _points_row_from_json(row: list[list[float] | None]) -> list[list[float]]:
    """Invert :func:`_points_row_to_json`, turning ``null`` back into ``NaN``."""
    return [
        [math.nan, math.nan] if cell is None else [float(cell[0]), float(cell[1])] for cell in row
    ]


def track_result_to_json_dict(
    result: TrackResult,
    *,
    tracker: str,
    video_path: str,
    frame_range: tuple[int, int],
    direction: str,
    queries: list[Query],
) -> dict[str, Any]:
    """Serialise a :class:`TrackResult` (plus its run metadata) to a JSON-safe dict.

    Args:
        result: The tracker's output.
        tracker: Name of the plugin that produced it.
        video_path: Path to the source video, recorded for provenance.
        frame_range: The ``(start, end)`` range that was requested.
        direction: The tracking direction that was requested.
        queries: The anchor query for each point, in the same order as ``result``.

    Returns:
        A ``json.dumps``-safe dict; see :func:`track_result_from_json_dict` for
        the inverse.
    """
    return {
        "version": 1,
        "tracker": tracker,
        "video": video_path,
        "frame_count": result.n_frames,
        "frame_range": list(frame_range),
        "direction": direction,
        "points": [
            {
                "query": query.to_dict(),
                "points": _points_row_to_json(result.points[i]),
                "confidence": result.confidence[i].tolist(),
                "visible": result.visible[i].tolist(),
            }
            for i, query in enumerate(queries)
        ],
    }


def track_result_from_json_dict(data: dict[str, Any]) -> TrackResult:
    """Rebuild a :class:`TrackResult` from :func:`track_result_to_json_dict` output.

    Args:
        data: A dict previously produced by :func:`track_result_to_json_dict`.

    Returns:
        The reconstructed :class:`TrackResult`; per-point queries are not part
        of the return value (read ``data["points"][i]["query"]`` for those).

    Raises:
        ContractError: If the arrays inside don't satisfy the contract.
    """
    points_entries = data["points"]
    if not points_entries:
        return TrackResult.empty(0, int(data["frame_count"]))
    return TrackResult(
        points=np.asarray([_points_row_from_json(entry["points"]) for entry in points_entries]),
        confidence=np.asarray([entry["confidence"] for entry in points_entries]),
        visible=np.asarray([entry["visible"] for entry in points_entries]),
    )


def _cli_progress(quiet: bool) -> Any:
    """Return a progress callback that prints a single overwriting line to stderr."""

    def _report(fraction: float, message: str = "") -> None:
        if quiet:
            return
        bar = f"\rtracking... {fraction * 100:5.1f}%"
        if message:
            bar += f"  {message}"
        print(bar, end="", file=sys.stderr, flush=True)

    return _report


def run_track(args: argparse.Namespace) -> int:
    """Execute the ``track`` subcommand.

    Args:
        args: Parsed arguments from the parser built by :func:`add_track_parser`.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a recognised failure.
    """
    try:
        video = PyAVVideoSource(args.video)
    except StableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frame_range = args.frame_range if args.frame_range is not None else (0, video.frame_count)

    if args.tracker not in registry.discover():
        print(
            f"error: no tracker named {args.tracker!r}; available: {registry.discover()}",
            file=sys.stderr,
        )
        return 1

    tracker = None
    try:
        tracker = create_tracker(args.tracker, device=args.device)
        result = tracker.track(
            video,
            args.points,
            frame_range,
            args.direction,
            progress=_cli_progress(args.quiet),
        )
    except StableError as exc:
        if not args.quiet:
            print(file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tracker is not None:
            tracker.release()
        video.close()

    if not args.quiet:
        print(file=sys.stderr)

    if args.out.endswith(".npz"):
        result.save_npz(args.out)
    else:
        payload = track_result_to_json_dict(
            result,
            tracker=args.tracker,
            video_path=args.video,
            frame_range=frame_range,
            direction=args.direction,
            queries=args.points,
        )
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    print(f"tracked {result.n_points} point(s) over {result.n_frames} frame(s) -> {args.out}")
    return 0
