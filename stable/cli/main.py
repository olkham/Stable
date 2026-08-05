"""Entry point for the ``stable`` console script."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .. import __version__
from ..core.device import device_info
from ..trackers.registry import describe_trackers
from .models import add_models_parser, run_models
from .motion import print_filters
from .sample import add_sample_parser, run_sample
from .serve import add_serve_parser, run_serve
from .solve import add_solve_parser, run_solve
from .stabilize import add_stabilize_parser, run_stabilize
from .track import add_track_parser, run_track

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        A parser with the Phase 0 subcommands attached. Later phases add their
        subcommands here.
    """
    parser = argparse.ArgumentParser(
        prog="stable",
        description="Open-source point tracking and 2D stabilization.",
    )
    parser.add_argument("--version", action="version", version=f"stable {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    trackers = subparsers.add_parser("trackers", help="list installed tracker plugins")
    trackers.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    devices = subparsers.add_parser("devices", help="show the compute devices Stable can use")
    devices.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    filters = subparsers.add_parser("filters", help="list the smoothing filters and their options")
    filters.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    add_track_parser(subparsers)
    add_solve_parser(subparsers)
    add_stabilize_parser(subparsers)
    add_models_parser(subparsers)
    add_serve_parser(subparsers)
    add_sample_parser(subparsers)
    return parser


def _cmd_trackers(as_json: bool) -> int:
    """Print every discoverable tracker plugin."""
    rows = describe_trackers()
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no trackers installed")
        return 1
    width = max(len(row["name"]) for row in rows)
    for row in rows:
        flags = []
        if row["requires_gpu"]:
            flags.append("gpu-only")
        if not row["available"]:
            flags.append("unavailable")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        print(f"{row['name']:<{width}}  {row['display_name']}  ({row['license']}){suffix}")
        # An unavailable tracker that does not say what is missing sends the
        # reader to the docs to guess; the plugin already knows the answer.
        if not row["available"] and row.get("unavailable_reason"):
            print(f"{'':<{width}}  -> {row['unavailable_reason']}")
    return 0


def _cmd_devices(as_json: bool) -> int:
    """Print the compute environment Stable sees."""
    info = device_info()
    if as_json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"selected: {info['selected']}")
    print(f"torch:    {info['torch'] or 'not installed'}")
    print(f"cuda:     {info['cuda']}")
    for name in info["cuda_devices"]:
        print(f"  - {name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "trackers":
        return _cmd_trackers(args.json)
    if args.command == "devices":
        return _cmd_devices(args.json)
    if args.command == "filters":
        return print_filters(args.json)
    if args.command == "track":
        return run_track(args)
    if args.command == "solve":
        return run_solve(args)
    if args.command == "stabilize":
        return run_stabilize(args)
    if args.command == "models":
        return run_models(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "sample":
        return run_sample(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script shim
    raise SystemExit(main())
