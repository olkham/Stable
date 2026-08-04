"""The ``stable sample`` subcommand: a ready-to-run example project.

Headless entry point for::

    stable sample
    stable sample ./my-sample --tracker lk --border auto_zoom

Copies the bundled ~5s synthetic clip (:mod:`stable.examples`), tracks its
three known-good feature points with a classic tracker (no GPU or PyTorch
needed), solves the motion, and renders a stabilized output — so there is
something to look at right after installing, without recording or finding a
real clip first. It is exactly the ``track`` -> ``solve`` -> ``stabilize``
pipeline those three subcommands run individually — this wires them together
against the bundled clip's known points as one command, by calling
:func:`stable.cli.main.main` the same way a shell script chaining the three
would.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from ..examples import sample_clip_path, sample_points

__all__ = ["add_sample_parser", "run_sample"]


def add_sample_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``sample`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "sample",
        help="write a ready-to-run example project from the bundled sample clip",
        description=(
            "Copy the bundled ~5s sample clip into a new directory, track its three "
            "known-good feature points, solve, and render a stabilized output -- "
            "everything `stable track`/`solve`/`stabilize` do, run as one command."
        ),
    )
    parser.add_argument(
        "out", nargs="?", default="stable-sample", help="output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--tracker", default="lk", help="tracker plugin to use (default: %(default)s)"
    )
    parser.add_argument(
        "--border",
        default="auto_zoom",
        help="border policy for the stabilized render (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress and summaries")
    return parser


def run_sample(args: argparse.Namespace) -> int:
    """Execute the ``sample`` subcommand: track, solve, and stabilize the bundled clip.

    Args:
        args: Parsed arguments from :func:`add_sample_parser`.

    Returns:
        Process exit code: ``0`` on success, or the first failing step's code.
    """
    from .main import main as run_cli  # noqa: PLC0415 - avoids a main<->sample import cycle

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_path = out_dir / "clip.mp4"
    shutil.copyfile(sample_clip_path(), clip_path)

    tracks_path = out_dir / "tracks.json"
    track_argv = ["track", str(clip_path), "--tracker", args.tracker, "--out", str(tracks_path)]
    for x, y in sample_points():
        track_argv += ["--point", f"{x:.2f},{y:.2f}@0"]
    if args.quiet:
        track_argv.append("--quiet")
    rc = run_cli(track_argv)
    if rc != 0:
        return rc

    curves_path = out_dir / "curves.csv"
    solve_argv = ["solve", str(tracks_path), "--out", str(curves_path), "--video", str(clip_path)]
    if args.quiet:
        solve_argv.append("--quiet")
    rc = run_cli(solve_argv)
    if rc != 0:
        return rc

    stabilized_path = out_dir / "stabilized.mp4"
    stabilize_argv = [
        "stabilize",
        str(clip_path),
        "--tracks",
        str(tracks_path),
        "--out",
        str(stabilized_path),
        "--border",
        args.border,
    ]
    if args.quiet:
        stabilize_argv.append("--quiet")
    rc = run_cli(stabilize_argv)
    if rc != 0:
        return rc

    if not args.quiet:
        print(f"\nsample project written to {out_dir}/")
        print(f"  {clip_path.name:<16} the bundled sample clip")
        print(f"  {tracks_path.name:<16} tracked points ('stable track' output)")
        print(f"  {curves_path.name:<16} solved X/Y/R/S curves ('stable solve' output)")
        print(f"  {stabilized_path.name:<16} stabilized render ('stable stabilize' output)")
        print("\nOpen the videos directly, or run `stable serve` and upload clip.mp4 there.")
    return 0
