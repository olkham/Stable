"""The ``stable models`` subcommand: list, download, or remove checkpoints.

Headless entry point for::

    stable models
    stable models download bootstapir
    stable models download cotracker3_offline --accept-noncommercial
    stable models remove bootstapir

A thin CLI over :mod:`stable.trackers.weights` — the same UI the browser's
model manager gives (see :mod:`stable.server.routes.models`), for scripting
and for machines that never run the web UI at all. The same fail-closed rules
apply: an unpinned checkpoint needs ``--allow-unpinned`` (or
``STABLE_ALLOW_UNPINNED_WEIGHTS=1``), and a non-commercial one needs
``--accept-noncommercial`` (or ``STABLE_ACCEPT_NONCOMMERCIAL=1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ..core.errors import StableError
from ..trackers.registry import registry
from ..trackers.weights import clear_weights, describe_models, ensure_weights, registered_models

__all__ = ["add_models_parser", "run_models"]


def add_models_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``models`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "models",
        help="list, download, or remove tracker model checkpoints",
        description=(
            "Manage the model checkpoints deep-learning tracker plugins need, cached "
            "under ~/.stable/models (override with STABLE_MODEL_DIR)."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON (list only)"
    )
    actions = parser.add_subparsers(dest="models_action")

    actions.add_parser("list", help="list every registered checkpoint (default)")

    download = actions.add_parser("download", help="download and verify one checkpoint")
    download.add_argument("key", help="the checkpoint's key, from `stable models`")
    download.add_argument(
        "--accept-noncommercial",
        action="store_true",
        help="accept a non-commercial licence (e.g. CoTracker3's CC-BY-NC)",
    )
    download.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="download even without a verified checksum pinned",
    )

    remove = actions.add_parser("remove", help="delete a cached checkpoint")
    remove.add_argument("key", help="the checkpoint's key, from `stable models`")

    return parser


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print a one-line-per-checkpoint human-readable table."""
    if not rows:
        print("no installed tracker plugin registers a checkpoint")
        return
    width = max(len(row["key"]) for row in rows)
    for row in rows:
        flags = []
        if row["downloaded"]:
            flags.append("cached")
        if not row["pinned"]:
            flags.append("unpinned")
        if row["non_commercial"]:
            flags.append("non-commercial")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{row['key']:<{width}}  {row['license']}{suffix}")


def _cli_progress(fraction: float, message: str = "") -> None:
    """Print a single overwriting progress line to stderr."""
    bar = f"\r{message or 'downloading'}... {fraction * 100:5.1f}%"
    print(bar, end="", file=sys.stderr, flush=True)


def run_models(args: argparse.Namespace) -> int:
    """Execute the ``models`` subcommand.

    Args:
        args: Parsed arguments from :func:`add_models_parser`.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a recognised failure.
    """
    registry.discover()  # import every installed plugin so it has registered its checkpoint(s)
    action = getattr(args, "models_action", None) or "list"

    if action == "list":
        rows = describe_models()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            _print_table(rows)
        return 0

    by_key = {spec.key: spec for spec in registered_models()}
    spec = by_key.get(args.key)
    if spec is None:
        print(
            f"error: no registered model {args.key!r}; run `stable models` to list them",
            file=sys.stderr,
        )
        return 1

    if action == "download":
        try:
            path = ensure_weights(
                spec,
                progress=_cli_progress,
                accept_noncommercial=args.accept_noncommercial,
                allow_unpinned=args.allow_unpinned,
            )
        except StableError as exc:
            print(file=sys.stderr)
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(file=sys.stderr)
        print(f"{spec.key} ready at {path}")
        return 0

    # action == "remove"
    removed = clear_weights(spec)
    print(f"{spec.key} {'removed' if removed else 'was not cached'}")
    return 0
