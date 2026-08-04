"""The one bundled sample asset: a tiny synthetic ~5s clip, shipped in the package.

PLAN.md's Phase 6 acceptance criteria ask for two things that are best served
by exactly the same file rather than two separate ones: a "bundled ~5s test
clip" for CI's headless integration test, and a "sample project" a new user
can run the pipeline against immediately after installing. Both point here.

The clip (``data/sample_clip.mp4``) is synthetic — a textured background moved
by a known handheld-camera path, the same technique
``tests/conftest.py``'s ``WarpedClipSource`` uses for the Phase 3 acceptance
tests — never real footage, and it carries no model weights (the deep
trackers' checkpoints are downloaded separately; see
:mod:`stable.trackers.weights`). ``data/sample_clip.json`` records three of
its feature points' pixel coordinates at frame 0, chosen to be well separated
so :func:`stable.solver.solve_motion` gets a genuine X/Y/R/S solve (not just
X/Y) straight away.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["sample_clip_path", "sample_metadata", "sample_points"]


def sample_clip_path() -> Path:
    """Return the filesystem path to the bundled sample clip.

    Uses :mod:`importlib.resources` (not a ``__file__``-relative path) so this
    works the same whether Stable is running from a source checkout or an
    installed wheel.
    """
    return resources.files(__name__).joinpath("data", "sample_clip.mp4")  # type: ignore[return-value]


def sample_metadata() -> dict[str, Any]:
    """Return the bundled clip's metadata: fps, size, frame count and feature points."""
    text = resources.files(__name__).joinpath("data", "sample_clip.json").read_text("utf-8")
    return dict(json.loads(text))


def sample_points() -> list[tuple[float, float]]:
    """Return three well-separated, genuinely trackable ``(x, y)`` points at frame 0.

    Used by ``stable sample`` to place anchors without guessing, and by the
    headless integration test as a real (not hand-picked-on-the-spot) input.
    """
    return [(float(x), float(y)) for x, y in sample_metadata()["points"]]
