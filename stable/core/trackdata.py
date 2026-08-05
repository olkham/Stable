"""Interchange formats for 2D track data.

CSV and JSON (see :mod:`stable.server.routes.export`) answer "give me the
numbers". They do not answer the other half of todo 18 — *use in other
applications* — because no compositor imports either. This module adds the
format that closes that gap, as pure functions over positions: no file I/O, no
project objects, so they are as usable from the CLI as from a route and can be
tested against the spec rather than against a round trip.

After Effects keyframe data
---------------------------
The tab-separated text After Effects reads from the clipboard or a ``.txt``
file, and pastes straight onto a layer's Position property. Deliberately chosen
over a compositor's node format: it is plain text with a published, stable
layout that can be verified by reading it, where a Nuke ``Tracker4`` node is an
intricate binary-ish knob serialisation that would have to be checked against
Nuke itself to be trusted.

Coordinates match: After Effects measures pixels from the top-left with Y
downward, exactly as this project does, so positions are written through
unchanged. Frames with no position are *skipped* rather than written as zero —
a gap in a track must not import as the point jumping to the top-left corner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

__all__ = [
    "AE_FORMAT",
    "after_effects_keyframes",
]

AE_FORMAT = "ae"
"""The ``format=`` value that selects After Effects keyframe data."""

_AE_HEADER = "Adobe After Effects 8.0 Keyframe Data"


def _finite(value: float) -> bool:
    """Whether a coordinate is a real position rather than a gap marker."""
    return value is not None and math.isfinite(value)


def after_effects_keyframes(
    tracks: Iterable[tuple[str, Sequence[tuple[float, float]]]],
    *,
    fps: float,
    width: int,
    height: int,
) -> str:
    """Render tracks as After Effects keyframe data.

    Args:
        tracks: ``(name, positions)`` pairs, where ``positions`` is indexed by
            frame and holds ``(x, y)`` in source pixels; ``NaN`` marks a frame
            the point was not found on.
        fps: Frame rate, written as "Units Per Second".
        width: Source width in pixels.
        height: Source height in pixels.

    Returns:
        The complete document, tab-separated with CRLF line endings — After
        Effects rejects LF-only keyframe data pasted on Windows, and accepts
        CRLF everywhere.

    Raises:
        ValueError: If ``fps`` is not positive; the units line is what makes the
            keyframe times mean anything, so a zero there is silently wrong.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    lines: list[str] = [_AE_HEADER, ""]
    # One "Transform / Position" block per point. After Effects applies each
    # block to whatever layer is selected when it is pasted, so the blocks are
    # separated and named for the operator to paste one at a time.
    for name, positions in tracks:
        lines.append(f"\tUnits Per Second\t{fps:g}")
        lines.append(f"\tSource Width\t{width}")
        lines.append(f"\tSource Height\t{height}")
        lines.append("\tSource Pixel Aspect Ratio\t1")
        lines.append("\tComp Pixel Aspect Ratio\t1")
        lines.append("")
        lines.append(f"Transform\tPosition\t{name}")
        lines.append("\tFrame\tX pixels\tY pixels\tZ pixels")
        for frame, position in enumerate(positions):
            x, y = position
            if not (_finite(x) and _finite(y)):
                continue
            lines.append(f"\t{frame}\t{x:.4f}\t{y:.4f}\t0")
        lines.append("")

    lines.append("End of Keyframe Data")
    lines.append("")
    return "\r\n".join(lines)
