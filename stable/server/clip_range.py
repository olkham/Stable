"""The clip's working range: the In and Out markers (todo 1).

"It's not always the full length of the video we want to track." Everything in
Stable defaulted to the whole clip — :func:`stable.server.tracking.run_point_tracking`
hardcoded ``(0, frame_count)`` and the export rendered every frame — so a
thirty-second shot inside a ten-minute take cost ten minutes of tracking and ten
minutes of encoding to get at.

An In/Out pair is one piece of state that answers both. It bounds:

* **tracking** — runs cover the range and leave the rest untouched (``NaN``), so
  the operator's time goes where the shot is;
* **export** — the render writes exactly the frames between the markers.

Where it is kept
----------------
In ``clip_range.json``, a sidecar in the project directory beside
``solve.json`` / ``smoothing.json`` / ``render_options.json`` /
``tracker_options.json``. The locked v1 project schema stays untouched: this is
a per-project working preference, not part of the document a migration would
have to carry forward.

Conventions
-----------
``in`` is inclusive and ``out`` is **exclusive**, matching every other range in
this codebase (``frame_range``, ``bounded_frame_range``, Python slicing). A
missing sidecar, or one that has been cleared, means the whole clip — so nothing
that never sets a range behaves any differently than before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.errors import ProjectError

__all__ = ["CLIP_RANGE_CACHE", "ClipRange", "load_clip_range", "save_clip_range"]

CLIP_RANGE_CACHE = "clip_range.json"
"""File name of a project's In/Out sidecar."""


@dataclass(frozen=True)
class ClipRange:
    """The frames the operator is working on.

    Args:
        start: First frame, inclusive.
        end: One past the last frame, exclusive.

    Raises:
        ProjectError: If the range is empty or starts before zero.
    """

    start: int = 0
    end: int = 0

    def __post_init__(self) -> None:
        """Validate the range (dataclass hook)."""
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        if self.start < 0:
            raise ProjectError(f"the in point must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ProjectError(
                f"the out point must be after the in point, got in={self.start} out={self.end}"
            )

    @property
    def frames(self) -> int:
        """How many frames the range covers."""
        return self.end - self.start

    def as_tuple(self) -> tuple[int, int]:
        """Return the ``(start, end)`` pair every tracker API takes."""
        return (self.start, self.end)

    def contains(self, frame: int) -> bool:
        """Whether ``frame`` is inside the range."""
        return self.start <= int(frame) < self.end

    def clamped(self, frame_count: int) -> ClipRange:
        """Return this range fitted inside a clip of ``frame_count`` frames.

        A range can outlive the clip it was set on (a project re-pointed at a
        shorter file), and a run planned outside the clip is an error the
        operator did not make. Clamping keeps at least one frame.
        """
        total = max(1, int(frame_count))
        start = min(max(0, self.start), total - 1)
        end = min(max(start + 1, self.end), total)
        return ClipRange(start, end)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"in": self.start, "out": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, frame_count: int) -> ClipRange | None:
        """Rebuild from :meth:`to_dict` output, or ``None`` for the whole clip."""
        if not data:
            return None
        if data.get("in") is None and data.get("out") is None:
            return None
        raw_out = data.get("out")
        try:
            start = int(data.get("in") or 0)
            end = int(frame_count if raw_out is None else raw_out)
        except (TypeError, ValueError):  # pragma: no cover - hand-edited sidecar
            return None
        if end <= start:
            return None
        return ClipRange(start, end).clamped(frame_count)


def range_path(project_dir: Path) -> Path:
    """Return where a project's In/Out sidecar lives."""
    return Path(project_dir) / CLIP_RANGE_CACHE


def load_clip_range(project_dir: Path, frame_count: int) -> ClipRange | None:
    """Return the project's working range, or ``None`` for the whole clip.

    Args:
        project_dir: The project's directory.
        frame_count: The clip's length, used to clamp a stale range.

    Returns:
        The range, or ``None`` when none is set — which every caller must read
        as "the whole clip", so a project that never sets one is unaffected.
    """
    path = range_path(project_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar only
        return None
    if not isinstance(payload, dict):  # pragma: no cover - hand-edited sidecar
        return None
    return ClipRange.from_dict(payload, frame_count)


def save_clip_range(
    project_dir: Path, clip_range: ClipRange | None, frame_count: int
) -> ClipRange | None:
    """Persist the working range, or clear it.

    Args:
        project_dir: The project's directory.
        clip_range: The range to store, or ``None`` to go back to the whole clip.
        frame_count: The clip's length; the range is clamped into it.

    Returns:
        What is now in force (``None`` for the whole clip).
    """
    path = range_path(project_dir)
    if clip_range is None:
        path.unlink(missing_ok=True)
        return None
    fitted = clip_range.clamped(frame_count)
    path.write_text(json.dumps(fitted.to_dict(), indent=2), encoding="utf-8")
    return fitted


def effective_range(project_dir: Path, frame_count: int) -> tuple[int, int]:
    """Return the ``(start, end)`` frames to work on — the range, or the clip."""
    stored = load_clip_range(project_dir, frame_count)
    return stored.as_tuple() if stored else (0, int(frame_count))
