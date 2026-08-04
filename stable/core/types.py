"""The data contracts every tracker, solver and UI component speaks.

Three types matter here:

* :class:`Query` — a user-placed anchor keyframe: "at frame ``f`` this point is
  truly at ``(x, y)``". A point owns a *list* of these (see the multi-anchor
  section of ``PLAN.md``).
* :class:`TrackResult` — the output of one tracker run for ``N`` points, the
  single type everything downstream sees. This is what makes trackers swappable.
* :class:`PointTrack` — one point's slice of a :class:`TrackResult`. Per-point
  storage (``tracks/p1.npz``), corrections and multi-anchor merging all work at
  this granularity.

Timeline convention (fixed)
---------------------------
Array axis ``T`` is the **absolute frame index** of the clip, not an offset into
the tracked range: ``points[n, 12]`` is always frame 12 of the video. Frames a
run did not cover are ``NaN`` / confidence ``0`` / ``visible False``. This costs
a few kilobytes per point and removes every off-by-one between plugins, solver
and UI.

Immutability
------------
"Raw tracker output is immutable" is a hard requirement of the project (manual
edits are a non-destructive correction layer). Both containers copy their input
arrays and mark them read-only, so an accidental in-place write raises
``ValueError`` instead of silently corrupting a track.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ContractError

__all__ = ["Query", "TrackResult", "PointTrack"]


def _freeze(array: np.ndarray) -> np.ndarray:
    """Return a read-only copy of ``array`` (defensive copy, then lock)."""
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Query:
    """A user-placed anchor keyframe: a position known to be true.

    A point's ``queries`` list may hold several of these across the clip;
    tracking runs bidirectionally from each one and the runs are merged by
    :mod:`stable.trackers.merge`.

    Args:
        frame: Absolute frame index, ``>= 0``.
        x: Horizontal position in pixels (sub-pixel allowed).
        y: Vertical position in pixels (sub-pixel allowed).

    Raises:
        ContractError: If ``frame`` is negative or a coordinate is non-finite.
    """

    frame: int
    x: float
    y: float

    def __post_init__(self) -> None:
        """Normalise types and validate (dataclass hook)."""
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        if self.frame < 0:
            raise ContractError(f"Query.frame must be >= 0, got {self.frame}")
        if not (np.isfinite(self.x) and np.isfinite(self.y)):
            raise ContractError(f"Query position must be finite, got ({self.x}, {self.y})")

    @property
    def position(self) -> tuple[float, float]:
        """The ``(x, y)`` position as a tuple."""
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file form: ``{"frame": int, "x": float, "y": float}``."""
        return {"frame": self.frame, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Query:
        """Rebuild a query from :meth:`to_dict` output.

        Raises:
            ContractError: If a required key is missing.
        """
        try:
            return cls(frame=data["frame"], x=data["x"], y=data["y"])
        except KeyError as exc:
            raise ContractError(f"query is missing key {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class TrackResult:
    """The output of one tracker run — the universal downstream contract.

    Args:
        points: ``[N, T, 2]`` float32 positions in pixels; ``NaN`` where the
            position is unknown (untracked frame, lost point).
        confidence: ``[N, T]`` float32 in ``[0, 1]``, normalised per plugin so
            the solver can weight every tracker the same way. ``0`` means "no
            information"; ``NaN`` is not allowed.
        visible: ``[N, T]`` bool — False when the plugin believes the point is
            occluded or out of frame.

    Raises:
        ContractError: On any shape, dtype or range violation.
    """

    points: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray

    def __post_init__(self) -> None:
        """Coerce dtypes, validate the contract and freeze the arrays."""
        points = np.asarray(self.points, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        visible = np.asarray(self.visible, dtype=bool)

        if points.ndim != 3 or points.shape[2] != 2:
            raise ContractError(f"points must have shape [N, T, 2], got {points.shape}")
        n, t = points.shape[0], points.shape[1]
        if confidence.shape != (n, t):
            raise ContractError(
                f"confidence must have shape [N, T] = {(n, t)}, got {confidence.shape}"
            )
        if visible.shape != (n, t):
            raise ContractError(f"visible must have shape [N, T] = {(n, t)}, got {visible.shape}")
        if confidence.size and not np.isfinite(confidence).all():
            raise ContractError("confidence must be finite (no NaN/inf)")
        if confidence.size and (confidence.min() < 0.0 or confidence.max() > 1.0):
            raise ContractError(
                "confidence must lie in [0, 1]; plugins normalise their own scores"
            )
        if points.size and np.isinf(points).any():
            raise ContractError("points must be finite or NaN (no inf)")

        object.__setattr__(self, "points", _freeze(points))
        object.__setattr__(self, "confidence", _freeze(confidence))
        object.__setattr__(self, "visible", _freeze(visible))

    # -- shape helpers ---------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of tracked points ``N``."""
        return int(self.points.shape[0])

    @property
    def n_frames(self) -> int:
        """Length of the clip timeline ``T`` (absolute frame indices ``0..T-1``)."""
        return int(self.points.shape[1])

    def __len__(self) -> int:
        """Number of tracked points (same as :attr:`n_points`)."""
        return self.n_points

    @classmethod
    def empty(cls, n_points: int, n_frames: int) -> TrackResult:
        """Return an all-invalid result of the given size.

        Args:
            n_points: Number of points ``N``.
            n_frames: Clip length ``T``.

        Returns:
            A result full of ``NaN`` positions, zero confidence and
            ``visible=False`` — the correct starting point for a tracker that
            fills in the frames it covers.
        """
        if n_points < 0 or n_frames < 0:
            raise ContractError("n_points and n_frames must be >= 0")
        return cls(
            points=np.full((n_points, n_frames, 2), np.nan, dtype=np.float32),
            confidence=np.zeros((n_points, n_frames), dtype=np.float32),
            visible=np.zeros((n_points, n_frames), dtype=bool),
        )

    @classmethod
    def from_tracks(cls, tracks: list[PointTrack]) -> TrackResult:
        """Stack per-point tracks into a single result.

        Args:
            tracks: One :class:`PointTrack` per point; all must share ``T``.

        Returns:
            The stacked :class:`TrackResult`.

        Raises:
            ContractError: If the list is empty or lengths disagree.
        """
        if not tracks:
            raise ContractError("cannot stack an empty list of tracks; use TrackResult.empty")
        lengths = {t.n_frames for t in tracks}
        if len(lengths) != 1:
            raise ContractError(f"all tracks must share the same T, got {sorted(lengths)}")
        return cls(
            points=np.stack([t.points for t in tracks]),
            confidence=np.stack([t.confidence for t in tracks]),
            visible=np.stack([t.visible for t in tracks]),
        )

    def point(self, index: int) -> PointTrack:
        """Return point ``index`` as a standalone :class:`PointTrack`.

        Args:
            index: Point index in ``[0, N)``; negative indices count from the end.

        Returns:
            A :class:`PointTrack` copy of that point's arrays.
        """
        if not -self.n_points <= index < self.n_points:
            raise IndexError(f"point index {index} out of range for N={self.n_points}")
        return PointTrack(
            points=self.points[index],
            confidence=self.confidence[index],
            visible=self.visible[index],
        )

    def tracks(self) -> list[PointTrack]:
        """Return every point as a list of :class:`PointTrack`."""
        return [self.point(i) for i in range(self.n_points)]

    def select(self, indices: list[int]) -> TrackResult:
        """Return a result containing only the given point indices, in order."""
        return TrackResult(
            points=self.points[indices],
            confidence=self.confidence[indices],
            visible=self.visible[indices],
        )

    # -- persistence -----------------------------------------------------

    def save_npz(self, path: str | os.PathLike[str]) -> None:
        """Write the result to a compressed ``.npz`` file.

        This is the on-disk format referenced by ``points[].raw`` in the
        project file.

        Args:
            path: Destination file; parent directories are created.
        """
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points,
            confidence=self.confidence,
            visible=self.visible,
        )

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> TrackResult:
        """Read a result written by :meth:`save_npz`.

        Raises:
            ContractError: If the file lacks a required array.
        """
        with np.load(os.fspath(path)) as data:
            missing = {"points", "confidence", "visible"} - set(data.files)
            if missing:
                raise ContractError(f"{path} is missing array(s): {sorted(missing)}")
            return cls(
                points=data["points"],
                confidence=data["confidence"],
                visible=data["visible"],
            )


@dataclass(frozen=True)
class PointTrack:
    """One point's trajectory over the whole clip timeline.

    Args:
        points: ``[T, 2]`` float32 pixel positions, ``NaN`` where unknown.
        confidence: ``[T]`` float32 in ``[0, 1]``.
        visible: ``[T]`` bool.

    Raises:
        ContractError: On any shape, dtype or range violation.
    """

    points: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray

    def __post_init__(self) -> None:
        """Coerce dtypes, validate the contract and freeze the arrays."""
        points = np.asarray(self.points, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        visible = np.asarray(self.visible, dtype=bool)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ContractError(f"points must have shape [T, 2], got {points.shape}")
        t = points.shape[0]
        if confidence.shape != (t,):
            raise ContractError(f"confidence must have shape [T] = {(t,)}, got {confidence.shape}")
        if visible.shape != (t,):
            raise ContractError(f"visible must have shape [T] = {(t,)}, got {visible.shape}")
        if confidence.size and not np.isfinite(confidence).all():
            raise ContractError("confidence must be finite (no NaN/inf)")
        if confidence.size and (confidence.min() < 0.0 or confidence.max() > 1.0):
            raise ContractError("confidence must lie in [0, 1]")
        if points.size and np.isinf(points).any():
            raise ContractError("points must be finite or NaN (no inf)")
        object.__setattr__(self, "points", _freeze(points))
        object.__setattr__(self, "confidence", _freeze(confidence))
        object.__setattr__(self, "visible", _freeze(visible))

    @property
    def n_frames(self) -> int:
        """Length of the clip timeline ``T``."""
        return int(self.points.shape[0])

    def __len__(self) -> int:
        """Length of the clip timeline (same as :attr:`n_frames`)."""
        return self.n_frames

    @classmethod
    def empty(cls, n_frames: int) -> PointTrack:
        """Return an all-invalid track of length ``n_frames``."""
        if n_frames < 0:
            raise ContractError("n_frames must be >= 0")
        return cls(
            points=np.full((n_frames, 2), np.nan, dtype=np.float32),
            confidence=np.zeros(n_frames, dtype=np.float32),
            visible=np.zeros(n_frames, dtype=bool),
        )

    @property
    def valid(self) -> np.ndarray:
        """``[T]`` bool mask of frames with a finite position (read-only)."""
        mask = np.isfinite(self.points).all(axis=1)
        mask.setflags(write=False)
        return mask

    def position_at(self, frame: int) -> tuple[float, float] | None:
        """Return the position at ``frame``, or ``None`` if it is unknown.

        Args:
            frame: Absolute frame index.

        Returns:
            ``(x, y)`` if the sample is finite, otherwise ``None``.

        Raises:
            IndexError: If ``frame`` is outside ``[0, T)``.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range for T={self.n_frames}")
        x, y = self.points[frame]
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (float(x), float(y))

    def as_result(self) -> TrackResult:
        """Wrap this single track in a one-point :class:`TrackResult`."""
        return TrackResult(
            points=self.points[None],
            confidence=self.confidence[None],
            visible=self.visible[None],
        )

    def replace(
        self,
        *,
        points: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        visible: np.ndarray | None = None,
    ) -> PointTrack:
        """Return a new track with the given arrays swapped in.

        The receiver is never modified — this is the only sanctioned way to
        "edit" a track, and it is what keeps raw tracker output immutable.

        Args:
            points: Optional replacement ``[T, 2]`` positions.
            confidence: Optional replacement ``[T]`` confidences.
            visible: Optional replacement ``[T]`` visibility flags.

        Returns:
            A new :class:`PointTrack`.
        """
        return PointTrack(
            points=self.points if points is None else points,
            confidence=self.confidence if confidence is None else confidence,
            visible=self.visible if visible is None else visible,
        )

    def save_npz(self, path: str | os.PathLike[str]) -> None:
        """Write this track to ``path`` in the per-point ``.npz`` format."""
        self.as_result().save_npz(path)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> PointTrack:
        """Read a single-point ``.npz`` written by :meth:`save_npz`.

        Raises:
            ContractError: If the file holds more than one point.
        """
        result = TrackResult.load_npz(path)
        if result.n_points != 1:
            raise ContractError(f"{path} holds {result.n_points} points, expected exactly 1")
        return result.point(0)
