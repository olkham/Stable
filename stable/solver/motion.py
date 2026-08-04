"""Point tracks in, four motion curves out — with graceful degradation.

:func:`solve_motion` walks the clip and fits one similarity per frame against a
reference frame, producing the ``X, Y, R, S`` curves everything downstream (the
curve editor, the smoothing library, the stabilisation renderer, the CSV/JSON
exports) is written against, plus a per-frame quality record.

Curve convention (pivot-relative)
---------------------------------
:class:`~stable.core.geometry.SimilarityTransform` is origin-based, and that is
not negotiable — but reporting ``tx``/``ty`` about the top-left corner would make
the X and Y curves nearly useless to a human: a pure rotation about the frame
centre shows up as hundreds of pixels of "translation", tightly correlated with
the rotation curve, and smoothing the two independently then fights itself.

So the curves are reported **about a pivot** (the frame centre by default):

* ``x``, ``y`` — how far the pivot point moved, in pixels.
* ``r`` — rotation in radians, **unwrapped** across the clip so a shot passing
  through +/-pi produces a continuous curve a filter can work on.
* ``s`` — uniform scale.

Rotation and scale are pivot-independent, so only the translation pair changes
meaning. :meth:`MotionSolve.transform_at` folds the pivot back in via
:meth:`~stable.core.geometry.SimilarityTransform.about_pivot` and returns the
plain origin-based transform, so no other module has to know about any of this.

At the reference frame the fit is the identity by construction, so the curves
read ``(0, 0, 0, 1)`` there.

Graceful degradation (never raises for a bad frame)
---------------------------------------------------
Real tracks have holes. Every frame lands in one of five states, recorded in
:class:`SolveStatus` and surfaced through :attr:`MotionSolve.flagged`:

``SOLVED``
    Two or more usable correspondences; three or more also go through RANSAC.
``TRANSLATION_ONLY``
    Exactly one usable point — X and Y are real, rotation and scale are not
    observable and are held at identity.
``INTERPOLATED``
    No usable point at all, but solved frames exist on both sides: the curves
    are linearly interpolated across the hole.
``HELD``
    No usable point, and the hole runs off the start or end of the clip: the
    nearest solved value is held.
``IDENTITY``
    Not one frame in the whole clip could be solved.

Only a reference frame with no usable points at all raises, because that is a
configuration mistake rather than a data gap: nothing anywhere in the clip could
be measured against it.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from ..core.errors import SolverError
from ..core.geometry import CHANNELS, SimilarityTransform
from ..core.project import SolveSettings
from ..core.types import PointTrack, TrackResult
from ..trackers.base import ProgressCallback, noop_progress
from .ransac import RansacConfig, solve_similarity_ransac
from .similarity import SolveConstraints, transform_residuals, weighted_rms

__all__ = [
    "SolveStatus",
    "SolveConfig",
    "FrameSolve",
    "MotionSolve",
    "solve_motion",
    "solve_frame",
]

IDENTITY_CURVES: dict[str, float] = {"x": 0.0, "y": 0.0, "r": 0.0, "s": 1.0}
"""Curve values of the identity transform, used wherever nothing can be solved."""


class SolveStatus(IntEnum):
    """How a frame's transform was arrived at.

    Attributes:
        SOLVED: Fitted from two or more usable correspondences.
        TRANSLATION_ONLY: Fitted from a single point; rotation and scale held.
        INTERPOLATED: No usable point; interpolated between solved neighbours.
        HELD: No usable point and no solved frame on one side; nearest held.
        IDENTITY: Nothing in the clip could be solved at all.
    """

    SOLVED = 0
    TRANSLATION_ONLY = 1
    INTERPOLATED = 2
    HELD = 3
    IDENTITY = 4

    @property
    def label(self) -> str:
        """Lowercase name, for CSV/JSON exports and the UI."""
        return self.name.lower()


@dataclass(frozen=True)
class SolveConfig:
    """Everything :func:`solve_motion` needs beyond the tracks themselves.

    Args:
        reference_frame: Frame the solve is relative to; its transform is the
            identity by construction.
        channels: Which of ``x``, ``y``, ``r``, ``s`` the solve may use.
            Disabling ``r`` or ``s`` constrains the fit itself (see
            :class:`~stable.solver.similarity.SolveConstraints`); disabling
            ``x`` or ``y`` zeroes that pivot-relative translation curve, so the
            render simply never moves the frame along that axis.
        pivot: ``(x, y)`` the translation curves are reported about. ``None``
            means the centre of ``frame_size`` when that is known, otherwise the
            weighted centroid of the reference-frame point positions.
        min_confidence: Samples below this confidence are treated as missing.
        max_residual_px: Solved frames whose residual exceeds this are flagged
            for the user's attention (they are still used).
        ransac: RANSAC tuning, applied on frames with three or more points.
        interpolate_gaps: Fill unsolved frames from their solved neighbours.
            When False those frames are left at the identity and flagged.

    Raises:
        SolverError: If a value is out of range or a channel name is unknown.
    """

    reference_frame: int = 0
    channels: Mapping[str, bool] = field(
        default_factory=lambda: {"x": True, "y": True, "r": True, "s": True}
    )
    pivot: tuple[float, float] | None = None
    min_confidence: float = 0.05
    max_residual_px: float = 3.0
    ransac: RansacConfig = field(default_factory=RansacConfig)
    interpolate_gaps: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration (dataclass hook)."""
        if self.reference_frame < 0:
            raise SolverError("SolveConfig.reference_frame must be >= 0")
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise SolverError(f"unknown solve channel(s) {sorted(unknown)}; expected {CHANNELS}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise SolverError("SolveConfig.min_confidence must lie in [0, 1]")
        if self.max_residual_px <= 0.0:
            raise SolverError("SolveConfig.max_residual_px must be positive")

    @classmethod
    def from_settings(cls, settings: SolveSettings, **overrides: Any) -> SolveConfig:
        """Build a config from a project file's ``solve`` block.

        Args:
            settings: The project's :class:`~stable.core.project.SolveSettings`.
            **overrides: Any other :class:`SolveConfig` field.

        Returns:
            The corresponding :class:`SolveConfig`.
        """
        return cls(
            reference_frame=settings.reference_frame,
            channels=dict(settings.channels),
            **overrides,
        )

    def enabled(self, channel: str) -> bool:
        """Whether ``channel`` is enabled (unknown channels default to on)."""
        return bool(self.channels.get(channel, True))

    @property
    def constraints(self) -> SolveConstraints:
        """The rotation/scale locks implied by :attr:`channels`."""
        return SolveConstraints.from_channels(self.channels)


@dataclass(frozen=True)
class FrameSolve:
    """One frame's fit, before the curves are assembled.

    Args:
        frame: Absolute frame index.
        transform: The origin-based similarity taking reference coordinates to
            this frame, or ``None`` when the frame could not be solved.
        residual: Weighted RMS reprojection error over the inliers, in pixels.
        n_points: Usable correspondences the frame had.
        n_inliers: How many of those survived RANSAC.
        status: How the transform was arrived at.
    """

    frame: int
    transform: SimilarityTransform | None
    residual: float
    n_points: int
    n_inliers: int
    status: SolveStatus


def _freeze(array: np.ndarray) -> np.ndarray:
    """Return a read-only copy of ``array``."""
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class MotionSolve:
    """The solved motion of a clip: four curves plus per-frame solve quality.

    Instances are immutable and their arrays are read-only; smoothing produces a
    new object rather than editing this one, matching the project's rule that
    solved and raw data are never mutated in place.

    Args:
        curves: ``{"x", "y", "r", "s"}`` to ``[T]`` float64 arrays, reported
            about :attr:`pivot` (see the module docstring).
        residual: ``[T]`` weighted RMS reprojection error in pixels; ``NaN``
            where nothing was fitted.
        n_points: ``[T]`` int32 usable correspondences per frame.
        n_inliers: ``[T]`` int32 correspondences that survived RANSAC.
        status: ``[T]`` int8 :class:`SolveStatus` values.
        reference_frame: The frame the solve is relative to.
        pivot: ``(x, y)`` the translation curves are reported about.
        max_residual_px: Residual above which a solved frame is flagged.
    """

    curves: dict[str, np.ndarray]
    residual: np.ndarray
    n_points: np.ndarray
    n_inliers: np.ndarray
    status: np.ndarray
    reference_frame: int
    pivot: tuple[float, float]
    max_residual_px: float = 3.0

    def __post_init__(self) -> None:
        """Validate shapes and freeze every array (dataclass hook)."""
        missing = set(CHANNELS) - set(self.curves)
        if missing:
            raise SolverError(f"MotionSolve is missing curve(s) {sorted(missing)}")
        frozen = {c: _freeze(np.asarray(self.curves[c], dtype=np.float64)) for c in CHANNELS}
        length = frozen["x"].shape[0]
        for channel, values in frozen.items():
            if values.ndim != 1 or values.shape[0] != length:
                raise SolverError(f"curve {channel!r} must be a [T] array of length {length}")
        object.__setattr__(self, "curves", frozen)
        for name, dtype in (
            ("residual", np.float64),
            ("n_points", np.int32),
            ("n_inliers", np.int32),
            ("status", np.int8),
        ):
            array = np.asarray(getattr(self, name), dtype=dtype)
            if array.shape != (length,):
                raise SolverError(f"MotionSolve.{name} must have shape [{length}]")
            object.__setattr__(self, name, _freeze(array))
        object.__setattr__(self, "pivot", (float(self.pivot[0]), float(self.pivot[1])))

    # -- shape ------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Length of the clip timeline ``T``."""
        return int(self.curves["x"].shape[0])

    def __len__(self) -> int:
        """Length of the clip timeline (same as :attr:`n_frames`)."""
        return self.n_frames

    def curve(self, channel: str) -> np.ndarray:
        """Return the ``[T]`` read-only curve for ``channel``.

        Args:
            channel: One of ``x``, ``y``, ``r``, ``s``.

        Raises:
            SolverError: If ``channel`` is not a known channel.
        """
        if channel not in self.curves:
            raise SolverError(f"unknown channel {channel!r}; expected {CHANNELS}")
        return self.curves[channel]

    # -- transforms --------------------------------------------------------

    def transform_at(self, frame: int) -> SimilarityTransform:
        """Return the origin-based similarity for ``frame``.

        Args:
            frame: Absolute frame index.

        Returns:
            The transform taking reference-frame coordinates to ``frame``'s
            coordinates, with the pivot folded back into the translation.

        Raises:
            IndexError: If ``frame`` is outside ``[0, T)``.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range for T={self.n_frames}")
        return SimilarityTransform.about_pivot(
            self.pivot,
            rotation=float(self.curves["r"][frame]),
            scale=float(self.curves["s"][frame]),
            translation=(float(self.curves["x"][frame]), float(self.curves["y"][frame])),
        )

    def transforms(self) -> list[SimilarityTransform]:
        """Return every frame's origin-based similarity, in frame order."""
        return [self.transform_at(f) for f in range(self.n_frames)]

    # -- quality -----------------------------------------------------------

    @property
    def flagged(self) -> np.ndarray:
        """``[T]`` bool: frames the user should look at.

        True where the frame was not cleanly solved (degraded, interpolated,
        held or identity) or where its residual exceeds
        :attr:`max_residual_px`. This is the array a UI turns into a timeline
        heat strip — the "go fix this" signal for the solve, mirroring the
        confidence dips :mod:`stable.trackers.merge` produces for tracks.
        """
        residual_bad = np.nan_to_num(self.residual, nan=0.0) > self.max_residual_px
        mask: np.ndarray = (self.status != int(SolveStatus.SOLVED)) | residual_bad
        mask.setflags(write=False)
        return mask

    def status_at(self, frame: int) -> SolveStatus:
        """Return the :class:`SolveStatus` of ``frame``."""
        return SolveStatus(int(self.status[frame]))

    def summary(self) -> dict[str, Any]:
        """Return a small dict of headline solve-quality numbers.

        Returns:
            Frame counts per status, the flagged-frame count, and the mean and
            worst residual over the frames that were actually fitted.
        """
        finite = self.residual[np.isfinite(self.residual)]
        return {
            "frames": self.n_frames,
            "reference_frame": self.reference_frame,
            "pivot": list(self.pivot),
            "status_counts": {
                status.label: int(np.count_nonzero(self.status == int(status)))
                for status in SolveStatus
            },
            "flagged": int(np.count_nonzero(self.flagged)),
            "mean_residual_px": float(finite.mean()) if finite.size else 0.0,
            "max_residual_px": float(finite.max()) if finite.size else 0.0,
        }

    # -- exports -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the curves and per-frame quality."""
        return {
            "version": 1,
            "reference_frame": self.reference_frame,
            "pivot": list(self.pivot),
            "frames": self.n_frames,
            "curves": {c: self.curves[c].tolist() for c in CHANNELS},
            "residual": [None if not np.isfinite(v) else float(v) for v in self.residual],
            "n_points": self.n_points.tolist(),
            "n_inliers": self.n_inliers.tolist(),
            "status": [SolveStatus(int(v)).label for v in self.status],
            "flagged": self.flagged.tolist(),
            "summary": self.summary(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the solve serialised as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Return the curves as CSV text, one row per frame.

        Columns are ``frame, x, y, r_rad, r_deg, s, residual_px, n_points,
        n_inliers, status, flagged``. Rotation appears in both units so the file
        is readable by a human without giving up the radian convention the rest
        of the project uses.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "frame", "x", "y", "r_rad", "r_deg", "s",
                "residual_px", "n_points", "n_inliers", "status", "flagged",
            ]
        )
        flagged = self.flagged
        for frame in range(self.n_frames):
            rotation = float(self.curves["r"][frame])
            residual = float(self.residual[frame])
            writer.writerow(
                [
                    frame,
                    f"{float(self.curves['x'][frame]):.6f}",
                    f"{float(self.curves['y'][frame]):.6f}",
                    f"{rotation:.9f}",
                    f"{np.degrees(rotation):.6f}",
                    f"{float(self.curves['s'][frame]):.9f}",
                    "" if not np.isfinite(residual) else f"{residual:.6f}",
                    int(self.n_points[frame]),
                    int(self.n_inliers[frame]),
                    SolveStatus(int(self.status[frame])).label,
                    int(bool(flagged[frame])),
                ]
            )
        return buffer.getvalue()

    def save(self, path: str | os.PathLike[str]) -> str:
        """Write the solve to ``path`` as CSV or JSON, chosen by extension.

        Args:
            path: Destination; ``.csv`` writes CSV, anything else JSON.

        Returns:
            The absolute path written.
        """
        target = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        text = self.to_csv() if target.lower().endswith(".csv") else self.to_json() + "\n"
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return target


# -- solving -----------------------------------------------------------------


def _as_tracks(tracks: TrackResult | Sequence[PointTrack]) -> list[PointTrack]:
    """Normalise the accepted track containers to a list of per-point tracks."""
    if isinstance(tracks, TrackResult):
        return tracks.tracks()
    listed = list(tracks)
    for track in listed:
        if not isinstance(track, PointTrack):
            raise SolverError(
                "tracks must be a TrackResult or a sequence of PointTrack, "
                f"got a {type(track).__name__}"
            )
    return listed


def _default_pivot(
    frame_size: tuple[int, int] | None, source: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Pick the pivot the translation curves are reported about.

    The frame centre when the clip size is known (what a user means by "the shot
    moved this far"), otherwise the confidence-weighted centroid of the points
    that were actually usable in the reference frame.
    """
    if frame_size is not None:
        return (float(frame_size[0]) / 2.0, float(frame_size[1]) / 2.0)
    total = float(weights.sum())
    if total <= 0.0:  # pragma: no cover - guarded by the caller
        centre = source.mean(axis=0)
    else:
        centre = (weights[:, None] * source).sum(axis=0) / total
    return (float(centre[0]), float(centre[1]))


def _solve_frame(
    frame: int,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    config: SolveConfig,
) -> FrameSolve:
    """Fit one frame, degrading rather than failing when points run out."""
    count = source.shape[0]
    if count == 0:
        return FrameSolve(frame, None, float("nan"), 0, 0, SolveStatus.INTERPOLATED)
    if count == 1:
        shift = target[0] - source[0]
        transform = SimilarityTransform(tx=float(shift[0]), ty=float(shift[1]))
        return FrameSolve(frame, transform, 0.0, 1, 1, SolveStatus.TRANSLATION_ONLY)

    result = solve_similarity_ransac(
        source, target, weights, constraints=config.constraints, config=config.ransac
    )
    inliers = result.inliers
    if not inliers.any():  # pragma: no cover - RANSAC always returns a consensus
        inliers = np.ones(count, dtype=bool)
    errors = transform_residuals(result.transform, source[inliers], target[inliers])
    residual = weighted_rms(errors, weights[inliers])
    return FrameSolve(
        frame,
        result.transform,
        float(residual),
        int(count),
        int(np.count_nonzero(inliers)),
        SolveStatus.SOLVED,
    )


def solve_motion(
    tracks: TrackResult | Sequence[PointTrack],
    *,
    config: SolveConfig | None = None,
    frame_size: tuple[int, int] | None = None,
    progress: ProgressCallback = noop_progress,
) -> MotionSolve:
    """Solve a clip's ``X, Y, R, S`` motion from its point tracks.

    Args:
        tracks: A :class:`~stable.core.types.TrackResult` or a sequence of
            :class:`~stable.core.types.PointTrack`, all over the same clip
            timeline. Points the project disabled for solving should already
            have been filtered out by the caller.
        config: Solve settings; defaults to :class:`SolveConfig`.
        frame_size: ``(width, height)`` of the clip, used to default the pivot
            to the frame centre. Optional, but worth passing.
        progress: Progress reporter, called as the clip is walked.

    Returns:
        The :class:`MotionSolve`.

    Raises:
        SolverError: If there are no tracks, the tracks disagree on ``T``, the
            reference frame is past the end of the clip, or no point at all is
            usable in the reference frame.
    """
    settings = config or SolveConfig()
    listed = _as_tracks(tracks)
    if not listed:
        raise SolverError("solve_motion needs at least one track")
    lengths = {track.n_frames for track in listed}
    if len(lengths) != 1:
        raise SolverError(f"all tracks must share the same T, got {sorted(lengths)}")
    n_frames = lengths.pop()
    reference = settings.reference_frame
    if not 0 <= reference < n_frames:
        raise SolverError(
            f"reference_frame {reference} is outside the clip (T={n_frames})"
        )

    positions = np.stack([track.points.astype(np.float64) for track in listed])  # [N, T, 2]
    confidence = np.stack([track.confidence.astype(np.float64) for track in listed])  # [N, T]
    usable = np.isfinite(positions).all(axis=2) & (confidence >= settings.min_confidence)

    reference_usable = usable[:, reference]
    if not reference_usable.any():
        raise SolverError(
            f"no point is usable in reference frame {reference} "
            f"(need a finite position with confidence >= {settings.min_confidence}); "
            "pick a different reference frame or lower min_confidence"
        )

    source_all = positions[reference_usable, reference]  # [M, 2]
    reference_weight = confidence[reference_usable, reference]  # [M]
    pivot = settings.pivot or _default_pivot(frame_size, source_all, reference_weight)

    frame_positions = positions[reference_usable]  # [M, T, 2]
    frame_confidence = confidence[reference_usable]  # [M, T]
    frame_usable = usable[reference_usable]  # [M, T]

    solves: list[FrameSolve] = []
    for frame in range(n_frames):
        selector = frame_usable[:, frame]
        # A point only speaks for a frame if it is trustworthy at *both* ends of
        # the correspondence, so the reference confidence multiplies in.
        weights = reference_weight[selector] * frame_confidence[selector, frame]
        solves.append(
            _solve_frame(
                frame,
                source_all[selector],
                frame_positions[selector, frame],
                weights,
                settings,
            )
        )
        if n_frames and frame % 32 == 0:
            progress(frame / n_frames, f"frame {frame}/{n_frames}")
    progress(1.0, "solved")

    return _assemble(solves, settings, pivot, n_frames)


def _assemble(
    solves: list[FrameSolve],
    config: SolveConfig,
    pivot: tuple[float, float],
    n_frames: int,
) -> MotionSolve:
    """Turn per-frame fits into continuous curves, filling and flagging holes."""
    status = np.full(n_frames, int(SolveStatus.INTERPOLATED), dtype=np.int8)
    residual = np.full(n_frames, np.nan, dtype=np.float64)
    n_points = np.zeros(n_frames, dtype=np.int32)
    n_inliers = np.zeros(n_frames, dtype=np.int32)
    curves = {channel: np.full(n_frames, np.nan, dtype=np.float64) for channel in CHANNELS}

    solved_frames: list[int] = []
    for solve in solves:
        status[solve.frame] = int(solve.status)
        residual[solve.frame] = solve.residual
        n_points[solve.frame] = solve.n_points
        n_inliers[solve.frame] = solve.n_inliers
        if solve.transform is None:
            continue
        solved_frames.append(solve.frame)
        moved = solve.transform.apply(np.asarray(pivot, dtype=np.float64))
        curves["x"][solve.frame] = float(moved[0]) - pivot[0]
        curves["y"][solve.frame] = float(moved[1]) - pivot[1]
        curves["r"][solve.frame] = solve.transform.rotation
        curves["s"][solve.frame] = solve.transform.scale

    if not solved_frames:
        for channel in CHANNELS:
            curves[channel][:] = IDENTITY_CURVES[channel]
        status[:] = int(SolveStatus.IDENTITY)
        return MotionSolve(
            curves=curves,
            residual=residual,
            n_points=n_points,
            n_inliers=n_inliers,
            status=status,
            reference_frame=config.reference_frame,
            pivot=pivot,
            max_residual_px=config.max_residual_px,
        )

    known = np.asarray(solved_frames, dtype=np.int64)
    # Unwrap before interpolating: a shot passing through +/-pi otherwise gets a
    # 2pi step no filter can smooth, and a gap across the step interpolates the
    # long way round.
    curves["r"][known] = np.unwrap(curves["r"][known])

    missing = np.setdiff1d(np.arange(n_frames, dtype=np.int64), known, assume_unique=False)
    if missing.size:
        if config.interpolate_gaps:
            for channel in CHANNELS:
                curves[channel][missing] = np.interp(missing, known, curves[channel][known])
        else:
            for channel in CHANNELS:
                curves[channel][missing] = IDENTITY_CURVES[channel]
        # np.interp clamps outside the known range, which is exactly "hold the
        # nearest solved value" — a different (and worse) guarantee than an
        # interior fill, so it gets its own status.
        outside = (missing < known[0]) | (missing > known[-1])
        status[missing[outside]] = int(SolveStatus.HELD)
        status[missing[~outside]] = int(SolveStatus.INTERPOLATED)

    for channel in ("x", "y"):
        if not config.enabled(channel):
            curves[channel][:] = 0.0

    return MotionSolve(
        curves=curves,
        residual=residual,
        n_points=n_points,
        n_inliers=n_inliers,
        status=status,
        reference_frame=config.reference_frame,
        pivot=pivot,
        max_residual_px=config.max_residual_px,
    )


def solve_frame(
    tracks: TrackResult | Sequence[PointTrack],
    frame: int,
    *,
    config: SolveConfig | None = None,
) -> FrameSolve:
    """Solve a single frame — a convenience for previews, tuning and tests.

    Uses exactly the same path as one iteration of :func:`solve_motion`, so a
    preview cannot disagree with the full solve.

    Args:
        tracks: The same containers :func:`solve_motion` accepts.
        frame: Absolute frame index to solve.
        config: Solve settings; defaults to :class:`SolveConfig`.

    Returns:
        The frame's :class:`FrameSolve`. Its ``transform`` is the origin-based
        similarity taking reference coordinates to ``frame``, or ``None`` when
        no point was usable in both frames.

    Raises:
        SolverError: If there are no tracks, or nothing is usable in the
            reference frame.
    """
    settings = config or SolveConfig()
    listed = _as_tracks(tracks)
    if not listed:
        raise SolverError("solve_frame needs at least one track")
    reference = settings.reference_frame
    positions = np.stack([track.points.astype(np.float64) for track in listed])
    confidence = np.stack([track.confidence.astype(np.float64) for track in listed])
    usable = np.isfinite(positions).all(axis=2) & (confidence >= settings.min_confidence)
    if not usable[:, reference].any():
        raise SolverError(f"no point is usable in reference frame {reference}")
    selector = usable[:, reference] & usable[:, frame]
    weights = confidence[selector, reference] * confidence[selector, frame]
    return _solve_frame(
        frame,
        positions[selector, reference],
        positions[selector, frame],
        weights,
        settings,
    )
