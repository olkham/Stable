"""Smoothing the solved motion, and turning the result into a stabilisation warp.

:func:`smooth_motion` runs each channel's filter chain over the corresponding
curve of a :class:`~stable.solver.motion.MotionSolve` and returns a
:class:`SmoothedMotion` — the raw solve, the smoothed curves, and the one thing
the renderer actually consumes:

    warp(t) = smoothed(t) . raw(t)^-1

Read it right to left: take frame ``t``'s pixels back to reference-frame
coordinates with ``raw(t)^-1``, then re-apply the *motion you wish the camera
had had*. A channel that was not smoothed contributes ``smoothed == raw`` and
cancels out of the product exactly, which is why disabling a channel is
guaranteed to leave that axis untouched rather than approximately untouched.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

import numpy as np

from ..core.errors import SmoothingError
from ..core.geometry import CHANNELS, SimilarityTransform
from ..solver.motion import MotionSolve
from .channels import SmoothingSettings
from .filters import FilterContext

__all__ = ["SmoothedMotion", "smooth_motion", "MIN_SCALE"]

MIN_SCALE = 1e-3
"""Floor applied to the smoothed scale curve; a zero scale is not invertible."""


def _freeze(array: np.ndarray) -> np.ndarray:
    """Return a read-only copy of ``array``."""
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


class SmoothedMotion:
    """A solved clip plus the smoothed motion the render should follow.

    Args:
        solve: The raw :class:`~stable.solver.motion.MotionSolve`.
        curves: ``{"x", "y", "r", "s"}`` to ``[T]`` smoothed values, in the same
            pivot-relative convention as the solve's own curves.
        settings: The configuration that produced them.
        fps: Frame rate the filters were designed against.

    Raises:
        SmoothingError: If a curve is missing or the wrong length.
    """

    def __init__(
        self,
        solve: MotionSolve,
        curves: dict[str, np.ndarray],
        settings: SmoothingSettings,
        fps: float,
    ) -> None:
        """Store the smoothed curves; see the class docstring for the arguments."""
        missing = set(CHANNELS) - set(curves)
        if missing:
            raise SmoothingError(f"SmoothedMotion is missing curve(s) {sorted(missing)}")
        frozen: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            values = np.asarray(curves[channel], dtype=np.float64)
            if values.shape != (solve.n_frames,):
                raise SmoothingError(
                    f"smoothed curve {channel!r} must have shape [{solve.n_frames}], "
                    f"got {values.shape}"
                )
            frozen[channel] = _freeze(values)
        self.solve = solve
        self.curves = frozen
        self.settings = settings
        self.fps = float(fps)

    # -- shape -------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Length of the clip timeline ``T``."""
        return self.solve.n_frames

    def __len__(self) -> int:
        """Length of the clip timeline (same as :attr:`n_frames`)."""
        return self.n_frames

    @property
    def pivot(self) -> tuple[float, float]:
        """The pivot the translation curves are reported about."""
        return self.solve.pivot

    @property
    def reference_frame(self) -> int:
        """The frame the underlying solve is relative to."""
        return self.solve.reference_frame

    def curve(self, channel: str) -> np.ndarray:
        """Return the ``[T]`` read-only smoothed curve for ``channel``.

        Raises:
            SmoothingError: If ``channel`` is not a known channel.
        """
        if channel not in self.curves:
            raise SmoothingError(f"unknown channel {channel!r}; expected {CHANNELS}")
        return self.curves[channel]

    def correction(self, channel: str) -> np.ndarray:
        """Return ``smoothed - raw`` for ``channel`` — what the render will undo.

        This is the curve editor's "before/after" difference and the honest
        measure of how much the frame is being pushed around; large values mean
        a large crop will be needed.
        """
        return self.curve(channel) - self.solve.curve(channel)

    # -- transforms --------------------------------------------------------

    def transform_at(self, frame: int) -> SimilarityTransform:
        """Return the smoothed origin-based similarity for ``frame``.

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

    def stabilization_transform(self, frame: int) -> SimilarityTransform:
        """Return ``smoothed(frame) . raw(frame)^-1`` — the warp for one frame.

        Args:
            frame: Absolute frame index.

        Returns:
            The single affine that maps frame ``frame``'s pixels to their
            stabilised positions.
        """
        return self.transform_at(frame).compose(self.solve.transform_at(frame).inverse())

    def stabilization_transforms(self) -> list[SimilarityTransform]:
        """Return the stabilisation warp for every frame, in frame order."""
        return [self.stabilization_transform(f) for f in range(self.n_frames)]

    # -- exports -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict holding both the raw solve and the smoothing."""
        return {
            "version": 1,
            "fps": self.fps,
            "solve": self.solve.to_dict(),
            "smoothing": self.settings.to_dict(),
            "smoothed": {c: self.curves[c].tolist() for c in CHANNELS},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the smoothed motion serialised as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Return raw and smoothed curves side by side as CSV text.

        Columns are ``frame``, then ``raw_*`` and ``smooth_*`` for each channel,
        then ``residual_px`` and ``status`` from the solve — everything a user
        needs to see what the solve found and what the render will do about it.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            ["frame"]
            + [f"raw_{c}" for c in CHANNELS]
            + [f"smooth_{c}" for c in CHANNELS]
            + ["residual_px", "status", "flagged"]
        )
        from ..solver.motion import SolveStatus  # local: avoids a circular import at module load

        flagged = self.solve.flagged
        for frame in range(self.n_frames):
            residual = float(self.solve.residual[frame])
            writer.writerow(
                [frame]
                + [f"{float(self.solve.curve(c)[frame]):.9f}" for c in CHANNELS]
                + [f"{float(self.curves[c][frame]):.9f}" for c in CHANNELS]
                + [
                    "" if not np.isfinite(residual) else f"{residual:.6f}",
                    SolveStatus(int(self.solve.status[frame])).label,
                    int(bool(flagged[frame])),
                ]
            )
        return buffer.getvalue()

    def save(self, path: str | os.PathLike[str]) -> str:
        """Write the curves to ``path`` as CSV or JSON, chosen by extension.

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

    def __repr__(self) -> str:
        """Return a short debug representation naming the active channels."""
        active = ",".join(self.settings.active_channels) or "none"
        return f"<SmoothedMotion T={self.n_frames} fps={self.fps:g} smoothing={active}>"


def smooth_motion(
    solve: MotionSolve,
    settings: SmoothingSettings | None = None,
    *,
    fps: float = 25.0,
) -> SmoothedMotion:
    """Smooth every channel of a solve according to ``settings``.

    Args:
        solve: The raw motion solve.
        settings: Per-channel configuration; ``None`` means "smooth nothing",
            which yields a stabilisation warp of exactly the identity.
        fps: Frame rate, so frequency-domain filters mean what they say. Pass
            the clip's real rate.

    Returns:
        The :class:`SmoothedMotion`.

    Raises:
        SmoothingError: If a filter name or parameter is invalid, or ``fps`` is
            not positive.
    """
    configuration = settings or SmoothingSettings()
    context = FilterContext(
        fps=fps,
        reference_frame=solve.reference_frame,
        channel="x",
        n_frames=solve.n_frames,
    )
    curves: dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        smoothed = configuration.channel(channel).apply(solve.curve(channel), context)
        if channel == "s":
            # A smoothed scale of zero would make the warp non-invertible, and a
            # negative one would mirror the frame; neither is a camera move.
            smoothed = np.maximum(smoothed, MIN_SCALE)
        curves[channel] = smoothed
    return SmoothedMotion(solve=solve, curves=curves, settings=configuration, fps=fps)
