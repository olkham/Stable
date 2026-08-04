"""The :class:`VideoSource` protocol trackers are written against.

Trackers must never import the decoding stack directly: they take *any* object
satisfying this protocol. Phase 1 provides the real PyAV-backed implementation
in :mod:`stable.video`; :class:`SyntheticVideoSource` below is a dependency-free
reference implementation used by tests and by the ``dummy`` tracker.

Frame convention (fixed): ``get_frame`` returns ``uint8`` **RGB** of shape
``(H, W, 3)``. Plugins that want BGR or grayscale convert on their own.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import ContractError

__all__ = ["VideoSource", "SyntheticVideoSource"]


@runtime_checkable
class VideoSource(Protocol):
    """Random-access read-only view of a decoded clip.

    Implementations are expected to cache decoded frames; trackers assume
    ``get_frame`` is cheap for nearby indices and may walk the clip backwards.
    """

    @property
    def frame_count(self) -> int:
        """Total number of frames ``T`` in the clip."""
        ...

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...

    @property
    def fps(self) -> float:
        """Frame rate in frames per second."""
        ...

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``uint8`` RGB of shape ``(H, W, 3)``.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            The decoded frame.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        ...

    def iter_frames(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` pairs over ``[start, stop)`` in order.

        Args:
            start: First frame index, inclusive.
            stop: Last frame index, exclusive; ``None`` means end of clip.

        Yields:
            ``(index, frame)`` with frames as ``uint8`` RGB ``(H, W, 3)``.
        """
        ...


class SyntheticVideoSource:
    """A deterministic, procedurally generated clip — no I/O, no dependencies.

    The frame content is a static gradient plus a bright square whose centre
    follows a known circular path, so tests can compare tracker output against
    :meth:`true_position` ground truth.

    Args:
        frame_count: Number of frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Nominal frame rate.
        radius: Radius of the marker's circular path, in pixels.
        period: Number of frames per full revolution.
    """

    def __init__(
        self,
        frame_count: int = 24,
        width: int = 64,
        height: int = 48,
        fps: float = 24.0,
        *,
        radius: float = 8.0,
        period: float = 24.0,
    ) -> None:
        """Create a synthetic clip; see the class docstring for the arguments."""
        if frame_count < 0 or width <= 0 or height <= 0:
            raise ContractError("SyntheticVideoSource needs frame_count >= 0 and positive size")
        self._frame_count = int(frame_count)
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._radius = float(radius)
        self._period = float(period)
        self.path: str | None = None

    @property
    def frame_count(self) -> int:
        """Total number of frames ``T`` in the clip."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    @property
    def fps(self) -> float:
        """Frame rate in frames per second."""
        return self._fps

    def true_position(self, frame: int) -> tuple[float, float]:
        """Return the ground-truth marker centre at ``frame``.

        Args:
            frame: Absolute frame index.

        Returns:
            ``(x, y)`` in pixels.
        """
        angle = 2.0 * math.pi * (frame / self._period)
        return (
            self._width / 2.0 + self._radius * math.cos(angle),
            self._height / 2.0 + self._radius * math.sin(angle),
        )

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``uint8`` RGB of shape ``(H, W, 3)``.

        Raises:
            IndexError: If ``index`` is outside ``[0, frame_count)``.
        """
        if not 0 <= index < self._frame_count:
            raise IndexError(f"frame {index} out of range for T={self._frame_count}")
        ys, xs = np.mgrid[0 : self._height, 0 : self._width]
        base = ((xs + ys) % 256).astype(np.uint8)
        frame = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1)
        cx, cy = self.true_position(index)
        x0, x1 = int(round(cx)) - 2, int(round(cx)) + 3
        y0, y1 = int(round(cy)) - 2, int(round(cy)) + 3
        x0, x1 = max(x0, 0), min(x1, self._width)
        y0, y1 = max(y0, 0), min(y1, self._height)
        frame[y0:y1, x0:x1] = 255
        return frame

    def iter_frames(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` pairs over ``[start, stop)`` in order."""
        stop = self._frame_count if stop is None else min(stop, self._frame_count)
        for index in range(max(start, 0), stop):
            yield index, self.get_frame(index)
