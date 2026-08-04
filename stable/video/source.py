"""``PyAVVideoSource`` — the real, file-backed :class:`~stable.core.video.VideoSource`.

Phase 0 fixed the contract (:mod:`stable.core.video`); this module is Phase 1's
implementation of it against a real video file, using PyAV (FFmpeg bindings).

Random access strategy
-----------------------
FFmpeg decode is fundamentally sequential, so frame-accurate random access
means seeking to the nearest preceding keyframe and decoding forward to the
target. To make that reliable across containers (B-frames reorder presentation
order vs. decode order) the source decodes the whole clip once at construction
time to record every frame's presentation timestamp; that list is then the
source of truth for "which pts is frame N" used by every later seek.

Every frame decoded while seeking forward toward a target is cached, not just
the target itself, so the dominant access pattern in this project — a tracker
scanning sequentially forward or backward from an anchor frame — pays a seek
cost once per run rather than once per frame.
"""

from __future__ import annotations

import bisect
import logging
import os
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..core.errors import VideoError

__all__ = ["PyAVVideoSource"]

logger = logging.getLogger(__name__)


def _av() -> Any:
    """Import and return :mod:`av`, lazily.

    Deferring the import means merely importing this module (or anything that
    imports it transitively, like :mod:`stable.video`) never requires PyAV to
    be installed — only actually constructing a :class:`PyAVVideoSource` does.

    Returns:
        The imported ``av`` module.

    Raises:
        VideoError: If PyAV is not installed.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise VideoError(
            "PyAVVideoSource requires the 'av' package; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return av


class PyAVVideoSource:
    """Random-access decode of a real video file via PyAV.

    Implements the :class:`~stable.core.video.VideoSource` protocol. Frames are
    served as ``uint8`` RGB arrays of shape ``(H, W, 3)``, matching the fixed
    convention every tracker plugin is written against.

    Args:
        path: Path to a video file FFmpeg can demux and decode.
        cache_size: Maximum number of decoded frames kept in the LRU cache.

    Raises:
        VideoError: If PyAV is not installed, the file cannot be opened, or it
            has no video stream.
    """

    def __init__(self, path: str | os.PathLike[str], *, cache_size: int = 64) -> None:
        """Open ``path`` and index its frames; see the class docstring."""
        av = _av()
        self._path = os.fspath(path)
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        try:
            self._container = av.open(self._path)
        except Exception as exc:
            raise VideoError(f"cannot open video {self._path!r}: {exc}") from exc
        if not self._container.streams.video:
            self._container.close()
            raise VideoError(f"{self._path!r} has no video stream")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._width = int(self._stream.codec_context.width)
        self._height = int(self._stream.codec_context.height)
        self._pts_list: list[int] = self._build_pts_index()
        if not self._pts_list:
            raise VideoError(f"{self._path!r} decoded zero frames")
        self._fps = self._compute_fps()
        self._decoder: Iterator[Any] | None = None
        self._decoder_next_index = 0

    # -- construction helpers ---------------------------------------------

    def _build_pts_index(self) -> list[int]:
        """Decode the whole clip once to record every frame's presentation timestamp.

        Returns:
            Presentation timestamps in display order; this list's length is the
            clip's ``frame_count`` and each position is that frame's absolute
            index.
        """
        pts_list: list[int] = []
        for frame in self._container.decode(self._stream):
            pts_list.append(int(frame.pts) if frame.pts is not None else len(pts_list))
        return pts_list

    def _compute_fps(self) -> float:
        """Return the stream's frame rate, falling back to a duration estimate."""
        rate = self._stream.average_rate or self._stream.guessed_rate
        if rate:
            return float(rate)
        if len(self._pts_list) >= 2 and self._stream.time_base:  # pragma: no cover - rare container
            duration = (self._pts_list[-1] - self._pts_list[0]) * float(self._stream.time_base)
            if duration > 0:
                return (len(self._pts_list) - 1) / duration
        return 24.0  # pragma: no cover - only when a container reports nothing usable

    # -- VideoSource protocol ----------------------------------------------

    @property
    def frame_count(self) -> int:
        """Total number of frames ``T`` in the clip."""
        return len(self._pts_list)

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

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``uint8`` RGB of shape ``(H, W, 3)``.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            The decoded frame, safe for the caller to mutate (it is a copy of
            the cached array).

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the decoder cannot reach ``index``.
        """
        if not 0 <= index < self.frame_count:
            raise IndexError(f"frame {index} out of range for T={self.frame_count}")
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached.copy()
        if self._decoder is None or index < self._decoder_next_index:
            self._reseek(index)
        while index not in self._cache:
            self._decode_one()
        result = self._cache[index]
        self._cache.move_to_end(index)
        return result.copy()

    def iter_frames(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` pairs over ``[start, stop)`` in order.

        This is the efficient path for the common case of a tracker scanning a
        contiguous range: it seeks once and decodes straight through, rather
        than reseeking per frame.

        Args:
            start: First frame index, inclusive.
            stop: Last frame index, exclusive; ``None`` means end of clip.

        Yields:
            ``(index, frame)`` with frames as ``uint8`` RGB ``(H, W, 3)``.
        """
        start = max(start, 0)
        stop = self.frame_count if stop is None else min(stop, self.frame_count)
        if start >= stop:
            return
        for index in range(start, stop):
            yield index, self.get_frame(index)

    # -- decode machinery ---------------------------------------------------

    def _reseek(self, index: int) -> None:
        """Seek the container so decoding can reach frame ``index``.

        Seeks to the nearest keyframe at or before ``index`` and primes the
        live decoder; the frame the seek actually lands on is cached
        immediately since it was decoded "for free".
        """
        target_pts = self._pts_list[index]
        self._container.seek(target_pts, stream=self._stream, any_frame=False, backward=True)
        self._decoder = self._container.decode(self._stream)
        try:
            first = next(self._decoder)
        except StopIteration as exc:  # pragma: no cover - malformed file only
            raise VideoError(f"could not decode frame {index} in {self._path!r}") from exc
        landed_index = bisect.bisect_left(self._pts_list, int(first.pts))
        landed_index = min(landed_index, len(self._pts_list) - 1)
        self._store_in_cache(landed_index, first.to_ndarray(format="rgb24"))
        self._decoder_next_index = landed_index + 1

    def _decode_one(self) -> np.ndarray:
        """Pull the next frame from the live decoder, cache it, and return it.

        Raises:
            VideoError: If the decoder is exhausted before it should be.
        """
        assert self._decoder is not None  # noqa: S101 - internal invariant
        try:
            raw = next(self._decoder)
        except StopIteration as exc:
            raise VideoError(
                f"decoder for {self._path!r} ended before frame {self._decoder_next_index}"
            ) from exc
        array = raw.to_ndarray(format="rgb24")
        self._store_in_cache(self._decoder_next_index, array)
        self._decoder_next_index += 1
        return array

    def _store_in_cache(self, index: int, array: np.ndarray) -> None:
        """Insert a decoded frame into the LRU cache, evicting the oldest if full."""
        array = array.copy()
        array.setflags(write=False)
        self._cache[index] = array
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Release the underlying PyAV container. Safe to call more than once."""
        container = getattr(self, "_container", None)
        if container is not None:
            container.close()
            self._container = None
            self._decoder = None

    def __enter__(self) -> PyAVVideoSource:
        """Support use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the container on context-manager exit."""
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC timing is not deterministic
        """Best-effort cleanup if the source is garbage-collected without closing."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        """Return a short debug representation naming the path and size."""
        return (
            f"<PyAVVideoSource path={self._path!r} "
            f"{self._width}x{self._height} T={self.frame_count} fps={self._fps:.3f}>"
        )
