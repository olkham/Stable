"""``PyAVVideoSource`` — the real, file-backed :class:`~stable.core.video.VideoSource`.

Phase 0 fixed the contract (:mod:`stable.core.video`); this module is Phase 1's
implementation of it against a real video file, using PyAV (FFmpeg bindings).

Random access strategy
-----------------------
FFmpeg decode is fundamentally sequential, so frame-accurate random access
means seeking to the nearest preceding keyframe and decoding forward to the
target. To make that reliable across containers (B-frames reorder presentation
order vs. decode order) the source builds, at construction time, a list of every
frame's presentation timestamp in display order; that list is then the source of
truth for "which pts is frame N" used by every later seek.

That index is built by **demuxing packets, not decoding them** — a packet
carries the presentation timestamp of the frame inside it, and reading them
costs no pixel work at all. It is only trusted when the container's own frame
count agrees with the number of packets found; anything less tidy (a stream that
reports no frame count, packets without timestamps, several frames per packet)
falls back to the original decode-everything pass, which is always correct.
Opening a 240-frame 1080p clip drops from ~0.49 s to ~0.08 s.

Every frame decoded while seeking forward toward a target is cached, not just
the target itself, so the dominant access pattern in this project — a tracker
scanning sequentially forward or backward from an anchor frame — pays a seek
cost once per run rather than once per frame. The cache is
:class:`~stable.video.cache.FrameCache`, budgeted in bytes rather than frames
and shareable between sources (see :mod:`stable.video.pool`), so several runs
over one clip decode it once between them instead of once each.
"""

from __future__ import annotations

import bisect
import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..core.errors import VideoError
from .cache import FrameCache
from .telemetry import note_frame

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

    Safe to call from several threads: decoding is serialised on an internal
    lock, so one source can back concurrent jobs. Concurrent readers with
    *different* cursors will make each other seek, though, which is why
    :class:`~stable.video.pool.PooledVideoSource` gives each one its own decoder
    over a shared cache instead.

    Args:
        path: Path to a video file FFmpeg can demux and decode.
        cache_size: Maximum number of decoded frames kept in the LRU cache.
        cache: An existing :class:`~stable.video.cache.FrameCache` to use — pass
            the same one to several sources over the same file and they share
            decoded frames. ``None`` creates a private cache.
        decode_threads: Threads FFmpeg may use to decode one stream. ``0``
            (the default) lets FFmpeg choose, which is what saturates more than
            one core on H.264/H.265.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        cache_size: int = 64,
        cache: FrameCache | None = None,
        decode_threads: int = 0,
    ) -> None:
        """Open ``path`` and index its frames; see the class docstring."""
        av = _av()
        self._path = os.fspath(path)
        self._cache_size = max(1, int(cache_size))
        self._cache = cache if cache is not None else FrameCache(max_frames=self._cache_size)
        self._owns_cache = cache is None
        self._lock = threading.RLock()
        try:
            self._container = av.open(self._path)
        except Exception as exc:
            raise VideoError(f"cannot open video {self._path!r}: {exc}") from exc
        if not self._container.streams.video:
            self._container.close()
            raise VideoError(f"{self._path!r} has no video stream")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        if decode_threads:
            self._stream.codec_context.thread_count = int(decode_threads)
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
        """Record every frame's presentation timestamp, in display order.

        Tries the cheap route first (demux packet timestamps, no pixel decode)
        and falls back to decoding the whole clip when that cannot be trusted.

        Returns:
            Presentation timestamps in display order; this list's length is the
            clip's ``frame_count`` and each position is that frame's absolute
            index.
        """
        from_packets = self._index_from_packets()
        if from_packets is not None:
            return from_packets
        return self._index_by_decoding()

    def _index_from_packets(self) -> list[int] | None:
        """Return the pts index read from packet headers, or ``None`` if unsafe.

        A packet carries the presentation timestamp of the frame it holds, so
        demuxing gives the same list the decode pass would — for the ordinary
        one-frame-per-packet case. That assumption is *verified*, not assumed:
        the result is only used when the container's own frame count agrees.

        Returns:
            The timestamps sorted into display order, or ``None`` when the
            container does not give enough information to be sure.
        """
        declared = int(getattr(self._stream, "frames", 0) or 0)
        if declared <= 0:
            return None
        values: list[int] = []
        try:
            for packet in self._container.demux(self._stream):
                if packet.size == 0:  # the flush packet at end of stream
                    continue
                if packet.pts is None:
                    return None
                values.append(int(packet.pts))
                if len(values) > declared:
                    return None
        except Exception as exc:  # pragma: no cover - malformed container
            logger.debug("packet index failed for %r (%s); decoding instead", self._path, exc)
            return None
        finally:
            self._rewind()
        if len(values) != declared:
            return None
        values.sort()
        return values

    def _index_by_decoding(self) -> list[int]:
        """Decode the whole clip once to record every frame's presentation timestamp."""
        pts_list: list[int] = []
        for frame in self._container.decode(self._stream):
            pts_list.append(int(frame.pts) if frame.pts is not None else len(pts_list))
        self._rewind()
        return pts_list

    def _rewind(self) -> None:
        """Put the container back at the start after an index pass."""
        try:
            self._container.seek(0, stream=self._stream, backward=True, any_frame=False)
        except Exception as exc:  # pragma: no cover - malformed container
            logger.debug("could not rewind %r: %s", self._path, exc)

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

    @property
    def cache(self) -> FrameCache:
        """The frame cache this source reads through."""
        return self._cache

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``uint8`` RGB of shape ``(H, W, 3)``.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            The decoded frame, safe for the caller to mutate (it is a copy of
            the cached array). Use :meth:`get_frame_view` to skip that copy when
            the frame will only be read.

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the decoder cannot reach ``index``.
        """
        return self.get_frame_view(index).copy()

    def get_frame_view(self, index: int) -> np.ndarray:
        """Return frame ``index`` as a **read-only** array — no copy.

        The fast path for consumers that only read the pixels (colour
        conversion, resizing, warping all produce a new array anyway). Copying a
        1080p frame costs about 1 ms, which is a fifth of what decoding it costs,
        so skipping it where it is not needed is worth having.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            The cached frame with ``writeable=False``. Callers must not mutate
            it, and must not hold it past the point where the cache may evict
            it if they need it to stay valid (it stays valid — NumPy keeps the
            buffer alive — but it may no longer be the cache's copy).

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the decoder cannot reach ``index``.
        """
        if not 0 <= index < self.frame_count:
            raise IndexError(f"frame {index} out of range for T={self.frame_count}")
        cached = self._cache.get(index)
        if cached is not None:
            note_frame(index)
            return cached
        with self._lock:
            # Another thread may have decoded it while we waited for the lock.
            cached = self._cache.get(index)
            if cached is None:
                if self._decoder is None or index < self._decoder_next_index:
                    self._reseek(index)
                while (cached := self._cache.get(index)) is None:
                    self._decode_one()
        note_frame(index)
        return cached

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

    def iter_frame_views(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Like :meth:`iter_frames`, but yielding read-only frames without copying.

        Args:
            start: First frame index, inclusive.
            stop: Last frame index, exclusive; ``None`` means end of clip.

        Yields:
            ``(index, frame)`` with ``frame`` read-only; see :meth:`get_frame_view`.
        """
        start = max(start, 0)
        stop = self.frame_count if stop is None else min(stop, self.frame_count)
        for index in range(start, stop):
            yield index, self.get_frame_view(index)

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
        self._cache.put(landed_index, first.to_ndarray(format="rgb24"))
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
        # `to_ndarray` allocates a fresh array every call, so the cache can take
        # ownership of it directly -- copying here used to cost a full frame's
        # memcpy per decode for nothing.
        array = self._cache.put(self._decoder_next_index, raw.to_ndarray(format="rgb24"))
        self._decoder_next_index += 1
        return array

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Release the underlying PyAV container. Safe to call more than once."""
        container = getattr(self, "_container", None)
        if container is not None:
            container.close()
            self._container = None
            self._decoder = None
        if getattr(self, "_owns_cache", False):
            self._cache.clear()

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
