"""Concurrent decode of one clip: several decoders, one shared frame cache.

The single-decoder :class:`~stable.video.source.PyAVVideoSource` is the right
primitive and the wrong *object* to hand to a server. Two things go wrong there:

#. **Every pass re-decodes.** A multi-anchor track plans two runs per
   inter-anchor span; three points with three anchors each is twelve runs over
   the same footage. Measured on a 240-frame 1080p clip, that was 1355 decodes of
   240 distinct frames — 5.6x redundant, and decode was ~47% of the wall time.
#. **Decode blocks compute.** A frame-stepping tracker alternates "decode one
   frame" with "run optical flow on it". Both are real work and neither overlaps
   the other, so a 32-core machine sits mostly idle (the "barely uses 25% of the
   CPU" complaint) while the two take turns.

:class:`PooledVideoSource` fixes both without any tracker knowing. It owns a
byte-budgeted :class:`~stable.video.cache.FrameCache` that *all* of its decoders
write into, so the second pass over a span is free; and it watches the access
pattern, so when a consumer walks frames in order a background thread decodes
ahead of it into that same cache. The tracker's next ``get_frame`` then returns
immediately, and decode has happened *while* optical flow was running rather
than between two runs of it.

It also makes concurrent access safe and fast: each caller gets its own decoder
out of the pool, so two tracking jobs running in parallel do not drag one
decoder's cursor back and forth across the clip.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from ..core.errors import VideoError
from .cache import FrameCache, default_cache_bytes
from .source import PyAVVideoSource
from .telemetry import note_frame

__all__ = ["PooledVideoSource", "default_decoder_count"]

logger = logging.getLogger(__name__)

_MAX_DECODERS = 4
"""Ceiling on decoders per clip.

Each one holds its own FFmpeg decode context and reference frames, so this is
memory, not just threads. Four is enough to keep a read-ahead thread and a
couple of parallel tracking jobs from queueing behind each other; past that the
shared cache is doing the work anyway.
"""

DEFAULT_READAHEAD = 32
"""How many frames ahead of a sequential reader the prefetcher tries to stay."""


def default_decoder_count() -> int:
    """Return how many decoders to open per clip on this machine.

    Returns:
        Between 2 and :data:`_MAX_DECODERS`, scaled off the CPU count.
    """
    cpus = os.cpu_count() or 1
    return max(2, min(_MAX_DECODERS, cpus // 4 or 1))


class PooledVideoSource:
    """A thread-safe :class:`~stable.core.video.VideoSource` with read-ahead.

    Args:
        path: Path to a video file FFmpeg can demux and decode.
        max_decoders: Maximum concurrent decoders; ``None`` uses
            :func:`default_decoder_count`.
        cache_bytes: Budget for the shared decoded-frame cache; ``None`` uses
            :func:`~stable.video.cache.default_cache_bytes`.
        readahead: How many frames ahead of a sequential reader to decode. ``0``
            disables read-ahead (the cache and the decoder pool still apply).

    Raises:
        VideoError: If the file cannot be opened (raised from the first decoder).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_decoders: int | None = None,
        cache_bytes: int | None = None,
        readahead: int = DEFAULT_READAHEAD,
    ) -> None:
        """Open the clip and start with a single decoder; see the class docstring."""
        self._path = os.fspath(path)
        self._max_decoders = max(1, int(max_decoders or default_decoder_count()))
        self._cache = FrameCache(max_bytes=cache_bytes or default_cache_bytes())
        self._readahead = max(0, int(readahead))

        first = self._new_decoder()
        self._frame_count = first.frame_count
        self._width = first.width
        self._height = first.height
        self._fps = first.fps

        self._idle: list[PyAVVideoSource] = [first]
        self._created = 1
        self._available = threading.Condition(threading.Lock())
        self._closed = False

        self._cursors: dict[int, int] = {}
        self._prefetch_lock = threading.Lock()
        self._frontier: int | None = None
        self._prefetch_pending = 0
        self._executor: ThreadPoolExecutor | None = None

    def _new_decoder(self) -> PyAVVideoSource:
        """Open one more decoder over the same file, sharing the frame cache."""
        return PyAVVideoSource(self._path, cache=self._cache, cache_size=self._frame_cap())

    def _frame_cap(self) -> int:
        """Frame-count cap for the shared cache (bytes are the real bound)."""
        return 1 << 30

    # -- VideoSource protocol ----------------------------------------------

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

    @property
    def cache(self) -> FrameCache:
        """The shared decoded-frame cache every decoder writes into."""
        return self._cache

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as ``uint8`` RGB ``(H, W, 3)``, safe to mutate.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            A private copy of the frame.

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the source is closed or the frame cannot be decoded.
        """
        return self.get_frame_view(index).copy()

    def get_frame_view(self, index: int) -> np.ndarray:
        """Return frame ``index`` as a **read-only** array, without copying.

        Args:
            index: Absolute frame index in ``[0, frame_count)``.

        Returns:
            The cached frame with ``writeable=False``; see
            :meth:`stable.video.source.PyAVVideoSource.get_frame_view`.

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the source is closed or the frame cannot be decoded.
        """
        if not 0 <= index < self._frame_count:
            raise IndexError(f"frame {index} out of range for T={self._frame_count}")
        self._note_cursor(index)
        cached = self._cache.get(index)
        if cached is not None:
            note_frame(index)
            return cached
        with self._borrow() as decoder:
            frame = decoder.get_frame_view(index)
        return frame

    def iter_frames(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` pairs over ``[start, stop)`` in order."""
        start = max(start, 0)
        stop = self._frame_count if stop is None else min(stop, self._frame_count)
        for index in range(start, stop):
            yield index, self.get_frame(index)

    def iter_frame_views(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Like :meth:`iter_frames` but yielding read-only frames without copying."""
        start = max(start, 0)
        stop = self._frame_count if stop is None else min(stop, self._frame_count)
        for index in range(start, stop):
            yield index, self.get_frame_view(index)

    # -- decoder pool --------------------------------------------------------

    @contextmanager
    def _borrow(self) -> Iterator[PyAVVideoSource]:
        """Lend one decoder to the calling thread for the duration of the block.

        Decoders are handed out and returned **LIFO**, so a thread doing
        sequential work tends to get back the decoder it just used — already
        positioned at the next frame, needing no seek.

        Raises:
            VideoError: If the source has been closed.
        """
        decoder = self._acquire()
        try:
            yield decoder
        finally:
            self._release(decoder)

    def _acquire(self) -> PyAVVideoSource:
        """Take a decoder from the pool, growing it or waiting as needed."""
        with self._available:
            while True:
                if self._closed:
                    raise VideoError(f"video source for {self._path!r} is closed")
                if self._idle:
                    return self._idle.pop()
                if self._created < self._max_decoders:
                    self._created += 1
                    break
                self._available.wait()
        try:
            return self._new_decoder()
        except Exception:
            with self._available:
                self._created -= 1
                self._available.notify()
            raise

    def _release(self, decoder: PyAVVideoSource) -> None:
        """Give a decoder back to the pool (or close it if the source shut down)."""
        with self._available:
            if self._closed:
                decoder.close()
                self._created -= 1
            else:
                self._idle.append(decoder)
            self._available.notify()

    # -- read-ahead ----------------------------------------------------------

    def _note_cursor(self, index: int) -> None:
        """Record this thread's position and start read-ahead if it is scanning.

        Read-ahead only starts once a thread has actually stepped to an adjacent
        frame, so a UI serving scattered single frames never triggers it, and a
        tracker walking a run triggers it on its second frame.
        """
        if self._readahead <= 0 or self._closed:
            return
        thread_id = threading.get_ident()
        previous = self._cursors.get(thread_id)
        if len(self._cursors) > 64:  # pragma: no cover - only with thread churn
            self._cursors.clear()
        self._cursors[thread_id] = index
        if previous is None or abs(index - previous) != 1:
            return
        self._request_readahead(index, 1 if index > previous else -1)

    def _request_readahead(self, index: int, direction: int) -> None:
        """Queue a decode of the frames just past ``index``, if none is running."""
        with self._prefetch_lock:
            if self._prefetch_pending > 0:
                frontier = self._frontier
                # Still comfortably ahead of the reader: nothing to do.
                if frontier is not None and (frontier - index) * direction > self._readahead // 2:
                    return
                if self._prefetch_pending >= 2:
                    return
            start = index + direction
            end = index + direction * (self._readahead + 1)
            start = max(0, min(self._frame_count - 1, start))
            end = max(-1, min(self._frame_count, end))
            if start == end or (end - start) * direction <= 0:
                return
            self._frontier = end - direction
            self._prefetch_pending += 1
            executor = self._ensure_executor()
        future: Future[None] = executor.submit(self._prefetch_span, start, end, direction)
        future.add_done_callback(self._prefetch_finished)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Create the read-ahead thread pool on first use (caller holds the lock)."""
        if self._executor is None:
            # One read-ahead thread is enough: decoding runs at hundreds of
            # frames a second and the consumers it feeds run at tens. More would
            # only compete with the foreground reader for decoders.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="stable-readahead"
            )
        return self._executor

    def _prefetch_finished(self, _future: Future[None]) -> None:
        """Mark one read-ahead span as complete."""
        with self._prefetch_lock:
            self._prefetch_pending = max(0, self._prefetch_pending - 1)

    def _prefetch_span(self, start: int, end: int, direction: int) -> None:
        """Decode ``[start, end)`` in ``direction`` into the shared cache.

        Runs on a background thread. Failures are swallowed: a read-ahead that
        cannot keep up (or races a ``close()``) must never break the foreground
        read that follows it, which will simply decode the frame itself.
        """
        if self._closed:
            return
        # Decode ascending whatever the run direction: FFmpeg is sequential, so a
        # backward span costs one seek this way and one seek *per frame* the
        # other way.
        low, high = (start, end) if direction > 0 else (end + 1, start + 1)
        try:
            with self._borrow() as decoder:
                for frame_index in range(low, high):
                    if self._closed:
                        return
                    if frame_index in self._cache:
                        continue
                    decoder.get_frame_view(frame_index)
        except Exception as exc:  # noqa: BLE001 - read-ahead is best effort
            logger.debug("read-ahead for %r stopped: %s", self._path, exc)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Stop read-ahead, close every decoder and drop the cache. Idempotent."""
        with self._available:
            if self._closed:
                return
            self._closed = True
            idle, self._idle = self._idle, []
            self._created -= len(idle)
            self._available.notify_all()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False)
        for decoder in idle:
            decoder.close()
        self._cache.clear()
        self._cursors.clear()

    def __enter__(self) -> PooledVideoSource:
        """Support use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close every decoder on context-manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Return a short debug representation naming the path, size and pool."""
        return (
            f"<PooledVideoSource path={self._path!r} {self._width}x{self._height} "
            f"T={self._frame_count} decoders={self._created}/{self._max_decoders} "
            f"cache={self._cache!r}>"
        )
