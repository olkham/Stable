"""A byte-budgeted, thread-safe LRU cache of decoded frames.

Frame caching is the single highest-leverage thing this project can do with a
video file, because *every* consumer reads the same frames repeatedly: a
multi-anchor track plans two runs per inter-anchor span and each run walks the
same footage again, proxy generation reads it once more, and the stabilised
preview reads it a fourth time.

Counting frames is the wrong budget for that. A 128x96 test clip and a 4K clip
differ by a factor of ~500 in bytes per frame, so any single frame count is
either uselessly small for the first or an out-of-memory bug for the second.
This cache is therefore sized in **bytes**, with an optional count cap on top,
and eviction is plain LRU.

Cached arrays are stored read-only (``setflags(write=False)``) and handed out
by reference from :meth:`FrameCache.get`. That is what makes a cache hit free
rather than a 6 MB ``memcpy`` — callers that need to mutate a frame copy it
themselves, which is what :meth:`stable.video.source.PyAVVideoSource.get_frame`
does on their behalf.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

__all__ = ["FrameCache", "default_cache_bytes", "ENV_CACHE_MB"]

ENV_CACHE_MB = "STABLE_FRAME_CACHE_MB"
"""Environment variable overriding the decoded-frame cache budget, in MiB."""

_DEFAULT_CACHE_MB = 768
"""Frame cache budget when nothing overrides it.

Enough to hold ~120 frames of 1080p (or a whole clip at proxy resolution)
without being the reason a laptop starts swapping. Raise it with
:data:`ENV_CACHE_MB` on a machine with memory to spare — the tracking pipeline
gets faster in direct proportion to how much of the clip stays resident.
"""


def default_cache_bytes() -> int:
    """Return the default frame-cache budget in bytes.

    Returns:
        ``$STABLE_FRAME_CACHE_MB`` MiB when that variable is set to a positive
        integer, else :data:`_DEFAULT_CACHE_MB` MiB.
    """
    raw = os.environ.get(ENV_CACHE_MB)
    if raw:
        try:
            megabytes = int(raw)
        except ValueError:
            megabytes = _DEFAULT_CACHE_MB
        else:
            megabytes = max(1, megabytes)
    else:
        megabytes = _DEFAULT_CACHE_MB
    return megabytes * 1024 * 1024


class FrameCache:
    """Thread-safe LRU cache of decoded frames, bounded by total bytes.

    Args:
        max_bytes: Total budget for cached frame data. Values below one frame's
            size still keep a single frame, so the cache is never useless.
        max_frames: Optional hard cap on the number of frames, applied in
            addition to the byte budget. ``None`` means bytes are the only bound.
    """

    def __init__(self, max_bytes: int | None = None, max_frames: int | None = None) -> None:
        """Create an empty cache; see the class docstring for the arguments."""
        self.max_bytes = int(max_bytes if max_bytes is not None else default_cache_bytes())
        self.max_frames = None if max_frames is None else max(1, int(max_frames))
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, index: int) -> np.ndarray | None:
        """Return the cached frame for ``index``, or ``None`` on a miss.

        Args:
            index: Absolute frame index.

        Returns:
            The cached array, **read-only and by reference** — do not mutate it.
        """
        with self._lock:
            entry = self._entries.get(index)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(index)
            self.hits += 1
            return entry

    def put(self, index: int, array: np.ndarray) -> np.ndarray:
        """Insert a frame, evicting least-recently-used entries to stay in budget.

        The array is marked read-only in place — the cache takes ownership. Every
        caller in this package passes a freshly decoded array, so no copy is made.

        Args:
            index: Absolute frame index.
            array: The decoded frame.

        Returns:
            The stored (read-only) array, so callers can chain.
        """
        array.setflags(write=False)
        with self._lock:
            existing = self._entries.pop(index, None)
            if existing is not None:
                self._bytes -= existing.nbytes
            self._entries[index] = array
            self._bytes += array.nbytes
            self._evict_locked()
        return array

    def _evict_locked(self) -> None:
        """Drop oldest entries until both bounds are satisfied (caller holds the lock)."""
        while len(self._entries) > 1 and (
            self._bytes > self.max_bytes
            or (self.max_frames is not None and len(self._entries) > self.max_frames)
        ):
            _index, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes

    def __contains__(self, index: object) -> bool:
        """Whether a frame is cached, without disturbing its LRU position."""
        with self._lock:
            return index in self._entries

    def __len__(self) -> int:
        """Number of frames currently cached."""
        with self._lock:
            return len(self._entries)

    @property
    def nbytes(self) -> int:
        """Total bytes of frame data currently held."""
        with self._lock:
            return self._bytes

    @property
    def hit_rate(self) -> float:
        """Fraction of :meth:`get` calls that were hits (``0.0`` before any)."""
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total else 0.0

    def clear(self) -> None:
        """Drop every cached frame and reset the byte count (not the counters)."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def __repr__(self) -> str:
        """Return a short debug representation of size and hit rate."""
        return (
            f"<FrameCache {len(self)} frames / {self.nbytes / 1e6:.1f} MB "
            f"of {self.max_bytes / 1e6:.1f} MB, hit rate {self.hit_rate:.0%}>"
        )
