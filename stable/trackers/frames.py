"""Bounded per-run caches of colour-converted frames.

Every classic-CV tracker here works on a conversion of the frame rather than the
frame itself — grayscale for ``lk``, ``template``, ``correlation``, ``kalman``
and ``particle``, HSV for ``meanshift`` — and every one of them used to keep its
own ``dict`` of those conversions, keyed by frame index, with nothing evicting.

That is a leak with a very specific shape. A run walks the clip, so the dict
grows to hold *the whole clip*: at 1920x1080 a grayscale frame is 2.07 MB and an
HSV frame is 6.2 MB, so a thousand-frame shot costs 2 GB or 6 GB of cache that
is never read again. Mean-shift is simply the one that dies first, which is why
it was reported as a mean-shift bug::

    cv::OutOfMemoryError: Failed to allocate 24883200 bytes

(24,883,200 = 1920 x 1080 x 12 — an ordinary allocation, failing only because
the cache had already taken everything.)

The fix is not a new cache. :class:`stable.video.cache.FrameCache` is already a
byte-budgeted, thread-safe LRU built for exactly this, and reusing it means one
eviction policy for the whole application rather than a second one that drifts.
This module is the small amount of glue that points it at a conversion:
:func:`converted_frames` returns the ``gray(index)`` / ``hsv(index)`` callable
each tracker already had, backed by a bounded cache instead of a bare dict.

Why a *separate* cache from the decoder's
-----------------------------------------
The two hold different pixels — the decoder caches RGB, this caches the
conversion — and they are additive, so this one gets its own, smaller budget.
Sequential access means a small window is all a run ever re-reads:

* each direction of a run reads each frame once,
* the forward-backward check in ``lk`` re-reads the frame it just left,
* several queries in one ``track()`` call walk the same span again.

The first two need a handful of frames; the third is what the budget is for.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from ..video.cache import FrameCache, default_cache_bytes

__all__ = ["ENV_TRACKER_CACHE_MB", "converted_frames", "tracker_cache_bytes"]

ENV_TRACKER_CACHE_MB = "STABLE_TRACKER_CACHE_MB"
"""Environment variable overriding the converted-frame budget, in MiB."""

_DEFAULT_FRACTION = 0.25
"""Share of the decoded-frame budget a tracker's conversions may also use.

A quarter of the 768 MB default is ~192 MB: about 30 seconds of 1080p HSV, or
ten times that at proxy resolution — far more than the sequential access pattern
needs, and bounded, which is the whole point.
"""


def tracker_cache_bytes() -> int:
    """Return the byte budget for one tracker's converted-frame cache.

    Returns:
        ``$STABLE_TRACKER_CACHE_MB`` MiB when set to a positive integer, else a
        quarter of :func:`stable.video.cache.default_cache_bytes`, so raising
        the decoder's budget raises this with it.
    """
    raw = os.environ.get(ENV_TRACKER_CACHE_MB)
    if raw:
        try:
            return max(1, int(raw)) * 1024 * 1024
        except ValueError:  # pragma: no cover - a typo'd env var falls through
            pass
    return int(default_cache_bytes() * _DEFAULT_FRACTION)


def _read(video: Any, index: int) -> np.ndarray:
    """Return frame ``index`` without copying it where the source allows.

    ``VideoSource`` only promises ``get_frame`` (which copies, so callers may
    mutate). Every caller here converts the frame into a *new* array
    immediately, so the copy is pure cost — taken only when the source has no
    view to offer.
    """
    view = getattr(video, "get_frame_view", None)
    return view(index) if callable(view) else video.get_frame(index)


def converted_frames(
    video: Any,
    convert: Callable[[np.ndarray], np.ndarray],
    *,
    max_bytes: int | None = None,
) -> Callable[[int], np.ndarray]:
    """Return a cached ``f(index) -> converted frame`` reader for one run.

    Args:
        video: The clip being tracked; only ``get_frame`` (or ``get_frame_view``)
            is used.
        convert: Turns an RGB frame into whatever this tracker matches on, e.g.
            ``lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)``.
        max_bytes: Cache budget; defaults to :func:`tracker_cache_bytes`.

    Returns:
        A callable returning the converted frame for an index. Results are
        **read-only** (the cache takes ownership, as it does for decoded
        frames): callers that need to modify one must copy it, which every
        caller in this package already does via ``astype``.

        The callable exposes the cache as ``.cache`` so a caller — or a test
        asserting the bound actually holds — can read ``nbytes``/``hit_rate``.
    """
    cache = FrameCache(max_bytes=max_bytes if max_bytes is not None else tracker_cache_bytes())

    def read(index: int) -> np.ndarray:
        cached = cache.get(index)
        if cached is None:
            cached = cache.put(index, np.ascontiguousarray(convert(_read(video, index))))
        return cached

    read.cache = cache  # type: ignore[attr-defined]
    return read
