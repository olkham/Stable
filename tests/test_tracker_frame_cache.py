"""The converted-frame cache every classic tracker runs on, and its bound.

The reported failure was mean-shift dying with an OpenCV ``OutOfMemoryError`` on
an ordinary HD clip. The cause was not mean-shift's arithmetic but the plain
``dict`` it (and five other plugins) used to memoise colour-converted frames:
nothing evicted, so a run held a converted copy of the entire clip.

These tests pin both halves of the fix — the cache really is bounded, and
bounding it does not change what a tracker produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.video import SyntheticVideoSource
from stable.trackers.frames import ENV_TRACKER_CACHE_MB, converted_frames, tracker_cache_bytes
from stable.trackers.registry import create_tracker

# The plugins that keep a converted-frame cache, and how many bytes per pixel
# their conversion costs — the reason mean-shift was the first to fall over.
CACHING_TRACKERS = ["lk", "klt", "template", "correlation", "kalman", "particle", "meanshift"]


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """A stand-in conversion with no OpenCV dependency."""
    return rgb.mean(axis=2).astype(np.uint8)


def test_the_cache_stops_growing_at_its_budget():
    """The whole bug in one assertion: reading a long clip must not hold it all."""
    video = SyntheticVideoSource(frame_count=500, width=160, height=120)
    frame_bytes = 160 * 120  # one grayscale frame
    budget = frame_bytes * 8

    read = converted_frames(video, _to_gray, max_bytes=budget)
    for index in range(video.frame_count):
        read(index)

    assert read.cache.nbytes <= budget
    assert len(read.cache) <= 8
    # ...and the unbounded version would have held all 500.
    assert len(read.cache) < video.frame_count


def test_an_evicted_frame_is_recomputed_not_lost():
    video = SyntheticVideoSource(frame_count=40, width=64, height=48)
    read = converted_frames(video, _to_gray, max_bytes=64 * 48 * 2)

    first = np.array(read(0))
    for index in range(1, 40):  # push frame 0 out
        read(index)
    assert np.array_equal(read(0), first)


def test_a_cached_frame_is_returned_by_reference_and_read_only():
    """A hit must cost nothing, so it is the cache's array, not a copy of it."""
    video = SyntheticVideoSource(frame_count=4, width=32, height=24)
    read = converted_frames(video, _to_gray)

    first, second = read(2), read(2)
    assert first is second
    assert not first.flags.writeable
    assert read.cache.hits == 1


def test_the_budget_is_configurable_and_bounded_below(monkeypatch):
    monkeypatch.setenv(ENV_TRACKER_CACHE_MB, "8")
    assert tracker_cache_bytes() == 8 * 1024 * 1024
    monkeypatch.setenv(ENV_TRACKER_CACHE_MB, "0")
    assert tracker_cache_bytes() == 1024 * 1024
    monkeypatch.delenv(ENV_TRACKER_CACHE_MB)
    assert tracker_cache_bytes() > 0


@pytest.mark.parametrize("tracker_name", CACHING_TRACKERS)
def test_eviction_never_changes_what_a_tracker_produces(tracker_name, monkeypatch):
    """The bound is only safe if it is invisible in the output.

    Run each plugin twice over the same clip — once with room for the whole clip
    and once with room for a single frame, which forces an eviction on every
    step — and require the two tracks to be identical, not merely close.
    """
    from stable.core.types import Query

    video = SyntheticVideoSource(frame_count=30, width=96, height=72)
    query = [Query(frame=0, x=video.width / 2 + 8.0, y=video.height / 2)]

    def run(budget: int):
        monkeypatch.setattr("stable.trackers.frames.tracker_cache_bytes", lambda: budget)
        plugin = create_tracker(tracker_name)
        try:
            return plugin.track(video, query, (0, video.frame_count), "bidir")
        finally:
            plugin.release()

    roomy = run(96 * 72 * 3 * 64)
    cramped = run(1)  # one frame kept, every other read a miss

    assert np.array_equal(
        np.nan_to_num(roomy.points, nan=-1), np.nan_to_num(cramped.points, nan=-1)
    )
    assert np.array_equal(roomy.confidence, cramped.confidence)
    assert np.array_equal(roomy.visible, cramped.visible)


def test_a_long_hd_run_stays_inside_the_budget(monkeypatch):
    """The reported case: HD, hundreds of frames, the tracker with the fattest pixels.

    Before the fix this run held ``frames x 6.2 MB`` of HSV — gigabytes, and the
    reason an ordinary 24 MB allocation failed underneath it.
    """
    from stable.core.types import Query

    caches = []
    real_cache = converted_frames.__globals__["FrameCache"]

    class Watched(real_cache):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            caches.append(self)

    monkeypatch.setattr("stable.trackers.frames.FrameCache", Watched)
    budget = 16 * 1024 * 1024
    monkeypatch.setattr("stable.trackers.frames.tracker_cache_bytes", lambda: budget)

    video = SyntheticVideoSource(frame_count=300, width=1920, height=1080, radius=200.0)
    plugin = create_tracker("meanshift")
    try:
        plugin.track(video, [Query(frame=0, x=1160.0, y=540.0)], (0, 120), "fwd")
    finally:
        plugin.release()

    assert caches, "the tracker should have built a converted-frame cache"
    held = max(cache.nbytes for cache in caches)
    assert held <= budget, (
        f"cache grew to {held / 1e6:.0f} MB against a {budget / 1e6:.0f} MB bound"
    )
    # 120 HD HSV frames would have been ~750 MB in the dict this replaced.
    assert held < 120 * 1920 * 1080 * 3
