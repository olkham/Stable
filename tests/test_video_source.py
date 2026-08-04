"""Contract tests for PyAVVideoSource: the real, file-backed VideoSource."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import VideoError
from stable.core.video import VideoSource
from stable.video.source import PyAVVideoSource


def test_satisfies_the_video_source_protocol(tiny_clip):
    path, source = tiny_clip
    with PyAVVideoSource(path) as video:
        assert isinstance(video, VideoSource)
        assert video.frame_count == source.frame_count
        assert video.width == source.width
        assert video.height == source.height
        assert video.fps == pytest.approx(source.fps, rel=0.05)


def test_frames_are_uint8_rgb_of_the_right_shape(tiny_clip):
    path, source = tiny_clip
    with PyAVVideoSource(path) as video:
        frame = video.get_frame(0)
        assert frame.shape == (source.height, source.width, 3)
        assert frame.dtype == np.uint8


def test_out_of_range_index_raises(tiny_clip):
    path, _source = tiny_clip
    with PyAVVideoSource(path) as video:
        with pytest.raises(IndexError):
            video.get_frame(video.frame_count)
        with pytest.raises(IndexError):
            video.get_frame(-1)


def test_iter_frames_yields_every_index_in_order(tiny_clip):
    path, _source = tiny_clip
    with PyAVVideoSource(path) as video:
        indices = [i for i, _frame in video.iter_frames()]
        assert indices == list(range(video.frame_count))


def test_iter_frames_respects_start_and_stop(tiny_clip):
    path, _source = tiny_clip
    with PyAVVideoSource(path) as video:
        indices = [i for i, _frame in video.iter_frames(5, 10)]
        assert indices == [5, 6, 7, 8, 9]


def test_random_access_matches_sequential_decode(tiny_clip):
    """Jumping around must return the same pixels as decoding straight through."""
    path, _source = tiny_clip
    with PyAVVideoSource(path) as sequential, PyAVVideoSource(path) as random_access:
        sequential_frames = dict(sequential.iter_frames())
        # Visit out of order, including backward jumps, on a second instance.
        order = [10, 2, 25, 25, 0, 15, 3, 29]
        for index in order:
            if index >= random_access.frame_count:
                continue
            assert np.array_equal(random_access.get_frame(index), sequential_frames[index])


def test_random_access_is_correct_even_with_a_tiny_cache(tiny_clip):
    """Heavy eviction pressure (cache_size=2) must not break correctness.

    This forces almost every access to reseek and re-decode, exercising the
    keyframe-seek + forward-decode path far more than the default cache size
    (which never evicts anything for a clip this short) would.
    """
    path, _source = tiny_clip
    with PyAVVideoSource(path, cache_size=2) as tiny_cache, PyAVVideoSource(path) as reference:
        reference_frames = dict(reference.iter_frames())
        order = [20, 1, 29, 0, 14, 14, 5, 27, 2]
        for index in order:
            if index >= tiny_cache.frame_count:
                continue
            assert np.array_equal(tiny_cache.get_frame(index), reference_frames[index])


def test_get_frame_caches_and_returns_independent_copies(tiny_clip):
    path, _source = tiny_clip
    with PyAVVideoSource(path) as video:
        first = video.get_frame(3)
        first[:] = 0  # mutate the caller's copy
        second = video.get_frame(3)
        assert not np.array_equal(second, first)  # cache was untouched
        assert second.flags.writeable  # the returned copy is safe to mutate


def test_tracking_through_a_real_decoded_clip_recovers_ground_truth(tiny_clip):
    """A real encode/decode round trip should not perturb tracking meaningfully."""
    from stable.core.types import Query
    from stable.trackers.lk import LKTracker

    path, source = tiny_clip
    with PyAVVideoSource(path) as video:
        anchor_frame = 5
        query = Query(anchor_frame, *source.true_position(anchor_frame))
        result = LKTracker().track(video, [query], (0, video.frame_count), "bidir")

    for frame in range(source.frame_count):
        px, py = result.points[0, frame]
        if not np.isfinite(px):
            continue
        tx, ty = source.true_position(frame)
        assert (px - tx) ** 2 + (py - ty) ** 2 < 4.0**2  # within a few pixels


def test_missing_file_raises_video_error(tmp_path):
    with pytest.raises(VideoError):
        PyAVVideoSource(tmp_path / "does_not_exist.mp4")


def test_repr_is_informative(tiny_clip):
    path, _source = tiny_clip
    with PyAVVideoSource(path) as video:
        text = repr(video)
        assert "PyAVVideoSource" in text
        assert str(video.frame_count) in text
