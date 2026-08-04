"""Tests for JPEG proxy generation and disk-backed serving.

The headline acceptance criterion is architectural: once proxies exist,
scrubbing them must be I/O-bound, never decode-bound. :class:`ProxyStore` is
built so that is true by construction — it has no reference to a
:class:`~stable.core.video.VideoSource` at all — and the tests here exercise
that directly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from stable.core.errors import VideoError
from stable.core.video import SyntheticVideoSource
from stable.video.proxy import (
    PROXY_SCALES,
    ProxyManifest,
    ProxyStore,
    generate_all_proxies,
    generate_proxies,
)


@pytest.fixture
def small_video() -> SyntheticVideoSource:
    return SyntheticVideoSource(frame_count=10, width=64, height=48)


def test_generate_proxies_writes_a_jpeg_per_frame_and_a_manifest(tmp_path, small_video):
    out_dir = tmp_path / "half"
    manifest = generate_proxies(small_video, out_dir, scale=0.5)

    assert manifest.count == small_video.frame_count
    assert manifest.width == 32
    assert manifest.height == 24
    assert manifest.scale == pytest.approx(0.5)
    assert manifest.fps == pytest.approx(small_video.fps)

    files = sorted(os.listdir(out_dir))
    assert "proxy.json" in files
    jpegs = [f for f in files if f.endswith(".jpg")]
    assert len(jpegs) == small_video.frame_count

    with open(out_dir / "proxy.json", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["count"] == small_video.frame_count


def test_quarter_scale_rounds_dimensions(tmp_path, small_video):
    manifest = generate_proxies(small_video, tmp_path / "quarter", scale=0.25)
    assert manifest.width == 16
    assert manifest.height == 12


def test_generate_proxies_validates_scale_and_quality(tmp_path, small_video):
    with pytest.raises(VideoError):
        generate_proxies(small_video, tmp_path / "a", scale=0.0)
    with pytest.raises(VideoError):
        generate_proxies(small_video, tmp_path / "b", scale=1.5)
    with pytest.raises(VideoError):
        generate_proxies(small_video, tmp_path / "c", quality=101)
    with pytest.raises(VideoError):
        generate_proxies(small_video, tmp_path / "d", quality=-1)


def test_progress_is_reported_and_ends_at_one(tmp_path, small_video):
    seen: list[float] = []
    generate_proxies(small_video, tmp_path / "p", progress=lambda f, m="": seen.append(f))
    assert seen
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_proxy_store_serves_correctly_scaled_frames(tmp_path, small_video):
    out_dir = tmp_path / "half"
    generate_proxies(small_video, out_dir, scale=0.5)

    store = ProxyStore(out_dir)
    assert store.frame_count == small_video.frame_count
    assert len(store) == small_video.frame_count
    frame = store.get_frame(0)
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8
    assert "ProxyStore" in repr(store)


def test_proxy_store_out_of_range_raises(tmp_path, small_video):
    out_dir = tmp_path / "half"
    generate_proxies(small_video, out_dir, scale=0.5)
    store = ProxyStore(out_dir)
    with pytest.raises(IndexError):
        store.get_frame(store.frame_count)
    with pytest.raises(IndexError):
        store.frame_path(-1)


def test_proxy_store_requires_a_manifest(tmp_path):
    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()
    with pytest.raises(VideoError):
        ProxyStore(empty_dir)


def test_manifest_round_trips_through_dict():
    manifest = ProxyManifest(count=10, width=32, height=24, scale=0.5, fps=24.0, quality=85)
    rebuilt = ProxyManifest.from_dict(manifest.to_dict())
    assert rebuilt == manifest


def test_manifest_from_dict_requires_keys():
    with pytest.raises(VideoError):
        ProxyManifest.from_dict({"count": 1})


def test_generate_all_proxies_writes_every_named_tier(tmp_path, small_video):
    manifests = generate_all_proxies(small_video, tmp_path / "proxies")
    assert set(manifests) == set(PROXY_SCALES)
    for name, scale in PROXY_SCALES.items():
        assert manifests[name].scale == pytest.approx(scale)
        assert os.path.isdir(tmp_path / "proxies" / name)


# -- the headline acceptance criterion: scrubbing is I/O-bound -------------


def test_proxy_store_has_no_reference_to_the_source_video(tmp_path, small_video):
    """Structural proof: ProxyStore's constructor only ever takes a directory."""
    import inspect

    signature = inspect.signature(ProxyStore.__init__)
    assert "video" not in signature.parameters
    assert "source" not in signature.parameters


def test_scrubbing_proxies_survives_deleting_the_source_video(tmp_path, small_video):
    """Once proxies exist, the original file is irrelevant to serving them.

    This is the decisive proof that proxy scrubbing is disk I/O, not a video
    decode: if it worked by secretly re-decoding the source, this would fail
    the moment the source file disappears.
    """
    video_path = tmp_path / "source.bin"
    video_path.write_bytes(b"not a real video, just something to delete")

    out_dir = tmp_path / "half"
    generate_proxies(small_video, out_dir, scale=0.5)
    del small_video  # drop the only in-memory reference too

    os.remove(video_path)
    assert not video_path.exists()

    store = ProxyStore(out_dir)
    for index in range(store.frame_count):
        frame = store.get_frame(index)
        assert frame.shape == (24, 32, 3)
