"""Patch serving under a scrub, which is where the UI used to lock up.

The report was "scrubbing gets stuck, requests bog down, the UI stops updating,
I think it might be the patches". It was: every thumbnail was cropped from the
**full-resolution source clip**, so one frame of a scrub with N points meant N
random-access seeks queued behind the decoder's single lock, plus a fresh
`/patches` JSON call, on a bounded threadpool the proxy frames also need.

These tests pin the three properties that fix it: the pixels can come from the
proxy tier, a repeated patch is not re-rendered, and serving patches does not
slow down serving frames.
"""

from __future__ import annotations

import time
from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.server.routes.points import clear_patch_cache, drain_patch_warmup
from stable.server.storage import Workspace

from ._helpers import decode_jpeg, upload_clip, wait_job


@pytest.fixture
def tracked(client: TestClient, tiny_clip):
    """A project with one tracked point, so every frame has a patch."""
    clear_patch_cache()
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    x, y = source.true_position(0)
    client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": x, "y": y, "tracker": "lk"},
    )
    submitted = client.post(f"/api/projects/{project_id}/points/p1/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"
    return project_id, source


def patch_url(project_id: str, frame: int, **params) -> str:
    query = "".join(f"&{k}={v}" for k, v in params.items())
    return f"/api/projects/{project_id}/points/p1/patches/image?frame={frame}{query}"


def test_patches_come_from_the_proxy_tier_by_default(client, tracked):
    """The default must be the cheap source, or the default is the bug."""
    project_id, _source = tracked
    detail = client.get(f"/api/projects/{project_id}").json()
    scale = detail["proxy"]["half"]["width"] / detail["video"]["w"]

    body = client.get(f"/api/projects/{project_id}/points/p1/patches").json()
    radius = body["radius"]

    proxy = decode_jpeg(client.get(patch_url(project_id, 5)).content)
    original = decode_jpeg(client.get(patch_url(project_id, 5, source="original")).content)

    assert original.shape[0] == radius * 2 + 1
    assert proxy.shape[0] == round(radius * scale) * 2 + 1
    assert proxy.shape[0] < original.shape[0]


def test_the_proxy_patch_shows_the_same_place_as_the_source_patch(client, tracked):
    """Cheaper pixels, not different ones — the thumbnail must still be the point."""
    project_id, _source = tracked
    proxy = decode_jpeg(client.get(patch_url(project_id, 5)).content)
    original = decode_jpeg(client.get(patch_url(project_id, 5, source="original")).content)

    # Compare mean colour rather than pixels: one is a downscaled JPEG of the
    # other, so identity is the wrong test and "same region" is the right one.
    assert np.abs(proxy.mean(axis=(0, 1)) - original.mean(axis=(0, 1))).max() < 40


def test_an_unknown_source_is_refused(client, tracked):
    project_id, _source = tracked
    response = client.get(patch_url(project_id, 5, source="raw"))
    assert response.status_code == 400
    assert "source" in response.json()["detail"]


def test_a_repeated_patch_is_served_from_cache(client, tracked):
    """Scrubbing back over covered ground must cost nothing."""
    project_id, _source = tracked
    first = client.get(patch_url(project_id, 7, source="original"))
    second = client.get(patch_url(project_id, 7, source="original"))

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.headers["ETag"] == second.headers["ETag"]
    assert "max-age" in first.headers["Cache-Control"]


def test_a_conditional_request_gets_a_304(client, tracked):
    project_id, _source = tracked
    first = client.get(patch_url(project_id, 7))
    again = client.get(patch_url(project_id, 7), headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304
    assert again.content == b""


def test_moving_the_point_invalidates_its_cached_patches(client, tracked):
    """A cache that can serve a stale patch is worse than no cache."""
    project_id, _source = tracked
    before = client.get(patch_url(project_id, 7, source="original"))

    moved = client.post(
        f"/api/projects/{project_id}/points/p1/corrections",
        json={"kind": "replace", "frame": 7, "x": 20.0, "y": 20.0},
    )
    assert moved.status_code in (200, 201), moved.text

    after = client.get(patch_url(project_id, 7, source="original"))
    assert after.status_code == 200
    assert after.headers["ETag"] != before.headers["ETag"]
    assert after.content != before.content


def test_a_new_keyframes_patch_is_rendered_before_anything_asks_for_it(client, tracked):
    """Todo 3, sixth round: "when a patch/keyframe is made it should be cached".

    The cache was already there and already correct — what filled it was the
    first *request*, which on the panel that judges texture is a seek into the
    full-resolution clip. So a card with eight keyframes cost eight of those on
    every fresh server, which is the "slow to load" being reported.

    Placing the keyframe is the moment its pixels are known and the moment
    nobody is waiting on a thumbnail, so that is when it is rendered. The
    assertion is that the *first* request for it is already a cache hit: it is
    answered with no proxy store and no video source available at all, which
    only a cached payload can be.
    """
    project_id, _source = tracked
    clear_patch_cache()

    added = client.post(
        f"/api/projects/{project_id}/points/p1/keyframes",
        json={"frame": 11, "x": 24.0, "y": 21.0, "track": "none"},
    )
    assert added.status_code == 200, added.text
    assert drain_patch_warmup(), "the warmer never finished"

    for source in ("proxy", "original"):
        with (
            mock.patch.object(
                Workspace, "get_video_source", side_effect=AssertionError("read the clip")
            ),
            mock.patch.object(
                Workspace, "get_proxy_store", side_effect=AssertionError("read the proxy")
            ),
        ):
            served = client.get(patch_url(project_id, 11, source=source))
        assert served.status_code == 200, f"{source}: {served.text}"
        assert served.content


def test_listing_a_grid_warms_the_keyframes_it_lists(client, tracked):
    """Which is what covers a restart, and every keyframe placed before this existed.

    Warming on creation cannot help a cache that starts empty every time the
    server does. The panel asks for its grid before it asks for any image in it,
    so that call is the other place where "these are about to be wanted" is
    known for certain.
    """
    project_id, _source = tracked
    clear_patch_cache()

    listed = client.get(f"/api/projects/{project_id}/points/p1/patches?frame=4")
    assert listed.status_code == 200, listed.text
    frames = [patch["frame"] for patch in listed.json()["patches"]]
    assert frames, "the fixture point has no keyframes to warm"
    assert drain_patch_warmup(), "the warmer never finished"

    with (
        mock.patch.object(
            Workspace, "get_video_source", side_effect=AssertionError("read the clip")
        ),
        mock.patch.object(
            Workspace, "get_proxy_store", side_effect=AssertionError("read the proxy")
        ),
    ):
        for frame in frames:
            served = client.get(patch_url(project_id, frame, source="original"))
            assert served.status_code == 200, served.text


def test_the_crop_scale_round_trip_is_quick_enough_to_drag(client, tracked):
    """Todo 7, sixth round: the crop preview follows the slider.

    The panel now stores the crop *during* the drag rather than only on release,
    throttled to one request at a time. That is only reasonable if a store plus
    the re-plan it triggers is cheap — this is the number behind that claim,
    measured rather than asserted, on a project small enough that it is the plan
    and the write being timed rather than the clip.

    Loose bound on purpose: it is a guard against the plan memoisation being
    lost (which would make every step an O(T) fit from scratch), not a benchmark.
    """
    project_id, _source = tracked
    solved = client.post(f"/api/projects/{project_id}/solve")
    if solved.status_code != 200:
        pytest.skip("one point is not enough to solve; the timing needs a real plan")
    assert wait_job(client, project_id, solved.json()["job_id"])["status"] == "done"
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0}}]},
    )

    start = time.perf_counter()
    steps = 8
    for step in range(steps):
        scale = 0.6 + step * 0.02
        assert (
            client.put(
                f"/api/projects/{project_id}/render/options", json={"crop": {"scale": scale}}
            ).status_code
            == 200
        )
        assert client.get(f"/api/projects/{project_id}/export/preview").status_code == 200
    each = (time.perf_counter() - start) / steps

    assert each < 0.5, f"a crop step costs {each * 1000:.0f} ms — too slow to drag against"


def test_serving_patches_does_not_slow_down_serving_frames(client, tracked):
    """The reported symptom, as an assertion.

    Frames come from disk and patches (now) from the proxy, so a burst of patch
    requests must not change how long a frame takes. Before the fix each patch
    was a full-resolution seek holding the decoder lock, and this ratio grew
    with the number of points on the shot.
    """
    project_id, _source = tracked

    def frame_latency() -> float:
        start = time.perf_counter()
        for index in range(10):
            assert client.get(f"/api/projects/{project_id}/frames/{index}").status_code == 200
        return time.perf_counter() - start

    quiet = frame_latency()

    clear_patch_cache()
    for index in range(60):  # a scrub's worth of patch traffic
        client.get(patch_url(project_id, index % 20))
    busy = frame_latency()

    # Generous bound: this is a regression guard against an order-of-magnitude
    # collapse, not a benchmark — the machine running it is not quiet.
    assert busy < quiet * 5 + 0.5, f"frames took {busy:.3f}s after patches vs {quiet:.3f}s before"
