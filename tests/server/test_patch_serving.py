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

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.server.routes.points import clear_patch_cache

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
