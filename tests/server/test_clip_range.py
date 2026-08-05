"""In/Out markers: what tracking and export are limited to (todo 1).

"It's not always the full length of the video we want to track." The answer had
to bound both halves of the work, and the user confirmed both: a run covers the
range and nothing outside it, and an export writes exactly those frames.

The most important tests here are the two negative ones — a project that never
sets a range behaves exactly as it did before, and a stale range (set on a
longer clip) is clamped rather than crashing a run.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.errors import ProjectError
from stable.server.clip_range import CLIP_RANGE_CACHE, ClipRange, load_clip_range, save_clip_range

from ._helpers import upload_clip, wait_job


@pytest.fixture
def project(client: TestClient, tiny_clip):
    """A project with one point anchored mid-clip, so a range can exclude it."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    return project_id, source


def place(client: TestClient, project_id: str, source, frame: int) -> str:
    x, y = source.true_position(frame)
    return str(
        client.post(
            f"/api/projects/{project_id}/points",
            json={"frame": frame, "x": x, "y": y, "tracker": "lk"},
        ).json()["id"]
    )


# -- the value type ------------------------------------------------------------


def test_an_empty_or_inverted_range_is_refused():
    with pytest.raises(ProjectError, match="after the in point"):
        ClipRange(10, 10)
    with pytest.raises(ProjectError, match="after the in point"):
        ClipRange(10, 4)
    with pytest.raises(ProjectError, match=">= 0"):
        ClipRange(-1, 5)


def test_a_range_from_a_longer_clip_is_clamped_not_fatal(tmp_path):
    """A project re-pointed at a shorter file must still track."""
    save_clip_range(tmp_path, ClipRange(10, 500), frame_count=500)
    fitted = load_clip_range(tmp_path, frame_count=40)
    assert fitted is not None
    assert fitted.as_tuple() == (10, 40)


def test_no_sidecar_means_the_whole_clip(tmp_path):
    assert load_clip_range(tmp_path, frame_count=40) is None
    assert not (tmp_path / CLIP_RANGE_CACHE).exists()


# -- over HTTP -----------------------------------------------------------------


def test_the_range_defaults_to_the_whole_clip(client, project):
    project_id, source = project
    body = client.get(f"/api/projects/{project_id}/range").json()
    assert body["whole_clip"] is True
    assert body["in"] == 0
    assert body["out"] == source.frame_count


def test_one_marker_moves_without_disturbing_the_other(client, project):
    project_id, source = project
    client.put(f"/api/projects/{project_id}/range", json={"in": 5})
    after_in = client.get(f"/api/projects/{project_id}/range").json()
    assert (after_in["in"], after_in["out"]) == (5, source.frame_count)

    client.put(f"/api/projects/{project_id}/range", json={"out": 20})
    after_out = client.get(f"/api/projects/{project_id}/range").json()
    assert (after_out["in"], after_out["out"]) == (5, 20)
    assert after_out["frames"] == 15
    assert after_out["whole_clip"] is False


def test_clearing_the_range_goes_back_to_the_whole_clip(client, project):
    project_id, _source = project
    client.put(f"/api/projects/{project_id}/range", json={"in": 5, "out": 20})
    cleared = client.put(f"/api/projects/{project_id}/range", json={"in": None, "out": None})
    assert cleared.json()["whole_clip"] is True


def test_an_inverted_range_is_a_400_not_a_stored_mess(client, project):
    project_id, _source = project
    client.put(f"/api/projects/{project_id}/range", json={"in": 5, "out": 20})
    response = client.put(f"/api/projects/{project_id}/range", json={"out": 3})
    assert response.status_code == 400
    # ...and the stored range is untouched.
    assert client.get(f"/api/projects/{project_id}/range").json()["out"] == 20


# -- what it actually bounds ---------------------------------------------------


def test_tracking_covers_the_range_and_nothing_outside_it(client, project):
    project_id, source = project
    point_id = place(client, project_id, source, frame=12)
    client.put(f"/api/projects/{project_id}/range", json={"in": 8, "out": 20})

    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

    track = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()
    xs = np.asarray([np.nan if v is None else v for v in track["x"]], dtype=float)
    covered = np.isfinite(xs)
    assert covered[8:20].all(), "every frame in the range should be tracked"
    assert not covered[:8].any(), "nothing before the in point"
    assert not covered[20:].any(), "nothing after the out point"


def test_an_anchor_outside_the_range_is_reported_not_silently_ignored(client, project):
    project_id, source = project
    point_id = place(client, project_id, source, frame=2)
    client.put(f"/api/projects/{project_id}/range", json={"in": 10, "out": 20})

    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    job = wait_job(client, project_id, submitted.json()["job_id"])
    assert job["status"] == "error"
    assert "inside frames" in (job.get("error") or "")


def test_without_a_range_tracking_still_covers_the_whole_clip(client, project):
    """The regression that matters most: nothing changes for a project that never sets one."""
    project_id, source = project
    point_id = place(client, project_id, source, frame=0)
    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

    track = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()
    xs = np.asarray([np.nan if v is None else v for v in track["x"]], dtype=float)
    assert np.isfinite(xs).sum() == source.frame_count


def test_the_export_writes_exactly_the_ranged_frames(client, project, workspace):
    project_id, source = project
    point_id = place(client, project_id, source, frame=0)
    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"
    solving = client.post(f"/api/projects/{project_id}/solve", json={})
    assert wait_job(client, project_id, solving.json()["job_id"], timeout=120)["status"] == "done"
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0}}]},
    )
    client.put(f"/api/projects/{project_id}/range", json={"in": 4, "out": 14})

    exported = client.post(f"/api/projects/{project_id}/export/video", json={})
    job = wait_job(client, project_id, exported.json()["job_id"], timeout=180)
    assert job["status"] == "done", job
    assert job["result"]["frame_range"] == [4, 14]
    assert job["result"]["frames"] == 10

    import av  # noqa: PLC0415 - the video extra, imported where it is used

    path = workspace.renders_dir(project_id) / job["result"]["filename"]
    with av.open(str(path)) as container:
        written = sum(1 for _ in container.decode(video=0))
    assert written == 10

    # ...and the file *plays* for that long. Counting video packets is not
    # enough: the audio used to be copied in full, so a container holding ten
    # frames still reported the source's duration and every player froze on the
    # last frame for the remainder — which is what "the exported video was
    # still the full length" was (todo 2, second round).
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
        audio_seconds = samples / float(stream.sample_rate)
    assert audio_seconds == pytest.approx(10 / source.fps, abs=0.05)
