"""Point CRUD, anchor pins, and the real tracking pipeline as a background job."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from ._helpers import upload_clip, wait_job


@pytest.fixture
def project(client: TestClient, tiny_clip):
    """An uploaded project with its proxy already generated."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    wait_job(client, detail["id"], detail["proxy_job_id"])
    return detail, source


def test_create_update_delete_point(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    x, y = source.true_position(0)

    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y, "tracker": "lk"}
    )
    assert created.status_code == 200
    point = created.json()
    assert point["id"] == "p1"
    assert point["queries"] == [{"frame": 0, "x": x, "y": y}]
    assert point["has_track"] is False

    listed = client.get(f"/api/projects/{project_id}/points").json()
    assert [p["id"] for p in listed] == ["p1"]

    updated = client.patch(
        f"/api/projects/{project_id}/points/p1",
        json={"tracker": "template", "enabled_for_solve": False},
    )
    assert updated.status_code == 200
    assert updated.json()["tracker"] == "template"
    assert updated.json()["enabled_for_solve"] is False

    deleted = client.delete(f"/api/projects/{project_id}/points/p1")
    assert deleted.status_code == 200
    assert client.get(f"/api/projects/{project_id}/points").json() == []


def test_point_ids_increment(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    for _ in range(3):
        x, y = source.true_position(0)
        client.post(f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y})
    ids = [p["id"] for p in client.get(f"/api/projects/{project_id}/points").json()]
    assert ids == ["p1", "p2", "p3"]


def test_anchor_add_and_delete(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    x0, y0 = source.true_position(0)
    client.post(f"/api/projects/{project_id}/points", json={"frame": 0, "x": x0, "y": y0})

    x10, y10 = source.true_position(10)
    added = client.post(
        f"/api/projects/{project_id}/points/p1/anchors", json={"frame": 10, "x": x10, "y": y10}
    )
    assert added.status_code == 200
    frames = sorted(q["frame"] for q in added.json()["queries"])
    assert frames == [0, 10]

    # Cannot delete the last remaining anchor.
    client.delete(f"/api/projects/{project_id}/points/p1/anchors/0")
    single = client.get(f"/api/projects/{project_id}/points").json()[0]
    assert len(single["queries"]) == 1

    forbidden = client.delete(f"/api/projects/{project_id}/points/p1/anchors/10")
    assert forbidden.status_code == 400


def test_create_point_out_of_range_frame_rejected(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    response = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": source.frame_count + 50, "x": 1.0, "y": 1.0},
    )
    assert response.status_code == 400


def test_track_single_anchor_with_lk(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    x, y = source.true_position(0)
    client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y, "tracker": "lk"}
    )

    submitted = client.post(f"/api/projects/{project_id}/points/p1/track", json={"tracker": "lk"})
    assert submitted.status_code == 200
    job = wait_job(client, project_id, submitted.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["n_frames"] == source.frame_count

    track = client.get(f"/api/projects/{project_id}/points/p1/track").json()
    assert track["n_frames"] == source.frame_count
    # The anchor frame is exact ground truth by construction.
    assert track["x"][0] == pytest.approx(x, abs=1e-3)
    assert track["y"][0] == pytest.approx(y, abs=1e-3)
    assert all(0.0 <= c <= 1.0 for c in track["confidence"])
    assert track["anchor_frames"] == [0]

    summary = client.get(f"/api/projects/{project_id}/points").json()[0]
    assert summary["has_track"] is True
    assert summary["tracked_frames"] > 0


def test_track_not_found_returns_404(client: TestClient, project) -> None:
    detail, _source = project
    project_id = detail["id"]
    response = client.get(f"/api/projects/{project_id}/points/does-not-exist/track")
    assert response.status_code == 404


def test_track_before_running_is_404(client: TestClient, project) -> None:
    detail, source = project
    project_id = detail["id"]
    x, y = source.true_position(0)
    client.post(f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y})
    response = client.get(f"/api/projects/{project_id}/points/p1/track")
    assert response.status_code == 404


def test_multi_anchor_merge_passes_through_every_anchor(client: TestClient, project) -> None:
    """The server-side pipeline plans runs, tracks each, and merges — exact anchor passthrough."""
    detail, source = project
    project_id = detail["id"]
    frames = (0, source.frame_count // 2, source.frame_count - 1)
    x0, y0 = source.true_position(frames[0])
    client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": frames[0], "x": x0, "y": y0, "tracker": "lk"},
    )
    for frame in frames[1:]:
        x, y = source.true_position(frame)
        anchor = client.post(
            f"/api/projects/{project_id}/points/p1/anchors", json={"frame": frame, "x": x, "y": y}
        )
        assert anchor.status_code == 200

    submitted = client.post(f"/api/projects/{project_id}/points/p1/track", json={"tracker": "lk"})
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    assert job["status"] == "done", job
    assert sorted(job["result"]["anchor_frames"]) == sorted(frames)

    track = client.get(f"/api/projects/{project_id}/points/p1/track").json()
    for frame in frames:
        expected_x, expected_y = source.true_position(frame)
        assert track["x"][frame] == pytest.approx(expected_x, abs=1e-3)
        assert track["y"][frame] == pytest.approx(expected_y, abs=1e-3)
        assert track["confidence"][frame] == pytest.approx(1.0, abs=1e-6)
    # No sample should be NaN across the whole clip once every anchor is merged.
    assert all(v is not None for v in track["x"])
    assert all(not math.isnan(v) for v in track["confidence"])
