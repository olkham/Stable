"""Solve, smoothing, stabilized preview and export over the HTTP surface.

These cover the endpoints' shape and error handling. That the maths behind them
is really Phase 3's (and not the Phase 4 stand-ins that used to sit here) is the
subject of ``test_wireup.py``.
"""

from __future__ import annotations

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from ._helpers import upload_clip, wait_job


@pytest.fixture
def tracked_project(client: TestClient, tiny_clip):
    """A project with two tracked points, ready to solve."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])

    for dx in (0.0, 6.0):
        x, y = source.true_position(0)
        created = client.post(
            f"/api/projects/{project_id}/points",
            json={"frame": 0, "x": x + dx, "y": y, "tracker": "lk"},
        )
        point_id = created.json()["id"]
        submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
        wait_job(client, project_id, submitted.json()["job_id"], timeout=60)

    return project_id, source


def test_solve_requires_tracked_points(client: TestClient, tiny_clip) -> None:
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    response = client.post(f"/api/projects/{detail['id']}/solve")
    assert response.status_code == 400


def test_solve_then_smoothing_preview(client: TestClient, tracked_project) -> None:
    project_id, source = tracked_project

    submitted = client.post(f"/api/projects/{project_id}/solve")
    assert submitted.status_code == 200
    job = wait_job(client, project_id, submitted.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["n_points"] == 2

    solved = client.get(f"/api/projects/{project_id}/solve").json()
    curves = solved["curves"]
    assert len(curves["x"]) == source.frame_count
    # Reference frame is frame 0 -> identity transform there.
    assert curves["x"][0] == pytest.approx(0.0, abs=1e-6)
    assert curves["y"][0] == pytest.approx(0.0, abs=1e-6)
    assert curves["s"][0] == pytest.approx(1.0, abs=1e-6)

    smoothing = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": "x", "filter": "gaussian", "params": {"sigma": 2.0}},
                {"channel": "y", "filter": "moving_average", "params": {"window": 5}},
            ]
        },
    )
    assert smoothing.status_code == 200
    body = smoothing.json()
    assert len(body["curves"]["smoothed"]["x"]) == source.frame_count
    assert body["filters"][0]["filter"] == "gaussian"

    fetched = client.get(f"/api/projects/{project_id}/smoothing").json()
    assert fetched["curves"]["smoothed"] is not None


def test_tripod_filter_locks_to_reference_frame(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    submit = client.post(f"/api/projects/{project_id}/solve")
    wait_job(client, project_id, submit.json()["job_id"])

    result = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "tripod", "params": {}}]},
    ).json()
    smoothed_x = result["curves"]["smoothed"]["x"]
    assert all(v == pytest.approx(smoothed_x[0]) for v in smoothed_x)


def test_stabilized_frame_requires_solve_and_smoothing(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    response = client.get(f"/api/projects/{project_id}/frames/0/stabilized")
    assert response.status_code == 409

    submit = client.post(f"/api/projects/{project_id}/solve")
    wait_job(client, project_id, submit.json()["job_id"])
    # Solve alone seeds an (identity) smoothing cache too, since run_solve
    # recomputes smoothed curves from whatever filter list is set (empty by
    # default), so the preview should already be servable.
    ready = client.get(f"/api/projects/{project_id}/frames/0/stabilized")
    assert ready.status_code == 200
    assert ready.headers["content-type"] == "image/jpeg"


def test_export_tracks_csv_and_json(client: TestClient, tracked_project) -> None:
    project_id, source = tracked_project

    as_json = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "json"})
    assert as_json.status_code == 200
    payload = json.loads(as_json.text)
    assert len(payload["points"]) == 2 * source.frame_count

    as_csv = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "csv"})
    assert as_csv.status_code == 200
    rows = list(csv.DictReader(io.StringIO(as_csv.text)))
    assert len(rows) == 2 * source.frame_count
    assert set(rows[0]) == {"point_id", "frame", "x", "y", "confidence", "visible"}


def test_export_tracks_as_after_effects_keyframes(client: TestClient, tracked_project) -> None:
    """Todo 18's other half: a format a compositor will actually read."""
    project_id, source = tracked_project

    response = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "ae"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "tracks-ae.txt" in response.headers["content-disposition"]

    text = response.text
    assert text.startswith("Adobe After Effects 8.0 Keyframe Data")
    assert text.count("Transform\tPosition") == 2  # one block per tracked point
    assert f"\tSource Width\t{source.width}" in text
    assert "End of Keyframe Data" in text


def test_an_unknown_export_format_is_refused_by_name(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    response = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "xml"})
    assert response.status_code == 400
    assert "ae" in response.json()["detail"]


def test_after_effects_export_needs_something_to_export(client: TestClient, tiny_clip) -> None:
    """An empty keyframe document reads as a broken export, so it is refused."""
    detail = upload_clip(client, tiny_clip[0])
    wait_job(client, detail["id"], detail["proxy_job_id"])
    response = client.get(f"/api/projects/{detail['id']}/export/tracks", params={"format": "ae"})
    assert response.status_code == 400
    assert "tracked" in response.json()["detail"]


def test_export_curves_requires_solve(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    response = client.get(f"/api/projects/{project_id}/export/curves")
    assert response.status_code == 400


def test_export_curves_after_solve(client: TestClient, tracked_project) -> None:
    project_id, source = tracked_project
    submit = client.post(f"/api/projects/{project_id}/solve")
    wait_job(client, project_id, submit.json()["job_id"])

    as_json = client.get(f"/api/projects/{project_id}/export/curves", params={"format": "json"})
    assert as_json.status_code == 200
    payload = json.loads(as_json.text)
    assert len(payload["raw"]["x"]) == source.frame_count

    as_csv = client.get(f"/api/projects/{project_id}/export/curves", params={"format": "csv"})
    assert as_csv.status_code == 200
    rows = list(csv.DictReader(io.StringIO(as_csv.text)))
    assert len(rows) == source.frame_count


def smooth_a_little(client: TestClient, project_id: str) -> None:
    """Configure a real smoothing filter, so the render is not the identity.

    Without this the stabilisation warp is ``smoothed . raw^-1`` with
    ``smoothed == raw`` on every frame, and the export is refused as a no-op —
    see ``test_export_refuses_a_render_that_would_change_nothing``.
    """
    response = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": c, "filter": "butterworth", "params": {"cutoff_hz": 1.0}}
                for c in ("x", "y")
            ]
        },
    )
    assert response.status_code == 200, response.text


def test_export_video_end_to_end(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    submit = client.post(f"/api/projects/{project_id}/solve")
    wait_job(client, project_id, submit.json()["job_id"])
    smooth_a_little(client, project_id)

    render = client.post(
        f"/api/projects/{project_id}/export/video",
        json={"filename": "out.mp4", "codec": "h264", "crf": 20},
    )
    assert render.status_code == 200
    job = wait_job(client, project_id, render.json()["job_id"], timeout=60)
    assert job["status"] == "done", job

    download = client.get(f"/api/projects/{project_id}/export/download/out.mp4")
    assert download.status_code == 200
    assert download.headers["content-type"] == "video/mp4"
    assert len(download.content) > 0

    # A finished render must stay reachable after the export dialog forgets it.
    listed = client.get(f"/api/projects/{project_id}/export/renders")
    assert listed.status_code == 200
    assert [entry["filename"] for entry in listed.json()] == ["out.mp4"]
    assert listed.json()[0]["bytes"] > 0


def test_renders_listing_is_empty_before_any_export(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    assert client.get(f"/api/projects/{project_id}/export/renders").json() == []


def test_export_download_rejects_path_traversal(client: TestClient, tracked_project) -> None:
    project_id, _source = tracked_project
    response = client.get(f"/api/projects/{project_id}/export/download/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)
