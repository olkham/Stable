"""The smoothed track path the canvas draws (todo 3, fourth round).

The smoothing produces four *camera* curves, so "what did the smoothing do to
this track?" has no answer until the stabilisation warp is applied to the track
itself. ``GET /smoothing/tracks`` is that answer, and these tests pin the two
properties the canvas relies on: it is per point and over the whole timeline,
and it actually reflects the filters rather than echoing the source path back.
"""

from __future__ import annotations

import numpy as np

from ._helpers import upload_clip, wait_job


def _tracked_and_solved(client, tiny_clip):
    """Return ``(project_id, point_ids)`` for a solved two-point project."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])

    ids = []
    for dx, dy in ((0.0, 0.0), (6.0, -4.0)):
        x, y = source.true_position(0)
        created = client.post(
            f"/api/projects/{project_id}/points",
            json={"frame": 0, "x": x + dx, "y": y + dy, "tracker": "lk"},
        )
        point_id = created.json()["id"]
        ids.append(point_id)
        submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
        assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

    solving = client.post(f"/api/projects/{project_id}/solve", json={})
    assert wait_job(client, project_id, solving.json()["job_id"], timeout=120)["status"] == "done"
    return project_id, ids


def test_there_is_nothing_to_draw_before_a_solve(client, tiny_clip):
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])

    response = client.get(f"/api/projects/{project_id}/smoothing/tracks")
    assert response.status_code == 409
    assert "solve" in response.json()["detail"]


def test_every_tracked_point_gets_a_path_over_the_whole_clip(client, tiny_clip):
    project_id, ids = _tracked_and_solved(client, tiny_clip)
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0}}]},
    )

    body = client.get(f"/api/projects/{project_id}/smoothing/tracks").json()
    assert [row["point_id"] for row in body["points"]] == ids
    for row in body["points"]:
        # Absolute frame indices, like every other array this server serves.
        assert len(row["x"]) == body["n_frames"]
        assert len(row["y"]) == body["n_frames"]
        assert any(value is not None for value in row["x"])


def _column(row: dict, axis: str) -> np.ndarray:
    """The path's finite samples on one axis, as an array."""
    return np.asarray([v for v in row[axis] if v is not None], dtype=np.float64)


def test_with_no_filter_the_smoothed_path_is_the_tracked_path(client, tiny_clip):
    """The warp is the identity then, and the overlay must say so rather than lie."""
    project_id, ids = _tracked_and_solved(client, tiny_clip)
    client.put(f"/api/projects/{project_id}/smoothing", json={"filters": []})

    row = client.get(f"/api/projects/{project_id}/smoothing/tracks").json()["points"][0]
    tracked = client.get(f"/api/projects/{project_id}/points/{ids[0]}/track").json()

    assert np.allclose(_column(row, "x"), _column(tracked, "x"), atol=1e-6)
    assert np.allclose(_column(row, "y"), _column(tracked, "y"), atol=1e-6)


def test_enabling_a_filter_moves_the_smoothed_path(client, tiny_clip):
    """Otherwise it would be decoration: a second copy of the trail already drawn."""
    project_id, _ids = _tracked_and_solved(client, tiny_clip)

    client.put(f"/api/projects/{project_id}/smoothing", json={"filters": []})
    before = client.get(f"/api/projects/{project_id}/smoothing/tracks").json()["points"][0]

    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 0.5}}]},
    )
    after = client.get(f"/api/projects/{project_id}/smoothing/tracks").json()["points"][0]

    assert not np.allclose(_column(after, "x"), _column(before, "x"), atol=1e-3)
