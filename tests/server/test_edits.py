"""Phase 5 over HTTP: corrections, retracking, and undo/redo.

The frontend's whole editing vocabulary reduced to these endpoints — an on-frame
drag is one ``POST .../corrections``, a repair is a couple more, and every one of
them is a single undo step.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.types import PointTrack

from ._helpers import seed_tracks, upload_clip, wait_job


def diagonal_track(n_frames: int, *, x0: float = 12.0, y0: float = 10.0) -> PointTrack:
    """A confident point drifting slowly across a small clip."""
    frames = np.arange(n_frames, dtype=np.float32)
    return PointTrack(
        points=np.stack([x0 + 0.4 * frames, y0 + 0.3 * frames], axis=1),
        confidence=np.full(n_frames, 0.9, dtype=np.float32),
        visible=np.ones(n_frames, dtype=bool),
    )


@pytest.fixture
def edited_project(client: TestClient, workspace, tiny_clip):
    """A project with one seeded, tracked point ready to be edited."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    point_id = seed_tracks(workspace, project_id, [diagonal_track(source.frame_count)])[0]
    return project_id, point_id, source.frame_count


def track_of(client: TestClient, project_id: str, point_id: str) -> dict:
    """Fetch a point's per-frame track payload."""
    response = client.get(f"/api/projects/{project_id}/points/{point_id}/track")
    assert response.status_code == 200, response.text
    return dict(response.json())


def add(client: TestClient, project_id: str, point_id: str, body: dict) -> dict:
    """POST one correction and assert it was accepted."""
    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections", json=body
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- adding corrections ----------------------------------------------------------


def test_an_on_frame_drag_becomes_an_offset_correction(client, edited_project):
    project_id, point_id, _n = edited_project
    before = track_of(client, project_id, point_id)

    add(
        client,
        project_id,
        point_id,
        {
            "kind": "offset",
            "frame": 12,
            "dx": 3.5,
            "dy": -2.0,
            "falloff_before": 4,
            "falloff_after": 4,
        },
    )
    after = track_of(client, project_id, point_id)

    assert after["x"][12] == pytest.approx(before["x"][12] + 3.5, abs=1e-3)
    assert after["y"][12] == pytest.approx(before["y"][12] - 2.0, abs=1e-3)
    assert after["x"][8] == pytest.approx(before["x"][8], abs=1e-3)
    assert after["x"][20] == pytest.approx(before["x"][20], abs=1e-3)
    # The raw track is served alongside, so the editor can ghost before/after.
    assert after["raw_x"] is not None
    assert after["raw_x"][12] == pytest.approx(before["x"][12], abs=1e-6)


def test_a_shift_drag_becomes_a_hard_replace(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 7, "x": 40.0, "y": 30.0})
    after = track_of(client, project_id, point_id)
    assert (after["x"][7], after["y"][7]) == pytest.approx((40.0, 30.0), abs=1e-3)
    assert after["confidence"][7] == pytest.approx(1.0)


def test_delete_range_then_gap_fill_repairs_a_span(client, edited_project):
    project_id, point_id, _n = edited_project
    before = track_of(client, project_id, point_id)

    add(client, project_id, point_id, {"kind": "delete_range", "start": 10, "end": 15})
    deleted = track_of(client, project_id, point_id)
    assert deleted["x"][10:16] == [None] * 6
    assert deleted["gaps"] == [[10, 15]]

    add(
        client,
        project_id,
        point_id,
        {"kind": "gap_fill", "start": 10, "end": 15, "method": "spline", "confidence": 0.4},
    )
    filled = track_of(client, project_id, point_id)
    assert filled["gaps"] == []
    assert filled["x"][10:16] == pytest.approx(before["x"][10:16], abs=1e-2)
    assert filled["confidence"][12] == pytest.approx(0.4, abs=1e-6)


def test_corrections_can_be_toggled_off_to_compare(client, edited_project):
    project_id, point_id, _n = edited_project
    before = track_of(client, project_id, point_id)
    created = add(client, project_id, point_id, {"kind": "replace", "frame": 5, "x": 1.0, "y": 2.0})
    correction_id = created["corrections"][0]["id"]

    disabled = client.patch(
        f"/api/projects/{project_id}/points/{point_id}/corrections/{correction_id}",
        json={"enabled": False},
    )
    assert disabled.status_code == 200
    assert disabled.json()["corrections"][0]["enabled"] is False
    assert track_of(client, project_id, point_id)["x"][5] == pytest.approx(before["x"][5], abs=1e-3)

    client.patch(
        f"/api/projects/{project_id}/points/{point_id}/corrections/{correction_id}",
        json={"enabled": True},
    )
    assert track_of(client, project_id, point_id)["x"][5] == pytest.approx(1.0, abs=1e-3)


def test_corrections_can_be_deleted_and_cleared(client, edited_project):
    project_id, point_id, _n = edited_project
    first = add(client, project_id, point_id, {"kind": "replace", "frame": 4, "x": 1.0, "y": 1.0})
    add(client, project_id, point_id, {"kind": "replace", "frame": 6, "x": 2.0, "y": 2.0})
    correction_id = first["corrections"][0]["id"]

    removed = client.delete(
        f"/api/projects/{project_id}/points/{point_id}/corrections/{correction_id}"
    )
    assert removed.status_code == 200
    assert [c["frame"] for c in removed.json()["corrections"]] == [6]

    cleared = client.delete(f"/api/projects/{project_id}/points/{point_id}/corrections")
    assert cleared.status_code == 200
    assert cleared.json()["corrections"] == []


def test_malformed_and_misrouted_corrections_are_rejected(client, edited_project):
    project_id, point_id, _n = edited_project
    base = f"/api/projects/{project_id}/points/{point_id}/corrections"
    assert client.post(base, json={"kind": "teleport", "frame": 1}).status_code == 400
    assert client.post(base, json={"kind": "offset"}).status_code == 400
    assert client.post(base, json={"kind": "gap_fill", "start": 9, "end": 2}).status_code == 400
    retrack = client.post(base, json={"kind": "retrack", "frame": 1, "x": 0.0, "y": 0.0})
    assert retrack.status_code == 400
    assert "/retrack" in retrack.json()["detail"]
    assert client.delete(f"{base}/nope").status_code == 400


def test_corrections_survive_a_reload_and_feed_the_export(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 9, "x": 55.0, "y": 66.0})

    listed = client.get(f"/api/projects/{project_id}/points/{point_id}/corrections").json()
    assert [c["kind"] for c in listed["corrections"]] == ["replace"]

    exported = client.get(
        f"/api/projects/{project_id}/export/tracks", params={"format": "json"}
    ).json()
    row = next(r for r in exported["points"] if r["frame"] == 9 and r["point_id"] == point_id)
    assert (row["x"], row["y"]) == pytest.approx((55.0, 66.0), abs=1e-3)


# -- undo / redo -------------------------------------------------------------------


def test_undo_restores_the_exact_prior_state(client, edited_project):
    project_id, point_id, _n = edited_project
    original = track_of(client, project_id, point_id)

    add(client, project_id, point_id, {"kind": "offset", "frame": 10, "dx": 5.0, "dy": 5.0})
    add(client, project_id, point_id, {"kind": "delete_range", "start": 2, "end": 4})
    edited = track_of(client, project_id, point_id)
    assert edited["x"][2] is None

    undone = client.post(f"/api/projects/{project_id}/edits/undo")
    assert undone.status_code == 200
    assert undone.json()["applied"].startswith("delete_range")
    after_one = track_of(client, project_id, point_id)
    assert after_one["x"][2] is not None
    assert after_one["x"][10] == pytest.approx(original["x"][10] + 5.0, abs=1e-3)

    client.post(f"/api/projects/{project_id}/edits/undo")
    restored = track_of(client, project_id, point_id)
    assert restored["x"] == pytest.approx(original["x"], abs=0.0)
    assert restored["y"] == pytest.approx(original["y"], abs=0.0)
    assert restored["confidence"] == pytest.approx(original["confidence"], abs=0.0)
    assert restored["corrections"] == []


def test_redo_reapplies_what_undo_removed(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 8, "x": 77.0, "y": 88.0})
    edited = track_of(client, project_id, point_id)

    client.post(f"/api/projects/{project_id}/edits/undo")
    assert track_of(client, project_id, point_id)["x"][8] != pytest.approx(77.0, abs=1e-3)

    redone = client.post(f"/api/projects/{project_id}/edits/redo")
    assert redone.status_code == 200
    assert redone.json()["applied"].startswith("replace")
    assert track_of(client, project_id, point_id)["x"] == pytest.approx(edited["x"], abs=0.0)


def test_a_new_edit_clears_the_redo_stack(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 3, "x": 1.0, "y": 1.0})
    client.post(f"/api/projects/{project_id}/edits/undo")
    assert client.get(f"/api/projects/{project_id}/edits/history").json()["can_redo"] is True

    add(client, project_id, point_id, {"kind": "replace", "frame": 4, "x": 2.0, "y": 2.0})
    history = client.get(f"/api/projects/{project_id}/edits/history").json()
    assert history["can_redo"] is False
    assert history["can_undo"] is True


def test_undo_at_the_bottom_of_the_stack_is_a_no_op(client, edited_project):
    project_id, point_id, _n = edited_project
    before = track_of(client, project_id, point_id)
    response = client.post(f"/api/projects/{project_id}/edits/undo")
    assert response.status_code == 200
    assert response.json()["applied"] is None
    assert track_of(client, project_id, point_id)["x"] == pytest.approx(before["x"], abs=0.0)


def test_history_reports_what_each_step_would_do(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "offset", "frame": 11, "dx": 1.0, "dy": 0.0})
    history = client.get(f"/api/projects/{project_id}/edits/history").json()
    assert history["undo_label"] == f"offset {point_id} @ 11"
    assert history["undo_depth"] == 1
    assert history["redo_label"] is None


def test_history_survives_a_fresh_server_instance(client, workspace, edited_project):
    from fastapi.testclient import TestClient as Client  # noqa: PLC0415 - a second app instance

    from stable.server.app import create_app  # noqa: PLC0415

    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 6, "x": 3.0, "y": 3.0})

    with Client(create_app()) as reopened:
        assert reopened.get(f"/api/projects/{project_id}/edits/history").json()["can_undo"] is True
        reopened.post(f"/api/projects/{project_id}/edits/undo")
        assert track_of(reopened, project_id, point_id)["corrections"] == []


def test_a_failed_edit_leaves_no_undo_point_behind(client, edited_project):
    project_id, point_id, _n = edited_project
    add(client, project_id, point_id, {"kind": "replace", "frame": 2, "x": 1.0, "y": 1.0})
    depth = client.get(f"/api/projects/{project_id}/edits/history").json()["undo_depth"]

    assert client.delete(
        f"/api/projects/{project_id}/points/{point_id}/corrections/missing"
    ).status_code == 400
    assert client.get(f"/api/projects/{project_id}/edits/history").json()["undo_depth"] == depth


# -- retrack from here -------------------------------------------------------------


def test_retrack_from_a_corrected_position_splices_a_new_run(client, edited_project):
    project_id, point_id, n_frames = edited_project
    before = track_of(client, project_id, point_id)
    start = n_frames // 2

    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": start, "x": 30.0, "y": 25.0, "direction": "fwd", "tracker": "lk"},
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    assert job["result"]["covered_frames"] > 0

    after = track_of(client, project_id, point_id)
    assert [c["kind"] for c in after["corrections"]] == ["retrack"]
    assert after["corrections"][0]["source"]
    # Forward only: the corrected position lands, earlier frames keep the baseline.
    assert (after["x"][start], after["y"][start]) == pytest.approx((30.0, 25.0), abs=1e-3)
    assert after["x"][start - 3] == pytest.approx(before["x"][start - 3], abs=1e-3)
    assert after["x"][start + 2] != pytest.approx(before["x"][start + 2], abs=1e-3)


def test_a_retrack_is_undoable_like_any_other_edit(client, edited_project):
    project_id, point_id, n_frames = edited_project
    before = track_of(client, project_id, point_id)
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 4, "x": 20.0, "y": 18.0, "direction": "bidir", "tracker": "lk"},
    )
    wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert track_of(client, project_id, point_id)["corrections"]

    client.post(f"/api/projects/{project_id}/edits/undo")
    restored = track_of(client, project_id, point_id)
    assert restored["corrections"] == []
    assert restored["x"] == pytest.approx(before["x"], abs=0.0)


def test_retrack_validates_its_frame_and_point(client, edited_project):
    project_id, point_id, n_frames = edited_project
    missing = client.post(
        f"/api/projects/{project_id}/points/nope/retrack", json={"frame": 1, "x": 0.0, "y": 0.0}
    )
    assert missing.status_code == 404

    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": n_frames + 50, "x": 0.0, "y": 0.0},
    )
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    assert job["status"] == "error"
    assert "outside the clip" in job["error"]
