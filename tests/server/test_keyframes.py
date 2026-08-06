"""todo_6: pins and promoted corrections behind one keyframe API, plus auto-track.

The product used to show two of the same thing — "anchors" (pins in
``PointSpec.queries``) and "keyframes" (positions promoted from manual
corrections) — with two route pairs and two delete gestures. These tests pin the
unified surface: one list with provenance, one mutation vocabulary that decides
for itself which storage to rewrite, and a re-track that follows the edit
without anybody pressing a button.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from stable.server.jobs import reset_job_manager

from ._helpers import upload_clip, wait_job

TRACKER = "lk"


@pytest.fixture
def project(client: TestClient, tiny_clip):
    """An uploaded project whose proxy is ready, plus its ground-truth source."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    wait_job(client, detail["id"], detail["proxy_job_id"], timeout=60)
    return detail["id"], source


def place_point(client: TestClient, project_id: str, frame: int, x: float, y: float) -> str:
    """Create a point with its first keyframe and return its id."""
    response = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": frame, "x": x, "y": y, "tracker": TRACKER},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def keyframes(client: TestClient, project_id: str, point_id: str) -> list[dict]:
    """Fetch a point's unified keyframe list."""
    response = client.get(f"/api/projects/{project_id}/points/{point_id}/keyframes")
    assert response.status_code == 200, response.text
    return list(response.json())


def add_correction(client: TestClient, project_id: str, point_id: str, **body) -> dict:
    """Append one correction to a point."""
    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections", json=body
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- one list, whatever states it ----------------------------------------------


def test_the_list_merges_pins_and_corrections_with_their_provenance(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    )
    state = add_correction(
        client, project_id, point_id, kind="replace", frame=20, x=30.0, y=31.0
    )
    correction_id = state["corrections"][0]["id"]

    listed = keyframes(client, project_id, point_id)
    assert [k["frame"] for k in listed] == [0, 10, 20]
    assert [k["source"] for k in listed] == ["pin", "pin", "correction"]
    assert [k["correction_id"] for k in listed] == [None, None, correction_id]
    assert (listed[2]["x"], listed[2]["y"]) == (30.0, 31.0)


def test_a_correction_wins_over_a_pin_on_the_same_frame(client, project) -> None:
    """One frame, one truth — and the later statement is the correction's."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    )
    add_correction(client, project_id, point_id, kind="replace", frame=10, x=9.0, y=9.5)

    listed = keyframes(client, project_id, point_id)
    assert [k["frame"] for k in listed] == [0, 10]
    assert listed[1]["source"] == "correction"
    assert (listed[1]["x"], listed[1]["y"]) == (9.0, 9.5)


# -- POST: stating a position --------------------------------------------------


def test_posting_over_a_pin_replaces_it(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    )
    body = client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 7.0, "y": 8.0, "track": "none"},
    ).json()

    assert sorted(q["frame"] for q in body["queries"]) == [0, 10]
    assert [(k["frame"], k["x"], k["y"]) for k in body["keyframes"]][1] == (10, 7.0, 8.0)


def test_posting_over_a_correction_takes_the_frame_off_it(client, project) -> None:
    """Shadowing would leave the old claim displacing the position just stated."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_correction(client, project_id, point_id, kind="replace", frame=12, x=1.0, y=2.0)

    body = client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 12, "x": 40.0, "y": 41.0, "track": "none"},
    ).json()

    assert body["corrections"] == []
    listed = keyframes(client, project_id, point_id)
    assert [(k["frame"], k["source"]) for k in listed] == [(0, "pin"), (12, "pin")]
    assert (listed[1]["x"], listed[1]["y"]) == (40.0, 41.0)


def test_posting_over_a_drag_clears_its_offset_too(client, project) -> None:
    """A drag is an offset *plus* a promoted replace; half a removal is no removal."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/track", json={"tracker": TRACKER}
    )
    assert wait_job(client, project_id, submitted.json()["job_id"], timeout=120)["status"] == "done"
    state = add_correction(
        client, project_id, point_id, kind="offset", frame=14, dx=4.0, dy=-3.0
    )
    assert [c["kind"] for c in state["corrections"]] == ["offset", "replace"]

    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 14, "x": 22.0, "y": 23.0, "track": "none"},
    )
    track = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()
    assert track["corrections"] == []


# -- PATCH: moving one -----------------------------------------------------------


def test_patch_moves_a_pin_backed_keyframe(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    )

    moved = client.patch(
        f"/api/projects/{project_id}/points/{point_id}/keyframes/10",
        json={"frame": 15, "x": 11.0, "track": "none"},
    )
    assert moved.status_code == 200, moved.text
    # `y` was omitted, so it is carried over rather than reset.
    assert [(k["frame"], k["x"], k["y"]) for k in moved.json()["keyframes"]] == [
        (0, *source.true_position(0)),
        (15, 11.0, 6.0),
    ]


def test_patch_moves_a_correction_backed_keyframe_and_drops_the_correction(
    client, project
) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_correction(client, project_id, point_id, kind="replace", frame=12, x=1.0, y=2.0)

    moved = client.patch(
        f"/api/projects/{project_id}/points/{point_id}/keyframes/12",
        json={"frame": 18, "track": "none"},
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["corrections"] == []
    assert [(k["frame"], k["x"], k["y"], k["source"]) for k in moved.json()["keyframes"]] == [
        (0, *source.true_position(0), "pin"),
        (18, 1.0, 2.0, "pin"),
    ]


def test_patch_onto_an_occupied_frame_replaces_what_was_there(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    for frame in (10, 20):
        client.post(
            f"/api/projects/{project_id}/points/{point_id}/keyframes",
            json={"frame": frame, "x": float(frame), "y": 1.0, "track": "none"},
        )

    moved = client.patch(
        f"/api/projects/{project_id}/points/{point_id}/keyframes/10",
        json={"frame": 20, "track": "none"},
    )
    assert moved.status_code == 200, moved.text
    assert [(k["frame"], k["x"]) for k in moved.json()["keyframes"]] == [
        (0, source.true_position(0)[0]),
        (20, 10.0),
    ]


def test_patching_a_frame_with_no_keyframe_is_a_404(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    response = client.patch(
        f"/api/projects/{project_id}/points/{point_id}/keyframes/7",
        json={"x": 1.0, "track": "none"},
    )
    assert response.status_code == 404


# -- DELETE: removing whatever backs it -------------------------------------------


def test_delete_removes_a_backing_retrack_and_its_cached_run(client, workspace, project) -> None:
    """The retrack's ``.npz`` and violations sidecar go with the keyframe."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/track", json={"tracker": TRACKER}
    )
    assert wait_job(client, project_id, submitted.json()["job_id"], timeout=120)["status"] == "done"

    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 5, "direction": "fwd", "tracker": TRACKER, "max_frames": 4},
    )
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job

    tracks_dir = workspace.tracks_dir(project_id)
    cached = sorted(p.name for p in tracks_dir.iterdir() if "retrack-" in p.name)
    assert cached, "the retrack left no cached run to clean up"
    assert [k["source"] for k in keyframes(client, project_id, point_id)] == ["pin", "correction"]

    removed = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/5?track=none")
    assert removed.status_code == 200, removed.text
    assert removed.json()["corrections"] == []
    assert [k["frame"] for k in removed.json()["keyframes"]] == [0]
    assert [p.name for p in tracks_dir.iterdir() if "retrack-" in p.name] == []


def test_delete_removes_a_pin(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    )
    removed = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/10?track=none")
    assert removed.status_code == 200
    assert [q["frame"] for q in removed.json()["queries"]] == [0]


def test_deleting_the_last_keyframe_is_refused(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    refused = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/0?track=none")
    assert refused.status_code == 400
    assert "last remaining keyframe" in refused.json()["detail"]


def test_the_last_keyframe_may_be_a_correction(client, project) -> None:
    """The refusal counts stated positions, not pins: a point with none cannot track."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_correction(client, project_id, point_id, kind="replace", frame=10, x=1.0, y=2.0)

    dropped = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/0?track=none")
    assert dropped.status_code == 200, dropped.text
    assert dropped.json()["queries"] == []

    refused = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/10?track=none")
    assert refused.status_code == 400


def test_deleting_a_frame_with_no_keyframe_is_a_404(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    response = client.delete(f"/api/projects/{project_id}/points/{point_id}/keyframes/7?track=none")
    assert response.status_code == 404


# -- auto-track -------------------------------------------------------------------


def test_track_none_submits_nothing(client, project) -> None:
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    body = client.post(
        f"/api/projects/{project_id}/points/{point_id}/keyframes",
        json={"frame": 10, "x": 5.0, "y": 6.0, "track": "none"},
    ).json()

    assert body["job_id"] is None
    jobs = client.get(f"/api/projects/{project_id}/jobs").json()
    assert [j for j in jobs if j["kind"] == "track"] == []


def test_auto_track_supersedes_a_queued_run_for_the_same_point(client, project) -> None:
    """A drag emits an edit per stop; only the last one should still be worth running."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))

    # One worker, occupied, so both auto-tracks are still queued when the second
    # arrives — the race this is about is otherwise won by whichever runs first.
    manager = reset_job_manager(1)
    release = threading.Event()

    def _block(progress) -> dict:
        release.wait(timeout=30)
        return {}

    manager.submit(project_id, "proxy", _block)
    deadline = time.time() + 10
    while not manager.running and time.time() < deadline:
        time.sleep(0.01)
    assert manager.running, "the blocking job never started"

    try:
        first = client.post(
            f"/api/projects/{project_id}/points/{point_id}/keyframes",
            json={"frame": 10, "x": 5.0, "y": 6.0},
        ).json()["job_id"]
        second = client.patch(
            f"/api/projects/{project_id}/points/{point_id}/keyframes/10",
            json={"x": 7.0},
        ).json()["job_id"]

        assert first and second and first != second
        assert manager.get(first).status == "cancelled"
        assert manager.get(second).status == "queued"
        # Superseding is per point: another point's queued run is untouched.
        manager.cancel(second)
    finally:
        release.set()


def test_auto_track_leaves_another_points_queued_run_alone(client, project) -> None:
    project_id, source = project
    first_point = place_point(client, project_id, 0, *source.true_position(0))
    second_point = place_point(client, project_id, 0, *source.true_position(0))

    manager = reset_job_manager(1)
    release = threading.Event()

    def _block(progress) -> dict:
        release.wait(timeout=30)
        return {}

    manager.submit(project_id, "proxy", _block)
    deadline = time.time() + 10
    while not manager.running and time.time() < deadline:
        time.sleep(0.01)

    try:
        other = client.post(
            f"/api/projects/{project_id}/points/{first_point}/keyframes",
            json={"frame": 10, "x": 5.0, "y": 6.0},
        ).json()["job_id"]
        mine = client.post(
            f"/api/projects/{project_id}/points/{second_point}/keyframes",
            json={"frame": 10, "x": 5.0, "y": 6.0},
        ).json()["job_id"]

        assert manager.get(other).status == "queued"
        assert manager.get(mine).status == "queued"
        manager.cancel(other)
        manager.cancel(mine)
    finally:
        release.set()
