"""Todo 19 — "save and load tracking sessions" — verified, not rebuilt.

The plan said to check this before building anything, because a project already
persists to disk continuously (:mod:`stable.server.storage`) rather than on an
explicit save. That makes "pause and resume without losing progress" a claim
about the *existing* design, and a claim is worth a test: this walks a full
session (points, keyframes, a track, a manual correction, solve settings,
smoothing, render options, template settings), throws the entire server away,
and reopens the project from disk in a new one.

If this passes there is nothing to build for todo 19, and if it ever fails then
"your work is saved" was never true. That is the whole value of the test.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from stable.server.app import create_app
from stable.server.jobs import reset_job_manager
from stable.server.storage import Workspace, reset_workspace

from ._helpers import upload_clip, wait_job


def test_a_whole_session_survives_the_server_being_thrown_away(tmp_path, tiny_clip):
    path, source = tiny_clip
    root = tmp_path / "workspace"
    reset_workspace(root)
    reset_job_manager()

    # -- session one: do a real day's work ---------------------------------
    with TestClient(create_app()) as client:
        detail = upload_clip(client, path, name="Session Test")
        project_id = detail["id"]
        wait_job(client, project_id, detail["proxy_job_id"])

        def ok(response):
            """Every setup call must succeed, or the test proves nothing."""
            assert response.status_code == 200, response.text
            return response

        x, y = source.true_position(0)
        ok(
            client.post(
                f"/api/projects/{project_id}/points",
                json={"frame": 0, "x": x, "y": y, "tracker": "template"},
            )
        )
        mid = source.frame_count // 2
        mx, my = source.true_position(mid)
        ok(
            client.post(
                f"/api/projects/{project_id}/points/p1/keyframes",
                json={"frame": mid, "x": mx, "y": my, "track": "none"},
            )
        )
        ok(
            client.put(
                f"/api/projects/{project_id}/points/p1/patches", json={"template_radius": 6}
            )
        )
        submitted = ok(client.post(f"/api/projects/{project_id}/points/p1/track", json={}))
        assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

        ok(
            client.post(
                f"/api/projects/{project_id}/points/p1/corrections",
                json={"kind": "replace", "frame": 3, "x": 12.5, "y": 13.5},
            )
        )
        ok(client.put(f"/api/projects/{project_id}/solve/settings", json={"reference_frame": 2}))
        before = ok(client.get(f"/api/projects/{project_id}/points/p1/track")).json()

    # -- everything in memory is gone --------------------------------------
    reset_workspace(root)  # same directory on disk, brand-new Workspace object
    reset_job_manager()

    # -- session two: reopen and check nothing was lost ----------------------
    with TestClient(create_app()) as client:
        listed = client.get("/api/projects").json()
        assert [p["id"] for p in listed] == [project_id]
        assert listed[0]["name"] == "Session Test"

        reopened = client.get(f"/api/projects/{project_id}").json()
        assert [p["id"] for p in reopened["points"]] == ["p1"]
        assert reopened["points"][0]["tracker"] == "template"
        assert reopened["solve"]["reference_frame"] == 2

        after = client.get(f"/api/projects/{project_id}/points/p1/track").json()
        # The corrected path, the keyframes it was tracked through, and the
        # manual correction on top of it all came back byte for byte.
        assert after["x"] == before["x"]
        assert after["y"] == before["y"]
        assert after["anchor_frames"] == before["anchor_frames"]
        assert [c["kind"] for c in after["corrections"]] == ["replace"]
        assert after["x"][3] == 12.5

        # Per-point tracker settings live in a sidecar, and came back too.
        template = client.get(f"/api/projects/{project_id}/points/p1/patches").json()
        assert template["template_radius"] == 6

        # And the proxy frames are still servable without regenerating them.
        assert client.get(f"/api/projects/{project_id}/frames/0").status_code == 200


def test_deleting_a_project_really_removes_it_from_disk(tmp_path, tiny_clip):
    """The other half of "load a session": the list must not lie about what exists.

    On Windows this is only true because `delete_project` closes the cached
    video source first — an open handle silently defeats the directory removal.
    """
    path, _source = tiny_clip
    root = tmp_path / "workspace"
    workspace: Workspace = reset_workspace(root)
    reset_job_manager()

    with TestClient(create_app()) as client:
        detail = upload_clip(client, path)
        project_id = detail["id"]
        wait_job(client, project_id, detail["proxy_job_id"])
        # Force the source open, which is what a track or a frame request does.
        assert client.get(f"/api/projects/{project_id}/frames/0").status_code == 200
        assert client.delete(f"/api/projects/{project_id}").status_code == 200

    assert not workspace.project_dir(project_id).exists()
    reset_workspace(root)
    with TestClient(create_app()) as client:
        assert client.get("/api/projects").json() == []
