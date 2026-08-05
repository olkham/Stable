"""Deleting a track takes everything it owned with it (todo 9).

The report was "when I delete a point and add a new one back it restores the
previous track". Two independent causes, both fixed here:

#. ``delete_point`` removed ``tracks/<id>.npz`` and its report and **nothing
   else** — the retrack runs, the anchor-violation sidecar and the point's
   stored tracker settings all survived; and
#. ``new_point_id`` handed out the *lowest free* ``pN``, so the next point
   placed was given the dead one's name and inherited whatever was left.

(The third cause was the browser's own cached track, which is client-side and
fixed in ``web/src/components/TracksPanel.tsx``.)
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from stable.core.project import Project
from stable.server.storage import PROJECT_FILENAME
from stable.server.tracker_options import TRACKER_OPTIONS_CACHE

from ._helpers import upload_clip, wait_job


@pytest.fixture
def tracked(client: TestClient, tiny_clip):
    """A project with one tracked point that has settings and a correction."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    x, y = source.true_position(0)
    created = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": x, "y": y, "tracker": "template"},
    )
    point_id = created.json()["id"]
    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"
    client.put(f"/api/projects/{project_id}/points/{point_id}/patches", json={"template_radius": 9})
    return project_id, point_id, source


def test_deleting_a_point_removes_every_file_named_for_it(client, tracked, workspace):
    project_id, point_id, _source = tracked
    tracks_dir = workspace.tracks_dir(project_id)
    assert list(tracks_dir.glob(f"{point_id}.*")), "nothing to delete makes this test vacuous"

    response = client.delete(f"/api/projects/{project_id}/points/{point_id}")
    assert response.status_code == 200, response.text
    assert response.json()["removed"], "the sweep should report what it removed"

    assert list(tracks_dir.glob(f"{point_id}.*")) == []
    sidecar = workspace.project_dir(project_id) / TRACKER_OPTIONS_CACHE
    stored = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    assert point_id not in stored


def test_a_retrack_run_does_not_outlive_its_point(client, tracked, workspace):
    """The sidecar this used to leave behind is the one that carried a track back."""
    project_id, point_id, _source = tracked
    retrack = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 5, "direction": "fwd"},
    )
    assert retrack.status_code == 200, retrack.text
    wait_job(client, project_id, retrack.json()["job_id"])
    tracks_dir = workspace.tracks_dir(project_id)
    assert len(list(tracks_dir.glob(f"{point_id}.*"))) > 2, "expected a retrack sidecar"

    client.delete(f"/api/projects/{project_id}/points/{point_id}")
    assert list(tracks_dir.glob(f"{point_id}.*")) == []


def test_a_new_point_inherits_nothing_from_the_one_it_replaces(client, tracked):
    """The bug as the operator sees it: delete, re-add, and the old track is back.

    Note what is *not* asserted: that the id differs. Emptying a project does
    start numbering again at ``p1`` — and that is safe precisely because there
    is nothing left on disk for a reused name to pick up. The guarantee worth
    having is this one: a new track starts empty.
    """
    project_id, point_id, source = tracked
    client.delete(f"/api/projects/{project_id}/points/{point_id}")

    x, y = source.true_position(0)
    replacement = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y}
    ).json()

    assert replacement["has_track"] is False
    assert replacement["tracked_frames"] == 0
    assert (
        client.get(f"/api/projects/{project_id}/points/{replacement['id']}/track").status_code
        == 404
    )
    # ...and no inherited window setting either.
    patches = client.get(f"/api/projects/{project_id}/points/{replacement['id']}/patches").json()
    assert patches["custom"] is False


def test_a_replacement_alongside_a_survivor_gets_a_fresh_id(client, tracked):
    """While anything remains, ids count forward rather than filling the gap."""
    project_id, point_id, source = tracked
    x, y = source.true_position(0)
    survivor = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x + 5, "y": y + 5}
    ).json()["id"]

    client.delete(f"/api/projects/{project_id}/points/{point_id}")
    replacement = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y}
    ).json()

    assert replacement["id"] not in {point_id, survivor}


def test_deleting_one_point_leaves_its_neighbours_alone(client, tracked, workspace):
    """`p1` must not take `p10`'s files, and a live point must keep its own."""
    project_id, first, source = tracked
    x, y = source.true_position(0)
    second = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x + 3, "y": y + 3}
    ).json()["id"]
    submitted = client.post(f"/api/projects/{project_id}/points/{second}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

    client.delete(f"/api/projects/{project_id}/points/{first}")

    remaining = client.get(f"/api/projects/{project_id}/points").json()
    assert [p["id"] for p in remaining] == [second]
    assert remaining[0]["has_track"] is True
    assert list(workspace.tracks_dir(project_id).glob(f"{second}.*"))


def test_ids_count_forward_rather_than_filling_gaps():
    """A unit check of the rule, independent of the HTTP layer."""
    project = Project()
    first = project.new_point_id()
    assert first == "p1"

    from stable.core.project import PointSpec

    project.points.append(PointSpec(id=first))
    second = project.new_point_id()
    project.points.append(PointSpec(id=second))
    assert second == "p2"

    # Delete the first: the next id must not be the one just freed.
    project.points = [p for p in project.points if p.id != "p1"]
    assert project.new_point_id() == "p3"


def test_the_project_file_still_holds_only_the_surviving_points(client, tracked, workspace):
    project_id, point_id, _source = tracked
    client.delete(f"/api/projects/{project_id}/points/{point_id}")
    project = json.loads(
        (workspace.project_dir(project_id) / PROJECT_FILENAME).read_text(encoding="utf-8")
    )
    assert project["points"] == []


def test_deleting_an_unknown_point_is_a_404(client, tracked):
    project_id, _point_id, _source = tracked
    assert client.delete(f"/api/projects/{project_id}/points/nope").status_code == 404
