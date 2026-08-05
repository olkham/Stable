"""Template size and template patches (todo 14).

The patch grid is only useful if it shows what the *tracker* would build, so
these tests pin the two places that could drift apart: the panel's "available"
flag against :class:`~stable.trackers.template.TemplateTracker`'s own edge rule,
and the stored radius against the plugin the run actually constructs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.server.storage import PROJECT_FILENAME
from stable.server.tracker_options import TRACKER_OPTIONS_CACHE
from stable.trackers.registry import create_tracker
from stable.trackers.template import TemplateTracker

from ._helpers import decode_jpeg, upload_clip, wait_job


@pytest.fixture
def template_point(client: TestClient, tiny_clip):
    """A project with one template-tracked point anchored on the marker."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    x, y = source.true_position(0)
    client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": x, "y": y, "tracker": "template"},
    )
    return project_id, source


def test_defaults_come_from_the_plugin_not_a_duplicated_constant(client, template_point):
    project_id, _source = template_point
    body = client.get(f"/api/projects/{project_id}/points/p1/template").json()

    plugin = TemplateTracker()
    assert body["template_radius"] == plugin.template_radius
    assert body["defaults"]["template_radius"] == plugin.template_radius
    assert body["patch_size"] == plugin.template_radius * 2 + 1
    assert body["custom"] is False


def test_setting_a_radius_persists_and_is_reported_back(client, template_point, workspace):
    project_id, _source = template_point
    updated = client.put(
        f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 7}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["template_radius"] == 7
    assert updated.json()["patch_size"] == 15
    assert updated.json()["custom"] is True

    # Stored in a sidecar, so the v1 project schema is untouched.
    sidecar = workspace.project_dir(project_id) / TRACKER_OPTIONS_CACHE
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"p1": {"template_radius": 7}}
    project_file = workspace.project_dir(project_id) / PROJECT_FILENAME
    project = json.loads(project_file.read_text(encoding="utf-8"))
    assert "template_radius" not in json.dumps(project["points"])


def test_a_null_radius_restores_the_plugin_default(client, template_point):
    project_id, _source = template_point
    client.put(f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 30})
    reset = client.put(
        f"/api/projects/{project_id}/points/p1/template", json={"template_radius": None}
    )
    assert reset.json()["template_radius"] == TemplateTracker().template_radius
    assert reset.json()["custom"] is False


def test_an_out_of_range_radius_is_refused(client, template_point):
    project_id, _source = template_point
    response = client.put(
        f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 9999}
    )
    assert response.status_code == 400
    assert "template_radius" in response.json()["detail"]


def test_the_patch_is_the_pixels_the_tracker_would_match(client, template_point, workspace):
    """The thumbnail is the real patch, not an approximation of it."""
    project_id, _source = template_point
    client.put(f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 5})

    response = client.get(f"/api/projects/{project_id}/points/p1/template/patch?frame=0")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    patch = decode_jpeg(response.content)  # cv2.imdecode handles PNG too
    assert patch.shape == (11, 11, 3)

    # The same crop the tracker takes: centred on the anchor, 2r+1 square.
    project = workspace.load_project(project_id)
    query = project.point("p1").queries[0]
    video = workspace.get_video_source(project_id)
    cx, cy = int(round(query.x)), int(round(query.y))
    expected = np.asarray(video.get_frame(0))[cy - 5 : cy + 6, cx - 5 : cx + 6]
    assert np.array_equal(patch, expected)


def test_the_grid_previews_a_size_before_it_is_saved(client, template_point):
    """Dragging the slider must show the consequence without committing to it."""
    project_id, _source = template_point
    preview = client.get(f"/api/projects/{project_id}/points/p1/template/patch?frame=0&radius=3")
    assert preview.status_code == 200
    assert decode_jpeg(preview.content).shape == (7, 7, 3)
    # ...and the stored value is untouched by the preview.
    assert client.get(f"/api/projects/{project_id}/points/p1/template").json()["custom"] is False


def test_a_keyframe_too_close_to_the_edge_is_reported_not_hidden(client, tiny_clip):
    """A patch the tracker would refuse must be flagged, with the reason."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    # Two pixels in from the corner: no patch of any useful size fits here.
    client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": 2.0, "y": 2.0, "tracker": "template"},
    )

    body = client.get(f"/api/projects/{project_id}/points/p1/template").json()
    assert body["anchors"][0]["available"] is False

    patch = client.get(f"/api/projects/{project_id}/points/p1/template/patch?frame=0")
    assert patch.status_code == 400
    assert "outside the frame" in patch.json()["detail"]


def test_every_keyframe_gets_an_entry_in_the_grid(client, template_point):
    project_id, source = template_point
    for frame in (10, 20):
        x, y = source.true_position(frame)
        client.post(
            f"/api/projects/{project_id}/points/p1/anchors",
            json={"frame": frame, "x": x, "y": y},
        )
    body = client.get(f"/api/projects/{project_id}/points/p1/template").json()
    assert [a["frame"] for a in body["anchors"]] == [0, 10, 20]
    for anchor in body["anchors"]:
        assert client.get(
            f"/api/projects/{project_id}/points/p1/template/patch?frame={anchor['frame']}"
        ).status_code == (200 if anchor["available"] else 400)


def test_a_stored_radius_reaches_the_plugin_that_runs(client, template_point, workspace):
    """The setting has to change the track, not just the panel."""
    project_id, _source = template_point
    client.put(f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 9})

    from stable.server.tracker_options import point_options

    options = point_options(workspace.project_dir(project_id), "p1")
    plugin = create_tracker("template", options=options)
    assert isinstance(plugin, TemplateTracker)
    assert plugin.template_radius == 9


def test_options_a_plugin_does_not_accept_are_ignored_not_fatal(client, template_point, workspace):
    """Settings follow the point, so a tracker swap must not break tracking."""
    project_id, _source = template_point
    client.put(f"/api/projects/{project_id}/points/p1/template", json={"template_radius": 9})
    client.patch(f"/api/projects/{project_id}/points/p1", json={"tracker": "lk"})

    from stable.server.tracker_options import point_options

    options = point_options(workspace.project_dir(project_id), "p1")
    plugin = create_tracker("lk", options=options)  # must not raise
    assert not hasattr(plugin, "template_radius")

    submitted = client.post(f"/api/projects/{project_id}/points/p1/track", json={})
    job = wait_job(client, project_id, submitted.json()["job_id"])
    assert job["status"] == "done", job


def test_the_panel_is_scoped_to_a_real_point(client, template_point):
    project_id, _source = template_point
    assert client.get(f"/api/projects/{project_id}/points/nope/template").status_code == 404
    assert (
        client.get(f"/api/projects/{project_id}/points/p1/template/patch?frame=999").status_code
        == 404
    )
