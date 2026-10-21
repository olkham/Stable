"""Setting a track up tracker first: inputs, settings before a point exists, creation with them.

The UI asks which tracker, then shows that tracker's settings and the tools
for the inputs it takes, and only then lets the operator mark the video. These
are the three server answers that flow needs: what each tracker can be started
from, what its settings are (and the regions they describe) for values not yet
stored anywhere, and a point created with those values already in place.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ._helpers import upload_clip, wait_job


@pytest.fixture
def pid(client: TestClient, tiny_clip) -> str:
    """An uploaded project with its proxy ready."""
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    wait_job(client, detail["id"], detail["proxy_job_id"])
    return str(detail["id"])


def _trackers(client: TestClient) -> dict[str, dict]:
    return {row["name"]: row for row in client.get("/api/trackers").json()}


def test_every_tracker_says_what_a_track_can_start_from(client: TestClient):
    rows = _trackers(client)
    assert rows["lk"]["inputs"] == ["point"]
    assert rows["template"]["inputs"] == ["point"]
    # A box tracker prefers the box it was built for, and still takes a point.
    assert rows["ocean"]["inputs"] == ["box", "point"]
    assert rows["siamdw"]["inputs"] == ["box", "point"]
    for row in rows.values():
        assert ("box" in row["inputs"]) == row["tracks_boxes"]


def test_a_trackers_settings_are_described_before_any_track_uses_it(client: TestClient):
    body = client.get("/api/trackers/template/settings").json()
    assert body["tracker"] == "template"
    assert body["inputs"] == ["point"]
    rows = {row["attribute"]: row for row in body["settings"]}
    assert set(rows) == {"template_radius", "search_radius"}
    assert all(row["value"] == row["default"] and not row["stored"] for row in rows.values())
    assert body["window"] == {
        "radius": rows["template_radius"]["value"],
        "attribute": "template_radius",
    }
    assert body["search"]["radius"] == rows["search_radius"]["value"]


def test_candidate_values_resize_the_regions_the_viewer_previews(client: TestClient):
    body = client.get("/api/trackers/template/settings?template_radius=24&search_radius=60").json()
    rows = {row["attribute"]: row for row in body["settings"]}
    assert rows["template_radius"]["value"] == 24 and rows["template_radius"]["stored"]
    assert body["window"]["radius"] == 24
    assert body["search"]["radius"] == 60


def test_optical_flows_reach_follows_its_pyramid(client: TestClient):
    """LK has no search box; its reach is the window carried up the pyramid."""
    shallow = client.get("/api/trackers/lk/settings?max_level=1").json()["search"]
    deep = client.get("/api/trackers/lk/settings?max_level=3").json()["search"]
    assert deep["radius"] == shallow["radius"] * 4
    assert deep["attribute"] == "max_level"


def test_settings_a_tracker_would_not_act_on_are_refused(client: TestClient):
    assert client.get("/api/trackers/nope/settings").status_code == 404
    bogus = client.get("/api/trackers/template/settings?bogus=3")
    assert bogus.status_code == 400 and "not a settable" in bogus.json()["detail"]
    foreign = client.get("/api/trackers/template/settings?win_size=21")
    assert foreign.status_code == 400 and "has no 'win_size'" in foreign.json()["detail"]
    assert client.get("/api/trackers/template/settings?template_radius=999").status_code == 400
    assert client.get("/api/trackers/template/settings?template_radius=2.5").status_code == 400


def test_a_point_is_created_with_the_settings_chosen_for_it(client: TestClient, pid: str):
    point = client.post(
        f"/api/projects/{pid}/points",
        json={
            "frame": 0,
            "x": 30.0,
            "y": 20.0,
            "tracker": "template",
            "options": {"template_radius": 24, "search_radius": 60},
        },
    ).json()
    assert point["tracker"] == "template"
    assert point["window"]["radius"] == 24 and point["window"]["search_radius"] == 60
    patches = client.get(f"/api/projects/{pid}/points/{point['id']}/patches").json()
    assert patches["custom"] is True
    assert patches["template_radius"] == 24 and patches["search_radius"] == 60


def test_a_refused_setting_leaves_no_point_behind(client: TestClient, pid: str):
    response = client.post(
        f"/api/projects/{pid}/points",
        json={"frame": 0, "x": 30.0, "y": 20.0, "tracker": "lk", "options": {"template_radius": 9}},
    )
    assert response.status_code == 400
    assert client.get(f"/api/projects/{pid}").json()["points"] == []


def test_a_box_track_takes_no_pattern_settings(client: TestClient, pid: str):
    """Its pattern is the box drawn; a stored radius would describe nothing it does."""
    response = client.post(
        f"/api/projects/{pid}/points",
        json={
            "frame": 0,
            "x": 30.0,
            "y": 20.0,
            "w": 12.0,
            "h": 10.0,
            "tracker": "siamdw",
            "options": {"template_radius": 9},
        },
    )
    assert response.status_code == 400
    assert "box" in response.json()["detail"]
    assert client.get(f"/api/projects/{pid}").json()["points"] == []


def test_a_point_created_without_settings_stores_none(client: TestClient, pid: str):
    point = client.post(
        f"/api/projects/{pid}/points",
        json={"frame": 0, "x": 30.0, "y": 20.0, "tracker": "template"},
    ).json()
    patches = client.get(f"/api/projects/{pid}/points/{point['id']}/patches").json()
    assert patches["custom"] is False
