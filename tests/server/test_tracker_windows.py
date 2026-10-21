"""Per-point window and search sizes, for every tracker that has them.

The size control used to exist only where the plugin happened to call its window
``template_radius``; everywhere else the number was reported and could not be
changed. That is the wrong split — "the feature I am tracking is bigger than
33 px" is a statement about the subject, not about which plugin is selected — so
these tests pin the generalised contract:

#. a setting is offered exactly where the point's own plugin accepts it,
#. in that plugin's own units,
#. and it reaches the run rather than only the panel.

The last one is what makes the rest worth anything: a control that does not
change the next track is worse than no control.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stable.server.tracker_options import (
    SETTINGS,
    constructor_parameters,
    point_options,
    tracker_search,
    tracker_settings,
    tracker_window,
)
from stable.trackers.registry import create_tracker

from ._helpers import upload_clip, wait_job


@pytest.fixture
def project(client: TestClient, tiny_clip):
    """A project with one point, tracker chosen per test."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    return project_id, source


def place(client: TestClient, project_id: str, source, tracker: str) -> str:
    """Place a point on the marker with ``tracker`` selected, and return its id."""
    x, y = source.true_position(0)
    response = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": x, "y": y, "tracker": tracker},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


# -- the table itself ---------------------------------------------------------


def test_every_setting_names_an_argument_some_tracker_actually_takes():
    """A row nothing accepts is a control that could never appear."""
    accepted: set[str] = set()
    for name in (
        "lk", "klt", "template", "correlation", "meanshift", "kalman", "particle", "ocean", "siamdw"
    ):
        accepted |= set(constructor_parameters(name))
    assert {s.attribute for s in SETTINGS} <= accepted


@pytest.mark.parametrize(
    ("tracker", "attribute"),
    [
        ("lk", "win_size"),
        ("klt", "win_size"),
        ("template", "template_radius"),
        ("correlation", "patch_radius"),
        ("meanshift", "window_radius"),
        ("kalman", "template_radius"),
        ("particle", "template_radius"),
    ],
)
def test_each_classic_tracker_offers_its_own_window(tracker, attribute):
    rows = [s for s in tracker_settings(tracker) if s["role"] == "window"]
    assert [row["attribute"] for row in rows] == [attribute]
    assert rows[0]["value"] == rows[0]["default"]
    assert rows[0]["radius"] > 0


def test_a_subclass_that_forwards_kwargs_still_reports_the_window_it_inherits():
    """KLT takes `**kwargs`; reading its own signature alone would see no window."""
    assert tracker_window("klt") == tracker_window("lk")
    assert "win_size" in constructor_parameters("klt")


def test_the_search_area_is_reported_only_where_the_tracker_scans_one():
    # An explicit box that really is scanned.
    for name in ("template", "kalman"):
        radius, attribute, _note = tracker_search(name)
        assert attribute == "search_radius"
        assert radius and radius > 0

    # Optical flow reaches as far as its pyramid carries the window, which is a
    # real number and not a box: reported, attributed, but not a search setting.
    reach, attribute, note = tracker_search("lk")
    assert attribute == "max_level"
    assert reach == (21 // 2) * 2**3
    assert "pyramid" in note

    # Phase correlation searches nothing at all, and says so.
    radius, attribute, note = tracker_search("correlation")
    assert (radius, attribute) == (None, None)
    assert "never searches" in note


def test_a_bigger_window_moves_the_reported_search_reach_with_it():
    """The two are not independent for LK — a wider window reaches further."""
    base, _attribute, _note = tracker_search("lk")
    wider, _attribute, _note = tracker_search("lk", {"win_size": 41})
    assert wider > base


# -- over HTTP ----------------------------------------------------------------


def test_setting_lks_window_persists_and_reaches_the_plugin(client, project, workspace):
    """The setting has to change the track, not just the panel (any tracker)."""
    project_id, source = project
    point_id = place(client, project_id, source, "lk")

    updated = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches", json={"win_size": 41}
    )
    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["radius"] == 20  # a full side reported as a radius
    assert body["custom"] is True

    options = point_options(workspace.project_dir(project_id), point_id)
    assert create_tracker("lk", options=options).win_size == 41

    # ...and a run with it still completes.
    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"


def test_setting_the_search_area_persists_and_reaches_the_plugin(client, project, workspace):
    project_id, source = project
    point_id = place(client, project_id, source, "template")

    response = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches",
        json={"template_radius": 20, "search_radius": 48},
    )
    assert response.status_code == 200, response.text
    assert response.json()["search"]["radius"] == 48
    assert response.json()["window"]["size_px"] == 41

    options = point_options(workspace.project_dir(project_id), point_id)
    plugin = create_tracker("template", options=options)
    assert (plugin.template_radius, plugin.search_radius) == (20, 48)


def test_a_setting_the_point_s_tracker_cannot_take_is_refused(client, project):
    """Silently dropping it would look like it had been applied."""
    project_id, source = project
    point_id = place(client, project_id, source, "lk")

    response = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches", json={"search_radius": 40}
    )
    assert response.status_code == 400
    assert "search_radius" in response.json()["detail"]
    assert "lk" in response.json()["detail"]


def test_an_unknown_setting_is_refused(client, project):
    project_id, source = project
    point_id = place(client, project_id, source, "lk")
    response = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches", json={"update_rate": 0.5}
    )
    assert response.status_code == 400
    assert "settable" in response.json()["detail"]


def test_an_out_of_range_window_is_refused_in_the_plugins_own_units(client, project):
    project_id, source = project
    point_id = place(client, project_id, source, "lk")
    response = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches", json={"win_size": 3}
    )
    assert response.status_code == 400
    assert "win_size" in response.json()["detail"]


def test_clearing_a_setting_restores_the_plugins_own_default(client, project):
    project_id, source = project
    point_id = place(client, project_id, source, "meanshift")

    client.put(f"/api/projects/{project_id}/points/{point_id}/patches", json={"window_radius": 30})
    reset = client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches", json={"window_radius": None}
    )
    assert reset.json()["radius"] == tracker_window("meanshift")[0]
    assert reset.json()["custom"] is False


def test_the_point_summary_carries_the_sizes_the_canvas_draws(client, project):
    """The boxes on the frame come from the project payload, not a fetch per point."""
    project_id, source = project
    point_id = place(client, project_id, source, "template")
    client.put(
        f"/api/projects/{project_id}/points/{point_id}/patches",
        json={"template_radius": 25, "search_radius": 60},
    )

    detail = client.get(f"/api/projects/{project_id}").json()
    window = detail["points"][0]["window"]
    assert window["radius"] == 25
    assert window["search_radius"] == 60
    assert window["attribute"] == "template_radius"

    # ...and the same shape on the points list.
    listed = client.get(f"/api/projects/{project_id}/points").json()
    assert listed[0]["window"] == window


def test_a_deep_trackers_point_offers_no_window_control(client, project):
    """The deep trackers see whole frames; a size slider there would be a lie."""
    project_id, source = project
    point_id = place(client, project_id, source, "cotracker3")

    body = client.get(f"/api/projects/{project_id}/points/{point_id}/patches").json()
    assert body["settings"] == []
    assert body["adjustable"] is False
    assert body["radius_source"] is None
    assert body["search"]["radius"] is None

    # The same "there is no window here" on the point summary, which is what the
    # all-points patch strip filters on (todo 5, sixth round): a tracker with no
    # region to match has no patch to show, so neither the strip nor the card's
    # grid draws one for it. Two views of one fact, and only one place it is
    # decided.
    listed = client.get(f"/api/projects/{project_id}/points").json()
    assert next(p for p in listed if p["id"] == point_id)["window"]["attribute"] is None


@pytest.mark.parametrize("tracker", ["cotracker3", "tapir", "raft", "klt"])
def test_switching_to_another_tracker_does_not_carry_a_setting_it_cannot_take(
    client, project, workspace, tracker
):
    """The reported regression: a stored window size crashed the next tracker.

    Settings follow the point across a tracker change on purpose (switch back
    and the size you chose is still there), so every plugin has to tolerate
    another's. The ones that forward their arguments through ``**kwargs`` — the
    deep trackers over ``DeepTrackerPlugin``, ``klt`` over ``lk`` — used to be
    handed the lot and raised ``__init__() got an unexpected keyword argument``.
    """
    project_id, source = project
    point_id = place(client, project_id, source, "lk")
    client.put(f"/api/projects/{project_id}/points/{point_id}/patches", json={"win_size": 41})

    switched = client.patch(
        f"/api/projects/{project_id}/points/{point_id}", json={"tracker": tracker}
    )
    assert switched.status_code == 200, switched.text

    # The setting is still stored (switching back must not have lost it) ...
    options = point_options(workspace.project_dir(project_id), point_id)
    assert options["win_size"] == 41
    # ... and the plugin builds anyway, which is what tracking does first.
    plugin = create_tracker(tracker, options=options, prepare=False)
    assert plugin.name == tracker

    # The panel is honest about it either way.
    body = client.get(f"/api/projects/{project_id}/points/{point_id}/patches").json()
    assert body["tracker"] == tracker
    assert body["adjustable"] is (tracker == "klt")
