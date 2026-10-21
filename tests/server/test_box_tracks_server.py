"""Box tracks over the HTTP surface: creation, anchors, tracking, edits, patches, export.

A box track is a point track whose anchors carry a width and height. These tests
do not need a deep tracker or its weights: a small fake plugin that declares
``tracks_boxes`` and reports each anchor's box as its per-frame size is
registered for the duration of each test, which exercises everything the server
adds (resolving tracker and sizes, carrying sizes through merge, retrack and
corrections, exporting them) without testing a model.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.types import PointTrack, TrackResult
from stable.server import tracking
from stable.server.storage import Workspace
from stable.server.tracking import _combine_halves, default_box_tracker
from stable.trackers.dummy import DummyTracker
from stable.trackers.registry import register_tracker, registry

from ._helpers import upload_clip, wait_job

BOX = {"w": 14.0, "h": 10.0}


class FakeBoxTracker(DummyTracker):
    """The dummy tracker, declared box-capable: every covered frame gets the anchor's box."""

    name = "fakebox"
    display_name = "Fake box tracker (testing)"
    tracks_boxes = True

    def track(self, video, queries, frame_range, direction="bidir", progress=lambda *_: None):
        """Track like the dummy, then report each query's box on the frames it covered."""
        result = super().track(video, queries, frame_range, direction, progress)
        sizes = np.full((result.n_points, result.n_frames, 2), np.nan, dtype=np.float32)
        for index, query in enumerate(queries):
            if query.size is not None:
                sizes[index, np.isfinite(result.points[index]).all(axis=1)] = query.size
        return TrackResult(
            points=result.points,
            confidence=result.confidence,
            visible=result.visible,
            sizes=sizes if any(q.has_box for q in queries) else None,
        )


@pytest.fixture(autouse=True)
def fake_box_tracker() -> Iterator[None]:
    """Register the fake box plugin for one test."""
    registry.discover()
    register_tracker(FakeBoxTracker, replace=True)
    yield
    registry.unregister("fakebox")


@pytest.fixture
def project_id(client: TestClient, tiny_clip) -> str:
    """An uploaded project with its proxy ready."""
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    wait_job(client, detail["id"], detail["proxy_job_id"])
    return str(detail["id"])


def make_box(client: TestClient, pid: str, *, frame: int = 0, **extra: Any) -> dict[str, Any]:
    """Create a box track on ``fakebox`` and return its summary."""
    body = {"frame": frame, "x": 30.0, "y": 24.0, "tracker": "fakebox", **BOX, **extra}
    response = client.post(f"/api/projects/{pid}/points", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def track(client: TestClient, pid: str, point_id: str) -> dict[str, Any]:
    """Run a full track and return the finished job."""
    submitted = client.post(f"/api/projects/{pid}/points/{point_id}/track", json={})
    assert submitted.status_code == 200, submitted.text
    return wait_job(client, pid, submitted.json()["job_id"], timeout=60)


def seed_sized_track(workspace: Workspace, pid: str, point_id: str) -> None:
    """Give a point a raw track whose box grows by a pixel per frame (w = 10 + frame)."""
    project = workspace.load_project(pid)
    n = project.video.frames
    tracks_dir = workspace.tracks_dir(pid)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(n, dtype=np.float32)
    path = tracks_dir / f"{point_id}.npz"
    PointTrack(
        points=np.stack([np.full(n, 30.0), np.full(n, 24.0)], axis=1).astype(np.float32),
        confidence=np.ones(n, dtype=np.float32),
        visible=np.ones(n, dtype=bool),
        sizes=np.stack([10.0 + frames, 5.0 + frames], axis=1),
    ).save_npz(path)

    def _mutate(proj: Any) -> None:
        proj.point(point_id).raw = proj.relative_path(str(path))

    workspace.update_project(pid, _mutate)


# -- creating a box track -----------------------------------------------------


def test_a_box_point_is_created_with_its_box_and_named_tracker(client, project_id):
    summary = make_box(client, project_id)

    assert summary["box"] is True
    assert summary["tracker"] == "fakebox"
    assert summary["queries"] == [{"frame": 0, "x": 30.0, "y": 24.0, **BOX}]


def test_a_point_without_a_box_is_unchanged(client, project_id):
    response = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    )

    body = response.json()
    assert body["box"] is False
    assert body["tracker"] == "lk"
    assert body["queries"] == [{"frame": 0, "x": 30.0, "y": 24.0}]


def test_an_unnamed_box_tracker_defaults_to_the_first_available_box_tracker(
    client, project_id, monkeypatch
):
    rows = [
        {"name": "ocean", "available": False, "tracks_boxes": True},
        {"name": "fakebox", "available": True, "tracks_boxes": True},
        {"name": "lk", "available": True, "tracks_boxes": False},
    ]
    monkeypatch.setattr(tracking, "describe_trackers", lambda: rows)

    response = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0, **BOX}
    )

    assert response.status_code == 200, response.text
    assert response.json()["tracker"] == "fakebox"


def test_ocean_then_siamdw_are_preferred_when_they_can_run(monkeypatch):
    rows = [
        {"name": "aaa", "available": True, "tracks_boxes": True},
        {"name": "siamdw", "available": True, "tracks_boxes": True},
        {"name": "ocean", "available": True, "tracks_boxes": True},
    ]
    monkeypatch.setattr(tracking, "describe_trackers", lambda: rows)
    assert default_box_tracker() == "ocean"

    rows[2]["available"] = False
    assert default_box_tracker() == "siamdw"

    rows[1]["available"] = False
    assert default_box_tracker() == "aaa"


def test_no_installed_box_tracker_is_a_400(client, project_id, monkeypatch):
    monkeypatch.setattr(
        tracking,
        "describe_trackers",
        lambda: [{"name": "lk", "available": True, "tracks_boxes": False}],
    )

    response = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0, **BOX}
    )

    assert response.status_code == 400
    assert "no installed tracker tracks boxes" in response.json()["detail"]


def test_a_box_with_a_tracker_that_cannot_follow_boxes_is_refused_naming_the_ones_that_can(
    client, project_id
):
    response = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": 30.0, "y": 24.0, "tracker": "lk", **BOX},
    )

    assert response.status_code == 400
    assert "does not track boxes" in response.json()["detail"]
    assert "fakebox" in response.json()["detail"]
    assert client.get(f"/api/projects/{project_id}/points").json() == []


@pytest.mark.parametrize(
    "box",
    [{"w": 10.0}, {"h": 10.0}, {"w": 0.0, "h": 5.0}, {"w": 5.0, "h": -1.0}],
    ids=["w only", "h only", "zero width", "negative height"],
)
def test_a_malformed_box_is_rejected_by_validation(client, project_id, box):
    created = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": 0, "x": 30.0, "y": 24.0, "tracker": "fakebox", **box},
    )
    assert created.status_code == 422

    point_id = make_box(client, project_id)["id"]
    anchored = client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": 5, "x": 31.0, "y": 24.0, **box},
    )
    assert anchored.status_code == 422


# -- anchors ----------------------------------------------------------------------


def test_an_anchor_without_a_box_takes_the_tracks_size_at_that_frame(client, project_id, workspace):
    point_id = make_box(client, project_id)["id"]
    seed_sized_track(workspace, project_id, point_id)

    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": 10, "x": 32.0, "y": 24.0},
    )

    assert response.status_code == 200, response.text
    added = [q for q in response.json()["queries"] if q["frame"] == 10][0]
    # The seeded track is 20 x 15 at frame 10, not the first anchor's 14 x 10.
    assert (added["w"], added["h"]) == (20.0, 15.0)


def test_an_untracked_box_anchor_takes_the_nearest_anchors_size(client, project_id):
    point_id = make_box(client, project_id)["id"]
    base = f"/api/projects/{project_id}/points/{point_id}/anchors"
    client.post(base, json={"frame": 25, "x": 31.0, "y": 24.0, "w": 30.0, "h": 20.0})

    response = client.post(base, json={"frame": 20, "x": 31.0, "y": 24.0})

    added = [q for q in response.json()["queries"] if q["frame"] == 20][0]
    assert (added["w"], added["h"]) == (30.0, 20.0)  # frame 25 is nearer than frame 0


def test_a_given_box_is_used_and_replaces_the_anchor_on_the_same_frame(client, project_id):
    point_id = make_box(client, project_id)["id"]
    base = f"/api/projects/{project_id}/points/{point_id}/anchors"

    response = client.post(base, json={"frame": 0, "x": 31.0, "y": 25.0, "w": 22.0, "h": 18.0})

    assert response.json()["queries"] == [{"frame": 0, "x": 31.0, "y": 25.0, "w": 22.0, "h": 18.0}]


def test_a_box_cannot_be_added_to_a_point_track(client, project_id):
    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()

    response = client.post(
        f"/api/projects/{project_id}/points/{created['id']}/anchors",
        json={"frame": 5, "x": 31.0, "y": 24.0, **BOX},
    )

    assert response.status_code == 400
    assert "point track" in response.json()["detail"]


# -- changing the tracker -------------------------------------------------------------


def test_a_box_track_cannot_switch_to_a_tracker_that_drops_boxes(client, project_id):
    point_id = make_box(client, project_id)["id"]

    refused = client.patch(
        f"/api/projects/{project_id}/points/{point_id}", json={"tracker": "template"}
    )

    assert refused.status_code == 400
    assert "box track" in refused.json()["detail"]
    points = client.get(f"/api/projects/{project_id}/points").json()
    assert points[0]["tracker"] == "fakebox"


def test_a_box_track_may_switch_between_box_trackers_and_a_point_track_to_one(client, project_id):
    point_id = make_box(client, project_id)["id"]
    same = client.patch(
        f"/api/projects/{project_id}/points/{point_id}", json={"tracker": "fakebox"}
    )
    assert same.status_code == 200

    plain = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()
    moved = client.patch(
        f"/api/projects/{project_id}/points/{plain['id']}", json={"tracker": "fakebox"}
    )
    assert moved.status_code == 200
    assert moved.json()["tracker"] == "fakebox"
    assert moved.json()["box"] is False


# -- tracking ----------------------------------------------------------------------


def test_tracking_stores_sizes_and_the_track_payload_returns_them(client, project_id, workspace):
    point_id = make_box(client, project_id)["id"]

    job = track(client, project_id, point_id)

    assert job["status"] == "done", job
    body = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()
    assert body["box"] is True
    assert len(body["w"]) == len(body["h"]) == body["n_frames"]
    assert body["w"][5] == pytest.approx(BOX["w"])
    assert body["h"][5] == pytest.approx(BOX["h"])
    raw = workspace.load_project(project_id).point(point_id).raw
    stored = PointTrack.load_npz(workspace.load_project(project_id).resolve_path(raw))
    assert stored.sizes is not None


def test_a_point_tracks_payload_has_no_sizes(client, project_id):
    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()
    job = track(client, project_id, created["id"])
    assert job["status"] == "done", job

    body = client.get(f"/api/projects/{project_id}/points/{created['id']}/track").json()

    assert body["box"] is False
    assert body["w"] is None
    assert body["h"] is None


def test_unknown_frames_have_null_sizes(client, project_id):
    point_id = make_box(client, project_id, frame=10)["id"]
    client.put(f"/api/projects/{project_id}/range", json={"in": 8, "out": 20})
    track(client, project_id, point_id)

    body = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()

    assert body["w"][0] is None
    assert body["w"][12] == pytest.approx(BOX["w"])


def test_resizing_an_anchor_marks_the_track_stale(client, project_id):
    point_id = make_box(client, project_id)["id"]
    track(client, project_id, point_id)
    summary = client.get(f"/api/projects/{project_id}/points").json()[0]
    assert summary["anchors_changed"] is False

    client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": 0, "x": 30.0, "y": 24.0, "w": 20.0, "h": 10.0},
    )

    summary = client.get(f"/api/projects/{project_id}/points").json()[0]
    assert summary["anchors_changed"] is True


def test_a_hand_edited_box_track_on_a_plain_tracker_fails_with_a_clear_error(
    client, project_id, workspace
):
    point_id = make_box(client, project_id)["id"]

    def _swap(proj: Any) -> None:
        proj.point(point_id).tracker = "lk"

    workspace.update_project(project_id, _swap)

    job = track(client, project_id, point_id)

    assert job["status"] == "error"
    assert "does not track boxes" in job["error"]


# -- retrack ------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["fwd", "bwd", "bidir"])
def test_a_retrack_on_a_box_track_keeps_its_sizes(client, project_id, direction):
    point_id = make_box(client, project_id)["id"]
    track(client, project_id, point_id)

    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 12, "x": 36.0, "y": 24.0, "direction": direction},
    )
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)

    assert job["status"] == "done", job
    corrections = client.get(f"/api/projects/{project_id}/points/{point_id}/corrections").json()
    retrack = [c for c in corrections["corrections"] if c["kind"] == "retrack"][0]
    # Resolved from the anchor/track, since the request did not name a box.
    assert (retrack["w"], retrack["h"]) == (BOX["w"], BOX["h"])
    body = client.get(f"/api/projects/{project_id}/points/{point_id}/track").json()
    assert body["x"][12] == pytest.approx(36.0, abs=0.5)
    assert all(w == pytest.approx(BOX["w"]) for w in body["w"] if w is not None)
    assert body["w"][12] is not None


def test_a_retrack_box_follows_the_displayed_size_not_the_first_anchor(
    client, project_id, workspace
):
    point_id = make_box(client, project_id)["id"]
    seed_sized_track(workspace, project_id, point_id)

    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 10, "x": 30.0, "y": 24.0, "direction": "fwd"},
    )
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)

    assert job["status"] == "done", job
    corrections = client.get(f"/api/projects/{project_id}/points/{point_id}/corrections").json()
    retrack = [c for c in corrections["corrections"] if c["kind"] == "retrack"][0]
    assert (retrack["w"], retrack["h"]) == (20.0, 15.0)


def test_combining_retrack_halves_splices_sizes_with_positions():
    n = 10

    def half(covered: range, width: float) -> PointTrack:
        points = np.full((n, 2), np.nan, dtype=np.float32)
        sizes = np.full((n, 2), np.nan, dtype=np.float32)
        for frame in covered:
            points[frame] = (frame, frame)
            sizes[frame] = (width, width)
        valid = np.isfinite(points).all(axis=1)
        return PointTrack(
            points=points,
            confidence=valid.astype(np.float32),
            visible=valid,
            sizes=sizes,
        )

    combined = _combine_halves(half(range(6, 10), 7.0), half(range(0, 6), 3.0), 5)

    assert combined.sizes is not None
    assert combined.sizes[2, 0] == 3.0  # backward half owns the frames before the query
    assert combined.sizes[6, 0] == 7.0  # forward half owns everything after the query
    assert combined.sizes[8, 0] == 7.0
    assert not np.isnan(combined.sizes[np.isfinite(combined.points).all(axis=1)]).any()


# -- edits --------------------------------------------------------------------------


def test_an_offset_on_a_box_track_promotes_to_an_anchor_that_keeps_the_box(
    client, project_id, workspace
):
    point_id = make_box(client, project_id)["id"]
    seed_sized_track(workspace, project_id, point_id)

    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "offset", "frame": 10, "dx": 3.0, "dy": 0.0},
    )

    assert response.status_code == 200, response.text
    companion = [c for c in response.json()["corrections"] if c["kind"] == "replace"][0]
    assert (companion["w"], companion["h"]) == (20.0, 15.0)
    assert companion["x"] == pytest.approx(33.0)


def test_a_client_replace_without_a_box_gets_the_current_size(client, project_id, workspace):
    point_id = make_box(client, project_id)["id"]
    seed_sized_track(workspace, project_id, point_id)

    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "replace", "frame": 4, "x": 35.0, "y": 25.0},
    )

    replaced = response.json()["corrections"][0]
    assert (replaced["w"], replaced["h"]) == (14.0, 9.0)


def test_point_track_corrections_carry_no_box(client, project_id):
    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()
    track(client, project_id, created["id"])
    base = f"/api/projects/{project_id}/points/{created['id']}/corrections"

    replaced = client.post(base, json={"kind": "replace", "frame": 4, "x": 35.0, "y": 25.0})
    offset = client.post(base, json={"kind": "offset", "frame": 9, "dx": 2.0, "dy": 0.0})

    for correction in offset.json()["corrections"] + replaced.json()["corrections"]:
        assert "w" not in correction
        assert "h" not in correction


# -- patches ------------------------------------------------------------------------


def test_a_box_tracks_patch_is_the_box(client, project_id):
    point_id = make_box(client, project_id)["id"]
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": 5, "x": 30.0, "y": 24.0, "w": 11.0, "h": 21.0},
    )

    body = client.get(f"/api/projects/{project_id}/points/{point_id}/patches").json()

    assert body["adjustable"] is False
    assert body["radius_source"] == "box"
    assert body["settings"] == []
    assert body["radius"] == 11  # half of the longest side (21), rounded up
    assert body["patch_size"] == 23
    assert body["window"]["adjustable"] is False
    assert body["window"]["radius"] == 11
    image = client.get(
        f"/api/projects/{project_id}/points/{point_id}/patches/image", params={"frame": 0}
    )
    assert image.status_code == 200


def test_a_point_tracks_patch_is_not_a_box(client, project_id):
    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()

    body = client.get(f"/api/projects/{project_id}/points/{created['id']}/patches").json()

    assert body["radius_source"] != "box"
    assert body["settings"] != []


# -- export -------------------------------------------------------------------------


def test_exports_carry_w_and_h_columns_filled_only_for_boxes(client, project_id):
    box_id = make_box(client, project_id)["id"]
    plain_id = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 30.0, "y": 24.0}
    ).json()["id"]
    track(client, project_id, box_id)
    track(client, project_id, plain_id)

    response = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "csv"})
    rows = list(csv.DictReader(io.StringIO(response.text)))

    assert list(rows[0]) == ["point_id", "frame", "x", "y", "w", "h", "confidence", "visible"]
    boxed = [r for r in rows if r["point_id"] == box_id and r["frame"] == "5"][0]
    assert (float(boxed["w"]), float(boxed["h"])) == (BOX["w"], BOX["h"])
    plain = [r for r in rows if r["point_id"] == plain_id and r["frame"] == "5"][0]
    assert plain["w"] == "" and plain["h"] == ""

    data = client.get(f"/api/projects/{project_id}/export/tracks", params={"format": "json"})
    entries = data.json()["points"]
    assert [e["w"] for e in entries if e["point_id"] == plain_id and e["frame"] == 5] == [None]
    got = [e for e in entries if e["point_id"] == box_id and e["frame"] == 5][0]
    assert got["w"] == pytest.approx(BOX["w"])
    assert not math.isnan(got["h"])
