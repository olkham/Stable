"""Deriving a track from other tracks, over HTTP (todo 2, tenth round).

The maths is :mod:`stable.core.derive`'s and is tested there. These are about
the route: that the path it writes really is the per-frame midpoint of the
tracks named, that it is not time-warped by a source covering fewer frames
(the reported bug), and that the two span modes cover what they say.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.types import PointTrack

from ._helpers import seed_tracks, upload_clip, wait_job


def moving_track(n_frames: int, *, end: int, offset: float, y: float) -> PointTrack:
    """A point following a shared motion, offset sideways, stopping at ``end``."""
    motion = np.linspace(0.0, 20.0, n_frames) + np.sin(np.linspace(0.0, 5.0, n_frames)) * 4.0
    points = np.full((n_frames, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(n_frames, dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)
    points[: end + 1, 0] = motion[: end + 1] + offset
    points[: end + 1, 1] = y
    confidence[: end + 1] = 1.0
    visible[: end + 1] = True
    return PointTrack(points=points, confidence=confidence, visible=visible)


@pytest.fixture
def two_tracks(client: TestClient, workspace, tiny_clip):
    """Two tracks moving together, one covering half as many frames as the other.

    Deliberately mismatched: two features in one shot share a clock, and the
    thing that must not happen is the shorter one being replayed at a different
    speed so the midpoint drifts off the line joining them.
    """
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    n_frames = source.frame_count
    ids = seed_tracks(
        workspace,
        project_id,
        [
            moving_track(n_frames, end=max(2, n_frames // 2), offset=10.0, y=8.0),
            moving_track(n_frames, end=n_frames - 1, offset=30.0, y=20.0),
        ],
    )
    return project_id, ids, n_frames


def track_data(client: TestClient, project_id: str, point_id: str) -> dict:
    response = client.get(f"/api/projects/{project_id}/points/{point_id}/track")
    assert response.status_code == 200, response.text
    return dict(response.json())


def derive(client: TestClient, project_id: str, ids: list[str], **body) -> dict:
    response = client.post(
        f"/api/projects/{project_id}/points/derive", json={"point_ids": ids, **body}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_a_derived_track_sits_midway_between_its_sources_on_every_frame(
    client: TestClient, two_tracks
) -> None:
    """The report: draw the line joining them on a frame, take its midpoint."""
    project_id, ids, _n = two_tracks
    summary = derive(client, project_id, ids, mode="overlap")
    assert summary["id"] not in ids
    assert summary["derived_from"] == ids
    assert summary["has_track"] is True

    left = track_data(client, project_id, ids[0])
    right = track_data(client, project_id, ids[1])
    derived = track_data(client, project_id, summary["id"])
    start, end = summary["derived_frames"]
    assert end > start

    for frame in range(start, end + 1):
        assert derived["x"][frame] == pytest.approx(
            (left["x"][frame] + right["x"][frame]) / 2, abs=1e-3
        )
        assert derived["y"][frame] == pytest.approx(
            (left["y"][frame] + right["y"][frame]) / 2, abs=1e-3
        )


def test_the_derived_track_is_not_time_warped_by_the_shorter_source(
    client: TestClient, two_tracks
) -> None:
    """A source covering fewer frames must not be stretched across more of them.

    Stretching it is what made the derived path lead the sources early and lag
    them late. The two seeded tracks move identically, so the midpoint's own
    frame-to-frame motion must match theirs exactly — any resampling of either
    would change its speed.
    """
    project_id, ids, _n = two_tracks
    summary = derive(client, project_id, ids)
    start, end = summary["derived_frames"]

    source = track_data(client, project_id, ids[0])["x"][start : end + 1]
    derived = track_data(client, project_id, summary["id"])["x"][start : end + 1]
    steps_source = [b - a for a, b in zip(source, source[1:], strict=False)]
    steps_derived = [b - a for a, b in zip(derived, derived[1:], strict=False)]
    assert steps_derived == pytest.approx(steps_source, abs=1e-3)


def test_overlap_covers_only_the_frames_both_sources_reach(client: TestClient, two_tracks) -> None:
    project_id, ids, n_frames = two_tracks
    summary = derive(client, project_id, ids, mode="overlap")
    start, end = summary["derived_frames"]

    derived = track_data(client, project_id, summary["id"])
    covered = [f for f, value in enumerate(derived["x"]) if value is not None]
    assert covered == list(range(start, end + 1))
    # The shorter source stops half way, so the overlap must too.
    assert end < n_frames - 1


def test_union_covers_every_frame_either_source_reaches(client: TestClient, two_tracks) -> None:
    project_id, ids, n_frames = two_tracks
    summary = derive(client, project_id, ids, mode="union")
    assert summary["derived_frames"] == [0, n_frames - 1]

    # Past the short track's end the mean is simply the one still there.
    derived = track_data(client, project_id, summary["id"])
    long_track = track_data(client, project_id, ids[1])
    assert derived["x"][n_frames - 1] == pytest.approx(long_track["x"][n_frames - 1], abs=1e-3)


def test_a_derived_track_is_anchored_where_its_path_starts(client: TestClient, two_tracks) -> None:
    """A point with no anchor cannot be tracked, re-tracked or extended."""
    project_id, ids, _n = two_tracks
    summary = derive(client, project_id, ids)
    start = summary["derived_frames"][0]
    assert [q["frame"] for q in summary["queries"]] == [start]

    derived = track_data(client, project_id, summary["id"])
    assert summary["queries"][0]["x"] == pytest.approx(derived["x"][start], abs=1e-3)


def test_what_a_composite_is_made_of_survives_a_reload(
    client: TestClient, workspace, two_tracks
) -> None:
    """The panel needs this every time it draws the card, not just once.

    It began as a key on the create response alone, which meant a reload turned
    a composite back into what looked like an ordinary tracked point — tracker
    picker, window sizes, anchors and all.
    """
    project_id, ids, _n = two_tracks
    point_id = derive(client, project_id, ids)["id"]

    reopened = client.get(f"/api/projects/{project_id}").json()
    composite = next(p for p in reopened["points"] if p["id"] == point_id)
    assert composite["derived_from"] == ids
    # ...and an ordinary tracked point is plainly not one.
    assert next(p for p in reopened["points"] if p["id"] == ids[0])["derived_from"] == []

    listed = client.get(f"/api/projects/{project_id}/points").json()
    assert next(p for p in listed if p["id"] == point_id)["derived_from"] == ids


def test_a_composite_that_is_tracked_stops_being_one(client: TestClient, two_tracks) -> None:
    """Its path is then that run's, not the average of anything.

    The panel hides the button that gets here, but the route is open — and a
    point still claiming to be made of other tracks while holding a freshly
    tracked path would be a lie the panel would go on drawing.
    """
    project_id, ids, _n = two_tracks
    point_id = derive(client, project_id, ids)["id"]

    submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job

    reopened = client.get(f"/api/projects/{project_id}").json()
    assert next(p for p in reopened["points"] if p["id"] == point_id)["derived_from"] == []


def test_a_deleted_source_is_remembered_rather_than_pruned(client: TestClient, two_tracks) -> None:
    """ "Made from p1 and p2" is what happened; the baked path did not change."""
    project_id, ids, _n = two_tracks
    point_id = derive(client, project_id, ids)["id"]
    before = track_data(client, project_id, point_id)

    assert client.delete(f"/api/projects/{project_id}/points/{ids[0]}").status_code == 200

    reopened = client.get(f"/api/projects/{project_id}").json()
    composite = next(p for p in reopened["points"] if p["id"] == point_id)
    assert composite["derived_from"] == ids
    assert track_data(client, project_id, point_id)["x"] == pytest.approx(before["x"], abs=1e-6)


def test_a_derived_track_survives_a_reload_and_can_be_deleted_with_its_file(
    client: TestClient, workspace, two_tracks
) -> None:
    project_id, ids, _n = two_tracks
    point_id = derive(client, project_id, ids)["id"]

    reopened = client.get(f"/api/projects/{project_id}").json()
    assert point_id in [p["id"] for p in reopened["points"]]
    assert (workspace.tracks_dir(project_id) / f"{point_id}.npz").is_file()

    removed = client.delete(f"/api/projects/{project_id}/points/{point_id}")
    assert removed.status_code == 200
    assert f"{point_id}.npz" in removed.json()["removed"]
    assert not (workspace.tracks_dir(project_id) / f"{point_id}.npz").exists()


def test_deriving_needs_two_different_tracked_sources(client: TestClient, two_tracks) -> None:
    project_id, ids, _n = two_tracks

    one = client.post(f"/api/projects/{project_id}/points/derive", json={"point_ids": ids[:1]})
    assert one.status_code == 400
    assert "at least two" in one.json()["detail"]

    # The same track twice is one track, not two: counting it twice would weight
    # it twice, which is a different operation from the one being asked for.
    twice = client.post(
        f"/api/projects/{project_id}/points/derive", json={"point_ids": [ids[0], ids[0]]}
    )
    assert twice.status_code == 400

    missing = client.post(
        f"/api/projects/{project_id}/points/derive", json={"point_ids": [ids[0], "nope"]}
    )
    assert missing.status_code == 404


def test_deriving_from_an_untracked_point_says_so(client: TestClient, two_tracks) -> None:
    project_id, ids, _n = two_tracks
    fresh = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 5.0, "y": 5.0}
    ).json()

    response = client.post(
        f"/api/projects/{project_id}/points/derive", json={"point_ids": [ids[0], fresh["id"]]}
    )
    assert response.status_code == 400
    assert "no tracked frames" in response.json()["detail"]


def test_sources_that_share_no_frames_are_refused_with_the_way_out(
    client: TestClient, workspace, tiny_clip
) -> None:
    """Overlap cannot average tracks that never coexist, so it names what can."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    n_frames = source.frame_count
    half = n_frames // 2

    def half_track(start: int, end: int) -> PointTrack:
        points = np.full((n_frames, 2), np.nan, dtype=np.float32)
        confidence = np.zeros(n_frames, dtype=np.float32)
        visible = np.zeros(n_frames, dtype=bool)
        points[start:end, 0] = 12.0
        points[start:end, 1] = 9.0
        confidence[start:end] = 1.0
        visible[start:end] = True
        return PointTrack(points=points, confidence=confidence, visible=visible)

    ids = seed_tracks(workspace, project_id, [half_track(0, half), half_track(half, n_frames)])

    refused = client.post(
        f"/api/projects/{project_id}/points/derive",
        json={"point_ids": ids, "mode": "overlap"},
    )
    assert refused.status_code == 400
    assert "share no frames" in refused.json()["detail"]

    # ...and the way out it names actually works.
    joined = derive(client, project_id, ids, mode="union")
    assert joined["derived_frames"] == [0, n_frames - 1]
