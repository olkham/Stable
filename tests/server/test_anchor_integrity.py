"""W1: a user-placed keyframe is ground truth and is never silently overridden.

Todo 13 is the bug these tests were written against: anchors the operator placed
up front were ignored by tracking runs started elsewhere in the clip. The merge
(:mod:`stable.trackers.merge`) has guaranteed exact anchor passthrough since
Phase 0 — what was missing was the *server* handing every anchor to the run in
the first place, so a retrack simply spliced straight over them.

The other todos in the cluster are the same idea from different angles and are
covered here too: promotion of manual corrections (11), bounded runs (12, 20)
and the flagged-anchor signal the timeline draws red (15).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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
    """Create a point with its first anchor and return its id."""
    response = client.post(
        f"/api/projects/{project_id}/points",
        json={"frame": frame, "x": x, "y": y, "tracker": TRACKER},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def add_anchor(client: TestClient, project_id: str, point_id: str, frame, x, y) -> None:
    """Pin an extra anchor keyframe onto a point."""
    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": frame, "x": x, "y": y},
    )
    assert response.status_code == 200, response.text


def track(client: TestClient, project_id: str, point_id: str, **body) -> dict:
    """Run the full multi-anchor tracking job and return the finished job."""
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/track",
        json={"tracker": TRACKER, **body},
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    return dict(job)


def retrack(client: TestClient, project_id: str, point_id: str, **body) -> dict:
    """Run a retrack job and return the finished job."""
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"tracker": TRACKER, **body},
    )
    assert submitted.status_code == 200, submitted.text
    return dict(wait_job(client, project_id, submitted.json()["job_id"], timeout=120))


def track_of(client: TestClient, project_id: str, point_id: str) -> dict:
    """Fetch a point's per-frame track payload."""
    response = client.get(f"/api/projects/{project_id}/points/{point_id}/track")
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- todo 13: an anchor is never silently overridden ------------------------------


def test_a_retrack_honours_every_anchor_it_passes(client, project) -> None:
    """[BUG 13] A retrack started before an anchor used to splice straight over it.

    The operator's anchor at frame 15 is deliberately 12 px away from where the
    tracker would go, which is the whole point: it is the *user's* word about
    where the feature is, and a run started at frame 5 must not overwrite it.
    """
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)

    ax, ay = source.true_position(15)
    ax, ay = ax + 12.0, ay
    add_anchor(client, project_id, point_id, 15, ax, ay)
    track(client, project_id, point_id)

    merged = track_of(client, project_id, point_id)
    assert (merged["x"][15], merged["y"][15]) == pytest.approx((ax, ay), abs=1e-3)

    x5, y5 = source.true_position(5)
    retrack(client, project_id, point_id, frame=5, x=x5, y=y5, direction="fwd")

    after = track_of(client, project_id, point_id)
    assert (after["x"][15], after["y"][15]) == pytest.approx((ax, ay), abs=1e-3), (
        "the retrack overwrote a user anchor"
    )


def test_a_disagreeing_run_stops_at_the_anchor_and_reports_it(client, project) -> None:
    """A run that reaches an anchor it disagrees with stops there and says so."""
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)

    ax, ay = source.true_position(15)
    add_anchor(client, project_id, point_id, 15, ax + 12.0, ay)
    track(client, project_id, point_id)

    x5, y5 = source.true_position(5)
    job = retrack(client, project_id, point_id, frame=5, x=x5, y=y5, direction="fwd")

    violations = job["result"]["anchor_violations"]
    assert [v["frame"] for v in violations] == [15]
    assert violations[0]["distance"] > 6.0
    assert job["result"]["stopped_at"] == 15

    after = track_of(client, project_id, point_id)
    # The disagreement outlives the job: it is what the timeline paints red.
    assert 15 in after["flagged_anchors"]
    assert any(v["frame"] == 15 and v["distance"] > 6.0 for v in after["anchor_violations"])


def test_an_agreeing_run_carries_straight_on_through_a_keyframe(client, project) -> None:
    """The guard is about disagreement, not about anchors as such."""
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    add_anchor(client, project_id, point_id, 15, *source.true_position(15))
    track(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=5, direction="fwd")
    assert job["result"]["anchor_violations"] == []
    assert job["result"]["stopped_at"] is None
    assert job["result"]["covered_range"][1] > 15


# -- todo 11: a manual correction becomes a ground-truth keyframe -----------------


def test_a_drag_is_promoted_to_a_ground_truth_keyframe(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)
    before = track_of(client, project_id, point_id)

    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "offset", "frame": 10, "dx": 4.0, "dy": -3.0, "falloff_before": 3},
    )
    assert response.status_code == 200, response.text

    after = track_of(client, project_id, point_id)
    assert [c["kind"] for c in after["corrections"]] == ["offset", "replace"]
    expected = (before["x"][10] + 4.0, before["y"][10] - 3.0)
    assert (after["x"][10], after["y"][10]) == pytest.approx(expected, abs=1e-3)
    assert [q["frame"] for q in after["ground_truth"]] == [0, 10]
    assert (after["ground_truth"][1]["x"], after["ground_truth"][1]["y"]) == pytest.approx(
        expected, abs=1e-3
    )


def test_a_retrack_does_not_overwrite_an_earlier_manual_correction(client, project) -> None:
    """The corrected frame is honoured by a later run started before it."""
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)

    fixed = (30.0, 12.0)
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "replace", "frame": 18, "x": fixed[0], "y": fixed[1]},
    )
    retrack(client, project_id, point_id, frame=4, direction="fwd")

    after = track_of(client, project_id, point_id)
    assert (after["x"][18], after["y"][18]) == pytest.approx(fixed, abs=1e-3)
    assert 18 in after["flagged_anchors"]


def test_a_re_track_plans_through_the_promoted_keyframe(client, project) -> None:
    """The point of promotion: the *next* full track passes through the fix."""
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)

    fixed = (26.0, 30.0)
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "replace", "frame": 12, "x": fixed[0], "y": fixed[1]},
    )
    job = track(client, project_id, point_id)
    assert 12 in job["result"]["anchor_frames"]

    after = track_of(client, project_id, point_id)
    assert (after["raw_x"][12], after["raw_y"][12]) == pytest.approx(fixed, abs=1e-3)


def test_undo_un_promotes_the_keyframe_in_one_step(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)

    client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "offset", "frame": 9, "dx": 5.0, "dy": 0.0},
    )
    assert [q["frame"] for q in track_of(client, project_id, point_id)["ground_truth"]] == [0, 9]

    undone = client.post(f"/api/projects/{project_id}/edits/undo")
    assert undone.status_code == 200
    restored = track_of(client, project_id, point_id)
    assert restored["corrections"] == []
    assert [q["frame"] for q in restored["ground_truth"]] == [0]


# -- todos 12 and 20: bounded runs -------------------------------------------------


def test_a_run_can_stop_at_the_next_keyframe(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    add_anchor(client, project_id, point_id, 20, *source.true_position(20))
    track(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=6, direction="fwd", stop_at_keyframe=True)
    assert job["result"]["frame_range"] == [6, 21]
    assert job["result"]["covered_range"] == [6, 20]


def test_a_run_can_stop_at_the_previous_keyframe(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    add_anchor(client, project_id, point_id, 8, *source.true_position(8))
    track(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=20, direction="bwd", stop_at_keyframe=True)
    assert job["result"]["frame_range"] == [8, 21]
    assert job["result"]["covered_range"] == [8, 20]


def test_a_run_can_be_limited_to_a_number_of_frames(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)
    baseline = track_of(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=5, direction="fwd", max_frames=4)
    assert job["result"]["frame_range"] == [5, 10]
    assert job["result"]["covered_range"] == [5, 9]

    after = track_of(client, project_id, point_id)
    # Beyond the bound the merged baseline is untouched: a limited retrack is a
    # limited *edit*, not a whole-clip one that happens to stop early.
    assert after["x"][12] == pytest.approx(baseline["x"][12], abs=1e-6)


def test_a_bidirectional_run_is_limited_in_both_directions(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=14, direction="bidir", max_frames=3)
    assert job["result"]["frame_range"] == [11, 18]
    assert job["result"]["covered_range"] == [11, 17]


def test_a_non_positive_frame_limit_is_rejected(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"frame": 5, "x": 1.0, "y": 1.0, "max_frames": 0},
    )
    assert response.status_code == 422


def test_a_retrack_with_no_position_starts_from_where_the_point_is(client, project) -> None:
    project_id, source = project
    x0, y0 = source.true_position(0)
    point_id = place_point(client, project_id, 0, x0, y0)
    track(client, project_id, point_id)
    baseline = track_of(client, project_id, point_id)

    job = retrack(client, project_id, point_id, frame=7, direction="fwd", max_frames=2)
    assert job["result"]["x"] == pytest.approx(baseline["x"][7], abs=1e-3)
    assert job["result"]["y"] == pytest.approx(baseline["y"][7], abs=1e-3)


# -- todo 15: every keyframe must be intersected, or be flagged --------------------


def test_every_keyframe_is_intersected_or_flagged(client, project) -> None:
    """Tracking all keyframes forwards and backwards, with the misses named."""
    project_id, source = project
    frames = (0, 10, 20, 29)
    x0, y0 = source.true_position(frames[0])
    point_id = place_point(client, project_id, frames[0], x0, y0)
    for frame in frames[1:-1]:
        add_anchor(client, project_id, point_id, frame, *source.true_position(frame))
    # The last keyframe is 25 px from where the feature actually is, so no run
    # can reach it: it is exactly the "needs manual review" case.
    bad_x, bad_y = source.true_position(frames[-1])
    add_anchor(client, project_id, point_id, frames[-1], bad_x + 25.0, bad_y)

    job = track(client, project_id, point_id)
    assert sorted(job["result"]["anchor_frames"]) == sorted(frames)

    after = track_of(client, project_id, point_id)
    # Every keyframe is still passed through exactly — that is a hard contract.
    for frame in frames[:-1]:
        assert (after["x"][frame], after["y"][frame]) == pytest.approx(
            source.true_position(frame), abs=1e-3
        )
    # ... and the one no run could reconcile is flagged rather than pretended about.
    assert after["flagged_anchors"] == [frames[-1]]
    assert all(f not in after["flagged_anchors"] for f in frames[:-1])
    breach = next(v for v in after["anchor_violations"] if v["frame"] == frames[-1])
    assert breach["distance"] > 20.0
    assert breach["direction"] == "fwd"


def test_a_clean_multi_keyframe_track_flags_nothing(client, project) -> None:
    project_id, source = project
    frames = (0, 12, 24)
    x0, y0 = source.true_position(frames[0])
    point_id = place_point(client, project_id, frames[0], x0, y0)
    for frame in frames[1:]:
        add_anchor(client, project_id, point_id, frame, *source.true_position(frame))

    job = track(client, project_id, point_id)
    assert job["result"]["flagged_anchors"] == []
    assert track_of(client, project_id, point_id)["flagged_anchors"] == []
