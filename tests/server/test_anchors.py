"""Anchors are one concept: place, track from, delete (todo 6, seventh round).

The report these are written against is a UX one — "we have keyframes and anchor
points, the user doesn't understand what they are" — but behind the two words
sat three things the tool genuinely could not do, and this file is about those:

#. **Track forwards or backwards from an anchor**, on a point that has never
   been tracked. The bounded run lands as a correction, and a correction needs a
   baseline to splice onto, so without one the run was computed and then hidden.
#. **Delete any anchor.** Half of them were asserted by a manual correction
   rather than by the point, and deleting the point's half left the correction
   putting the anchor straight back.
#. **See the tracking follow.** A run rooted on an anchor goes away with it; the
   merged baseline cannot, so the point says its anchors have changed instead.

The vocabulary itself is checked where it is served: ``kind == "anchor"`` in
``tests/server/test_point_patches.py``, ``anchors`` (not ``ground_truth``) in
``tests/server/test_anchor_integrity.py``.
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


def add_anchor(client: TestClient, project_id: str, point_id: str, frame, x, y) -> dict:
    """Add an anchor and return the updated point summary."""
    response = client.post(
        f"/api/projects/{project_id}/points/{point_id}/anchors",
        json={"frame": frame, "x": x, "y": y},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def run_from(client: TestClient, project_id: str, point_id: str, **body) -> dict:
    """Track one direction from one frame and return the finished job."""
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/retrack",
        json={"tracker": TRACKER, **body},
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    return dict(job)


def track(client: TestClient, project_id: str, point_id: str) -> dict:
    """Run the full multi-anchor tracking job and return the finished job."""
    submitted = client.post(
        f"/api/projects/{project_id}/points/{point_id}/track", json={"tracker": TRACKER}
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    return dict(job)


def track_of(client: TestClient, project_id: str, point_id: str) -> dict:
    """Fetch a point's per-frame track payload."""
    response = client.get(f"/api/projects/{project_id}/points/{point_id}/track")
    assert response.status_code == 200, response.text
    return dict(response.json())


def summary(client: TestClient, project_id: str, point_id: str) -> dict:
    """Fetch one point's list summary."""
    response = client.get(f"/api/projects/{project_id}/points")
    assert response.status_code == 200, response.text
    return next(p for p in response.json() if p["id"] == point_id)


# -- tracking forwards / backwards from an anchor ----------------------------------


def test_a_fresh_anchor_can_be_tracked_forwards_without_a_full_track_first(client, project):
    """The core gesture: place an anchor, track on from it, see the result.

    It used to be impossible. The run needs somewhere to land, and a point with
    no ``raw`` had nowhere, so the panel hid the buttons behind "has this been
    fully tracked yet" and the only way to use a new anchor was to track the
    whole clip from every anchor at once.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))

    job = run_from(client, project_id, point_id, frame=0, direction="fwd", max_frames=8)
    assert job["result"]["covered_range"] == [0, 8]

    after = track_of(client, project_id, point_id)
    for frame in range(9):
        assert (after["x"][frame], after["y"][frame]) == pytest.approx(
            source.true_position(frame), abs=2.0
        )
    # ...and only those frames: a bounded run is a bounded statement.
    assert after["x"][20] is None


def test_a_run_from_an_anchor_stops_at_the_next_one(client, project):
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 12, *source.true_position(12))

    job = run_from(client, project_id, point_id, frame=0, direction="fwd", stop_at_anchor=True)
    assert job["result"]["frame_range"] == [0, 13]


def test_the_old_name_for_the_anchor_bound_still_works(client, project):
    """``stop_at_keyframe`` was this field's name until the two words became one.

    A browser tab left open across the rename must not silently lose its bound
    and track the whole clip.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 12, *source.true_position(12))

    job = run_from(client, project_id, point_id, frame=0, direction="fwd", stop_at_keyframe=True)
    assert job["result"]["frame_range"] == [0, 13]


def test_a_point_reports_the_frames_a_run_actually_covered(client, project):
    """The card's "N frames" counts what is on screen, not what the merge wrote.

    A point whose whole track came from a directional run has an empty baseline
    underneath it, so counting the raw ``.npz`` reported "0 frames · 0%" over a
    track plainly drawn on the frame.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    run_from(client, project_id, point_id, frame=0, direction="fwd", max_frames=6)

    row = summary(client, project_id, point_id)
    assert row["has_track"] is True
    assert row["tracked_frames"] == 7
    assert row["mean_confidence"] > 0.0


# -- deleting an anchor, and what goes with it -------------------------------------


def test_deleting_an_anchor_removes_what_was_tracked_from_it(client, project):
    """Delete an anchor and the tracking updates accordingly — exactly that.

    A run rooted on an anchor is stored as a correction rooted on the same
    frame, so removing the anchor removes the span it produced — no re-tracking,
    nothing left behind.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 14, *source.true_position(14))
    run_from(client, project_id, point_id, frame=14, direction="fwd", max_frames=5)
    assert track_of(client, project_id, point_id)["x"][16] is not None

    response = client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/14")
    assert response.status_code == 200, response.text

    after = track_of(client, project_id, point_id)
    assert [q["frame"] for q in after["anchors"]] == [0]
    assert after["x"][16] is None, "the run from the deleted anchor outlived it"
    assert after["corrections"] == []


def test_an_anchor_a_drag_created_can_be_deleted_like_any_other(client, project):
    """It could not before: its × was disabled and pointed at another panel."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    track(client, project_id, point_id)

    dragged = client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "offset", "frame": 10, "dx": 6.0, "dy": -4.0, "falloff_before": 2},
    )
    assert dragged.status_code == 200, dragged.text
    before = track_of(client, project_id, point_id)
    assert [q["frame"] for q in before["anchors"]] == [0, 10]

    response = client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/10")
    assert response.status_code == 200, response.text

    after = track_of(client, project_id, point_id)
    assert [q["frame"] for q in after["anchors"]] == [0]
    # The drag went with it — anchor and nudge were one gesture, and leaving the
    # nudge would move the point by a fraction of a pixel and look like nothing
    # happened at all.
    assert after["corrections"] == []
    assert (after["x"][10], after["y"][10]) == pytest.approx(
        (before["raw_x"][10], before["raw_y"][10]), abs=1e-3
    )


def test_deleting_an_anchor_is_undoable(client, project):
    """Even on a point with no track at all — the anchors are the state that matters.

    Read off the point summary rather than the track: an untracked point has no
    track to read, which is exactly the case worth covering here.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 9, *source.true_position(9))

    client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/9")
    assert [q["frame"] for q in summary(client, project_id, point_id)["queries"]] == [0]

    undone = client.post(f"/api/projects/{project_id}/edits/undo")
    assert undone.status_code == 200, undone.text
    assert [q["frame"] for q in summary(client, project_id, point_id)["queries"]] == [0, 9]


def test_undo_brings_back_an_anchor_with_the_run_it_carried(client, project):
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 14, *source.true_position(14))
    run_from(client, project_id, point_id, frame=14, direction="fwd", max_frames=5)
    covered = track_of(client, project_id, point_id)["x"][16]

    client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/14")
    assert client.post(f"/api/projects/{project_id}/edits/undo").status_code == 200

    restored = track_of(client, project_id, point_id)
    assert [q["frame"] for q in restored["anchors"]] == [0, 14]
    assert restored["x"][16] == pytest.approx(covered, abs=1e-6)


def test_a_point_keeps_its_last_anchor(client, project):
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))

    response = client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/0")
    assert response.status_code == 400
    assert "last remaining anchor" in response.json()["detail"]


def test_deleting_an_anchor_that_is_not_there_says_so(client, project):
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))

    response = client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/7")
    assert response.status_code == 400
    assert "no anchor on frame 7" in response.json()["detail"]


# -- the track and its anchors falling out of step ---------------------------------


def test_a_freshly_tracked_point_is_not_stale(client, project):
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    track(client, project_id, point_id)
    assert summary(client, project_id, point_id)["anchors_changed"] is False


def test_adding_an_anchor_marks_the_track_as_out_of_date(client, project):
    """Placing an anchor and watching nothing happen is how "what is this for?" starts."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    track(client, project_id, point_id)

    add_anchor(client, project_id, point_id, 18, *source.true_position(18))
    assert summary(client, project_id, point_id)["anchors_changed"] is True

    track(client, project_id, point_id)
    assert summary(client, project_id, point_id)["anchors_changed"] is False


def test_moving_an_anchor_marks_the_track_as_out_of_date(client, project):
    """The frame list is unchanged, so only the recorded *positions* can see this."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 18, *source.true_position(18))
    track(client, project_id, point_id)

    x, y = source.true_position(18)
    add_anchor(client, project_id, point_id, 18, x + 3.0, y)
    assert summary(client, project_id, point_id)["anchors_changed"] is True


def test_deleting_an_anchor_marks_the_merged_track_as_out_of_date(client, project):
    """The merge cannot be un-merged, so the point says so rather than pretending."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    add_anchor(client, project_id, point_id, 18, *source.true_position(18))
    track(client, project_id, point_id)

    client.delete(f"/api/projects/{project_id}/points/{point_id}/anchors/18")
    assert summary(client, project_id, point_id)["anchors_changed"] is True


def test_a_correction_does_not_make_the_track_out_of_date(client, project):
    """A correction applies itself; only the operator's own anchors need a re-run."""
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    track(client, project_id, point_id)

    client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "offset", "frame": 11, "dx": 2.0, "dy": 1.0},
    )
    assert summary(client, project_id, point_id)["anchors_changed"] is False


def test_a_track_run_with_corrections_in_place_is_not_born_stale(client, project):
    """The run records the point's own anchors, not the anchors it planned through.

    Those differ the moment a correction has been promoted, and recording the
    planned set would make every corrected point permanently claim to need a
    re-track — including the one it had just finished.
    """
    project_id, source = project
    point_id = place_point(client, project_id, 0, *source.true_position(0))
    track(client, project_id, point_id)
    client.post(
        f"/api/projects/{project_id}/points/{point_id}/corrections",
        json={"kind": "replace", "frame": 11, "x": 24.0, "y": 20.0},
    )

    track(client, project_id, point_id)
    assert summary(client, project_id, point_id)["anchors_changed"] is False
