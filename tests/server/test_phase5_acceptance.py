"""Phase 5 acceptance: break a track, repair it in a handful of clicks, re-solve.

``PLAN.md``'s criterion is *"deliberately break a track (occlusion clip), repair
it in <= 5 interactions, re-solve, no visible pop in stabilized output; undo
restores exact prior state."*

The break is **imposed** rather than hoped for. Handing a real occlusion clip to
a real tracker and assuming it fails in a useful way makes the assertion depend
on how badly ``lk`` happens to do today — the day it improves, the test stops
testing the repair path and nobody notices. Here the failure is exactly the one
occlusion causes (the point latches onto the background and drifts away,
confidently), so the repair path is exercised every run.

Measuring "no visible pop"
--------------------------
A pop is something the *viewer* sees, so it has to be measured in the output,
not in the warp. The clip's true per-frame motion is known, so where a
reference-frame feature lands in the stabilised output is known exactly::

    apparent(t) = stabilization_transform(t) . truth(t)

With a correct solve that collapses to the smoothed camera path — smooth by
construction. Where the solve is wrong, the scene jumps. Taking the largest
frame-to-frame *acceleration* of that path isolates the jump from the legitimate
camera move around it, in pixels.

Every repair step below goes through the same HTTP endpoints the UI calls, and
they are counted.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.types import PointTrack
from stable.server.routes.solve import load_smoothed

from ._helpers import seed_tracks, upload_clip, wait_job

BREAK_START, BREAK_END = 18, 25
"""The frames the point is lost over — the "occlusion"."""

SMOOTHING = [
    {"channel": channel, "filter": "butterworth", "params": {"cutoff_hz": 1.2, "order": 2}}
    for channel in ("x", "y", "r", "s")
]
"""A real stabilisation setting, so the warp being measured is one a user would ship."""

VISIBLE_PX = 1.0
"""Apparent acceleration below this is sub-pixel, i.e. nothing a viewer can see."""


def occlude(track: PointTrack) -> PointTrack:
    """Return ``track`` broken the way an occlusion breaks one.

    The tracker does not report failure: it locks onto whatever texture is now
    under the window and slides away from the true path, still confident. That
    "confidently wrong" shape is what actually poisons a solve — a track that
    simply went ``NaN`` would be down-weighted to nothing and cause no pop.
    """
    points = np.array(track.points, copy=True)
    confidence = np.array(track.confidence, copy=True)
    held = points[BREAK_START - 1].copy()
    for offset, frame in enumerate(range(BREAK_START, BREAK_END + 1)):
        # Accelerating away, as a window sliding down a moving occluder does;
        # a constant-velocity error would just look like a slightly different
        # camera move and is not what makes a shot unusable.
        points[frame] = held + (0.9 * offset**2, -0.6 * offset**2)
        confidence[frame] = 0.8
    return PointTrack(points=points, confidence=confidence, visible=track.visible)


def apparent_pop(motion, clip) -> float:
    """Return the worst apparent acceleration, in pixels, of the stabilised scene.

    See the module docstring: this is what the viewer sees move, not what the
    warp does — the two differ precisely because the warp is *supposed* to carry
    the inverse of the clip's judder.
    """
    centre = np.asarray(motion.pivot, dtype=np.float64)
    path = np.stack(
        [
            motion.stabilization_transform(frame)
            .compose(clip.relative_transform(frame, 0))
            .apply(centre)
            for frame in range(motion.n_frames)
        ]
    )
    return float(np.linalg.norm(np.diff(path, n=2, axis=0), axis=1).max())


def solve_and_smooth(client: TestClient, workspace, project_id: str):
    """Run a solve, apply the smoothing, and return the resulting smoothed motion."""
    submitted = client.post(f"/api/projects/{project_id}/solve")
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    assert job["status"] == "done", job
    applied = client.put(f"/api/projects/{project_id}/smoothing", json={"filters": SMOOTHING})
    assert applied.status_code == 200, applied.text
    motion = load_smoothed(workspace, project_id)
    assert motion is not None
    return motion


def make_project(client: TestClient, workspace, clip, path: str, tracks) -> str:
    """Upload ``path`` as a project and seed it with ready-made tracks."""
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    seed_tracks(workspace, project_id, tracks)
    return str(project_id)


@pytest.fixture
def clip_file(tmp_path, warped_clip) -> str:
    """The warped clip encoded once as a real H.264 file."""
    from conftest import encode_synthetic_clip  # noqa: PLC0415 - pytest puts tests/ on sys.path

    return str(encode_synthetic_clip(warped_clip, str(tmp_path / "occluded.mp4"), with_audio=False))


@pytest.fixture
def broken_project(client: TestClient, workspace, clip_file, warped_clip):
    """A project whose first of three points loses its target mid-shot."""
    clean = warped_clip.true_tracks(warped_clip.feature_points(3)).tracks()
    project_id = make_project(
        client, workspace, warped_clip, clip_file, [occlude(clean[0]), clean[1], clean[2]]
    )
    point_id = client.get(f"/api/projects/{project_id}/points").json()[0]["id"]
    return project_id, point_id, clean


def test_a_broken_track_is_repaired_in_five_interactions_with_no_pop(
    client, workspace, clip_file, warped_clip, broken_project
):
    project_id, broken_id, clean = broken_project

    # The baseline: the same shot solved from tracks that never broke.
    reference_id = make_project(client, workspace, warped_clip, clip_file, clean)
    clean_pop = apparent_pop(solve_and_smooth(client, workspace, reference_id), warped_clip)
    assert clean_pop < VISIBLE_PX, clean_pop

    # As the operator actually has it: one point is confidently wrong for 8 frames.
    broken_pop = apparent_pop(solve_and_smooth(client, workspace, project_id), warped_clip)
    assert broken_pop > 10.0, broken_pop

    # The solve flags the damage, so the user knows where to look rather than
    # having to spot it by scrubbing.
    curves = client.get(f"/api/projects/{project_id}/solve").json()["curves"]
    assert any(curves["flagged"][BREAK_START : BREAK_END + 1])

    # The repair: one interaction — throw the bad span away and let the solver's
    # graceful degradation carry those frames on the remaining points.
    interactions = 0
    base = f"/api/projects/{project_id}/points/{broken_id}/corrections"
    assert (
        client.post(
            base, json={"kind": "delete_range", "start": BREAK_START, "end": BREAK_END}
        ).status_code
        == 200
    )
    interactions += 1
    assert interactions <= 5

    repaired_pop = apparent_pop(solve_and_smooth(client, workspace, project_id), warped_clip)
    assert repaired_pop < VISIBLE_PX, repaired_pop
    # Not merely "better": indistinguishable from the shot that never broke.
    assert repaired_pop == pytest.approx(clean_pop, abs=0.05), (repaired_pop, clean_pop)


def test_gap_fill_brings_the_point_itself_back(client, workspace, warped_clip, broken_project):
    """When the point is *needed* — not just excluded — filling recovers its path.

    Deleting the span is enough to fix the solve while other points cover those
    frames. Filling is the repair for when the point has to exist: exporting its
    track, or a shot where it is the only usable feature.
    """
    project_id, broken_id, clean = broken_project
    track_url = f"/api/projects/{project_id}/points/{broken_id}/track"
    truth = np.asarray(clean[0].points, dtype=np.float64)
    span = slice(BREAK_START, BREAK_END + 1)

    def drift(payload: dict) -> float:
        got = np.array(
            [[payload["x"][f], payload["y"][f]] for f in range(BREAK_START, BREAK_END + 1)],
            dtype=np.float64,
        )
        return float(np.linalg.norm(got - truth[span], axis=1).max())

    broken_pop = apparent_pop(solve_and_smooth(client, workspace, project_id), warped_clip)
    before = drift(client.get(track_url).json())
    assert before > 20.0, before

    base = f"/api/projects/{project_id}/points/{broken_id}/corrections"
    calls = [
        {"kind": "delete_range", "start": BREAK_START, "end": BREAK_END},
        {
            "kind": "gap_fill",
            "start": BREAK_START,
            "end": BREAK_END,
            "method": "spline",
            "confidence": 0.6,
        },
    ]
    for body in calls:
        assert client.post(base, json=body).status_code == 200
    assert len(calls) <= 5

    filled = client.get(track_url).json()
    assert filled["gaps"] == []
    assert drift(filled) < before / 10.0
    # The interpolation cannot invent the handheld judder it spans, so a little
    # residual is expected — but the pop is gone.
    repaired_pop = apparent_pop(solve_and_smooth(client, workspace, project_id), warped_clip)
    assert repaired_pop < broken_pop / 5.0, (repaired_pop, broken_pop)


def test_retrack_from_here_pulls_the_point_back_onto_its_target(client, broken_project):
    """The third repair route: re-track forward from a corrected position."""
    project_id, broken_id, clean = broken_project
    track_url = f"/api/projects/{project_id}/points/{broken_id}/track"
    truth = np.asarray(clean[0].points, dtype=np.float64)

    def drift(payload: dict, frames: range) -> float:
        got = np.array([[payload["x"][f], payload["y"][f]] for f in frames], dtype=np.float64)
        return float(np.linalg.norm(got - truth[frames.start : frames.stop], axis=1).max())

    span = range(BREAK_START, BREAK_END + 1)
    before = drift(client.get(track_url).json(), span)
    assert before > 5.0, before

    submitted = client.post(
        f"/api/projects/{project_id}/points/{broken_id}/retrack",
        json={
            "frame": BREAK_START,
            "x": float(truth[BREAK_START][0]),
            "y": float(truth[BREAK_START][1]),
            "direction": "fwd",
            "tracker": "lk",
        },
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job

    assert drift(client.get(track_url).json(), span) < before / 3.0


def test_undo_restores_the_exact_prior_state_after_a_repair(client, workspace, broken_project):
    project_id, broken_id, _clean = broken_project
    track_url = f"/api/projects/{project_id}/points/{broken_id}/track"
    before = client.get(track_url).json()
    project_before = workspace.load_project(project_id).to_json()

    base = f"/api/projects/{project_id}/points/{broken_id}/corrections"
    client.post(base, json={"kind": "delete_range", "start": BREAK_START, "end": BREAK_END})
    client.post(base, json={"kind": "gap_fill", "start": BREAK_START, "end": BREAK_END})
    client.post(base, json={"kind": "offset", "frame": 30, "dx": 2.0, "dy": 1.0})
    assert client.get(track_url).json()["x"] != before["x"]

    for _ in range(3):
        assert client.post(f"/api/projects/{project_id}/edits/undo").status_code == 200

    after = client.get(track_url).json()
    # Exact, not approximate: the correction layer is dropped wholesale, so the
    # displayed track is recomputed from the untouched raw .npz.
    assert after["x"] == before["x"]
    assert after["y"] == before["y"]
    assert after["confidence"] == before["confidence"]
    assert after["visible"] == before["visible"]
    assert after["corrections"] == []
    assert workspace.load_project(project_id).to_json() == project_before


def test_the_repair_can_be_replayed_with_redo(client, broken_project):
    project_id, broken_id, _clean = broken_project
    base = f"/api/projects/{project_id}/points/{broken_id}/corrections"
    client.post(base, json={"kind": "delete_range", "start": BREAK_START, "end": BREAK_END})
    client.post(base, json={"kind": "gap_fill", "start": BREAK_START, "end": BREAK_END})
    repaired = client.get(f"/api/projects/{project_id}/points/{broken_id}/track").json()

    client.post(f"/api/projects/{project_id}/edits/undo")
    client.post(f"/api/projects/{project_id}/edits/undo")
    client.post(f"/api/projects/{project_id}/edits/redo")
    client.post(f"/api/projects/{project_id}/edits/redo")

    assert client.get(f"/api/projects/{project_id}/points/{broken_id}/track").json() == repaired
