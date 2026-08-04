"""The server really runs Phase 3's solver / smoothing / renderer, not a stub.

Each test here would have passed against the Phase 4 stand-ins for the wrong
reason, so they assert the things only the real modules do: RANSAC's per-frame
quality record, pivot-relative curves, the genuine filter catalogue, and the
three border policies.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.geometry import CHANNELS
from stable.core.project import SmoothingSpec
from stable.smoothing.channels import SmoothingSettings
from stable.smoothing.motion import smooth_motion
from stable.solver.motion import SolveConfig, solve_motion

from ._helpers import decode_jpeg, seed_tracks, upload_clip, wait_job


@pytest.fixture
def solved_clip(client: TestClient, workspace, tmp_path, warped_clip):
    """A project over a clip warped by a known motion, with ground-truth tracks.

    The tracks are seeded rather than tracked, so what is under test is the
    solve/smooth/render chain and nothing else.
    """
    from conftest import encode_synthetic_clip  # noqa: PLC0415 - pytest puts tests/ on sys.path

    path = encode_synthetic_clip(warped_clip, str(tmp_path / "warped.mp4"), with_audio=False)
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)

    features = warped_clip.feature_points(6)
    seed_tracks(workspace, project_id, warped_clip.true_tracks(features).tracks())
    return project_id, warped_clip, features


def run_solve(client: TestClient, project_id: str) -> dict:
    """Submit a solve, wait for it, and return the fetched curves."""
    submitted = client.post(f"/api/projects/{project_id}/solve")
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    assert job["status"] == "done", job
    curves = client.get(f"/api/projects/{project_id}/solve").json()["curves"]
    assert curves is not None
    return dict(curves)


# -- the solver ----------------------------------------------------------------


def test_solve_returns_the_real_per_frame_quality_record(client, solved_clip):
    project_id, _clip, _features = solved_clip
    curves = run_solve(client, project_id)

    # Phase 3's SolveStatus / RANSAC inliers / flagged strip — none of which the
    # RANSAC-free stub could produce.
    assert set(curves) >= {"status", "n_inliers", "flagged", "pivot", "summary", "residual"}
    assert set(curves["status"]) <= {
        "solved",
        "translation_only",
        "interpolated",
        "held",
        "identity",
    }
    assert all(status == "solved" for status in curves["status"])
    assert all(n == 6 for n in curves["n_inliers"])
    assert curves["summary"]["status_counts"]["solved"] == len(curves["x"])


def test_solved_curves_are_pivot_relative_and_recover_the_known_motion(client, solved_clip):
    project_id, clip, _features = solved_clip
    curves = run_solve(client, project_id)

    detail = client.get(f"/api/projects/{project_id}").json()
    width, height = detail["video"]["w"], detail["video"]["h"]
    assert curves["pivot"] == pytest.approx([width / 2.0, height / 2.0])

    pivot = np.array([width / 2.0, height / 2.0])
    for frame in (0, 7, 19, len(curves["x"]) - 1):
        truth = clip.relative_transform(frame, 0)
        moved = truth.apply(pivot) - pivot
        assert curves["x"][frame] == pytest.approx(float(moved[0]), abs=0.3)
        assert curves["y"][frame] == pytest.approx(float(moved[1]), abs=0.3)
        assert curves["r"][frame] == pytest.approx(truth.rotation, abs=2e-3)
        assert curves["s"][frame] == pytest.approx(truth.scale, abs=2e-3)

    # The whole point of the pivot convention: a centre rotation must not read as
    # hundreds of pixels of translation.
    assert max(abs(v) for v in curves["x"]) < 40.0


def test_the_curves_match_calling_the_solver_directly(client, workspace, solved_clip):
    project_id, clip, features = solved_clip
    curves = run_solve(client, project_id)

    project = workspace.load_project(project_id)
    tracks = [t for t in clip.true_tracks(features).tracks()]
    trimmed = [
        type(t)(
            points=t.points[: project.video.frames],
            confidence=t.confidence[: project.video.frames],
            visible=t.visible[: project.video.frames],
        )
        for t in tracks
    ]
    expected = solve_motion(
        trimmed,
        config=SolveConfig.from_settings(project.solve),
        frame_size=(project.video.w, project.video.h),
    )
    for channel in CHANNELS:
        assert curves[channel] == pytest.approx(expected.curve(channel).tolist(), abs=1e-9)


# -- the filter catalogue --------------------------------------------------------


def test_filter_catalogue_is_served_with_its_parameter_schema(client):
    catalogue = client.get("/api/filters").json()
    names = {entry["name"] for entry in catalogue}
    assert names == {"moving_average", "gaussian", "butterworth", "savgol", "spline", "tripod"}
    butter = next(e for e in catalogue if e["name"] == "butterworth")
    params = {p["name"]: p for p in butter["params"]}
    assert set(params) == {"cutoff_hz", "order", "strength"}
    assert params["order"]["kind"] == "int"
    assert params["strength"]["maximum"] == 1.0


def test_butterworth_is_a_real_zero_phase_butterworth(client, workspace, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)

    response = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 0.8, "order": 2}}
            ]
        },
    )
    assert response.status_code == 200, response.text
    smoothed = response.json()["curves"]["smoothed"]

    # Compare against the library computing the same thing from the same solve.
    from stable.server.routes.solve import load_solve  # noqa: PLC0415 - test-local import

    solve = load_solve(workspace, project_id)
    project = workspace.load_project(project_id)
    expected = smooth_motion(
        solve,
        SmoothingSettings.from_specs(
            [
                SmoothingSpec(
                    channel="x", filter="butterworth", params={"cutoff_hz": 0.8, "order": 2}
                )
            ]
        ),
        fps=project.video.fps,
    )
    assert smoothed["x"] == pytest.approx(expected.curve("x").tolist(), abs=1e-9)
    # Untouched channels pass through raw, so the warp cannot move them at all.
    assert smoothed["y"] == pytest.approx(solve.curve("y").tolist(), abs=1e-12)


def test_stub_era_filter_parameters_are_now_rejected(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    # The stub took spline {"smoothing": 3}; the real spline has lam/cutoff_hz,
    # and silently ignoring a typo'd parameter is how a filter looks broken.
    response = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "spline", "params": {"smoothing": 3}}]},
    )
    assert response.status_code == 400
    assert "smoothing" in response.json()["detail"]


def test_a_rejected_filter_list_does_not_poison_the_project(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    good = [{"channel": "x", "filter": "gaussian", "params": {"sigma": 5.0}}]
    accepted = client.put(f"/api/projects/{project_id}/smoothing", json={"filters": good})
    assert accepted.status_code == 200
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "nope", "params": {}}]},
    )
    kept = client.get(f"/api/projects/{project_id}/smoothing").json()
    assert [f["filter"] for f in kept["filters"]] == ["gaussian"]
    assert kept["curves"]["smoothed"] is not None


def test_curve_editor_directives_round_trip_through_the_api(client, solved_clip):
    project_id, _clip, _features = solved_clip
    curves = run_solve(client, project_id)
    pinned = 12

    response = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": "x", "filter": "gaussian", "params": {"sigma": 10.0}},
                {"channel": "x", "filter": "@range", "params": {"start": 5, "end": 30, "blend": 3}},
                {"channel": "x", "filter": "@pin", "params": {"frame": pinned, "radius": 4}},
            ]
        },
    )
    assert response.status_code == 200, response.text
    smoothed = response.json()["curves"]["smoothed"]
    assert smoothed["x"][pinned] == pytest.approx(curves["x"][pinned], abs=1e-9)
    assert smoothed["x"][0] == pytest.approx(curves["x"][0], abs=1e-9)
    assert smoothed["settings"]["x"]["range"] == [5, 30]
    assert smoothed["settings"]["x"]["pins"] == [pinned]


# -- the renderer ------------------------------------------------------------------


def tripod_everything(client: TestClient, project_id: str) -> None:
    """Lock every channel, so the stabilisation warp becomes exactly ``raw^-1``."""
    response = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": c, "filter": "tripod", "params": {}} for c in CHANNELS]},
    )
    assert response.status_code == 200, response.text


def centre_crop(image: np.ndarray, fraction: float = 0.5) -> np.ndarray:
    """Return the centred sub-image covering ``fraction`` of each axis."""
    h, w = image.shape[:2]
    dy, dx = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    return image[dy : h - dy, dx : w - dx]


def test_tripod_stabilisation_lands_back_on_the_reference_frame(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    tripod_everything(client, project_id)
    border = client.put(f"/api/projects/{project_id}/render", json={"border": "transparent"})
    assert border.status_code == 200, border.text

    reference = decode_jpeg(
        client.get(f"/api/projects/{project_id}/frames/0").content
    ).astype(float)
    for frame in (9, 21, 33):
        moved = decode_jpeg(
            client.get(f"/api/projects/{project_id}/frames/{frame}").content
        ).astype(float)
        fixed = decode_jpeg(
            client.get(f"/api/projects/{project_id}/frames/{frame}/stabilized").content
        ).astype(float)
        assert fixed.shape == reference.shape

        drift_before = np.abs(centre_crop(moved) - centre_crop(reference)).mean()
        drift_after = np.abs(centre_crop(fixed) - centre_crop(reference)).mean()
        # Un-warping by raw^-1 must put the frame back on the reference. Rebuilding
        # the transform from raw channel values instead (the old stub's bug, which
        # reads pivot-relative curves as origin-based) leaves a rotation about the
        # top-left corner in — several pixels of misregistration on this texture.
        assert drift_after < drift_before / 2.0, (frame, drift_before, drift_after)
        assert drift_after < 6.0, (frame, drift_after)


def test_border_policies_change_the_preview_geometry(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    tripod_everything(client, project_id)
    proxy = decode_jpeg(client.get(f"/api/projects/{project_id}/frames/12").content)

    client.put(f"/api/projects/{project_id}/render", json={"border": "auto_zoom"})
    zoomed = decode_jpeg(
        client.get(f"/api/projects/{project_id}/frames/12/stabilized").content
    )
    assert zoomed.shape == proxy.shape

    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    cropped = decode_jpeg(
        client.get(f"/api/projects/{project_id}/frames/12/stabilized").content
    )
    assert cropped.shape[0] < proxy.shape[0] and cropped.shape[1] < proxy.shape[1]


def test_auto_zoom_preview_leaves_no_uncovered_edges(client, workspace, tmp_path, flat_clip):
    from conftest import encode_synthetic_clip  # noqa: PLC0415 - pytest puts tests/ on sys.path

    path = encode_synthetic_clip(flat_clip, str(tmp_path / "flat.mp4"), with_audio=False)
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    seed_tracks(workspace, project_id, flat_clip.true_tracks(flat_clip.feature_points(6)).tracks())

    run_solve(client, project_id)
    tripod_everything(client, project_id)
    client.put(f"/api/projects/{project_id}/render", json={"border": "auto_zoom"})

    # Every source pixel is white, so any dark pixel in the output is a gap the
    # auto-zoom failed to cover.
    for frame in (0, 14, 29):
        image = decode_jpeg(
            client.get(f"/api/projects/{project_id}/frames/{frame}/stabilized").content
        )
        assert image.min() > 120, (frame, int(image.min()))


def test_transparent_border_cannot_be_exported_as_video(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    client.put(f"/api/projects/{project_id}/render", json={"border": "transparent"})
    response = client.post(
        f"/api/projects/{project_id}/export/video", json={"filename": "out.mp4"}
    )
    assert response.status_code == 400
    assert "alpha" in response.json()["detail"]


def test_export_video_reports_the_render_geometry(client, solved_clip):
    project_id, _clip, _features = solved_clip
    run_solve(client, project_id)
    tripod_everything(client, project_id)
    client.put(f"/api/projects/{project_id}/render", json={"border": "auto_zoom"})

    submitted = client.post(
        f"/api/projects/{project_id}/export/video", json={"filename": "out.mp4", "crf": 24}
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=120)
    assert job["status"] == "done", job
    assert job["result"]["border"] == "auto_zoom"
    assert job["result"]["zoom"] > 1.0
    assert client.get(f"/api/projects/{project_id}/export/download/out.mp4").status_code == 200


def test_invalid_border_policy_is_rejected(client, solved_clip):
    project_id, _clip, _features = solved_clip
    response = client.put(f"/api/projects/{project_id}/render", json={"border": "feather"})
    assert response.status_code == 400
