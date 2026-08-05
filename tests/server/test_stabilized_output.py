"""Todo 17: the stabilised output video, and the options that make it usable.

The bug
-------
A real project taken off disk showed the failure exactly: one tracked point,
solved cleanly, smoothing filters present but *disabled*, border ``transparent``
— and a 7.8 MB ``stabilized.mp4`` in ``renders/`` that was a pixel-for-pixel
re-encode of the source. The clip's own solve says the camera moved 151 px in x
and 89 px in y; the export removed none of it, reported ``zoom: 1.0`` and
``status: done``, and gave the operator no reason to think anything was wrong.

The cause is structural rather than arithmetical: the warp is
``smoothed(t) . raw(t)^-1``, so with no filter enabled ``smoothed == raw`` and
every frame's warp is the identity. ``stable stabilize`` never hits it because
the CLI applies ``DEFAULT_SMOOTHING`` when the user does not choose one — which
is why the same clip stabilises fine from the command line and not from the
browser.

``test_export_refuses_a_render_that_would_change_nothing`` is that project,
reconstructed through the HTTP API. The rest cover the three options todo 17
asks for: a stabilisation target, a crop size around it, and a border fill.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ._helpers import decode_jpeg, seed_tracks, upload_clip, wait_job


@pytest.fixture
def solved_project(client: TestClient, workspace, tmp_path, warped_clip):
    """A project over a clip with a known handheld motion, solved, not smoothed.

    This is the state the reported project was in: real motion in the clip, a
    clean solve, and nothing configured that would make the render move a pixel.
    """
    from conftest import encode_synthetic_clip  # noqa: PLC0415 - pytest puts tests/ on sys.path

    path = encode_synthetic_clip(warped_clip, str(tmp_path / "warped.mp4"), with_audio=False)
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)

    features = warped_clip.feature_points(6)
    point_ids = seed_tracks(workspace, project_id, warped_clip.true_tracks(features).tracks())
    submitted = client.post(f"/api/projects/{project_id}/solve")
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    assert job["status"] == "done", job
    return project_id, warped_clip, point_ids


def export(client: TestClient, project_id: str, filename: str = "out.mp4") -> dict:
    """Submit an export, wait for it, and return the finished job."""
    submitted = client.post(
        f"/api/projects/{project_id}/export/video", json={"filename": filename, "crf": 24}
    )
    assert submitted.status_code == 200, submitted.text
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=180)
    assert job["status"] == "done", job
    return dict(job["result"])


def decoded_frames(path) -> list[np.ndarray]:
    """Decode a rendered file to a list of RGB frames."""
    import av  # noqa: PLC0415 - the video extra, imported where it is used

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


# -- the bug -------------------------------------------------------------------


def test_export_refuses_a_render_that_would_change_nothing(client, solved_project):
    """The reported bug: a solved project with no smoothing exported a copy."""
    project_id, _clip, _points = solved_project

    response = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "out.mp4"})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    # The message has to name both ways out, or it is just a different dead end.
    assert "re-encode the source" in detail
    assert "smoothing" in detail and "stabilisation target" in detail


def test_disabled_smoothing_filters_still_count_as_no_smoothing(client, solved_project):
    """The exact shape of the reported project: filters present, all disabled."""
    project_id, _clip, _points = solved_project
    smoothing = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {
                    "channel": c,
                    "filter": "butterworth",
                    "params": {"cutoff_hz": 1.0, "order": 2, "strength": 1.0},
                    "enabled": False,
                }
                for c in ("x", "y")
            ]
        },
    )
    assert smoothing.status_code == 200, smoothing.text
    response = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "out.mp4"})
    assert response.status_code == 400
    assert "re-encode the source" in response.json()["detail"]


def test_a_stabilisation_target_alone_produces_a_real_render(client, workspace, solved_project):
    """No smoothing at all, one target: the export goes ahead and really moves."""
    project_id, clip, point_ids = solved_project

    updated = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "point", "point_id": point_ids[0]}},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["options"]["target"]["point_id"] == point_ids[0]

    result = export(client, project_id)
    # The clip's real motion is tens of pixels; the old identity render moved 0.
    assert result["max_displacement"] > 5.0
    assert result["target"] == f"point {point_ids[0]}"

    frames = decoded_frames(workspace.renders_dir(project_id) / "out.mp4")
    assert len(frames) == clip.frame_count


def test_the_target_point_stops_moving_in_the_output(client, workspace, solved_project):
    """The point named as the target is pinned to where it sat on the anchor frame."""
    project_id, clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "point", "point_id": point_ids[0]}, "crop": {"scale": 0.8}},
    )
    export(client, project_id)

    # Where the target lands in the output, measured from the plan the server
    # actually rendered with — the same object, not a reimplementation of it.
    from stable.server import motion as motion_glue  # noqa: PLC0415 - test-local import
    from stable.server.routes.solve import load_options, load_smoothed  # noqa: PLC0415

    project = workspace.load_project(project_id)
    motion = load_smoothed(workspace, project_id, project)
    plan = motion_glue.build_plan(project, motion, options=load_options(workspace, project_id))

    feature = clip.feature_points(6)[0]
    landed = [
        plan.output_transform(f).apply(np.asarray(clip.transform_at(f).apply(feature)))
        for f in range(plan.n_frames)
    ]
    spread = np.ptp(np.asarray(landed), axis=0)
    assert spread.max() < 0.5, spread

    raw = np.asarray([clip.transform_at(f).apply(feature) for f in range(plan.n_frames)])
    # The clip has to actually move, or "it stopped moving" proves nothing.
    assert np.ptp(raw, axis=0).max() > 10.0


def test_centroid_target_uses_every_solve_enabled_point(client, solved_project):
    project_id, _clip, _points = solved_project
    updated = client.put(
        f"/api/projects/{project_id}/render/options", json={"target": {"mode": "centroid"}}
    )
    assert updated.status_code == 200, updated.text
    result = export(client, project_id)
    assert result["target"] == "all points"
    assert result["max_displacement"] > 5.0


def test_a_target_naming_an_unknown_point_is_rejected(client, solved_project):
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "point", "point_id": "nope"}},
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


# -- crop ----------------------------------------------------------------------


def test_crop_scale_sets_the_output_size(client, solved_project):
    project_id, clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "centroid"}, "crop": {"scale": 0.5}},
    )
    result = export(client, project_id)
    assert result["width"] == pytest.approx(clip.width // 2, abs=2)
    assert result["height"] == pytest.approx(clip.height // 2, abs=2)
    assert result["crop_scale"] == pytest.approx(0.5)
    # Cropping harder than the shot needs cannot expose anything.
    assert result["gaps"] is False


def test_a_crop_looser_than_the_coverage_reports_gaps(client, solved_project):
    """Keeping more than the shot covers is allowed — that is what the fill is for."""
    project_id, _clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {"mode": "centroid"},
            "crop": {"scale": 1.0},
            "fill": {"mode": "blur"},
        },
    )
    result = export(client, project_id)
    assert result["coverage"] < 1.0
    assert result["gaps"] is True
    assert result["fill"] == "blur"


def test_an_out_of_range_crop_scale_is_rejected(client, solved_project):
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 1.5}}
    )
    assert response.status_code == 400
    assert "crop scale" in response.json()["detail"]


# -- border fill ----------------------------------------------------------------


def test_border_fill_replaces_the_black_edge_in_the_preview(client, solved_project):
    """The uncovered edge stops being black once a fill is chosen."""
    project_id, _clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "centroid"}, "crop": {"scale": 1.0}},
    )

    def black_pixels(frame_index: int) -> int:
        """How much of the frame is the letterbox rather than the picture."""
        image = decode_jpeg(
            client.get(f"/api/projects/{project_id}/frames/{frame_index}/stabilized").content
        )
        return int(np.count_nonzero(image.max(axis=2) < 8))

    # Solid black is the default, and this clip's edges really are exposed.
    letterboxed = max(black_pixels(f) for f in (5, 20, 35))
    assert letterboxed > 200, letterboxed

    for mode in ("mirror", "stretch", "blur"):
        client.put(f"/api/projects/{project_id}/render/options", json={"fill": {"mode": mode}})
        filled = max(black_pixels(f) for f in (5, 20, 35))
        assert filled < letterboxed / 10, (mode, letterboxed, filled)

    # A solid fill in a colour is still a fill: no black, just not the picture.
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"fill": {"mode": "solid", "color": [220, 30, 30]}},
    )
    assert max(black_pixels(f) for f in (5, 20, 35)) < letterboxed / 10


def test_every_fill_mode_renders(client, solved_project):
    project_id, _clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    for mode in ("solid", "stretch", "mirror", "blur"):
        updated = client.put(
            f"/api/projects/{project_id}/render/options",
            json={
                "target": {"mode": "centroid"},
                "crop": {"scale": 1.0},
                "fill": {"mode": mode, "color": [10, 200, 30], "feather": 3.0},
            },
        )
        assert updated.status_code == 200, updated.text
        preview = client.get(f"/api/projects/{project_id}/frames/12/stabilized")
        assert preview.status_code == 200, (mode, preview.text)
        assert decode_jpeg(preview.content).shape[2] == 3


def test_an_unknown_fill_mode_is_rejected(client, solved_project):
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options", json={"fill": {"mode": "feathered"}}
    )
    assert response.status_code == 400
    assert "border fill" in response.json()["detail"]


# -- options plumbing -------------------------------------------------------------


def test_options_round_trip_and_offer_the_valid_choices(client, solved_project):
    project_id, _clip, point_ids = solved_project
    fetched = client.get(f"/api/projects/{project_id}/render/options").json()
    assert fetched["options"]["target"]["mode"] == "frame"
    assert fetched["target_modes"] == ["frame", "centroid", "point", "map"]
    assert fetched["fill_modes"] == ["solid", "stretch", "mirror", "blur"]
    # Only tracked points can be a target, so the UI cannot offer a broken one.
    assert [p["id"] for p in fetched["points"]] == point_ids

    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"fill": {"mode": "blur", "feather": 6.0}},
    )
    # A partial update must not wipe the blocks it did not mention.
    client.put(f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 0.9}})
    again = client.get(f"/api/projects/{project_id}/render/options").json()["options"]
    assert again["fill"]["mode"] == "blur"
    assert again["fill"]["feather"] == 6.0
    assert again["crop"]["scale"] == pytest.approx(0.9)


def test_options_survive_a_reload_and_drive_the_preview(client, workspace, solved_project):
    project_id, _clip, point_ids = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "point", "point_id": point_ids[1]}, "crop": {"scale": 0.6}},
    )
    # Stored beside the solve/smoothing caches, not in the locked v1 schema.
    assert (workspace.project_dir(project_id) / "render_options.json").is_file()
    stored = workspace.load_project(project_id).to_dict()["render"]
    assert set(stored) == {"border", "codec", "crf"}

    preview = client.get(f"/api/projects/{project_id}/frames/8/stabilized")
    assert preview.status_code == 200
    image = decode_jpeg(preview.content)
    # 60% of a half-res proxy of the clip.
    assert image.shape[0] == pytest.approx(_clip_half(workspace, project_id)[1] * 0.6, abs=3)


def _clip_half(workspace, project_id: str) -> tuple[int, int]:
    """Return the half-res proxy's ``(width, height)``."""
    manifest = workspace.get_proxy_store(project_id, "half").manifest
    return int(manifest.width), int(manifest.height)
