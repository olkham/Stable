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

``test_a_project_with_no_targets_is_refused_rather_than_re_encoded`` is that
project, reconstructed through the HTTP API: with no target points the render
falls back to the smoothing warp, and with no filter enabled that warp is the
identity. The export says what to change instead of writing a copy. Placing a
single target closes it, because a target pins a path to a mark and that moves
the picture whatever the smoothing is doing.

The rest cover the options: which paths are held and where in the delivered
frame, a crop size around them, a delivery resolution, and a border fill.
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


def test_a_project_with_no_targets_is_refused_rather_than_re_encoded(client, solved_project):
    """The reported bug: no targets, no smoothing, and a copy of the source.

    The refusal has to name both ways out, or it is a different dead end.
    """
    project_id, _clip, _points = solved_project

    response = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "o.mp4"})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "re-encode the source" in detail
    assert "smoothing" in detail and "target" in detail


def test_one_target_makes_that_same_project_export_a_real_render(client, solved_project):
    """...and placing one is the fix, with no smoothing configured at all."""
    project_id, _clip, point_ids = solved_project
    added = client.post(
        f"/api/projects/{project_id}/render/targets", json={"point_ids": [point_ids[0]]}
    )
    assert added.status_code == 200, added.text

    result = export(client, project_id)
    assert result["target"] == point_ids[0]
    assert result["max_displacement"] > 5.0


def test_disabled_smoothing_filters_still_count_as_no_smoothing(client, solved_project):
    """The exact shape of the reported project: filters present, all disabled."""
    project_id, _clip, point_ids = solved_project
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
        json={"target": {"points": [{"id": "t1", "point_ids": [point_ids[0]]}]}},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["options"]["target"]["points"][0]["point_ids"] == [point_ids[0]]

    result = export(client, project_id)
    # The clip's real motion is tens of pixels; the old identity render moved 0.
    assert result["max_displacement"] > 5.0
    assert result["target"] == point_ids[0]

    frames = decoded_frames(workspace.renders_dir(project_id) / "out.mp4")
    assert len(frames) == clip.frame_count


def test_the_target_point_stops_moving_in_the_output(client, workspace, solved_project):
    """The point named as the target is pinned to where it sat on the anchor frame."""
    project_id, clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {"points": [{"id": "t1", "point_ids": [point_ids[0]]}]},
            "crop": {"scale": 0.8},
        },
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
        f"/api/projects/{project_id}/render/options", json={"target": {"points": [{"id": "t1"}]}}
    )
    assert updated.status_code == 200, updated.text
    result = export(client, project_id)
    assert result["target"] == "all tracks"
    assert result["max_displacement"] > 5.0


def test_a_target_naming_an_unknown_point_is_rejected(client, solved_project):
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1", "point_ids": ["nope"]}]}},
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


# -- crop ----------------------------------------------------------------------


def test_crop_scale_sets_the_output_size(client, solved_project):
    project_id, clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1"}]}, "crop": {"scale": 0.5}},
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
            "target": {"points": [{"id": "t1"}]},
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


# -- the target's position in the delivered frame (todo 1, eighth round) -------------------------


@pytest.mark.parametrize("border", ["auto_zoom", "crop"])
@pytest.mark.parametrize("place", [{"u": 1 / 3, "v": 1 / 3}, {"u": 0.75, "v": 0.5}])
def test_the_target_lands_where_it_was_placed_in_the_output(
    client, workspace, solved_project, border, place
):
    """The whole point of todo 1, eighth round: a mark is a position in the delivered frame.

    Stated as a fraction of the output, and honoured exactly — not "near the
    left third" but *on* it, on every frame, at whatever crop the render ends up
    choosing. Measured through the plan the server would render with, so this is
    the file's geometry rather than a second implementation of it.
    """
    project_id, clip, point_ids = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": border})
    stored = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1", "point_ids": [point_ids[0]], "place": place}]}},
    )
    assert stored.status_code == 200, stored.text
    assert stored.json()["options"]["target"]["points"][0]["place"] == pytest.approx(place)

    from stable.server import motion as motion_glue  # noqa: PLC0415 - test-local import
    from stable.server.routes.solve import load_options, load_smoothed  # noqa: PLC0415

    project = workspace.load_project(project_id)
    motion = load_smoothed(workspace, project_id, project)
    plan = motion_glue.build_plan(project, motion, options=load_options(workspace, project_id))

    wanted = (place["u"] * plan.geometry.width, place["v"] * plan.geometry.height)
    feature = clip.feature_points(6)[0]
    for frame in (0, plan.n_frames // 2, plan.n_frames - 1):
        landed = plan.output_transform(frame).apply(
            np.asarray(clip.transform_at(frame).apply(feature))
        )
        assert landed == pytest.approx(wanted, abs=0.5)


def test_on_smoothed_track_reports_the_point_on_its_own_smoothed_trail(
    client, workspace, solved_project
):
    """The crosshair has to ride the dashed trail, so the numbers behind it must.

    'On track' holds the raw path on the mark, so the crosshair sits on the
    solid trail on every frame. 'On smoothed track' lets the point drift by
    whatever the smoothing left in, so a crosshair nailed to the stored mark
    would sit where nothing is — the viewer draws it at ``landed`` instead, and
    that number has to be the *smoothed* trail (``GET /smoothing/tracks``) put
    through the export's own fit, or the crosshair rides a curve nobody drew.
    """
    project_id, _clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {
                    "channel": c,
                    "filter": "butterworth",
                    "params": {"cutoff_hz": 1.0, "order": 2, "strength": 1.0},
                    "enabled": True,
                }
                for c in ("x", "y", "r", "s")
            ]
        },
    )
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {
                "points": [{"id": "t1", "point_ids": [point_ids[0]], "source": "smoothed"}]
            }
        },
    )

    trail = {
        row["point_id"]: row
        for row in client.get(f"/api/projects/{project_id}/smoothing/tracks").json()["points"]
    }[point_ids[0]]

    from stable.server import motion as motion_glue  # noqa: PLC0415 - test-local import
    from stable.server.routes.solve import load_options, load_smoothed  # noqa: PLC0415

    project = workspace.load_project(project_id)
    motion = load_smoothed(workspace, project_id, project)
    plan = motion_glue.build_plan(project, motion, options=load_options(workspace, project_id))
    mark = [plan.geometry.width / 2, plan.geometry.height / 2]

    drifted = False
    for frame in (0, plan.n_frames // 2, plan.n_frames - 1):
        row = client.get(f"/api/projects/{project_id}/export/preview?frame={frame}").json()[
            "targets"
        ][0]
        # The dashed trail is what is held, so it is *on* the mark, on every
        # frame — which is the whole point of naming it as the source.
        assert row["landed"] == pytest.approx(mark, abs=0.5)
        assert row["residual"] == pytest.approx(0.0, abs=0.5)
        back = plan.output_transform(frame).inverse().apply(np.asarray(row["landed"]))
        assert back == pytest.approx(
            [trail["x"][frame], trail["y"][frame]], abs=0.5
        ), "what landed on the mark is not the dashed trail the viewer draws"
        drifted = drifted or abs(trail["x"][frame] - trail["x"][0]) > 1.0
    # ...and the trail really does move, or this proves nothing.
    assert drifted, "the dashed trail never moved, so nothing was tested"


def test_on_track_keeps_the_point_on_the_mark_so_the_crosshair_never_drifts(
    client, solved_project
):
    """The other half of the pair: a lock reports zero drift on every frame."""
    project_id, _clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1", "point_ids": [point_ids[0]]}]}},
    )
    for frame in (0, 5, 9):
        row = client.get(f"/api/projects/{project_id}/export/preview?frame={frame}").json()[
            "targets"
        ][0]
        assert row["residual"] == pytest.approx(0.0, abs=0.5)


def test_the_crop_is_re_solved_around_wherever_the_mark_is_put(client, solved_project):
    """Moving the mark changes the crop, and the mark is hit either way.

    The rectangle is grown *around* the mark rather than around the frame
    centre, so pushing the mark to one edge leaves room on only one side of it —
    a different rectangle, and generally a different size. Which way it goes
    depends on where the shot's common area sits, so the claim here is that it
    is re-solved, not that off-centre is always worse.
    """
    project_id, _clip, point_ids = solved_project
    base = {"id": "t1", "point_ids": [point_ids[0]]}
    reports = []
    for u in (0.05, 0.95):
        client.put(
            f"/api/projects/{project_id}/render/options",
            json={"target": {"points": [{**base, "place": {"u": u, "v": 0.5}}]}},
        )
        reports.append(client.get(f"/api/projects/{project_id}/export/preview").json())

    assert reports[0]["coverage"] != pytest.approx(reports[1]["coverage"])
    # ...and the mark is hit exactly in both, which is the whole contract.
    for report in reports:
        row = report["targets"][0]
        assert row["point_id"] == point_ids[0]
        assert row["residual"] == pytest.approx(0.0, abs=0.5)


def test_a_placement_outside_the_output_frame_is_rejected(client, solved_project):
    project_id, _clip, point_ids = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {
                "points": [
                    {"id": "t1", "point_ids": [point_ids[0]], "place": {"u": 1.4, "v": 0.5}}
                ]
            }
        },
    )
    assert response.status_code == 400
    assert "placement" in response.json()["detail"]


def test_averaging_a_chosen_few_points_is_a_target_of_its_own(client, solved_project):
    project_id, _clip, point_ids = solved_project
    stored = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1", "point_ids": point_ids[:2]}]}},
    )
    assert stored.status_code == 200, stored.text
    assert stored.json()["options"]["target"]["points"][0]["point_ids"] == point_ids[:2]
    result = export(client, project_id)
    assert result["target"] == "mean(p1+p2)"
    assert result["max_displacement"] > 5.0


def test_selecting_no_points_at_all_falls_back_to_averaging_them_all(client, solved_project):
    """Unticking the last point is a thing to answer, not a thing to refuse.

    The value type refuses an empty selection outright — there is nothing to
    average — but reading one from a stored payload normalises it to the mode
    that means the same thing with no points named. That is also what makes
    unticking the last checkbox in the panel work without a special case there.
    """
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1", "point_ids": []}]}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["options"]["target"]["points"][0]["point_ids"] == []


# -- the export range, and the crop measured over it (todo 1, eighth round) ----------------------


def test_the_export_range_overrides_the_in_out_markers(client, solved_project):
    """Tracking wants a margin either side of the shot; delivery does not."""
    project_id, clip, _points = solved_project
    client.post(f"/api/projects/{project_id}/render/targets", json={})
    client.put(f"/api/projects/{project_id}/range", json={"in": 2, "out": clip.frame_count})

    # With no override the render follows the markers.
    following = client.get(f"/api/projects/{project_id}/render/options").json()
    assert following["export_window"] == [2, clip.frame_count]

    overridden = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"export_range": {"start": 5, "end": 11}},
    )
    assert overridden.status_code == 200, overridden.text
    assert overridden.json()["export_window"] == [5, 11]
    # ...and the markers are untouched, because they still bound tracking.
    assert client.get(f"/api/projects/{project_id}/range").json()["in"] == 2

    result = export(client, project_id)
    assert result["frame_range"] == [5, 11]
    assert result["frames"] == 6


def test_clearing_the_export_range_goes_back_to_following_the_markers(client, solved_project):
    project_id, clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/range", json={"in": 3, "out": clip.frame_count - 1})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"export_range": {"start": 6, "end": 9}},
    )
    cleared = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"export_range": {"start": None, "end": None}},
    )
    assert cleared.json()["export_window"] == [3, clip.frame_count - 1]


def test_an_inverted_export_range_is_refused(client, solved_project):
    project_id, _clip, _points = solved_project
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"export_range": {"start": 9, "end": 4}},
    )
    assert response.status_code == 400
    assert "end after it starts" in response.json()["detail"]


def test_fitting_the_crop_to_the_export_range_keeps_more_of_the_frame(client, solved_project):
    """Frames that are not being written get no vote on the crop.

    The clip's motion accumulates, so the whole-clip crop is set by its worst
    moment. An export of a few frames in the middle should be croppable as if
    the rest were not there — which is what the operator means by "the max crop
    that doesn't include regions outside the video between these markers".
    """
    project_id, _clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {"points": [{"id": "t1", "point_ids": [point_ids[0]]}]},
            "export_range": {"start": 0, "end": 4},
        },
    )
    whole = client.get(f"/api/projects/{project_id}/export/preview").json()

    fitted = client.put(
        f"/api/projects/{project_id}/render/options", json={"crop": {"limit_to_range": True}}
    )
    assert fitted.status_code == 200, fitted.text
    ranged = client.get(f"/api/projects/{project_id}/export/preview").json()

    assert ranged["crop_measured_frames"] == 4
    assert whole["crop_measured_frames"] > 4
    assert ranged["coverage"] > whole["coverage"]
    assert ranged["zoom"] < whole["zoom"]
    # Still gapless over the frames actually written — that is the promise.
    assert ranged["gaps"] is False


def test_moving_an_in_marker_re_plans_a_range_fitted_crop(client, solved_project):
    """The memoised plan has to notice the markers, not just the options.

    The export range falls back to them, so a crop fitted to that range depends
    on where they are — and a cache key that did not say so would keep serving
    the crop from where the In point used to be.
    """
    project_id, clip, point_ids = solved_project
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {"points": [{"id": "t1", "point_ids": [point_ids[0]]}]},
            "crop": {"limit_to_range": True},
        },
    )
    wide = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert wide["crop_measured_frames"] == clip.frame_count

    client.put(f"/api/projects/{project_id}/range", json={"in": 0, "out": 4})
    narrow = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert narrow["crop_measured_frames"] == 4
    assert narrow["coverage"] > wide["coverage"]


# -- border fill ----------------------------------------------------------------


def test_border_fill_replaces_the_black_edge_in_the_preview(client, solved_project):
    """The uncovered edge stops being black once a fill is chosen."""
    project_id, _clip, _points = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"points": [{"id": "t1"}]}, "crop": {"scale": 1.0}},
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
                "target": {"points": [{"id": "t1"}]},
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
    # Every mode holds *something*, so the default is the one that always
    # resolves for a project that can be exported at all (todo 1, eighth round).
    assert fetched["options"]["target"]["points"] == []
    assert fetched["target_sources"] == ["raw", "smoothed"]
    assert fetched["fill_modes"] == ["solid", "stretch", "mirror", "blur"]
    # Only tracked points can be a target, so the UI cannot offer a broken one.
    assert [p["id"] for p in fetched["points"]] == point_ids
    # The frames a render would write, and the markers they fall back to.
    assert fetched["export_window"] == fetched["clip_range"]

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
        json={
            "target": {"points": [{"id": "t1", "point_ids": [point_ids[1]]}]},
            "crop": {"scale": 0.6},
        },
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


# -- target points: adding, moving, deleting, mapping -----------------------------


def test_targets_are_added_moved_and_deleted_one_at_a_time(client, solved_project):
    """The panel and the canvas both address one mark, so the API does too.

    A whole-list PUT would make moving one mark carry a possibly stale copy of
    the others, which is exactly the class of bug the single-mark routes exist
    to remove.
    """
    project_id, _clip, point_ids = solved_project

    first = client.post(
        f"/api/projects/{project_id}/render/targets",
        json={"point_ids": [point_ids[0]], "place": {"u": 0.25, "v": 0.25}},
    )
    assert first.status_code == 200, first.text
    second = client.post(
        f"/api/projects/{project_id}/render/targets",
        json={"point_ids": [point_ids[1]], "source": "smoothed"},
    )
    points = second.json()["options"]["target"]["points"]
    # The server mints the ids, so two clients adding at once cannot collide.
    assert [p["id"] for p in points] == ["t1", "t2"]
    assert points[1]["source"] == "smoothed"

    moved = client.patch(
        f"/api/projects/{project_id}/render/targets/t1", json={"place": {"u": 0.75, "v": 0.6}}
    )
    assert moved.json()["options"]["target"]["points"][0]["place"] == pytest.approx(
        {"u": 0.75, "v": 0.6}
    )
    # ...and moving one leaves the other exactly as it was.
    assert moved.json()["options"]["target"]["points"][1] == points[1]

    dropped = client.delete(f"/api/projects/{project_id}/render/targets/t1")
    assert [p["id"] for p in dropped.json()["options"]["target"]["points"]] == ["t2"]
    assert client.delete(f"/api/projects/{project_id}/render/targets/t9").status_code == 404


def test_deleting_the_last_target_falls_back_to_the_smoothing_warp(client, solved_project):
    """Removing every mark is allowed, and means something."""
    project_id, _clip, point_ids = solved_project
    client.post(f"/api/projects/{project_id}/render/targets", json={"point_ids": [point_ids[0]]})
    cleared = client.delete(f"/api/projects/{project_id}/render/targets/t1")
    assert cleared.json()["options"]["target"]["points"] == []

    preview = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert preview["target_label"] == "smoothing only"
    # No smoothing is configured either, so it is honestly a no-op.
    assert preview["is_no_op"] is True


def test_a_mark_over_several_tracks_holds_their_mean(client, solved_project):
    project_id, _clip, point_ids = solved_project
    added = client.post(
        f"/api/projects/{project_id}/render/targets", json={"point_ids": point_ids[:3]}
    )
    assert added.status_code == 200, added.text
    result = export(client, project_id)
    assert result["target"] == f"mean({'+'.join(point_ids[:3])})"
    assert result["max_displacement"] > 5.0


def test_two_marks_fix_rotation_and_scale_and_both_land_exactly(client, solved_project):
    """Item 4: the picture is fitted to the marks, least squares."""
    project_id, _clip, point_ids = solved_project
    client.post(
        f"/api/projects/{project_id}/render/targets",
        json={"point_ids": [point_ids[0]], "place": {"u": 0.3, "v": 0.35}},
    )
    client.post(
        f"/api/projects/{project_id}/render/targets",
        json={"point_ids": [point_ids[1]], "place": {"u": 0.7, "v": 0.65}},
    )
    preview = client.get(f"/api/projects/{project_id}/export/preview?frame=5").json()
    assert preview["uses_rotation_and_scale"] is True
    assert len(preview["targets"]) == 2
    for row in preview["targets"]:
        assert row["residual"] == pytest.approx(0.0, abs=0.5)


def test_three_marks_that_disagree_are_a_least_squares_fit_not_a_failure(client, solved_project):
    """A similarity has four degrees of freedom; three marks ask for six."""
    project_id, _clip, point_ids = solved_project
    for point_id, place in zip(
        point_ids[:3],
        ({"u": 0.2, "v": 0.2}, {"u": 0.8, "v": 0.2}, {"u": 0.2, "v": 0.9}),
        strict=True,
    ):
        client.post(
            f"/api/projects/{project_id}/render/targets",
            json={"point_ids": [point_id], "place": place},
        )
    preview = client.get(f"/api/projects/{project_id}/export/preview?frame=5").json()
    assert len(preview["targets"]) == 3
    misses = [row["residual"] for row in preview["targets"]]
    assert max(misses) > 1.0, "this arrangement should not be exactly satisfiable"
    # ...and the error is spread rather than one mark being abandoned.
    assert max(misses) < 200.0


# -- delivery resolution ----------------------------------------------------------


def test_the_output_resolution_scales_the_crop(client, solved_project):
    """Item 7: delivery size is a separate decision from field of view."""
    project_id, _clip, point_ids = solved_project
    client.put(f"/api/projects/{project_id}/render", json={"border": "crop"})
    client.post(f"/api/projects/{project_id}/render/targets", json={"point_ids": [point_ids[0]]})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"crop": {"scale": 0.5}, "resolution": {"width": 640, "height": 360}},
    )
    result = export(client, project_id)
    assert (result["width"], result["height"]) == (640, 360)
    # The crop still says how much of the frame survives, not how big the file is.
    assert result["crop_scale"] == pytest.approx(0.5)


def test_a_square_delivery_crops_a_square_window(client, solved_project):
    """The crop window takes the delivered shape rather than being squeezed into it."""
    project_id, _clip, point_ids = solved_project
    client.post(f"/api/projects/{project_id}/render/targets", json={"point_ids": [point_ids[0]]})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"resolution": {"width": 480, "height": 480}},
    )
    preview = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert (preview["output"]["width"], preview["output"]["height"]) == (480, 480)
    # A mark at the centre of a square output still lands there exactly.
    row = preview["targets"][0]
    assert row["target"] == pytest.approx([240.0, 240.0])
    assert row["residual"] == pytest.approx(0.0, abs=0.5)


def test_clearing_the_output_resolution_goes_back_to_the_crop_size(client, solved_project):
    project_id, _clip, point_ids = solved_project
    client.post(f"/api/projects/{project_id}/render/targets", json={"point_ids": [point_ids[0]]})
    client.put(
        f"/api/projects/{project_id}/render/options", json={"resolution": {"width": 800}}
    )
    scaled = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert scaled["output"]["width"] == 800

    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"resolution": {"width": None, "height": None}},
    )
    natural = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert natural["output"]["width"] != 800
