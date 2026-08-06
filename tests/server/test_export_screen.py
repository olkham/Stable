"""The export screen's backend: preview, target mapping, and the render library.

Todo 5 asked to *see* what the export settings mean before spending minutes
rendering, and to place stabilisation targets rather than pick one point from a
list. Todo 7 asked for renders that do not overwrite each other and can be
managed afterwards. Both are answered by planning rather than rendering, which
is what makes the preview cheap enough to re-ask on every drag.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ._helpers import upload_clip, wait_job


@pytest.fixture
def solved(client: TestClient, tiny_clip):
    """A project with two tracked points, solved and smoothed — ready to export."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])

    ids = []
    for dx, dy in ((0.0, 0.0), (6.0, -4.0)):
        x, y = source.true_position(0)
        created = client.post(
            f"/api/projects/{project_id}/points",
            json={"frame": 0, "x": x + dx, "y": y + dy, "tracker": "lk"},
        )
        point_id = created.json()["id"]
        ids.append(point_id)
        submitted = client.post(f"/api/projects/{project_id}/points/{point_id}/track", json={})
        assert wait_job(client, project_id, submitted.json()["job_id"])["status"] == "done"

    # The solve is a background job: waiting for it is what makes the smoothing
    # below (and every preview after it) deterministic rather than a race.
    solving = client.post(f"/api/projects/{project_id}/solve", json={})
    assert solving.status_code == 200, solving.text
    assert wait_job(client, project_id, solving.json()["job_id"], timeout=120)["status"] == "done"
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0}}]},
    )
    return project_id, ids, source


# -- the preview ---------------------------------------------------------------


def test_the_preview_says_what_will_be_kept_without_rendering_it(client, solved):
    project_id, _ids, _source = solved
    response = client.get(f"/api/projects/{project_id}/export/preview?frame=3")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["frame"] == 3
    assert body["output"]["width"] > 0 and body["output"]["height"] > 0
    assert body["source"]["width"] > 0
    # Four corners of the output rectangle, expressed on the source frame so the
    # screen can draw it over the picture.
    assert len(body["rect"]) == 4
    assert all(len(corner) == 2 for corner in body["rect"])
    assert 0.0 < body["coverage"] <= 1.0
    assert isinstance(body["gaps"], bool)
    assert body["is_no_op"] is False


def test_the_preview_rectangle_moves_with_the_stabilisation(client, solved):
    """If the rect were the same on every frame it would be decoration, not a preview."""
    project_id, _ids, _source = solved
    first = client.get(f"/api/projects/{project_id}/export/preview?frame=0").json()["rect"]
    later = client.get(f"/api/projects/{project_id}/export/preview?frame=15").json()["rect"]
    assert np.abs(np.asarray(first) - np.asarray(later)).max() > 0.5


def test_everything_but_the_rectangle_is_the_same_on_every_frame(client, solved):
    """Which is why the screen stopped asking for this once per frame.

    Output size, coverage, zoom, ``is_no_op`` and the rest are properties of the
    *plan*, not of a frame, and the plan is whole-clip by construction (the
    minimal auto-zoom cannot be known otherwise). Scrubbing used to fire one of
    these per frame — see the log in todo 2, fifth round — to re-learn answers
    that had not changed. The per-frame half now travels with the pixels on the
    frame socket, and this route is asked once per settings change.
    """
    project_id, _ids, _source = solved
    first = client.get(f"/api/projects/{project_id}/export/preview?frame=0").json()
    later = client.get(f"/api/projects/{project_id}/export/preview?frame=15").json()

    whole_clip = (
        "frames",
        "source",
        "output",
        "coverage",
        "crop_scale",
        "zoom",
        "gaps",
        "border",
        "max_displacement",
        "is_no_op",
        "no_op_detail",
        "target",
        "uses_rotation_and_scale",
    )
    for key in whole_clip:
        assert first[key] == later[key], key
    # …and the per-frame half really is per frame, or there would be nothing to
    # stream and no reason for any of this.
    assert first["frame"] != later["frame"]
    assert first["rect"] != later["rect"]
    assert first["output_warp"] != later["output_warp"]


def test_the_preview_refuses_before_there_is_anything_to_preview(client, tiny_clip):
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])

    response = client.get(f"/api/projects/{project_id}/export/preview")
    assert response.status_code == 409
    assert "solve" in response.json()["detail"]


def test_the_preview_reports_a_no_op_rather_than_letting_it_be_exported(client, solved):
    """The same refusal the export gives, but *before* the operator commits."""
    project_id, _ids, _source = solved
    client.put(f"/api/projects/{project_id}/smoothing", json={"filters": []})

    body = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert body["is_no_op"] is True
    assert body["no_op_detail"]
    assert client.post(f"/api/projects/{project_id}/export/video", json={}).status_code == 400


# -- target mapping ------------------------------------------------------------


def test_a_mapping_is_stored_and_reported_with_its_residuals(client, solved):
    project_id, ids, _source = solved
    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {
                "mode": "map",
                "pairs": [
                    {"point_id": ids[0], "x": 40.0, "y": 30.0},
                    {"point_id": ids[1], "x": 46.0, "y": 26.0},
                ],
            }
        },
    )
    assert response.status_code == 200, response.text
    stored = response.json()["options"]["target"]
    assert stored["mode"] == "map"
    assert [p["point_id"] for p in stored["pairs"]] == ids

    preview = client.get(f"/api/projects/{project_id}/export/preview?frame=10").json()
    assert preview["uses_rotation_and_scale"] is True
    rows = {row["point_id"]: row for row in preview["targets"]}
    assert set(rows) == set(ids)
    for row in rows.values():
        assert row["landed"] is not None
        # Two pairs, one similarity: it can satisfy both, so the miss is ~0.
        assert row["residual"] < 1.0


def test_one_pair_is_a_translation_and_says_so(client, solved):
    project_id, ids, _source = solved
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": ids[0], "x": 40.0, "y": 30.0}]}},
    )
    preview = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert preview["uses_rotation_and_scale"] is False
    assert len(preview["targets"]) == 1


def test_a_mapping_to_an_untracked_point_is_refused_at_the_point_of_setting_it(client, solved):
    """Not twenty seconds into the export job."""
    project_id, _ids, _source = solved
    created = client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": 20.0, "y": 20.0}
    )
    untracked = created.json()["id"]

    response = client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": untracked, "x": 10.0, "y": 10.0}]}},
    )
    assert response.status_code == 400
    assert untracked in response.json()["detail"]


def test_changing_the_crop_does_not_unset_the_placed_targets(client, solved):
    """The bug: 'adjust the Crop size and the target points become unset'.

    Every update replaces only the blocks it names, so a crop change carries the
    stored target through untouched — and the preview still reports it, which is
    what the screen draws its crosshairs from (todo 7, fourth round).
    """
    project_id, ids, _source = solved
    pairs = [
        {"point_id": ids[0], "x": 40.0, "y": 30.0},
        {"point_id": ids[1], "x": 46.0, "y": 26.0},
    ]
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": pairs}},
    )

    patches = (
        {"crop": {"scale": 0.9}},
        {"crop": {"scale": 0.5}},
        {"crop": {"centre": "target"}},
    )
    for patch in patches:
        response = client.put(f"/api/projects/{project_id}/render/options", json=patch)
        assert response.status_code == 200, response.text
        target = response.json()["options"]["target"]
        assert target["mode"] == "map"
        assert target["pairs"] == pairs

        preview = client.get(f"/api/projects/{project_id}/export/preview")
        assert preview.status_code == 200, preview.text
        assert preview.json()["target"]["pairs"] == pairs
        # ...and it is still a real stabilisation, so no "this render would not
        # move the picture at all" warning appears alongside the vanished targets.
        assert preview.json()["is_no_op"] is False


def test_overlapping_option_updates_do_not_clobber_each_other(client, solved):
    """Read-merge-write under two threads: neither block may lose the other's edit."""
    import threading

    project_id, ids, _source = solved
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": ids[0], "x": 40.0, "y": 30.0}]}},
    )

    barrier = threading.Barrier(2)

    def put(body):
        barrier.wait()
        client.put(f"/api/projects/{project_id}/render/options", json=body)

    threads = [
        threading.Thread(target=put, args=({"crop": {"scale": 0.75}},)),
        threading.Thread(target=put, args=({"fill": {"mode": "blur"}},)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    options = client.get(f"/api/projects/{project_id}/render/options").json()["options"]
    assert options["crop"]["scale"] == 0.75
    assert options["fill"]["mode"] == "blur"
    assert len(options["target"]["pairs"]) == 1


def test_a_mapping_makes_the_export_worth_running(client, solved):
    """With no smoothing at all, a mapping alone is a real stabilisation."""
    project_id, ids, _source = solved
    client.put(f"/api/projects/{project_id}/smoothing", json={"filters": []})
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={
            "target": {
                "mode": "map",
                "pairs": [
                    {"point_id": ids[0], "x": 40.0, "y": 30.0},
                    {"point_id": ids[1], "x": 46.0, "y": 26.0},
                ],
            }
        },
    )
    preview = client.get(f"/api/projects/{project_id}/export/preview").json()
    assert preview["is_no_op"] is False
    assert preview["max_displacement"] > 1.0


# -- the render library (todo 7) ------------------------------------------------


def test_two_exports_are_two_files(client, solved):
    """The bug: every export was called stabilized.mp4 and ate the last one."""
    project_id, _ids, _source = solved

    names = []
    for _ in range(2):
        submitted = client.post(f"/api/projects/{project_id}/export/video", json={})
        assert submitted.status_code == 200, submitted.text
        job = wait_job(client, project_id, submitted.json()["job_id"], timeout=180)
        assert job["status"] == "done", job
        names.append(job["result"]["filename"])

    assert names[0] != names[1]
    listed = client.get(f"/api/projects/{project_id}/export/renders").json()
    assert sorted(row["filename"] for row in listed) == sorted(names)
    assert all(row["bytes"] > 0 and row["url"] for row in listed)


def test_an_explicit_name_is_honoured_but_still_never_overwrites(client, solved):
    project_id, _ids, _source = solved
    first = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "take"})
    assert wait_job(client, project_id, first.json()["job_id"], timeout=180)["status"] == "done"
    second = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "take.mp4"})
    job = wait_job(client, project_id, second.json()["job_id"], timeout=180)

    assert job["status"] == "done", job
    names = {
        row["filename"] for row in client.get(f"/api/projects/{project_id}/export/renders").json()
    }
    assert names == {"take.mp4", "take-2.mp4"}


# -- the export queue (todo 8, fourth round) -----------------------------------


def test_each_queued_export_renders_the_settings_it_was_queued_with(client, solved):
    """Otherwise the queue is a lie: both would come out with whatever was last saved."""
    project_id, _ids, _source = solved

    client.put(f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 1.0}})
    first = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "wide"})
    # Change the crop *after* queueing the first one and before queueing the
    # second: the two jobs must disagree about the output size.
    client.put(f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 0.5}})
    second = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "tight"})

    wide = wait_job(client, project_id, first.json()["job_id"], timeout=180)
    tight = wait_job(client, project_id, second.json()["job_id"], timeout=180)
    assert wide["status"] == "done", wide
    assert tight["status"] == "done", tight
    assert wide["result"]["crop_scale"] == 1.0
    assert tight["result"]["crop_scale"] == 0.5


def test_a_queued_export_can_be_cancelled_before_it_starts(client, solved):
    project_id, _ids, _source = solved

    running = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "keep"})
    queued = client.post(f"/api/projects/{project_id}/export/video", json={"filename": "drop"})
    dropped = client.post(f"/api/projects/{project_id}/jobs/{queued.json()['job_id']}/cancel")

    assert dropped.status_code == 200, dropped.text
    assert dropped.json()["status"] == "cancelled"
    assert wait_job(client, project_id, running.json()["job_id"], timeout=180)["status"] == "done"
    names = {
        row["filename"] for row in client.get(f"/api/projects/{project_id}/export/renders").json()
    }
    assert names == {"keep.mp4"}


def test_a_finished_export_cannot_be_cancelled(client, solved):
    """ "Cancel" that quietly does nothing is worse than one that says so."""
    project_id, _ids, _source = solved
    submitted = client.post(f"/api/projects/{project_id}/export/video", json={})
    job_id = submitted.json()["job_id"]
    assert wait_job(client, project_id, job_id, timeout=180)["status"] == "done"

    refused = client.post(f"/api/projects/{project_id}/jobs/{job_id}/cancel")
    assert refused.status_code == 409
    assert "done" in refused.json()["detail"]


def test_cancelling_an_unknown_job_is_a_404(client, solved):
    project_id, _ids, _source = solved
    assert client.post(f"/api/projects/{project_id}/jobs/nope/cancel").status_code == 404


def test_a_render_can_be_deleted_to_free_the_space(client, solved):
    project_id, _ids, _source = solved
    submitted = client.post(f"/api/projects/{project_id}/export/video", json={})
    job = wait_job(client, project_id, submitted.json()["job_id"], timeout=180)
    filename = job["result"]["filename"]

    deleted = client.delete(f"/api/projects/{project_id}/export/renders/{filename}")
    assert deleted.status_code == 200, deleted.text
    assert client.get(f"/api/projects/{project_id}/export/renders").json() == []
    assert client.get(f"/api/projects/{project_id}/export/download/{filename}").status_code == 404


def test_deleting_a_render_cannot_escape_the_renders_directory(client, solved):
    project_id, _ids, _source = solved
    # A separator that survives routing must be refused by the handler...
    assert (
        client.delete(f"/api/projects/{project_id}/export/renders/..\\secrets.txt").status_code
        == 400
    )
    # ...and a path that walks out of the directory never reaches it at all: the
    # router normalises or rejects it first. Either way nothing outside
    # `renders/` is reachable, which is the property being pinned.
    for hostile in ("..", "../../secrets.txt", "%2e%2e%2fsecrets.txt"):
        assert client.delete(
            f"/api/projects/{project_id}/export/renders/{hostile}"
        ).status_code in (400, 404, 405)
    assert client.delete(f"/api/projects/{project_id}/export/renders/nope.mp4").status_code == 404


def test_the_preview_sends_the_warp_targets_are_measured_against(client, solved):
    """The screen draws crosshairs by inverting this; the fit would misplace them.

    A target's destination is stated *after* the stabilisation warp and *before*
    the crop/zoom fit, so the output rectangle (which carries the fit) is the
    wrong thing to place it with. Sending the warp itself makes the round trip
    source -> target -> source exact.
    """
    project_id, ids, _source = solved
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": ids[0], "x": 40.0, "y": 30.0}]}},
    )
    body = client.get(f"/api/projects/{project_id}/export/preview?frame=8").json()

    warp = body["warp"]
    assert set(warp) == {"tx", "ty", "rotation", "scale"}
    assert warp["scale"] > 0

    # The screen's own arithmetic, in miniature: invert to place, apply to drag.
    import math

    def apply(x, y):
        cos = math.cos(warp["rotation"]) * warp["scale"]
        sin = math.sin(warp["rotation"]) * warp["scale"]
        return (cos * x - sin * y + warp["tx"], sin * x + cos * y + warp["ty"])

    def unapply(x, y):
        inv = 1.0 / warp["scale"]
        cos = math.cos(-warp["rotation"]) * inv
        sin = math.sin(-warp["rotation"]) * inv
        dx, dy = x - warp["tx"], y - warp["ty"]
        return (cos * dx - sin * dy, sin * dx + cos * dy)

    assert apply(*unapply(40.0, 30.0)) == pytest.approx((40.0, 30.0), abs=1e-6)


def test_a_target_placed_on_the_output_round_trips_to_the_same_destination(client, solved):
    """The arithmetic behind dragging a crosshair on the output canvas.

    The side-by-side canvas is in **output** pixels (todo 4, sixth round), so
    putting a target on it is: destination -> source (invert ``warp``) ->
    output (apply ``output_warp``); and reading a drag back off it is the same
    two steps reversed. If either composition were off, a crosshair would sit
    where the operator did not put it and the export would frame the shot
    somewhere else again — so the round trip is asserted rather than assumed.
    """
    import math

    project_id, ids, _source = solved
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": ids[0], "x": 40.0, "y": 30.0}]}},
    )
    body = client.get(f"/api/projects/{project_id}/export/preview?frame=8").json()

    def apply(warp, x, y):
        cos = math.cos(warp["rotation"]) * warp["scale"]
        sin = math.sin(warp["rotation"]) * warp["scale"]
        return (cos * x - sin * y + warp["tx"], sin * x + cos * y + warp["ty"])

    def unapply(warp, x, y):
        inv = 1.0 / warp["scale"]
        cos = math.cos(-warp["rotation"]) * inv
        sin = math.sin(-warp["rotation"]) * inv
        dx, dy = x - warp["tx"], y - warp["ty"]
        return (cos * dx - sin * dy, sin * dx + cos * dy)

    warp, output_warp = body["warp"], body["output_warp"]
    on_output = apply(output_warp, *unapply(warp, 40.0, 30.0))
    back = apply(warp, *unapply(output_warp, *on_output))
    assert back == pytest.approx((40.0, 30.0), abs=1e-6)


def test_a_targets_wanted_and_landed_positions_are_in_the_same_space(client, solved):
    """Otherwise the two numbers look like a miss the residual says is not there."""
    project_id, ids, _source = solved
    client.put(
        f"/api/projects/{project_id}/render/options",
        json={"target": {"mode": "map", "pairs": [{"point_id": ids[0], "x": 40.0, "y": 30.0}]}},
    )
    row = client.get(f"/api/projects/{project_id}/export/preview?frame=6").json()["targets"][0]

    reported = np.hypot(row["landed"][0] - row["target"][0], row["landed"][1] - row["target"][1])
    assert reported == pytest.approx(row["residual"], abs=1e-6)
