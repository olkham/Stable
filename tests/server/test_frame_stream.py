"""The frame WebSocket: one message carrying a frame and the geometry over it.

Todo 2, fifth round, asked whether frames could be streamed rather than fetched
one GET at a time, with one non-negotiable: *the frames must be in sync with the
track data*. Both halves are asserted here.

The **sync** half is structural rather than timing-dependent: a want answers
with the pixels and the export geometry for that frame in a single binary
message, so there is no ordering in which a client could draw one frame's
overlay over another frame's picture. The tests below pin that the geometry in
that message is the same geometry ``GET /export/preview?frame=n`` reports, and
that the pixels are the same bytes the REST frame routes serve — one
implementation behind both (:mod:`stable.server.frames`), so the socket and the
routes cannot drift apart and quietly disagree about what frame 7 looks like.

The **performance** half is that superseded work is dropped: a burst of wants
during a scrub is answered for the newest frame, not for every frame passed
over. That is what a request-per-frame transport structurally cannot do.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from stable.server.frames import frame_geometry, geometry_context

from ._helpers import seed_tracks, upload_clip, wait_job

FRAMES_WS = "/ws/projects/{}/frames"


def read_frame(socket) -> tuple[dict, bytes]:
    """Read one binary frame message: ``[uint32 header length][header][JPEG]``."""
    raw = socket.receive_bytes()
    size = int.from_bytes(raw[:4], "big")
    return json.loads(raw[4 : 4 + size]), raw[4 + size :]


@pytest.fixture
def streamable(client: TestClient, workspace, tmp_path, warped_clip):
    """A project with seeded tracks, solved and smoothed — so both variants exist.

    Tracks are seeded rather than tracked: what is under test is the transport,
    not how well a tracker does on a synthetic clip.
    """
    from conftest import encode_synthetic_clip  # noqa: PLC0415 - pytest puts tests/ on sys.path

    path = encode_synthetic_clip(warped_clip, str(tmp_path / "warped.mp4"), with_audio=False)
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)

    features = warped_clip.feature_points(6)
    seed_tracks(workspace, project_id, warped_clip.true_tracks(features).tracks())
    submitted = client.post(f"/api/projects/{project_id}/solve")
    assert wait_job(client, project_id, submitted.json()["job_id"], timeout=60)["status"] == "done"
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={"filters": [{"channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0}}]},
    )
    return project_id


def test_a_source_frame_is_the_same_bytes_the_rest_route_serves(client, streamable):
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 1, "frame": 5, "variants": ["source"]})
        header, payload = read_frame(socket)

    assert header["type"] == "frame"
    assert header["frame"] == 5
    assert header["variant"] == "source"
    assert header["seq"] == 1
    assert payload == client.get(f"/api/projects/{project_id}/frames/5").content


def test_an_output_frame_is_the_same_bytes_the_stabilized_route_serves(client, streamable):
    """The picture the side-by-side canvas draws is the picture the export writes.

    Both go through :func:`stable.server.frames.output_frame_jpeg`, so this is a
    guard against that ever becoming two implementations again.
    """
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 2, "frame": 5, "variants": ["output"]})
        header, payload = read_frame(socket)

    assert header["variant"] == "output"
    rest = client.get(f"/api/projects/{project_id}/frames/5/stabilized")
    assert rest.status_code == 200, rest.text
    assert payload == rest.content


def test_the_geometry_arrives_with_the_pixels_and_matches_the_preview(client, streamable):
    """The no-negotiation requirement: overlay and picture are one message.

    A client drawing from this message cannot put frame 7's output rectangle
    over frame 4's picture, whatever the network does — there is nothing to
    interleave.
    """
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 3, "frame": 7, "variants": ["source", "output"]})
        first, _ = read_frame(socket)
        second, _ = read_frame(socket)

    assert {first["variant"], second["variant"]} == {"source", "output"}
    # Both pictures of the frame carry the same geometry, so the two canvases
    # and the overlay between them are describing one moment in the clip.
    assert first["geometry"] == second["geometry"]

    preview = client.get(f"/api/projects/{project_id}/export/preview?frame=7").json()
    geometry = first["geometry"]
    assert geometry["frame"] == 7
    assert geometry["rect"] == preview["rect"]
    assert geometry["warp"] == preview["warp"]
    assert geometry["targets"] == preview["targets"]
    assert geometry["output"] == preview["output"]


def test_the_output_warp_is_the_transform_that_produced_the_output_rectangle(client, streamable):
    """What the side-by-side canvas draws its crosshairs with (todo 4, sixth round).

    ``rect`` is where the output rectangle sits on the *source* frame; the export
    panel's right-hand canvas is in **output** pixels, so it needs the map the
    other way — source in, output out — to put a stabilisation target on the
    picture it will actually be delivered in, and to read a drag back off it.

    That map is not new information: ``rect`` is its inverse applied to the
    output corners. This asserts exactly that, which is what makes sending it
    safe — the two can never describe different framings, because one is
    computed from the other.
    """
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 9, "frame": 6, "variants": ["source"]})
        header, _ = read_frame(socket)

    geometry = header["geometry"]
    warp = geometry["output_warp"]
    width = geometry["output"]["width"]
    height = geometry["output"]["height"]

    cos = math.cos(warp["rotation"]) * warp["scale"]
    sin = math.sin(warp["rotation"]) * warp["scale"]
    corners = [
        (cos * x - sin * y + warp["tx"], sin * x + cos * y + warp["ty"])
        for x, y in geometry["rect"]
    ]
    expected = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    for (got_x, got_y), (want_x, want_y) in zip(corners, expected, strict=True):
        assert got_x == pytest.approx(want_x, abs=1e-6)
        assert got_y == pytest.approx(want_y, abs=1e-6)

    # And the REST preview says the same thing, because it is the socket's
    # fallback: a viewer that loses its socket must not draw a different frame.
    preview = client.get(f"/api/projects/{project_id}/export/preview?frame=6").json()
    assert preview["output_warp"] == warp


def test_superseded_wants_are_dropped_rather_than_queued(client, streamable):
    """A scrub past eighteen frames must not cost eighteen warps.

    The newest want is always answered; the ones the playhead has already left
    behind are not. That is the whole performance argument for the socket, and
    it is the one thing a request-per-frame transport cannot do — over HTTP the
    server has no way to learn that frame 244 stopped mattering.

    In practice this settles at two replies (the one already in flight when the
    burst started, then the newest); the assertion is only that it is fewer than
    one per want, which is the claim being made.
    """
    project_id = streamable
    wanted = list(range(2, 20))
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        for seq, frame in enumerate(wanted):
            socket.send_json({"type": "want", "seq": seq, "frame": frame, "variants": ["output"]})
        served = []
        while not served or served[-1] != wanted[-1]:
            header, _ = read_frame(socket)
            served.append(header["frame"])

    assert served[-1] == wanted[-1]
    assert len(served) < len(wanted), f"served every want instead of coalescing: {served}"


def test_an_unsolved_project_gets_an_error_for_output_and_pixels_for_source(client, tiny_clip):
    """'Not solved yet' is a state the viewer shows, not a dropped connection."""
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)

    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 1, "frame": 0, "variants": ["output"]})
        failure = socket.receive_json()
        assert failure["type"] == "error"
        assert failure["status"] == 409
        assert failure["variant"] == "output"

        # The same socket still serves the raw proxy, which is what the viewer
        # needs to show anything at all before a solve exists.
        socket.send_json({"type": "want", "seq": 2, "frame": 0, "variants": ["source"]})
        header, payload = read_frame(socket)
        assert header["variant"] == "source"
        assert header["geometry"] is None
        assert payload == client.get(f"/api/projects/{project_id}/frames/0").content


def test_the_revision_changes_when_the_render_options_do(client, streamable):
    """How the client knows a cached output frame is stale.

    Output frames cannot be cached by URL — their pixels change with the crop,
    the fill and the smoothing — so every one is stamped with the identity of
    the inputs it was rendered from, and the client drops everything under an
    older stamp. Source frames are unaffected by any of that.
    """
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 1, "frame": 3, "variants": ["source", "output"]})
        first, _ = read_frame(socket)
        second, _ = read_frame(socket)
    stamps = {first["variant"]: first["rev"], second["variant"]: second["rev"]}

    changed = client.put(
        f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 0.8}}
    )
    assert changed.status_code == 200, changed.text

    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "want", "seq": 2, "frame": 3, "variants": ["source", "output"]})
        first, _ = read_frame(socket)
        second, _ = read_frame(socket)
    after = {first["variant"]: first["rev"], second["variant"]: second["rev"]}

    assert after["output"] != stamps["output"]
    assert after["source"] == stamps["source"]


def test_prefetch_is_served_and_carries_no_geometry(client, streamable):
    """Guesses about where the playhead is going are pixels only.

    Geometry describes the frame being drawn *now*, so a prefetch reply has no
    ``geometry`` key at all — not a null one. A client assembling "this frame,
    whole" must not be able to mistake a guess for an answer about the frame it
    is drawing, which is the desynchronisation this round exists to remove.
    """
    project_id = streamable
    with client.websocket_connect(FRAMES_WS.format(project_id)) as socket:
        socket.send_json({"type": "prefetch", "frames": [9], "variants": ["source"]})
        header, payload = read_frame(socket)

    assert header["frame"] == 9
    assert "geometry" not in header
    assert payload == client.get(f"/api/projects/{project_id}/frames/9").content


# -- per frame, only per-frame work (REVIEW.md P1, R3) ----------------------------


def _hold(client: TestClient, project_id: str, *marks: dict) -> None:
    response = client.put(
        f"/api/projects/{project_id}/render/options", json={"target": {"points": list(marks)}}
    )
    assert response.status_code == 200, response.text


def test_the_geometry_context_is_built_once_and_reused_until_an_input_changes(
    client, workspace, streamable
):
    """The export tab asks for geometry on every frame it draws.

    Everything except the frame's own rows is the same for all of them, so it
    is built once and handed back unchanged — and rebuilt the moment any file
    it came from changes, or the preview would describe settings that are gone.
    """
    project_id = streamable
    first = geometry_context(workspace, project_id)
    assert first is not None
    assert geometry_context(workspace, project_id) is first

    _hold(client, project_id, {"id": "t1", "point_ids": ["p1"], "place": {"u": 0.5, "v": 0.5}})
    second = geometry_context(workspace, project_id)
    assert second is not first
    assert second.revision != first.revision
    assert second.targets is not None
    assert geometry_context(workspace, project_id) is second


def test_cached_geometry_is_the_geometry_computed_from_scratch(client, workspace, streamable):
    """The cache changes what is computed per frame, never what comes out.

    One raw mark and one smoothed mark, the costly kind: a smoothed path is
    every frame of the track pushed through the smoothing's warp.
    """
    project_id = streamable
    _hold(
        client,
        project_id,
        {"id": "t1", "point_ids": ["p1"], "place": {"u": 0.4, "v": 0.4}},
        {"id": "t2", "point_ids": ["p2"], "source": "smoothed", "place": {"u": 0.6, "v": 0.6}},
    )
    context = geometry_context(workspace, project_id)
    assert context is not None and context.targets is not None
    for frame in (0, 7, 20):
        cached = frame_geometry(
            context.project,
            context.plan,
            context.options,
            frame,
            context.motion,
            targets=context.targets,
        )
        fresh = frame_geometry(
            context.project, context.plan, context.options, frame, context.motion
        )
        assert cached == fresh
        assert all(row["landed"] is not None for row in cached["targets"])


def test_a_change_to_what_a_target_holds_reaches_the_preview(client, streamable):
    """A target holds the *current* tracks, so changing them changes the plan.

    The interactive edits (corrections, undo, anchors) clear the plan cache
    themselves, but nothing else that rewrites the project did — a finished
    track or retrack job, a point taken out of the solve, a derived or deleted
    point. After one of those the preview kept the old fit while reporting
    against the new tracks, and the export, which plans afresh, did something
    else again. The plan is now keyed on the project file, so every writer
    counts.
    """
    project_id = streamable
    # Held: the mean of every solve-enabled track, the default kind of target.
    _hold(client, project_id, {"id": "t1", "point_ids": [], "place": {"u": 0.5, "v": 0.5}})
    before = client.get(f"/api/projects/{project_id}/export/preview?frame=10").json()
    assert before["targets"][0]["residual"] < 0.5  # one mark is held exactly

    dropped = client.patch(
        f"/api/projects/{project_id}/points/p1", json={"enabled_for_solve": False}
    )
    assert dropped.status_code == 200, dropped.text

    after = client.get(f"/api/projects/{project_id}/export/preview?frame=10").json()
    # Still held exactly, which is only true if the plan was fitted to the mean
    # of the five tracks now held, not the six it was first fitted to.
    assert after["targets"][0]["residual"] < 0.5
    assert after["output_warp"] != before["output_warp"]
