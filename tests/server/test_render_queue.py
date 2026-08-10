"""The render queue: queue, reorder, drop, estimate, process (todo 6, tenth round).

The report is "when a render is queued don't auto render it" plus four things
that only become possible once that is true. So these tests are mostly about the
*not* — an entry that sits there until it is asked for — and about the settings
it carries while it sits, since a queue whose entries all render whatever was
saved last is not a queue.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stable.core.types import PointTrack
from stable.server import render_queue

from ._helpers import seed_tracks, upload_clip, wait_job


def drifting_track(n_frames: int, *, x0: float, y0: float) -> PointTrack:
    """A point moving with a wobble on it, so a solve has something to recover."""
    frames = np.arange(n_frames, dtype=np.float32)
    return PointTrack(
        points=np.stack([x0 + 0.5 * frames + np.sin(frames / 3.0), y0 + 0.2 * frames], axis=1),
        confidence=np.full(n_frames, 0.9, dtype=np.float32),
        visible=np.ones(n_frames, dtype=bool),
    )


@pytest.fixture
def renderable(client: TestClient, workspace, tiny_clip):
    """A solved, smoothed project — the state an export can actually be made from."""
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)
    seed_tracks(
        workspace,
        project_id,
        [
            drifting_track(source.frame_count, x0=10.0, y0=10.0),
            drifting_track(source.frame_count, x0=30.0, y0=14.0),
        ],
    )
    submitted = client.post(f"/api/projects/{project_id}/solve")
    wait_job(client, project_id, submitted.json()["job_id"], timeout=60)
    # A render that would move nothing is refused, so give it something to undo.
    smoothed = client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": c, "filter": "gaussian", "params": {"sigma": 4.0}, "enabled": True}
                for c in ("x", "y")
            ]
        },
    )
    assert smoothed.status_code == 200, smoothed.text
    return project_id


def queue_of(client: TestClient, project_id: str) -> dict:
    response = client.get(f"/api/projects/{project_id}/export/queue")
    assert response.status_code == 200, response.text
    return dict(response.json())


def enqueue(client: TestClient, project_id: str, **body) -> dict:
    response = client.post(f"/api/projects/{project_id}/export/queue", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_queueing_a_render_does_not_start_one(client: TestClient, renderable, workspace) -> None:
    """The whole report, in one assertion: queued is not started."""
    project_id = renderable
    payload = enqueue(client, project_id, codec="h264", crf=28)

    assert payload["pending"] == 1
    entry = payload["entries"][0]
    assert entry["status"] == "pending"
    assert entry["job_id"] is None
    # Nothing submitted, and nothing written.
    assert client.get(f"/api/projects/{project_id}/jobs").json() == [] or all(
        job["kind"] != "export" for job in client.get(f"/api/projects/{project_id}/jobs").json()
    )
    assert client.get(f"/api/projects/{project_id}/export/renders").json() == []


def test_a_queued_entry_carries_the_settings_it_was_queued_with(
    client: TestClient, renderable
) -> None:
    """Two entries, two crops. Each must render its own, not the last one saved."""
    project_id = renderable
    client.put(f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 0.9}})
    first = enqueue(client, project_id, filename="wide", codec="h264", crf=20)["entries"][0]

    client.put(f"/api/projects/{project_id}/render/options", json={"crop": {"scale": 0.5}})
    entries = enqueue(client, project_id, filename="tight", codec="vp9", crf=30)["entries"]

    assert [e["filename"] for e in entries] == ["wide.mp4", "tight.mp4"]
    assert first["options"]["crop"]["scale"] == pytest.approx(0.9)
    assert entries[1]["options"]["crop"]["scale"] == pytest.approx(0.5)
    assert [e["codec"] for e in entries] == ["h264", "vp9"]
    assert [e["crf"] for e in entries] == [20, 30]


def test_the_smoothing_in_force_is_captured_too(client: TestClient, renderable) -> None:
    """ "Queue up many different runs" is mostly two smoothings, not two crops."""
    project_id = renderable
    soft = enqueue(client, project_id, filename="soft")["entries"][0]
    assert [spec["params"]["sigma"] for spec in soft["smoothing"]] == [4.0, 4.0]

    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": c, "filter": "gaussian", "params": {"sigma": 12.0}, "enabled": True}
                for c in ("x", "y")
            ]
        },
    )
    entries = enqueue(client, project_id, filename="hard")["entries"]
    assert [spec["params"]["sigma"] for spec in entries[0]["smoothing"]] == [4.0, 4.0]
    assert [spec["params"]["sigma"] for spec in entries[1]["smoothing"]] == [12.0, 12.0]


def test_queued_renders_never_share_a_filename(client: TestClient, renderable) -> None:
    """Three unnamed entries queued in the same second are still three files."""
    project_id = renderable
    for _ in range(3):
        enqueue(client, project_id)
    names = [e["filename"] for e in queue_of(client, project_id)["entries"]]
    assert len(set(names)) == 3


def test_entries_can_be_reordered_and_dropped(client: TestClient, renderable) -> None:
    project_id = renderable
    for name in ("a", "b", "c"):
        enqueue(client, project_id, filename=name)
    ids = [e["id"] for e in queue_of(client, project_id)["entries"]]

    reordered = client.put(
        f"/api/projects/{project_id}/export/queue/order",
        json={"ids": [ids[2], ids[0], ids[1]]},
    )
    assert reordered.status_code == 200
    assert [e["filename"] for e in reordered.json()["entries"]] == ["c.mp4", "a.mp4", "b.mp4"]

    dropped = client.delete(f"/api/projects/{project_id}/export/queue/{ids[0]}")
    assert dropped.status_code == 200
    assert [e["filename"] for e in dropped.json()["entries"]] == ["c.mp4", "b.mp4"]
    assert dropped.json()["pending"] == 2

    assert client.delete(f"/api/projects/{project_id}/export/queue/nope").status_code == 404


def test_a_partial_reorder_leaves_the_unnamed_entries_where_they_were(
    client: TestClient, renderable
) -> None:
    project_id = renderable
    for name in ("a", "b", "c"):
        enqueue(client, project_id, filename=name)
    ids = [e["id"] for e in queue_of(client, project_id)["entries"]]

    reordered = client.put(f"/api/projects/{project_id}/export/queue/order", json={"ids": [ids[1]]})
    assert [e["filename"] for e in reordered.json()["entries"]] == ["b.mp4", "a.mp4", "c.mp4"]


def test_every_entry_carries_an_estimate_and_says_whether_it_is_measured(
    client: TestClient, renderable
) -> None:
    project_id = renderable
    payload = enqueue(client, project_id)
    entry = payload["entries"][0]

    assert entry["frames"] > 0
    assert entry["estimate_seconds"] > 0.0
    # Nothing has been rendered on this machine yet, so the number comes from the
    # seed constant and says so rather than passing a guess off as a measurement.
    assert entry["estimate_measured"] is False
    assert payload["estimate_seconds"] == pytest.approx(entry["estimate_seconds"])


def test_a_finished_render_is_what_the_next_estimate_is_built_from(
    client: TestClient, renderable, workspace
) -> None:
    project_id = renderable
    directory = workspace.project_dir(project_id)
    before, measured_before = render_queue.seconds_per_frame(directory, 1920, 1080)
    assert measured_before is False

    render_queue.record_render(directory, seconds=60.0, frames=100, width=1920, height=1080)
    after, measured_after = render_queue.seconds_per_frame(directory, 1920, 1080)
    assert measured_after is True
    assert after == pytest.approx(0.6, rel=1e-6)
    assert after != pytest.approx(before)


def test_processing_the_queue_renders_every_entry_in_order(
    client: TestClient, renderable, workspace
) -> None:
    project_id = renderable
    for name in ("first", "second"):
        enqueue(client, project_id, filename=name)

    started = client.post(f"/api/projects/{project_id}/export/queue/run")
    assert started.status_code == 200, started.text
    assert started.json()["started"] is True
    assert render_queue.drain_runners(timeout=180.0), "the queue runner did not finish"

    entries = queue_of(client, project_id)["entries"]
    assert [e["status"] for e in entries] == ["done", "done"], entries
    assert all(e["job_id"] for e in entries)

    rendered = [
        f["filename"] for f in client.get(f"/api/projects/{project_id}/export/renders").json()
    ]
    assert sorted(rendered) == ["first.mp4", "second.mp4"]
    # Every entry ran on its own job, one at a time.
    jobs = [
        j for j in client.get(f"/api/projects/{project_id}/jobs").json() if j["kind"] == "export"
    ]
    assert len(jobs) == 2
    assert all(job["status"] == "done" for job in jobs)


def test_running_the_queue_teaches_the_estimator(client: TestClient, renderable, workspace) -> None:
    """The estimate stops being a seed as soon as one render has finished."""
    project_id = renderable
    enqueue(client, project_id, filename="measured")
    client.post(f"/api/projects/{project_id}/export/queue/run")
    assert render_queue.drain_runners(timeout=180.0)

    enqueue(client, project_id, filename="next")
    entry = next(e for e in queue_of(client, project_id)["entries"] if e["filename"] == "next.mp4")
    assert entry["estimate_measured"] is True
    assert entry["estimate_seconds"] > 0.0


def test_an_unrenderable_project_is_refused_at_queue_time(client: TestClient, tiny_clip) -> None:
    """Refusing at the front of a queue, minutes later, is refusing too late."""
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"], timeout=60)

    response = client.post(f"/api/projects/{project_id}/export/queue", json={})
    assert response.status_code == 400
    assert "solve" in response.json()["detail"]
    assert queue_of(client, project_id)["entries"] == []


def test_a_render_that_would_do_nothing_never_enters_the_queue(
    client: TestClient, renderable
) -> None:
    project_id = renderable
    client.put(
        f"/api/projects/{project_id}/smoothing",
        json={
            "filters": [
                {"channel": c, "filter": "gaussian", "params": {"sigma": 4.0}, "enabled": False}
                for c in ("x", "y")
            ]
        },
    )
    response = client.post(f"/api/projects/{project_id}/export/queue", json={})
    assert response.status_code == 400
    assert "would not move the picture" in response.json()["detail"]
    assert queue_of(client, project_id)["entries"] == []


def test_the_queue_survives_a_fresh_server_instance(
    client: TestClient, renderable, workspace
) -> None:
    """It is a sidecar, not process state: closing the tab must not lose it."""
    project_id = renderable
    enqueue(client, project_id, filename="kept")
    reloaded = render_queue.load_queue(workspace.project_dir(project_id))
    assert [entry.filename for entry in reloaded] == ["kept.mp4"]
    assert reloaded[0].is_pending
