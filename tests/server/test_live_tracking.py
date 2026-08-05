"""Live progress out of a running track: the tracker's cursor and partial path.

Todos 4 and 16 ask for the same thing from two directions — do not make the
operator wait for a whole clip before there is anything to look at. Two things
stream out of a job while it runs, and both are checked here:

* the frame the decoder is on, published by the worker from
  :mod:`stable.video.telemetry` (:class:`stable.server.jobs.JobManager`);
* the path found so far, published per completed run by
  :class:`stable.server.tracking.LivePartialTrack`.

The important negative case has its own test: a plain ``(fraction, message)``
callback — what the CLI and every pre-existing test pass — must keep working and
simply publish nothing, because the publishing API is duck-typed on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.types import PointTrack
from stable.server.jobs import Job, JobProgress
from stable.server.tracking import LivePartialTrack, run_point_tracking

from ._helpers import upload_clip, wait_job


def _run(n_frames: int, covered: range, *, start: float = 0.0) -> PointTrack:
    """A run that found positions only over ``covered`` (NaN elsewhere)."""
    points = np.full((n_frames, 2), np.nan, dtype=np.float32)
    for frame in covered:
        points[frame] = (start + frame, start + 2.0 * frame)
    confidence = np.zeros(n_frames, dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)
    for frame in covered:
        confidence[frame] = 1.0
        visible[frame] = True
    return PointTrack(points=points, confidence=confidence, visible=visible)


@pytest.fixture
def progress() -> JobProgress:
    """A real :class:`JobProgress` over a throwaway job, so `partial` is observable."""
    return JobProgress(Job(id="j1", project_id="p1", kind="track"))


def test_a_partial_path_is_published_as_each_run_lands(progress):
    live = LivePartialTrack(10, progress, total_runs=2)
    live.add_run(_run(10, range(5, 10)))

    partial = progress._job.partial
    assert partial is not None
    assert partial["runs_done"] == 1
    assert partial["runs_total"] == 2
    # Only the frames that run covered are filled in; the rest stay unknown.
    assert partial["track"]["x"][:5] == [None] * 5
    assert partial["track"]["x"][5:] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert partial["track"]["n_frames"] == 10


def test_later_runs_extend_the_path_without_erasing_earlier_ones(progress):
    live = LivePartialTrack(10, progress, total_runs=2)
    live.add_run(_run(10, range(5, 10)))
    live.add_run(_run(10, range(0, 5)))

    partial = progress._job.partial
    assert partial["runs_done"] == 2
    assert None not in partial["track"]["x"]  # the two halves now cover the clip
    assert partial["track"]["y"][9] == pytest.approx(18.0)


def test_a_plain_progress_callback_disables_publishing_entirely():
    """The CLI passes a bare function; nothing here may assume otherwise."""
    calls: list[tuple[float, str]] = []

    def callback(fraction: float, message: str = "") -> None:
        calls.append((fraction, message))

    live = LivePartialTrack(10, callback, total_runs=1)
    assert live.enabled is False
    live.add_run(_run(10, range(0, 10)))  # must not raise
    assert calls == []


def test_publishing_survives_a_run_longer_than_the_clip(progress):
    """A run trimmed against a re-pointed, shorter clip must not overflow."""
    live = LivePartialTrack(4, progress, total_runs=1)
    live.add_run(_run(8, range(0, 8)))
    assert len(progress._job.partial["track"]["x"]) == 4


def test_the_job_event_carries_the_partial_to_subscribers(progress):
    live = LivePartialTrack(4, progress, total_runs=1)
    live.add_run(_run(4, range(0, 4)))
    event = progress._job.to_event()
    assert event["partial"]["track"]["x"] == [0.0, 1.0, 2.0, 3.0]


def test_a_real_track_publishes_its_path_before_it_finishes(client, tiny_clip, workspace):
    """End to end: the partial path exists *during* the run, not only after it.

    This is the whole point of todos 4 and 16 — asserting the plumbing in
    isolation would not catch `run_point_tracking` forgetting to call it.
    """
    path, source = tiny_clip
    detail = upload_clip(client, path)
    project_id = detail["id"]
    wait_job(client, project_id, detail["proxy_job_id"])
    x, y = source.true_position(0)
    client.post(
        f"/api/projects/{project_id}/points", json={"frame": 0, "x": x, "y": y, "tracker": "lk"}
    )

    seen: list[dict] = []

    class Recording(JobProgress):
        """Snapshots every publish, since a finished job clears its `partial`."""

        def merge(self, **fields):
            super().merge(**fields)
            snapshot = dict(self._job.partial or {})
            if "track" in snapshot:
                snapshot["track"] = dict(snapshot["track"])
            seen.append(snapshot)

    recording = Recording(Job(id="j", project_id=project_id, kind="track", target="p1"))

    result = run_point_tracking(
        workspace,
        project_id,
        "p1",
        tracker="lk",
        direction="bidir",
        device=None,
        progress=recording,
    )

    assert result["n_frames"] == source.frame_count
    assert seen, "nothing was published while the track was running"
    covered = [
        sum(1 for value in snapshot["track"]["x"] if value is not None)
        for snapshot in seen
        if "track" in snapshot
    ]
    assert covered, "no partial path was published"
    # Coverage only ever grows, and the last publish covers the whole clip.
    assert covered == sorted(covered)
    assert covered[-1] == source.frame_count


def test_a_finished_job_clears_its_partial(monkeypatch):
    """`partial` means "in flight"; a completed job's answer is `result`."""
    from stable.server.jobs import reset_job_manager

    manager = reset_job_manager(1)

    def body(progress):
        progress.merge(frame=3)
        assert progress._job.partial == {"frame": 3}
        return {"ok": True}

    job = manager.submit("p1", "track", body)
    for _ in range(200):
        if job.status in ("done", "error"):
            break
        import time

        time.sleep(0.02)
    assert job.status == "done", job.error
    assert job.result == {"ok": True}
    assert job.partial is None
