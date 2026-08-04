"""Plain helper functions shared by the Phase 4 server tests.

Deliberately *not* named ``conftest.py``: pytest's rootless import mode
inserts each test directory onto ``sys.path`` under its bare filename, so a
second module literally named ``conftest`` here would shadow (or be shadowed
by) ``tests/conftest.py`` the moment any test does ``from conftest import
...`` — whichever one python's module cache loads first wins for every later
import in the process. Fixtures still live in ``tests/server/conftest.py``
(pytest needs that exact filename to auto-discover them); these are just
functions, so they can live anywhere and be imported by name.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from stable.core.project import PointSpec, Project
from stable.core.types import PointTrack, Query
from stable.server.storage import Workspace


def wait_job(
    client: TestClient, project_id: str, job_id: str, *, timeout: float = 30.0
) -> dict[str, Any]:
    """Poll a job's REST status until it finishes (or raise on timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/projects/{project_id}/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def upload_clip(
    client: TestClient, clip_path: str, *, name: str = "Test Project"
) -> dict[str, Any]:
    """Create a project from a clip file and return the created project's detail dict."""
    with open(clip_path, "rb") as handle:
        response = client.post(
            "/api/projects",
            files={"file": ("clip.mp4", handle, "video/mp4")},
            data={"name": name},
        )
    assert response.status_code == 200, response.text
    return response.json()


def seed_tracks(
    workspace: Workspace,
    project_id: str,
    tracks: Sequence[PointTrack],
    *,
    tracker: str = "lk",
) -> list[str]:
    """Write ready-made tracks into a project, as if they had been tracked.

    Server tests about *solving*, *editing* and *rendering* should not depend on
    how well a tracker happens to do on a synthetic clip — seeding known tracks
    makes their assertions about the code under test instead. The tracks are
    trimmed to the decoded clip's real frame count, since an encoder is free to
    round the length.

    Returns:
        The ids of the points created, in order.
    """
    project = workspace.load_project(project_id)
    n_frames = project.video.frames
    tracks_dir = workspace.tracks_dir(project_id)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    for track in tracks:
        trimmed = PointTrack(
            points=track.points[:n_frames],
            confidence=track.confidence[:n_frames],
            visible=track.visible[:n_frames],
        )
        point_id = project.new_point_id()
        npz_path = tracks_dir / f"{point_id}.npz"
        trimmed.save_npz(npz_path)
        first = int(np.nonzero(trimmed.valid)[0][0])
        x, y = trimmed.points[first]
        project.points.append(
            PointSpec(
                id=point_id,
                queries=[Query(frame=first, x=float(x), y=float(y))],
                tracker=tracker,
                raw=project.relative_path(str(npz_path)),
            )
        )
        ids.append(point_id)
    project.validate()
    workspace.save_project(project_id, project)
    return ids


def load_project(workspace: Workspace, project_id: str) -> Project:
    """Read a project straight off disk, bypassing the HTTP layer."""
    return workspace.load_project(project_id)


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode a JPEG response body into an RGB array."""
    import cv2

    array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert array is not None, "response body was not decodable as JPEG"
    return np.asarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
