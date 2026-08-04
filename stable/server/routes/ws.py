"""WebSocket progress streaming for background jobs.

Two sockets share one polling loop: ``/ws/projects/{project_id}`` for
project-scoped jobs (track/solve/render/export/proxy) and ``/ws/models`` for
model-checkpoint downloads, which are process-wide rather than tied to any
project — see :data:`stable.server.jobs.MODELS_JOB_SCOPE` and
:mod:`stable.server.routes.models`. On connect each sends a snapshot of every
job currently tracked for its scope, then polls
:class:`~stable.server.jobs.JobManager` a few times a second and pushes any
job whose event payload changed. Polling (rather than a thread-to-asyncio
pub/sub bridge) is a deliberate simplification — see the module docstring in
:mod:`stable.server.jobs` for why that tradeoff is fine for a local,
single-user tool.
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs import MODELS_JOB_SCOPE, JobManager, get_job_manager

__all__ = ["router"]

router = APIRouter()

_POLL_INTERVAL = 0.15


async def _stream_jobs(websocket: WebSocket, scope: str) -> None:
    """Push job-status events for one :class:`JobManager` scope until disconnect.

    Args:
        websocket: The (not-yet-accepted) connection.
        scope: A project id, or :data:`MODELS_JOB_SCOPE` for model downloads.
    """
    await websocket.accept()
    jobs: JobManager = get_job_manager()
    last: dict[str, dict] = {}
    try:
        while True:
            for job in jobs.jobs_for_project(scope):
                event = job.to_event()
                if last.get(job.id) != event:
                    last[job.id] = event
                    await websocket.send_json({"type": "job", **event})
            await asyncio.sleep(_POLL_INTERVAL)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@router.websocket("/ws/projects/{project_id}")
async def project_progress(websocket: WebSocket, project_id: str) -> None:
    """Stream job status updates for one project until the client disconnects."""
    await _stream_jobs(websocket, project_id)


@router.websocket("/ws/models")
async def models_progress(websocket: WebSocket) -> None:
    """Stream model-checkpoint download job status until the client disconnects."""
    await _stream_jobs(websocket, MODELS_JOB_SCOPE)
