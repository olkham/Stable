"""Project CRUD, video upload, proxy-frame serving.

A project is created by uploading a video file (multipart form) — the server
is local, but the browser cannot hand over an arbitrary filesystem path, so an
upload is the only reliable "load a video" affordance from a web UI. The bytes
are copied into the project directory (see :mod:`stable.server.storage`) and a
half-resolution proxy generation job is kicked off immediately so the viewer
has something to scrub as soon as it is ready.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ...core.errors import StableError
from ...core.project import Project
from ...video.proxy import generate_proxies
from .. import motion as motion_glue
from ..jobs import JobManager, get_job_manager
from ..schemas import ProjectRenameRequest, RenderSettingsUpdate
from ..serialize import project_detail
from ..storage import Workspace, get_workspace

__all__ = ["router"]

router = APIRouter(prefix="/projects", tags=["projects"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


def _submit_proxy_job(workspace: Workspace, jobs: JobManager, project_id: str) -> str:
    """Kick off half-res proxy generation for a freshly created project."""

    def _run(progress: Any) -> dict[str, Any]:
        video = workspace.get_video_source(project_id)
        out_dir = workspace.proxies_dir(project_id) / "half"
        manifest = generate_proxies(video, out_dir, scale=0.5, progress=progress)
        return {
            "tier": "half",
            "count": manifest.count,
            "width": manifest.width,
            "height": manifest.height,
        }

    return jobs.submit(project_id, "proxy", _run).id


@router.post("")
async def create_project(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, Any]:
    """Upload a video and create a new project from it.

    Returns:
        The new project's detail payload plus ``proxy_job_id`` so the client
        can subscribe to proxy-generation progress before frames are servable.
    """
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    try:
        meta, _project = workspace.create_project(
            video_bytes=video_bytes, original_filename=file.filename or "video.mp4", name=name
        )
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = _submit_proxy_job(workspace, jobs, meta.id)
    detail = project_detail(workspace, meta.id)
    detail["proxy_job_id"] = job_id
    return detail


@router.get("")
def list_projects(workspace: Workspace = Depends(_workspace)) -> list[dict[str, Any]]:
    """List every stored project, newest first."""
    return [m.to_dict() for m in workspace.list_projects()]


@router.get("/{project_id}")
def get_project(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return a project's full current state."""
    return project_detail(workspace, project_id)


@router.patch("/{project_id}")
def rename_project(
    project_id: str, body: ProjectRenameRequest, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Rename a project."""
    workspace.rename_project(project_id, body.name)
    return project_detail(workspace, project_id)


@router.put("/{project_id}/render")
def update_render_settings(
    project_id: str, body: RenderSettingsUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Update the border policy, codec and/or CRF used by the stabilized render.

    The border policy is a *planning* decision — it changes the stabilized
    preview as well as the export — so the memoised render plan is dropped here.

    Raises:
        HTTPException: 400 if the border policy is not one of
            :data:`stable.core.project.BORDER_POLICIES`.
    """

    def _mutate(project: Project) -> None:
        if body.border is not None:
            project.render.border = body.border
            project.render.__post_init__()
        if body.codec is not None:
            project.render.codec = body.codec
        if body.crf is not None:
            project.render.crf = body.crf

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    motion_glue.clear_plan_cache(project_id)
    return project.render.to_dict()


@router.delete("/{project_id}")
def delete_project(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, str]:
    """Delete a project and all of its derived data."""
    motion_glue.clear_plan_cache(project_id)
    workspace.delete_project(project_id)
    return {"status": "deleted"}


@router.post("/{project_id}/proxy")
def regenerate_proxy(
    project_id: str,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Re-run proxy generation (e.g. after it failed, or was never started)."""
    workspace.load_project(project_id)  # 404s if the project does not exist
    job_id = _submit_proxy_job(workspace, jobs, project_id)
    return {"job_id": job_id}


@router.get("/{project_id}/frames/{index}")
def get_frame(
    project_id: str,
    index: int,
    tier: str = "half",
    workspace: Workspace = Depends(_workspace),
) -> FileResponse:
    """Serve one proxy frame as a JPEG file.

    This is disk I/O only (see :class:`stable.video.proxy.ProxyStore`) — no
    video decode happens on this path, which is what keeps timeline scrubbing
    fast.
    """
    try:
        store = workspace.get_proxy_store(project_id, tier)
        path = store.frame_path(index)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{project_id}/jobs")
def list_jobs(project_id: str, jobs: JobManager = Depends(_jobs)) -> list[dict[str, Any]]:
    """Return every tracked job for a project.

    REST fallback for the initial paint, before the WebSocket connects.
    """
    return jobs.snapshot(project_id)


@router.get("/{project_id}/jobs/{job_id}")
def get_job(project_id: str, job_id: str, jobs: JobManager = Depends(_jobs)) -> dict[str, Any]:
    """Return one job's current status."""
    job = jobs.get(job_id)
    if job is None or job.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r} for project {project_id!r}")
    return job.to_event()
