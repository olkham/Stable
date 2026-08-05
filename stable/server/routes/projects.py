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
from ..clip_range import ClipRange, load_clip_range, save_clip_range
from ..jobs import JobManager, get_job_manager
from ..schemas import ClipRangeUpdate, ProjectRenameRequest, RenderSettingsUpdate
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

    A proxy frame only changes when the proxies are regenerated, so it is worth
    caching in the browser: scrubbing back over ground already covered should
    not re-fetch anything. ``FileResponse`` already sends an ``ETag`` and
    ``Last-Modified`` from the file itself, so the ``max-age`` below is a
    revalidation window rather than a promise — a regenerated proxy is picked up
    on the next conditional request instead of being cached forever.
    """
    try:
        store = workspace.get_proxy_store(project_id, tier)
        path = store.frame_path(index)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"}
    )


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


# -- the working range: In / Out markers (todo 1) -------------------------------


@router.get("/{project_id}/range")
def get_clip_range(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return the frames tracking and export are limited to.

    Returns:
        ``in``/``out`` (out exclusive) and ``frames``, plus ``whole_clip``
        saying whether any limit is set at all — ``null`` markers and "the
        markers happen to be the ends of the clip" are different states, and the
        timeline draws them differently.
    """
    project = workspace.load_project(project_id)
    total = int(project.video.frames)
    stored = load_clip_range(workspace.project_dir(project_id), total)
    current = stored or ClipRange(0, max(total, 1))
    return {
        "in": current.start,
        "out": current.end,
        "frames": current.frames,
        "clip_frames": total,
        "whole_clip": stored is None,
    }


@router.put("/{project_id}/range")
def set_clip_range(
    project_id: str, body: ClipRangeUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Set or clear the In/Out markers.

    Both fields are optional: sending ``{"in": 100}`` moves the in point and
    leaves the out point alone, which is what dragging one handle means. Sending
    ``{"in": null, "out": null}`` clears the range back to the whole clip.

    Note the interaction with the solve: it is measured *from*
    ``solve.reference_frame``, so a reference frame outside this range will have
    no tracked points on it and the solve refuses. That refusal already names
    the fix, and moving the reference frame here would be a surprising thing for
    a range to do, so the two stay independent and the UI warns instead.

    Raises:
        HTTPException: 400 if the result would be empty or inverted.
    """
    project = workspace.load_project(project_id)
    total = int(project.video.frames)
    directory = workspace.project_dir(project_id)
    current = load_clip_range(directory, total) or ClipRange(0, max(total, 1))

    # `by_alias` matters: the field is `in_` (a Python keyword) exposed as
    # `in`, and dumping by field name would silently drop the in point.
    payload = body.model_dump(by_alias=True, exclude_unset=True)
    if payload.get("in") is None and payload.get("out") is None and payload:
        save_clip_range(directory, None, total)
        return get_clip_range(project_id, workspace)

    start = current.start if payload.get("in") is None else int(payload["in"])
    end = current.end if payload.get("out") is None else int(payload["out"])
    try:
        save_clip_range(directory, ClipRange(start, end), total)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_clip_range(project_id, workspace)
