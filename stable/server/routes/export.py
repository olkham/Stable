"""Export dialog backend: data (CSV/JSON) and stabilized video.

Data export reads the same corrected tracks and solved/smoothed curves the rest
of the server produces. Video export plans the render with
:class:`stable.render.StabilizationPlan` — real border policies, exact minimal
auto-zoom — and streams it through :func:`stable.render.render_stabilized`,
which hands frames to Phase 1's :func:`stable.video.encode.export_video` with
the source clip's audio passed through untouched.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...core.errors import ProjectError, RenderError
from ...render.stabilize import render_stabilized
from .. import motion as motion_glue
from ..jobs import JobManager, get_job_manager
from ..schemas import ExportVideoRequest
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track
from . import solve as solve_routes

__all__ = ["router"]

router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


# -- data export ---------------------------------------------------------------


@router.get("/tracks")
def export_tracks(
    project_id: str, format: str = "json", workspace: Workspace = Depends(_workspace)
) -> Response:
    """Export every point's (corrected) track as CSV or JSON.

    Args:
        format: ``"json"`` or ``"csv"``.
    """
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'csv'")
    project = workspace.load_project(project_id)

    rows: list[dict[str, Any]] = []
    for point in project.points:
        track = load_display_track(project, point)
        if track is None:
            continue
        for frame in range(track.n_frames):
            x, y = track.points[frame]
            rows.append(
                {
                    "point_id": point.id,
                    "frame": frame,
                    "x": None if math.isnan(x) else float(x),
                    "y": None if math.isnan(y) else float(y),
                    "confidence": float(track.confidence[frame]),
                    "visible": bool(track.visible[frame]),
                }
            )

    if format == "json":
        body = json.dumps({"points": rows}, indent=2)
        return Response(content=body, media_type="application/json")

    buffer = io.StringIO()
    fieldnames = ["point_id", "frame", "x", "y", "confidence", "visible"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tracks.csv"},
    )


@router.get("/curves")
def export_curves(
    project_id: str, format: str = "json", workspace: Workspace = Depends(_workspace)
) -> Response:
    """Export the solved (and smoothed, if available) X/Y/R/S curves as CSV or JSON.

    Raises:
        HTTPException: 400 if the project has not been solved yet.
    """
    if format not in ("json", "csv"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'csv'")
    solve = solve_routes.load_solve(workspace, project_id)
    if solve is None:
        raise HTTPException(status_code=400, detail="run solve before exporting curves")
    smoothed = solve_routes.load_smoothed(workspace, project_id)

    if format == "json":
        body = json.dumps(
            {
                "raw": motion_glue.solve_payload(solve),
                "smoothed": (
                    motion_glue.smoothed_payload(smoothed) if smoothed is not None else None
                ),
            },
            indent=2,
        )
        return Response(content=body, media_type="application/json")

    # The CSV is the library's own: raw and smoothed side by side with the
    # solve's per-frame quality record, so a spreadsheet shows the same numbers
    # `stable solve` writes from the CLI.
    text = smoothed.to_csv() if smoothed is not None else solve.to_csv()
    return Response(
        content=text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=curves.csv"},
    )


# -- video export ----------------------------------------------------------------


@router.post("/video")
def export_video_route(
    project_id: str,
    body: ExportVideoRequest,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Submit a background job that renders the stabilized MP4.

    The whole render is planned before a pixel is read (see
    :class:`stable.render.StabilizationPlan`), because the minimal auto-zoom is
    only knowable once every frame's warp is — then frames are streamed to the
    encoder one at a time, so clip length is bounded by disk, not memory.

    Raises:
        HTTPException: 400 if the project has not been solved, or if its border
            policy is ``transparent`` (no codec here carries an alpha channel).
    """
    project = workspace.load_project(project_id)
    if solve_routes.load_smoothed(workspace, project_id, project) is None:
        raise HTTPException(
            status_code=400, detail="run solve and smoothing before exporting a stabilized video"
        )
    if project.render.border == "transparent":
        raise HTTPException(
            status_code=400,
            detail=(
                "the 'transparent' border produces RGBA frames and no codec here carries an "
                "alpha channel; switch the border policy to 'crop' or 'auto_zoom' to export video"
            ),
        )
    filename = body.filename if body.filename.endswith(".mp4") else f"{body.filename}.mp4"

    def _run(progress: Any) -> dict[str, Any]:
        proj = workspace.load_project(project_id)
        video = workspace.get_video_source(project_id)
        motion = solve_routes.load_smoothed(workspace, project_id, proj)
        if motion is None:  # pragma: no cover - re-checked at submit
            raise ProjectError("solve/smoothing cache missing")

        plan = motion_glue.build_plan(proj, motion)
        out_dir = workspace.renders_dir(project_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        render_stabilized(
            video,
            plan,
            out_path,
            codec=body.codec,
            crf=body.crf,
            audio_source=proj.resolve_path(proj.video.path),
            frame_range=(0, min(plan.n_frames, video.frame_count)),
            progress=progress,
        )
        return {
            "filename": filename,
            "border": plan.border,
            "zoom": plan.geometry.zoom,
            "width": plan.geometry.width,
            "height": plan.geometry.height,
        }

    try:
        job = jobs.submit(project_id, "export", _run)
    except RenderError as exc:  # pragma: no cover - planning happens inside the job
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.id}


@router.get("/download/{filename}")
def download_export(
    project_id: str, filename: str, workspace: Workspace = Depends(_workspace)
) -> FileResponse:
    """Download a previously rendered export from ``renders/``.

    Raises:
        HTTPException: 404 if the filename is unknown, or 400 if it tries to
            escape the renders directory.
    """
    if "/" in filename or "\\" in filename or filename in ("..", "."):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = workspace.renders_dir(project_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no export named {filename!r}")
    return FileResponse(path, media_type="video/mp4", filename=filename)
