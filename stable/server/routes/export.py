"""Export dialog backend: data (CSV/JSON) and stabilized video.

Data export reads the same corrected tracks and solved/smoothed curves the rest
of the server produces. Video export plans the render with
:class:`stable.render.StabilizationPlan` — real border policies, exact minimal
auto-zoom, the operator's stabilisation target / crop / border fill — and
streams it through :func:`stable.render.render_stabilized`, which hands frames
to Phase 1's :func:`stable.video.encode.export_video` with the source clip's
audio passed through untouched.

Refusing a render that would do nothing
---------------------------------------
The stabilisation warp is ``smoothed(t) . raw(t)^-1``. A project with no
smoothing filter enabled and no stabilisation target therefore has a warp of
exactly the identity on every frame, and the export used to succeed — writing a
re-encode of the source, reporting ``zoom: 1.0``, and looking to the operator
exactly like "generating the stabilised output video doesn't work". Unlike
``stable stabilize``, which applies a default Butterworth when the user does not
choose one, the server has no default to fall back on. So this route checks the
plan before it queues one (:attr:`~stable.render.StabilizationPlan.is_no_op`)
and answers with what to change instead of spending minutes producing a copy.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...core.errors import ProjectError, RenderError, StableError
from ...core.project import Project
from ...core.trackdata import AE_FORMAT, after_effects_keyframes
from ...render.stabilize import render_stabilized
from ...video.encode import AudioPlan, plan_audio
from .. import motion as motion_glue
from ..jobs import JobManager, get_job_manager
from ..schemas import ExportVideoRequest
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track
from . import solve as solve_routes

__all__ = ["router", "TRACK_FORMATS"]

NO_OP_DETAIL = (
    "this render would not move the picture at all, so it would just re-encode the source: "
    "the stabilisation warp is 'smoothed motion' minus 'solved motion', and nothing has "
    "asked them to differ. Enable at least one smoothing filter in the Smoothing panel "
    "(Butterworth at ~1 Hz is the usual starting point), or set a stabilisation target so a "
    "tracked point is held still."
)
"""Why an all-identity export is refused, and the two ways to fix it."""

router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


# -- data export ---------------------------------------------------------------


TRACK_FORMATS = ("json", "csv", AE_FORMAT)
"""Formats ``GET /export/tracks`` can produce.

``json``/``csv`` are the numbers; ``ae`` is After Effects keyframe data, which
is the half of todo 18 that "use in other applications" actually needs — see
:mod:`stable.core.trackdata`.
"""


@router.get("/tracks")
def export_tracks(
    project_id: str, format: str = "json", workspace: Workspace = Depends(_workspace)
) -> Response:
    """Export every point's (corrected) track as CSV, JSON or AE keyframe data.

    Args:
        format: One of :data:`TRACK_FORMATS`.
    """
    if format not in TRACK_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"format must be one of {', '.join(TRACK_FORMATS)}"
        )
    project = workspace.load_project(project_id)

    if format == AE_FORMAT:
        return _export_after_effects(project)

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


def _export_after_effects(project: Project) -> Response:
    """Render every tracked point as After Effects keyframe data.

    Raises:
        HTTPException: 400 if no point has been tracked yet — an empty keyframe
            document imports as nothing at all, which looks like a broken
            export rather than an empty project.
    """
    tracks: list[tuple[str, list[tuple[float, float]]]] = []
    for point in project.points:
        track = load_display_track(project, point)
        if track is None:
            continue
        tracks.append((point.id, [(float(x), float(y)) for x, y in track.points]))
    if not tracks:
        raise HTTPException(status_code=400, detail="no point has been tracked yet")

    body = after_effects_keyframes(
        tracks,
        fps=project.video.fps,
        width=project.video.w,
        height=project.video.h,
    )
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=tracks-ae.txt"},
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
    encoder one at a time, so clip length is bounded by disk, not memory. It is
    planned *here*, at submit time, rather than only inside the job: planning is
    the cheap part (no pixels), and it is what tells us whether the render is
    worth running at all.

    Raises:
        HTTPException: 400 if the project has not been solved, if its border
            policy is ``transparent`` (no codec here carries an alpha channel),
            if the stabilisation target cannot be resolved, or if the render
            would be a no-op (see :data:`NO_OP_DETAIL`).
    """
    project = workspace.load_project(project_id)
    motion = solve_routes.load_smoothed(workspace, project_id, project)
    if motion is None:
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
    options = solve_routes.load_options(workspace, project_id)
    try:
        planned = motion_glue.build_plan(project, motion, options=options)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if planned.is_no_op:
        raise HTTPException(status_code=400, detail=NO_OP_DETAIL)
    filename = body.filename if body.filename.endswith(".mp4") else f"{body.filename}.mp4"

    def _run(progress: Any) -> dict[str, Any]:
        proj = workspace.load_project(project_id)
        video = workspace.get_video_source(project_id)
        smoothed = solve_routes.load_smoothed(workspace, project_id, proj)
        if smoothed is None:  # pragma: no cover - re-checked at submit
            raise ProjectError("solve/smoothing cache missing")

        plan = motion_glue.build_plan(
            proj, smoothed, options=solve_routes.load_options(workspace, project_id)
        )
        out_dir = workspace.renders_dir(project_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        source_clip = proj.resolve_path(proj.video.path)
        # Asked here as well as inside the encoder so the answer can be reported:
        # a clip whose audio had to be converted should say so rather than leave
        # the operator to notice it in the file (see stable/video/encode.py).
        audio = plan_audio(source_clip, out_path) if source_clip else AudioPlan("none")
        render_stabilized(
            video,
            plan,
            out_path,
            codec=body.codec,
            crf=body.crf,
            audio_source=source_clip,
            frame_range=(0, min(plan.n_frames, video.frame_count)),
            progress=progress,
        )
        return {
            "filename": filename,
            "border": plan.border,
            "zoom": plan.geometry.zoom,
            "width": plan.geometry.width,
            "height": plan.geometry.height,
            "crop_scale": plan.geometry.crop_scale,
            "coverage": plan.geometry.coverage,
            "gaps": plan.geometry.gaps,
            "target": plan_target(options),
            "fill": None if plan.fill is None else plan.fill.mode,
            "max_displacement": plan.max_displacement,
            "audio": audio.mode,
            "audio_note": audio.reason,
        }

    try:
        job = jobs.submit(project_id, "export", _run)
    except RenderError as exc:  # pragma: no cover - planning happens inside the job
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.id}


def plan_target(options: Any) -> str:
    """Return a short human label for what the render held still."""
    target = options.target
    if target.mode == "point":
        return f"point {target.point_id}"
    return {"frame": "smoothing only", "centroid": "all points"}.get(target.mode, target.mode)


@router.get("/renders")
def list_renders(
    project_id: str, workspace: Workspace = Depends(_workspace)
) -> list[dict[str, Any]]:
    """List the stabilized videos already rendered for this project, newest first.

    Without this the only route to a finished export is the download link the
    export dialog holds in memory — close the dialog or reload the page and a
    perfectly good render becomes unreachable.
    """
    directory = workspace.renders_dir(project_id)
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"filename": p.name, "bytes": p.stat().st_size, "modified_at": p.stat().st_mtime}
        for p in files
    ]


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
