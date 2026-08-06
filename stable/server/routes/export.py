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
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...core.errors import RenderError, StableError
from ...core.project import Project
from ...core.trackdata import AE_FORMAT, after_effects_keyframes
from ...render.stabilize import render_stabilized
from ...video.encode import AudioPlan, plan_audio
from .. import motion as motion_glue
from ..clip_range import effective_range
from ..frames import frame_geometry
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

RENDER_PREFIX = "stabilized"
"""Stem every auto-named render starts with."""


def unique_render_name(directory: Path, requested: str | None) -> str:
    """Return a render filename that does not already exist in ``directory``.

    Args:
        directory: The project's ``renders/`` directory (need not exist yet).
        requested: What the operator asked for, or ``None`` for an automatic
            ``stabilized-YYYYmmdd-HHMMSS.mp4``.

    Returns:
        A ``.mp4`` filename, suffixed ``-2``, ``-3``… if that name is taken.

    An export is minutes of work and the thing you compare against the last one,
    so a name collision must not silently destroy the previous file. Timestamps
    make that rare; the suffix makes it impossible.
    """
    stem = (requested or f"{RENDER_PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}").strip()
    if stem.endswith(".mp4"):
        stem = stem[: -len(".mp4")]
    stem = stem or RENDER_PREFIX
    candidate = f"{stem}.mp4"
    index = 2
    while (directory / candidate).exists():
        candidate = f"{stem}-{index}.mp4"
        index += 1
    return candidate


def _render_range(workspace: Workspace, project_id: str, total: int) -> tuple[int, int]:
    """Return the frames an export should write: the In/Out range, or all of them."""
    return effective_range(workspace.project_dir(project_id), total)


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
) -> dict[str, Any]:
    """Queue a background job that renders the stabilized MP4.

    The whole render is planned before a pixel is read (see
    :class:`stable.render.StabilizationPlan`), because the minimal auto-zoom is
    only knowable once every frame's warp is — then frames are streamed to the
    encoder one at a time, so clip length is bounded by disk, not memory.

    Everything the render depends on is captured **here, at submit time**: the
    plan, the frame range and the encode settings (todo 8, fourth round). It
    used to be re-read inside the worker, which is fine for one export and a lie
    for a queue — line up "this crop" and then "that crop" and both would render
    whichever was last saved. What you queued is what gets rendered, and the
    O(T) plan built for the checks below is the one that runs, not a second copy
    of it.

    The job is ``exclusive``, so several exports wait for each other rather than
    running at once: two FFmpeg encodes fight over the same cores to nobody's
    benefit, and a queue that all starts together is not a queue.

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
        plan = motion_glue.build_plan(project, motion, options=options)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if plan.is_no_op:
        raise HTTPException(status_code=400, detail=NO_OP_DETAIL)
    filename = unique_render_name(workspace.renders_dir(project_id), body.filename)
    codec, crf = body.codec, body.crf
    clip = workspace.get_video_source(project_id)
    render_range = _render_range(workspace, project_id, min(plan.n_frames, clip.frame_count))

    def _run(progress: Any) -> dict[str, Any]:
        proj = workspace.load_project(project_id)
        video = workspace.get_video_source(project_id)
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
            codec=codec,
            crf=crf,
            audio_source=source_clip,
            # The In/Out markers as they were when this was queued, or the whole
            # clip when none were set (todo 1). The plan still covers every
            # frame — the warp for frame N is the warp for frame N whatever is
            # written out — so this narrows the *render*, not the stabilisation.
            frame_range=render_range,
            progress=progress,
        )
        return {
            "filename": filename,
            "frame_range": list(render_range),
            "frames": render_range[1] - render_range[0],
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
        job = jobs.submit(project_id, "export", _run, target=filename, exclusive=True)
    except RenderError as exc:  # pragma: no cover - planning happens inside the job
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.id, "filename": filename}


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
        {
            "filename": p.name,
            "bytes": p.stat().st_size,
            "modified_at": p.stat().st_mtime,
            "url": f"/api/projects/{project_id}/export/download/{p.name}",
        }
        for p in files
    ]


@router.get("/preview")
def export_preview(
    project_id: str,
    frame: int = 0,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Describe what the current export settings would produce, without rendering.

    Everything the export screen draws comes from here, and all of it is
    *planned* rather than rendered — no pixels are touched, so this is cheap
    enough to re-ask on every drag of a target.

    "Cheap" is only true because the plan is memoised. A plan is an O(T) fit
    over the whole clip (the minimal auto-zoom cannot be known otherwise), and
    this route used to build one per request while the export screen asked for
    one per frame of a drag; it now goes through
    :func:`stable.server.routes.solve.preview_plan`, which exists for exactly
    that and re-plans only when the caches or the options change.

    The per-frame half of the answer — ``rect``, ``warp`` and ``targets`` — is
    :func:`stable.server.frames.frame_geometry`, which is also what rides along
    with the pixels on the frame WebSocket. That is deliberate: the socket
    delivers a frame and the geometry drawn over it in **one message**, so they
    cannot be a frame apart on the canvas, and this route cannot describe the
    frame differently from the way the socket does.

    Everything else here is a property of the whole clip, not of ``frame``, so a
    client that is scrubbing has no reason to re-ask this at all.

    Returns:
        ``output``/``source`` sizes; ``rect``, the output rectangle mapped back
        into source pixels so the screen can draw it over the frame; the
        measured ``coverage``, requested ``crop_scale`` and whether the fill
        will be visible (``gaps``); ``max_displacement`` and ``is_no_op``, which
        say whether this is a stabilisation at all; and ``targets``, one row per
        mapped point saying where it lands and by how much it misses.

    Raises:
        HTTPException: 409 if the project has not been solved and smoothed yet
            (there is nothing to preview), 400 if the options cannot be planned.
    """
    project = workspace.load_project(project_id)
    motion = solve_routes.load_smoothed(workspace, project_id, project)
    if motion is None:
        raise HTTPException(
            status_code=409,
            detail="solve and smooth this project before previewing the export",
        )
    options = solve_routes.load_options(workspace, project_id)
    try:
        plan = solve_routes.preview_plan(workspace, project_id, project, motion)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    geometry = plan.geometry
    return {
        **frame_geometry(project, plan, options, frame),
        "coverage": float(geometry.coverage),
        "crop_scale": float(geometry.crop_scale),
        "zoom": float(geometry.zoom),
        "gaps": bool(geometry.gaps),
        "border": geometry.border,
        "max_displacement": float(plan.max_displacement),
        "is_no_op": bool(plan.is_no_op),
        "no_op_detail": NO_OP_DETAIL if plan.is_no_op else None,
        "target": options.target.to_dict(),
    }


@router.delete("/renders/{filename}")
def delete_render(
    project_id: str, filename: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, str]:
    """Delete one previously rendered export, freeing the disk space.

    Renders are minutes of encoding and hundreds of megabytes each; now that
    they are uniquely named rather than overwriting each other, they accumulate,
    so there has to be a way to remove one from the same list that offers them.

    Raises:
        HTTPException: 400 if the filename tries to escape the renders
            directory, 404 if there is no such render.
    """
    _reject_path_escape(filename)
    path = workspace.renders_dir(project_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no export named {filename!r}")
    try:
        path.unlink()
    except OSError as exc:  # pragma: no cover - a locked file on Windows only
        raise HTTPException(
            status_code=409, detail=f"could not delete {filename!r}: {exc}"
        ) from exc
    return {"status": "deleted", "filename": filename}


def _reject_path_escape(filename: str) -> None:
    """Raise 400 unless ``filename`` names a file directly inside ``renders/``."""
    if "/" in filename or "\\" in filename or filename in ("..", "."):
        raise HTTPException(status_code=400, detail="invalid filename")


@router.get("/download/{filename}")
def download_export(
    project_id: str, filename: str, workspace: Workspace = Depends(_workspace)
) -> FileResponse:
    """Download a previously rendered export from ``renders/``.

    Raises:
        HTTPException: 404 if the filename is unknown, or 400 if it tries to
            escape the renders directory.
    """
    _reject_path_escape(filename)
    path = workspace.renders_dir(project_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no export named {filename!r}")
    return FileResponse(path, media_type="video/mp4", filename=filename)
