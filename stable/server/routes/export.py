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
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ...core.errors import RenderError, StableError
from ...core.project import Project, SmoothingSpec
from ...core.trackdata import AE_FORMAT, after_effects_keyframes
from ...render.options import RenderOptions
from ...render.stabilize import render_stabilized
from ...video.encode import AudioPlan, plan_audio
from .. import motion as motion_glue
from .. import render_queue
from ..clip_range import effective_range, export_window
from ..frames import frame_geometry
from ..jobs import JobManager, get_job_manager
from ..schemas import ExportVideoRequest, RenderQueueOrder
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track
from . import solve as solve_routes

__all__ = ["router", "TRACK_FORMATS"]

NO_OP_DETAIL = (
    "this render would not move the picture at all, so it would just re-encode the source. "
    "There are no target points, so the render falls back to undoing what the Smoothing panel "
    "removed — and no filter is enabled there, so there is nothing to undo. Add a target point "
    "(hold a track somewhere in the delivered frame), or enable a smoothing filter "
    "(Butterworth at ~1 Hz is the usual starting point)."
)
"""Why an all-identity export is refused, and the two ways to fix it.

Only a project with no targets can reach it: a target pins a path to a mark on
every frame, which moves the picture unless the shot was already perfectly still.
"""

router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])

RENDER_PREFIX = "stabilized"
"""Stem every auto-named render starts with."""


def unique_render_name(
    directory: Path, requested: str | None, *, taken: Collection[str] = ()
) -> str:
    """Return a render filename that nothing else has claimed.

    Args:
        directory: The project's ``renders/`` directory (need not exist yet).
        requested: What the operator asked for, or ``None`` for an automatic
            ``stabilized-YYYYmmdd-HHMMSS.mp4``.
        taken: Names claimed but not yet written — the queued renders (todo 6,
            tenth round). A queued entry has no file yet, so the existence check
            below cannot see it, and three renders queued in one second with no
            name would otherwise all be handed the same timestamped one.

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
    claimed = set(taken)
    candidate = f"{stem}.mp4"
    index = 2
    while (directory / candidate).exists() or candidate in claimed:
        candidate = f"{stem}-{index}.mp4"
        index += 1
    return candidate


def _render_range(
    workspace: Workspace, project_id: str, total: int, options: Any
) -> tuple[int, int]:
    """Return the frames an export should write.

    The export's own range where it states one, the In/Out markers otherwise —
    see :func:`stable.server.clip_range.export_window`.
    """
    return export_window(workspace.project_dir(project_id), total, options.export_range)


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


def _planned_export(
    workspace: Workspace,
    project_id: str,
    *,
    project: Project | None = None,
    border: str | None = None,
    options: Any = None,
    smoothing: list[SmoothingSpec] | None = None,
) -> tuple[Project, Any, Any, tuple[int, int]]:
    """Plan a render and refuse the ones that cannot or should not be made.

    Shared by the immediate export and the queue (todo 6, tenth round), so both
    refuse the same things for the same reasons and neither can drift into
    accepting a render the other would not. The queue calls it twice: once when
    an entry is *added*, so a render that would do nothing is refused while the
    operator is still looking at the settings that caused it, and once when the
    entry runs, from its own stored snapshot.

    Args:
        project: The project, when the caller already has it.
        border: The border policy to plan for; ``None`` uses the project's.
        options: The render options to plan with; ``None`` loads the stored ones.
        smoothing: Smoothing filters to smooth with; ``None`` uses the project's
            own. This is what lets a queued entry render the filters it was
            queued with rather than whatever is saved by the time it runs.

    Returns:
        ``(project, options, plan, frame_range)``.

    Raises:
        HTTPException: 400 if the project has not been solved and smoothed, if
            the border policy is ``transparent`` (no codec here carries an alpha
            channel), if the options cannot be planned, or if the render would
            be a no-op (see :data:`NO_OP_DETAIL`).
    """
    project = project or workspace.load_project(project_id)
    if smoothing is not None:
        # A *copy* of the project, not the project: this is a planning input, and
        # writing the entry's filter list onto the shared object would leak one
        # queued render's settings into everything else asking about this
        # project for as long as the object lived.
        project = replace(project, smoothing=list(smoothing))
    motion = solve_routes.load_smoothed(workspace, project_id, project)
    if motion is None:
        raise HTTPException(
            status_code=400, detail="run solve and smoothing before exporting a stabilized video"
        )
    policy = border or project.render.border
    if policy == "transparent":
        raise HTTPException(
            status_code=400,
            detail=(
                "the 'transparent' border produces RGBA frames and no codec here carries an "
                "alpha channel; switch the border policy to 'crop' or 'auto_zoom' to export video"
            ),
        )
    chosen = (
        options
        if options is not None
        else solve_routes.load_options(workspace, project_id, project)
    )
    clip = workspace.get_video_source(project_id)
    total = min(motion.n_frames, clip.frame_count)
    # Worked out before the plan, not after: a crop asked to fit the export
    # range needs to know that range in order to be planned at all.
    render_range = _render_range(workspace, project_id, total, chosen)
    try:
        plan = motion_glue.build_plan(
            project, motion, border=policy, options=chosen, export_range=render_range
        )
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if plan.is_no_op:
        raise HTTPException(status_code=400, detail=NO_OP_DETAIL)
    return project, chosen, plan, render_range


def _render_body(
    workspace: Workspace,
    project_id: str,
    *,
    filename: str,
    codec: str,
    crf: int,
    options: Any,
    plan: Any,
    render_range: tuple[int, int],
) -> Any:
    """Return the job body that writes one planned render to disk.

    The plan, the range and the encode settings are all closed over here rather
    than re-read inside the worker (todo 8, fourth round) — line up "this crop"
    and then "that crop" and each renders the one it was queued with, instead of
    both rendering whichever was saved last.
    """

    def _run(progress: Any) -> dict[str, Any]:
        started = time.monotonic()
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
            # The export range as it was when this was queued — its own, or the
            # In/Out markers (todo 1, todo 1, eighth round). The plan still covers every frame
            # — the warp for frame N is the warp for frame N whatever is written
            # out — so this narrows the *render*, not the stabilisation.
            frame_range=render_range,
            progress=progress,
        )
        elapsed = time.monotonic() - started
        # What this machine actually took, so the queue's estimates stop being a
        # seed constant after the first render (todo 6, tenth round).
        render_queue.record_render(
            workspace.project_dir(project_id),
            seconds=elapsed,
            frames=render_range[1] - render_range[0],
            width=plan.geometry.width,
            height=plan.geometry.height,
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
            "crop_measured_frames": plan.geometry.measured_frames,
            "gaps": plan.geometry.gaps,
            "target": motion_glue.target_label(options.target),
            "targets": [point.to_dict() for point in options.target.points],
            "resolution": [plan.geometry.width, plan.geometry.height],
            "fill": None if plan.fill is None else plan.fill.mode,
            "max_displacement": plan.max_displacement,
            "audio": audio.mode,
            "audio_note": audio.reason,
            "seconds": elapsed,
        }

    return _run


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

    This starts the render **now**. The UI uses ``POST /export/queue`` instead
    (todo 6, tenth round) — an export is minutes of work, and lining several up
    with different settings is worth being able to do before any of them
    starts. This route stays because an immediate render is still the right
    answer for a script or a one-off.

    Raises:
        HTTPException: 400 if the project has not been solved, if its border
            policy is ``transparent`` (no codec here carries an alpha channel),
            if the stabilisation target cannot be resolved, or if the render
            would be a no-op (see :data:`NO_OP_DETAIL`).
    """
    _project, options, plan, render_range = _planned_export(workspace, project_id)
    filename = unique_render_name(
        workspace.renders_dir(project_id),
        body.filename,
        taken=render_queue.reserved_names(workspace.project_dir(project_id)),
    )
    body_fn = _render_body(
        workspace,
        project_id,
        filename=filename,
        codec=body.codec,
        crf=body.crf,
        options=options,
        plan=plan,
        render_range=render_range,
    )
    try:
        job = jobs.submit(project_id, "export", body_fn, target=filename, exclusive=True)
    except RenderError as exc:  # pragma: no cover - planning happens inside the job
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.id, "filename": filename}


# -- the render queue (todo 6, tenth round) ---------------------------------------
#
# The list above renders at once. This is the other half of the report: queue
# several runs with different settings, look at them, reorder or drop any of
# them, and then start the lot. Everything about *what* a queued entry is lives
# in :mod:`stable.server.render_queue`; these are its five verbs.


def _queue_payload(workspace: Workspace, project_id: str) -> dict[str, Any]:
    """Return the queue with an estimate attached to every entry."""
    directory = workspace.project_dir(project_id)
    entries = render_queue.load_queue(directory)
    rows: list[dict[str, Any]] = []
    total = 0.0
    measured = True
    for entry in entries:
        seconds, from_measurement = render_queue.estimate_seconds(directory, entry)
        measured = measured and from_measurement
        if entry.is_pending:
            total += seconds
        rows.append(
            {
                **entry.to_dict(),
                "frames": entry.frames,
                "estimate_seconds": seconds,
                "estimate_measured": from_measurement,
            }
        )
    return {
        "entries": rows,
        "pending": sum(1 for entry in entries if entry.is_pending),
        # How long the *unstarted* work should take, which is the number the
        # button is really asking about.
        "estimate_seconds": total,
        "estimate_measured": bool(entries) and measured,
        "running": render_queue.runner_active(project_id),
    }


@router.get("/queue")
def get_render_queue(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return the queued renders, in order, each with how long it should take."""
    workspace.load_project(project_id)  # 404s if the project does not exist
    return _queue_payload(workspace, project_id)


@router.post("/queue")
def enqueue_render(
    project_id: str, body: ExportVideoRequest, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Capture the current settings as a queued render, **without starting it**.

    The entry carries everything the render depends on — the options, the border
    policy, the smoothing filter list, the encode settings and the frame range —
    so queueing "this crop at 1 Hz" and then "that crop at 0.5 Hz" gives two
    renders of two different things rather than two of whatever was saved last.

    The plan is built here and thrown away: it exists to answer "would this
    render anything at all?" while the operator is still looking at the settings
    that caused it, rather than minutes later at the front of a queue. It is
    rebuilt from the snapshot when the entry actually runs.

    Raises:
        HTTPException: the same 400s as ``POST /export/video`` — this refuses
            exactly what an immediate render would refuse.
    """
    project = workspace.load_project(project_id)
    _project, options, plan, render_range = _planned_export(workspace, project_id)
    entry = render_queue.QueuedRender(
        filename=unique_render_name(
            workspace.renders_dir(project_id),
            body.filename,
            taken=render_queue.reserved_names(workspace.project_dir(project_id)),
        ),
        codec=body.codec,
        crf=body.crf,
        border=project.render.border,
        options=options.to_dict(),
        smoothing=[spec.to_dict() for spec in project.smoothing],
        frame_range=render_range,
        width=plan.geometry.width,
        height=plan.geometry.height,
    )
    render_queue.append_entry(workspace.project_dir(project_id), entry)
    return _queue_payload(workspace, project_id)


@router.put("/queue/order")
def reorder_render_queue(
    project_id: str, body: RenderQueueOrder, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Put the pending entries in the order given.

    Only pending entries move: one already handed to a worker has a job and soon
    a file, and shuffling it would be as much of a lie as a reorder the runner
    ignored. Pending entries the body does not name keep their relative order,
    after the ones it does.
    """
    workspace.load_project(project_id)
    render_queue.reorder_queue(workspace.project_dir(project_id), body.ids)
    return _queue_payload(workspace, project_id)


@router.delete("/queue/{entry_id}")
def remove_queued_render(
    project_id: str, entry_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Drop one entry from the queue.

    A finished or failed entry is dropped as readily as a pending one — it is a
    record by then, and dismissing records is what keeps the list a queue. A
    *running* entry is only forgotten here; its job carries on, because there is
    no safe point to abandon an encode at.

    Raises:
        HTTPException: 404 if there is no such entry.
    """
    workspace.load_project(project_id)
    directory = workspace.project_dir(project_id)
    if all(entry.id != entry_id for entry in render_queue.load_queue(directory)):
        raise HTTPException(status_code=404, detail=f"no queued render {entry_id!r}")
    render_queue.remove_entry(directory, entry_id)
    return _queue_payload(workspace, project_id)


@router.post("/queue/run")
def run_render_queue(
    project_id: str,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, Any]:
    """Start working through the pending entries, one render at a time.

    One at a time, in order, driven by a thread of its own — see
    :func:`stable.server.render_queue.start_runner` for why that is not "submit
    them all and let the exclusive lock sort it out".

    An entry that can no longer be planned (its tracks deleted, the solve
    re-run into something unusable) is marked failed and the queue carries on:
    one bad entry must not strand the ones behind it.

    Calling this while a run is already going is not an error — the entries just
    queued are picked up by the run in progress, which is what the operator
    meant by pressing it.
    """
    workspace.load_project(project_id)

    def _submit(entry: render_queue.QueuedRender) -> str:
        project = workspace.load_project(project_id)
        try:
            _project, options, plan, render_range = _planned_export(
                workspace,
                project_id,
                project=project,
                border=entry.border,
                options=RenderOptions.from_dict(entry.options),
                smoothing=[SmoothingSpec.from_dict(spec) for spec in entry.smoothing],
            )
        except HTTPException as exc:
            # There is no response to attach a status code to out here — this
            # runs on the runner's thread, long after the request that started
            # it — so the refusal becomes the entry's error text, which is where
            # the operator will actually read it.
            raise RuntimeError(str(exc.detail)) from exc
        # The entry's own range, not a freshly resolved one: the In/Out markers
        # may well have moved since it was queued, and what was queued is what
        # gets rendered.
        window = entry.frame_range if entry.frames else render_range
        job = jobs.submit(
            project_id,
            "export",
            _render_body(
                workspace,
                project_id,
                filename=entry.filename,
                codec=entry.codec,
                crf=entry.crf,
                options=options,
                plan=plan,
                render_range=window,
            ),
            target=entry.filename,
            exclusive=True,
        )
        return job.id

    def _status_of(job_id: str) -> tuple[str, str | None]:
        job = jobs.get(job_id)
        # A job the manager has forgotten (evicted, or a server restarted under
        # a queue) is finished as far as the queue is concerned; leaving the
        # entry "running" for ever would stall everything behind it.
        if job is None:  # pragma: no cover - only after eviction
            return ("error", "the render job is no longer tracked")
        return (job.status, job.error)

    started = render_queue.start_runner(
        project_id, workspace.project_dir(project_id), _submit, _status_of
    )
    return {**_queue_payload(workspace, project_id), "started": started}


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
        measured ``coverage``, requested ``crop_scale``, how many frames the
        coverage was measured over (``crop_measured_frames``) and whether the
        fill will be visible (``gaps``); ``export_window`` and ``clip_range``,
        the frames that would be written and the In/Out markers they may differ
        from; ``max_displacement`` and ``is_no_op``, which say whether this is a
        stabilisation at all; and ``targets``, one row per mark saying where it
        lands and by how much it misses.

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
    options = solve_routes.load_options(workspace, project_id, project)
    try:
        plan = solve_routes.preview_plan(workspace, project_id, project, motion)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    geometry = plan.geometry
    window = _render_range(workspace, project_id, plan.n_frames, options)
    return {
        **frame_geometry(project, plan, options, frame, motion),
        "coverage": float(geometry.coverage),
        "crop_scale": float(geometry.crop_scale),
        "crop_measured_frames": int(geometry.measured_frames),
        "zoom": float(geometry.zoom),
        "gaps": bool(geometry.gaps),
        "border": geometry.border,
        "max_displacement": float(plan.max_displacement),
        "is_no_op": bool(plan.is_no_op),
        "no_op_detail": NO_OP_DETAIL if plan.is_no_op else None,
        "target": options.target.to_dict(),
        "target_label": motion_glue.target_label(options.target),
        "export_window": list(window),
        "clip_range": list(effective_range(workspace.project_dir(project_id), plan.n_frames)),
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
