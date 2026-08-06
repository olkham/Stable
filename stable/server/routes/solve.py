"""Solve settings + job, smoothing preview, and the stabilized-frame preview.

The curve math is the real Phase 3 stack — :mod:`stable.solver`,
:mod:`stable.smoothing`, :mod:`stable.render` — reached through
:mod:`stable.server.motion`, which owns the on-disk cache and the JSON shapes.
This module is only the HTTP surface: settings persistence, job submission and
serving one warped frame.

It also owns the stabilisation *options* — target, crop and border fill (todo
17). They are not part of the locked v1 project schema, so they persist in a
sidecar next to the solve and smoothing caches (see
:data:`stable.server.motion.RENDER_OPTIONS_CACHE`); the request models for them
live here rather than in ``schemas.py`` for the same reason the correction
models do: one definition of what a valid option is, in
:mod:`stable.render.options`, instead of a Pydantic mirror that can drift.

Note:
    Solved curves are pivot-relative, so a stabilisation warp is never rebuilt
    from raw channel values here. See :mod:`stable.server.motion`.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ...core.errors import StableError
from ...core.project import Project, SmoothingSpec
from ...render.fill import FILL_MODES
from ...render.options import CROP_CENTRES, TARGET_MODES, RenderOptions
from .. import motion as motion_glue
from ..frames import (
    FrameUnavailable,
    load_options,
    load_smoothed,
    load_solve,
    output_frame_jpeg,
    preview_plan,
    smoothing_cache_path,
    solve_cache_path,
)
from ..jobs import JobManager, get_job_manager
from ..schemas import SmoothingUpdate, SolveSettingsUpdate
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track

__all__ = [
    "router",
    "load_solve",
    "load_smoothed",
    "recompute_smoothed",
    "load_options",
    "preview_plan",
]

router = APIRouter(prefix="/projects/{project_id}", tags=["solve"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


def recompute_smoothed(
    workspace: Workspace, project_id: str, project: Project
) -> dict[str, Any] | None:
    """Recompute and cache the smoothed curves from the cached raw solve, if any."""
    solve = load_solve(workspace, project_id)
    if solve is None:
        return None
    smoothed = motion_glue.smooth_solve(project, solve)
    payload = motion_glue.smoothed_payload(smoothed)
    motion_glue.write_json(smoothing_cache_path(workspace, project_id), payload)
    motion_glue.clear_plan_cache(project_id)
    return payload


# -- solve settings + job -----------------------------------------------------


@router.put("/solve/settings")
def update_solve_settings(
    project_id: str, body: SolveSettingsUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Update the reference frame and/or enabled channels."""

    def _mutate(project: Project) -> None:
        if body.reference_frame is not None:
            project.solve.reference_frame = body.reference_frame
        if body.channels is not None:
            project.solve.channels = {**project.solve.channels, **body.channels}

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return project.solve.to_dict()


@router.get("/solve")
def get_solve(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return the current solve settings plus the cached raw curves, if solved."""
    project = workspace.load_project(project_id)
    solve = load_solve(workspace, project_id)
    return {
        "settings": project.solve.to_dict(),
        "curves": motion_glue.solve_payload(solve) if solve is not None else None,
    }


@router.post("/solve")
def run_solve(
    project_id: str,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Submit a solve job over every tracked, solve-enabled point.

    The job runs :func:`stable.solver.solve_motion` — a confidence-weighted
    Umeyama fit with RANSAC, degrading gracefully frame by frame — over each
    point's *corrected* track, so manual edits feed straight into the solve.
    """
    project = workspace.load_project(project_id)
    candidates = [p for p in project.points if p.enabled_for_solve and p.raw]
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="no tracked, solve-enabled points — track at least one point first",
        )

    def _run(progress: Any) -> dict[str, Any]:
        proj = workspace.load_project(project_id)
        points = [p for p in proj.points if p.enabled_for_solve and p.raw]
        tracks = []
        for index, point in enumerate(points):
            track = load_display_track(proj, point)
            if track is not None:
                tracks.append(track)
            progress(
                0.5 * (index + 1) / max(len(points), 1),
                f"gathering point {index + 1}/{len(points)}",
            )

        def fit_progress(fraction: float, message: str = "") -> None:
            progress(0.5 + 0.5 * fraction, message)

        solve = motion_glue.solve_tracks(proj, tracks, progress=fit_progress)
        motion_glue.write_json(
            solve_cache_path(workspace, project_id), motion_glue.solve_to_cache(solve)
        )
        recompute_smoothed(workspace, project_id, proj)
        progress(1.0, "done")
        return {"n_points": len(tracks), "n_frames": solve.n_frames, **solve.summary()}

    job = jobs.submit(project_id, "solve", _run)
    return {"job_id": job.id}


# -- smoothing settings + preview ---------------------------------------------


@router.get("/smoothing")
def get_smoothing(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return the current smoothing filter list plus cached raw/smoothed curves."""
    project = workspace.load_project(project_id)
    solve = load_solve(workspace, project_id)
    smoothed = motion_glue.read_json(smoothing_cache_path(workspace, project_id))
    return {
        "filters": [s.to_dict() for s in project.smoothing],
        "curves": {
            "raw": motion_glue.solve_payload(solve) if solve is not None else None,
            "smoothed": smoothed,
        },
    }


@router.put("/smoothing")
def update_smoothing(
    project_id: str, body: SmoothingUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Replace the smoothing filter list and recompute the preview curves.

    The list may carry curve-editor directives alongside filters (see
    :data:`stable.smoothing.channels.DIRECTIVES`) — that is how a sub-range and
    the pinned keyframes round-trip through the locked v1 project schema.

    Recomputation is synchronous (not a job): it operates on small, already-
    cached ``[T]`` arrays, so it is fast enough for the smoothing panel's
    live raw-vs-smoothed preview to feel immediate as the user adjusts a slider.
    """
    previous = workspace.load_project(project_id).smoothing

    def _mutate(project: Project) -> None:
        project.smoothing = [
            SmoothingSpec(channel=f.channel, filter=f.filter, params=f.params, enabled=f.enabled)
            for f in body.filters
        ]

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        smoothed = recompute_smoothed(workspace, project_id, project)
    except StableError as exc:
        # A bad filter name or parameter must not leave the project holding a
        # configuration that can never be smoothed again — put the old one back.
        def _restore(proj: Project) -> None:
            proj.smoothing = previous

        workspace.update_project(project_id, _restore)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    solve = load_solve(workspace, project_id)
    return {
        "filters": [s.to_dict() for s in project.smoothing],
        "curves": {
            "raw": motion_glue.solve_payload(solve) if solve is not None else None,
            "smoothed": smoothed,
        },
    }


@router.get("/smoothing/tracks")
def get_smoothed_tracks(
    project_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Return every tracked point's path after the smoothing's stabilisation warp.

    The smoothing produces four *camera* curves, not a path per point, so
    "what did the smoothing do to this track?" has no answer until the warp
    ``smoothed(t) . raw(t)^-1`` is applied to the track itself. That is what this
    returns: where each feature ends up once the shot is stabilised, which is
    what the canvas draws over the source trail (todo 3, fourth round).

    Deliberately *not* the render's ``output_transform``: that folds in the
    export's stabilisation target and crop fit, so the trail would move whenever
    the crop moved. The export preview owns that; this owns the smoothing.

    Returns:
        ``n_frames`` and one ``{point_id, x, y}`` row per tracked point, with
        ``null`` on frames the point has no position for.

    Raises:
        HTTPException: 409 if the project has not been solved and smoothed yet —
            there is no warp to apply, which is a state rather than a failure.
    """
    project = workspace.load_project(project_id)
    motion = load_smoothed(workspace, project_id, project)
    if motion is None:
        raise HTTPException(
            status_code=409, detail="solve and smooth this project to see the smoothed tracks"
        )
    warps = motion.stabilization_transforms()
    rows: list[dict[str, Any]] = []
    for point in project.points:
        track = load_display_track(project, point)
        if track is None:
            continue
        positions = np.asarray(track.points, dtype=np.float64)
        count = min(len(warps), positions.shape[0])
        xs: list[float | None] = [None] * positions.shape[0]
        ys: list[float | None] = [None] * positions.shape[0]
        for frame in range(count):
            source = positions[frame]
            if not np.isfinite(source).all():
                continue
            moved = warps[frame].apply(source)
            xs[frame] = float(moved[0])
            ys[frame] = float(moved[1])
        rows.append({"point_id": point.id, "x": xs, "y": ys})
    return {"n_frames": motion.n_frames, "points": rows}


# -- stabilisation options (target / crop / fill) -------------------------------


class TargetPairIn(BaseModel):
    """One point-to-destination pair of a ``map`` stabilisation target."""

    point_id: str
    x: float
    y: float


class TargetUpdate(BaseModel):
    """The ``target`` block of ``PUT /api/projects/{id}/render/options``."""

    mode: str | None = None
    point_id: str | None = None
    anchor_frame: int | None = None
    # Sent whole rather than merged key-by-key: a pair list is a set of
    # positions the operator has arranged on the canvas, and merging halves of
    # two arrangements would produce one nobody asked for.
    pairs: list[TargetPairIn] | None = None


class CropUpdate(BaseModel):
    """The ``crop`` block of ``PUT /api/projects/{id}/render/options``."""

    scale: float | None = None
    centre: str | None = None


class FillUpdate(BaseModel):
    """The ``fill`` block of ``PUT /api/projects/{id}/render/options``."""

    mode: str | None = None
    color: list[int] | None = None
    blur: float | None = None
    tint: float | None = None
    feather: float | None = None


class RenderOptionsUpdate(BaseModel):
    """Body for ``PUT /api/projects/{id}/render/options``.

    Every block is optional and replaces only the keys it names, so the export
    dialog can change one control without having to send back a whole
    configuration it might be holding a stale copy of.
    """

    target: TargetUpdate | None = None
    crop: CropUpdate | None = None
    fill: FillUpdate | None = None


def _options_payload(workspace: Workspace, project_id: str, project: Project) -> dict[str, Any]:
    """Return the options plus everything the UI needs to offer valid choices."""
    return {
        "options": load_options(workspace, project_id).to_dict(),
        "target_modes": list(TARGET_MODES),
        "crop_centres": list(CROP_CENTRES),
        "fill_modes": list(FILL_MODES),
        "points": motion_glue.target_choices(project),
    }


@router.get("/render/options")
def get_render_options(
    project_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Return the stabilisation target, crop and border fill, plus the valid choices."""
    project = workspace.load_project(project_id)
    return _options_payload(workspace, project_id, project)


_OPTIONS_LOCK = threading.Lock()
"""Serialises the read-merge-write below.

Each request loads the whole sidecar, replaces the block it was given and writes
all three back, so two overlapping updates would each carry the other's stale
copy of a block neither touched — "I moved the crop slider and my placed targets
disappeared" (todo 7, fourth round). One lock for every project: these are
sub-millisecond file writes, and a per-project map would be more bookkeeping
than the contention it saves.
"""


@router.put("/render/options")
def update_render_options(
    project_id: str, body: RenderOptionsUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Update the stabilisation target, crop scale and/or border fill.

    Only the blocks named in the body change: adjusting the crop leaves the
    placed stabilisation targets exactly as they were, and vice versa.

    These drive the stabilized preview as well as the export, so the memoised
    render plan is dropped here — the same rule the border policy follows.

    Raises:
        HTTPException: 400 if a mode, colour or number is not a valid choice
            (validated by :mod:`stable.render.options`, not re-implemented here).
    """
    project = workspace.load_project(project_id)
    with _OPTIONS_LOCK:
        current = load_options(workspace, project_id).to_dict()
        for block, update in (("target", body.target), ("crop", body.crop), ("fill", body.fill)):
            if update is None:
                continue
            current[block] = {
                **current[block],
                **{k: v for k, v in update.model_dump(exclude_unset=True).items()},
            }
        try:
            options = RenderOptions.from_dict(current)
            # Fail now, with a message naming the target, rather than inside the
            # export job twenty seconds later.
            if motion_glue.resolve_mapped_targets(project, options.target) is None:
                motion_glue.resolve_target_positions(project, options.target)
        except StableError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        motion_glue.save_render_options(workspace.project_dir(project_id), options)
        motion_glue.clear_plan_cache(project_id)
    return _options_payload(workspace, project_id, project)


@router.get("/frames/{index}/stabilized")
def get_stabilized_frame(
    project_id: str, index: int, tier: str = "half", workspace: Workspace = Depends(_workspace)
) -> Response:
    """Return one proxy frame warped by the solved+smoothed stabilization transform.

    The warp is the render's own ``output_transform`` — stabilisation (including
    the stabilisation target's lock) plus the project's border policy and crop —
    re-expressed for the proxy's resolution by
    :func:`stable.render.warp.rescale_transform`, so what the preview shows is
    what the export will produce, border fill and all.

    The pixels themselves come from :func:`stable.server.frames.output_frame_jpeg`,
    which is also what the frame WebSocket serves — one implementation, so the
    streamed picture and the one this route answers with cannot drift apart.

    Raises:
        HTTPException: 409 if the project has not been solved yet; 404 if the
            frame index or proxy tier is invalid.
    """
    try:
        payload, _revision = output_frame_jpeg(workspace, project_id, index, tier)
    except FrameUnavailable as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return Response(content=payload, media_type="image/jpeg")
