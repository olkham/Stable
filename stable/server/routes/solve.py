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
from ...render.options import (
    TARGET_SOURCES,
    Placement,
    RenderOptions,
    StabilizationTarget,
    TargetPoint,
)
from .. import motion as motion_glue
from ..clip_range import effective_range
from ..frames import (
    FrameUnavailable,
    clear_geometry_contexts,
    export_range,
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
    "export_range",
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
    clear_geometry_contexts(project_id)
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


class PlacementIn(BaseModel):
    """Where a mark sits in the delivered frame, as a fraction of it."""

    u: float
    v: float


class TargetPointIn(BaseModel):
    """One mark on the output, and the tracked path held on it."""

    id: str | None = None
    place: PlacementIn | None = None
    point_ids: list[str] | None = None
    source: str | None = None
    weight: float | None = None


class TargetUpdate(BaseModel):
    """The ``target`` block of ``PUT /api/projects/{id}/render/options``.

    The mark list is sent whole rather than merged mark-by-mark: it is one
    arrangement the operator made on the canvas, and merging halves of two
    arrangements would produce a third nobody asked for. The single-mark
    endpoints below are the convenient way to add, move and delete one.
    """

    points: list[TargetPointIn] | None = None


class CropUpdate(BaseModel):
    """The ``crop`` block of ``PUT /api/projects/{id}/render/options``."""

    scale: float | None = None
    limit_to_range: bool | None = None


class ResolutionUpdate(BaseModel):
    """The ``resolution`` block of ``PUT /api/projects/{id}/render/options``.

    Either side may be ``null`` on purpose — that side follows the crop — so
    like the export range this is one of the few places a sent ``null`` means
    something other than "leave it alone".
    """

    width: int | None = None
    height: int | None = None


class ExportRangeUpdate(BaseModel):
    """The ``export_range`` block of ``PUT /api/projects/{id}/render/options``.

    Either end may be ``null`` on purpose — that end follows the clip's In/Out
    marker — so this is one of the few places a sent ``null`` means something
    other than "leave it alone". ``exclude_unset`` on the merge below is what
    keeps the two apart.
    """

    start: int | None = None
    end: int | None = None


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
    export_range: ExportRangeUpdate | None = None
    resolution: ResolutionUpdate | None = None


def _options_payload(workspace: Workspace, project_id: str, project: Project) -> dict[str, Any]:
    """Return the options plus everything the UI needs to offer valid choices.

    ``export_window`` is the frames a render would actually write with these
    options — the In/Out markers, or the export's own range where it states one.
    The panel shows it rather than making the operator work out which of the two
    is in force.
    """
    options = load_options(workspace, project_id, project)
    total = int(project.video.frames or 0)
    window = export_range(workspace, project_id, options, total) if total else (0, 0)
    return {
        "options": options.to_dict(),
        "target_sources": list(TARGET_SOURCES),
        "fill_modes": list(FILL_MODES),
        "points": motion_glue.target_choices(project),
        "source_size": [int(project.video.w or 0), int(project.video.h or 0)],
        "clip_frames": total,
        "clip_range": list(effective_range(workspace.project_dir(project_id), total))
        if total
        else [0, 0],
        "export_window": list(window),
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
    """Update the targets, crop, border fill, export range and/or resolution.

    Only the blocks named in the body change: adjusting the crop leaves the
    placed target points exactly as they were, and vice versa.

    These drive the stabilized preview as well as the export, so the memoised
    render plan is dropped here — the same rule the border policy follows.

    Raises:
        HTTPException: 400 if a source, colour or number is not a valid choice
            (validated by :mod:`stable.render.options`, not re-implemented here).
    """
    project = workspace.load_project(project_id)
    return _store_options(workspace, project, project_id, lambda cur: _merge_options(cur, body))


def _merge_options(current: dict[str, Any], body: RenderOptionsUpdate) -> dict[str, Any]:
    """Fold a partial update into the stored options, block by block."""
    merged = dict(current)
    for block, update in (
        ("target", body.target),
        ("crop", body.crop),
        ("fill", body.fill),
        ("export_range", body.export_range),
        ("resolution", body.resolution),
    ):
        if update is None:
            continue
        merged[block] = {
            **merged[block],
            **{k: v for k, v in update.model_dump(exclude_unset=True).items()},
        }
    return merged


def _store_options(
    workspace: Workspace, project: Project, project_id: str, mutate: Any
) -> dict[str, Any]:
    """Read, mutate, validate and write the options sidecar, under the lock.

    Every route that changes an option goes through here, so the read-merge-write
    is serialised in one place rather than once per endpoint — see
    :data:`_OPTIONS_LOCK` for why that matters.
    """
    with _OPTIONS_LOCK:
        current = load_options(workspace, project_id, project).to_dict()
        try:
            options = RenderOptions.from_dict(mutate(current))
            _check_targets(project, options)
        except StableError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        motion_glue.save_render_options(workspace.project_dir(project_id), options)
        motion_glue.clear_plan_cache(project_id)
    return _options_payload(workspace, project_id, project)


def _check_targets(project: Project, options: RenderOptions) -> None:
    """Fail now if a target names a track that cannot supply a path.

    Skipped entirely while the project has no tracked point: a mark over "every
    track" is unresolvable on a fresh project, and refusing to store a crop
    scale until something has been tracked would be a wrong answer to a right
    question. The export checks again before it queues.
    """
    if not any(point.raw for point in project.points):
        return
    for mark in options.target.points:
        for point_id in mark.point_ids:
            motion_glue.named_track(project, point_id)


class AddTargetRequest(BaseModel):
    """Body for ``POST /api/projects/{id}/render/targets``."""

    place: PlacementIn | None = None
    point_ids: list[str] | None = None
    source: str | None = None


@router.post("/render/targets")
def add_target(
    project_id: str, body: AddTargetRequest, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Add one target point, with an id nothing else is using.

    The id is minted here rather than by the client: the list is addressed by id
    afterwards, so two panels adding a mark at once must not be able to land on
    the same one and make each other's un-deletable.
    """
    project = workspace.load_project(project_id)

    def _add(current: dict[str, Any]) -> dict[str, Any]:
        target = StabilizationTarget.from_dict(current.get("target"))
        mark = TargetPoint(
            id=target.next_id(),
            place=Placement(**body.place.model_dump()) if body.place else Placement(),
            point_ids=tuple(body.point_ids or ()),
            source=body.source or "raw",
        )
        points = [*current["target"]["points"], mark.to_dict()]
        return {**current, "target": {"points": points}}

    return _store_options(workspace, project, project_id, _add)


class PatchTargetRequest(BaseModel):
    """Body for ``PATCH /api/projects/{id}/render/targets/{target_id}``."""

    place: PlacementIn | None = None
    point_ids: list[str] | None = None
    source: str | None = None
    weight: float | None = None


@router.patch("/render/targets/{target_id}")
def patch_target(
    project_id: str,
    target_id: str,
    body: PatchTargetRequest,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Move one target point, or change which track it holds.

    Raises:
        HTTPException: 404 if there is no such mark, 400 if the change is not a
            valid one.
    """
    project = workspace.load_project(project_id)
    patch = body.model_dump(exclude_unset=True)

    def _patch(current: dict[str, Any]) -> dict[str, Any]:
        points = list(current["target"]["points"])
        index = next((i for i, p in enumerate(points) if p.get("id") == target_id), None)
        if index is None:
            raise HTTPException(status_code=404, detail=f"no target point {target_id!r}")
        points[index] = {**points[index], **patch}
        return {**current, "target": {"points": points}}

    return _store_options(workspace, project, project_id, _patch)


@router.delete("/render/targets/{target_id}")
def delete_target(
    project_id: str, target_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Remove one target point.

    Removing the last one is allowed, and means something: with nothing left to
    fit, the render goes back to undoing what the Smoothing panel removed.

    Raises:
        HTTPException: 404 if there is no such mark.
    """
    project = workspace.load_project(project_id)

    def _delete(current: dict[str, Any]) -> dict[str, Any]:
        points = [p for p in current["target"]["points"] if p.get("id") != target_id]
        if len(points) == len(current["target"]["points"]):
            raise HTTPException(status_code=404, detail=f"no target point {target_id!r}")
        return {**current, "target": {"points": points}}

    return _store_options(workspace, project, project_id, _delete)


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
