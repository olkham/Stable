"""Solve settings + job, smoothing preview, and the stabilized-frame preview.

The curve math is the real Phase 3 stack — :mod:`stable.solver`,
:mod:`stable.smoothing`, :mod:`stable.render` — reached through
:mod:`stable.server.motion`, which owns the on-disk cache and the JSON shapes.
This module is only the HTTP surface: settings persistence, job submission and
serving one warped frame.

Note:
    Solved curves are pivot-relative, so a stabilisation warp is never rebuilt
    from raw channel values here. See :mod:`stable.server.motion`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from ...core.errors import StableError
from ...core.project import Project, SmoothingSpec
from ...render.warp import rescale_transform, warp_frame
from ...smoothing.motion import SmoothedMotion
from ...solver.motion import MotionSolve
from .. import motion as motion_glue
from ..jobs import JobManager, get_job_manager
from ..schemas import SmoothingUpdate, SolveSettingsUpdate
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track

__all__ = ["router", "load_solve", "load_smoothed", "recompute_smoothed"]

router = APIRouter(prefix="/projects/{project_id}", tags=["solve"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


def _solve_path(workspace: Workspace, project_id: str) -> Any:
    return workspace.project_dir(project_id) / motion_glue.SOLVE_CACHE


def _smoothing_path(workspace: Workspace, project_id: str) -> Any:
    return workspace.project_dir(project_id) / motion_glue.SMOOTHING_CACHE


def load_solve(workspace: Workspace, project_id: str) -> MotionSolve | None:
    """Return a project's cached raw solve, or ``None`` if it has not been solved.

    Shared with :mod:`stable.server.routes.export`, which needs the same cache.
    """
    payload = motion_glue.read_json(_solve_path(workspace, project_id))
    return None if payload is None else motion_glue.solve_from_cache(payload)


def load_smoothed(
    workspace: Workspace, project_id: str, project: Project | None = None
) -> SmoothedMotion | None:
    """Return a project's smoothed motion, or ``None`` if it has not been solved.

    The smoothed *curves* are cached, but the object handed back is rebuilt on
    top of the cached solve so callers get the real
    :class:`~stable.smoothing.motion.SmoothedMotion` — the thing that knows how
    to produce a stabilisation warp — rather than four bare lists.
    """
    solve = load_solve(workspace, project_id)
    if solve is None:
        return None
    if not _smoothing_path(workspace, project_id).is_file():
        return None
    return motion_glue.smooth_solve(project or workspace.load_project(project_id), solve)


def recompute_smoothed(
    workspace: Workspace, project_id: str, project: Project
) -> dict[str, Any] | None:
    """Recompute and cache the smoothed curves from the cached raw solve, if any."""
    solve = load_solve(workspace, project_id)
    if solve is None:
        return None
    smoothed = motion_glue.smooth_solve(project, solve)
    payload = motion_glue.smoothed_payload(smoothed)
    motion_glue.write_json(_smoothing_path(workspace, project_id), payload)
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
            _solve_path(workspace, project_id), motion_glue.solve_to_cache(solve)
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
    smoothed = motion_glue.read_json(_smoothing_path(workspace, project_id))
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


# -- stabilized frame preview --------------------------------------------------


def _preview_plan(workspace: Workspace, project_id: str, project: Project, motion: SmoothedMotion):
    """Return the render plan for the preview, memoised on the caches' mtimes."""
    key = (
        _solve_path(workspace, project_id).stat().st_mtime_ns,
        _smoothing_path(workspace, project_id).stat().st_mtime_ns,
        project.render.border,
        project.video.w,
        project.video.h,
    )
    return motion_glue.cached_plan(
        project_id, key, lambda: motion_glue.build_plan(project, motion)
    )


@router.get("/frames/{index}/stabilized")
def get_stabilized_frame(
    project_id: str, index: int, tier: str = "half", workspace: Workspace = Depends(_workspace)
) -> Response:
    """Return one proxy frame warped by the solved+smoothed stabilization transform.

    The warp is the render's own ``output_transform`` — stabilisation plus the
    project's border policy (crop / auto_zoom / transparent) — re-expressed for
    the proxy's resolution by :func:`stable.render.warp.rescale_transform`, so
    what the preview shows is what the export will produce.

    Raises:
        HTTPException: 409 if the project has not been solved yet; 404 if the
            frame index or proxy tier is invalid.
    """
    project = workspace.load_project(project_id)
    motion = load_smoothed(workspace, project_id, project)
    if motion is None:
        raise HTTPException(
            status_code=409, detail="run solve and smoothing before requesting a stabilized preview"
        )
    if not 0 <= index < motion.n_frames:
        raise HTTPException(status_code=404, detail=f"frame {index} outside the solved range")
    try:
        store = workspace.get_proxy_store(project_id, tier)
        frame = store.get_frame(index)
        plan = _preview_plan(workspace, project_id, project, motion)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    scale = float(store.manifest.scale)
    warped = warp_frame(
        frame,
        rescale_transform(plan.output_transform(index), scale),
        size=(
            max(int(round(plan.geometry.width * scale)), 1),
            max(int(round(plan.geometry.height * scale)), 1),
        ),
        border=plan.border,
        interpolation=plan.interpolation,
    )

    import cv2  # noqa: PLC0415 - lazy, matches the rest of the video stack

    # JPEG carries no alpha, so the transparent policy's uncovered area is shown
    # as the black it was composited against rather than failing the request.
    rgb = warped[:, :, :3] if warped.shape[2] == 4 else warped
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - encoder failure only
        raise HTTPException(status_code=500, detail="could not encode stabilized frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
