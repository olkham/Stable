"""Solve / smooth / stabilise for the server — the real Phase 3 stack, cached.

This module replaces Phase 4's ``solve_stub`` / ``smoothing_stub`` /
``render_stub``. It owns nothing mathematical: :mod:`stable.solver`,
:mod:`stable.smoothing` and :mod:`stable.render` do all of it. What lives here
is the glue a *server* needs and a library must not have — reading tracks out of
a project, caching a solve on disk so the smoothing panel does not re-fit the
clip on every slider drag, and turning the results into JSON the frontend reads.

The pivot gotcha
----------------
Solved curves are **pivot-relative** (Phase 3): ``x``/``y`` are how far the
frame's *centre* moved, not an origin-based ``tx``/``ty``. Rebuilding a
transform from raw channel values — which the old ``render_stub`` did — is
therefore wrong by a whole rotation-about-the-corner. Nothing in this module
ever constructs a transform from channels: it goes through
:meth:`~stable.solver.motion.MotionSolve.transform_at`,
:meth:`~stable.smoothing.motion.SmoothedMotion.stabilization_transform` and
:meth:`~stable.render.stabilize.StabilizationPlan.output_transform`, which fold
the pivot back in themselves. The stabilisation *target* added here obeys the
same rule: it is resolved to pixel positions from the tracks and turned into a
translation composed on top of the sanctioned warp, never into channel values.

Render options live beside the project file
-------------------------------------------
The v1 project schema is locked, and its ``render`` block is ``border`` /
``codec`` / ``crf``. The operator's stabilisation target, its position in the
delivered frame, the crop scale, the border fill and the export's own frame
range are therefore stored in :data:`RENDER_OPTIONS_CACHE`, a sidecar in the
project directory alongside ``solve.json`` and ``smoothing.json`` — derived
render state, in the same place as the rest of it, with no invented schema keys
and nothing for a project-file migration to carry.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ProjectError
from ..core.geometry import CHANNELS
from ..core.project import Project
from ..core.types import PointTrack
from ..render.options import RenderOptions, StabilizationTarget, average_positions
from ..render.stabilize import StabilizationPlan
from ..smoothing.channels import SmoothingSettings
from ..smoothing.motion import SmoothedMotion, smooth_motion
from ..solver.motion import MotionSolve, SolveConfig, SolveStatus, solve_motion
from ..trackers.base import ProgressCallback, noop_progress

__all__ = [
    "SOLVE_CACHE",
    "SMOOTHING_CACHE",
    "RENDER_OPTIONS_CACHE",
    "solve_tracks",
    "smooth_solve",
    "solve_to_cache",
    "solve_from_cache",
    "solve_payload",
    "smoothed_payload",
    "load_render_options",
    "save_render_options",
    "named_track",
    "smoothed_paths",
    "resolve_targets",
    "target_label",
    "target_residuals",
    "target_choices",
    "build_plan",
    "cached_plan",
    "clear_plan_cache",
]

SOLVE_CACHE = "solve.json"
"""File name of a project's cached raw solve, inside its project directory."""

SMOOTHING_CACHE = "smoothing.json"
"""File name of a project's cached smoothed curves."""

RENDER_OPTIONS_CACHE = "render_options.json"
"""File name of a project's stabilisation target / crop / border-fill choices."""


# -- solving ------------------------------------------------------------------


def solve_tracks(
    project: Project,
    tracks: list[PointTrack],
    *,
    progress: ProgressCallback = noop_progress,
) -> MotionSolve:
    """Solve a project's motion from the (corrected) tracks of its enabled points.

    Args:
        project: The project, for its solve settings and clip size.
        tracks: One track per solve-enabled point, all over the clip timeline.
        progress: Progress reporter forwarded to :func:`stable.solver.solve_motion`.

    Returns:
        The :class:`~stable.solver.motion.MotionSolve`.

    Raises:
        ProjectError: If no usable track was supplied.
        stable.core.errors.SolverError: If the solve cannot run at all (e.g. no
            point is usable in the reference frame).
    """
    if not tracks:
        raise ProjectError("no usable tracked points to solve from")
    frame_size = (
        (int(project.video.w), int(project.video.h))
        if project.video.w and project.video.h
        else None
    )
    config = SolveConfig.from_settings(project.solve)
    return solve_motion(tracks, config=config, frame_size=frame_size, progress=progress)


def smooth_solve(project: Project, solve: MotionSolve) -> SmoothedMotion:
    """Smooth a solve with the project's filter list.

    Args:
        project: The project, for its ``smoothing`` specs and frame rate.
        solve: The raw solve to smooth.

    Returns:
        The :class:`~stable.smoothing.motion.SmoothedMotion`.

    Raises:
        stable.core.errors.SmoothingError: If a filter name or parameter is invalid.
    """
    settings = SmoothingSettings.from_specs(project.smoothing)
    return smooth_motion(solve, settings, fps=project.video.fps or 25.0)


# -- disk cache ----------------------------------------------------------------


def solve_to_cache(solve: MotionSolve) -> dict[str, Any]:
    """Return the JSON-safe dict written to ``solve.json``.

    :meth:`~stable.solver.motion.MotionSolve.to_dict` carries everything except
    the flag threshold, which is added here so the round trip is exact.
    """
    payload = solve.to_dict()
    payload["max_residual_px"] = solve.max_residual_px
    return payload


def solve_from_cache(payload: dict[str, Any]) -> MotionSolve:
    """Rebuild a :class:`~stable.solver.motion.MotionSolve` from :func:`solve_to_cache`.

    Raises:
        ProjectError: If the cached payload is missing a field or names a status
            this build does not know.
    """
    try:
        stored = payload["curves"]
        curves = {channel: np.asarray(stored[channel], dtype=np.float64) for channel in CHANNELS}
        residual = np.asarray(
            [np.nan if value is None else float(value) for value in payload["residual"]],
            dtype=np.float64,
        )
        status = np.asarray(
            [int(SolveStatus[str(label).upper()]) for label in payload["status"]], dtype=np.int8
        )
        pivot = payload["pivot"]
        return MotionSolve(
            curves=curves,
            residual=residual,
            n_points=np.asarray(payload["n_points"], dtype=np.int32),
            n_inliers=np.asarray(payload["n_inliers"], dtype=np.int32),
            status=status,
            reference_frame=int(payload["reference_frame"]),
            pivot=(float(pivot[0]), float(pivot[1])),
            max_residual_px=float(payload.get("max_residual_px", 3.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(f"cached solve is unreadable: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    """Return a JSON file's contents, or ``None`` if it does not exist."""
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


# -- JSON payloads for the frontend --------------------------------------------


def solve_payload(solve: MotionSolve) -> dict[str, Any]:
    """Return the ``curves`` object the solve/smoothing/export endpoints serve.

    The four channels sit at the top level (``x``, ``y``, ``r``, ``s``) so the
    shape the Phase 4 frontend was written against still holds; the per-frame
    quality record Phase 3 added (status, inliers, flagged) rides alongside and
    is what the curve editor colours its "go fix this" markers from. ``residual``
    is ``null`` on frames nothing was fitted for, rather than a fake ``0``.
    """
    return {
        **{channel: solve.curve(channel).tolist() for channel in CHANNELS},
        "residual": [None if not np.isfinite(v) else float(v) for v in solve.residual],
        "n_points": [int(v) for v in solve.n_points],
        "n_inliers": [int(v) for v in solve.n_inliers],
        "status": [SolveStatus(int(v)).label for v in solve.status],
        "flagged": [bool(v) for v in solve.flagged],
        "reference_frame": solve.reference_frame,
        "pivot": list(solve.pivot),
        "summary": solve.summary(),
    }


def smoothed_payload(motion: SmoothedMotion) -> dict[str, Any]:
    """Return the smoothed ``{x, y, r, s}`` curves plus the per-channel settings."""
    return {
        **{channel: motion.curve(channel).tolist() for channel in CHANNELS},
        "settings": motion.settings.to_dict(),
        "fps": motion.fps,
    }


# -- stabilisation options ------------------------------------------------------


def options_path(project_dir: Path) -> Path:
    """Return where a project's render options sidecar lives."""
    return Path(project_dir) / RENDER_OPTIONS_CACHE


def load_render_options(
    project_dir: Path, frame_size: tuple[int, int] | None = None
) -> RenderOptions:
    """Return a project's stabilisation options, or the defaults if never set.

    A missing or unreadable sidecar is not an error: the default target is the
    average of every tracked point held at the centre of the frame, which is
    what a project that has never opened the export screen should render as.

    Args:
        project_dir: The project's directory.
        frame_size: The clip's ``(width, height)``. Only used to read a sidecar
            written before todo 1, eighth round, whose ``map`` marks were stored as pixels;
            without it those fall back to the output centre rather than being
            misread as fractions.
    """
    try:
        payload = read_json(options_path(project_dir))
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar only
        return RenderOptions()
    return RenderOptions.from_dict(payload, frame_size)


def save_render_options(project_dir: Path, options: RenderOptions) -> dict[str, Any]:
    """Persist stabilisation options and return the JSON payload written."""
    payload = options.to_dict()
    write_json(options_path(project_dir), payload)
    return payload


def target_choices(project: Project) -> list[dict[str, Any]]:
    """Return the points that can serve as a stabilisation target, for the UI.

    Only tracked points qualify — a target with no path cannot pin anything —
    so the export dialog can offer exactly the choices that will work instead
    of a list that fails at render time.
    """
    from .tracking import load_display_track  # noqa: PLC0415 - avoids an import cycle

    out: list[dict[str, Any]] = []
    for point in project.points:
        track = load_display_track(project, point)
        if track is None:
            continue
        out.append(
            {
                "id": point.id,
                "enabled_for_solve": bool(point.enabled_for_solve),
                "frames": int(np.count_nonzero(track.valid)),
            }
        )
    return out


def named_track(project: Project, point_id: str) -> Any:
    """Return one point's corrected track, by id.

    Raises:
        ProjectError: If the project has no such point, or it has no track —
            a target is only as real as the track underneath it, and finding
            that out here beats finding it out inside an export job.
    """
    from .tracking import load_display_track  # noqa: PLC0415 - avoids an import cycle

    point = next((p for p in project.points if p.id == point_id), None)
    if point is None:
        raise ProjectError(
            f"a target point holds track {point_id!r}, which this project does not have; "
            "remove that target or point it at a track that exists"
        )
    track = load_display_track(project, point)
    if track is None:
        raise ProjectError(f"the target's track {point_id!r} has not been tracked yet")
    return track


def smoothed_paths(project: Project, motion: SmoothedMotion) -> dict[str, np.ndarray]:
    """Return every tracked point's path **after** the smoothing's own warp.

    The dashed trail the viewer draws, and what ``GET /smoothing/tracks``
    serves — one implementation, so a target that says "hold the smoothed
    estimate" holds exactly the curve the operator is looking at rather than one
    that merely resembles it.
    """
    from .tracking import load_display_track  # noqa: PLC0415 - avoids an import cycle

    warps = motion.stabilization_transforms()
    out: dict[str, np.ndarray] = {}
    for point in project.points:
        track = load_display_track(project, point)
        if track is None:
            continue
        positions = np.asarray(track.points, dtype=np.float64)
        moved = np.full_like(positions, np.nan)
        for frame in range(min(len(warps), positions.shape[0])):
            source = positions[frame]
            if np.isfinite(source).all():
                moved[frame] = warps[frame].apply(source)
        out[point.id] = moved
    return out


def resolve_targets(
    project: Project, target: StabilizationTarget, motion: SmoothedMotion
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Turn the operator's marks into the arrays the plan fits.

    Args:
        project: The project holding the tracks.
        target: The marks, in order.
        motion: The smoothed motion, needed only by ``smoothed`` marks — that is
            what turns a track into the dashed trail.

    Returns:
        ``(sources[K, T, 2], placements[K, 2], weights[K, T])``, or ``None``
        when there are no marks at all (the render then falls back to the
        smoothing's own warp).

    Raises:
        ProjectError: If a mark names a track the project does not have or that
            has not been tracked, or asks for the mean of nothing.
    """
    if not target.points:
        return None
    smoothed: dict[str, np.ndarray] | None = None
    sources: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    placements: list[tuple[float, float]] = []
    for mark in target.points:
        ids = list(mark.point_ids)
        if not ids:
            ids = [p.id for p in project.points if p.enabled_for_solve and p.raw]
            if not ids:
                raise ProjectError(
                    f"target {mark.id} holds every track, and this project has no tracked, "
                    "solve-enabled point; track one, or point the target at a specific track"
                )
        tracks = [named_track(project, point_id) for point_id in ids]
        if mark.is_smoothed:
            if smoothed is None:
                smoothed = smoothed_paths(project, motion)
            paths = [smoothed[point_id] for point_id in ids]
        else:
            paths = [np.asarray(t.points, dtype=np.float64) for t in tracks]
        sources.append(paths[0] if len(paths) == 1 else average_positions(paths))
        confidences = np.stack([np.asarray(t.confidence, dtype=np.float64) for t in tracks])
        weights.append(confidences.mean(axis=0) * mark.weight)
        placements.append(mark.place.fraction)
    return (
        np.stack(sources),
        np.asarray(placements, dtype=np.float64),
        np.stack(weights),
    )


def target_label(target: StabilizationTarget) -> str:
    """Return the short phrase the export screen and the job record use."""
    if not target.points:
        return "smoothing only"
    if len(target.points) == 1:
        return target.points[0].label()
    return f"{len(target.points)} targets"


def target_residuals(
    project: Project,
    plan: Any,
    target: StabilizationTarget,
    frame: int,
    motion: SmoothedMotion | None = None,
) -> list[dict[str, Any]]:
    """Report where each held path actually lands, and how far that misses by.

    One mark is satisfied exactly and reads ~0 px, which is itself worth seeing:
    it says the render is doing what the panel claims. Two are satisfied exactly
    too. Three or more generally *cannot* all be — a similarity has four degrees
    of freedom and they ask for six — so the fit is least-squares and the miss is
    shown per mark rather than an average being presented as if it were the
    answer.

    Args:
        project: The project holding the tracks.
        plan: The built :class:`~stable.render.stabilize.StabilizationPlan`.
        target: The marks, in order.
        frame: Which frame to report for.
        motion: The smoothed motion, needed only when a mark holds a smoothed
            path. Without it those rows report no landing rather than guessing.

    Returns:
        One row per mark: its id, the tracks it holds, what it was asked for,
        where the plan puts it, and the distance between them — all in **output
        pixels**, so they read against each other and against the output canvas.
    """
    size = (float(plan.geometry.width), float(plan.geometry.height))
    index = max(0, min(int(frame), len(plan.transforms) - 1)) if plan.transforms else 0
    output_transform = plan.geometry.fit.compose(
        plan.transforms[index] if plan.transforms else plan.geometry.fit
    )
    resolved = (
        resolve_targets(project, target, motion)
        if motion is not None and target.points
        else None
    )

    rows: list[dict[str, Any]] = []
    for k, mark in enumerate(target.points):
        wanted = mark.place.pixels(size)
        row: dict[str, Any] = {
            "id": mark.id,
            # `point_id` is what the viewer colours a crosshair from; a mark
            # over several tracks belongs to none of them and gets none.
            "point_id": mark.point_ids[0] if len(mark.point_ids) == 1 else None,
            "point_ids": list(mark.point_ids),
            "source": mark.source,
            "label": mark.label(),
            "target": [float(wanted[0]), float(wanted[1])],
            "landed": None,
            "residual": None,
        }
        source_point = (
            resolved[0][k][index]
            if resolved is not None and index < resolved[0].shape[1]
            else None
        )
        if source_point is not None and np.isfinite(source_point).all():
            landed = output_transform.apply(source_point)
            row["landed"] = [float(landed[0]), float(landed[1])]
            row["residual"] = float(
                np.hypot(landed[0] - wanted[0], landed[1] - wanted[1])
            )
        rows.append(row)
    return rows


# -- render planning ------------------------------------------------------------


def build_plan(
    project: Project,
    motion: SmoothedMotion,
    *,
    border: str | None = None,
    options: RenderOptions | None = None,
    export_range: tuple[int, int] | None = None,
) -> StabilizationPlan:
    """Plan the stabilised render for a project's smoothed motion.

    Args:
        project: The project, for its clip size, border policy and tracks.
        motion: The smoothed solve.
        border: Override for the project's border policy.
        options: The operator's target / crop / fill / resolution choices;
            ``None`` means the defaults.
        export_range: The frames the export will write. Only consulted when the
            crop was asked to fit that range
            (:attr:`~stable.render.options.CropSpec.limit_to_range`) — the plan
            always covers the whole clip, because the warp for frame N is the
            warp for frame N whatever ends up being written.

    Returns:
        The :class:`~stable.render.stabilize.StabilizationPlan`.

    Raises:
        stable.core.errors.RenderError: If the border policy is unknown or an
            automatic crop finds the warped frames share no common area.
        ProjectError: If a target names a track this project cannot supply.
    """
    size = (int(project.video.w), int(project.video.h))
    chosen = options or RenderOptions()
    resolved = resolve_targets(project, chosen.target, motion)
    return StabilizationPlan.build(
        motion,
        source_size=size,
        border=border or project.render.border,
        sources=None if resolved is None else resolved[0],
        placements=None if resolved is None else resolved[1],
        weights=None if resolved is None else resolved[2],
        crop=chosen.crop,
        fill=chosen.fill,
        coverage_range=export_range if chosen.crop.limit_to_range else None,
        resolution=chosen.resolution,
    )


_PLAN_CACHE: dict[str, tuple[Any, StabilizationPlan]] = {}
_PLAN_LOCK = threading.Lock()
_PLAN_CACHE_LIMIT = 8


def cached_plan(project_id: str, key: Any, factory: Any) -> StabilizationPlan:
    """Return a memoised render plan, rebuilding it only when ``key`` changes.

    The minimal auto-zoom is a whole-clip computation, and a preview scrub asks
    for one frame at a time — so planning per request would redo an O(T) solve
    for every frame the user drags past. ``key`` is whatever identifies the
    inputs (the cache files' modification times and the border policy); when it
    matches, the stored plan is reused.

    Args:
        project_id: The project the plan belongs to.
        key: Any hashable/comparable identity of the plan's inputs.
        factory: Zero-argument callable building the plan on a miss.

    Returns:
        The cached or freshly built :class:`StabilizationPlan`.
    """
    with _PLAN_LOCK:
        entry = _PLAN_CACHE.get(project_id)
        if entry is not None and entry[0] == key:
            return entry[1]
    plan = factory()
    with _PLAN_LOCK:
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT and project_id not in _PLAN_CACHE:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)), None)
        _PLAN_CACHE[project_id] = (key, plan)
    return plan


def clear_plan_cache(project_id: str | None = None) -> None:
    """Drop cached render plans — for one project, or all of them."""
    with _PLAN_LOCK:
        if project_id is None:
            _PLAN_CACHE.clear()
        else:
            _PLAN_CACHE.pop(project_id, None)
