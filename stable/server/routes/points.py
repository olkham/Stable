"""Point CRUD, anchor pins, the tracking job trigger, and template previews.

A "point" is a :class:`~stable.core.project.PointSpec`: a set of anchor
keyframes plus a tracker choice. Placing more anchors on the timeline (pins) is
what feeds :mod:`stable.trackers.merge`'s multi-anchor blend — this module is
the HTTP surface for that, plus submitting the actual tracking job.

Template matching (todo 14) adds two routes on top: the per-point tracker
settings (``/template``) and the patch itself at any one keyframe
(``/template/patch``), rendered on demand — see
:mod:`stable.server.tracker_options` for why nothing about a patch is stored.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from ...core.errors import ProjectError, StableError
from ...core.project import PointSpec, Project
from ...core.types import Query
from ...trackers.template import TemplateTracker
from ..jobs import JobManager, get_job_manager
from ..schemas import (
    AnchorCreate,
    PointCreate,
    PointUpdate,
    TemplateOptionsUpdate,
    TrackRequest,
)
from ..serialize import point_summary
from ..storage import Workspace, get_workspace
from ..tracker_options import (
    SEARCH_RADIUS_RANGE,
    TEMPLATE_RADIUS_RANGE,
    patch_bounds,
    point_options,
    save_point_options,
    template_patch,
)
from ..tracking import (
    effective_anchors,
    load_anchor_violations,
    load_display_track,
    load_raw_track,
    load_report,
    report_path_for,
    run_point_tracking,
)

__all__ = ["router"]

router = APIRouter(prefix="/projects/{project_id}/points", tags=["points"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


def _nan_to_none(values: list[float]) -> list[float | None]:
    return [None if math.isnan(v) else v for v in values]


@router.get("")
def list_points(
    project_id: str, workspace: Workspace = Depends(_workspace)
) -> list[dict[str, Any]]:
    """List every point on a project, with lightweight track stats."""
    project = workspace.load_project(project_id)
    return [point_summary(project, p) for p in project.points]


@router.post("")
def create_point(
    project_id: str, body: PointCreate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Place a new point: its first anchor keyframe.

    Raises:
        HTTPException: 400 if the anchor is invalid (e.g. past the end of the clip).
    """
    try:
        project = workspace.load_project(project_id)
        point_id = project.new_point_id()
        spec = PointSpec(
            id=point_id,
            queries=[Query(frame=body.frame, x=body.x, y=body.y)],
            tracker=body.tracker or "lk",
        )
        project.points.append(spec)
        project.validate()
        workspace.save_project(project_id, project)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return point_summary(project, project.point(point_id))


@router.patch("/{point_id}")
def update_point(
    project_id: str, point_id: str, body: PointUpdate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Update a point's tracker choice or solve-eligibility flag."""

    def _mutate(project: Project) -> None:
        spec = project.point(point_id)
        if body.tracker is not None:
            spec.tracker = body.tracker
        if body.enabled_for_solve is not None:
            spec.enabled_for_solve = body.enabled_for_solve

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return point_summary(project, project.point(point_id))


@router.delete("/{point_id}")
def delete_point(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, str]:
    """Delete a point and its cached raw track."""

    def _mutate(project: Project) -> None:
        before = len(project.points)
        project.points = [p for p in project.points if p.id != point_id]
        if len(project.points) == before:
            raise ProjectError(f"no point with id {point_id!r}")

    try:
        workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    npz_path = workspace.tracks_dir(project_id) / f"{point_id}.npz"
    npz_path.unlink(missing_ok=True)
    report_path_for(npz_path).unlink(missing_ok=True)
    return {"status": "deleted"}


@router.post("/{point_id}/anchors")
def add_anchor(
    project_id: str, point_id: str, body: AnchorCreate, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Add an anchor keyframe (a timeline pin) to a point.

    Adding an anchor at a frame that already has one replaces it — a pin drag
    in the UI is expressed as delete-then-add at the new frame, so this only
    needs to handle "two anchors, same frame" as a same-position edit.
    """

    def _mutate(project: Project) -> None:
        spec = project.point(point_id)
        spec.queries = [q for q in spec.queries if q.frame != body.frame]
        spec.queries.append(Query(frame=body.frame, x=body.x, y=body.y))

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return point_summary(project, project.point(point_id))


@router.delete("/{point_id}/anchors/{frame}")
def delete_anchor(
    project_id: str, point_id: str, frame: int, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Remove an anchor keyframe.

    Raises:
        HTTPException: 400 if this would leave the point with zero anchors —
            every point needs at least one truth position to track from.
    """

    def _mutate(project: Project) -> None:
        spec = project.point(point_id)
        remaining = [q for q in spec.queries if q.frame != frame]
        if not remaining:
            raise ProjectError("cannot delete a point's last remaining anchor")
        spec.queries = remaining

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return point_summary(project, project.point(point_id))


@router.post("/{point_id}/track")
def track_point(
    project_id: str,
    point_id: str,
    body: TrackRequest,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Submit a background job that tracks a point from every one of its anchors.

    Runs :func:`stable.trackers.merge.plan_anchor_runs`, executes each run with
    the chosen tracker plugin, and merges the results — the real Phase 0-2
    pipeline, not a stub. See :mod:`stable.server.tracking`.

    "Every one of its anchors" includes the ones the operator's manual
    corrections promoted (:func:`stable.server.tracking.effective_anchors`), and
    a run that reaches a keyframe it disagrees with is stopped there rather than
    drifting past it — the job result names the frame and the pixel distance.
    """
    project = workspace.load_project(project_id)
    try:
        project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _run(progress: Any) -> dict[str, Any]:
        return run_point_tracking(
            workspace,
            project_id,
            point_id,
            tracker=body.tracker,
            direction=body.direction,
            device=body.device,
            progress=progress,
            anchor_tolerance_px=body.anchor_tolerance_px,
        )

    job = jobs.submit(project_id, "track", _run, target=point_id)
    return {"job_id": job.id}


def _gaps(track: Any) -> list[list[int]]:
    """Return the inclusive ``[start, end]`` runs of frames with no position.

    The edit panel offers a one-click gap fill per entry, so the user does not
    have to read a range off the timeline by eye.
    """
    valid = track.valid
    runs: list[list[int]] = []
    start: int | None = None
    for frame in range(track.n_frames):
        if not valid[frame] and start is None:
            start = frame
        elif valid[frame] and start is not None:
            runs.append([start, frame - 1])
            start = None
    if start is not None:
        runs.append([start, track.n_frames - 1])
    return runs


@router.get("/{point_id}/track")
def get_track(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Return a point's full per-frame track: position, confidence, visibility.

    This is what the canvas overlay (track path, confidence colouring), the
    timeline's per-point confidence heat-strip and the editor's before/after
    ghosting are built from — hence both the corrected track and, when
    corrections exist, the raw one underneath it.

    Raises:
        HTTPException: 404 if the point has not been tracked yet.
    """
    project = workspace.load_project(project_id)
    try:
        point = project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    track = load_display_track(project, point)
    if track is None:
        raise HTTPException(status_code=404, detail=f"point {point_id!r} has not been tracked yet")

    report = None
    if point.raw:
        path = project.resolve_path(point.raw)
        if path:
            report = load_report(report_path_for(Path(path)))

    anchors = effective_anchors(point)
    violations = load_anchor_violations(project, point)
    payload: dict[str, Any] = {
        "point_id": point_id,
        "n_frames": track.n_frames,
        "x": _nan_to_none(track.points[:, 0].tolist()),
        "y": _nan_to_none(track.points[:, 1].tolist()),
        "confidence": track.confidence.tolist(),
        "visible": track.visible.tolist(),
        "disagreement": (report or {}).get("disagreement", [None] * track.n_frames),
        "contributions": (report or {}).get("contributions", [0] * track.n_frames),
        "anchor_frames": (report or {}).get("anchor_frames", [q.frame for q in anchors]),
        # Every position the operator has stated, pins plus promoted manual
        # corrections — what the panel lists and the timeline pins.
        "ground_truth": [q.to_dict() for q in anchors],
        # Keyframes a run reached and disagreed with: drawn red on the timeline
        # and the reason a track needs manual review (todos 13, 15).
        "anchor_violations": violations,
        "flagged_anchors": sorted({int(v["frame"]) for v in violations}),
        "corrections": [c.to_dict() for c in point.corrections],
        "gaps": _gaps(track),
        "raw_x": None,
        "raw_y": None,
    }
    if any(c.enabled for c in point.corrections):
        raw = load_raw_track(project, point)
        if raw is not None:
            payload["raw_x"] = _nan_to_none(raw.points[:, 0].tolist())
            payload["raw_y"] = _nan_to_none(raw.points[:, 1].tolist())
    return payload


# -- template matching (todo 14) ---------------------------------------------


def _template_defaults() -> tuple[int, int]:
    """The template tracker's own defaults, read from the plugin not duplicated."""
    plugin = TemplateTracker()
    return plugin.template_radius, plugin.search_radius


@router.get("/{point_id}/template")
def get_template_options(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Return the template settings in force for a point, and where its patches are.

    Everything the panel needs to draw itself in one call: the effective radii
    (stored, or the plugin's own defaults), the range the sliders may cover, and
    one entry per keyframe saying whether a patch can actually be built there.
    A keyframe too close to the frame edge is reported rather than 404-ing
    later, so the grid can show *why* a thumbnail is missing.
    """
    project = workspace.load_project(project_id)
    try:
        point = project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    default_radius, default_search = _template_defaults()
    stored = point_options(workspace.project_dir(project_id), point_id)
    radius = int(stored.get("template_radius", default_radius))
    search = int(stored.get("search_radius", default_search))

    video = workspace.get_video_source(project_id)
    anchors = [
        {
            "frame": q.frame,
            "x": q.x,
            "y": q.y,
            "available": patch_bounds((q.x, q.y), radius, video.width, video.height) is not None,
        }
        for q in effective_anchors(point)
    ]
    return {
        "point_id": point_id,
        "tracker": point.tracker,
        "template_radius": radius,
        "search_radius": search,
        "patch_size": radius * 2 + 1,
        "defaults": {"template_radius": default_radius, "search_radius": default_search},
        "template_radius_range": list(TEMPLATE_RADIUS_RANGE),
        "search_radius_range": list(SEARCH_RADIUS_RANGE),
        "custom": bool(stored),
        "anchors": anchors,
    }


@router.put("/{point_id}/template")
def set_template_options(
    project_id: str,
    point_id: str,
    body: TemplateOptionsUpdate,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Set (or clear) a point's template settings.

    A ``null`` for either radius restores the plugin's own default, which is
    what the panel's "reset" does — storing the default explicitly would freeze
    it against any later change to the tracker.
    """
    project = workspace.load_project(project_id)
    try:
        project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = body.model_dump(exclude_unset=True)
    try:
        save_point_options(workspace.project_dir(project_id), point_id, payload)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_template_options(project_id, point_id, workspace)


@router.get("/{point_id}/template/patch")
def get_template_patch(
    project_id: str,
    point_id: str,
    frame: int,
    radius: int | None = None,
    workspace: Workspace = Depends(_workspace),
) -> Response:
    """Return the template patch at one keyframe as a PNG.

    Args:
        frame: Which keyframe to show. Any frame works, but the panel asks for
            the point's anchors.
        radius: Override the stored radius, so the grid can preview a size the
            slider is still being dragged through without saving it first.

    Raises:
        HTTPException: 404 if the point or frame does not exist, 400 if the
            patch would fall outside the frame.
    """
    project = workspace.load_project(project_id)
    try:
        point = project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    default_radius, _ = _template_defaults()
    stored = point_options(workspace.project_dir(project_id), point_id)
    size = int(radius if radius is not None else stored.get("template_radius", default_radius))
    low, high = TEMPLATE_RADIUS_RANGE
    if not low <= size <= high:
        raise HTTPException(status_code=400, detail=f"radius must lie in [{low}, {high}]")

    position = next(((q.x, q.y) for q in effective_anchors(point) if q.frame == frame), None)
    if position is None:
        raise HTTPException(status_code=404, detail=f"point has no keyframe at frame {frame}")

    video = workspace.get_video_source(project_id)
    patch = template_patch(video, frame, position, size)
    if patch is None:
        raise HTTPException(
            status_code=400,
            detail=f"a {size * 2 + 1}px template at frame {frame} falls outside the frame",
        )
    return Response(content=_encode_png(patch), media_type="image/png")


def _encode_png(patch: Any) -> bytes:
    """Encode an RGB patch as PNG bytes.

    PNG rather than JPEG: these are tiny (a 25px square) and the operator is
    judging whether the patch has enough texture to track, which is exactly the
    detail JPEG throws away first.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HTTPException(status_code=500, detail="opencv is required to render patches") from exc
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - only on an encoder failure
        raise HTTPException(status_code=500, detail="could not encode the template patch")
    return bytes(io.BytesIO(encoded.tobytes()).getvalue())
