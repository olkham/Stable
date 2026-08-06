"""Point CRUD, keyframes, the tracking job trigger, and template previews.

A "point" is a :class:`~stable.core.project.PointSpec`: a set of keyframes plus
a tracker choice. Placing more keyframes on the timeline is what feeds
:mod:`stable.trackers.merge`'s multi-anchor blend — this module is the HTTP
surface for that, plus submitting the actual tracking job. Since todo_6 a
keyframe edit re-tracks the point by itself unless the caller opts out; see
:func:`_auto_track`.

Todo 14 adds two routes on top: what patches a point has (``/patches``) and the
pixels of one of them (``/patches/image``), rendered on demand — see
:mod:`stable.server.tracker_options` for why nothing about a patch is stored.
Both work for *any* tracker, not only template matching; see the section
comment above them.
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from ...core.corrections import GROUND_TRUTH_KINDS
from ...core.errors import ProjectError, StableError
from ...core.project import PointSpec, Project
from ...core.types import Query
from ...trackers.template import TemplateTracker
from ..jobs import JobManager, get_job_manager
from ..schemas import (
    KeyframeCreate,
    KeyframeUpdate,
    PointCreate,
    PointUpdate,
    TemplateOptionsUpdate,
    TrackMode,
    TrackRequest,
)
from ..serialize import point_summary
from ..storage import PROJECT_FILENAME, Workspace, get_workspace
from ..tracker_options import (
    SEARCH_RADIUS_RANGE,
    TEMPLATE_RADIUS_RANGE,
    TRACKER_OPTIONS_CACHE,
    constructor_parameters,
    forget_point,
    load_tracker_options,
    patch_bounds,
    point_options,
    proxy_patch,
    save_point_options,
    setting_for,
    template_patch,
    tracker_search,
    tracker_settings,
    tracker_window,
)
from ..tracking import (
    effective_anchors,
    effective_keyframes,
    load_display_track,
    load_keyframe_violations,
    load_raw_track,
    load_report,
    report_path_for,
    run_point_tracking,
)
from .edits import discard_corrections

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
    stored = load_tracker_options(workspace.project_dir(project_id))
    return [point_summary(project, p, stored.get(p.id, {})) for p in project.points]


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
    # Render its patch now rather than when the panel first asks for it (todo 3,
    # sixth round): making a keyframe is the moment the pixels it names are
    # known, and it is not a moment anybody is waiting on a thumbnail.
    warm_patches(workspace, project_id, point_id, [(body.frame, body.x, body.y)])
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
) -> dict[str, Any]:
    """Delete a point and **everything on disk that was keyed by its id**.

    Deleting only ``tracks/<id>.npz`` and its report was the server half of "when
    I delete a point and add a new one back it restores the previous track": the
    retrack runs (``tracks/<id>.<correction_id>.npz``), the anchor-violation
    sidecar and the point's stored tracker settings all survived, and a new
    point handed the same id inherited them. Ids are no longer recycled either
    (:meth:`stable.core.project.Project.new_point_id`), but a point's files
    outliving it is a leak whether or not anything picks them up.

    Returns:
        The status and the artefacts actually removed, so the caller can see
        that the sweep did something rather than trusting that it did.
    """

    def _mutate(project: Project) -> None:
        before = len(project.points)
        project.points = [p for p in project.points if p.id != point_id]
        if len(project.points) == before:
            raise ProjectError(f"no point with id {point_id!r}")

    try:
        workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    removed: list[str] = []
    tracks = workspace.tracks_dir(project_id)
    if tracks.is_dir():
        # Everything this point owns is named for it: `p1.npz`, `p1.report.json`,
        # `p1.<correction>.npz`, `p1.violations.json`. Matching on the id plus a
        # separator is what stops `p1` from taking `p10`'s files with it.
        for path in sorted(tracks.iterdir()):
            if path.is_file() and path.name.startswith(f"{point_id}."):
                path.unlink(missing_ok=True)
                removed.append(path.name)

    # ...and the per-point settings sidecar, which is keyed by id but lives
    # outside tracks/ (see stable/server/tracker_options.py).
    if forget_point(workspace.project_dir(project_id), point_id):
        removed.append(f"{TRACKER_OPTIONS_CACHE}:{point_id}")

    clear_patch_cache()
    return {"status": "deleted", "removed": removed}


# -- keyframes (todo_6) --------------------------------------------------------
#
# A keyframe is one thing to the operator — "the feature is HERE on this frame" —
# and two things in storage: a pin in `PointSpec.queries`, or a manual correction
# carrying an absolute position that was promoted to ground truth. Both had their
# own route pair, their own chip and their own delete gesture, and nobody could
# say why. These routes are where that stops: they all speak
# `stable.server.tracking.effective_keyframes` and decide for themselves which
# storage a given edit has to rewrite. The project file is untouched by the
# change — v1 files keep loading, and the corrections layer keeps its
# raw-is-immutable, undo-is-drop-a-correction contract.


def submit_track_job(
    workspace: Workspace,
    jobs: JobManager,
    project_id: str,
    point_id: str,
    request: TrackRequest,
) -> str:
    """Queue the full multi-keyframe track for one point and return the job id."""

    def _run(progress: Any) -> dict[str, Any]:
        return run_point_tracking(
            workspace,
            project_id,
            point_id,
            tracker=request.tracker,
            direction=request.direction,
            device=request.device,
            progress=progress,
            anchor_tolerance_px=request.anchor_tolerance_px,
        )

    return jobs.submit(project_id, "track", _run, target=point_id).id


def _auto_track(
    workspace: Workspace,
    jobs: JobManager,
    project_id: str,
    point_id: str,
    mode: TrackMode,
) -> str | None:
    """Re-track a point after a keyframe edit, unless the caller asked not to.

    Full-point is the right grain even for a single moved keyframe: the merge
    re-runs globally regardless, and its cost is capped at roughly twice the
    clip however many keyframes there are. What is *not* free is queueing one
    run per edit while the operator drags, so any run for this point that has
    not started yet is superseded first (see
    :meth:`stable.server.jobs.JobManager.cancel_queued_matching`).

    Returns:
        The submitted job's id, or ``None`` for ``track="none"``.
    """
    if mode != "auto":
        return None
    jobs.cancel_queued_matching(project_id, "track", point_id)
    return submit_track_job(workspace, jobs, project_id, point_id, TrackRequest())


def _claims_on(point: PointSpec, frame: int) -> list[str]:
    """Return the ids of enabled corrections that place the point on ``frame``.

    Those are what a new statement about that frame has to clear out. Ground
    truth kinds (``replace``, ``retrack``) are the obvious ones — they *are* the
    keyframe. ``offset`` is here too because it is half of one: a drag is stored
    as an offset plus a companion ``replace`` (see
    :func:`stable.server.routes.edits._promotion_for`), and dropping only the
    replace would leave the nudge to displace the very position the operator
    just restated.
    """
    displacing = (*GROUND_TRUTH_KINDS, "offset")
    return [
        correction.id
        for correction in point.corrections
        if correction.enabled
        and correction.kind in displacing
        and getattr(correction, "frame", None) == frame
    ]


def _keyframe_payload(project: Project, point: PointSpec, job_id: str | None) -> dict[str, Any]:
    """Return the shape every mutating keyframe route answers with.

    The point summary, so a caller can refresh its card from the same response;
    the resulting keyframe list *and* correction list, because a keyframe edit
    may have discarded a correction and a panel showing the stale one would be
    the two-concepts problem all over again; and the id of the auto-track this
    edit triggered (``None`` when it triggered none), so the caller can subscribe
    to the run over the jobs WebSocket.
    """
    return {
        **point_summary(project, point),
        "keyframes": [k.to_dict() for k in effective_keyframes(point)],
        "corrections": [c.to_dict() for c in point.corrections],
        "job_id": job_id,
    }


def _unlink(paths: Iterable[Path]) -> None:
    """Delete files that nothing references any more, best effort."""
    for path in paths:
        path.unlink(missing_ok=True)


@router.get("/{point_id}/keyframes")
def list_keyframes(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> list[dict[str, Any]]:
    """List every position the operator has stated for a point, in frame order.

    Each entry carries where it came from — ``source`` is ``"pin"`` or
    ``"correction"``, with the promoting ``correction_id`` for the latter. The
    UI has no use for the distinction; a caller reconciling this list with the
    corrections panel does.
    """
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)
    return [k.to_dict() for k in effective_keyframes(point)]


@router.post("/{point_id}/keyframes")
def add_keyframe(
    project_id: str,
    point_id: str,
    body: KeyframeCreate,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, Any]:
    """State where a point truly is on one frame, replacing whatever said otherwise.

    One frame, one truth: a pin already on that frame is overwritten, and a
    correction claiming it is discarded rather than shadowed. Shadowing was the
    tempting shortcut and it is wrong — the correction would keep displacing the
    displayed track at that frame while the pin quietly told the next merge
    something else.

    Raises:
        HTTPException: 400 if the frame lies past the end of the clip.
    """
    orphaned: list[Path] = []

    def _mutate(project: Project) -> None:
        spec = project.point(point_id)
        orphaned.extend(discard_corrections(project, spec, _claims_on(spec, body.frame)))
        spec.queries = [q for q in spec.queries if q.frame != body.frame]
        spec.queries.append(Query(frame=body.frame, x=body.x, y=body.y))

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _unlink(orphaned)
    # Same as placing a point: the patch is rendered while the operator is still
    # looking at the frame they keyed, not when they next open the card.
    warm_patches(workspace, project_id, point_id, [(body.frame, body.x, body.y)])
    job_id = _auto_track(workspace, jobs, project_id, point_id, body.track)
    return _keyframe_payload(project, project.point(point_id), job_id)


@router.patch("/{point_id}/keyframes/{frame}")
def move_keyframe(
    project_id: str,
    point_id: str,
    frame: int,
    body: KeyframeUpdate,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, Any]:
    """Move a keyframe — along the timeline, across the image, or both — in one write.

    A drag is one gesture and has to be one mutation. Expressed as
    delete-then-add it is two, and between them the point is missing a truth
    position: a delete of the last keyframe is refused outright, and an
    auto-track that slipped into the gap would have tracked from a state the
    operator never asked for.

    Whatever backed the keyframe is replaced by a pin at the new location; a
    correction-backed one is discarded with its cached run, because the operator
    has just restated that position and the old claim could only fight the new
    one at the next merge. Landing on another keyframe's frame replaces it, the
    same way :func:`add_keyframe` does.

    Raises:
        HTTPException: 404 if the point has no keyframe on ``frame``, 400 if the
            destination lies past the end of the clip.
    """
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)
    current = next((k for k in effective_keyframes(point) if k.frame == frame), None)
    if current is None:
        raise HTTPException(
            status_code=404, detail=f"point {point_id!r} has no keyframe on frame {frame}"
        )
    target = Query(
        frame=frame if body.frame is None else body.frame,
        x=current.x if body.x is None else body.x,
        y=current.y if body.y is None else body.y,
    )
    orphaned: list[Path] = []

    def _mutate(proj: Project) -> None:
        spec = proj.point(point_id)
        claimed = {*_claims_on(spec, frame), *_claims_on(spec, target.frame)}
        orphaned.extend(discard_corrections(proj, spec, claimed))
        spec.queries = [q for q in spec.queries if q.frame not in (frame, target.frame)]
        spec.queries.append(target)

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _unlink(orphaned)
    warm_patches(workspace, project_id, point_id, [(target.frame, target.x, target.y)])
    job_id = _auto_track(workspace, jobs, project_id, point_id, body.track)
    return _keyframe_payload(project, project.point(point_id), job_id)


@router.delete("/{point_id}/keyframes/{frame}")
def delete_keyframe(
    project_id: str,
    point_id: str,
    frame: int,
    track: TrackMode = "auto",
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, Any]:
    """Remove the keyframe on a frame, whatever states it.

    A pin is dropped; a promoting correction is discarded along with the cached
    run of a retrack, since leaving it would put the keyframe straight back. The
    caller does not have to know which it was — that asymmetry is precisely what
    todo_6 exists to hide.

    Raises:
        HTTPException: 404 if there is no keyframe on that frame, 400 if it is
            the point's last one. The count is of *effective* keyframes, not
            pins: a point whose only truth position came from a correction is
            just as unable to be tracked as one with no pins at all.
    """
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)
    keyframes = effective_keyframes(point)
    if not any(k.frame == frame for k in keyframes):
        raise HTTPException(
            status_code=404, detail=f"point {point_id!r} has no keyframe on frame {frame}"
        )
    if len(keyframes) == 1:
        raise HTTPException(
            status_code=400, detail="cannot delete a point's last remaining keyframe"
        )
    orphaned: list[Path] = []

    def _mutate(proj: Project) -> None:
        spec = proj.point(point_id)
        orphaned.extend(discard_corrections(proj, spec, _claims_on(spec, frame)))
        spec.queries = [q for q in spec.queries if q.frame != frame]

    try:
        project = workspace.update_project(project_id, _mutate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _unlink(orphaned)
    job_id = _auto_track(workspace, jobs, project_id, point_id, track)
    return _keyframe_payload(project, project.point(point_id), job_id)


@router.post("/{point_id}/track")
def track_point(
    project_id: str,
    point_id: str,
    body: TrackRequest,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Submit a background job that tracks a point from every one of its keyframes.

    Runs :func:`stable.trackers.merge.plan_anchor_runs`, executes each run with
    the chosen tracker plugin, and merges the results — the real Phase 0-2
    pipeline, not a stub. See :mod:`stable.server.tracking`.

    "Every one of its keyframes" includes the ones the operator's manual
    corrections promoted (:func:`stable.server.tracking.effective_keyframes`),
    and a run that reaches a keyframe it disagrees with is stopped there rather
    than drifting past it — the job result names the frame and the pixel
    distance.
    """
    project = workspace.load_project(project_id)
    try:
        project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job_id": submit_track_job(workspace, jobs, project_id, point_id, body)}


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
    violations = load_keyframe_violations(project, point)
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
        # corrections — what the panel lists and the timeline ticks.
        "keyframes": [k.to_dict() for k in effective_keyframes(point)],
        # Keyframes a run reached and disagreed with: drawn red on the timeline
        # and the reason a track needs manual review (todos 13, 15).
        "keyframe_violations": violations,
        "flagged_keyframes": sorted({int(v["frame"]) for v in violations}),
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


# -- tracked patches (todo 14) ------------------------------------------------
#
# "See the template patch" is the *template* tracker's phrasing of a need every
# tracker has: show me the pixels being matched. So this is not gated on the
# template tracker — every point gets a patch grid, sized from whatever window
# its own plugin declares (:func:`stable.server.tracker_options.tracker_window`).
# Only the size *control* is template-specific, because only that plugin takes
# its window from a stored setting; for the others the size is reported as the
# tracker's own and the panel says so rather than pretending it is adjustable.


def _template_defaults() -> tuple[int, int]:
    """The template tracker's own defaults, read from the plugin not duplicated."""
    plugin = TemplateTracker()
    return plugin.template_radius, plugin.search_radius


def _point_or_404(project: Project, point_id: str) -> PointSpec:
    """Return a point, or raise the 404 every route in this section wants."""
    try:
        return project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _patch_position(
    project: Project, point: PointSpec, frame: int
) -> tuple[tuple[float, float], str] | None:
    """Where a point is on ``frame``, and whether that is stated or tracked.

    A keyframe wins over the track: on a frame the operator has pinned, the
    patch they want to see is the one at the position they stated, which is also
    what a run would build its template from.

    Returns:
        ``((x, y), kind)`` with ``kind`` in ``{"keyframe", "tracked"}``, or
        ``None`` if the point has neither there.
    """
    for anchor in effective_anchors(point):
        if anchor.frame == frame:
            return (anchor.x, anchor.y), "keyframe"
    track = load_display_track(project, point)
    if track is None or not 0 <= frame < track.n_frames:
        return None
    position = track.position_at(frame)
    return (position, "tracked") if position is not None else None


@router.get("/{point_id}/patches")
def get_point_patches(
    project_id: str,
    point_id: str,
    frame: int | None = None,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Describe every patch this point has, so the panel can draw itself in one call.

    Args:
        frame: The frame being viewed, added to the list as a ``"tracked"``
            entry when the point has a position there and it is not already a
            keyframe. This is what makes the panel show the region being
            followed *now*, not only where it was pinned.

    Returns:
        The effective window radius and where it came from, every setting this
        point's tracker lets the operator change (``settings``, one control per
        row), the pattern/search summary the canvas draws its boxes from, and
        one entry per patch with ``available`` saying whether it can actually be
        built — a keyframe too close to the frame edge is reported rather than
        404-ing later, so the grid can show *why* a thumbnail is missing.
    """
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)

    stored = point_options(workspace.project_dir(project_id), point_id)
    radius, source_attribute = tracker_window(point.tracker, stored)
    settings = tracker_settings(point.tracker, stored)
    search_radius, search_attribute, search_note = tracker_search(point.tracker, stored)
    default_radius, default_search = _template_defaults()

    video = workspace.get_video_source(project_id)

    def entry(position: tuple[float, float], at: int, kind: str) -> dict[str, Any]:
        return {
            "frame": at,
            "x": position[0],
            "y": position[1],
            "kind": kind,
            "available": patch_bounds(position, radius, video.width, video.height) is not None,
        }

    patches = [entry((q.x, q.y), q.frame, "keyframe") for q in effective_anchors(point)]
    current: dict[str, Any] | None = None
    if frame is not None:
        found = _patch_position(project, point, frame)
        if found is not None:
            current = entry(found[0], frame, found[1])

    window_setting = next((s for s in settings if s["role"] == "window"), None)
    # The grid this describes is about to ask for one image per keyframe, and on
    # a cold cache each of those is a seek into the clip. Rendering them now, in
    # the background, is what makes the panel appear filled rather than filling
    # (todo 3, sixth round) — and it covers the keyframes that pre-date the
    # warm-on-create above, plus every server restart, which an eager hook at
    # creation time cannot.
    warm_patches(workspace, project_id, point_id, [(p["frame"], p["x"], p["y"]) for p in patches])
    return {
        "point_id": point_id,
        "tracker": point.tracker,
        "radius": radius,
        "patch_size": radius * 2 + 1,
        # Which constructor argument the radius came from, or null when the
        # tracker has no local window at all (the deep trackers see whole
        # frames) and this is only a preview size.
        "radius_source": source_attribute,
        # True when this tracker's window is the operator's to set — which is
        # every classic-CV plugin, each under its own argument name, not only
        # the template matcher's `template_radius`.
        "adjustable": window_setting is not None,
        # One row per control the panel should draw, in order. The panel is
        # deliberately ignorant of which tracker is selected; this is why.
        "settings": settings,
        # The pattern and search regions in one shape, in source pixels, for
        # the canvas overlay. `search.radius` is null where the algorithm has
        # no search area at all, and `note` says why.
        "window": {
            "radius": radius,
            "size_px": radius * 2 + 1,
            "attribute": source_attribute,
            "adjustable": window_setting is not None,
        },
        "search": {
            "radius": search_radius,
            "size_px": None if search_radius is None else search_radius * 2 + 1,
            "attribute": search_attribute,
            "adjustable": any(s["role"] == "search" for s in settings),
            "note": search_note,
        },
        "template_radius": int(stored.get("template_radius", default_radius)),
        "search_radius": int(stored.get("search_radius", default_search)),
        "defaults": {"template_radius": default_radius, "search_radius": default_search},
        "template_radius_range": list(TEMPLATE_RADIUS_RANGE),
        "search_radius_range": list(SEARCH_RADIUS_RANGE),
        "custom": bool(stored),
        "patches": patches,
        "current": current,
    }


def point_options_schema(tracker_name: str) -> set[str]:
    """Return the constructor arguments ``tracker_name`` accepts."""
    return set(constructor_parameters(tracker_name))


@router.put("/{point_id}/patches")
def set_point_patch_options(
    project_id: str,
    point_id: str,
    body: TemplateOptionsUpdate,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Set (or clear) this point's tracker sizes — pattern window, search area.

    Any argument in :data:`stable.server.tracker_options.SETTINGS` that the
    point's *own* tracker accepts may be sent, under the plugin's own name and
    in the plugin's own units (LK's ``win_size`` is a full side, the radii are
    half-sizes). One that the tracker would not take is a 400 rather than a
    silent no-op: the whole point of storing these per point is that they reach
    the run, and a setting that could not is a bug in the caller.

    A ``null`` restores the plugin's own default, which is what the panel's
    "reset" does — storing the default explicitly would freeze it against any
    later change to the tracker.
    """
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)

    payload = body.model_dump(exclude_unset=True)
    accepted = point_options_schema(point.tracker)
    for key in payload:
        if setting_for(key) is None:
            raise HTTPException(status_code=400, detail=f"{key!r} is not a settable tracker option")
        if key not in accepted:
            raise HTTPException(
                status_code=400,
                detail=f"tracker {point.tracker!r} has no {key!r} setting",
            )
    try:
        save_point_options(workspace.project_dir(project_id), point_id, payload)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_point_patches(project_id, point_id, None, workspace)


# Rendered patches, kept in memory and keyed by everything that can change them.
#
# Scrubbing re-asks for the same patches constantly (back and forth over the
# same frames, one request per point per frame). Rendering them again each time
# is what made the UI stall: see `PATCH_SOURCES` below and the `stale_key`
# helper, which is what lets a cached PNG be trusted rather than merely hoped
# for — an edit, a re-track or a moved anchor changes the key, so a stale patch
# cannot be served.
_PATCH_CACHE: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
_PATCH_CACHE_LOCK = threading.Lock()
_PATCH_CACHE_BYTES = 64 * 1024 * 1024
"""Budget for the rendered-patch cache. A 33px PNG is ~2 KB, so this is tens of
thousands of them — more than a session generates, and bounded regardless."""

PATCH_SOURCES = ("proxy", "original")
"""Where a patch's pixels may come from.

``proxy`` reads the half-resolution JPEG the timeline already scrubs on: a disk
read, no video decode, no decoder lock. ``original`` seeks the source clip for
1:1 pixels. Thumbnails are drawn at 72px whatever they contain, so ``proxy`` is
the default and ``original`` is for the one panel where texture is being judged.
"""


def _cache_get(key: tuple[Any, ...]) -> bytes | None:
    with _PATCH_CACHE_LOCK:
        payload = _PATCH_CACHE.get(key)
        if payload is not None:
            _PATCH_CACHE.move_to_end(key)
        return payload


def _cache_put(key: tuple[Any, ...], payload: bytes) -> None:
    with _PATCH_CACHE_LOCK:
        _PATCH_CACHE[key] = payload
        total = sum(len(v) for v in _PATCH_CACHE.values())
        while total > _PATCH_CACHE_BYTES and len(_PATCH_CACHE) > 1:
            _key, evicted = _PATCH_CACHE.popitem(last=False)
            total -= len(evicted)


def clear_patch_cache() -> None:
    """Drop every cached patch (used by tests and by workspace resets)."""
    with _PATCH_CACHE_LOCK:
        _PATCH_CACHE.clear()


# -- warming the cache (todo 3, sixth round) -----------------------------------
#
# The cache above answers "has this patch been rendered before?", and until now
# the only thing that ever filled it was the first request for a patch — the slow
# one. That is what "patches are slow to load, they seem to come from the video
# file" describes: the panel that judges texture asks for ``source="original"``,
# which is a random-access seek into the full-resolution clip per keyframe, so
# opening a card with eight of them is eight seeks queued behind the decoder's
# single lock, on every fresh server.
#
# So the patch is rendered when the keyframe is *made* (and, for the anchors that
# pre-date this and for every restart, when the grid that will show them is
# listed). One worker, because the cost being avoided is contention for the
# decoder — several warmers would recreate exactly the pile-up they exist to
# prevent — and a de-duplicating set, because listing a grid twice in a row must
# not queue it twice.

_WARM_WORKERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="patch-warm")
_WARM_PENDING: set[tuple[Any, ...]] = set()
_WARM_LOCK = threading.Lock()

_WARM_QUEUE_LIMIT = 64
"""How many patches may be waiting to be warmed at once.

A bound rather than a budget: warming is a head start, and a backlog of them is
a head start on frames nobody is looking at any more. It also keeps process
shutdown prompt — the executor is joined at exit, so an unbounded queue would be
an unbounded wait to quit. Anything turned away here is simply rendered on
demand, exactly as it was before this existed."""


def _warm_patch(
    workspace: Workspace,
    project_id: str,
    key: tuple[Any, ...],
    frame: int,
    position: tuple[float, float],
    radius: int,
    source: str,
) -> None:
    """Render one patch into the cache, or give up quietly.

    Everything about *where* the patch is has already been decided by the caller
    and is passed in. That is not tidiness: this runs on a background thread
    while request threads are writing the project file, and on Windows a reader
    holding it open makes the writer's atomic ``os.replace`` fail outright. A
    warmer that can break the save it was triggered by is worse than no warmer,
    so this one never opens the project.
    """
    try:
        if _cache_get(key) is not None:
            return
        patch = _render_patch(workspace, project_id, frame, position, radius, source)
        if patch is not None:
            _cache_put(key, _encode_png(patch))
    except Exception:  # noqa: BLE001 - a warm cache is an optimisation, never a failure
        return
    finally:
        with _WARM_LOCK:
            _WARM_PENDING.discard(key)


def warm_patches(
    workspace: Workspace,
    project_id: str,
    point_id: str,
    positions: Iterable[tuple[int, float, float]],
) -> None:
    """Render the given keyframes' patches in the background, if it is worth it.

    Args:
        positions: ``(frame, x, y)`` per patch, in source pixels — resolved by
            the caller, which already has them, so the background work is
            nothing but "read pixels, encode, store".

    Skipped entirely for a tracker with no local window: it matches whole frames,
    the UI draws no patch for it (todo 5, sixth round), and rendering images
    nothing will show is the opposite of the point.
    """
    try:
        project = workspace.load_project(project_id)
        stored = point_options(workspace.project_dir(project_id), point_id)
        radius, source_attribute = tracker_window(project.point(point_id).tracker, stored)
    except StableError:
        return
    if source_attribute is None:
        return
    stale = _stale_key(workspace, project_id, point_id)
    for frame, x, y in positions:
        for source in PATCH_SOURCES:
            key = (project_id, point_id, int(frame), radius, source, stale)
            with _WARM_LOCK:
                if key in _WARM_PENDING or len(_WARM_PENDING) >= _WARM_QUEUE_LIMIT:
                    continue
                _WARM_PENDING.add(key)
            _WARM_WORKERS.submit(
                _warm_patch, workspace, project_id, key, int(frame), (x, y), radius, source
            )


def drain_patch_warmup(timeout: float = 30.0) -> bool:
    """Wait until nothing is queued for warming. For tests, which cannot guess.

    Returns:
        Whether the queue emptied within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _WARM_LOCK:
            if not _WARM_PENDING:
                return True
        time.sleep(0.01)
    return False


def _stale_key(workspace: Workspace, project_id: str, point_id: str) -> tuple[float, float]:
    """Return a key that changes whenever a point's patches could have changed.

    The pixels at a frame are a function of *where the point is*, which comes
    from the project file (anchors) and the cached track (everything else). Two
    ``stat`` calls therefore answer "is a cached patch still the right answer?"
    exactly, without the client having to remember to bust anything.
    """

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    directory = workspace.project_dir(project_id)
    return (
        mtime(directory / PROJECT_FILENAME),
        mtime(workspace.tracks_dir(project_id) / f"{point_id}.npz"),
    )


@router.get("/{point_id}/patches/image")
def get_point_patch_image(
    project_id: str,
    point_id: str,
    frame: int,
    request: Request,
    radius: int | None = None,
    source: str = "proxy",
    workspace: Workspace = Depends(_workspace),
) -> Response:
    """Return the patch at one frame as a PNG — a keyframe's, or the tracked one.

    Args:
        frame: Any frame the point has a position on, pinned or tracked.
        radius: Override the effective radius, so the grid can preview a size
            the slider is still being dragged through without saving it first.
        source: ``"proxy"`` (default) reads the pixels from the proxy tier the
            timeline already uses — no video decode, so a scrub does not queue
            full-resolution seeks behind the decoder's lock. ``"original"``
            forces 1:1 source pixels for the panel that is judging texture.

    Raises:
        HTTPException: 404 if the point does not exist or has no position on
            that frame, 400 if the patch would fall outside the frame.
    """
    if source not in PATCH_SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {list(PATCH_SOURCES)}")
    project = workspace.load_project(project_id)
    point = _point_or_404(project, point_id)

    stored = point_options(workspace.project_dir(project_id), point_id)
    effective, _source = tracker_window(point.tracker, stored)
    size = int(radius if radius is not None else effective)
    low, high = TEMPLATE_RADIUS_RANGE
    if not low <= size <= high:
        raise HTTPException(status_code=400, detail=f"radius must lie in [{low}, {high}]")

    key = (project_id, point_id, frame, size, source, _stale_key(workspace, project_id, point_id))
    etag = f'"{abs(hash(key)):x}"'
    headers = {
        # Short and revalidated rather than immutable: the URL does not change
        # when an edit moves the point, but the ETag does, so a scrub back over
        # covered frames costs nothing while an edit is still seen immediately.
        "Cache-Control": "private, max-age=30",
        "ETag": etag,
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    payload = _cache_get(key)
    if payload is None:
        found = _patch_position(project, point, frame)
        if found is None:
            raise HTTPException(
                status_code=404, detail=f"point {point_id!r} has no position on frame {frame}"
            )
        patch = _render_patch(workspace, project_id, frame, found[0], size, source)
        if patch is None:
            raise HTTPException(
                status_code=400,
                detail=f"a {size * 2 + 1}px patch at frame {frame} falls outside the frame",
            )
        payload = _encode_png(patch)
        _cache_put(key, payload)

    return Response(content=payload, media_type="image/png", headers=headers)


def _render_patch(
    workspace: Workspace,
    project_id: str,
    frame: int,
    position: tuple[float, float],
    size: int,
    source: str,
) -> Any:
    """Cut one patch out of the pixels, or return ``None`` if it falls off the frame.

    Shared by the route above and the background warmer, so a warmed patch and a
    requested one are byte-identical — a warmer that rendered anything different
    would be a cache that serves the wrong picture.
    """
    if source == "proxy":
        try:
            store = workspace.get_proxy_store(project_id, "half")
        except StableError:
            store = None
        if store is not None:
            patch = proxy_patch(store, frame, position, size)
            if patch is not None:
                return patch
    # Either the caller asked for source pixels, or the proxy could not supply
    # them (not generated yet, or the scaled patch fell off the edge). The
    # source is always able to answer.
    return template_patch(workspace.get_video_source(project_id), frame, position, size)


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
