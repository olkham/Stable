"""Manual track editing: the corrections layer, retracking, and undo/redo.

Every edit the operator makes on the frame or in the curve editor arrives here
as one typed :class:`~stable.core.corrections.Correction` appended to a point.
Nothing rewrites a track: the raw ``.npz`` is untouched, and what the overlay,
the solver and the exports see is always
``apply_corrections(raw, point.corrections)``. That is what makes the whole
editing vocabulary undoable by swapping a small JSON list (see
:mod:`stable.server.history`).

Endpoints, all under ``/projects/{project_id}``:

============================================ ===================================
``GET/POST  /points/{id}/corrections``       list / append an edit
``PATCH/DELETE /points/{id}/corrections/{c}``  toggle / remove one
``POST      /points/{id}/retrack``           re-track from a corrected position
``GET       /edits/history``                 undo/redo availability
``POST      /edits/undo`` / ``/edits/redo``  step the corrections stack
============================================ ===================================
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.corrections import (
    Correction,
    OffsetCorrection,
    ReplaceCorrection,
    apply_corrections,
    correction_from_dict,
)
from ...core.errors import ProjectError, StableError
from ...core.project import PointSpec, Project
from .. import motion as motion_glue
from ..history import HISTORY_FILENAME, EditHistory
from ..jobs import JobManager, get_job_manager
from ..schemas import CorrectionIn, CorrectionUpdate, RetrackRequest
from ..serialize import point_summary
from ..storage import Workspace, get_workspace
from ..tracking import load_display_track, retrack_resolver, run_point_retrack

__all__ = ["router", "history_for"]

router = APIRouter(prefix="/projects/{project_id}", tags=["edits"])


def _workspace() -> Workspace:
    return get_workspace()


def _jobs() -> JobManager:
    return get_job_manager()


def history_for(workspace: Workspace, project_id: str) -> EditHistory:
    """Return the project's undo/redo stack, loaded from disk.

    A fresh instance per request is deliberate: the stack only ever changes
    under the workspace lock, and reading a few kilobytes of JSON is cheaper
    than keeping process state that a second browser tab could disagree with.
    """
    return EditHistory(workspace.project_dir(project_id) / HISTORY_FILENAME)


def _describe(project: Project, point: PointSpec, history: EditHistory) -> dict[str, Any]:
    """Return the payload every mutating edit endpoint answers with."""
    return {
        "point": point_summary(project, point),
        "corrections": [c.to_dict() for c in point.corrections],
        "history": history.to_dict(),
    }


def _edit(
    workspace: Workspace,
    project_id: str,
    point_id: str,
    label: str,
    mutate: Any,
) -> dict[str, Any]:
    """Record an undo point, apply ``mutate``, persist, and return the payload.

    ``mutate`` is called as ``mutate(project, point)`` — the project is handed
    over too because promoting an edit to a ground-truth anchor needs to read
    the point's current track, which only the project knows how to find.

    The snapshot is taken inside the same read-modify-write as the mutation, so
    what undo restores is exactly the state the edit replaced — and a mutation
    that raises leaves both the project and the history file untouched.
    """
    history = history_for(workspace, project_id)

    def _run(project: Project) -> None:
        history.record(project, label)
        mutate(project, project.point(point_id))

    try:
        project = workspace.update_project(project_id, _run)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    history.save()
    motion_glue.clear_plan_cache(project_id)
    return _describe(project, project.point(point_id), history)


# -- corrections ---------------------------------------------------------------


@router.get("/points/{point_id}/corrections")
def list_corrections(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Return a point's correction list and the project's undo/redo status."""
    project = workspace.load_project(project_id)
    try:
        point = project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _describe(project, point, history_for(workspace, project_id))


def _promotion_for(
    project: Project, point: PointSpec, correction: Correction
) -> ReplaceCorrection | None:
    """Return the ground-truth anchor a manual correction should be promoted to.

    Todo 11: *every* manual correction that says where the feature is becomes a
    keyframe, so a later re-track plans through it instead of overwriting it.

    ``replace`` and ``retrack`` already carry an absolute position, so they are
    ground truth as they stand (see
    :data:`stable.core.corrections.GROUND_TRUTH_KINDS`) and need no companion. A
    drag arrives as an ``offset``, which is *relative* — replaying it on top of
    a freshly re-tracked path would nudge an already-correct position a second
    time. So the resulting absolute position is captured here, once, as a
    companion ``replace`` at the same frame: the drag keeps blending its
    neighbours, and the frame the operator actually pointed at is pinned
    exactly, before and after any re-track.

    Returns:
        The companion correction to append, or ``None`` when the edit states no
        position (``gap_fill``, ``delete_range``) or the track has no sample at
        that frame to offset.
    """
    if not isinstance(correction, OffsetCorrection) or not correction.enabled:
        return None
    track = load_display_track(project, point)
    if track is None:
        return None
    edited = apply_corrections(track, [correction], resolve_retrack=retrack_resolver(project))
    position = (
        edited.position_at(correction.frame) if 0 <= correction.frame < edited.n_frames else None
    )
    if position is None:
        return None
    return ReplaceCorrection(frame=correction.frame, x=position[0], y=position[1])


@router.post("/points/{point_id}/corrections")
def add_correction(
    project_id: str,
    point_id: str,
    body: CorrectionIn,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Append one typed correction to a point, promoting it to ground truth.

    The body is the correction's own project-file form — ``kind`` plus that
    type's fields — so the HTTP schema is the storage schema and there is no
    second definition of an edit to keep in sync. An on-frame drag becomes
    ``{"kind": "offset", ...}`` or ``{"kind": "replace", ...}``; the curve
    editor's repair buttons become ``gap_fill`` and ``delete_range``.

    A correction that states a position is also promoted to an anchor keyframe
    (see :func:`_promotion_for`), which is what stops the next track or retrack
    from overwriting the operator's work. Promotion happens inside the same
    read-modify-write, so it is part of the *same* undo step — and because it
    lives entirely in the corrections layer, undo un-promotes it for free.

    Raises:
        HTTPException: 400 if the correction is malformed, 404 if the point or
            project does not exist.
    """
    try:
        correction = correction_from_dict(body.payload())
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if correction.kind == "retrack":
        raise HTTPException(
            status_code=400,
            detail="a retrack has to run the tracker; POST to /points/{id}/retrack instead",
        )

    def _mutate(project: Project, point: PointSpec) -> None:
        promoted = _promotion_for(project, point, correction)
        appended = [correction] if promoted is None else [correction, promoted]
        point.corrections = [*point.corrections, *appended]

    return _edit(
        workspace,
        project_id,
        point_id,
        _label(correction, point_id),
        _mutate,
    )


def _label(correction: Correction, point_id: str) -> str:
    """Return the short human description an undo step is shown under."""
    frame = getattr(correction, "frame", None)
    if frame is not None:
        return f"{correction.kind} {point_id} @ {frame}"
    start = getattr(correction, "start", None)
    end = getattr(correction, "end", None)
    if start is not None:
        return f"{correction.kind} {point_id} [{start}-{end}]"
    return f"{correction.kind} {point_id}"  # pragma: no cover - every kind has one or the other


@router.patch("/points/{point_id}/corrections/{correction_id}")
def update_correction(
    project_id: str,
    point_id: str,
    correction_id: str,
    body: CorrectionUpdate,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Enable or disable one correction without deleting it.

    "Toggle my fix off to compare" is a first-class operation — the correction
    stays in the project so the comparison is reversible.

    Raises:
        HTTPException: 404 if the correction does not exist on that point.
    """

    def _mutate(_project: Project, point: PointSpec) -> None:
        found = False
        updated = []
        for correction in point.corrections:
            if correction.id == correction_id and body.enabled is not None:
                found = True
                data = correction.to_dict()
                data["enabled"] = bool(body.enabled)
                updated.append(correction_from_dict(data))
            else:
                found = found or correction.id == correction_id
                updated.append(correction)
        if not found:
            raise ProjectError(f"no correction {correction_id!r} on point {point_id!r}")
        point.corrections = updated

    action = "enable" if body.enabled else "disable"
    return _edit(workspace, project_id, point_id, f"{action} edit on {point_id}", _mutate)


@router.delete("/points/{point_id}/corrections/{correction_id}")
def delete_correction(
    project_id: str,
    point_id: str,
    correction_id: str,
    workspace: Workspace = Depends(_workspace),
) -> dict[str, Any]:
    """Remove one correction from a point.

    Raises:
        HTTPException: 404 if the correction does not exist on that point.
    """

    def _mutate(_project: Project, point: PointSpec) -> None:
        remaining = [c for c in point.corrections if c.id != correction_id]
        if len(remaining) == len(point.corrections):
            raise ProjectError(f"no correction {correction_id!r} on point {point_id!r}")
        point.corrections = remaining

    return _edit(workspace, project_id, point_id, f"remove edit from {point_id}", _mutate)


@router.delete("/points/{point_id}/corrections")
def clear_corrections(
    project_id: str, point_id: str, workspace: Workspace = Depends(_workspace)
) -> dict[str, Any]:
    """Drop every correction on a point, reverting it to the raw merged track."""

    def _mutate(_project: Project, point: PointSpec) -> None:
        point.corrections = []

    return _edit(workspace, project_id, point_id, f"revert {point_id} to raw", _mutate)


# -- retrack from here ----------------------------------------------------------


@router.post("/points/{point_id}/retrack")
def retrack_point(
    project_id: str,
    point_id: str,
    body: RetrackRequest,
    workspace: Workspace = Depends(_workspace),
    jobs: JobManager = Depends(_jobs),
) -> dict[str, str]:
    """Submit a job re-tracking a point from a corrected position.

    Unlike adding an anchor — which re-runs the whole multi-anchor merge — this
    repairs one direction from one frame and splices the result in as a
    correction, leaving the merged baseline intact underneath.

    The run is **bounded**: ``stop_at_keyframe`` takes it only as far as the
    neighbouring keyframe (todo 12), ``max_frames`` only as far as the operator
    asked (todo 20), and any keyframe it crosses and disagrees with stops it
    dead (todo 13). Omitting ``x``/``y`` starts it from wherever the point
    already is at that frame.

    Raises:
        HTTPException: 404 if the point does not exist.
    """
    project = workspace.load_project(project_id)
    try:
        project.point(point_id)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    history = history_for(workspace, project_id)
    history.record(project, f"retrack {point_id} @ {body.frame}")
    history.save()

    def _run(progress: Any) -> dict[str, Any]:
        return run_point_retrack(
            workspace,
            project_id,
            point_id,
            frame=body.frame,
            x=body.x,
            y=body.y,
            direction=body.direction,
            tracker=body.tracker,
            device=body.device,
            progress=progress,
            max_frames=body.max_frames,
            stop_at_keyframe=body.stop_at_keyframe,
            anchor_tolerance_px=body.anchor_tolerance_px,
        )

    job = jobs.submit(project_id, "retrack", _run, target=point_id)
    return {"job_id": job.id}


# -- undo / redo -----------------------------------------------------------------


@router.get("/edits/history")
def get_history(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Return whether undo/redo are available, and what they would do."""
    workspace.load_project(project_id)  # 404s if the project does not exist
    return history_for(workspace, project_id).to_dict()


def _step(workspace: Workspace, project_id: str, forward: bool) -> dict[str, Any]:
    """Apply one undo or redo step and return the resulting project state."""
    history = history_for(workspace, project_id)
    applied: list[str | None] = []

    def _run(project: Project) -> None:
        applied.append(history.redo(project) if forward else history.undo(project))

    try:
        project = workspace.update_project(project_id, _run)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    history.save()
    motion_glue.clear_plan_cache(project_id)
    return {
        "applied": applied[0],
        "history": history.to_dict(),
        "points": [point_summary(project, p) for p in project.points],
    }


@router.post("/edits/undo")
def undo(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Restore the corrections layer to its state before the last edit.

    ``applied`` is ``null`` when there was nothing to undo; the project is then
    left exactly as it was rather than being rewritten with identical content.
    """
    return _step(workspace, project_id, forward=False)


@router.post("/edits/redo")
def redo(project_id: str, workspace: Workspace = Depends(_workspace)) -> dict[str, Any]:
    """Re-apply the most recently undone edit."""
    return _step(workspace, project_id, forward=True)
