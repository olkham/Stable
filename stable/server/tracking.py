"""Runs one point's tracking job: plan anchor runs, execute, merge, persist.

This is the server-side glue between the project file (a point's anchors and
tracker choice) and the Phase 0-2 tracking stack
(:mod:`stable.trackers.registry`, :mod:`stable.trackers.merge`). Nothing here
is a stub — it is the real multi-anchor tracking pipeline PLAN.md describes,
just wired to run as a background job and to persist its result where the
project file expects it (``points[].raw``).

Phase 5 adds the second kind of run: *retrack from here*. Where a full track
starts from every anchor and merges, a retrack starts from one corrected
position the user just placed and covers only the frames in one direction from
it. Its output is cached beside the raw track and referenced by the
:class:`~stable.core.corrections.RetrackCorrection` that splices it in, so the
raw multi-anchor baseline is never overwritten — reopening the project (or
disabling the correction) brings the original back.

Anchors are ground truth, everywhere
------------------------------------
Both kinds of run go through :func:`effective_anchors`, which is the single
answer to "what has the operator told us about this point": its anchor
keyframes *plus* every manual correction that states a position (see
:func:`stable.core.corrections.ground_truth_anchors`). That matters twice.

#. **Planning.** A full track plans its runs through every one of them, so the
   merge passes exactly through the operator's corrections as well as their
   pins — a manual fix improves the next track instead of being overwritten by
   it.
#. **Guarding.** A run that crosses an anchor and disagrees with it is stopped
   there by :func:`stable.trackers.merge.enforce_anchor_agreement` instead of
   drifting past, and the disagreement is reported (frame + pixels) all the way
   out to the job result, the sidecar report and the timeline.

Before this, a retrack simply spliced over whatever anchors lay in its
direction: the user's keyframes were silently discarded by the very run that was
supposed to respect them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..core.corrections import (
    DIRECTIONS,
    RetrackCorrection,
    RetrackResolver,
    apply_corrections,
    ground_truth_anchors,
)
from ..core.errors import ProjectError, StableError
from ..core.project import PointSpec, Project
from ..core.types import PointTrack, Query
from ..trackers.base import Direction
from ..trackers.merge import (
    DEFAULT_ANCHOR_TOLERANCE_PX,
    AnchorViolation,
    MergeResult,
    TrackRun,
    enforce_anchor_agreement,
    merge_anchor_runs,
    plan_anchor_runs,
)
from ..trackers.registry import create_tracker
from .storage import Workspace

__all__ = [
    "run_point_tracking",
    "run_point_retrack",
    "effective_anchors",
    "bounded_frame_range",
    "report_path_for",
    "retrack_path_for",
    "violations_path_for",
    "load_report",
    "load_violations",
    "load_anchor_violations",
    "load_display_track",
    "load_raw_track",
    "retrack_resolver",
]


def report_path_for(npz_path: Path) -> Path:
    """Return the sidecar diagnostics path next to a track's ``.npz`` file."""
    return npz_path.with_suffix("").with_suffix(".report.json")


def retrack_path_for(tracks_dir: Path, point_id: str, correction_id: str) -> Path:
    """Return where one retrack run's ``.npz`` is cached, beside the raw track."""
    return tracks_dir / f"{point_id}.retrack-{correction_id}.npz"


def violations_path_for(npz_path: Path) -> Path:
    """Return the anchor-disagreement sidecar for any run's ``.npz``.

    Deliberately not :func:`report_path_for`: that one strips the *whole* suffix
    chain, so a retrack's ``p1.retrack-ab12.npz`` would resolve to the raw
    track's own ``p1.report.json`` and the two would overwrite each other.
    """
    return npz_path.with_suffix(".violations.json")


def load_violations(path: Path) -> list[dict[str, Any]]:
    """Load a violations sidecar, or return an empty list if there is none."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
        return []
    return [dict(entry) for entry in data] if isinstance(data, list) else []


def effective_anchors(point: PointSpec) -> list[Query]:
    """Return every position the operator has stated for a point, by frame.

    The union of the point's anchor keyframes and the anchors its enabled
    manual corrections promote to (todo 11). A correction wins over a pin on the
    same frame: it is the later statement of where the feature is.

    Args:
        point: The point whose ground truth is wanted.

    Returns:
        Anchors sorted by frame, ready for
        :func:`~stable.trackers.merge.plan_anchor_runs`.
    """
    by_frame = {int(q.frame): q for q in point.sorted_queries()}
    for query in ground_truth_anchors(point.corrections):
        by_frame[int(query.frame)] = query
    return [by_frame[frame] for frame in sorted(by_frame)]


def bounded_frame_range(
    direction: str,
    frame: int,
    n_frames: int,
    *,
    anchors: list[Query] | None = None,
    max_frames: int | None = None,
    stop_at_keyframe: bool = False,
) -> tuple[int, int]:
    """Return the half-open frame range a directional run should cover.

    A run defaults to "everything in that direction". Two options bound it, and
    the tighter of the two wins:

    * ``stop_at_keyframe`` — stop at the next keyframe forwards / previous
      keyframe backwards (todo 12). With no keyframe that way it is a no-op.
    * ``max_frames`` — cover at most this many frames beyond ``frame``
      (todo 20), so a repair costs seconds rather than the whole clip.

    Args:
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
        frame: Absolute frame the run starts from.
        n_frames: Clip length ``T``.
        anchors: The point's ground-truth anchors, needed by ``stop_at_keyframe``.
        max_frames: Maximum number of frames to cover beyond ``frame``.
        stop_at_keyframe: Stop at the neighbouring keyframe.

    Returns:
        A half-open ``(start, end)`` absolute range that always contains
        ``frame``.

    Raises:
        ProjectError: If ``max_frames`` is not positive.
    """
    if max_frames is not None and max_frames <= 0:
        raise ProjectError(f"max_frames must be positive, got {max_frames}")
    frames = [int(q.frame) for q in (anchors or []) if int(q.frame) != int(frame)]

    reach_fwd = n_frames - 1 - frame
    reach_bwd = frame
    if stop_at_keyframe:
        after = [f for f in frames if f > frame]
        before = [f for f in frames if f < frame]
        if after:
            reach_fwd = min(reach_fwd, min(after) - frame)
        if before:
            reach_bwd = min(reach_bwd, frame - max(before))
    if max_frames is not None:
        reach_fwd = min(reach_fwd, max_frames)
        reach_bwd = min(reach_bwd, max_frames)

    start = frame - reach_bwd if direction in ("bwd", "bidir") else frame
    end = frame + reach_fwd + 1 if direction in ("fwd", "bidir") else frame + 1
    return (max(0, start), min(n_frames, end))


def _save_report(
    path: Path, result: MergeResult, violations: list[AnchorViolation] | None = None
) -> None:
    """Write a :class:`~stable.trackers.merge.MergeReport` as JSON.

    ``anchor_violations`` (and the ``flagged_anchors`` frame list derived from
    it) ride along in the same sidecar so reopening a project keeps the "this
    keyframe was never actually reached" signal the timeline draws red — it is a
    property of the run, not something recomputable from the merged track, whose
    anchor passthrough is exact by construction.
    """
    report = result.report
    path.parent.mkdir(parents=True, exist_ok=True)
    disagreement = [None if math.isnan(v) else float(v) for v in report.disagreement.tolist()]
    breaches = violations or []
    payload = {
        "contributions": [int(v) for v in report.contributions.tolist()],
        "disagreement": disagreement,
        "anchor_frames": list(report.anchor_frames),
        "anchor_violations": [v.to_dict() for v in breaches],
        "flagged_anchors": sorted({v.frame for v in breaches}),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_report(path: Path) -> dict[str, Any] | None:
    """Load a report written by :func:`_save_report`, or ``None`` if absent."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_point_tracking(
    workspace: Workspace,
    project_id: str,
    point_id: str,
    *,
    tracker: str | None,
    direction: str,
    device: str | None,
    progress: Any,
    anchor_tolerance_px: float = DEFAULT_ANCHOR_TOLERANCE_PX,
) -> dict[str, Any]:
    """Track one point's anchors, merge the runs, and persist the result.

    Every anchor the operator has stated — pins *and* promoted manual
    corrections, see :func:`effective_anchors` — is planned through, so the
    merged track passes exactly through all of them. Runs are guarded: one that
    reaches a keyframe it disagrees with is stopped there rather than allowed to
    drift on, and that keyframe comes back flagged.

    Args:
        workspace: The server's project storage.
        project_id: The owning project.
        point_id: The point to track.
        tracker: Tracker plugin override; ``None`` uses the point's own.
        direction: Accepted for API symmetry and ignored here. A full track is
            *by definition* every anchor forwards and backwards:
            :func:`~stable.trackers.merge.plan_anchor_runs` fixes the direction
            of each run from the anchor pair it spans, and dropping half of them
            would leave the clip only partly covered. A run in one direction
            from one frame is :func:`run_point_retrack`.
        device: Compute device override; ``None`` selects automatically.
        progress: A ``(fraction, message)`` progress callback.
        anchor_tolerance_px: How far a run may land from a keyframe before it is
            stopped there.

    Returns:
        A JSON-safe summary: ``point_id``, ``n_frames``, ``anchor_frames``,
        ``mean_confidence``, ``worst_frames`` (the merge's "go fix this" list),
        ``anchor_violations`` and ``flagged_anchors`` (todos 13 and 15).

    Raises:
        ProjectError: If the point has no anchors.
        stable.core.errors.TrackerError: If the tracker plugin fails.
    """
    project = workspace.load_project(project_id)
    point = project.point(point_id)
    video = workspace.get_video_source(project_id)
    # A promoted correction can outlive the clip it was made against (a project
    # re-pointed at a shorter file), and merging rejects an out-of-range anchor.
    anchors = [q for q in effective_anchors(point) if 0 <= q.frame < video.frame_count]
    if not anchors:
        raise ProjectError(f"point {point_id!r} has no anchor keyframes to track from")

    frame_range = (0, video.frame_count)
    run_specs = plan_anchor_runs(anchors, frame_range)
    tracker_name = tracker or point.tracker

    plugin = create_tracker(tracker_name, device=device)
    track_runs: list[TrackRun] = []
    violations: list[AnchorViolation] = []
    try:
        n_runs = len(run_specs)
        for index, spec in enumerate(run_specs):

            def sub_progress(
                fraction: float, message: str = "", _index: int = index, _n: int = n_runs
            ) -> None:
                progress((_index + fraction) / _n, message or f"run {_index + 1}/{_n}")

            # A run always has exactly one anchor (its own); tracker.track()
            # still takes a list, matching the single-run TrackerPlugin contract.
            result = plugin.track(
                video, [spec.anchor], spec.frame_range, spec.direction, progress=sub_progress
            )
            guarded, breaches = enforce_anchor_agreement(
                result.point(0),
                anchors,
                origin=spec.anchor.frame,
                direction=spec.direction,
                tolerance_px=anchor_tolerance_px,
            )
            violations.extend(breaches)
            track_runs.append(TrackRun.from_spec(spec, guarded))
    finally:
        plugin.release()

    merge_result = merge_anchor_runs(track_runs, anchors, video.frame_count)

    tracks_dir = workspace.tracks_dir(project_id)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = tracks_dir / f"{point_id}.npz"
    merge_result.track.save_npz(npz_path)
    _save_report(report_path_for(npz_path), merge_result, violations)

    def _mutate(p: Project) -> None:
        spec = p.point(point_id)
        spec.raw = p.relative_path(str(npz_path))
        if tracker:
            spec.tracker = tracker

    workspace.update_project(project_id, _mutate)

    conf = merge_result.track.confidence
    progress(1.0, "done")
    return {
        "point_id": point_id,
        "n_frames": video.frame_count,
        "anchor_frames": list(merge_result.report.anchor_frames),
        "mean_confidence": float(conf.mean()) if conf.size else 0.0,
        "worst_frames": merge_result.report.worst_frames(),
        "anchor_violations": [v.to_dict() for v in violations],
        "flagged_anchors": sorted({v.frame for v in violations}),
    }


def run_point_retrack(
    workspace: Workspace,
    project_id: str,
    point_id: str,
    *,
    frame: int,
    x: float | None = None,
    y: float | None = None,
    direction: str,
    tracker: str | None,
    device: str | None,
    progress: Any,
    max_frames: int | None = None,
    stop_at_keyframe: bool = False,
    anchor_tolerance_px: float = DEFAULT_ANCHOR_TOLERANCE_PX,
) -> dict[str, Any]:
    """Re-track a point from a corrected position and append the correction.

    The run is executed once, cached on disk, and referenced by a new
    :class:`~stable.core.corrections.RetrackCorrection` appended to the point —
    so it is undoable like every other edit and costs nothing to re-apply when
    the project is reopened.

    The run is bounded and guarded rather than open-ended:

    * ``stop_at_keyframe`` / ``max_frames`` cap how far it goes (todos 12, 20);
    * any keyframe it *does* reach and disagree with stops it there (todo 13),
      so a retrack can never silently overwrite the operator's own anchors.

    Because a correction only ever splices the frames its cached run actually
    covered, bounding the run bounds the edit — no extra range bookkeeping is
    needed in :func:`~stable.core.corrections.apply_corrections`.

    Args:
        workspace: The server's project storage.
        project_id: The owning project.
        point_id: The point to repair.
        frame: Frame the retrack starts from.
        x: Corrected horizontal position, used as the run's query. ``None``
            reuses the point's current position at ``frame`` — which is what
            "track on from here to the next keyframe" means.
        y: Corrected vertical position; see ``x``.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
        tracker: Tracker plugin override; ``None`` uses the point's own.
        device: Compute device override; ``None`` selects automatically.
        progress: A ``(fraction, message)`` progress callback.
        max_frames: Cover at most this many frames beyond ``frame``.
        stop_at_keyframe: Stop at the next/previous keyframe.
        anchor_tolerance_px: How far the run may land from a keyframe it crosses
            before it is stopped there.

    Returns:
        A JSON-safe summary: the new ``correction_id``, the frames it covers,
        the mean confidence of the spliced span, and the anchor disagreements
        (if any) that cut it short.

    Raises:
        ProjectError: If the frame lies outside the clip, the direction is
            unknown, ``max_frames`` is not positive, or no position is known at
            ``frame`` and none was given.
        stable.core.errors.TrackerError: If the tracker plugin fails.
    """
    project = workspace.load_project(project_id)
    point = project.point(point_id)
    video = workspace.get_video_source(project_id)
    n_frames = video.frame_count
    if not 0 <= frame < n_frames:
        raise ProjectError(f"frame {frame} is outside the clip (T={n_frames})")
    if direction not in DIRECTIONS:
        raise ProjectError(f"unknown direction {direction!r}; expected {DIRECTIONS}")

    anchors = effective_anchors(point)
    position = _resolve_start_position(project, point, frame, x, y)
    query = Query(frame=frame, x=position[0], y=position[1])
    run_range = bounded_frame_range(
        direction,
        frame,
        n_frames,
        anchors=anchors,
        max_frames=max_frames,
        stop_at_keyframe=stop_at_keyframe,
    )

    plugin_name = tracker or point.tracker
    correction = RetrackCorrection(
        frame=frame, x=query.x, y=query.y, direction=direction, tracker=plugin_name
    )
    plugin = create_tracker(plugin_name, device=device)
    try:
        if direction == "bidir":
            # A bidirectional retrack is two single-direction runs (the plugin
            # contract is single-run only); the halves cover disjoint spans, so
            # they combine by simply taking whichever one covered each frame.
            forward = plugin.track(
                video, [query], (frame, run_range[1]), "fwd", progress=_scaled(progress, 0, 2)
            ).point(0)
            backward = plugin.track(
                video, [query], (run_range[0], frame + 1), "bwd", progress=_scaled(progress, 1, 2)
            ).point(0)
            run = _combine_halves(forward, backward, frame)
        else:
            run = plugin.track(
                video,
                [query],
                run_range,
                cast(Direction, direction),
                progress=progress,
            ).point(0)
    finally:
        plugin.release()

    run, violations = enforce_anchor_agreement(
        run,
        anchors,
        origin=frame,
        direction=direction,
        tolerance_px=anchor_tolerance_px,
    )

    tracks_dir = workspace.tracks_dir(project_id)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = retrack_path_for(tracks_dir, point_id, correction.id)
    run.save_npz(npz_path)
    violations_path_for(npz_path).write_text(
        json.dumps([v.to_dict() for v in violations]), encoding="utf-8"
    )

    def _mutate(proj: Project) -> None:
        spec = proj.point(point_id)
        spec.corrections = [
            *spec.corrections,
            RetrackCorrection(
                id=correction.id,
                frame=frame,
                x=query.x,
                y=query.y,
                direction=direction,
                tracker=correction.tracker,
                source=proj.relative_path(str(npz_path)),
            ),
        ]

    workspace.update_project(project_id, _mutate)

    covered = run.valid
    covered_frames = np.nonzero(covered)[0]
    progress(1.0, "done")
    return {
        "point_id": point_id,
        "correction_id": correction.id,
        "frame": frame,
        "x": query.x,
        "y": query.y,
        "direction": direction,
        "tracker": correction.tracker,
        "frame_range": list(run_range),
        "covered_frames": int(covered.sum()),
        "covered_range": (
            [int(covered_frames[0]), int(covered_frames[-1])] if covered_frames.size else None
        ),
        "mean_confidence": float(run.confidence[covered].mean()) if covered.any() else 0.0,
        "anchor_violations": [v.to_dict() for v in violations],
        "flagged_anchors": sorted({v.frame for v in violations}),
        "stopped_at": violations[0].frame if violations else None,
    }


def _resolve_start_position(
    project: Project,
    point: PointSpec,
    frame: int,
    x: float | None,
    y: float | None,
) -> tuple[float, float]:
    """Return where a retrack should start from, filling in ``x``/``y`` if absent.

    Omitting the position is how "carry on tracking from here" is expressed: the
    run starts from wherever the point already is at that frame — an anchor if
    one sits there, otherwise the corrected track.

    Raises:
        ProjectError: If no position is known at ``frame``.
    """
    if x is not None and y is not None:
        return (float(x), float(y))
    for query in effective_anchors(point):
        if int(query.frame) == int(frame):
            return (float(query.x), float(query.y))
    track = load_display_track(project, point)
    known = track.position_at(frame) if track is not None else None
    if known is None:
        raise ProjectError(
            f"point {point.id!r} has no position at frame {frame} to retrack from; "
            "place the point there first or pass x/y"
        )
    return known


def _scaled(progress: Any, index: int, total: int) -> Any:
    """Return a progress callback reporting into slice ``index`` of ``total``."""

    def report(fraction: float, message: str = "") -> None:
        progress((index + fraction) / total, message or f"run {index + 1}/{total}")

    return report


def _combine_halves(forward: PointTrack, backward: PointTrack, frame: int) -> PointTrack:
    """Merge the two halves of a bidirectional retrack into one clip-length track."""
    points = forward.points.copy()
    confidence = forward.confidence.copy()
    visible = forward.visible.copy()
    take = backward.valid.copy()
    take[frame + 1 :] = False  # the forward half owns everything after the query
    points[take] = backward.points[take]
    confidence[take] = backward.confidence[take]
    visible[take] = backward.visible[take]
    return PointTrack(points=points, confidence=confidence, visible=visible)


def retrack_resolver(project: Project) -> RetrackResolver:
    """Return a resolver turning a retrack correction into its cached run.

    See :data:`stable.core.corrections.RetrackResolver`: this is the seam that
    keeps :func:`~stable.core.corrections.apply_corrections` a pure array
    function with no knowledge of the filesystem.
    """

    def resolve(correction: RetrackCorrection) -> PointTrack | None:
        if not correction.source:
            return None
        path = project.resolve_path(correction.source)
        if not path or not Path(path).is_file():
            return None
        try:
            return PointTrack.load_npz(path)
        except StableError:  # pragma: no cover - corrupt cache
            return None

    return resolve


def load_display_track(project: Project, point: PointSpec) -> PointTrack | None:
    """Return a point's displayable track: raw plus its correction layer.

    This is the single funnel every consumer goes through — the overlay, the
    heat strip, the solver and the data export all see the *corrected* track, so
    a repair the user makes is immediately what the rest of the tool works from.

    Returns:
        The corrected :class:`~stable.core.types.PointTrack`, or ``None`` if the
        point has not been tracked yet.
    """
    raw = load_raw_track(project, point)
    if raw is None:
        return None
    try:
        return apply_corrections(raw, point.corrections, resolve_retrack=retrack_resolver(project))
    except StableError:
        # A correction this build cannot apply must not hide the track itself.
        return raw


def load_anchor_violations(project: Project, point: PointSpec) -> list[dict[str, Any]]:
    """Return every keyframe disagreement recorded for a point, in frame order.

    A track can fail to reach an operator's keyframe in two places: the
    multi-anchor run behind the raw track, and any enabled retrack correction
    layered on top. Both are collected here so the timeline paints exactly the
    keyframes that are still unreached in what the user is looking at — a
    retrack whose correction is toggled off stops flagging anything.
    """
    found: list[dict[str, Any]] = []
    if point.raw:
        raw_path = project.resolve_path(point.raw)
        if raw_path:
            report = load_report(report_path_for(Path(raw_path))) or {}
            found.extend(dict(entry) for entry in report.get("anchor_violations", []))
    for correction in point.corrections:
        if not correction.enabled or not isinstance(correction, RetrackCorrection):
            continue
        source = project.resolve_path(correction.source) if correction.source else None
        if source:
            found.extend(load_violations(violations_path_for(Path(source))))
    return sorted(found, key=lambda entry: int(entry.get("frame", 0)))


def load_raw_track(project: Project, point: PointSpec) -> PointTrack | None:
    """Return a point's untouched merged track, with no corrections applied.

    The curve editor's "before" ghost and the correction panel's revert preview
    are drawn from this.
    """
    if not point.raw:
        return None
    path = project.resolve_path(point.raw)
    if not path or not Path(path).is_file():
        return None
    return PointTrack.load_npz(path)
