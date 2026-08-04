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
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from ..core.corrections import DIRECTIONS, RetrackCorrection, RetrackResolver, apply_corrections
from ..core.errors import ProjectError, StableError
from ..core.project import PointSpec, Project
from ..core.types import PointTrack, Query
from ..trackers.base import Direction
from ..trackers.merge import MergeResult, TrackRun, merge_anchor_runs, plan_anchor_runs
from ..trackers.registry import create_tracker
from .storage import Workspace

__all__ = [
    "run_point_tracking",
    "run_point_retrack",
    "report_path_for",
    "retrack_path_for",
    "load_report",
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


def _save_report(path: Path, result: MergeResult) -> None:
    """Write a :class:`~stable.trackers.merge.MergeReport` as JSON."""
    report = result.report
    path.parent.mkdir(parents=True, exist_ok=True)
    disagreement = [None if math.isnan(v) else float(v) for v in report.disagreement.tolist()]
    payload = {
        "contributions": [int(v) for v in report.contributions.tolist()],
        "disagreement": disagreement,
        "anchor_frames": list(report.anchor_frames),
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
) -> dict[str, Any]:
    """Track one point's anchors, merge the runs, and persist the result.

    Args:
        workspace: The server's project storage.
        project_id: The owning project.
        point_id: The point to track.
        tracker: Tracker plugin override; ``None`` uses the point's own.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"`` — forwarded to every
            planned run (``plan_anchor_runs`` already fixes fwd/bwd per run
            around each anchor pair; this only affects single-anchor spans).
        device: Compute device override; ``None`` selects automatically.
        progress: A ``(fraction, message)`` progress callback.

    Returns:
        A JSON-safe summary: ``point_id``, ``n_frames``, ``anchor_frames``,
        ``mean_confidence``, ``worst_frames`` (the merge's "go fix this" list).

    Raises:
        ProjectError: If the point has no anchors.
        stable.core.errors.TrackerError: If the tracker plugin fails.
    """
    project = workspace.load_project(project_id)
    point = project.point(point_id)
    anchors = point.sorted_queries()
    if not anchors:
        raise ProjectError(f"point {point_id!r} has no anchor keyframes to track from")

    video = workspace.get_video_source(project_id)
    frame_range = (0, video.frame_count)
    run_specs = plan_anchor_runs(anchors, frame_range)
    tracker_name = tracker or point.tracker

    plugin = create_tracker(tracker_name, device=device)
    track_runs: list[TrackRun] = []
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
            track_runs.append(TrackRun.from_spec(spec, result.point(0)))
    finally:
        plugin.release()

    merge_result = merge_anchor_runs(track_runs, anchors, video.frame_count)

    tracks_dir = workspace.tracks_dir(project_id)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = tracks_dir / f"{point_id}.npz"
    merge_result.track.save_npz(npz_path)
    _save_report(report_path_for(npz_path), merge_result)

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
    }


def _retrack_frame_range(direction: str, frame: int, n_frames: int) -> tuple[int, int]:
    """Return the half-open frame range one retrack direction has to cover."""
    if direction == "fwd":
        return (frame, n_frames)
    if direction == "bwd":
        return (0, frame + 1)
    return (0, n_frames)


def run_point_retrack(
    workspace: Workspace,
    project_id: str,
    point_id: str,
    *,
    frame: int,
    x: float,
    y: float,
    direction: str,
    tracker: str | None,
    device: str | None,
    progress: Any,
) -> dict[str, Any]:
    """Re-track a point from a corrected position and append the correction.

    The run is executed once, cached on disk, and referenced by a new
    :class:`~stable.core.corrections.RetrackCorrection` appended to the point —
    so it is undoable like every other edit and costs nothing to re-apply when
    the project is reopened.

    Args:
        workspace: The server's project storage.
        project_id: The owning project.
        point_id: The point to repair.
        frame: Frame the retrack starts from.
        x: Corrected horizontal position, used as the run's query.
        y: Corrected vertical position, used as the run's query.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
        tracker: Tracker plugin override; ``None`` uses the point's own.
        device: Compute device override; ``None`` selects automatically.
        progress: A ``(fraction, message)`` progress callback.

    Returns:
        A JSON-safe summary: the new ``correction_id``, the frames it covers and
        the mean confidence of the spliced span.

    Raises:
        ProjectError: If the frame lies outside the clip.
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

    plugin_name = tracker or point.tracker
    correction = RetrackCorrection(
        frame=frame, x=x, y=y, direction=direction, tracker=plugin_name
    )
    query = Query(frame=frame, x=x, y=y)
    plugin = create_tracker(plugin_name, device=device)
    try:
        if direction == "bidir":
            # A bidirectional retrack is two single-direction runs (the plugin
            # contract is single-run only); the halves cover disjoint spans, so
            # they combine by simply taking whichever one covered each frame.
            forward = plugin.track(
                video, [query], (frame, n_frames), "fwd", progress=_scaled(progress, 0, 2)
            ).point(0)
            backward = plugin.track(
                video, [query], (0, frame + 1), "bwd", progress=_scaled(progress, 1, 2)
            ).point(0)
            run = _combine_halves(forward, backward, frame)
        else:
            run = plugin.track(
                video,
                [query],
                _retrack_frame_range(direction, frame, n_frames),
                cast(Direction, direction),
                progress=progress,
            ).point(0)
    finally:
        plugin.release()

    tracks_dir = workspace.tracks_dir(project_id)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    npz_path = retrack_path_for(tracks_dir, point_id, correction.id)
    run.save_npz(npz_path)

    def _mutate(proj: Project) -> None:
        spec = proj.point(point_id)
        spec.corrections = [
            *spec.corrections,
            RetrackCorrection(
                id=correction.id,
                frame=frame,
                x=x,
                y=y,
                direction=direction,
                tracker=correction.tracker,
                source=proj.relative_path(str(npz_path)),
            ),
        ]

    workspace.update_project(project_id, _mutate)

    covered = run.valid
    progress(1.0, "done")
    return {
        "point_id": point_id,
        "correction_id": correction.id,
        "frame": frame,
        "direction": direction,
        "tracker": correction.tracker,
        "covered_frames": int(covered.sum()),
        "mean_confidence": float(run.confidence[covered].mean()) if covered.any() else 0.0,
    }


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
