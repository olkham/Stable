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

Watching a run happen
---------------------
A track over a long clip takes minutes, and until it finished there was nothing
to look at but a percentage. Two things now stream out of a running job (todos 4
and 16), both through the :class:`~stable.server.jobs.JobProgress` the job body
is handed:

* **Where the tracker is** — published by the worker itself from
  :mod:`stable.video.telemetry`, needing nothing from this module.
* **What it has found** — :class:`LivePartialTrack` below publishes the merged
  positions after each planned run completes, so the path appears a segment at a
  time and the operator can scrub back over frames already covered rather than
  waiting for the whole clip.

Both are best-effort: the publishing calls are duck-typed, so every caller that
passes a plain ``(fraction, message)`` callback — the CLI, and every test
written before this existed — keeps working untouched and simply publishes
nothing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ..core.corrections import (
    DIRECTIONS,
    RetrackCorrection,
    RetrackResolver,
    apply_corrections,
    ground_truth_claims,
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
from .clip_range import effective_range
from .storage import Workspace
from .tracker_options import point_options

__all__ = [
    "run_point_tracking",
    "run_point_retrack",
    "Keyframe",
    "effective_keyframes",
    "effective_anchors",
    "bounded_frame_range",
    "report_path_for",
    "retrack_path_for",
    "violations_path_for",
    "load_report",
    "load_violations",
    "load_keyframe_violations",
    "load_display_track",
    "load_raw_track",
    "retrack_resolver",
    "LivePartialTrack",
]


class LivePartialTrack:
    """Accumulates finished runs and publishes the path found so far.

    A tracking job plans several runs and only merges at the end, so until now
    nothing existed to look at mid-job. This holds one ``x``/``y`` array over the
    whole clip, fills in each run as it lands, and publishes the pair — the same
    shape ``GET .../points/{id}/track`` returns, with ``null`` for frames not yet
    covered, so the viewer draws a partial path with exactly the code that draws
    a finished one.

    This is deliberately *not* the merge: blending runs is
    :func:`~stable.trackers.merge.merge_anchor_runs`' job and running it early,
    on a partial set, could only produce a different answer from the real one.
    Later runs overwrite earlier ones where they overlap, which is enough for a
    preview and cannot influence the result that gets persisted.

    Args:
        n_frames: Clip length, fixing the published arrays' length.
        progress: The job's progress reporter. Anything without a ``merge``
            method (a plain callback) disables publishing entirely.
        total_runs: How many runs are planned, for the ``runs_done``/
            ``runs_total`` counters the UI shows.
    """

    def __init__(self, n_frames: int, progress: Any, *, total_runs: int) -> None:
        """Create an all-unknown partial track bound to one job."""
        self._n_frames = max(0, int(n_frames))
        self._publish = getattr(progress, "merge", None)
        self._total_runs = max(0, int(total_runs))
        self._runs_done = 0
        self._x: list[float | None] = [None] * self._n_frames
        self._y: list[float | None] = [None] * self._n_frames

    @property
    def enabled(self) -> bool:
        """Whether anything is listening (a plain progress callback means no)."""
        return callable(self._publish)

    def add_run(self, track: PointTrack) -> None:
        """Record one finished run's positions and publish the result so far.

        Args:
            track: The run's guarded output, ``NaN`` outside the frames it
                covered.
        """
        self._runs_done += 1
        if not self.enabled:
            return
        points = np.asarray(track.points, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1)
        for frame in np.nonzero(valid)[0]:
            if frame >= self._n_frames:
                break
            self._x[frame] = float(points[frame, 0])
            self._y[frame] = float(points[frame, 1])
        self.publish()

    def publish(self) -> None:
        """Push the current partial path onto the job event."""
        if not self.enabled:
            return
        assert self._publish is not None  # noqa: S101 - `enabled` just checked it
        self._publish(
            track={"x": self._x, "y": self._y, "n_frames": self._n_frames},
            runs_done=self._runs_done,
            runs_total=self._total_runs,
        )


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


KeyframeSource = Literal["pin", "correction"]
"""What in the project file states a keyframe's position."""


@dataclass(frozen=True)
class Keyframe:
    """One position the operator has stated, and what states it.

    To the operator this is a single concept — "the feature is *here* on this
    frame" (todo_6). In storage it is two: a pin in ``PointSpec.queries``, or a
    manual correction that carries an absolute position and was promoted to
    ground truth. Nothing about that split is interesting to a user, but it is
    the difference between which list a delete has to rewrite, so it rides along
    as ``source``/``correction_id`` rather than being guessed at again by every
    caller.

    Args:
        frame: Absolute frame the position is stated on.
        x: Horizontal position in source pixels.
        y: Vertical position in source pixels.
        source: ``"pin"`` for a ``queries`` entry, ``"correction"`` for a
            promoted edit.
        correction_id: The promoting correction's id, or ``None`` for a pin.
    """

    frame: int
    x: float
    y: float
    source: KeyframeSource
    correction_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form the keyframe routes answer with."""
        return {
            "frame": self.frame,
            "x": self.x,
            "y": self.y,
            "source": self.source,
            "correction_id": self.correction_id,
        }

    @property
    def query(self) -> Query:
        """This keyframe as the plain anchor the tracking stack takes."""
        return Query(frame=self.frame, x=self.x, y=self.y)


def effective_keyframes(point: PointSpec) -> list[Keyframe]:
    """Return every position the operator has stated for a point, with its origin.

    The union of the point's pins and the anchors its enabled manual corrections
    promote to (todo 11). A correction wins over a pin on the same frame: it is
    the later statement of where the feature is.

    Args:
        point: The point whose ground truth is wanted.

    Returns:
        Keyframes sorted by frame.
    """
    by_frame: dict[int, Keyframe] = {
        int(q.frame): Keyframe(
            frame=int(q.frame), x=float(q.x), y=float(q.y), source="pin", correction_id=None
        )
        for q in point.sorted_queries()
    }
    for query, correction in ground_truth_claims(point.corrections):
        frame = int(query.frame)
        by_frame[frame] = Keyframe(
            frame=frame,
            x=float(query.x),
            y=float(query.y),
            source="correction",
            correction_id=correction.id,
        )
    return [by_frame[frame] for frame in sorted(by_frame)]


def effective_anchors(point: PointSpec) -> list[Query]:
    """Return every position the operator has stated for a point, by frame.

    :func:`effective_keyframes` without the provenance — what the engine wants,
    since :func:`~stable.trackers.merge.plan_anchor_runs` neither knows nor
    cares whether a position came from a pin or from a promoted correction.

    Args:
        point: The point whose ground truth is wanted.

    Returns:
        Anchors sorted by frame, ready for
        :func:`~stable.trackers.merge.plan_anchor_runs`.
    """
    return [keyframe.query for keyframe in effective_keyframes(point)]


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

    ``keyframe_violations`` (and the ``flagged_keyframes`` frame list derived
    from it) ride along in the same sidecar so reopening a project keeps the
    "this keyframe was never actually reached" signal the timeline draws red —
    it is a property of the run, not something recomputable from the merged
    track, whose anchor passthrough is exact by construction.
    """
    report = result.report
    path.parent.mkdir(parents=True, exist_ok=True)
    disagreement = [None if math.isnan(v) else float(v) for v in report.disagreement.tolist()]
    breaches = violations or []
    payload = {
        "contributions": [int(v) for v in report.contributions.tolist()],
        "disagreement": disagreement,
        "anchor_frames": list(report.anchor_frames),
        "keyframe_violations": [v.to_dict() for v in breaches],
        "flagged_keyframes": sorted({v.frame for v in breaches}),
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
        ``keyframe_violations`` and ``flagged_keyframes`` (todos 13 and 15).

    Raises:
        ProjectError: If the point has no anchors.
        stable.core.errors.TrackerError: If the tracker plugin fails.
    """
    project = workspace.load_project(project_id)
    point = project.point(point_id)
    video = workspace.get_video_source(project_id)
    # The In/Out markers, or the whole clip when none are set (todo 1). This is
    # what makes "track the bit I care about" cost what that bit costs.
    frame_range = effective_range(workspace.project_dir(project_id), video.frame_count)
    # A promoted correction can outlive the clip it was made against (a project
    # re-pointed at a shorter file), and merging rejects an out-of-range anchor.
    # Anchors outside the working range are skipped for the same reason: a run
    # cannot start from a frame it is not allowed to read.
    anchors = [q for q in effective_anchors(point) if frame_range[0] <= q.frame < frame_range[1]]
    if not anchors:
        raise ProjectError(
            f"point {point_id!r} has no anchor keyframes inside frames "
            f"[{frame_range[0]}, {frame_range[1]}) to track from"
        )

    run_specs = plan_anchor_runs(anchors, frame_range)
    tracker_name = tracker or point.tracker

    options = point_options(workspace.project_dir(project_id), point_id)
    plugin = create_tracker(tracker_name, device=device, options=options)
    track_runs: list[TrackRun] = []
    violations: list[AnchorViolation] = []
    n_runs = len(run_specs)
    live = LivePartialTrack(video.frame_count, progress, total_runs=n_runs)
    try:
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
            live.add_run(guarded)
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
        "keyframe_violations": [v.to_dict() for v in violations],
        "flagged_keyframes": sorted({v.frame for v in violations}),
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
    options = point_options(workspace.project_dir(project_id), point_id)
    plugin = create_tracker(plugin_name, device=device, options=options)
    live = LivePartialTrack(n_frames, progress, total_runs=2 if direction == "bidir" else 1)
    try:
        if direction == "bidir":
            # A bidirectional retrack is two single-direction runs (the plugin
            # contract is single-run only); the halves cover disjoint spans, so
            # they combine by simply taking whichever one covered each frame.
            forward = plugin.track(
                video, [query], (frame, run_range[1]), "fwd", progress=_scaled(progress, 0, 2)
            ).point(0)
            live.add_run(forward)
            backward = plugin.track(
                video, [query], (run_range[0], frame + 1), "bwd", progress=_scaled(progress, 1, 2)
            ).point(0)
            run = _combine_halves(forward, backward, frame)
            live.add_run(backward)
        else:
            run = plugin.track(
                video,
                [query],
                run_range,
                cast(Direction, direction),
                progress=progress,
            ).point(0)
            live.add_run(run)
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
        "keyframe_violations": [v.to_dict() for v in violations],
        "flagged_keyframes": sorted({v.frame for v in violations}),
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


def load_keyframe_violations(project: Project, point: PointSpec) -> list[dict[str, Any]]:
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
            # ``anchor_violations`` is what sidecars written before todo_6 call
            # the same list; a project from an older build must still flag.
            entries = report.get("keyframe_violations", report.get("anchor_violations", []))
            found.extend(dict(entry) for entry in entries)
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
