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
answer to "what has the operator told us about this point": its own anchors
*plus* every manual correction that states a position (see
:func:`stable.core.corrections.ground_truth_anchors`). That matters twice.

#. **Planning.** A full track plans its runs through every one of them, so the
   merge passes exactly through the operator's corrections as well as their own
   anchors — a manual fix improves the next track instead of being overwritten
   by it.
#. **Guarding.** A run that crosses an anchor and disagrees with it is stopped
   there by :func:`stable.trackers.merge.enforce_anchor_agreement` instead of
   drifting past, and the disagreement is reported (frame + pixels) all the way
   out to the job result, the sidecar report and the timeline.

Before this, a retrack simply spliced over whatever anchors lay in its
direction: the user's anchors were silently discarded by the very run that was
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
from .clip_range import effective_range
from .storage import Workspace
from .tracker_options import point_options

__all__ = [
    "run_point_tracking",
    "run_point_retrack",
    "effective_anchors",
    "anchors_changed",
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


def effective_anchors(point: PointSpec) -> list[Query]:
    """Return every position the operator has stated for a point, by frame.

    The union of the point's own anchors and the anchors its enabled manual
    corrections promote to (todo 11). A correction wins over a stored anchor on
    the same frame: it is the later statement of where the feature is.

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
    stop_at_anchor: bool = False,
) -> tuple[int, int]:
    """Return the half-open frame range a directional run should cover.

    A run defaults to "everything in that direction". Two options bound it, and
    the tighter of the two wins:

    * ``stop_at_anchor`` — stop at the next anchor forwards / previous anchor
      backwards (todo 12). With no anchor that way it is a no-op.
    * ``max_frames`` — cover at most this many frames beyond ``frame``
      (todo 20), so a repair costs seconds rather than the whole clip.

    Args:
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
        frame: Absolute frame the run starts from.
        n_frames: Clip length ``T``.
        anchors: The point's ground-truth anchors, needed by ``stop_at_anchor``.
        max_frames: Maximum number of frames to cover beyond ``frame``.
        stop_at_anchor: Stop at the neighbouring anchor.

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
    if stop_at_anchor:
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
    path: Path,
    result: MergeResult,
    violations: list[AnchorViolation] | None = None,
    anchors: list[Query] | None = None,
) -> None:
    """Write a :class:`~stable.trackers.merge.MergeReport` as JSON.

    ``anchor_violations`` (and the ``flagged_anchors`` frame list derived from
    it) ride along in the same sidecar so reopening a project keeps the "this
    anchor was never actually reached" signal the timeline draws red — it is a
    property of the run, not something recomputable from the merged track, whose
    anchor passthrough is exact by construction.

    ``anchors`` records the point's *own* anchors (``points[].queries``) as they
    stood when the run was planned, with their positions, which is what
    :func:`anchors_changed` compares against. Positions, because moving an
    anchor without moving it to another frame leaves ``anchor_frames``
    identical while the merged track still runs through where it used to be;
    the point's own anchors rather than the effective set, because the promoted
    ones belong to corrections, and a correction applies itself to the displayed
    track whether or not a run has been planned through it.
    """
    report = result.report
    path.parent.mkdir(parents=True, exist_ok=True)
    disagreement = [None if math.isnan(v) else float(v) for v in report.disagreement.tolist()]
    breaches = violations or []
    payload = {
        "contributions": [int(v) for v in report.contributions.tolist()],
        "disagreement": disagreement,
        "anchor_frames": list(report.anchor_frames),
        "anchors": [q.to_dict() for q in (anchors or [])],
        "anchor_violations": [v.to_dict() for v in breaches],
        "flagged_anchors": sorted({v.frame for v in breaches}),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def anchors_changed(point: PointSpec, report: dict[str, Any] | None) -> bool:
    """Whether a point's anchors have moved on since the track was computed.

    An anchor is an *input* to tracking, so changing one leaves the stored track
    answering a question nobody is asking any more — it still passes exactly
    through an anchor that has been deleted, and not at all through one just
    placed. Re-running is the operator's call (a full track is minutes of GPU on
    a real clip, and starting one behind their back would be worse than saying
    nothing), so this is the flag the card raises next to the button.

    Only the point's *own* anchors count. A correction-promoted anchor arrives
    with the correction that applies it, so the displayed track already honours
    it; it is the operator's own anchors that a run has to be re-planned through
    before anything changes.

    Args:
        point: The point, with its current anchors.
        report: The sidecar of the run that produced its track, or ``None`` when
            there is no track to be stale.

    Returns:
        Whether the point's anchors differ from the ones the run recorded. False
        when there is no track yet (an untracked point is not out of date) and
        false for a report written before this was recorded — a sidecar that
        cannot answer the question must not be read as a "yes".
    """
    if report is None:
        return False
    recorded = report.get("anchors")
    if not isinstance(recorded, list):
        return False
    try:
        was = [Query.from_dict(dict(entry)) for entry in recorded]
    except StableError:  # pragma: no cover - corrupt sidecar
        return False
    now = point.sorted_queries()
    return [(q.frame, q.x, q.y) for q in now] != [(q.frame, q.x, q.y) for q in was]


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

    Every anchor the operator has stated — the point's own *and* the ones its
    manual corrections were promoted to, see :func:`effective_anchors` — is
    planned through, so the merged track passes exactly through all of them.
    Runs are guarded: one that reaches an anchor it disagrees with is stopped
    there rather than allowed to drift on, and that anchor comes back flagged.

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
        anchor_tolerance_px: How far a run may land from an anchor before it is
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
            f"point {point_id!r} has no anchors inside frames "
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
    # The point's *own* anchors, not the effective set this run planned through:
    # see `anchors_changed`, which reads this back to answer "have the operator's
    # anchors moved on since?". Recording the effective set would answer "yes"
    # the moment the point has any correction at all, which is not a change and
    # not something re-tracking would settle.
    _save_report(report_path_for(npz_path), merge_result, violations, point.sorted_queries())

    def _mutate(p: Project) -> None:
        spec = p.point(point_id)
        spec.raw = p.relative_path(str(npz_path))
        if tracker:
            spec.tracker = tracker
        # A composite that has just been tracked is not a composite any more:
        # its path is this run's, not the average of anything. The UI hides the
        # button that gets here for one, but the API is open and a point whose
        # `derived_from` outlived the path it described would be a lie the
        # panel would then draw.
        spec.derived_from = []

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
    stop_at_anchor: bool = False,
    anchor_tolerance_px: float = DEFAULT_ANCHOR_TOLERANCE_PX,
) -> dict[str, Any]:
    """Track one direction from one frame, and splice the run in as a correction.

    This is "track forwards / backwards from this anchor" (and its
    from-the-playhead sibling, "carry on from here"). The run is executed once,
    cached on disk, and referenced by a new
    :class:`~stable.core.corrections.RetrackCorrection` appended to the point —
    so it is undoable like every other edit, costs nothing to re-apply when the
    project is reopened, and goes away again with the anchor it started from
    (see :func:`stable.core.corrections.corrections_anchored_at`).

    The run is bounded and guarded rather than open-ended:

    * ``stop_at_anchor`` / ``max_frames`` cap how far it goes (todos 12, 20);
    * any anchor it *does* reach and disagree with stops it there (todo 13),
      so a run can never silently overwrite the operator's own anchors.

    Because a correction only ever splices the frames its cached run actually
    covered, bounding the run bounds the edit — no extra range bookkeeping is
    needed in :func:`~stable.core.corrections.apply_corrections`.

    A point that has never been tracked gets an all-``NaN`` baseline written for
    it here (todo 6, seventh round). Corrections splice *onto* a raw track, and
    without one this whole operation computed a run that
    :func:`load_display_track` would then refuse to show — which is why the UI
    could only offer it once a full track already existed, and why "place an
    anchor, then track forwards from it" was not a thing the tool could do.

    Args:
        workspace: The server's project storage.
        project_id: The owning project.
        point_id: The point to track.
        frame: Frame the run starts from.
        x: Corrected horizontal position, used as the run's query. ``None``
            reuses the point's current position at ``frame`` — an anchor there
            if it has one, otherwise its tracked position, which is what
            "track on from here to the next anchor" means.
        y: Corrected vertical position; see ``x``.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
        tracker: Tracker plugin override; ``None`` uses the point's own.
        device: Compute device override; ``None`` selects automatically.
        progress: A ``(fraction, message)`` progress callback.
        max_frames: Cover at most this many frames beyond ``frame``.
        stop_at_anchor: Stop at the next/previous anchor.
        anchor_tolerance_px: How far the run may land from an anchor it crosses
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
        stop_at_anchor=stop_at_anchor,
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
    baseline_path = _ensure_baseline(project, point, tracks_dir, n_frames)

    def _mutate(proj: Project) -> None:
        spec = proj.point(point_id)
        if baseline_path is not None:
            spec.raw = proj.relative_path(str(baseline_path))
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


def _ensure_baseline(
    project: Project, point: PointSpec, tracks_dir: Path, n_frames: int
) -> Path | None:
    """Give a never-tracked point an empty track for corrections to splice onto.

    The correction layer is defined as ``apply_corrections(raw, corrections)``,
    so with no ``raw`` there is nothing for a directional run to land on and
    :func:`load_display_track` returns ``None`` — the run happens, is cached,
    and is invisible. An all-``NaN`` baseline is the honest starting state
    ("this point has a track; no frame of it is known yet"), and one run over
    frames 100–160 then reads as exactly the frames it covered.

    Args:
        project: The project, for resolving the point's existing ``raw`` path.
        point: The point as loaded before the run.
        tracks_dir: The project's ``tracks/`` directory (already created).
        n_frames: Clip length ``T``.

    Returns:
        The baseline's path when one was written, or ``None`` when the point
        already had a usable track and nothing needed doing.
    """
    if point.raw:
        existing = project.resolve_path(point.raw)
        if existing and Path(existing).is_file():
            return None
    path = tracks_dir / f"{point.id}.npz"
    if path.is_file():
        # On disk but not referenced (a project edited by hand, an interrupted
        # write): adopt it rather than throwing a real track away.
        return path
    PointTrack.empty(n_frames).save_npz(path)
    return path


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
        return apply_corrections(
            raw,
            point.corrections,
            resolve_retrack=retrack_resolver(project),
            # A per-track smoothing correction can be cut by frequency, and only
            # the clip knows how many frames a second is. Everything else in the
            # correction layer is in frames and ignores this.
            fps=project.video.fps,
        )
    except StableError:
        # A correction this build cannot apply must not hide the track itself.
        return raw


def load_anchor_violations(project: Project, point: PointSpec) -> list[dict[str, Any]]:
    """Return every anchor disagreement recorded for a point, in frame order.

    A track can fail to reach an operator's anchor in two places: the
    multi-anchor run behind the raw track, and any enabled directional run
    layered on top. Both are collected here so the timeline paints exactly the
    anchors that are still unreached in what the user is looking at — a run
    whose correction is toggled off stops flagging anything.
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
