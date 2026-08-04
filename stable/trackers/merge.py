"""Multi-anchor merging — tracker-agnostic, applied to every plugin.

The robustness feature that separates Stable from a plain "track from one
template" tool: a point owns **several user-placed anchor keyframes**
(:class:`~stable.core.types.Query`) spread across the clip. Tracking runs
bidirectionally from every anchor, and the runs are merged here — never inside a
plugin. Plugins only ever do single-run tracking (see
:class:`~stable.trackers.base.TrackerPlugin`).

The pipeline
------------
#. :func:`plan_anchor_runs` turns a point's anchors into the minimal set of
   single-direction runs. Between two adjacent anchors exactly two runs overlap
   (forward from the left anchor, backward from the right one); outside the
   anchor hull there is exactly one. Total work stays ~2x the clip length no
   matter how many anchors the user adds.
#. The caller executes each :class:`RunSpec` with any plugin and wraps the
   output in a :class:`TrackRun`.
#. :func:`merge_anchor_runs` blends them into one
   :class:`~stable.core.types.PointTrack` under three rules:

   * **Exact anchor passthrough** — the merged track passes through every anchor
     position exactly. This is a hard constraint, not a weighting.
   * **Confidence-weighted blend with distance-to-anchor falloff** — a run's
     say over a frame decays the further that frame is from the run's anchor, so
     drift accumulated far from an anchor loses to a fresher run.
   * **Disagreement lowers confidence** — where overlapping runs land in
     different places the merged confidence drops. That dip is the UI's
     "go fix this" signal: the user adds another anchor there and the problem
     region re-merges cleanly.

How the blend actually works
---------------------------
For a frame covered by more than one run, each contributing run gets a weight::

    w_r(t) = falloff_r(t) * confidence_r(t)

    falloff_r(t) = floor + (1 - floor) * max(0, 1 - |t - anchor_r| / scale_r)

so a run speaks with full authority at its own anchor and decays linearly to
``falloff_floor`` ``scale_r`` frames away, where ``scale_r`` defaults to how far
that run actually reached (which, between two anchors, is the distance to the
other anchor — giving exactly the crossfade you want). Position is the weighted
mean; confidence is the weighted mean confidence multiplied by a disagreement
penalty that reaches zero at :attr:`MergeConfig.disagreement_px`.

Frames covered by a single run pass through untouched, and anchor frames are
overwritten with the user's ground truth afterwards, so neither is subject to
the blend at all.

Runs that cross an anchor
-------------------------
:func:`plan_anchor_runs` never plans a run *through* an anchor — each span is
bounded by the anchors either side of it — so within a full multi-anchor track
the blend above is the whole story. Runs started somewhere else (a "retrack from
here", or a bounded run the operator asked for) do cross anchors, and for those
:func:`enforce_anchor_agreement` is the guard: a run that arrives at a
user-placed keyframe more than ``tolerance_px`` away from it is **stopped there**
rather than allowed to carry on from a position the user has already said is
wrong. The disagreement is returned as an :class:`AnchorViolation` so the caller
can report *where* and *by how much* instead of silently losing frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..core.errors import ContractError
from ..core.types import PointTrack, Query

__all__ = [
    "RunSpec",
    "TrackRun",
    "MergeConfig",
    "MergeReport",
    "MergeResult",
    "AnchorViolation",
    "DEFAULT_ANCHOR_TOLERANCE_PX",
    "plan_anchor_runs",
    "merge_anchor_runs",
    "enforce_anchor_agreement",
]

DEFAULT_ANCHOR_TOLERANCE_PX = 6.0
"""How far a run may land from a user anchor before the run is stopped there.

A few pixels of disagreement is ordinary tracker noise; more than that means the
run and the operator are describing different features, and every frame the run
produces beyond that point is derived from a position the operator has already
contradicted.
"""

RunDirection = Literal["fwd", "bwd"]
"""A planned run is always single-direction; ``bidir`` is a caller convenience."""

_WEIGHT_EPS = 1e-9
"""Floor added to blend weights so an all-zero-confidence frame still merges."""


@dataclass(frozen=True)
class RunSpec:
    """One single-direction tracking run to execute for one point.

    Args:
        anchor: The anchor the run starts from. Its frame always lies inside
            ``frame_range``.
        direction: ``"fwd"`` (increasing frames) or ``"bwd"`` (decreasing).
        frame_range: Half-open ``(start, end)`` absolute frame range the run
            must cover — the same convention as
            :meth:`~stable.trackers.base.TrackerPlugin.track`.
    """

    anchor: Query
    direction: RunDirection
    frame_range: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate the run span (dataclass hook)."""
        start, end = self.frame_range
        if end <= start:
            raise ContractError(f"RunSpec frame_range must be non-empty, got [{start}, {end})")
        if not start <= self.anchor.frame < end:
            raise ContractError(
                f"anchor frame {self.anchor.frame} is outside frame_range [{start}, {end})"
            )
        if self.direction not in ("fwd", "bwd"):
            raise ContractError(f"RunSpec.direction must be 'fwd' or 'bwd', got {self.direction!r}")

    @property
    def n_frames(self) -> int:
        """Number of frames this run covers."""
        return self.frame_range[1] - self.frame_range[0]


@dataclass(frozen=True)
class TrackRun:
    """The result of executing one :class:`RunSpec` for one point.

    Args:
        anchor: The anchor this run started from.
        direction: The direction it ran in.
        track: The run's output over the full clip timeline; frames it did not
            cover must be ``NaN``.
    """

    anchor: Query
    direction: RunDirection
    track: PointTrack

    @classmethod
    def from_spec(cls, spec: RunSpec, track: PointTrack) -> TrackRun:
        """Pair a planned run with the track a plugin produced for it."""
        return cls(anchor=spec.anchor, direction=spec.direction, track=track)


@dataclass(frozen=True)
class MergeConfig:
    """Tuning for :func:`merge_anchor_runs`.

    Args:
        falloff_frames: Distance from a run's anchor, in frames, at which that
            run's weight has decayed to ``falloff_floor``. ``None`` derives it
            from the spacing of the surrounding anchors.
        falloff_floor: Residual weight of a run infinitely far from its anchor,
            in ``(0, 1]``. Keeps a lone distant run usable.
        disagreement_px: Distance between overlapping runs, in pixels, at which
            merged confidence is driven to zero. Smaller = stricter.
        min_confidence: Samples below this confidence are treated as missing.
        anchor_confidence: Confidence written at anchor frames — anchors are
            ground truth, so this is ``1.0`` unless a caller wants otherwise.
    """

    falloff_frames: float | None = None
    falloff_floor: float = 0.05
    disagreement_px: float = 8.0
    min_confidence: float = 0.0
    anchor_confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate the tuning values (dataclass hook)."""
        if self.falloff_frames is not None and self.falloff_frames <= 0:
            raise ContractError("MergeConfig.falloff_frames must be positive or None")
        if not 0.0 < self.falloff_floor <= 1.0:
            raise ContractError("MergeConfig.falloff_floor must lie in (0, 1]")
        if self.disagreement_px <= 0:
            raise ContractError("MergeConfig.disagreement_px must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ContractError("MergeConfig.min_confidence must lie in [0, 1]")
        if not 0.0 <= self.anchor_confidence <= 1.0:
            raise ContractError("MergeConfig.anchor_confidence must lie in [0, 1]")


@dataclass(frozen=True)
class MergeReport:
    """Per-frame diagnostics from a merge — what the UI turns into a heat strip.

    Args:
        contributions: ``[T]`` int — how many runs supplied a usable sample.
        disagreement: ``[T]`` float — the largest distance in pixels between any
            two contributing runs; ``NaN`` where fewer than two contributed.
        anchor_frames: Sorted frames the merged track passes through exactly.
    """

    contributions: np.ndarray
    disagreement: np.ndarray
    anchor_frames: tuple[int, ...] = field(default_factory=tuple)

    @property
    def n_frames(self) -> int:
        """Length of the clip timeline ``T``."""
        return int(self.contributions.shape[0])

    def worst_frames(self, limit: int = 5) -> list[int]:
        """Return the frames with the largest disagreement, worst first.

        Args:
            limit: Maximum number of frames to return.

        Returns:
            Frame indices where overlapping runs disagreed most — the places to
            suggest adding another anchor.
        """
        finite = np.where(np.isfinite(self.disagreement))[0]
        if finite.size == 0:
            return []
        order = finite[np.argsort(-self.disagreement[finite], kind="stable")]
        return [int(f) for f in order[:limit]]


@dataclass(frozen=True)
class MergeResult:
    """The merged track plus its diagnostics.

    Args:
        track: The merged :class:`~stable.core.types.PointTrack`.
        report: Per-frame :class:`MergeReport`.
    """

    track: PointTrack
    report: MergeReport


@dataclass(frozen=True)
class AnchorViolation:
    """A run that reached a user-placed anchor and disagreed with it.

    Produced by :func:`enforce_anchor_agreement`. It is the "where, and by how
    much" the operator needs: the run was stopped at :attr:`frame`, having
    wandered :attr:`distance` pixels away from the keyframe they placed there.

    Args:
        frame: The anchor frame the run was stopped at.
        anchor: The operator's ground-truth ``(x, y)`` at that frame.
        tracked: Where the run had actually got to.
        distance: Distance between the two, in pixels.
        origin: Frame the run started from.
        direction: Direction the run was travelling in when it was stopped.
    """

    frame: int
    anchor: tuple[float, float]
    tracked: tuple[float, float]
    distance: float
    origin: int
    direction: RunDirection

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form the server reports to the UI."""
        return {
            "frame": int(self.frame),
            "anchor": [float(self.anchor[0]), float(self.anchor[1])],
            "tracked": [float(self.tracked[0]), float(self.tracked[1])],
            "distance": float(self.distance),
            "origin": int(self.origin),
            "direction": self.direction,
        }


def plan_anchor_runs(anchors: list[Query], frame_range: tuple[int, int]) -> list[RunSpec]:
    """Turn a point's anchors into the minimal set of single-direction runs.

    Args:
        anchors: The point's anchor keyframes, in any order. Duplicate frames
            are rejected — two truths for one frame is a UI-level error.
        frame_range: Half-open ``(start, end)`` absolute range to cover.

    Returns:
        Runs ordered by ``(start frame, direction)``. Between adjacent anchors
        there are exactly two overlapping runs; outside the anchor hull exactly
        one. Anchors outside ``frame_range`` are ignored.

    Raises:
        ContractError: If ``anchors`` is empty (after clipping), the range is
            empty, or two anchors share a frame.
    """
    start, end = int(frame_range[0]), int(frame_range[1])
    if end <= start:
        raise ContractError(f"frame_range must be non-empty, got [{start}, {end})")
    frames = [q.frame for q in anchors]
    if len(set(frames)) != len(frames):
        duplicates = sorted({f for f in frames if frames.count(f) > 1})
        raise ContractError(f"anchors must sit on distinct frames; duplicates at {duplicates}")
    inside = sorted((q for q in anchors if start <= q.frame < end), key=lambda q: q.frame)
    if not inside:
        raise ContractError(f"no anchor lies inside frame_range [{start}, {end})")

    runs: list[RunSpec] = []
    first, last = inside[0], inside[-1]
    if first.frame > start:
        runs.append(RunSpec(first, "bwd", (start, first.frame + 1)))
    for left, right in zip(inside, inside[1:], strict=False):
        span = (left.frame, right.frame + 1)
        runs.append(RunSpec(left, "fwd", span))
        runs.append(RunSpec(right, "bwd", span))
    if last.frame < end - 1:
        runs.append(RunSpec(last, "fwd", (last.frame, end)))
    if len(inside) == 1 and not runs:
        # A single anchor on a single-frame range: one degenerate forward run.
        runs.append(RunSpec(first, "fwd", (first.frame, first.frame + 1)))
    runs.sort(key=lambda r: (r.frame_range[0], r.direction))
    return runs


def enforce_anchor_agreement(
    track: PointTrack,
    anchors: Sequence[Query],
    *,
    origin: int,
    direction: str = "bidir",
    tolerance_px: float = DEFAULT_ANCHOR_TOLERANCE_PX,
) -> tuple[PointTrack, list[AnchorViolation]]:
    """Stop a run at the first anchor it disagrees with, in each direction.

    A user-placed keyframe is ground truth. A run travelling away from
    ``origin`` that arrives at one more than ``tolerance_px`` away from it has
    lost the feature: it is not merely imprecise there, everything it produces
    *beyond* that frame is stepped off a position the operator has already said
    is wrong. Such a run is truncated from that frame onwards (inclusive) rather
    than allowed to drift past, and the disagreement is returned so the caller
    can tell the operator where it happened and by how much.

    Anchors are examined in travel order, so a run that agrees with the first
    keyframe it meets carries on to the next one — several anchors in a row are
    all honoured, which is what makes "track everything forwards and backwards"
    across many keyframes safe.

    Args:
        track: The run's output over the whole clip timeline.
        anchors: The point's anchor keyframes. The one sitting on ``origin``
            (the run's own query) is ignored — a run never disagrees with the
            position it started from.
        origin: Absolute frame the run started from.
        direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``; which way(s) to walk.
        tolerance_px: Disagreement, in pixels, that stops the run.

    Returns:
        ``(track, violations)`` — the (possibly truncated) track and one
        :class:`AnchorViolation` per direction that was stopped, in travel
        order. When nothing disagreed the track is returned unchanged.

    Raises:
        ContractError: If ``tolerance_px`` is not positive or ``direction`` is
            not one of the three accepted values.
    """
    if tolerance_px <= 0:
        raise ContractError(f"tolerance_px must be positive, got {tolerance_px}")
    if direction not in ("fwd", "bwd", "bidir"):
        raise ContractError(f"unknown direction {direction!r}; expected fwd, bwd or bidir")

    n_frames = track.n_frames
    guarded = {
        int(a.frame): (float(a.x), float(a.y))
        for a in anchors
        if int(a.frame) != int(origin) and 0 <= int(a.frame) < n_frames
    }
    if not guarded:
        return track, []

    steps: list[RunDirection] = []
    if direction in ("fwd", "bidir"):
        steps.append("fwd")
    if direction in ("bwd", "bidir"):
        steps.append("bwd")

    points = track.points
    violations: list[AnchorViolation] = []
    cut = np.zeros(n_frames, dtype=bool)
    for run_direction in steps:
        step = 1 if run_direction == "fwd" else -1
        for frame in range(int(origin) + step, n_frames if step > 0 else -1, step):
            anchor = guarded.get(frame)
            if anchor is None:
                continue
            x, y = float(points[frame, 0]), float(points[frame, 1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            distance = float(np.hypot(x - anchor[0], y - anchor[1]))
            if distance <= tolerance_px:
                continue
            violations.append(
                AnchorViolation(
                    frame=frame,
                    anchor=anchor,
                    tracked=(x, y),
                    distance=distance,
                    origin=int(origin),
                    direction=run_direction,
                )
            )
            if step > 0:
                cut[frame:] = True
            else:
                cut[: frame + 1] = True
            break

    if not violations:
        return track, []

    trimmed_points = np.array(track.points, dtype=np.float32, copy=True)
    trimmed_conf = np.array(track.confidence, dtype=np.float32, copy=True)
    trimmed_visible = np.array(track.visible, dtype=bool, copy=True)
    trimmed_points[cut] = np.nan
    trimmed_conf[cut] = 0.0
    trimmed_visible[cut] = False
    return (
        PointTrack(points=trimmed_points, confidence=trimmed_conf, visible=trimmed_visible),
        violations,
    )


def _gather(
    runs: list[TrackRun], n_frames: int, min_confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack run arrays and mask out samples that are missing or too weak.

    Returns:
        ``(positions [R, T, 2], confidence [R, T], visible [R, T], usable [R, T])``.
    """
    positions = np.stack([r.track.points.astype(np.float64) for r in runs])
    confidence = np.stack([r.track.confidence.astype(np.float64) for r in runs])
    visible = np.stack([r.track.visible for r in runs])
    # A confidence of exactly 0 is still a sample; only a positive gate drops one.
    usable: np.ndarray = np.asarray(np.isfinite(positions).all(axis=2), dtype=bool)
    if min_confidence > 0.0:
        usable = np.asarray(usable & (confidence >= min_confidence), dtype=bool)
    return positions, confidence, visible, usable


def _falloff(
    runs: list[TrackRun], usable: np.ndarray, n_frames: int, config: MergeConfig
) -> np.ndarray:
    """Return ``[R, T]`` distance-to-anchor weights for every run.

    Each run's weight is ``1`` at its own anchor frame and decays linearly to
    :attr:`MergeConfig.falloff_floor` at ``scale`` frames away, staying there
    beyond. ``scale`` is :attr:`MergeConfig.falloff_frames` when the caller sets
    it, otherwise the furthest the run actually reached from its anchor — which,
    for the pair of runs spanning two adjacent anchors, is the distance between
    those anchors, so the two runs crossfade exactly across the gap.
    """
    frames = np.arange(n_frames, dtype=np.float64)
    weights = np.empty((len(runs), n_frames), dtype=np.float64)
    for index, run in enumerate(runs):
        distance = np.abs(frames - float(run.anchor.frame))
        if config.falloff_frames is not None:
            scale = float(config.falloff_frames)
        else:
            covered = np.where(usable[index])[0]
            reach = float(distance[covered].max()) if covered.size else 0.0
            scale = max(reach, 1.0)
        ramp = np.clip(1.0 - distance / scale, 0.0, 1.0)
        weights[index] = config.falloff_floor + (1.0 - config.falloff_floor) * ramp
    return weights


def _disagreement(positions: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Return ``[T]`` max pairwise distance between contributing runs.

    ``NaN`` where fewer than two runs contributed.
    """
    n_runs, n_frames = usable.shape
    out = np.full(n_frames, np.nan)
    if n_runs < 2:
        return out
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            both = usable[i] & usable[j]
            if not both.any():
                continue
            delta = positions[i][both] - positions[j][both]
            distance = np.hypot(delta[:, 0], delta[:, 1])
            current = out[both]
            out[both] = np.where(np.isnan(current), distance, np.maximum(current, distance))
    return out


def merge_anchor_runs(
    runs: list[TrackRun],
    anchors: list[Query],
    n_frames: int,
    config: MergeConfig | None = None,
) -> MergeResult:
    """Merge bidirectional per-anchor runs into one track through every anchor.

    Args:
        runs: The executed runs for a single point, typically produced from
            :func:`plan_anchor_runs`. May be empty.
        anchors: The point's anchor keyframes. The merged track passes through
            each of these exactly.
        n_frames: Clip length ``T``; every run's track must be this long.
        config: Blending tuning; defaults to :class:`MergeConfig`.

    Returns:
        A :class:`MergeResult` holding the merged track and its per-frame report.

    Raises:
        ContractError: If a run's track length differs from ``n_frames``, or an
            anchor lies outside the clip.
    """
    config = config or MergeConfig()
    n_frames = int(n_frames)
    if n_frames < 0:
        raise ContractError("n_frames must be >= 0")
    for run in runs:
        if run.track.n_frames != n_frames:
            raise ContractError(
                f"run from anchor {run.anchor.frame} has T={run.track.n_frames}, "
                f"expected {n_frames}"
            )
    for anchor in anchors:
        if not 0 <= anchor.frame < n_frames:
            raise ContractError(
                f"anchor at frame {anchor.frame} is outside the clip (T={n_frames})"
            )

    merged_points = np.full((n_frames, 2), np.nan, dtype=np.float64)
    merged_conf = np.zeros(n_frames, dtype=np.float64)
    merged_visible = np.zeros(n_frames, dtype=bool)

    # Anchor frames are overwritten at the end with the user's ground truth
    # whatever the runs say, so runs meeting on an anchor (a point's own forward
    # and backward runs always do) is never a conflict.
    if runs:
        positions, confidence, visible, usable = _gather(runs, n_frames, config.min_confidence)
        contributions = usable.sum(axis=0).astype(np.int32)
        disagreement = _disagreement(positions, usable)

        # One run covering a frame is not a blend: pass it through untouched, so
        # a single-anchor point is bit-for-bit the tracker's own output.
        single = contributions == 1
        if single.any():
            which = np.argmax(usable, axis=0)
            frames = np.where(single)[0]
            merged_points[frames] = positions[which[frames], frames]
            merged_conf[frames] = confidence[which[frames], frames]
            merged_visible[frames] = visible[which[frames], frames]

        overlapped = np.where(contributions > 1)[0]
        if overlapped.size:
            falloff = _falloff(runs, usable, n_frames, config)
            # `+ _WEIGHT_EPS` keeps a frame where every run reports zero
            # confidence from dividing by zero: the positions still merge (by
            # falloff alone) and the merged confidence still comes out at zero.
            weights = np.where(usable, falloff * (confidence + _WEIGHT_EPS), 0.0)
            columns = weights[:, overlapped]
            total = columns.sum(axis=0)
            share = columns / total
            # Runs that do not cover these frames hold NaN there and a share of
            # 0; zero them out first, since 0 * NaN would poison the sums.
            live = usable[:, overlapped]
            clean_points = np.where(live[:, :, None], positions[:, overlapped, :], 0.0)
            clean_conf = np.where(live, confidence[:, overlapped], 0.0)

            merged_points[overlapped] = np.einsum("rf,rfc->fc", share, clean_points)
            blended_conf = np.einsum("rf,rf->f", share, clean_conf)

            # Where the runs disagree the merged position is a compromise nobody
            # measured, so it must not be trusted like a corroborated one. This
            # dip is the UI's "add an anchor here" signal.
            gap = np.nan_to_num(disagreement[overlapped], nan=0.0)
            penalty = np.clip(1.0 - gap / config.disagreement_px, 0.0, 1.0)
            merged_conf[overlapped] = np.clip(blended_conf * penalty, 0.0, 1.0)
            clean_visible = np.where(live, visible[:, overlapped], False).astype(np.float64)
            merged_visible[overlapped] = np.einsum("rf,rf->f", share, clean_visible) >= 0.5
    else:
        contributions = np.zeros(n_frames, dtype=np.int32)
        disagreement = np.full(n_frames, np.nan)

    # Hard constraint: the merged track passes through every anchor exactly.
    for anchor in anchors:
        merged_points[anchor.frame] = (anchor.x, anchor.y)
        merged_conf[anchor.frame] = config.anchor_confidence
        merged_visible[anchor.frame] = True

    contributions.setflags(write=False)
    disagreement.setflags(write=False)
    return MergeResult(
        track=PointTrack(points=merged_points, confidence=merged_conf, visible=merged_visible),
        report=MergeReport(
            contributions=contributions,
            disagreement=disagreement,
            anchor_frames=tuple(sorted(a.frame for a in anchors)),
        ),
    )
