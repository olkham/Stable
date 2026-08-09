"""Non-destructive manual edits — the correction layer.

Hard requirement from ``PLAN.md``: *raw tracker output is immutable*. A user
edit never rewrites a track; it appends a typed, serialisable
:class:`Correction` to the point's ``corrections`` list. The displayed track is
always ``apply_corrections(raw, corrections)``, so:

* undo is "drop the last correction" — no snapshots, no history of arrays;
* re-tracking a point does not destroy the user's edits;
* the project file stays small and human-readable.

Corrections apply **in list order**; each one is a pure function of the track it
receives. The five types cover the editing vocabulary of Phase 5:

===================== ===============================================================
:class:`OffsetCorrection`      nudge a point, blended out over a falloff range
:class:`ReplaceCorrection`     hard-set the position at one frame
:class:`GapFillCorrection`     interpolate across a range with no valid samples
:class:`DeleteRangeCorrection` mark a range invalid (bad/occluded tracking)
:class:`RetrackCorrection`     re-run the tracker from a corrected position
===================== ===============================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np

from .errors import ContractError
from .types import PointTrack, Query

__all__ = [
    "Correction",
    "OffsetCorrection",
    "ReplaceCorrection",
    "GapFillCorrection",
    "DeleteRangeCorrection",
    "RetrackCorrection",
    "CORRECTION_TYPES",
    "GROUND_TRUTH_KINDS",
    "ANCHORING_KINDS",
    "correction_from_dict",
    "apply_corrections",
    "ground_truth_anchors",
    "corrections_anchored_at",
    "RetrackResolver",
    "FALLOFF_SHAPES",
    "INTERPOLATION_METHODS",
    "DIRECTIONS",
]

FALLOFF_SHAPES: tuple[str, ...] = ("linear", "smooth")
"""How an offset decays to zero away from its own frame (``smooth`` = smoothstep)."""

INTERPOLATION_METHODS: tuple[str, ...] = ("linear", "spline")
"""Gap-fill interpolation methods."""

DIRECTIONS: tuple[str, ...] = ("fwd", "bwd", "bidir")
"""Tracking directions, shared with the tracker plugin contract."""


def _new_id() -> str:
    """Return a short unique identifier for a correction."""
    return uuid4().hex[:12]


@dataclass(frozen=True, kw_only=True)
class Correction:
    """Base class for every typed, serialisable manual edit.

    Args:
        id: Stable identifier, unique within a point. Auto-generated if omitted;
            the UI uses it for selection and for undo.
        enabled: When False the edit is kept in the project but not applied —
            this is how "toggle my fix off to compare" works.

    Attributes:
        kind: Class-level discriminator written to the project file.
    """

    kind: ClassVar[str] = "base"

    id: str = field(default_factory=_new_id)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe project-file form, including ``kind``."""
        data: dict[str, Any] = {"kind": self.kind}
        for f in fields(self):
            data[f.name] = getattr(self, f.name)
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Correction:
        """Rebuild the correct subclass from :meth:`to_dict` output.

        Args:
            data: A dict carrying a ``kind`` key.

        Returns:
            The concrete :class:`Correction` subclass instance.

        Raises:
            ContractError: If ``kind`` is missing or unknown.
        """
        return correction_from_dict(data)


@dataclass(frozen=True, kw_only=True)
class OffsetCorrection(Correction):
    """Nudge the track by ``(dx, dy)`` at ``frame``, blended out either side.

    The full offset is applied at ``frame`` and decays to zero over
    ``falloff_before`` frames before it and ``falloff_after`` frames after it, so
    a local fix does not introduce a step at its edges. Falloff ``0`` means the
    offset applies to that single frame only.

    Args:
        frame: Absolute frame index the full offset is applied at.
        dx: Horizontal offset in pixels.
        dy: Vertical offset in pixels.
        falloff_before: Frames over which the offset ramps in.
        falloff_after: Frames over which the offset ramps out.
        shape: One of :data:`FALLOFF_SHAPES`.

    Raises:
        ContractError: If the frame is negative, a falloff is negative, or the
            shape is unknown.
    """

    kind: ClassVar[str] = "offset"

    frame: int
    dx: float
    dy: float
    falloff_before: int = 0
    falloff_after: int = 0
    shape: str = "smooth"

    def __post_init__(self) -> None:
        """Validate the offset (dataclass hook)."""
        if self.frame < 0:
            raise ContractError(f"OffsetCorrection.frame must be >= 0, got {self.frame}")
        if self.falloff_before < 0 or self.falloff_after < 0:
            raise ContractError("OffsetCorrection falloff values must be >= 0")
        if self.shape not in FALLOFF_SHAPES:
            raise ContractError(f"unknown falloff shape {self.shape!r}; expected {FALLOFF_SHAPES}")

    @property
    def frame_range(self) -> tuple[int, int]:
        """Inclusive ``(start, end)`` frame span this correction can influence."""
        return (max(0, self.frame - self.falloff_before), self.frame + self.falloff_after)


@dataclass(frozen=True, kw_only=True)
class ReplaceCorrection(Correction):
    """Hard-set the tracked position at a single frame.

    Unlike :class:`OffsetCorrection` this ignores the raw sample entirely; it is
    the "I know exactly where this is" edit. Confidence at that frame becomes
    ``1.0`` and the point is marked visible.

    Args:
        frame: Absolute frame index.
        x: Corrected horizontal position in pixels.
        y: Corrected vertical position in pixels.

    Raises:
        ContractError: If ``frame`` is negative.
    """

    kind: ClassVar[str] = "replace"

    frame: int
    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate the replacement (dataclass hook)."""
        if self.frame < 0:
            raise ContractError(f"ReplaceCorrection.frame must be >= 0, got {self.frame}")


@dataclass(frozen=True, kw_only=True)
class GapFillCorrection(Correction):
    """Interpolate positions across ``[start, end]`` (inclusive).

    Used where the tracker lost the point: the surrounding valid samples drive
    the interpolation. Filled frames get a reduced confidence so the solver can
    still down-weight them.

    Args:
        start: First frame of the gap, inclusive.
        end: Last frame of the gap, inclusive.
        method: One of :data:`INTERPOLATION_METHODS`.
        confidence: Confidence assigned to the interpolated frames, in ``[0, 1]``.

    Raises:
        ContractError: On a negative/inverted range, unknown method, or a
            confidence outside ``[0, 1]``.
    """

    kind: ClassVar[str] = "gap_fill"

    start: int
    end: int
    method: str = "spline"
    confidence: float = 0.5

    def __post_init__(self) -> None:
        """Validate the gap range (dataclass hook)."""
        if self.start < 0:
            raise ContractError(f"GapFillCorrection.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ContractError(f"invalid range [{self.start}, {self.end}]: end < start")
        if self.method not in INTERPOLATION_METHODS:
            raise ContractError(
                f"unknown interpolation method {self.method!r}; expected {INTERPOLATION_METHODS}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("GapFillCorrection.confidence must lie in [0, 1]")


@dataclass(frozen=True, kw_only=True)
class DeleteRangeCorrection(Correction):
    """Mark ``[start, end]`` (inclusive) invalid: NaN position, zero confidence.

    Args:
        start: First frame to discard, inclusive.
        end: Last frame to discard, inclusive.

    Raises:
        ContractError: On a negative or inverted range.
    """

    kind: ClassVar[str] = "delete_range"

    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate the deleted range (dataclass hook)."""
        if self.start < 0:
            raise ContractError(f"DeleteRangeCorrection.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ContractError(f"invalid range [{self.start}, {self.end}]: end < start")


@dataclass(frozen=True, kw_only=True)
class RetrackCorrection(Correction):
    """Re-run the tracker from a corrected position and splice the result in.

    The corrected position becomes a new query for the plugin. The resulting run
    is cached on disk (``source``) so reopening the project does not require a
    GPU; when ``source`` is absent the segment must be recomputed.

    Note:
        A retrack that the user wants to keep *permanently* is better expressed
        as an extra anchor in the point's ``queries`` list — that feeds the
        multi-anchor merge. This correction exists for local, after-the-fact
        repairs that must not disturb the merged baseline.

    Args:
        frame: Frame the retrack starts from.
        x: Corrected horizontal position used as the new query.
        y: Corrected vertical position used as the new query.
        direction: One of :data:`DIRECTIONS`.
        tracker: Tracker name to use; ``None`` means the point's own tracker.
        source: Optional path (relative to the project file) of the cached run.

    Raises:
        ContractError: If the frame is negative or the direction is unknown.
    """

    kind: ClassVar[str] = "retrack"

    frame: int
    x: float
    y: float
    direction: str = "fwd"
    tracker: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate the retrack request (dataclass hook)."""
        if self.frame < 0:
            raise ContractError(f"RetrackCorrection.frame must be >= 0, got {self.frame}")
        if self.direction not in DIRECTIONS:
            raise ContractError(f"unknown direction {self.direction!r}; expected {DIRECTIONS}")


CORRECTION_TYPES: dict[str, type[Correction]] = {
    cls.kind: cls
    for cls in (
        OffsetCorrection,
        ReplaceCorrection,
        GapFillCorrection,
        DeleteRangeCorrection,
        RetrackCorrection,
    )
}
"""Discriminator (``kind``) to class map used for project-file deserialisation."""

GROUND_TRUTH_KINDS: tuple[str, ...] = ("replace", "retrack")
"""Correction kinds that state an absolute position, i.e. *are* ground truth.

These are the edits where the operator said "the feature is **here**", in
pixels, so they promote to anchors (see :func:`ground_truth_anchors`).
``offset`` is deliberately absent: it is a relative nudge, meaningful only
against the track it was applied to. The server promotes a drag by recording the
resulting absolute position as a companion ``replace`` at the same frame, which
is what makes the promotion survive a re-track without doubling the nudge.
``gap_fill`` and ``delete_range`` state no position at all.
"""

ANCHORING_KINDS: tuple[str, ...] = ("replace", "retrack", "offset")
"""Correction kinds that :func:`corrections_anchored_at` takes to *be* an anchor.

:data:`GROUND_TRUTH_KINDS` plus ``offset``, and the extra entry is the whole
reason this list exists separately. A drag on the frame is stored as an
``offset`` and a companion ``replace`` at the same frame (see the server's
``_promotion_for``); the ``replace`` is what shows up as an anchor, but the two
are one gesture, and removing the anchor while leaving the nudge that placed it
would move the point by a fraction of a pixel and look like the deletion did
nothing at all.
"""


def correction_from_dict(data: dict[str, Any]) -> Correction:
    """Rebuild a typed correction from its project-file dict.

    Args:
        data: Dict with a ``kind`` key plus that type's fields. Unknown keys are
            ignored so newer project files degrade gracefully.

    Returns:
        The concrete :class:`Correction` instance.

    Raises:
        ContractError: If ``kind`` is missing/unknown or a required field is absent.
    """
    if "kind" not in data:
        raise ContractError("correction dict is missing the 'kind' discriminator")
    kind = data["kind"]
    cls = CORRECTION_TYPES.get(kind)
    if cls is None:
        raise ContractError(
            f"unknown correction kind {kind!r}; expected {sorted(CORRECTION_TYPES)}"
        )
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ContractError(f"invalid {kind!r} correction: {exc}") from exc


def ground_truth_anchors(corrections: Sequence[Correction]) -> list[Query]:
    """Return the anchors a point's manual corrections amount to.

    Every enabled correction that states an absolute position (see
    :data:`GROUND_TRUTH_KINDS`) is a place the operator has told the tool where
    the feature actually is — which is precisely what an anchor is. Promoting
    them means a later re-track plans its runs *through* those positions and
    merges to pass exactly through them, instead of overwriting the operator's
    work with a fresh guess.

    Args:
        corrections: A point's correction layer, in order. Disabled entries are
            skipped, so toggling a fix off also retires the anchor it implied.

    Returns:
        One :class:`~stable.core.types.Query` per frame, sorted by frame. Where
        several corrections land on the same frame the last one wins, matching
        the order :func:`apply_corrections` applies them in.
    """
    by_frame: dict[int, Query] = {}
    for correction in corrections:
        if not correction.enabled or correction.kind not in GROUND_TRUTH_KINDS:
            continue
        if not isinstance(correction, ReplaceCorrection | RetrackCorrection):
            continue  # pragma: no cover - a third-party subclass reusing the kind
        frame = int(correction.frame)
        by_frame[frame] = Query(frame=frame, x=float(correction.x), y=float(correction.y))
    return [by_frame[frame] for frame in sorted(by_frame)]


def corrections_anchored_at(corrections: Sequence[Correction], frame: int) -> list[Correction]:
    """Return the corrections that put an anchor on ``frame``.

    The counterpart to :func:`ground_truth_anchors`: that one answers "which
    anchors do these edits imply", this one answers "which edits does deleting
    that anchor have to take with it". An anchor the operator can see and can
    delete must be *one* thing, so removing it removes everything that asserted
    it — otherwise a promoted anchor comes straight back the moment the list is
    re-read, which is what "delete the correction on the Edit tab" used to mean.

    Disabled corrections are skipped: a disabled edit states nothing, shows no
    anchor, and is something the operator deliberately kept to compare against.

    Args:
        corrections: The point's correction layer, in order.
        frame: Absolute frame index of the anchor being removed.

    Returns:
        The corrections to drop, in list order — see :data:`ANCHORING_KINDS` for
        why a plain ``offset`` at that frame is one of them.
    """
    return [
        correction
        for correction in corrections
        if correction.enabled
        and correction.kind in ANCHORING_KINDS
        and int(getattr(correction, "frame", -1)) == int(frame)
    ]


RetrackResolver = Callable[["RetrackCorrection"], PointTrack | None]
"""Looks up the cached run a :class:`RetrackCorrection` refers to.

Splicing a retrack needs pixels, and :func:`apply_corrections` is a pure array
function that must never import the tracking stack. The caller therefore hands
in a resolver that turns a correction's ``source`` into a
:class:`~stable.core.types.PointTrack` (or ``None`` when the run has not been
computed yet). See :mod:`stable.server.tracking` for the server's.
"""


def _smoothstep(values: np.ndarray) -> np.ndarray:
    """Return the C1 smoothstep of ``values`` (already clamped to ``[0, 1]``)."""
    return values * values * (3.0 - 2.0 * values)


def _falloff_weights(correction: OffsetCorrection, n_frames: int) -> np.ndarray:
    """Return the ``[T]`` influence of an offset, ``1`` at its own frame.

    The weight ramps in over ``falloff_before`` frames and out over
    ``falloff_after``; a zero falloff on one side means the offset simply does
    not reach that way, so a single-frame nudge stays a single-frame nudge.
    """
    distance = np.arange(n_frames, dtype=np.float64) - float(correction.frame)
    weights = np.zeros(n_frames, dtype=np.float64)
    before, after = distance < 0.0, distance > 0.0
    if correction.falloff_before > 0:
        weights[before] = 1.0 + distance[before] / float(correction.falloff_before)
    if correction.falloff_after > 0:
        weights[after] = 1.0 - distance[after] / float(correction.falloff_after)
    weights[distance == 0.0] = 1.0
    np.clip(weights, 0.0, 1.0, out=weights)
    return _smoothstep(weights) if correction.shape == "smooth" else weights


def _catmull_rom(knots: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Evaluate a Catmull-Rom cubic through ``(knots, values)`` at ``targets``.

    Tangents are the central finite differences of the neighbouring knots
    (one-sided at the ends), which is what makes the curve pass through every
    knot while staying C1 — a linear fill would put a visible corner at each end
    of a repaired gap, and that corner is exactly what the operator is trying to
    get rid of. Targets outside the knot hull clamp to the nearest knot, matching
    :func:`numpy.interp`.
    """
    if knots.shape[0] < 3:
        return np.interp(targets, knots, values)
    spans = np.diff(knots)
    tangents = np.empty_like(values)
    tangents[1:-1] = (values[2:] - values[:-2]) / (knots[2:] - knots[:-2])
    tangents[0] = (values[1] - values[0]) / spans[0]
    tangents[-1] = (values[-1] - values[-2]) / spans[-1]

    index = np.clip(np.searchsorted(knots, targets, side="right") - 1, 0, knots.shape[0] - 2)
    width = spans[index]
    step = np.clip((targets - knots[index]) / width, 0.0, 1.0)
    step2 = step * step
    step3 = step2 * step
    return (
        (2.0 * step3 - 3.0 * step2 + 1.0) * values[index]
        + (step3 - 2.0 * step2 + step) * width * tangents[index]
        + (-2.0 * step3 + 3.0 * step2) * values[index + 1]
        + (step3 - step2) * width * tangents[index + 1]
    )


def _interpolate(
    knots: np.ndarray, values: np.ndarray, targets: np.ndarray, method: str
) -> np.ndarray:
    """Fill ``targets`` from ``(knots, values)`` with the requested method."""
    if knots.shape[0] == 0:
        return np.full(targets.shape, np.nan, dtype=np.float64)
    if knots.shape[0] == 1:
        return np.full(targets.shape, float(values[0]), dtype=np.float64)
    if method == "spline":
        return _catmull_rom(knots, values, targets)
    return np.interp(targets, knots, values)


def _range_mask(start: int, end: int, n_frames: int) -> np.ndarray:
    """Return a ``[T]`` bool mask for the inclusive frame range ``[start, end]``."""
    mask = np.zeros(n_frames, dtype=bool)
    lo, hi = max(int(start), 0), min(int(end) + 1, n_frames)
    if hi > lo:
        mask[lo:hi] = True
    return mask


def _retrack_span(correction: RetrackCorrection, n_frames: int) -> np.ndarray:
    """Return the ``[T]`` mask of frames a retrack is allowed to overwrite."""
    if correction.direction == "fwd":
        return _range_mask(correction.frame, n_frames - 1, n_frames)
    if correction.direction == "bwd":
        return _range_mask(0, correction.frame, n_frames)
    return np.ones(n_frames, dtype=bool)


class _Editable:
    """Mutable working copy of a track's three arrays, applied one edit at a time."""

    def __init__(self, track: PointTrack) -> None:
        """Take a writable copy of ``track``; the original is never touched."""
        self.points = np.array(track.points, dtype=np.float64, copy=True)
        self.confidence = np.array(track.confidence, dtype=np.float64, copy=True)
        self.visible = np.array(track.visible, dtype=bool, copy=True)
        self.n_frames = int(self.points.shape[0])

    def valid(self) -> np.ndarray:
        """Return the ``[T]`` mask of frames whose position is currently finite."""
        return np.asarray(np.isfinite(self.points).all(axis=1), dtype=bool)

    def set_position(self, frame: int, x: float, y: float, confidence: float) -> None:
        """Hard-set one frame's position, confidence and visibility."""
        if not 0 <= frame < self.n_frames:
            return
        self.points[frame] = (float(x), float(y))
        self.confidence[frame] = float(confidence)
        self.visible[frame] = True

    def finish(self) -> PointTrack:
        """Return the edited arrays as a fresh, frozen :class:`PointTrack`."""
        return PointTrack(
            points=self.points.astype(np.float32),
            confidence=np.clip(self.confidence, 0.0, 1.0).astype(np.float32),
            visible=self.visible,
        )


def _apply_offset(state: _Editable, correction: OffsetCorrection) -> None:
    """Blend a nudge over the offset's falloff window."""
    weights = _falloff_weights(correction, state.n_frames)
    live = (weights > 0.0) & state.valid()
    if not live.any():
        return
    state.points[live, 0] += correction.dx * weights[live]
    state.points[live, 1] += correction.dy * weights[live]


def _apply_gap_fill(state: _Editable, correction: GapFillCorrection) -> None:
    """Interpolate a range from the valid samples surrounding it."""
    gap = _range_mask(correction.start, correction.end, state.n_frames)
    if not gap.any():
        return
    knots_mask = state.valid() & ~gap
    knots = np.nonzero(knots_mask)[0].astype(np.float64)
    targets = np.nonzero(gap)[0].astype(np.float64)
    filled = np.stack(
        [
            _interpolate(knots, state.points[knots_mask, axis], targets, correction.method)
            for axis in (0, 1)
        ],
        axis=1,
    )
    usable = np.isfinite(filled).all(axis=1)
    rows = np.nonzero(gap)[0][usable]
    state.points[rows] = filled[usable]
    state.confidence[rows] = correction.confidence
    state.visible[rows] = True


def _apply_delete_range(state: _Editable, correction: DeleteRangeCorrection) -> None:
    """Mark a range unusable: no position, no confidence, not visible."""
    dead = _range_mask(correction.start, correction.end, state.n_frames)
    state.points[dead] = np.nan
    state.confidence[dead] = 0.0
    state.visible[dead] = False


def _apply_retrack(
    state: _Editable, correction: RetrackCorrection, resolved: PointTrack | None
) -> None:
    """Splice a cached re-tracking run in, anchored on the corrected position."""
    if resolved is not None:
        if resolved.n_frames != state.n_frames:
            raise ContractError(
                f"retrack run covers {resolved.n_frames} frames but the track has "
                f"{state.n_frames}; both must span the whole clip timeline"
            )
        span = _retrack_span(correction, state.n_frames)
        run_valid = np.isfinite(resolved.points).all(axis=1)
        rows = np.nonzero(span & run_valid)[0]
        state.points[rows] = resolved.points[rows]
        state.confidence[rows] = resolved.confidence[rows]
        state.visible[rows] = resolved.visible[rows]
    # The user's position is ground truth and outranks whatever the run said,
    # so it is re-asserted after the splice rather than before it.
    state.set_position(correction.frame, correction.x, correction.y, 1.0)


def apply_corrections(
    track: PointTrack,
    corrections: Sequence[Correction],
    *,
    resolve_retrack: RetrackResolver | None = None,
) -> PointTrack:
    """Return ``track`` with the correction layer applied, leaving it untouched.

    This is the single funnel through which edited tracks are produced. It is a
    pure function: the same ``(raw, corrections)`` pair always yields the same
    result, which is what makes undo/redo just a list operation.

    Each correction sees the output of the ones before it, so a gap fill can
    repair what a delete-range removed, and an offset can nudge a spliced
    retrack. The five semantics are:

    * :class:`OffsetCorrection` — add ``(dx, dy)`` scaled by the offset's
      falloff weight, only where a position already exists (a nudge cannot
      invent a sample).
    * :class:`ReplaceCorrection` — hard-set one frame; confidence becomes
      ``1.0`` and the point becomes visible.
    * :class:`GapFillCorrection` — interpolate the range from the valid samples
      outside it (``spline`` is a Catmull-Rom through them, ``linear`` a plain
      lerp); filled frames take the correction's own confidence.
    * :class:`DeleteRangeCorrection` — ``NaN`` position, zero confidence, not
      visible.
    * :class:`RetrackCorrection` — splice ``resolve_retrack``'s cached run over
      the direction's frame span, then re-assert the corrected position.

    Args:
        track: The immutable raw track (or the output of an earlier stage).
        corrections: Corrections to apply, in order. Disabled ones are skipped.
        resolve_retrack: Looks up a :class:`RetrackCorrection`'s cached run; see
            :data:`RetrackResolver`. When omitted (or when it returns ``None``)
            a retrack still contributes its corrected position, so the edit is
            never silently lost — only the re-tracked span is missing until the
            run is computed.

    Returns:
        A new :class:`PointTrack`; ``track`` itself is never modified.

    Raises:
        ContractError: If a resolved retrack run does not span the same clip
            timeline as ``track``.
    """
    active = [c for c in corrections if c.enabled]
    if not active:
        return track.replace()

    state = _Editable(track)
    for correction in active:
        if isinstance(correction, OffsetCorrection):
            _apply_offset(state, correction)
        elif isinstance(correction, ReplaceCorrection):
            state.set_position(correction.frame, correction.x, correction.y, 1.0)
        elif isinstance(correction, GapFillCorrection):
            _apply_gap_fill(state, correction)
        elif isinstance(correction, DeleteRangeCorrection):
            _apply_delete_range(state, correction)
        elif isinstance(correction, RetrackCorrection):
            resolved = resolve_retrack(correction) if resolve_retrack is not None else None
            _apply_retrack(state, correction, resolved)
        else:  # pragma: no cover - a third-party Correction subclass
            raise ContractError(
                f"don't know how to apply correction kind {correction.kind!r}; "
                f"expected one of {sorted(CORRECTION_TYPES)}"
            )
    return state.finish()
