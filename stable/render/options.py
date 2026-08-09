"""User-facing stabilisation options: what to hold still, where, how much to crop.

The Phase 3 renderer answered one question — "remove the motion the smoothing
took out" — and answered it well. It left five the operator has to be able to
answer too, and this module is their vocabulary:

**What is being held, and where?**
    A :class:`StabilizationTarget` is a list of :class:`TargetPoint` — one to N
    marks on the delivered frame, each naming the tracked path that goes on it.
    A mark holds one track, the mean of several, or the mean of all of them, and
    holds either their raw positions or their smoothed estimates
    (:data:`TARGET_SOURCES`). The render is the similarity that best satisfies
    every mark at once, so one mark shifts the picture, two also rotate and
    scale it, and three or more are a least-squares fit whose miss is reported
    per mark. There is no mode to choose: what you map is what gets fitted.

    An empty list is a meaningful state — with nothing to fit, the render falls
    back to undoing what the Smoothing panel removed.

**Where does a mark sit?**
    :class:`Placement`, a position in **output** coordinates as a fraction of
    the output frame, so "hold this feature on the left thirds line" is a thing
    the operator states rather than one they discover after the render. Storing
    the fraction rather than a pixel is what keeps the composition intact when
    the crop scale or the delivery size changes: a mark on the thirds line stays
    on the thirds line.

**What size is delivered?**
    :class:`OutputResolution`. Delivery size and field of view are different
    decisions, so cropping hard into a shot and still handing over 1920×1080 is
    two controls rather than a compromise between one.

**How much of the frame survives?**
    :class:`CropSpec`. ``auto`` is Phase 3's exact minimal-gapless crop — now
    optionally measured over the *export* range only
    (:attr:`CropSpec.limit_to_range`), which is what makes a thirty-second
    section of a shaky ten-minute take croppable at all. An explicit ``scale``
    overrides it in either direction.

**Which frames get written?**
    :class:`ExportRange`. The In/Out markers bound tracking *and* the render,
    which is right until the operator wants to deliver a different section than
    the one they tracked; this overrides them for the export alone.

**What goes in the exposed edge?**
    :class:`~stable.render.fill.BorderFill` — see that module.

:class:`RenderOptions` bundles them and round-trips as JSON, so a server or a
CLI can persist an operator's choices without either of them re-deriving what a
valid choice is.

Nothing here touches the pivot-relative curve convention: a mark is resolved to
*positions in pixels* (from the tracks), and the plan fits a similarity to them.
The smoothing's own warp is what produces the ``smoothed`` paths and what a
render with no marks falls back to; it is never rebuilt from raw channel values,
and it is never silently in series with a fit.

Why the placement is a fraction and not a pixel
-----------------------------------------------
The output frame's size is not known until the crop is planned, and the crop is
planned from the transforms the placement helps determine. A fraction is the one
statement of "put it here" that survives that ordering — and it is also the one
that survives the operator afterwards deciding to crop harder. The plan resolves
it to pixels exactly once, in :func:`stable.render.warp.plan_output`, which is
the only place that knows the output size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import RenderError
from .fill import BorderFill

__all__ = [
    "TARGET_SOURCES",
    "LEGACY_TARGET_MODES",
    "Placement",
    "StabilizationTarget",
    "TargetPoint",
    "CropSpec",
    "ExportRange",
    "OutputResolution",
    "RenderOptions",
    "average_positions",
    "hold_gaps",
]

TARGET_SOURCES: tuple[str, ...] = ("raw", "smoothed")
"""Which of a track's two paths a target holds on its mark.

``raw``
    The tracked positions themselves — the solid trail on the viewer's canvas.
    Holding these nails the feature: it does not move in the delivered frame,
    and every wobble the tracker saw (real or not) is taken out of the picture.
``smoothed``
    The smoothed estimate for the same track — the **dashed trail** the viewer
    draws beside it, and what ``GET /smoothing/tracks`` serves. Holding these
    nails *that* curve instead, so the fit is driven by a de-noised estimate of
    where the feature is rather than by every sample of it.

The difference is not cosmetic and is worth stating plainly: a ``raw`` target
lands its feature on the mark exactly, every frame. A ``smoothed`` target lands
the *dashed* trail on the mark exactly, and the feature keeps whatever the
smoothing treated as camera motion — a gentler correction, and a picture that is
not driven by tracker noise. Which one is right is the operator's call, per
target, which is why it is a control and not a constant.
"""


@dataclass(frozen=True)
class Placement:
    """Where in the delivered frame a held path is put.

    Args:
        u: Horizontal position as a fraction of the output width, ``0``-``1``.
        v: Vertical position as a fraction of the output height.

    Raises:
        RenderError: If either value is not finite or falls outside ``[0, 1]``.

    A fraction, not a pixel, on purpose — see the module docstring. ``(0.5,
    0.5)`` is the output centre, ``1/3`` and ``2/3`` are the thirds lines, which
    is why the export screen can snap to them without knowing the output size.
    """

    u: float = 0.5
    v: float = 0.5

    def __post_init__(self) -> None:
        """Validate the fractions (dataclass hook)."""
        for name in ("u", "v"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RenderError(
                    f"target placement {name} must be finite, got {getattr(self, name)!r}"
                )
            if not 0.0 <= value <= 1.0:
                raise RenderError(
                    f"target placement {name} is a fraction of the output frame and must be in "
                    f"[0, 1], got {value}"
                )
            object.__setattr__(self, name, value)

    @property
    def fraction(self) -> tuple[float, float]:
        """The placement as a ``(u, v)`` tuple."""
        return (self.u, self.v)

    def pixels(self, size: tuple[float, float]) -> tuple[float, float]:
        """Return this placement in output pixels for an output of ``size``."""
        return (self.u * float(size[0]), self.v * float(size[1]))

    @classmethod
    def from_pixels(cls, x: float, y: float, size: tuple[float, float]) -> Placement:
        """Return the placement of an output-pixel position, clamped to the frame."""
        width, height = float(size[0]), float(size[1])
        if width <= 0 or height <= 0:
            raise RenderError(f"output size must be positive, got {size!r}")
        return cls(
            u=min(max(float(x) / width, 0.0), 1.0),
            v=min(max(float(y) / height, 0.0), 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"u": self.u, "v": self.v}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Placement:
        """Rebuild from :meth:`to_dict` output; missing keys mean the centre."""
        if not data:
            return cls()
        return cls(u=float(data.get("u", 0.5)), v=float(data.get("v", 0.5)))


@dataclass(frozen=True)
class TargetPoint:
    """One mark on the output, and which tracked path is held on it.

    This is the whole stabilisation vocabulary now: a render states one or more
    of these, and the warp is the similarity that best puts every named path on
    its mark. There is no mode to pick and no base warp underneath — what you
    map is what gets fitted, which is why a ``smoothed`` mark lands the dashed
    trail on its mark exactly rather than approximately.

    Args:
        id: A short, stable id, so the panel and the canvas address the same
            mark across a drag, a re-order or a reload.
        place: Where this mark sits in the **output** frame, as a fraction of it.
        point_ids: The tracks whose path is held here. Several means their mean
            position, which is steadier than any one of them. **Empty means
            every solve-enabled tracked point** — the "hold the whole frame
            still" mark, which needs no maintenance as points come and go.
        source: One of :data:`TARGET_SOURCES` — the raw tracked positions, or
            the smoothed estimate for the same tracks.
        weight: How much this mark counts for in the fit, relative to the
            others. Three or more marks generally cannot all be satisfied, so
            saying which one matters most is the difference between a fit and a
            compromise nobody chose.

    Raises:
        RenderError: If the source is unknown, a track is named twice, or the
            weight is not a positive, finite number.
    """

    id: str = "t1"
    place: Placement = field(default_factory=Placement)
    point_ids: tuple[str, ...] = ()
    source: str = "raw"
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the mark (dataclass hook)."""
        if not self.id:
            raise RenderError("a target point needs an id")
        if self.source not in TARGET_SOURCES:
            raise RenderError(
                f"unknown target source {self.source!r}; expected {list(TARGET_SOURCES)}"
            )
        object.__setattr__(self, "point_ids", tuple(self.point_ids))
        if len(set(self.point_ids)) != len(self.point_ids):
            raise RenderError(
                f"a track can only be mapped once to a target point; got {list(self.point_ids)}"
            )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise RenderError(f"a target point's weight must be positive, got {self.weight!r}")
        object.__setattr__(self, "weight", weight)

    @property
    def holds_every_point(self) -> bool:
        """Whether this mark holds the mean of every solve-enabled tracked point."""
        return not self.point_ids

    @property
    def is_smoothed(self) -> bool:
        """Whether the held path is the smoothed estimate rather than the raw track."""
        return self.source == "smoothed"

    def label(self, separator: str = "+") -> str:
        """Return a short human phrase for what this mark holds."""
        what = separator.join(self.point_ids) if self.point_ids else "all tracks"
        if len(self.point_ids) > 1:
            what = f"mean({what})"
        return f"smoothed({what})" if self.is_smoothed else what

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {
            "id": self.id,
            "place": self.place.to_dict(),
            "point_ids": list(self.point_ids),
            "source": self.source,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], frame_size: tuple[int, int] | None = None
    ) -> TargetPoint:
        """Rebuild from :meth:`to_dict` output.

        Args:
            data: The stored mark. ``place`` is the current form; a bare
                ``u``/``v`` or the ``x``/``y`` pixels earlier versions wrote are
                both read too.
            frame_size: The clip's ``(width, height)``, needed only to read a
                legacy pixel position — without it such a mark falls back to the
                output centre rather than being taken for a fraction it is not.
        """
        stored = data.get("place")
        if stored is None and ("u" in data or "v" in data):
            stored = data
        if stored is None and frame_size and "x" in data and "y" in data:
            spot = Placement.from_pixels(
                float(data.get("x", 0.0)), float(data.get("y", 0.0)), frame_size
            )
        else:
            spot = Placement.from_dict(stored)
        # `point_id` is what a single-track mark used to be called.
        ids = data.get("point_ids")
        if ids is None:
            single = data.get("point_id")
            ids = [single] if single else []
        return cls(
            id=str(data.get("id") or "t1"),
            place=spot,
            point_ids=tuple(str(p) for p in ids if p),
            source=str(data.get("source", "raw")),
            weight=float(data.get("weight", 1.0)),
        )


LEGACY_TARGET_MODES: dict[str, str] = {
    # The modes this replaced, and the source each one is read as. `frame` held
    # nothing at all and becomes no marks; every other one becomes exactly one.
    "frame": "",
    "point": "raw",
    "track": "raw",
    "centroid": "raw",
    "average": "raw",
    "selected": "raw",
    "smoothed_track": "smoothed",
}
"""Target modes stored by earlier versions, and the source they are read as."""


@dataclass(frozen=True)
class StabilizationTarget:
    """Every mark the render is fitting to, in the order they were placed.

    Args:
        points: One to N :class:`TargetPoint`. **Empty is a meaningful state**:
            with nothing to fit, the render falls back to undoing what the
            Smoothing panel removed, which is what a project renders as before
            anybody opens the export screen.

    Raises:
        RenderError: If two marks share an id.
    """

    points: tuple[TargetPoint, ...] = ()

    def __post_init__(self) -> None:
        """Validate the set of marks (dataclass hook)."""
        object.__setattr__(self, "points", tuple(self.points))
        ids = [point.id for point in self.points]
        if len(set(ids)) != len(ids):
            raise RenderError(f"every target point needs its own id; got duplicates in {ids}")

    @property
    def is_active(self) -> bool:
        """Whether anything is being fitted at all."""
        return bool(self.points)

    @property
    def uses_rotation_and_scale(self) -> bool:
        """Whether the fit can rotate and scale the picture, not only shift it.

        One correspondence is a translation however it is solved; two or more
        are what let the fit take out camera rotation and zoom as well as
        position. The export screen says which of the two is in force, because
        the difference is invisible in the controls.
        """
        return len(self.points) >= 2

    def next_id(self) -> str:
        """Return an id no existing mark is using."""
        taken = {point.id for point in self.points}
        index = 1
        while f"t{index}" in taken:
            index += 1
        return f"t{index}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"points": [point.to_dict() for point in self.points]}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, frame_size: tuple[int, int] | None = None
    ) -> StabilizationTarget:
        """Rebuild from :meth:`to_dict` output, migrating the shapes before it.

        Three generations of sidecar are read here rather than rejected: this
        list of marks, the single ``mode``/``place`` target that preceded it, and
        the ``mode``/``pairs`` mapping before that. Mode names map onto a source
        through :data:`LEGACY_TARGET_MODES`; ``frame``, which held nothing,
        becomes no marks at all.
        """
        if not data:
            return cls()
        stored = data.get("points")
        if stored is not None:
            return cls(
                points=tuple(
                    TargetPoint.from_dict(p, frame_size) for p in stored if isinstance(p, dict)
                )
            )
        return cls(points=_migrate_target(data, frame_size))


def _migrate_target(
    data: dict[str, Any], frame_size: tuple[int, int] | None
) -> tuple[TargetPoint, ...]:
    """Read a pre-list stabilisation target as a tuple of marks."""
    mode = str(data.get("mode", "frame"))
    source = LEGACY_TARGET_MODES.get(mode, "raw")
    if not source:
        return ()
    pairs = [p for p in (data.get("pairs") or []) if isinstance(p, dict)]
    if pairs:
        return tuple(
            TargetPoint.from_dict({**pair, "id": f"t{index + 1}", "source": "raw"}, frame_size)
            for index, pair in enumerate(pairs)
        )
    ids = [p for p in (data.get("point_ids") or []) if p]
    if not ids and data.get("point_id"):
        ids = [str(data["point_id"])]
    # `centroid` / `average` named no point and meant "all of them", which is
    # exactly what an empty `point_ids` means now.
    if mode in ("centroid", "average"):
        ids = []
    return (
        TargetPoint(
            id="t1",
            place=Placement.from_dict(data.get("place")),
            point_ids=tuple(str(p) for p in ids),
            source=source,
        ),
    )
@dataclass(frozen=True)
class CropSpec:
    """How much of the frame the output keeps, and what "enough" is measured over.

    Args:
        scale: Fraction of the source kept on each axis, in ``(0, 1]``. ``None``
            means "auto": exactly the minimal crop that no frame pokes out of,
            which is what Phase 3 computed and still the default.
        limit_to_range: Measure that automatic crop over the **export range**
            only. Off, it is measured over the whole clip, so one violent second
            at the end of a ten-minute take crops the thirty seconds you are
            actually delivering down to nothing. On, frames outside the export
            markers are not consulted at all — which is the honest reading of
            "the largest crop with no gaps", given that those frames are not
            being written.

    Raises:
        RenderError: If ``scale`` is outside ``(0, 1]``.
    """

    scale: float | None = None
    limit_to_range: bool = False

    def __post_init__(self) -> None:
        """Validate the scale (dataclass hook)."""
        object.__setattr__(self, "limit_to_range", bool(self.limit_to_range))
        if self.scale is not None:
            value = float(self.scale)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise RenderError(f"crop scale must be in (0, 1], got {self.scale!r}")
            object.__setattr__(self, "scale", value)

    @property
    def is_auto(self) -> bool:
        """Whether the crop is the automatically computed minimal one."""
        return self.scale is None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"scale": self.scale, "limit_to_range": self.limit_to_range}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CropSpec:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        if not data:
            return cls()
        scale = data.get("scale")
        return cls(
            scale=(None if scale is None else float(scale)),
            limit_to_range=bool(data.get("limit_to_range", False)),
        )


@dataclass(frozen=True)
class ExportRange:
    """Which frames the render writes, when that is not the In/Out markers.

    Args:
        start: First frame written, inclusive. ``None`` follows the clip's In
            marker.
        end: One past the last frame written, exclusive. ``None`` follows the
            clip's Out marker.

    Raises:
        RenderError: If ``start`` is negative or the range is empty.

    The In/Out markers bound tracking *and* the export, which is right up to the
    moment the operator wants to deliver a different section than the one they
    tracked — tracking wants a margin either side of the shot, delivery does
    not. This is that override, and it is deliberately a separate piece of state
    rather than a second pair of markers on the same timeline: the markers stay
    the tracking range, and this stays a property of the render.
    """

    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        """Validate the bounds (dataclass hook)."""
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, int(value))
        if self.start is not None and self.start < 0:
            raise RenderError(f"the export range must start at frame 0 or later, got {self.start}")
        if self.end is not None and self.end <= (self.start or 0):
            raise RenderError(
                f"the export range must end after it starts, got start={self.start} end={self.end}"
            )

    @property
    def is_custom(self) -> bool:
        """Whether this overrides the clip's In/Out markers at all."""
        return self.start is not None or self.end is not None

    def resolve(self, total: int, fallback: tuple[int, int]) -> tuple[int, int]:
        """Return the half-open ``(start, end)`` frames to write.

        Args:
            total: The clip's length, which the answer is clamped into.
            fallback: The range to use for either end this does not state —
                the In/Out markers, or the whole clip when none are set.

        Returns:
            A non-empty ``(start, end)`` inside ``[0, total)``.
        """
        count = max(1, int(total))
        start = fallback[0] if self.start is None else self.start
        end = fallback[1] if self.end is None else self.end
        start = min(max(0, int(start)), count - 1)
        end = min(max(start + 1, int(end)), count)
        return (start, end)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExportRange:
        """Rebuild from :meth:`to_dict` output; missing keys mean "follow In/Out"."""
        if not data:
            return cls()
        start, end = data.get("start"), data.get("end")
        return cls(
            start=(None if start is None else int(start)),
            end=(None if end is None else int(end)),
        )


@dataclass(frozen=True)
class OutputResolution:
    """The pixel size the crop window is delivered at.

    Args:
        width: Output width in pixels, or ``None`` for the crop's own size.
        height: Output height in pixels, or ``None`` for the crop's own size.

    Raises:
        RenderError: If either side is present and not a positive integer.

    Delivery size and field of view are two different decisions, and keeping
    them separate is what lets an operator crop hard into a shot and still hand
    over a 1920×1080 file. The crop window takes *this* aspect ratio when both
    sides are set, so a square delivery crops a square window out of a 16:9
    source rather than squeezing the picture into one.
    """

    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        """Validate the size (dataclass hook)."""
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is None:
                continue
            number = int(value)
            if number < 2:
                raise RenderError(
                    f"the output {name} must be at least 2 pixels, got {value!r}"
                )
            object.__setattr__(self, name, number)

    @property
    def is_source(self) -> bool:
        """Whether the output is delivered at whatever size the crop came out."""
        return self.width is None and self.height is None

    @property
    def aspect(self) -> tuple[float, float] | None:
        """The shape the crop window should take, or ``None`` to keep the source's."""
        if self.width is None or self.height is None:
            return None
        return (float(self.width), float(self.height))

    def resolve(self, crop_size: tuple[int, int]) -> tuple[int, int]:
        """Return the delivered ``(width, height)`` for a crop of ``crop_size``.

        A side that was left unset follows the crop, scaled by whatever the
        other side asked for — so setting only the width keeps the shape and
        changes the size, which is the common request.
        """
        crop_w, crop_h = max(int(crop_size[0]), 1), max(int(crop_size[1]), 1)
        if self.width is None and self.height is None:
            return (crop_w, crop_h)
        if self.height is None:
            scale = float(self.width or crop_w) / crop_w
            return (int(self.width or crop_w), _even(round(crop_h * scale)))
        if self.width is None:
            scale = float(self.height) / crop_h
            return (_even(round(crop_w * scale)), int(self.height))
        return (int(self.width), int(self.height))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OutputResolution:
        """Rebuild from :meth:`to_dict` output; missing keys mean the crop's size."""
        if not data:
            return cls()
        width, height = data.get("width"), data.get("height")
        return cls(
            width=(None if width is None else int(width)),
            height=(None if height is None else int(height)),
        )


def _even(value: int) -> int:
    """Round ``value`` down to an even number (>= 2); most encoders demand it."""
    return max(int(value) - (int(value) % 2), 2)


@dataclass(frozen=True)
class RenderOptions:
    """The stabilisation choices an operator makes, together.

    Args:
        target: The marks being fitted, and which path is held on each.
        crop: How much of the frame survives, and what that is measured over.
        fill: What goes in the uncovered edge.
        export_range: Which frames get written.
        resolution: What size the crop is delivered at.
    """

    target: StabilizationTarget = field(default_factory=StabilizationTarget)
    crop: CropSpec = field(default_factory=CropSpec)
    fill: BorderFill = field(default_factory=BorderFill)
    export_range: ExportRange = field(default_factory=ExportRange)
    resolution: OutputResolution = field(default_factory=OutputResolution)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form (what a project sidecar stores)."""
        return {
            "target": self.target.to_dict(),
            "crop": self.crop.to_dict(),
            "fill": self.fill.to_dict(),
            "export_range": self.export_range.to_dict(),
            "resolution": self.resolution.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, frame_size: tuple[int, int] | None = None
    ) -> RenderOptions:
        """Rebuild from :meth:`to_dict` output; anything missing takes its default.

        Args:
            data: The stored options.
            frame_size: The clip's ``(width, height)``, used only to read the
                pixel positions earlier versions stored for a mapped target.

        Raises:
            RenderError: If a value is present but invalid.
        """
        if not data:
            return cls()
        return cls(
            target=StabilizationTarget.from_dict(data.get("target"), frame_size),
            crop=CropSpec.from_dict(data.get("crop")),
            fill=BorderFill.from_dict(data.get("fill")),
            export_range=ExportRange.from_dict(data.get("export_range")),
            resolution=OutputResolution.from_dict(data.get("resolution")),
        )


# -- resolving a target to per-frame positions ---------------------------------


def average_positions(tracks: Sequence[np.ndarray]) -> np.ndarray:
    """Return the per-frame mean of several ``[T, 2]`` point paths, ignoring gaps.

    Args:
        tracks: One ``[T, 2]`` array per point; ``NaN`` marks an untracked frame.

    Returns:
        A ``[T, 2]`` float64 array; frames where *every* point is missing stay
        ``NaN`` (:func:`hold_gaps` is what closes those).

    Raises:
        RenderError: If no track is given or they disagree on ``T``.
    """
    if not tracks:
        raise RenderError("an averaged stabilisation target needs at least one tracked point")
    stacked = np.stack([np.asarray(t, dtype=np.float64).reshape(-1, 2) for t in tracks], axis=0)
    if stacked.ndim != 3 or stacked.shape[2] != 2:  # pragma: no cover - reshape guarantees it
        raise RenderError(f"target tracks must be [T, 2], got {stacked.shape}")
    valid = np.asarray(np.isfinite(stacked).all(axis=2))
    counts = np.asarray(valid.sum(axis=0), dtype=np.float64)
    summed = np.asarray(np.where(valid[:, :, None], stacked, 0.0).sum(axis=0), dtype=np.float64)
    out = np.full((stacked.shape[1], 2), np.nan, dtype=np.float64)
    covered = counts > 0
    out[covered] = summed[covered] / counts[covered, None]
    return out


def hold_gaps(positions: np.ndarray) -> np.ndarray:
    """Fill ``NaN`` frames of a ``[T, 2]`` path by holding the nearest valid sample.

    A target that vanishes for a few frames must not make the render lurch back
    to an un-stabilised warp and then snap out of it again: holding the last
    known position keeps the correction continuous, which is the whole point of
    stabilising on it.

    Args:
        positions: ``[T, 2]`` path, possibly with ``NaN`` frames.

    Returns:
        A ``[T, 2]`` float64 copy with no ``NaN``.

    Raises:
        RenderError: If the path has no valid sample at all.
    """
    path = np.asarray(positions, dtype=np.float64).reshape(-1, 2).copy()
    valid = np.isfinite(path).all(axis=1)
    if not valid.any():
        raise RenderError(
            "the stabilisation target has no tracked position anywhere in the clip; "
            "track the point (or pick a different target) before rendering"
        )
    indices = np.where(valid, np.arange(path.shape[0]), -1)
    forward = np.maximum.accumulate(indices)
    backward = np.minimum.accumulate(
        np.where(valid, np.arange(path.shape[0]), path.shape[0])[::-1]
    )[::-1]
    source = np.where(forward >= 0, forward, backward)
    return np.asarray(path[source])
