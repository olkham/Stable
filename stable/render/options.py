"""User-facing stabilisation options: what to hold still, how much to crop, what to fill.

The Phase 3 renderer answered one question — "remove the motion the smoothing
took out" — and answered it well. It left three the operator has to be able to
answer too, and this module is their vocabulary:

**Where should the picture hold still?**
    :class:`StabilizationTarget`. By default the render only undoes what the
    smoothing removed, which means a project with no smoothing configured
    renders a warp of exactly the identity — a re-encode of the source that
    looks, correctly, like nothing happened. A *target* is the other way to ask
    for stabilisation: name a position (the average of every tracked point, or
    one chosen point) and it is pinned to where it sat on an anchor frame.

**How much of the frame survives?**
    :class:`CropSpec`. ``auto`` is Phase 3's exact minimal-gapless crop. An
    explicit ``scale`` overrides it in either direction: crop harder than needed
    for a safety margin, or crop *less* than needed and let the fill deal with
    what is exposed. Its centre can follow the target, so "crop around the thing
    I am stabilising on" means what it says.

**What goes in the exposed edge?**
    :class:`~stable.render.fill.BorderFill` — see that module.

:class:`RenderOptions` bundles the three and round-trips as JSON, so a server or
a CLI can persist an operator's choices without either of them re-deriving what
a valid choice is.

Nothing here touches the pivot-relative curve convention: a target is resolved
to *positions in pixels* (from the tracks), and the plan turns those into an
extra translation composed on top of
:meth:`~stable.smoothing.motion.SmoothedMotion.stabilization_transform`. No
transform is ever rebuilt from raw channel values.
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
    "TARGET_MODES",
    "CROP_CENTRES",
    "StabilizationTarget",
    "TargetPair",
    "CropSpec",
    "RenderOptions",
    "average_positions",
    "hold_gaps",
]

TARGET_MODES: tuple[str, ...] = ("frame", "centroid", "point", "map")
"""What the render holds still.

``frame``
    Nothing extra — the warp is exactly what the smoothing asked for. This is
    Phase 3's behaviour and stays the default.
``centroid``
    The average position of every solve-enabled tracked point.
``point``
    One named point, held wherever it sat on the anchor frame.
``map``
    Named points held at positions the operator *chose* — one pair per point.
    One pair is "put this feature here" (a translation, and the same thing
    ``point`` does with the destination picked for you). Two or more are "put
    this here **and** that there", which a translation cannot do: the fit then
    uses rotation and scale as well, which is how mapping two tracked points
    onto two chosen positions compensates for a rotating, zooming camera.
"""


@dataclass(frozen=True)
class TargetPair:
    """One tracked point and where the output should hold it.

    Args:
        point_id: The tracked point being mapped.
        x: Destination x in **output** pixels.
        y: Destination y in output pixels.

    Raises:
        RenderError: If the point id is empty or the destination is not finite.
    """

    point_id: str
    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate the pair (dataclass hook)."""
        if not self.point_id:
            raise RenderError("a stabilisation target pair needs a point_id")
        for name in ("x", "y"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RenderError(f"target pair {name} must be finite, got {getattr(self, name)!r}")
            object.__setattr__(self, name, value)

    @property
    def position(self) -> tuple[float, float]:
        """The destination as an ``(x, y)`` tuple."""
        return (self.x, self.y)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {"point_id": self.point_id, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetPair:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            point_id=str(data.get("point_id") or ""),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
        )


CROP_CENTRES: tuple[str, ...] = ("frame", "target")
"""Where the output rectangle is centred: the frame's centre, or the target's."""


@dataclass(frozen=True)
class StabilizationTarget:
    """The position the stabilised render holds still.

    Args:
        mode: One of :data:`TARGET_MODES`.
        point_id: Which point, when ``mode`` is ``"point"``.
        anchor_frame: The frame whose target position is held. ``None`` uses the
            solve's reference frame, so "stabilise onto this point" keeps the
            framing the operator chose their reference on rather than inventing
            a new one. Unused by ``map``, which states its destinations outright.
        pairs: The point-to-destination pairs, when ``mode`` is ``"map"``.

    Raises:
        RenderError: If the mode is unknown, ``point`` names no point, or
            ``map`` has no pairs or names the same point twice.
    """

    mode: str = "frame"
    point_id: str | None = None
    anchor_frame: int | None = None
    pairs: tuple[TargetPair, ...] = ()

    def __post_init__(self) -> None:
        """Validate the mode (dataclass hook)."""
        if self.mode not in TARGET_MODES:
            raise RenderError(
                f"unknown stabilisation target {self.mode!r}; expected {list(TARGET_MODES)}"
            )
        if self.mode == "point" and not self.point_id:
            raise RenderError("stabilisation target 'point' needs a point_id")
        if self.anchor_frame is not None and int(self.anchor_frame) < 0:
            raise RenderError(f"anchor_frame must be >= 0, got {self.anchor_frame}")
        object.__setattr__(self, "pairs", tuple(self.pairs))
        if self.mode == "map":
            if not self.pairs:
                raise RenderError(
                    "a 'map' stabilisation target needs at least one point-to-target pair"
                )
            ids = [pair.point_id for pair in self.pairs]
            if len(set(ids)) != len(ids):
                raise RenderError(
                    f"each point can be mapped to one target position; got duplicates in {ids}"
                )

    @property
    def is_active(self) -> bool:
        """Whether this target adds anything to the smoothing's own warp."""
        return self.mode != "frame"

    @property
    def uses_rotation_and_scale(self) -> bool:
        """Whether this target can rotate/scale the picture, not only shift it.

        One correspondence is a translation however it is solved; two or more
        are what make a mapping compensate for camera rotation and zoom. The
        export screen says which of the two the operator has asked for, because
        the difference is not visible in the controls.
        """
        return self.mode == "map" and len(self.pairs) >= 2

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {
            "mode": self.mode,
            "point_id": self.point_id,
            "anchor_frame": self.anchor_frame,
            "pairs": [pair.to_dict() for pair in self.pairs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StabilizationTarget:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        if not data:
            return cls()
        anchor = data.get("anchor_frame")
        raw_pairs = data.get("pairs") or []
        return cls(
            mode=str(data.get("mode", "frame")),
            point_id=(str(data["point_id"]) if data.get("point_id") else None),
            anchor_frame=(None if anchor is None else int(anchor)),
            pairs=tuple(TargetPair.from_dict(p) for p in raw_pairs if isinstance(p, dict)),
        )


@dataclass(frozen=True)
class CropSpec:
    """How much of the frame the output keeps, and where that rectangle sits.

    Args:
        scale: Fraction of the source kept on each axis, in ``(0, 1]``. ``None``
            means "auto": exactly the minimal crop that no frame pokes out of,
            which is what Phase 3 computed and still the default.
        centre: One of :data:`CROP_CENTRES`. ``"target"`` centres the output on
            the stabilisation target instead of the frame centre; with no active
            target it falls back to the frame centre, since a moving centre
            would re-frame the shot every frame.

    Raises:
        RenderError: If ``scale`` is outside ``(0, 1]`` or ``centre`` is unknown.
    """

    scale: float | None = None
    centre: str = "frame"

    def __post_init__(self) -> None:
        """Validate the scale and centre (dataclass hook)."""
        if self.centre not in CROP_CENTRES:
            raise RenderError(f"unknown crop centre {self.centre!r}; expected {list(CROP_CENTRES)}")
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
        return {"scale": self.scale, "centre": self.centre}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CropSpec:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        if not data:
            return cls()
        scale = data.get("scale")
        return cls(
            scale=(None if scale is None else float(scale)),
            centre=str(data.get("centre", "frame")),
        )


@dataclass(frozen=True)
class RenderOptions:
    """The three stabilisation choices an operator makes, together.

    Args:
        target: What the render holds still.
        crop: How much of the frame survives, and where it is centred.
        fill: What goes in the uncovered edge.
    """

    target: StabilizationTarget = field(default_factory=StabilizationTarget)
    crop: CropSpec = field(default_factory=CropSpec)
    fill: BorderFill = field(default_factory=BorderFill)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form (what a project sidecar stores)."""
        return {
            "target": self.target.to_dict(),
            "crop": self.crop.to_dict(),
            "fill": self.fill.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RenderOptions:
        """Rebuild from :meth:`to_dict` output; anything missing takes its default.

        Raises:
            RenderError: If a value is present but invalid.
        """
        if not data:
            return cls()
        return cls(
            target=StabilizationTarget.from_dict(data.get("target")),
            crop=CropSpec.from_dict(data.get("crop")),
            fill=BorderFill.from_dict(data.get("fill")),
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
        raise RenderError("a centroid stabilisation target needs at least one tracked point")
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
