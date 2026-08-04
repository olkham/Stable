"""What goes in the exposed edges: solid colour, blur, stretch, mirror — or a mix.

Moving a frame to stabilise it necessarily uncovers area the camera never shot.
:mod:`stable.render.warp`'s three border *policies* answer "how much of the
frame do we keep"; this module answers the other half — "what do we put in the
part that has no source pixels", which is the difference between a shot that
can be delivered without a crop and one that cannot.

The four plates
---------------
``solid``
    A flat colour. Black is the classic letterbox; any RGB works.
``stretch``
    The frame's outermost row/column smeared outwards (``BORDER_REPLICATE``).
    Reads as a natural continuation on soft edges (sky, walls), streaky on busy
    ones.
``mirror``
    The frame reflected across its own edge (``BORDER_REFLECT_101``). Keeps real
    texture and detail out there, at the cost of a visible symmetry.
``blur``
    The mirrored plate, blurred. The usual choice: no symmetry to spot, no hard
    seam, and the colours match the frame because they *are* the frame.

Combinations
------------
The four are a *base plate*, and two modifiers stack on top of any of them, so
"a combination of these" is expressible rather than a fifth mode:

* ``blur`` — Gaussian sigma applied to the plate (``mirror`` + blur is the
  ``blur`` mode; ``stretch`` + blur softens the streaks).
* ``tint`` — blend the plate toward ``color`` (e.g. a mirrored edge faded 60%
  into black, which reads as a vignette rather than a border).
* ``feather`` — soften the join itself, so the eye cannot find the boundary.

Cost
----
The plain case (solid fill, no feather) is exactly one ``warpAffine`` — the same
call the renderer always made. Anything else costs two more warps (the plate and
a coverage mask) and a blend, which is why :meth:`BorderFill.is_plain` exists and
the fast path is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import RenderError

__all__ = ["FILL_MODES", "BorderFill", "composite_fill"]

FILL_MODES: tuple[str, ...] = ("solid", "stretch", "mirror", "blur")
"""Base plates for the uncovered area, in increasing order of cleverness."""


def _clamp01(value: float, name: str) -> float:
    """Return ``value`` as a float in ``[0, 1]``, or raise."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RenderError(f"BorderFill.{name} must be between 0 and 1, got {value!r}")
    return number


@dataclass(frozen=True)
class BorderFill:
    """How the uncovered edge area is filled.

    Args:
        mode: Base plate, one of :data:`FILL_MODES`.
        color: RGB fill for ``solid``, and the colour ``tint`` blends toward.
        blur: Gaussian sigma in output pixels applied to the plate. ``0``
            leaves it sharp; the ``blur`` mode defaults it to a sensible
            fraction of the frame when left at ``0`` (see :meth:`plate_sigma`).
        tint: ``0``-``1`` blend of the plate toward ``color``.
        feather: Width in pixels of the soft join between real pixels and the
            plate. ``0`` is a hard edge.

    Raises:
        RenderError: If the mode is unknown or a number is out of range.
    """

    mode: str = "solid"
    color: tuple[int, int, int] = (0, 0, 0)
    blur: float = 0.0
    tint: float = 0.0
    feather: float = 0.0

    def __post_init__(self) -> None:
        """Validate the mode and coerce the numbers (dataclass hook)."""
        if self.mode not in FILL_MODES:
            raise RenderError(f"unknown border fill {self.mode!r}; expected {list(FILL_MODES)}")
        rgb = tuple(int(np.clip(int(c), 0, 255)) for c in self.color)
        if len(rgb) != 3:
            raise RenderError(f"BorderFill.color must be an RGB triple, got {self.color!r}")
        object.__setattr__(self, "color", rgb)
        for name in ("blur", "feather"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise RenderError(f"BorderFill.{name} must be >= 0, got {value!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tint", _clamp01(self.tint, "tint"))

    @property
    def is_plain(self) -> bool:
        """Whether this is the plain constant-colour fill the renderer always did.

        True means one ``warpAffine`` with ``BORDER_CONSTANT`` reproduces it
        exactly, so :func:`composite_fill` can be skipped entirely.
        """
        return self.mode == "solid" and self.feather == 0.0 and self.tint == 0.0

    def plate_sigma(self, width: int, height: int) -> float:
        """Return the Gaussian sigma actually used for the plate, in pixels.

        ``blur`` mode with no explicit sigma picks 4% of the frame's short side —
        enough that no detail survives to give the mirror away, and scale-free so
        it looks the same on a proxy as on the delivery render.
        """
        if self.blur > 0.0:
            return float(self.blur)
        if self.mode == "blur":
            return max(2.0, 0.04 * min(int(width), int(height)))
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict form."""
        return {
            "mode": self.mode,
            "color": list(self.color),
            "blur": self.blur,
            "tint": self.tint,
            "feather": self.feather,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BorderFill:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults.

        Raises:
            RenderError: If a value is present but unusable.
        """
        if not data:
            return cls()
        color = data.get("color", (0, 0, 0))
        try:
            rgb = (int(color[0]), int(color[1]), int(color[2]))
        except (TypeError, IndexError, ValueError) as exc:
            raise RenderError(f"BorderFill.color must be an RGB triple, got {color!r}") from exc
        return cls(
            mode=str(data.get("mode", "solid")),
            color=rgb,
            blur=float(data.get("blur", 0.0) or 0.0),
            tint=float(data.get("tint", 0.0) or 0.0),
            feather=float(data.get("feather", 0.0) or 0.0),
        )


def _border_mode(cv2: Any, mode: str) -> int:
    """Return the OpenCV ``borderMode`` that produces ``mode``'s plate directly."""
    return {
        "solid": cv2.BORDER_CONSTANT,
        "stretch": cv2.BORDER_REPLICATE,
        "mirror": cv2.BORDER_REFLECT_101,
        "blur": cv2.BORDER_REFLECT_101,
    }[mode]


def composite_fill(
    cv2: Any,
    source: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int],
    *,
    flag: int,
    fill: BorderFill,
) -> np.ndarray:
    """Warp ``source`` and fill the uncovered area according to ``fill``.

    The real pixels are warped once with ``BORDER_REFLECT_101`` (so the outermost
    row does not blend toward a constant and leave a dark seam), the plate is
    warped once more with the mode's own border mode, and a coverage mask —
    a white frame warped with ``BORDER_CONSTANT`` — decides which is which.

    Args:
        cv2: The imported OpenCV module.
        source: ``uint8`` RGB ``(H, W, 3)`` source frame.
        matrix: The ``(2, 3)`` forward affine.
        size: Output ``(width, height)``.
        flag: OpenCV interpolation flag.
        fill: The fill specification.

    Returns:
        ``uint8`` RGB ``(out_h, out_w, 3)``.
    """
    out_w, out_h = int(size[0]), int(size[1])
    colour = np.asarray(fill.color, dtype=np.float32)

    def warp(image: np.ndarray, border_mode: int, value: Any = 0) -> np.ndarray:
        return np.asarray(
            cv2.warpAffine(
                image, matrix, (out_w, out_h), flags=flag, borderMode=border_mode,
                borderValue=value,
            )
        )

    content = warp(source, cv2.BORDER_REFLECT_101).astype(np.float32)

    if fill.mode == "solid":
        plate = np.empty((out_h, out_w, 3), dtype=np.float32)
        plate[:] = colour
    elif fill.mode == "mirror" or fill.mode == "blur":
        plate = content.copy()
    else:  # stretch
        plate = warp(source, _border_mode(cv2, "stretch")).astype(np.float32)

    sigma = fill.plate_sigma(out_w, out_h)
    if sigma > 0.0 and fill.mode != "solid":
        plate = np.asarray(cv2.GaussianBlur(plate, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE))
    if fill.tint > 0.0:
        plate = plate * (1.0 - fill.tint) + colour * fill.tint

    ones = np.full(source.shape[:2], 255, dtype=np.uint8)
    mask = warp(ones, cv2.BORDER_CONSTANT, 0).astype(np.float32) / 255.0
    if fill.feather > 0.0:
        # Erode first, so the softened join eats into the real pixels instead of
        # smearing the plate over them (which would haze the whole frame edge).
        radius = max(1, int(round(fill.feather)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = np.asarray(cv2.erode(mask, kernel))
        mask = np.asarray(cv2.GaussianBlur(mask, (0, 0), max(fill.feather / 2.0, 0.5)))
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]

    blended = content * mask + plate * (1.0 - mask)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
