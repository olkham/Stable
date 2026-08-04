"""2D similarity geometry — the only motion model Stable solves for.

A :class:`SimilarityTransform` has four degrees of freedom that map one-to-one
onto the ``X, Y, R, S`` channels used everywhere downstream (solver, smoothing,
render, project file, exports).

Conventions (fixed; everything downstream assumes them)
-------------------------------------------------------
* Coordinates are **pixels in image space**, origin at the top-left corner of
  the frame, ``x`` right and ``y`` down. Sub-pixel positions are expected.
* The mapping is ``p' = s * R(theta) @ p + t`` with
  ``R(theta) = [[cos, -sin], [sin, cos]]``. Because ``y`` points down, a
  *positive* ``theta`` appears **clockwise** on screen.
* ``rotation`` is stored and serialised in **radians**. ``rotation_deg`` exists
  for display only; no other module should convert units.
* Rotation and scale are about the **origin**. To build a transform that rotates
  about an arbitrary pivot (e.g. the frame centre) use :meth:`about_pivot`,
  which folds the pivot into the translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ContractError

__all__ = ["SimilarityTransform", "CHANNELS", "Channel"]

Channel = str
"""One of the four solved motion channels: ``"x"``, ``"y"``, ``"r"``, ``"s"``."""

CHANNELS: tuple[str, str, str, str] = ("x", "y", "r", "s")
"""Canonical channel order: translation x, translation y, rotation, scale."""


@dataclass(frozen=True)
class SimilarityTransform:
    """An immutable 2D similarity: translation, rotation and uniform scale.

    Args:
        tx: Translation along x, in pixels.
        ty: Translation along y, in pixels.
        rotation: Rotation in radians (positive = clockwise on screen).
        scale: Uniform scale factor; must be finite and non-zero.

    Raises:
        ContractError: If any component is non-finite or ``scale`` is zero.
    """

    tx: float = 0.0
    ty: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate the four components (dataclass hook)."""
        for field_name in ("tx", "ty", "rotation", "scale"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ContractError(
                    f"SimilarityTransform.{field_name} must be finite, got {value!r}"
                )
            object.__setattr__(self, field_name, value)
        if self.scale == 0.0:
            raise ContractError("SimilarityTransform.scale must be non-zero")

    # -- constructors ---------------------------------------------------

    @classmethod
    def identity(cls) -> SimilarityTransform:
        """Return the identity transform (no translation, rotation or scale)."""
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, *, tol: float = 1e-4) -> SimilarityTransform:
        """Build a transform from a ``2x3`` or ``3x3`` affine matrix.

        The matrix must actually be a similarity: the linear part has to be a
        scaled rotation (``a == d`` and ``b == -c``).

        Args:
            matrix: Affine matrix, shape ``(2, 3)`` or ``(3, 3)``.
            tol: Absolute tolerance on the similarity check, relative to scale.

        Returns:
            The equivalent :class:`SimilarityTransform`.

        Raises:
            ContractError: If the shape is wrong or the linear part is not a
                scaled rotation within ``tol``.
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (3, 3):
            if not np.allclose(m[2], (0.0, 0.0, 1.0), atol=1e-9):
                raise ContractError("3x3 matrix must have a [0, 0, 1] bottom row (affine only)")
            m = m[:2]
        if m.shape != (2, 3):
            raise ContractError(f"expected a (2, 3) or (3, 3) matrix, got {m.shape}")
        a, b, tx = m[0]
        c, d, ty = m[1]
        cos_s = (a + d) / 2.0
        sin_s = (c - b) / 2.0
        scale = math.hypot(cos_s, sin_s)
        if scale == 0.0:
            raise ContractError("degenerate matrix: zero scale")
        if abs(a - d) > tol * scale or abs(b + c) > tol * scale:
            raise ContractError(
                "matrix linear part is not a scaled rotation "
                f"(a={a:.6g}, b={b:.6g}, c={c:.6g}, d={d:.6g})"
            )
        return cls(float(tx), float(ty), math.atan2(sin_s, cos_s), scale)

    @classmethod
    def about_pivot(
        cls,
        pivot: tuple[float, float] | np.ndarray,
        *,
        rotation: float = 0.0,
        scale: float = 1.0,
        translation: tuple[float, float] = (0.0, 0.0),
    ) -> SimilarityTransform:
        """Rotate/scale about ``pivot``, then translate.

        Equivalent to ``T(translation) . T(pivot) . R(rotation)S(scale) . T(-pivot)``
        expressed as a single origin-based similarity.

        Args:
            pivot: ``(x, y)`` centre of rotation/scaling, in pixels.
            rotation: Rotation in radians (positive = clockwise on screen).
            scale: Uniform scale factor.
            translation: Extra translation applied after the pivoted rotation.

        Returns:
            The equivalent origin-based :class:`SimilarityTransform`.
        """
        px, py = float(pivot[0]), float(pivot[1])
        cos_s = scale * math.cos(rotation)
        sin_s = scale * math.sin(rotation)
        tx = translation[0] + px - (cos_s * px - sin_s * py)
        ty = translation[1] + py - (sin_s * px + cos_s * py)
        return cls(tx, ty, rotation, scale)

    @classmethod
    def from_channels(cls, channels: dict[str, float]) -> SimilarityTransform:
        """Rebuild a transform from a ``{"x", "y", "r", "s"}`` mapping.

        Args:
            channels: Mapping with keys ``x``, ``y``, ``r`` (radians) and ``s``.

        Returns:
            The corresponding :class:`SimilarityTransform`.

        Raises:
            ContractError: If a channel key is missing.
        """
        try:
            return cls(
                float(channels["x"]),
                float(channels["y"]),
                float(channels["r"]),
                float(channels["s"]),
            )
        except KeyError as exc:  # pragma: no cover - message clarity only
            raise ContractError(f"missing channel {exc.args[0]!r}; expected {CHANNELS}") from exc

    # -- accessors ------------------------------------------------------

    @property
    def rotation_deg(self) -> float:
        """Rotation in degrees. Display convenience only — storage is radians."""
        return math.degrees(self.rotation)

    @property
    def translation(self) -> tuple[float, float]:
        """Translation as an ``(x, y)`` tuple, in pixels."""
        return (self.tx, self.ty)

    def to_matrix(self) -> np.ndarray:
        """Return the ``(2, 3)`` float64 affine matrix for this transform."""
        cos_s = self.scale * math.cos(self.rotation)
        sin_s = self.scale * math.sin(self.rotation)
        return np.array(
            [[cos_s, -sin_s, self.tx], [sin_s, cos_s, self.ty]],
            dtype=np.float64,
        )

    def to_matrix3(self) -> np.ndarray:
        """Return the homogeneous ``(3, 3)`` float64 matrix for this transform."""
        m = np.eye(3, dtype=np.float64)
        m[:2] = self.to_matrix()
        return m

    def to_channels(self) -> dict[str, float]:
        """Return ``{"x", "y", "r", "s"}`` — ``r`` in radians, ``s`` unitless."""
        return {"x": self.tx, "y": self.ty, "r": self.rotation, "s": self.scale}

    # -- algebra --------------------------------------------------------

    def inverse(self) -> SimilarityTransform:
        """Return the transform that undoes this one.

        ``t.inverse().compose(t)`` is the identity (within float precision).
        """
        inv_scale = 1.0 / self.scale
        cos_i = math.cos(-self.rotation) * inv_scale
        sin_i = math.sin(-self.rotation) * inv_scale
        return SimilarityTransform(
            tx=-(cos_i * self.tx - sin_i * self.ty),
            ty=-(sin_i * self.tx + cos_i * self.ty),
            rotation=-self.rotation,
            scale=inv_scale,
        )

    def compose(self, other: SimilarityTransform) -> SimilarityTransform:
        """Return ``self . other`` — ``other`` is applied first, then ``self``.

        Args:
            other: The transform applied first.

        Returns:
            The composed similarity, so that
            ``a.compose(b).apply(p) == a.apply(b.apply(p))``.
        """
        cos_a = math.cos(self.rotation)
        sin_a = math.sin(self.rotation)
        s = self.scale
        return SimilarityTransform(
            tx=self.tx + s * (cos_a * other.tx - sin_a * other.ty),
            ty=self.ty + s * (sin_a * other.tx + cos_a * other.ty),
            rotation=self.rotation + other.rotation,
            scale=self.scale * other.scale,
        )

    def __matmul__(self, other: SimilarityTransform) -> SimilarityTransform:
        """Operator form of :meth:`compose` (``a @ b`` applies ``b`` first)."""
        if not isinstance(other, SimilarityTransform):
            return NotImplemented
        return self.compose(other)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform an array of points.

        Args:
            points: Float array of shape ``(..., 2)``. NaNs propagate.

        Returns:
            Float64 array of the same shape holding the transformed points.

        Raises:
            ContractError: If the trailing dimension is not 2.
        """
        p = np.asarray(points, dtype=np.float64)
        if p.ndim == 0 or p.shape[-1] != 2:
            raise ContractError(f"points must have shape (..., 2), got {p.shape}")
        m = self.to_matrix()
        return p @ m[:, :2].T + m[:, 2]

    @staticmethod
    def lerp(a: SimilarityTransform, b: SimilarityTransform, t: float) -> SimilarityTransform:
        """Interpolate between two transforms.

        Translation is linear, rotation takes the shortest angular path, and
        scale is interpolated geometrically (linear in log space) so that
        halfway between 1x and 4x is 2x.

        Args:
            a: Transform at ``t == 0``.
            b: Transform at ``t == 1``.
            t: Interpolation parameter; values outside ``[0, 1]`` extrapolate.

        Returns:
            The interpolated :class:`SimilarityTransform`.
        """
        delta = (b.rotation - a.rotation + math.pi) % (2.0 * math.pi) - math.pi
        return SimilarityTransform(
            tx=a.tx + (b.tx - a.tx) * t,
            ty=a.ty + (b.ty - a.ty) * t,
            rotation=a.rotation + delta * t,
            scale=math.exp(math.log(abs(a.scale)) * (1.0 - t) + math.log(abs(b.scale)) * t)
            * (1.0 if a.scale > 0 else -1.0),
        )

    def allclose(self, other: SimilarityTransform, *, atol: float = 1e-6) -> bool:
        """Return True if ``other`` matches this transform within ``atol``.

        Args:
            other: Transform to compare against.
            atol: Absolute tolerance applied to every component (rotation is
                compared modulo a full turn).

        Returns:
            Whether the two transforms are equal within tolerance.
        """
        delta = (other.rotation - self.rotation + math.pi) % (2.0 * math.pi) - math.pi
        return (
            abs(other.tx - self.tx) <= atol
            and abs(other.ty - self.ty) <= atol
            and abs(delta) <= atol
            and abs(other.scale - self.scale) <= atol
        )

    def is_identity(self, *, atol: float = 1e-9) -> bool:
        """Return True if this transform is the identity within ``atol``."""
        return self.allclose(SimilarityTransform.identity(), atol=atol)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Return the JSON-safe dict form (same keys as :meth:`to_channels`)."""
        return self.to_channels()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimilarityTransform:
        """Rebuild a transform from :meth:`to_dict` output."""
        return cls.from_channels({k: float(v) for k, v in data.items() if k in CHANNELS})
