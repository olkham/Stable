"""Per-frame inverse warp and the three border policies.

Stabilising a frame is a single affine — ``smoothed(t) . raw(t)^-1``, composed
once by :mod:`stable.smoothing` — pushed through ``cv2.warpAffine``. What makes
the difference between "stabilised" and "shippable" is what happens at the
edges, because moving the frame necessarily uncovers area the camera never shot.

The three policies
------------------
``crop``
    Output the largest centred rectangle of the original aspect ratio that
    every frame covers. Nothing is scaled, so the picture stays pixel-exact;
    the output is simply smaller than the input.
``auto_zoom``
    The same rectangle, scaled back up to the original size — the *minimal*
    zoom that covers every frame. This is the default: same delivery
    resolution, guaranteed no gaps, and the zoom is only ever as much as the
    shot actually needed.
``transparent``
    No cropping at all. The output is RGBA with alpha 0 wherever no source
    pixel exists, for compositing the gaps in another application.

Finding the minimal zoom
------------------------
For each frame the warped source rectangle is a convex quad (a similarity
cannot bend it). :func:`coverage_scale` asks, for every frame, how far a
centred rectangle of the source aspect ratio can grow before it pokes out of
that quad — a linear-programming question with four constraints per edge that
has a closed form — and takes the tightest answer over the whole clip. The
zoom is the reciprocal. Because it is exact rather than iterative, "minimal
zoom covering all frames" is a guarantee, not a search that might stop early.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import RenderError
from ..core.geometry import SimilarityTransform
from ..core.project import BORDER_POLICIES

__all__ = [
    "INTERPOLATIONS",
    "OutputGeometry",
    "coverage_scale",
    "plan_output",
    "warp_frame",
    "rescale_transform",
    "cuda_available",
    "load_cv2",
]

INTERPOLATIONS: tuple[str, ...] = ("nearest", "linear", "cubic", "lanczos")
"""Resampling kernels, coarsest to finest. ``linear`` is the sane default."""


def load_cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Deferring the import means merely importing :mod:`stable.render` never
    requires OpenCV — only actually warping a frame does.

    Returns:
        The imported ``cv2`` module.

    Raises:
        RenderError: If OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RenderError(
            "the stabilisation renderer requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


def _interpolation_flag(cv2: Any, name: str) -> int:
    """Map an :data:`INTERPOLATIONS` name onto an OpenCV flag.

    Raises:
        RenderError: If ``name`` is not a known kernel.
    """
    flags = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    try:
        return int(flags[name])
    except KeyError as exc:
        raise RenderError(
            f"unknown interpolation {name!r}; expected one of {list(INTERPOLATIONS)}"
        ) from exc


def cuda_available() -> bool:
    """Whether OpenCV was built with a usable CUDA device attached.

    The GPU warp path is an optimisation, never a requirement: every caller
    falls back to the CPU silently when this is False.
    """
    try:
        cv2 = load_cv2()
        return bool(cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:  # pragma: no cover - depends on the OpenCV build
        return False


@dataclass(frozen=True)
class OutputGeometry:
    """Where the stabilised pixels land, and how much had to be given up.

    Args:
        width: Output frame width in pixels.
        height: Output frame height in pixels.
        fit: The extra similarity applied *after* the stabilisation warp — the
            crop offset or the auto-zoom, expressed so the renderer still does
            exactly one affine per frame.
        coverage: Largest fraction of the frame, in ``(0, 1]``, that every
            stabilised frame covers. ``1.0`` means no zoom or crop was needed.
        zoom: The scale factor ``fit`` applies; ``1 / coverage`` for auto-zoom.
        border: The policy this geometry implements.
        clamped: True when ``max_zoom`` capped the zoom, so gaps may remain.
    """

    width: int
    height: int
    fit: SimilarityTransform
    coverage: float
    zoom: float
    border: str
    clamped: bool = False

    @property
    def size(self) -> tuple[int, int]:
        """Output size as ``(width, height)``."""
        return (self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the output geometry."""
        return {
            "width": self.width,
            "height": self.height,
            "border": self.border,
            "coverage": self.coverage,
            "zoom": self.zoom,
            "clamped": self.clamped,
            "fit": self.fit.to_dict(),
        }


def rescale_transform(transform: SimilarityTransform, scale: float) -> SimilarityTransform:
    """Re-express a warp for frames at a different pixel resolution.

    A stabilisation warp is solved in full-resolution pixels, but a UI preview
    warps a half- or quarter-res proxy. Conjugating by the resolution change
    (``S . W . S^-1``) leaves rotation and the similarity's own scale untouched —
    both are dimensionless — and scales only the translation.

    Args:
        transform: The warp in full-resolution pixel coordinates.
        scale: Resolution ratio of the frames to be warped, e.g. ``0.5`` for a
            half-res proxy.

    Returns:
        The equivalent warp in the proxy's pixel coordinates.

    Raises:
        RenderError: If ``scale`` is not positive.
    """
    if scale <= 0.0:
        raise RenderError(f"resolution scale must be positive, got {scale}")
    return SimilarityTransform(
        tx=transform.tx * scale,
        ty=transform.ty * scale,
        rotation=transform.rotation,
        scale=transform.scale,
    )


def _frame_quad(transform: SimilarityTransform, width: int, height: int) -> np.ndarray:
    """Return the four corners of the source frame after ``transform``."""
    corners = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    return np.asarray(transform.apply(corners), dtype=np.float64)


def coverage_scale(
    transforms: Sequence[SimilarityTransform],
    width: int,
    height: int,
    *,
    centre: tuple[float, float] | None = None,
) -> float:
    """Return the largest centred rectangle every warped frame fully covers.

    Args:
        transforms: One stabilisation warp per frame.
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        centre: Centre of the rectangle in output coordinates; defaults to the
            frame centre. A centred crop is what keeps the framing where the
            operator put it — an off-centre one would silently re-frame the shot.

    Returns:
        A factor in ``[0, 1]``: the rectangle is ``coverage * width`` by
        ``coverage * height``. ``1.0`` means no crop is needed at all; ``0.0``
        means the frames share no common area.
    """
    if width <= 0 or height <= 0:
        raise RenderError(f"frame size must be positive, got {width}x{height}")
    if not transforms:
        return 1.0
    origin = (
        np.array([width / 2.0, height / 2.0], dtype=np.float64)
        if centre is None
        else np.asarray(centre, dtype=np.float64)
    )
    # Corner directions of a centred rectangle with the source aspect ratio.
    directions = np.array(
        [
            [-width / 2.0, -height / 2.0],
            [width / 2.0, -height / 2.0],
            [width / 2.0, height / 2.0],
            [-width / 2.0, height / 2.0],
        ],
        dtype=np.float64,
    )

    best = 1.0
    for transform in transforms:
        quad = _frame_quad(transform, width, height)
        quad_centre = quad.mean(axis=0)
        for index in range(4):
            start, end = quad[index], quad[(index + 1) % 4]
            edge = end - start
            normal = np.array([edge[1], -edge[0]], dtype=np.float64)
            length = float(np.hypot(normal[0], normal[1]))
            if length == 0.0:  # pragma: no cover - a degenerate (zero-scale) warp
                continue
            normal /= length
            if float(normal @ (quad_centre - start)) > 0.0:
                normal = -normal  # make it point out of the quad
            slack = float(normal @ (start - origin))
            if slack <= 0.0:
                return 0.0  # the crop centre itself is outside this frame
            for direction in directions:
                projection = float(normal @ direction)
                if projection > 0.0:
                    best = min(best, slack / projection)
    if best >= 1.0 - 1e-9:
        # Snap to exactly 1: a shot that needed no crop must not pick up a
        # 1.0000000000000004 zoom and resample every frame for nothing.
        return 1.0
    return float(max(best, 0.0))


def _even(value: int) -> int:
    """Round ``value`` down to an even number (>= 2); most encoders demand it."""
    return max(int(value) - (int(value) % 2), 2)


def plan_output(
    transforms: Sequence[SimilarityTransform],
    width: int,
    height: int,
    border: str = "auto_zoom",
    *,
    max_zoom: float | None = None,
) -> OutputGeometry:
    """Work out the output size and fit transform for a border policy.

    Args:
        transforms: One stabilisation warp per frame.
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        border: One of :data:`~stable.core.project.BORDER_POLICIES`.
        max_zoom: Optional cap on the auto-zoom factor. When the shot needs more
            than this, the zoom stops there and the result is marked
            :attr:`OutputGeometry.clamped` — visible gaps, but the user keeps
            the framing they asked for rather than losing half the frame.

    Returns:
        The :class:`OutputGeometry`.

    Raises:
        RenderError: If the policy is unknown, the size is not positive, or the
            frames share no common area (crop and auto-zoom only).
    """
    if border not in BORDER_POLICIES:
        raise RenderError(f"unknown border policy {border!r}; expected {list(BORDER_POLICIES)}")
    if width <= 0 or height <= 0:
        raise RenderError(f"frame size must be positive, got {width}x{height}")

    if border == "transparent":
        # Nothing is hidden, so nothing needs measuring: the gaps become alpha.
        return OutputGeometry(
            width=int(width),
            height=int(height),
            fit=SimilarityTransform.identity(),
            coverage=1.0,
            zoom=1.0,
            border=border,
        )

    coverage = coverage_scale(transforms, width, height)
    if coverage <= 0.0:
        raise RenderError(
            "the stabilised frames share no common area, so there is nothing to crop to; "
            "reduce the smoothing strength, disable a channel, or use the 'transparent' border"
        )

    centre = (width / 2.0, height / 2.0)
    if border == "crop":
        crop_w = _even(int(coverage * width))
        crop_h = _even(int(coverage * height))
        offset = SimilarityTransform(
            tx=-(width - crop_w) / 2.0,
            ty=-(height - crop_h) / 2.0,
        )
        return OutputGeometry(
            width=crop_w,
            height=crop_h,
            fit=offset,
            coverage=coverage,
            zoom=1.0,
            border=border,
        )

    zoom = 1.0 / coverage
    clamped = False
    if max_zoom is not None and zoom > float(max_zoom):
        if float(max_zoom) < 1.0:
            raise RenderError(f"max_zoom must be >= 1, got {max_zoom}")
        zoom, clamped = float(max_zoom), True
    return OutputGeometry(
        width=int(width),
        height=int(height),
        fit=SimilarityTransform.about_pivot(centre, scale=zoom),
        coverage=coverage,
        zoom=zoom,
        border=border,
        clamped=clamped,
    )


def _warp_cuda(cv2: Any, source: np.ndarray, matrix: np.ndarray, size: tuple[int, int],
               flag: int, border_value: tuple[float, ...]) -> np.ndarray:
    """Warp one frame on the GPU. Raises if the CUDA path is unusable."""
    gpu = cv2.cuda_GpuMat()
    gpu.upload(source)
    warped = cv2.cuda.warpAffine(
        gpu,
        np.asarray(matrix, dtype=np.float64),
        size,
        flags=flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return np.asarray(warped.download())


def warp_frame(
    frame: np.ndarray,
    transform: SimilarityTransform,
    *,
    size: tuple[int, int] | None = None,
    border: str = "crop",
    interpolation: str = "linear",
    background: tuple[int, int, int] = (0, 0, 0),
    use_gpu: bool = False,
) -> np.ndarray:
    """Apply one stabilisation affine to one frame.

    Args:
        frame: ``uint8`` RGB ``(H, W, 3)`` source frame.
        transform: The complete output transform for this frame — the
            stabilisation warp with the crop/zoom fit already composed in, so
            exactly one resample happens.
        size: Output ``(width, height)``; defaults to the source size.
        border: ``"transparent"`` returns RGBA with alpha 0 in the uncovered
            area; any other policy fills it with ``background``.
        interpolation: One of :data:`INTERPOLATIONS`.
        background: RGB fill for uncovered area when not using alpha.
        use_gpu: Try OpenCV's CUDA warp first, falling back to the CPU silently.

    Returns:
        ``uint8`` ``(out_h, out_w, 3)`` RGB — or ``(out_h, out_w, 4)`` RGBA when
        ``border`` is ``"transparent"``.

    Raises:
        RenderError: If OpenCV is missing, the frame is not RGB, or the
            interpolation name is unknown.
    """
    cv2 = load_cv2()
    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[2] != 3:
        raise RenderError(f"frames must be RGB (H, W, 3), got shape {source.shape}")
    if source.dtype != np.uint8:
        source = np.clip(source, 0, 255).astype(np.uint8)
    height, width = source.shape[:2]
    out_size = (int(width), int(height)) if size is None else (int(size[0]), int(size[1]))
    flag = _interpolation_flag(cv2, interpolation)
    matrix = transform.to_matrix()

    if border == "transparent":
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        source = np.concatenate([source, alpha], axis=2)
        border_value: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    else:
        border_value = (float(background[0]), float(background[1]), float(background[2]))

    if use_gpu:
        try:
            return _warp_cuda(cv2, source, matrix, out_size, flag, border_value)
        except Exception:  # pragma: no cover - depends on the OpenCV build
            pass
    return np.asarray(
        cv2.warpAffine(
            source,
            matrix,
            out_size,
            flags=flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    )
