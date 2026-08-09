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
rectangle of the source aspect ratio can grow before it pokes out of that quad —
a linear-programming question with four constraints per edge that has a closed
form — and takes the tightest answer over the frames it is given. The zoom is
the reciprocal. Because it is exact rather than iterative, "minimal zoom
covering all frames" is a guarantee, not a search that might stop early.

Which frames get a say is a choice (todo 1, eighth round): pass ``coverage_range`` and the
answer covers only the frames the export will actually write, so one violent
second at the end of a take stops dictating the crop of the thirty seconds being
delivered.

Where that rectangle sits
-------------------------
Not necessarily centred. :func:`plan_output` takes an ``origin`` — a position in
warped coordinates, normally the stabilisation target's — and a ``placement``,
the fraction of the output frame that origin has to land on. The default
``(0.5, 0.5)`` is "centre it", which is what every caller wanted before targets
could be placed. Anything else composes the shot: ``(1/3, 0.5)`` puts the held
feature on the left thirds line, and the crop grows as far as it can *around
that*, which is a different (and generally smaller) rectangle than the centred
one.

This is the whole reason a target's placement can be stated in output
coordinates at all. The fit transform is built to satisfy ``fit(origin) =
placement × output size`` exactly, so "the point lands where I put it" is an
identity of the geometry rather than something the renderer approximates.

Choosing the crop yourself
--------------------------
That minimal answer is the *default*, not the only one. A
:class:`~stable.render.options.CropSpec` overrides it with an explicit scale,
in either direction: tighter than necessary (a safety margin), or looser — in
which case the output deliberately includes area no frame covers, and
:class:`~stable.render.fill.BorderFill` decides what goes there. The geometry
reports both numbers (``coverage`` measured, ``crop_scale`` requested) and sets
:attr:`OutputGeometry.gaps` when the second exceeds the first, so nobody has to
guess whether the fill will be visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import RenderError
from ..core.geometry import SimilarityTransform
from ..core.project import BORDER_POLICIES
from .fill import BorderFill, composite_fill
from .options import CropSpec, OutputResolution

__all__ = [
    "INTERPOLATIONS",
    "OutputGeometry",
    "coverage_scale",
    "plan_output",
    "warp_frame",
    "rescale_transform",
    "window_size",
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
            This is *measured*, not requested — compare it with ``crop_scale``.
        zoom: The scale factor ``fit`` applies; ``1 / crop_scale`` for auto-zoom.
        border: The policy this geometry implements.
        clamped: True when ``max_zoom`` capped the zoom, so gaps may remain.
        crop_scale: The fraction of the frame actually kept. Equals ``coverage``
            for an automatic crop; an explicit
            :class:`~stable.render.options.CropSpec` overrides it.
        centre: The origin the output rectangle was grown around, in warped
            coordinates — the stabilisation target's position, or the frame
            centre when there is nothing to hold.
        placement: Where that origin lands in the output, as a ``(u, v)``
            fraction of the output frame. ``fit(centre)`` is exactly
            ``placement × (width, height)``, which is what lets a target's
            position be stated in output coordinates.
        gaps: True when the output includes area no frame covers, so the border
            fill will be visible. Honest rather than implied: an explicit crop
            or a capped zoom can both leave uncovered edges on purpose.
        measured_frames: How many frames the coverage was measured over — the
            whole clip, or just the export range when the crop was asked to fit
            that (:attr:`~stable.render.options.CropSpec.limit_to_range`).
    """

    width: int
    height: int
    fit: SimilarityTransform
    coverage: float
    zoom: float
    border: str
    clamped: bool = False
    crop_scale: float = 1.0
    centre: tuple[float, float] = (0.0, 0.0)
    placement: tuple[float, float] = (0.5, 0.5)
    gaps: bool = False
    measured_frames: int = 0

    @property
    def size(self) -> tuple[int, int]:
        """Output size as ``(width, height)``."""
        return (self.width, self.height)

    @property
    def placement_pixels(self) -> tuple[float, float]:
        """Where :attr:`centre` lands, in output pixels."""
        return (self.placement[0] * self.width, self.placement[1] * self.height)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the output geometry."""
        return {
            "width": self.width,
            "height": self.height,
            "border": self.border,
            "coverage": self.coverage,
            "zoom": self.zoom,
            "clamped": self.clamped,
            "crop_scale": self.crop_scale,
            "centre": list(self.centre),
            "placement": list(self.placement),
            "gaps": self.gaps,
            "measured_frames": self.measured_frames,
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
    origin: tuple[float, float] | None = None,
    placement: tuple[float, float] = (0.5, 0.5),
    window: tuple[float, float] | None = None,
) -> float:
    """Return the largest rectangle, positioned as asked, every warped frame covers.

    Args:
        transforms: One stabilisation warp per frame — only the frames that
            matter, so a caller that renders a sub-range passes that sub-range.
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        origin: The point in warped coordinates the rectangle is pinned to;
            defaults to the frame centre.
        placement: Where ``origin`` sits inside the rectangle, as a ``(u, v)``
            fraction of it. ``(0.5, 0.5)`` grows a centred rectangle, which is
            what keeps the framing where it was; ``(1/3, 0.5)`` grows one whose
            left thirds line is on ``origin``, which is how a placed target
            composes the shot.
        window: The rectangle's shape as a ``(width, height)`` ratio, or
            ``None`` for the source frame's. A delivery resolution with a
            different aspect crops a rectangle of *that* shape — a square
            delivery takes the largest square, rather than squeezing 16:9 into
            it — so the shape has to be known before the size is solved for.

    Returns:
        A factor in ``[0, 1]``: the rectangle is that fraction of the largest
        window of the requested shape that fits the source frame. ``1.0`` means
        no crop is needed at all; ``0.0`` means the frames share no common area
        at that placement.
    """
    if width <= 0 or height <= 0:
        raise RenderError(f"frame size must be positive, got {width}x{height}")
    if not transforms:
        return 1.0
    pinned = (
        np.array([width / 2.0, height / 2.0], dtype=np.float64)
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    # Corner offsets of a rectangle of the requested shape holding the origin at
    # `placement`. With the source shape and (0.5, 0.5) these are the centred
    # corners the whole module used before a target could be placed off-centre
    # or a delivery aspect asked for.
    box_w, box_h = window_size(width, height, window)
    u, v = float(placement[0]), float(placement[1])
    left, right = -u * box_w, (1.0 - u) * box_w
    top, bottom = -v * box_h, (1.0 - v) * box_h
    directions = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
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
            slack = float(normal @ (start - pinned))
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


def window_size(
    width: int, height: int, window: tuple[float, float] | None
) -> tuple[float, float]:
    """Return the largest rectangle of ``window``'s shape that fits the frame.

    This is what a crop scale of ``1.0`` means once delivery can ask for a
    different aspect than the source: all of the frame when the shapes agree,
    and the biggest rectangle of the delivered shape inside it when they do not.
    ``None`` keeps the source's own shape, which is the whole frame.

    Args:
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        window: The delivered ``(width, height)`` ratio, or ``None``.

    Returns:
        The window's ``(width, height)`` in source pixels at scale ``1.0``.
    """
    if window is None:
        return (float(width), float(height))
    ratio_w, ratio_h = float(window[0]), float(window[1])
    if ratio_w <= 0.0 or ratio_h <= 0.0:
        raise RenderError(f"the output aspect must be positive, got {window!r}")
    scale = min(width / ratio_w, height / ratio_h)
    return (ratio_w * scale, ratio_h * scale)


def plan_output(
    transforms: Sequence[SimilarityTransform],
    width: int,
    height: int,
    border: str = "auto_zoom",
    *,
    max_zoom: float | None = None,
    origin: tuple[float, float] | None = None,
    placement: tuple[float, float] = (0.5, 0.5),
    crop: CropSpec | None = None,
    coverage_range: tuple[int, int] | None = None,
    allow_zoom: bool = True,
    resolution: OutputResolution | None = None,
) -> OutputGeometry:
    """Work out the output size and fit transform for a border policy.

    Args:
        transforms: One stabilisation warp per frame, for the whole clip.
        width: Source frame width in pixels.
        height: Source frame height in pixels.
        border: One of :data:`~stable.core.project.BORDER_POLICIES`.
        max_zoom: Optional cap on the auto-zoom factor. When the shot needs more
            than this, the zoom stops there and the result is marked
            :attr:`OutputGeometry.clamped` — visible gaps, but the user keeps
            the framing they asked for rather than losing half the frame.
        origin: The position in warped coordinates the output rectangle is
            pinned to — the stabilisation target's held position. Defaults to
            the frame centre, which is what "nothing in particular is held"
            means.
        placement: Where ``origin`` lands in the output, as a ``(u, v)``
            fraction of it. The fit is built so that ``fit(origin)`` is exactly
            ``placement × output size``, which is how a target's position can be
            stated in output coordinates and be *right* rather than close.
        crop: How much of the frame to keep. ``None`` (or an ``auto``
            :class:`~stable.render.options.CropSpec`) reproduces the exact
            minimal gapless crop; an explicit ``scale`` overrides it and may
            leave uncovered edges for the border fill to handle.
        coverage_range: Half-open ``(start, end)`` frames the automatic crop is
            measured over. ``None`` measures the whole clip. Narrowing it is how
            "the biggest crop with no gaps *between the export markers*" is
            computed — the frames outside are not being written, so letting them
            veto the crop is answering a question nobody asked.
        allow_zoom: Whether the ``auto_zoom`` policy may actually zoom. False
            for a fit of two or more marks: those state the framing completely,
            so any zoom on top would slide every point off the mark it was
            placed on. The geometry still reports the coverage, so the gaps such
            a fit leaves are visible rather than silent.
        resolution: The pixel size to deliver the crop window at. ``None`` (or
            an unset :class:`~stable.render.options.OutputResolution`) delivers
            it at whatever size it came out, which is what every policy did
            before delivery size was a choice.

    Returns:
        The :class:`OutputGeometry`.

    Raises:
        RenderError: If the policy is unknown, the size is not positive, the
            coverage range is empty, or an automatic crop finds the frames share
            no common area.
    """
    if border not in BORDER_POLICIES:
        raise RenderError(f"unknown border policy {border!r}; expected {list(BORDER_POLICIES)}")
    if width <= 0 or height <= 0:
        raise RenderError(f"frame size must be positive, got {width}x{height}")
    spec = crop or CropSpec()
    delivery = resolution or OutputResolution()
    window = delivery.aspect
    box_w, box_h = window_size(width, height, window)
    pinned = (width / 2.0, height / 2.0) if origin is None else (
        float(origin[0]),
        float(origin[1]),
    )
    place = (float(placement[0]), float(placement[1]))

    if border == "transparent":
        # Nothing is hidden, so nothing needs measuring: the gaps become alpha.
        return OutputGeometry(
            width=int(width),
            height=int(height),
            fit=SimilarityTransform.identity(),
            coverage=1.0,
            zoom=1.0,
            border=border,
            crop_scale=1.0,
            centre=(width / 2.0, height / 2.0),
            placement=(0.5, 0.5),
            measured_frames=len(transforms),
        )

    measured = _coverage_frames(transforms, coverage_range)
    coverage = coverage_scale(
        measured, width, height, origin=pinned, placement=place, window=window
    )
    if spec.is_auto:
        if coverage <= 0.0:
            raise RenderError(
                "the stabilised frames share no common area, so there is nothing to crop to; "
                "reduce the smoothing strength, disable a channel, move the stabilisation "
                "target back towards the middle of the frame, set an explicit crop scale with a "
                "border fill, or use the 'transparent' border"
            )
        requested = coverage
    else:
        requested = float(spec.scale or 1.0)
    gaps = requested > coverage + 1e-9

    if border == "crop":
        crop_w = _even(int(requested * box_w))
        crop_h = _even(int(requested * box_h))
        out_w, out_h = delivery.resolve((crop_w, crop_h))
        # Put the origin where the placement asks for inside the crop window,
        # then scale that window up to the delivered size. Both steps are one
        # similarity, so the frame is still resampled exactly once.
        offset = SimilarityTransform(
            tx=place[0] * crop_w - pinned[0],
            ty=place[1] * crop_h - pinned[1],
        )
        deliver = SimilarityTransform(scale=out_w / crop_w) if out_w != crop_w else None
        return OutputGeometry(
            width=out_w,
            height=out_h,
            fit=offset if deliver is None else deliver.compose(offset),
            coverage=coverage,
            zoom=1.0 if deliver is None else out_w / crop_w,
            border=border,
            crop_scale=requested,
            centre=pinned,
            placement=place,
            gaps=gaps,
            measured_frames=len(measured),
        )

    zoom = 1.0 / requested if allow_zoom else 1.0
    clamped = False
    if not allow_zoom:
        gaps = gaps or coverage < 1.0 - 1e-9
    elif max_zoom is not None and zoom > float(max_zoom):
        if float(max_zoom) < 1.0:
            raise RenderError(f"max_zoom must be >= 1, got {max_zoom}")
        zoom, clamped = float(max_zoom), True
        gaps = gaps or zoom * coverage < 1.0 - 1e-9
    # `auto_zoom` delivers the window at its own size unless asked otherwise;
    # the window is the frame when the shapes agree, which is the usual case.
    out_w, out_h = delivery.resolve((_even(int(box_w)), _even(int(box_h))))
    scale = zoom * out_w / box_w
    recentre = SimilarityTransform(
        tx=place[0] * out_w - pinned[0], ty=place[1] * out_h - pinned[1]
    )
    return OutputGeometry(
        width=out_w,
        height=out_h,
        fit=recentre.compose(SimilarityTransform.about_pivot(pinned, scale=scale)),
        coverage=coverage,
        zoom=zoom,
        border=border,
        clamped=clamped,
        crop_scale=requested if allow_zoom else 1.0,
        centre=pinned,
        placement=place,
        gaps=gaps,
        measured_frames=len(measured),
    )


def _coverage_frames(
    transforms: Sequence[SimilarityTransform], coverage_range: tuple[int, int] | None
) -> Sequence[SimilarityTransform]:
    """Return the warps an automatic crop should be measured over.

    Raises:
        RenderError: If the range is empty or falls outside the planned frames.
    """
    if coverage_range is None:
        return transforms
    start, end = int(coverage_range[0]), int(coverage_range[1])
    if end <= start:
        raise RenderError(f"the crop's frame range must be non-empty, got [{start}, {end})")
    window = transforms[max(start, 0) : min(end, len(transforms))]
    if not window:
        raise RenderError(
            f"the crop's frame range [{start}, {end}) does not overlap the "
            f"{len(transforms)} planned frames"
        )
    return window


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
    fill: BorderFill | None = None,
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
            area; any other policy fills it with ``background`` or ``fill``.
        interpolation: One of :data:`INTERPOLATIONS`.
        background: RGB fill for uncovered area when not using alpha. Ignored
            when ``fill`` is given.
        fill: What to put in the uncovered area — solid colour, stretched or
            mirrored edge, blur, or a combination (see
            :class:`~stable.render.fill.BorderFill`). ``None`` keeps the plain
            constant-colour behaviour, which stays a single ``warpAffine``.
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

    # Alpha is itself an answer to "what goes in the gap", so a fill only
    # applies to the policies that have to put real pixels there.
    if fill is not None and border != "transparent":
        if fill.is_plain:
            background = fill.color
        else:
            return composite_fill(cv2, source, matrix, out_size, flag=flag, fill=fill)

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
