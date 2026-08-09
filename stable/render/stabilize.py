"""Driving the stabilised render: plan once, then stream frames out.

A :class:`StabilizationPlan` is the whole render decided up front — one output
transform per frame, the output size, and the border policy — computed from a
:class:`~stable.smoothing.motion.SmoothedMotion` before a single pixel is
touched. That ordering is not incidental: the minimal auto-zoom can only be
known after *every* frame's warp is known, so any renderer that decided
per-frame would either over-crop or leave gaps.

Frames are then produced lazily and handed straight to
:func:`~stable.video.encode.export_video`, which passes the source clip's audio
through untouched. Nothing holds more than one frame at a time, so clip length
is bounded by disk, not memory.

Targets, and the fit
--------------------
A :class:`~stable.render.options.StabilizationTarget` is a list of marks on the
output frame, each naming the tracked path that goes on it. The plan is the
similarity that best satisfies all of them at once::

    S(t)   = argmin_S sum_k w_k |S . p_k(t) - g_k|^2
    out(t) = fit . S(t)

and that fit **is** the whole stabilisation — there is nothing composed
underneath it. That matters for more than tidiness: ``p_k`` is whichever path
the operator picked, raw or smoothed, and a base warp in series would move the
smoothed path somewhere else before the fit ever saw it. With nothing composed
under it, a ``smoothed`` mark lands the dashed trail on its mark exactly, every
frame, which is what picking it says it will do.

With no marks at all the plan falls back to
:meth:`~stable.smoothing.motion.SmoothedMotion.stabilization_transforms` —
``smoothed(t) . raw(t)^-1``, never rebuilt from raw channel values, because the
curves are pivot-relative and reconstructing a transform from them is wrong by a
rotation about the frame corner. That is what a project renders as before
anybody opens the export screen, and what produces the smoothed source paths
themselves. :attr:`StabilizationPlan.max_displacement` reports how much the plan
actually moves the picture, which is how a caller tells a real render from a
no-op.

Where a mark lands
------------------
A mark is a fraction of the *delivered* frame, and the plan honours it exactly:
:func:`~stable.render.warp.plan_output` builds the crop/zoom fit so that
``fit(origin) == placement × output size``, where ``origin`` is the marks'
centroid in the fitted space. That identity is why an operator can state "hold
this feature on the left thirds line" and have it be true of the file rather
than nearly true: the framing is solved for, not measured afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import RenderError
from ..core.geometry import SimilarityTransform
from ..core.project import BORDER_POLICIES, RenderSettings
from ..core.video import VideoSource
from ..smoothing.motion import SmoothedMotion
from ..trackers.base import ProgressCallback, noop_progress
from ..video.encode import export_video
from .fill import BorderFill
from .options import CropSpec, OutputResolution, RenderOptions, hold_gaps
from .warp import OutputGeometry, load_cv2, plan_output, warp_frame, window_size

__all__ = [
    "IDENTITY_TOLERANCE_PX",
    "StabilizationPlan",
    "iter_stabilized_frames",
    "render_stabilized",
    "write_frame_sequence",
    "stabilize",
    "stabilized_frame",
]

IDENTITY_TOLERANCE_PX = 0.01
"""Below this much corner movement a plan is a no-op, not a stabilisation.

A hundredth of a pixel is far under anything a viewer or a codec can represent,
so a plan that moves the frame less than this would render a file identical to
the source. Callers use it to refuse rather than to spend minutes encoding one.
"""


def _identity_warps(count: int) -> list[SimilarityTransform]:
    """Return ``count`` identity transforms.

    The base a target fit is composed onto. It is the identity *on purpose*: the
    fit is the entire stabilisation, so the paths the operator chose are the
    paths that get put on the marks — nothing is applied underneath that would
    move them somewhere else first. The smoothing's own warp is still what a
    render with no marks uses, and it is still what produces the ``smoothed``
    source paths; it is just not silently in series with them.
    """
    return [SimilarityTransform.identity() for _ in range(count)]


def _mapped_transforms(
    warps: Sequence[SimilarityTransform],
    sources: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray | None = None,
) -> list[SimilarityTransform]:
    """Compose the similarity that puts each tracked point on its chosen target.

    This is the whole stabilisation. Pinning *one* point can only translate the
    picture, so a shot that rotates or zooms still rotates and zooms around it.
    Given two or more point-to-target pairs, the transform that satisfies them
    **is** the rotation and scale the shot needs::

        S(t) = argmin_S sum_k w_k |S . p_k(t) - g_k|^2

    ``S`` is the weighted Umeyama fit — the same
    :func:`~stable.solver.similarity.solve_similarity` the camera solve uses,
    not a second implementation of it.

    With one pair the fit degrades to a pure translation (rotation and scale are
    unobservable from a single correspondence), which is precisely the
    single-target behaviour. With three or more it is least-squares: a
    similarity generally cannot satisfy them all, and the residual per pair is
    reported rather than hidden — see
    :func:`stable.server.motion.target_residuals`.

    Two or more pairs also spend the framing budget completely. The similarity
    that satisfies them is unique, so any crop/zoom fit applied afterwards would
    slide every point off the mark it was placed on; :meth:`StabilizationPlan.build`
    therefore plans those with ``allow_zoom=False``. Gaps are then real and
    reported rather than zoomed away — the way to close them is to move the
    marks closer together.

    Args:
        warps: The base each fit is composed onto; the identity, from
            :func:`_identity_warps`.
        sources: ``[K, T, 2]`` tracked paths in source pixels, gaps already held.
        destinations: ``[K, 2]`` chosen positions in output pixels.
        weights: Optional ``[K, T]`` per-frame confidences.

    Returns:
        One transform per frame.

    Raises:
        RenderError: If the shapes disagree or no pair is given.
    """
    from ..solver.similarity import solve_similarity  # noqa: PLC0415 - avoids an import cycle

    src = np.asarray(sources, dtype=np.float64)
    dst = np.asarray(destinations, dtype=np.float64).reshape(-1, 2)
    if src.ndim != 3 or src.shape[0] == 0 or src.shape[2] != 2:
        raise RenderError(f"mapped targets need a [K, T, 2] source array, got {src.shape}")
    if src.shape[0] != dst.shape[0]:
        raise RenderError(
            f"{src.shape[0]} mapped tracks against {dst.shape[0]} target positions; "
            "every pair needs both halves"
        )
    if src.shape[1] < len(warps):
        raise RenderError(
            f"the stabilisation targets cover {src.shape[1]} frames but the render plans "
            f"{len(warps)}; the target tracks were made from a different clip"
        )

    all_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    out: list[SimilarityTransform] = []
    for frame, warp in enumerate(warps):
        moved = warp.apply(src[:, frame, :])
        frame_weights: np.ndarray | None = None
        if all_weights is not None:
            candidate = all_weights[:, frame]
            # Confidence weights the fit — except where there is no confidence
            # to go on. Outside the tracked span every pair scores zero, and a
            # zero-sum weighting is not "trust nobody", it is "no information":
            # the plan still has to produce a transform for those frames, so
            # they fall back to trusting the pairs equally. Without this the
            # whole render fails on the first untracked frame.
            if np.isfinite(candidate).all() and candidate.sum() > 0.0:
                frame_weights = candidate
        fit = solve_similarity(moved, dst, frame_weights)
        out.append(fit.compose(warp))
    return out


@dataclass(frozen=True)
class StabilizationPlan:
    """Every decision the render needs, made before any pixel is read.

    Args:
        transforms: One stabilisation warp per frame, ``smoothed . raw^-1``.
        geometry: Output size and the crop/zoom fit transform.
        source_size: ``(width, height)`` of the input clip.
        interpolation: Resampling kernel; see
            :data:`~stable.render.warp.INTERPOLATIONS`.
        background: RGB fill for uncovered area (unused for ``transparent``).
        fill: What goes in the uncovered area. ``None`` means the plain
            ``background`` colour, which is what Phase 3 always did.
    """

    transforms: tuple[SimilarityTransform, ...]
    geometry: OutputGeometry
    source_size: tuple[int, int]
    interpolation: str = "linear"
    background: tuple[int, int, int] = (0, 0, 0)
    fill: BorderFill | None = field(default=None)

    @classmethod
    def build(
        cls,
        motion: SmoothedMotion,
        *,
        source_size: tuple[int, int],
        border: str = "auto_zoom",
        max_zoom: float | None = None,
        interpolation: str = "linear",
        background: tuple[int, int, int] = (0, 0, 0),
        sources: np.ndarray | None = None,
        placements: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        crop: CropSpec | None = None,
        fill: BorderFill | None = None,
        coverage_range: tuple[int, int] | None = None,
        resolution: OutputResolution | None = None,
    ) -> StabilizationPlan:
        """Plan a render from smoothed motion.

        Args:
            motion: The smoothed solve; its
                :meth:`~stable.smoothing.motion.SmoothedMotion.stabilization_transforms`
                are what get applied.
            source_size: ``(width, height)`` of the clip.
            border: One of :data:`~stable.core.project.BORDER_POLICIES`.
            max_zoom: Optional cap on the auto-zoom factor.
            interpolation: Resampling kernel.
            background: RGB fill for uncovered area.
            sources: ``[K, T, 2]`` source-pixel paths, one per target mark, in
                the order the marks were given. Gaps are held
                (:func:`~stable.render.options.hold_gaps`) rather than left to
                make the correction lurch. ``None`` (or no marks at all) plans
                the plain smoothing warp, which is what a project renders as
                before anybody places a target.
            placements: ``[K, 2]`` marks as ``(u, v)`` fractions of the output.
            weights: Optional ``[K, T]`` per-frame confidences for the fit.
            crop: How much of the frame the output keeps; ``None`` is the exact
                minimal gapless crop.
            fill: What goes in the uncovered area.
            coverage_range: Half-open ``(start, end)`` the automatic crop is
                measured over; ``None`` measures every planned frame. The plan
                itself always covers the whole clip — this narrows what "no
                gaps" is a promise about, not what gets rendered.
            resolution: The size to deliver the crop window at.

        Returns:
            The :class:`StabilizationPlan`.

        Raises:
            RenderError: If the border policy is unknown, an automatic crop
                finds the warped frames share no common area, or a source path
                does not cover the clip.
        """
        width, height = int(source_size[0]), int(source_size[1])
        warps: Sequence[SimilarityTransform] = motion.stabilization_transforms()
        origin: tuple[float, float] | None = None
        place = (0.5, 0.5)
        allow_zoom = True
        marks = (
            np.zeros((0, 2))
            if placements is None
            else np.asarray(placements, dtype=np.float64).reshape(-1, 2)
        )
        if sources is not None and marks.shape[0]:
            held = np.stack([hold_gaps(path) for path in np.asarray(sources)])
            # The marks are fractions of the delivered frame, and the fit is the
            # *whole* stabilisation — nothing is composed underneath it. That is
            # what lets a `smoothed` source land its own trail on its mark: the
            # path the operator picked is the path that gets fitted, rather than
            # a path pushed through a warp they did not ask for.
            #
            # They resolve against the *crop window*, not the source frame. The
            # window carries the delivered aspect ratio, so the fit's own scale
            # is the only thing between these and the output pixels — which is
            # what keeps two marks exactly on their fractions when the delivery
            # is a different shape than the source.
            box = np.asarray(
                window_size(width, height, (resolution or OutputResolution()).aspect),
                dtype=np.float64,
            )
            destinations = marks * box
            warps = _mapped_transforms(
                _identity_warps(len(warps)), held, destinations, weights=weights
            )
            centroid = destinations.mean(axis=0)
            origin = (float(centroid[0]), float(centroid[1]))
            place = (float(marks[:, 0].mean()), float(marks[:, 1].mean()))
            # Two or more marks pin rotation and scale as well as position, so
            # the similarity is already determined and there is no zoom left to
            # apply. One mark is a translation, which leaves the zoom free.
            allow_zoom = marks.shape[0] < 2
        transforms = tuple(warps)
        geometry = plan_output(
            transforms,
            width,
            height,
            border,
            max_zoom=max_zoom,
            origin=origin,
            placement=place,
            crop=crop,
            coverage_range=coverage_range,
            allow_zoom=allow_zoom,
            resolution=resolution,
        )
        return cls(
            transforms=transforms,
            geometry=geometry,
            source_size=(width, height),
            interpolation=interpolation,
            background=background,
            fill=fill,
        )

    @classmethod
    def from_settings(
        cls,
        motion: SmoothedMotion,
        settings: RenderSettings,
        *,
        source_size: tuple[int, int],
        options: RenderOptions | None = None,
        sources: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        **overrides: Any,
    ) -> StabilizationPlan:
        """Plan a render from a project file's ``render`` block.

        Args:
            motion: The smoothed solve.
            settings: The project's :class:`~stable.core.project.RenderSettings`
                — only its border policy is a planning decision; codec and CRF
                belong to the encode step.
            source_size: ``(width, height)`` of the clip.
            options: The operator's target / crop / fill / resolution choices.
                The marks' *paths* are supplied resolved, as ``sources``,
                because turning "the smoothed estimate of p2" into a path needs
                the tracks and this module deliberately knows nothing about them.
            sources: ``[K, T, 2]`` paths, one per mark, in ``options`` order.
            weights: Optional ``[K, T]`` confidences for the fit.
            **overrides: Any other :meth:`build` argument.

        Returns:
            The :class:`StabilizationPlan`.
        """
        chosen = options or RenderOptions()
        return cls.build(
            motion,
            source_size=source_size,
            border=settings.border,
            sources=sources,
            placements=np.asarray(
                [point.place.fraction for point in chosen.target.points], dtype=np.float64
            ).reshape(-1, 2),
            weights=weights,
            crop=chosen.crop,
            fill=chosen.fill,
            resolution=chosen.resolution,
            **overrides,
        )

    @property
    def border(self) -> str:
        """The border policy this plan implements."""
        return self.geometry.border

    @property
    def n_frames(self) -> int:
        """Number of frames the plan covers."""
        return len(self.transforms)

    @property
    def has_alpha(self) -> bool:
        """Whether rendered frames carry an alpha channel (RGBA)."""
        return self.border == "transparent"

    @property
    def max_displacement(self) -> float:
        """Largest distance, in source pixels, that any frame corner is moved.

        Measured on the stabilisation warps alone — the crop/zoom fit is
        framing, not stabilisation, and would make a plain zoom look like
        motion correction. ``0`` means the render would reproduce the source.
        """
        width, height = float(self.source_size[0]), float(self.source_size[1])
        corners = np.array(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64
        )
        worst = 0.0
        for transform in self.transforms:
            moved = np.asarray(transform.apply(corners), dtype=np.float64) - corners
            worst = max(worst, float(np.hypot(moved[:, 0], moved[:, 1]).max()))
        return worst

    @property
    def is_no_op(self) -> bool:
        """Whether this plan would render the source back unchanged.

        True when nothing asked the picture to move: no smoothing filter is
        enabled and no stabilisation target is set, so
        ``smoothed(t) . raw(t)^-1`` is the identity on every frame. Rendering it
        is minutes of encoding for a copy of the input, which is exactly what
        "the stabilised output doesn't work" looks like from the outside.
        """
        return self.max_displacement < IDENTITY_TOLERANCE_PX

    def output_transform(self, frame: int) -> SimilarityTransform:
        """Return the single affine applied to ``frame``.

        The stabilisation warp with the crop offset or auto-zoom composed in, so
        the frame is resampled exactly once however many stages produced it.

        Args:
            frame: Absolute frame index.

        Raises:
            IndexError: If ``frame`` is outside the planned range.
        """
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} out of range for {self.n_frames} planned frames")
        return self.geometry.fit.compose(self.transforms[frame])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of the plan (without the per-frame matrices)."""
        return {
            "frames": self.n_frames,
            "source_size": list(self.source_size),
            "interpolation": self.interpolation,
            "geometry": self.geometry.to_dict(),
            "max_displacement": self.max_displacement,
            "fill": None if self.fill is None else self.fill.to_dict(),
        }


def iter_stabilized_frames(
    video: VideoSource,
    plan: StabilizationPlan,
    *,
    frame_range: tuple[int, int] | None = None,
    use_gpu: bool = False,
    progress: ProgressCallback = noop_progress,
) -> Iterator[np.ndarray]:
    """Yield stabilised frames one at a time.

    Args:
        video: The clip to read from.
        plan: The render plan.
        frame_range: Half-open ``(start, end)`` range to render; defaults to the
            whole planned range.
        use_gpu: Try the CUDA warp, falling back to the CPU silently.
        progress: Progress reporter, called once per frame.

    Yields:
        ``uint8`` RGB ``(H, W, 3)`` frames — or RGBA ``(H, W, 4)`` under the
        ``transparent`` policy.

    Raises:
        RenderError: If the range is invalid or reaches past the plan.
    """
    start, end = (
        (0, plan.n_frames)
        if frame_range is None
        else (
            int(frame_range[0]),
            int(frame_range[1]),
        )
    )
    if end <= start:
        raise RenderError(f"frame_range must be non-empty, got [{start}, {end})")
    if start < 0 or end > plan.n_frames:
        raise RenderError(
            f"frame_range [{start}, {end}) is outside the planned range [0, {plan.n_frames})"
        )
    if end > video.frame_count:
        raise RenderError(
            f"frame_range [{start}, {end}) runs past the clip (T={video.frame_count})"
        )

    total = end - start
    for offset, index in enumerate(range(start, end)):
        yield warp_frame(
            video.get_frame(index),
            plan.output_transform(index),
            size=plan.geometry.size,
            border=plan.border,
            interpolation=plan.interpolation,
            background=plan.background,
            fill=plan.fill,
            use_gpu=use_gpu,
        )
        progress((offset + 1) / total, f"frame {offset + 1}/{total}")


def _looks_like_directory(path: str) -> bool:
    """Whether ``path`` names a directory (an existing one, or an extension-less name)."""
    if os.path.isdir(path):
        return True
    if path.endswith(("/", "\\")):
        return True
    return os.path.splitext(path)[1] == ""


def write_frame_sequence(
    frames: Iterable[np.ndarray],
    directory: str | os.PathLike[str],
    *,
    prefix: str = "frame",
    start: int = 0,
    digits: int = 6,
    progress: ProgressCallback = noop_progress,
) -> list[str]:
    """Write frames as a numbered PNG sequence.

    PNG is the escape hatch for the ``transparent`` border: the video codecs
    this project encodes to carry no alpha channel, so an image sequence is the
    only honest way to deliver one.

    Args:
        frames: ``uint8`` RGB or RGBA frames, in order.
        directory: Destination directory; created if missing.
        prefix: File name stem.
        start: Number of the first frame.
        digits: Zero-padding width of the frame number.
        progress: Progress reporter, called once per frame.

    Returns:
        The absolute paths written, in order.

    Raises:
        RenderError: If OpenCV is missing, a frame has an unexpected shape, or a
            file cannot be written.
    """
    cv2 = load_cv2()
    target = os.path.abspath(os.fspath(directory))
    os.makedirs(target, exist_ok=True)
    written: list[str] = []
    for offset, frame in enumerate(frames):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise RenderError(f"frames must be RGB or RGBA, got shape {array.shape}")
        converted = cv2.cvtColor(
            array, cv2.COLOR_RGBA2BGRA if array.shape[2] == 4 else cv2.COLOR_RGB2BGR
        )
        path = os.path.join(target, f"{prefix}_{start + offset:0{digits}d}.png")
        if not cv2.imwrite(path, converted):
            raise RenderError(f"could not write {path!r}")
        written.append(path)
        progress(0.0, f"frame {offset + 1}")
    if not written:
        raise RenderError("write_frame_sequence received no frames")
    progress(1.0, "done")
    return written


def render_stabilized(
    video: VideoSource,
    plan: StabilizationPlan,
    out_path: str | os.PathLike[str],
    *,
    fps: float | None = None,
    codec: str = "h264",
    crf: int = 18,
    audio_source: str | os.PathLike[str] | None = None,
    frame_range: tuple[int, int] | None = None,
    use_gpu: bool = False,
    progress: ProgressCallback = noop_progress,
) -> str:
    """Render a stabilised clip to a video file or a PNG sequence.

    ``out_path`` decides which: a path with a file extension is encoded as a
    video (audio passed through from ``audio_source``), and an extension-less
    path or an existing directory becomes a numbered PNG sequence.

    Args:
        video: The clip to read from.
        plan: The render plan.
        out_path: Destination file or directory.
        fps: Output frame rate; defaults to the source's.
        codec: One of :data:`~stable.video.encode.CODECS`' keys.
        crf: Constant rate factor (quality); lower is better.
        audio_source: Path to the original clip, whose audio track is copied
            into the output verbatim. ``None`` renders silent video.
        frame_range: Half-open range to render; defaults to the whole clip.
        use_gpu: Try the CUDA warp, falling back to the CPU silently.
        progress: Progress reporter.

    Returns:
        The absolute path written — the file, or the directory for a sequence.

    Raises:
        RenderError: If the ``transparent`` policy is aimed at a video file
            (no codec here carries alpha), or the render cannot be produced.
        VideoError: If encoding fails.
    """
    target = os.path.abspath(os.fspath(out_path))
    frames = iter_stabilized_frames(
        video, plan, frame_range=frame_range, use_gpu=use_gpu, progress=progress
    )

    if _looks_like_directory(target):
        write_frame_sequence(frames, target)
        return target

    if plan.has_alpha:
        raise RenderError(
            "the 'transparent' border produces RGBA frames, and the video codecs available "
            f"here carry no alpha channel, so {os.path.basename(target)!r} would silently lose "
            "it; render to a directory for a PNG sequence, or use the 'crop' or 'auto_zoom' "
            "border for a video file"
        )

    start, end = (
        (0, plan.n_frames)
        if frame_range is None
        else (
            int(frame_range[0]),
            int(frame_range[1]),
        )
    )
    return export_video(
        frames,
        target,
        fps=float(video.fps) if fps is None else float(fps),
        codec=codec,
        crf=crf,
        audio_source=audio_source,
        audio_range=_audio_window(video, plan, start, end),
        frame_count=end - start,
    )


def _audio_window(
    video: VideoSource, plan: StabilizationPlan, start: int, end: int
) -> tuple[float, float] | None:
    """Return the seconds of source audio a render of ``[start, end)`` should carry.

    ``None`` when the render covers the whole clip, so the untrimmed path stays
    exactly what it was rather than merely equivalent to it. Anything narrower
    has to trim: a container's duration is the longer of its streams, so a full
    audio track behind a partial video one leaves a player holding the last
    frame for the remainder (see :mod:`stable.video.encode`).

    The window is measured on the *source's* frame rate, since that is the
    timeline the audio was recorded against — an ``fps`` override changes how
    fast the output plays, not where the sound was.
    """
    fps = float(video.fps)
    if fps <= 0.0:  # pragma: no cover - a clip with no declared rate
        return None
    if start <= 0 and end >= plan.n_frames:
        return None
    return (start / fps, end / fps)


def stabilize(
    video: VideoSource,
    motion: SmoothedMotion,
    out_path: str | os.PathLike[str],
    *,
    border: str = "auto_zoom",
    max_zoom: float | None = None,
    interpolation: str = "linear",
    sources: np.ndarray | None = None,
    placements: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    crop: CropSpec | None = None,
    fill: BorderFill | None = None,
    coverage_range: tuple[int, int] | None = None,
    resolution: OutputResolution | None = None,
    **kwargs: Any,
) -> tuple[str, StabilizationPlan]:
    """Plan and render in one call — the convenience wrapper the CLI uses.

    Args:
        video: The clip to stabilise.
        motion: The smoothed solve.
        out_path: Destination file or directory.
        border: One of :data:`~stable.core.project.BORDER_POLICIES`.
        max_zoom: Optional cap on the auto-zoom factor.
        interpolation: Resampling kernel.
        sources: ``[K, T, 2]`` paths to put on the marks, if any.
        placements: ``[K, 2]`` marks, as ``(u, v)`` fractions of the output.
        weights: Optional ``[K, T]`` confidences for the fit.
        crop: How much of the frame to keep.
        fill: What goes in the uncovered area.
        coverage_range: Frames the automatic crop is measured over.
        resolution: The size to deliver the crop window at.
        **kwargs: Any other :func:`render_stabilized` argument.

    Returns:
        ``(path, plan)`` — what was written, and the plan that produced it (so
        the caller can report the zoom and output size).

    Raises:
        RenderError: As :func:`render_stabilized`.
    """
    if border not in BORDER_POLICIES:
        raise RenderError(f"unknown border policy {border!r}; expected {list(BORDER_POLICIES)}")
    plan = StabilizationPlan.build(
        motion,
        source_size=(int(video.width), int(video.height)),
        border=border,
        max_zoom=max_zoom,
        interpolation=interpolation,
        sources=sources,
        placements=placements,
        weights=weights,
        crop=crop,
        fill=fill,
        coverage_range=coverage_range,
        resolution=resolution,
    )
    return render_stabilized(video, plan, out_path, **kwargs), plan


def stabilized_frame(
    video: VideoSource,
    plan: StabilizationPlan,
    frame: int,
    *,
    use_gpu: bool = False,
) -> np.ndarray:
    """Render a single stabilised frame — what a preview scrub needs.

    Args:
        video: The clip to read from.
        plan: The render plan.
        frame: Absolute frame index.
        use_gpu: Try the CUDA warp, falling back to the CPU silently.

    Returns:
        The stabilised frame as ``uint8`` RGB (or RGBA under ``transparent``).
    """
    return warp_frame(
        video.get_frame(frame),
        plan.output_transform(frame),
        size=plan.geometry.size,
        border=plan.border,
        interpolation=plan.interpolation,
        background=plan.background,
        fill=plan.fill,
        use_gpu=use_gpu,
    )
