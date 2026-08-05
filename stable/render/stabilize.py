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

The stabilisation target
------------------------
The base warp is ``smoothed(t) . raw(t)^-1``, so a clip with *no* smoothing
configured produces the identity on every frame — a render that is a re-encode
of the source and looks, correctly, like nothing happened. That is the whole
failure mode behind "the stabilised output doesn't work": nothing errors,
:attr:`OutputGeometry.zoom` comes back as ``1.0``, and the file is just the
input again.

A :class:`~stable.render.options.StabilizationTarget` is the direct way to ask
for stabilisation instead of asking for it sideways through the smoothing
panel. Given the target's tracked position ``p(t)``, the plan holds it still::

    q(t) = W(t) . p(t)                     # where the base warp puts it
    plan(t) = T(q(anchor) - q(t)) . W(t)   # nail it to its anchor-frame place

``W(t)`` comes from
:meth:`~stable.smoothing.motion.SmoothedMotion.stabilization_transform`, never
from raw channel values — the curves are pivot-relative, and rebuilding a
transform from them is wrong by a rotation about the frame corner. The extra
term is a pure translation *composed on top*, so that stays true. With no
smoothing ``W(t)`` is the identity and the target lock is the entire
stabilisation; with smoothing it removes whatever drift the filters left behind.
:attr:`StabilizationPlan.max_displacement` reports how much the plan actually
moves the picture, which is how a caller tells a real render from a no-op.
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
from .options import CropSpec, RenderOptions, hold_gaps
from .warp import OutputGeometry, load_cv2, plan_output, warp_frame

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


def _target_transforms(
    warps: Sequence[SimilarityTransform],
    positions: np.ndarray,
    anchor_frame: int,
) -> list[SimilarityTransform]:
    """Compose a translation onto each warp that pins ``positions`` to their anchor.

    Args:
        warps: The base stabilisation warp per frame.
        positions: ``[T, 2]`` target positions in source pixels, gaps already
            held (see :func:`~stable.render.options.hold_gaps`).
        anchor_frame: The frame whose stabilised target position is held.

    Returns:
        One transform per frame.

    Raises:
        RenderError: If the positions do not cover the planned frames, or the
            anchor frame is outside them.
    """
    path = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if path.shape[0] < len(warps):
        raise RenderError(
            f"the stabilisation target covers {path.shape[0]} frames but the render plans "
            f"{len(warps)}; the target track was made from a different clip"
        )
    if not 0 <= anchor_frame < len(warps):
        raise RenderError(
            f"stabilisation anchor frame {anchor_frame} is outside the clip [0, {len(warps)})"
        )
    held = warps[anchor_frame].apply(path[anchor_frame])
    out: list[SimilarityTransform] = []
    for frame, warp in enumerate(warps):
        moved = warp.apply(path[frame])
        offset = held - moved
        out.append(SimilarityTransform(tx=float(offset[0]), ty=float(offset[1])).compose(warp))
    return out


def _mapped_transforms(
    warps: Sequence[SimilarityTransform],
    sources: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray | None = None,
) -> list[SimilarityTransform]:
    """Compose the similarity that puts each tracked point on its chosen target.

    The generalisation of :func:`_target_transforms`, and the reason it is worth
    having: pinning *one* point can only translate the picture, so a shot that
    rotates or zooms still rotates and zooms around it. Given two or more
    point-to-target pairs, the transform that satisfies them **is** the
    rotation and scale the shot needs::

        q_k(t) = W(t) . p_k(t)                 # where the base warp puts point k
        S(t)   = argmin_S sum_k w_k |S q_k(t) - g_k|^2
        plan(t) = S(t) . W(t)

    ``S`` is the weighted Umeyama fit — the same
    :func:`~stable.solver.similarity.solve_similarity` the camera solve uses,
    not a second implementation of it — and it is composed **on top of** the
    sanctioned warp exactly as the single-target lock is, so nothing here
    reconstructs a transform from channel values.

    With one pair the fit degrades to a pure translation (rotation and scale are
    unobservable from a single correspondence), which is precisely the
    single-target behaviour. With three or more it is least-squares: a
    similarity generally cannot satisfy them all, and the residual per pair is
    reported rather than hidden — see
    :func:`stable.server.motion.target_residuals`.

    Args:
        warps: The base stabilisation warp per frame.
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
        target_positions: np.ndarray | None = None,
        anchor_frame: int | None = None,
        mapped_sources: np.ndarray | None = None,
        mapped_destinations: np.ndarray | None = None,
        mapped_weights: np.ndarray | None = None,
        crop: CropSpec | None = None,
        fill: BorderFill | None = None,
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
            target_positions: ``[T, 2]`` source-pixel path of the stabilisation
                target, or ``None`` for the plain smoothing-only warp. Gaps are
                held (:func:`~stable.render.options.hold_gaps`) rather than left
                to make the correction lurch.
            anchor_frame: Frame whose target position is held; defaults to the
                solve's reference frame, so the framing does not move.
            mapped_sources: ``[K, T, 2]`` tracked paths for a ``map`` target,
                mutually exclusive with ``target_positions``.
            mapped_destinations: ``[K, 2]`` chosen output positions for those
                paths. Two or more pairs is what lets the plan use rotation and
                scale rather than only translation — see
                :func:`_mapped_transforms`.
            mapped_weights: Optional ``[K, T]`` confidences for the fit.
            crop: How much of the frame the output keeps; ``None`` is the exact
                minimal gapless crop.
            fill: What goes in the uncovered area.

        Returns:
            The :class:`StabilizationPlan`.

        Raises:
            RenderError: If the border policy is unknown, an automatic crop
                finds the warped frames share no common area, or the target
                path does not cover the clip.
        """
        width, height = int(source_size[0]), int(source_size[1])
        warps: Sequence[SimilarityTransform] = motion.stabilization_transforms()
        target_centre: tuple[float, float] | None = None
        if mapped_sources is not None and mapped_destinations is not None:
            held_paths = np.stack([hold_gaps(path) for path in np.asarray(mapped_sources)])
            warps = _mapped_transforms(
                warps, held_paths, mapped_destinations, weights=mapped_weights
            )
            # The destinations are already where the operator wants the picture
            # framed, so a "crop around the target" centres on them.
            centroid = np.asarray(mapped_destinations, dtype=np.float64).reshape(-1, 2).mean(axis=0)
            target_centre = (float(centroid[0]), float(centroid[1]))
        elif target_positions is not None:
            held_path = hold_gaps(target_positions)
            frame = motion.reference_frame if anchor_frame is None else int(anchor_frame)
            warps = _target_transforms(warps, held_path, frame)
            anchor_point = warps[frame].apply(held_path[frame])
            target_centre = (float(anchor_point[0]), float(anchor_point[1]))
        # "Crop around the target" only means anything once the target is pinned:
        # without a lock its stabilised position moves, and a moving crop centre
        # would re-frame the shot every frame instead of cropping it.
        centre = target_centre if (crop is not None and crop.centre == "target") else None
        transforms = tuple(warps)
        geometry = plan_output(
            transforms, width, height, border, max_zoom=max_zoom, centre=centre, crop=crop
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
        target_positions: np.ndarray | None = None,
        **overrides: Any,
    ) -> StabilizationPlan:
        """Plan a render from a project file's ``render`` block.

        Args:
            motion: The smoothed solve.
            settings: The project's :class:`~stable.core.project.RenderSettings`
                — only its border policy is a planning decision; codec and CRF
                belong to the encode step.
            source_size: ``(width, height)`` of the clip.
            options: The operator's stabilisation target / crop / fill choices.
                The *target* is supplied resolved, as ``target_positions``,
                because turning "the point called p2" into a path needs the
                tracks and this module deliberately knows nothing about them.
            target_positions: ``[T, 2]`` path for ``options.target``.
            **overrides: Any other :meth:`build` argument.

        Returns:
            The :class:`StabilizationPlan`.
        """
        chosen = options or RenderOptions()
        return cls.build(
            motion,
            source_size=source_size,
            border=settings.border,
            target_positions=target_positions,
            anchor_frame=chosen.target.anchor_frame,
            crop=chosen.crop,
            fill=chosen.fill,
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
    target_positions: np.ndarray | None = None,
    anchor_frame: int | None = None,
    crop: CropSpec | None = None,
    fill: BorderFill | None = None,
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
        target_positions: ``[T, 2]`` path of the stabilisation target, if any.
        anchor_frame: Frame whose target position is held.
        crop: How much of the frame to keep.
        fill: What goes in the uncovered area.
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
        target_positions=target_positions,
        anchor_frame=anchor_frame,
        crop=crop,
        fill=fill,
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
