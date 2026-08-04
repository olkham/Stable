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
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import RenderError
from ..core.geometry import SimilarityTransform
from ..core.project import BORDER_POLICIES, RenderSettings
from ..core.video import VideoSource
from ..smoothing.motion import SmoothedMotion
from ..trackers.base import ProgressCallback, noop_progress
from ..video.encode import export_video
from .warp import OutputGeometry, load_cv2, plan_output, warp_frame

__all__ = [
    "StabilizationPlan",
    "iter_stabilized_frames",
    "render_stabilized",
    "write_frame_sequence",
    "stabilize",
    "stabilized_frame",
]


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
    """

    transforms: tuple[SimilarityTransform, ...]
    geometry: OutputGeometry
    source_size: tuple[int, int]
    interpolation: str = "linear"
    background: tuple[int, int, int] = (0, 0, 0)

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

        Returns:
            The :class:`StabilizationPlan`.

        Raises:
            RenderError: If the border policy is unknown or the warped frames
                share no common area.
        """
        width, height = int(source_size[0]), int(source_size[1])
        transforms = tuple(motion.stabilization_transforms())
        geometry = plan_output(transforms, width, height, border, max_zoom=max_zoom)
        return cls(
            transforms=transforms,
            geometry=geometry,
            source_size=(width, height),
            interpolation=interpolation,
            background=background,
        )

    @classmethod
    def from_settings(
        cls,
        motion: SmoothedMotion,
        settings: RenderSettings,
        *,
        source_size: tuple[int, int],
        **overrides: Any,
    ) -> StabilizationPlan:
        """Plan a render from a project file's ``render`` block.

        Args:
            motion: The smoothed solve.
            settings: The project's :class:`~stable.core.project.RenderSettings`
                — only its border policy is a planning decision; codec and CRF
                belong to the encode step.
            source_size: ``(width, height)`` of the clip.
            **overrides: Any other :meth:`build` argument.

        Returns:
            The :class:`StabilizationPlan`.
        """
        return cls.build(motion, source_size=source_size, border=settings.border, **overrides)

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
    start, end = (0, plan.n_frames) if frame_range is None else (
        int(frame_range[0]),
        int(frame_range[1]),
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

    start, end = (0, plan.n_frames) if frame_range is None else frame_range
    return export_video(
        frames,
        target,
        fps=float(video.fps) if fps is None else float(fps),
        codec=codec,
        crf=crf,
        audio_source=audio_source,
        frame_count=int(end) - int(start),
    )


def stabilize(
    video: VideoSource,
    motion: SmoothedMotion,
    out_path: str | os.PathLike[str],
    *,
    border: str = "auto_zoom",
    max_zoom: float | None = None,
    interpolation: str = "linear",
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
        use_gpu=use_gpu,
    )
