"""FFmpeg encode via PyAV, with audio passthrough — the export half of video I/O.

:func:`export_video` is the utility every future render/export path (Phase 3's
stabilized render, Phase 4's export dialog) builds on: it takes a sequence of
RGB frames, encodes them with a real codec, and — when given the original
clip as ``audio_source`` — copies its audio track into the output verbatim, no
re-encode. That satisfies the Stack decision "FFmpeg encode with audio
passthrough" without any subsystem needing to touch PyAV directly.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import numpy as np

from ..core.errors import VideoError
from ..trackers.base import ProgressCallback, noop_progress

__all__ = ["CODECS", "export_video"]

CODECS: dict[str, str] = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
"""Map from a project file's ``render.codec`` name to an FFmpeg encoder name."""


def _av() -> Any:
    """Import and return :mod:`av`, lazily.

    Returns:
        The imported ``av`` module.

    Raises:
        VideoError: If PyAV is not installed.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise VideoError(
            "export_video requires the 'av' package; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return av


def export_video(
    frames: Iterable[np.ndarray],
    out_path: str | os.PathLike[str],
    *,
    fps: float,
    codec: str = "h264",
    crf: int = 18,
    audio_source: str | os.PathLike[str] | None = None,
    frame_count: int | None = None,
    progress: ProgressCallback = noop_progress,
) -> str:
    """Encode an RGB frame sequence to a video file, passing audio through untouched.

    Args:
        frames: ``uint8`` RGB ``(H, W, 3)`` frames, in order. All frames must
            share the same shape; the first frame's size becomes the output's.
        out_path: Destination file path; the container is inferred by FFmpeg
            from its extension (``.mp4`` is the expected case).
        fps: Output frame rate.
        codec: One of :data:`CODECS`' keys.
        crf: Constant rate factor passed to the encoder (lower = higher quality).
        audio_source: Optional path to a video/audio file whose first audio
            stream is copied into the output verbatim (no re-encode) — the
            "audio passthrough" the Stack section requires.
        frame_count: Total frame count, for progress reporting only; if omitted
            progress is reported as ``0`` until the final ``1.0``.
        progress: Progress reporter, called after every frame.

    Returns:
        The absolute path written.

    Raises:
        VideoError: If PyAV is unavailable, ``codec`` is unknown, ``frames`` is
            empty, or encoding fails.
    """
    av = _av()
    if codec not in CODECS:
        raise VideoError(f"unknown codec {codec!r}; expected one of {sorted(CODECS)}")
    out_path = os.fspath(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    output = av.open(out_path, mode="w")
    video_stream = None
    audio_in_container = None
    audio_in_stream = None
    audio_out_stream = None
    written = 0
    try:
        if audio_source is not None:
            audio_in_container = av.open(os.fspath(audio_source))
            if audio_in_container.streams.audio:
                audio_in_stream = audio_in_container.streams.audio[0]
                audio_out_stream = output.add_stream_from_template(audio_in_stream)

        rate = Fraction(fps).limit_denominator(1000)
        for frame in frames:
            array = np.ascontiguousarray(frame, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise VideoError(f"frames must be RGB (H, W, 3), got shape {array.shape}")
            if video_stream is None:
                height, width = array.shape[:2]
                video_stream = output.add_stream(CODECS[codec], rate=rate)
                video_stream.width = width
                video_stream.height = height
                video_stream.pix_fmt = "yuv420p"
                video_stream.options = {"crf": str(crf)}
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            video_frame.pts = written
            for packet in video_stream.encode(video_frame):
                output.mux(packet)
            written += 1
            progress((written / frame_count) if frame_count else 0.0, f"frame {written}")

        if written == 0 or video_stream is None:
            raise VideoError("export_video received no frames")

        for packet in video_stream.encode(None):
            output.mux(packet)

        if audio_in_container is not None and audio_out_stream is not None:
            for packet in audio_in_container.demux(audio_in_stream):
                if packet.dts is None:
                    continue
                packet.stream = audio_out_stream
                output.mux(packet)
    except VideoError:
        raise
    except Exception as exc:
        raise VideoError(f"failed to export video to {out_path!r}: {exc}") from exc
    finally:
        output.close()
        if audio_in_container is not None:
            audio_in_container.close()

    progress(1.0, "done")
    return os.path.abspath(out_path)
