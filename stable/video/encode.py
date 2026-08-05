"""FFmpeg encode via PyAV, with audio passthrough — the export half of video I/O.

:func:`export_video` is the utility every future render/export path (Phase 3's
stabilized render, Phase 4's export dialog) builds on: it takes a sequence of
RGB frames, encodes them with a real codec, and — when given the original
clip as ``audio_source`` — copies its audio track into the output verbatim, no
re-encode. That satisfies the Stack decision "FFmpeg encode with audio
passthrough" without any subsystem needing to touch PyAV directly.

When passthrough is not possible
--------------------------------
"Copy the audio track verbatim" quietly assumes the output container can hold
whatever the source recorded, and MP4 cannot hold nearly as much as MOV can.
A camera that records **uncompressed PCM** (``pcm_s16be`` is common) produces a
``.MP4`` whose audio the MP4 muxer will accept packet by packet and then refuse
at ``av_write_trailer`` — so the failure arrived at the *end*, as
``Invalid argument: '...stabilized.mp4' returned 22``, after every frame had
already been encoded. On a 900-frame 4K clip that is minutes of work thrown
away with a message naming nothing that caused it.

So the audio decision is made **before the first frame is encoded**, by
:func:`plan_audio`, and it is made by *trying it* rather than by consulting a
table: a one-packet container is written to a temporary file with the real
output's extension. ``OutputContainer.supported_codecs`` cannot be used for
this — it reports ``pcm_s16be`` as supported by MP4, which is exactly the case
that fails. Codec support is a property of the muxer, and the only reliable
oracle for it is the muxer.

When passthrough is refused the audio is **re-encoded to AAC** rather than
dropped: silence is a worse answer than a lossy copy for anyone stabilising a
shot they intend to cut with sound. Passthrough remains what happens whenever
it can, so the Stack decision holds wherever it is actually applicable.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from ..core.errors import VideoError
from ..trackers.base import ProgressCallback, noop_progress

__all__ = ["CODECS", "AudioPlan", "plan_audio", "export_video"]

logger = logging.getLogger(__name__)

CODECS: dict[str, str] = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}
"""Map from a project file's ``render.codec`` name to an FFmpeg encoder name."""

FALLBACK_AUDIO_CODEC = "aac"
"""What incompatible audio is re-encoded to. Universally playable in MP4."""

AudioMode = Literal["none", "passthrough", "transcode"]


@dataclass(frozen=True)
class AudioPlan:
    """What will happen to the source's audio, decided before any frame is encoded.

    Args:
        mode: ``"none"`` (no audio track, or none usable), ``"passthrough"``
            (copied verbatim) or ``"transcode"`` (re-encoded to
            :data:`FALLBACK_AUDIO_CODEC`).
        reason: Human-readable explanation, for logs and the UI. Empty when
            nothing surprising happened.
        codec: The source audio codec's name, when there is one.
    """

    mode: AudioMode
    reason: str = ""
    codec: str | None = None


def _first_audio_packet(container: Any, stream: Any) -> Any | None:
    """Return the first real (non-empty, timestamped) packet of an audio stream."""
    for packet in container.demux(stream):
        if packet.dts is not None and packet.size:
            return packet
    return None


def plan_audio(audio_source: str | os.PathLike[str], out_path: str | os.PathLike[str]) -> AudioPlan:
    """Decide how the source's audio can reach an output with ``out_path``'s extension.

    Writes a one-packet throwaway container with the same suffix as ``out_path``
    and asks the muxer to finish it. If that succeeds the real export can copy
    the audio verbatim; if it fails, only re-encoding will work. See the module
    docstring for why this is done empirically.

    Args:
        audio_source: Path to the clip whose audio track is wanted.
        out_path: Where the real export will be written — only its extension
            matters, since that is what selects the muxer.

    Returns:
        The :class:`AudioPlan` the export should follow. Never raises for a
        missing, unreadable or audio-less source: those are ``mode="none"``,
        because a video that exports without sound beats one that does not
        export.
    """
    av = _av()
    try:
        with av.open(os.fspath(audio_source)) as source:
            if not source.streams.audio:
                return AudioPlan("none", "the source clip has no audio track")
            stream = source.streams.audio[0]
            codec = getattr(stream.codec_context, "name", None)
            packet = _first_audio_packet(source, stream)
            if packet is None:
                return AudioPlan("none", "the source's audio track is empty", codec)
            if _passthrough_writes(av, stream, packet, out_path):
                return AudioPlan("passthrough", "", codec)
    except Exception as exc:  # noqa: BLE001 - audio must never sink the whole render
        logger.warning("could not inspect audio in %r: %s", str(audio_source), exc)
        return AudioPlan("none", f"the source's audio could not be read ({exc})")

    suffix = os.path.splitext(os.fspath(out_path))[1] or ".mp4"
    return AudioPlan(
        "transcode",
        f"{suffix} cannot store {codec} audio, so it was re-encoded to {FALLBACK_AUDIO_CODEC}",
        codec,
    )


def _passthrough_writes(
    av: Any, stream: Any, packet: Any, out_path: str | os.PathLike[str]
) -> bool:
    """Whether a container like ``out_path`` can be *finished* holding this stream.

    Muxing a packet is not the test — the MP4 muxer accepts PCM packets happily
    and only rejects them when the trailer is written, which is the whole reason
    this function exists. The throwaway container is therefore closed, not just
    written to.
    """
    suffix = os.path.splitext(os.fspath(out_path))[1] or ".mp4"
    handle, probe_path = tempfile.mkstemp(suffix=suffix, prefix="stable-audio-probe-")
    os.close(handle)
    try:
        probe = av.open(probe_path, mode="w")
        try:
            packet.stream = probe.add_stream_from_template(stream)
            probe.mux(packet)
        finally:
            probe.close()  # the trailer is written here, and is what can fail
        return True
    except Exception as exc:  # noqa: BLE001 - a refusal is the answer, not an error
        logger.debug("audio passthrough probe failed for %s: %s", suffix, exc)
        return False
    finally:
        try:
            os.unlink(probe_path)
        except OSError:  # pragma: no cover - Windows may hold it briefly
            pass


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
            stream is carried into the output — copied verbatim where the
            container allows it (the "audio passthrough" the Stack section
            requires), re-encoded to :data:`FALLBACK_AUDIO_CODEC` where it does
            not. See the module docstring.
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

    # Decided up front, on purpose: finding out at the trailer that the audio
    # cannot be written means discarding a render that is already complete.
    plan = plan_audio(audio_source, out_path) if audio_source is not None else AudioPlan("none")
    if plan.reason:
        logger.warning("audio for %r: %s", out_path, plan.reason)

    try:
        output = av.open(out_path, mode="w")
    except Exception as exc:
        raise VideoError(f"cannot open {out_path!r} for writing: {exc}") from exc

    video_stream = None
    audio_in_container = None
    audio_in_stream = None
    audio_out_stream = None
    written = 0
    failure: BaseException | None = None
    try:
        if plan.mode != "none" and audio_source is not None:
            audio_in_container = av.open(os.fspath(audio_source))
            audio_in_stream = audio_in_container.streams.audio[0]
            if plan.mode == "passthrough":
                audio_out_stream = output.add_stream_from_template(audio_in_stream)
            else:
                audio_out_stream = _add_transcoded_audio_stream(output, audio_in_stream)

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
            if plan.mode == "passthrough":
                _copy_audio(output, audio_in_container, audio_in_stream, audio_out_stream)
            else:
                _transcode_audio(av, output, audio_in_container, audio_in_stream, audio_out_stream)
    except VideoError as exc:
        failure = exc
        raise
    except Exception as exc:
        failure = exc
        raise VideoError(f"failed to export video to {out_path!r}: {exc}") from exc
    finally:
        # `close()` writes the trailer and can fail on its own. Letting that
        # propagate from `finally` would replace whatever actually went wrong
        # with a message naming only the file -- which is precisely how the
        # original PCM-in-MP4 failure managed to say nothing useful.
        try:
            output.close()
        except Exception as exc:  # noqa: BLE001 - see above
            if failure is None:
                raise VideoError(f"failed to finish writing {out_path!r}: {exc}") from exc
            logger.debug("closing %r also failed: %s", out_path, exc)
        finally:
            if audio_in_container is not None:
                audio_in_container.close()

    progress(1.0, "done")
    return os.path.abspath(out_path)


def _add_transcoded_audio_stream(output: Any, source_stream: Any) -> Any:
    """Add an AAC output stream matched to ``source_stream``'s rate and channel count.

    The channel layout is rebuilt from the channel *count* rather than copied.
    PCM tracks routinely carry an unnamed layout (PyAV shows it as
    ``'2 channels'``), and handing that to the AAC encoder fails
    ``avcodec_open2`` with the same ``Invalid argument`` this module exists to
    avoid. Anything wider than stereo is left to the source's own layout, which
    for real multichannel audio is a named one.
    """
    context = source_stream.codec_context
    channels = int(getattr(getattr(context, "layout", None), "nb_channels", 0) or 2)
    stream = output.add_stream(FALLBACK_AUDIO_CODEC, rate=context.sample_rate)
    if channels <= 1:
        stream.codec_context.layout = "mono"
    elif channels == 2:
        stream.codec_context.layout = "stereo"
    else:
        stream.codec_context.layout = context.layout
    return stream


def _copy_audio(output: Any, container: Any, in_stream: Any, out_stream: Any) -> None:
    """Remux an audio track verbatim into ``output``."""
    for packet in container.demux(in_stream):
        if packet.dts is None:
            continue
        packet.stream = out_stream
        output.mux(packet)


def _transcode_audio(av: Any, output: Any, container: Any, in_stream: Any, out_stream: Any) -> None:
    """Decode, resample and re-encode an audio track into ``output``.

    Timestamps are dropped before encoding (``frame.pts = None``) so the encoder
    lays down its own monotonic timeline: the resampler changes the sample rate
    and layout, which makes the source's timestamps no longer describe the
    frames being written.
    """
    context = out_stream.codec_context
    resampler = av.audio.resampler.AudioResampler(
        format=context.format, layout=context.layout, rate=context.sample_rate
    )
    for frame in container.decode(in_stream):
        frame.pts = None
        for resampled in resampler.resample(frame):
            for packet in out_stream.encode(resampled):
                output.mux(packet)
    for resampled in resampler.resample(None):
        for packet in out_stream.encode(resampled):
            output.mux(packet)
    for packet in out_stream.encode(None):
        output.mux(packet)
