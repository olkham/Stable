"""Tests for FFmpeg (PyAV) encode with audio passthrough."""

from __future__ import annotations

import av
import numpy as np
import pytest
from conftest import encode_synthetic_clip

from stable.core.errors import VideoError
from stable.core.video import SyntheticVideoSource
from stable.video.encode import export_video, plan_audio
from stable.video.source import PyAVVideoSource


def _synthetic_frames(source: SyntheticVideoSource):
    for _index, frame in source.iter_frames():
        yield frame


def test_export_video_round_trips_frame_count_and_size(tmp_path):
    source = SyntheticVideoSource(frame_count=20, width=64, height=48, fps=24.0)
    out_path = tmp_path / "out.mp4"

    written = export_video(_synthetic_frames(source), out_path, fps=24.0, frame_count=20)

    with PyAVVideoSource(written) as decoded:
        assert decoded.frame_count == 20
        assert decoded.width == 64
        assert decoded.height == 48
        assert decoded.fps == pytest.approx(24.0, rel=0.05)


def test_export_video_reports_progress(tmp_path):
    source = SyntheticVideoSource(frame_count=10, width=32, height=24, fps=24.0)
    out_path = tmp_path / "out.mp4"
    seen: list[float] = []

    export_video(
        _synthetic_frames(source),
        out_path,
        fps=24.0,
        frame_count=10,
        progress=lambda f, m="": seen.append(f),
    )

    assert seen
    assert seen[-1] == pytest.approx(1.0)


def test_export_video_rejects_unknown_codec(tmp_path):
    source = SyntheticVideoSource(frame_count=3, width=16, height=16)
    with pytest.raises(VideoError, match="unknown codec"):
        export_video(_synthetic_frames(source), tmp_path / "out.mp4", fps=24.0, codec="av1")


def test_export_video_rejects_empty_frame_sequence(tmp_path):
    with pytest.raises(VideoError):
        export_video(iter(()), tmp_path / "out.mp4", fps=24.0)


def test_export_video_rejects_non_rgb_frames(tmp_path):
    def bad_frames():
        yield np.zeros((8, 8), dtype=np.uint8)  # missing the channel axis

    with pytest.raises(VideoError):
        export_video(bad_frames(), tmp_path / "out.mp4", fps=24.0)


def test_audio_passthrough_survives_export(tmp_path):
    """The Stack decision: 'FFmpeg encode with audio passthrough'."""
    source = SyntheticVideoSource(frame_count=24, width=48, height=32, fps=24.0)
    audio_source_path = str(tmp_path / "with_audio.mp4")
    encode_synthetic_clip(source, audio_source_path, with_audio=True)

    with av.open(audio_source_path) as probe:
        assert len(probe.streams.audio) == 1

    out_path = tmp_path / "restabilized.mp4"
    export_video(
        _synthetic_frames(source),
        out_path,
        fps=24.0,
        frame_count=24,
        audio_source=audio_source_path,
    )

    with av.open(str(out_path)) as result:
        assert len(result.streams.audio) == 1
        assert len(result.streams.video) == 1


def test_export_without_audio_source_has_no_audio_stream(tmp_path):
    source = SyntheticVideoSource(frame_count=5, width=32, height=24, fps=24.0)
    out_path = tmp_path / "silent.mp4"

    export_video(_synthetic_frames(source), out_path, fps=24.0, frame_count=5)

    with av.open(str(out_path)) as result:
        assert len(result.streams.audio) == 0


# -- audio the output container cannot hold ----------------------------------
#
# A camera that records uncompressed PCM produces a clip whose audio the MP4
# muxer accepts packet by packet and then rejects at the trailer. That made a
# finished 4K render fail with `Invalid argument: '...stabilized.mp4' returned
# 22` after every frame had already been encoded. These tests are written
# against a real PCM clip, because the failure only exists in the muxer.


#
# Reproducing it needs the camera's exact stream, and the discriminating detail
# is narrower than "PCM": the reported clip's audio carries an *unspecified*
# channel layout (PyAV shows it as `'2 channels'`, not `'stereo'`). PCM with a
# named layout muxes into MP4 quite happily -- a fixture written with
# `layout="stereo"` passes through and proves nothing. So the fixture below
# builds the unspecified layout deliberately, and asserts it got one.


def _clip_with_pcm_audio(path: str, source: SyntheticVideoSource, *, layout: str) -> str:
    """Encode a clip whose audio is uncompressed ``pcm_s16be``, as cameras record."""
    output = av.open(path, mode="w")
    try:
        video_stream = output.add_stream("libx264", rate=int(round(source.fps)) or 24)
        video_stream.width = source.width
        video_stream.height = source.height
        video_stream.pix_fmt = "yuv420p"
        audio_stream = output.add_stream("pcm_s16be", rate=48000)
        audio_stream.codec_context.layout = layout

        for index, frame in source.iter_frames():
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            for packet in video_stream.encode(video_frame):
                output.mux(packet)
        for packet in video_stream.encode(None):
            output.mux(packet)

        # `s16` is packed (interleaved), so one row of samples * channels.
        samples = np.zeros((1, 48000 * 2), dtype=np.int16)
        audio_frame = av.AudioFrame.from_ndarray(samples, format="s16", layout=layout)
        audio_frame.sample_rate = 48000
        audio_frame.pts = 0
        for packet in audio_stream.encode(audio_frame):
            output.mux(packet)
        for packet in audio_stream.encode(None):
            output.mux(packet)
    finally:
        output.close()
    return path


@pytest.fixture
def pcm_clip(tmp_path):
    """A clip matching the reported camera's audio: PCM with an unspecified layout."""
    source = SyntheticVideoSource(frame_count=12, width=48, height=32, fps=24.0)
    path = _clip_with_pcm_audio(str(tmp_path / "camera.mov"), source, layout="2 channels")
    with av.open(path) as probe:
        stream = probe.streams.audio[0]
        assert stream.codec_context.name == "pcm_s16be"
        # If this ever reads "stereo" the fixture has stopped reproducing the bug.
        assert str(stream.codec_context.layout.name) == "2 channels"
    return path, source


def test_plan_audio_refuses_passthrough_that_the_container_cannot_finish(pcm_clip, tmp_path):
    path, _source = pcm_clip
    plan = plan_audio(path, tmp_path / "out.mp4")
    assert plan.mode == "transcode"
    assert plan.codec == "pcm_s16be"
    assert "aac" in plan.reason


def test_plan_audio_keeps_passthrough_where_the_container_allows_it(pcm_clip, tmp_path):
    """The same audio into .mov is fine — this is a muxer limit, not a codec one."""
    path, _source = pcm_clip
    assert plan_audio(path, tmp_path / "out.mov").mode == "passthrough"


def test_pcm_with_a_named_layout_still_passes_through_into_mp4(tmp_path):
    """The refusal is specific, not "PCM is banned" — it must not over-reach."""
    source = SyntheticVideoSource(frame_count=8, width=32, height=24, fps=24.0)
    path = _clip_with_pcm_audio(str(tmp_path / "named.mov"), source, layout="stereo")
    assert plan_audio(path, tmp_path / "out.mp4").mode == "passthrough"


def test_incompatible_audio_is_re_encoded_instead_of_sinking_the_render(pcm_clip, tmp_path):
    """The reported bug: this used to raise at the trailer, losing the whole export."""
    path, source = pcm_clip
    out_path = tmp_path / "stabilized.mp4"

    written = export_video(
        _synthetic_frames(source),
        out_path,
        fps=24.0,
        frame_count=source.frame_count,
        audio_source=path,
    )

    with av.open(written) as result:
        assert len(result.streams.video) == 1
        # The audio is kept, converted -- silence would be a worse answer.
        assert len(result.streams.audio) == 1
        assert result.streams.audio[0].codec_context.name == "aac"


def test_compatible_audio_is_still_copied_verbatim(tmp_path):
    """Passthrough is not abandoned; it just stops being assumed."""
    source = SyntheticVideoSource(frame_count=12, width=48, height=32, fps=24.0)
    audio_path = str(tmp_path / "aac.mp4")
    encode_synthetic_clip(source, audio_path, with_audio=True)

    plan = plan_audio(audio_path, tmp_path / "out.mp4")
    assert plan.mode == "passthrough"
    assert plan.reason == ""


def test_a_source_with_no_audio_plans_nothing(tmp_path):
    source = SyntheticVideoSource(frame_count=6, width=32, height=24, fps=24.0)
    silent = str(tmp_path / "silent_source.mp4")
    encode_synthetic_clip(source, silent, with_audio=False)

    plan = plan_audio(silent, tmp_path / "out.mp4")
    assert plan.mode == "none"
    assert "no audio track" in plan.reason


def test_an_unreadable_audio_source_does_not_stop_the_export(tmp_path):
    """A video exported without sound beats one that does not export."""
    source = SyntheticVideoSource(frame_count=5, width=32, height=24, fps=24.0)
    missing = tmp_path / "not_a_video.mp4"
    missing.write_bytes(b"this is not a video file")

    assert plan_audio(missing, tmp_path / "out.mp4").mode == "none"

    written = export_video(
        _synthetic_frames(source),
        tmp_path / "out.mp4",
        fps=24.0,
        frame_count=5,
        audio_source=missing,
    )
    with av.open(written) as result:
        assert len(result.streams.video) == 1
        assert len(result.streams.audio) == 0


def test_a_failure_to_open_the_output_names_the_output(tmp_path):
    """`av.open` used to sit outside the try, so its errors escaped unwrapped."""
    source = SyntheticVideoSource(frame_count=3, width=16, height=16)
    with pytest.raises(VideoError, match="cannot open"):
        export_video(_synthetic_frames(source), tmp_path / "out.unknownext", fps=24.0)
