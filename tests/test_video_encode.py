"""Tests for FFmpeg (PyAV) encode with audio passthrough."""

from __future__ import annotations

import av
import numpy as np
import pytest
from conftest import encode_synthetic_clip

from stable.core.errors import VideoError
from stable.core.video import SyntheticVideoSource
from stable.video.encode import export_video
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
