"""``stable stabilize``: the whole Phase 3 pipeline as one command."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from conftest import WarpedClipSource, encode_synthetic_clip, handheld_transforms

from stable.cli.main import main
from stable.cli.track import track_result_to_json_dict
from stable.core.types import Query

pytest.importorskip("cv2")
pytest.importorskip("av")

GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0), (160.0, 120.0)]


@pytest.fixture(scope="module")
def stabilize_inputs(tmp_path_factory):
    """A real handheld-ish clip on disk (with audio) plus its tracks file."""
    directory = tmp_path_factory.mktemp("stabilize")
    clip = WarpedClipSource(
        handheld_transforms(30, pivot=(160.0, 120.0), amplitude=7.0), fps=25.0
    )
    video_path = str(directory / "clip.mp4")
    encode_synthetic_clip(clip, video_path, with_audio=True, crf=12)

    payload = track_result_to_json_dict(
        clip.true_tracks(GRID),
        tracker="lk",
        video_path=video_path,
        frame_range=(0, clip.frame_count),
        direction="bidir",
        queries=[Query(frame=0, x=x, y=y) for x, y in GRID],
    )
    tracks_path = str(directory / "tracks.json")
    with open(tracks_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return video_path, tracks_path, clip


def has_audio(path: str) -> bool:
    """Whether ``path`` carries at least one audio stream."""
    import av

    with av.open(path) as container:
        return bool(container.streams.audio)


def frame_count(path: str) -> int:
    """Return how many frames the file at ``path`` decodes to."""
    from stable.video.source import PyAVVideoSource

    with PyAVVideoSource(path) as video:
        return video.frame_count


# -- the happy path ----------------------------------------------------------


def test_stabilize_writes_a_video_and_reports_what_it_did(tmp_path, stabilize_inputs, capsys):
    """The default run: solve, smooth, auto-zoom, encode."""
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "stable.mp4")
    assert main(["stabilize", video, "--tracks", tracks, "--out", out]) == 0

    printed = capsys.readouterr().out
    assert "solved" in printed
    assert "border: auto_zoom" in printed
    assert "smoothing:" in printed
    assert os.path.exists(out)
    assert frame_count(out) == clip.frame_count


def test_audio_survives_the_stabilised_export(tmp_path, stabilize_inputs):
    """The acceptance criterion, through the command a user would actually run."""
    video, tracks, _ = stabilize_inputs
    assert has_audio(video), "the fixture clip must have audio for this to mean anything"

    out = str(tmp_path / "with_audio.mp4")
    assert main(["stabilize", video, "--tracks", tracks, "--out", out, "--quiet"]) == 0
    assert has_audio(out)


def test_no_audio_drops_the_track_on_request(tmp_path, stabilize_inputs):
    """The passthrough is the default, not the only option."""
    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "silent.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--quiet", "--no-audio"]
    ) == 0
    assert not has_audio(out)


# -- border policies ---------------------------------------------------------

def test_crop_produces_a_smaller_file(tmp_path, stabilize_inputs, capsys):
    """Crop trades resolution for keeping every pixel real."""
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "cropped.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--border", "crop"]
    ) == 0

    from stable.video.source import PyAVVideoSource

    with PyAVVideoSource(out) as decoded:
        assert decoded.width < clip.width
        assert decoded.height < clip.height
        assert decoded.width % 2 == 0
    assert "border: crop" in capsys.readouterr().out


def test_transparent_renders_a_png_sequence(tmp_path, stabilize_inputs):
    """A directory destination carries the alpha a video container cannot."""
    video, tracks, clip = stabilize_inputs
    out = tmp_path / "frames"
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(out),
            "--border", "transparent", "--quiet",
        ]
    ) == 0
    written = sorted(os.listdir(out))
    assert len(written) == clip.frame_count
    assert written[0].endswith(".png")


def test_transparent_into_a_video_file_fails_with_advice(tmp_path, stabilize_inputs, capsys):
    """Losing the alpha silently would be worse than refusing."""
    video, tracks, _ = stabilize_inputs
    assert main(
        [
            "stabilize", video, "--tracks", tracks,
            "--out", str(tmp_path / "no.mp4"), "--border", "transparent",
        ]
    ) == 1
    assert "PNG sequence" in capsys.readouterr().err


def test_max_zoom_warns_when_it_clamps(tmp_path, stabilize_inputs, capsys):
    """The user asked to keep their framing, so they are told what it cost."""
    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "capped.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--max-zoom", "1.001"]
    ) == 0
    assert "--max-zoom capped" in capsys.readouterr().err


# -- smoothing options -------------------------------------------------------


def test_tripod_locks_the_shot(tmp_path, stabilize_inputs):
    """``--tripod`` renders a locked-off camera and needs a bigger crop for it."""
    video, tracks, _ = stabilize_inputs
    locked = str(tmp_path / "tripod.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", locked, "--tripod", "--quiet"]
    ) == 0
    assert os.path.exists(locked)


def test_strength_zero_leaves_the_clip_alone(tmp_path, stabilize_inputs, capsys):
    """A bypass must not crop, zoom or otherwise touch the picture."""
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "bypass.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--strength", "0"]
    ) == 0
    printed = capsys.readouterr().out
    assert "zoom 1.0000" in printed

    from stable.video.source import PyAVVideoSource

    with PyAVVideoSource(out) as decoded:
        assert (decoded.width, decoded.height) == (clip.width, clip.height)


def test_per_channel_smoothing_reaches_the_render(tmp_path, stabilize_inputs, capsys):
    """``--smooth-channel`` and ``--channels`` are reported and applied."""
    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "channels.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--channels", "xy", "--smooth-channel", "r:tripod",
        ]
    ) == 0
    report = capsys.readouterr().out
    assert "r=tripod@1" in report
    assert "s=off" in report


def test_the_curves_can_be_exported_alongside_the_video(tmp_path, stabilize_inputs):
    """One command, both deliverables."""
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "both.mp4")
    curves = tmp_path / "both.csv"
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--curves", str(curves), "--quiet"]
    ) == 0
    assert len(curves.read_text(encoding="utf-8").splitlines()) == clip.frame_count + 1


def test_a_frame_range_renders_part_of_the_clip(tmp_path, stabilize_inputs):
    """Useful for checking a fix without waiting for the whole shot."""
    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "part.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--frame-range", "4:14", "--quiet",
        ]
    ) == 0
    assert frame_count(out) == 10


def test_the_gpu_flag_falls_back_to_the_cpu(tmp_path, stabilize_inputs):
    """``--gpu`` is an optimisation; without CUDA the render still completes."""
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "gpu.mp4")
    assert main(["stabilize", video, "--tracks", tracks, "--out", out, "--gpu", "--quiet"]) == 0
    assert frame_count(out) == clip.frame_count


# -- failures ----------------------------------------------------------------


def test_tracks_from_a_different_clip_are_refused(tmp_path, stabilize_inputs, capsys):
    """A frame-count mismatch is the tell, and it is caught before rendering."""
    video, _, clip = stabilize_inputs
    short = WarpedClipSource(handheld_transforms(9, pivot=(160.0, 120.0)))
    payload = track_result_to_json_dict(
        short.true_tracks(GRID),
        tracker="lk",
        video_path="other.mp4",
        frame_range=(0, 9),
        direction="bidir",
        queries=[Query(frame=0, x=x, y=y) for x, y in GRID],
    )
    other = tmp_path / "other.json"
    other.write_text(json.dumps(payload), encoding="utf-8")

    assert main(
        ["stabilize", video, "--tracks", str(other), "--out", str(tmp_path / "x.mp4")]
    ) == 1
    assert "different clip" in capsys.readouterr().err


def test_a_missing_video_is_reported(tmp_path, stabilize_inputs, capsys):
    """No traceback for a wrong path."""
    _, tracks, _ = stabilize_inputs
    assert main(
        ["stabilize", str(tmp_path / "nope.mp4"), "--tracks", tracks, "--out", "x.mp4"]
    ) == 1
    assert "error:" in capsys.readouterr().err


def test_an_impossible_filter_setting_is_reported(tmp_path, stabilize_inputs, capsys):
    """A cutoff above Nyquist is caught by argparse, before any work starts."""
    video, tracks, _ = stabilize_inputs
    with pytest.raises(SystemExit):
        main(
            [
                "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "x.mp4"),
                "--smooth", "butterworth:order=99",
            ]
        )
    assert "order" in capsys.readouterr().err


def test_the_stabilised_clip_really_is_steadier(tmp_path, stabilize_inputs):
    """The output moves less than the input — the reason the command exists."""
    from stable.cli.motion import load_tracks
    from stable.solver import solve_motion
    from stable.video.source import PyAVVideoSource

    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "steadier.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", out, "--tripod", "--quiet"]
    ) == 0

    raw = solve_motion(load_tracks(tracks), frame_size=(320, 240))
    # A tripod render pins the frame to the reference, so the motion it removed
    # is the whole of the raw motion.
    assert np.abs(raw.curve("x")).max() > 3.0

    with PyAVVideoSource(out) as decoded:
        first = decoded.get_frame(0).astype(np.float64)
        last = decoded.get_frame(decoded.frame_count - 1).astype(np.float64)
    with PyAVVideoSource(video) as decoded:
        raw_first = decoded.get_frame(0).astype(np.float64)
        raw_last = decoded.get_frame(decoded.frame_count - 1).astype(np.float64)

    # Frame-to-frame difference across the whole shot: much smaller once locked.
    assert np.abs(last - first).mean() < np.abs(raw_last - raw_first).mean()
