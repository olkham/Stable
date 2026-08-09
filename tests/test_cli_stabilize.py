"""``stable stabilize``: the whole Phase 3 pipeline as one command."""

from __future__ import annotations

import json
import os
import re

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


# -- todo 17: stabilisation target, crop size and border fill -----------------


def test_a_target_stabilises_with_no_smoothing_at_all(tmp_path, stabilize_inputs, capsys):
    """`--target` is the direct way to ask for stabilisation.

    Without it, `--smooth none` leaves `smoothed == raw`, the warp is the
    identity on every frame and the render is a re-encode — the web UI's
    version of exactly that is the bug behind todo 17.
    """
    from stable.video.source import PyAVVideoSource

    video, tracks, clip = stabilize_inputs
    flat = str(tmp_path / "flat.mp4")
    assert main(
        ["stabilize", video, "--tracks", tracks, "--out", flat, "--channels", "none"]
    ) == 0
    assert "re-encode of the input" in capsys.readouterr().err

    pinned = str(tmp_path / "pinned.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", pinned,
            "--channels", "none", "--target", "track:4",
        ]
    ) == 0
    printed = capsys.readouterr()
    assert "holding point 4" in printed.out
    assert "(50%, 50%) of the output" in printed.out
    assert "re-encode of the input" not in printed.err

    # Point 4 of GRID is the frame centre, so pinning it must steady the middle.
    def centre_drift(path: str) -> float:
        with PyAVVideoSource(path) as decoded:
            first = decoded.get_frame(0).astype(np.float64)
            last = decoded.get_frame(decoded.frame_count - 1).astype(np.float64)
        h, w = first.shape[:2]
        box = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
        return float(np.abs(last[box] - first[box]).mean())

    assert centre_drift(pinned) < centre_drift(flat) / 2.0
    assert frame_count(pinned) == clip.frame_count


def test_average_target_needs_no_point_index(tmp_path, stabilize_inputs, capsys):
    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "average.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--channels", "none", "--target", "average",
        ]
    ) == 0
    assert "holding every point" in capsys.readouterr().out


def test_a_target_index_outside_the_tracks_is_reported(tmp_path, stabilize_inputs, capsys):
    video, tracks, _ = stabilize_inputs
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "x.mp4"),
            "--target", "track:99",
        ]
    ) == 1
    assert "out of range" in capsys.readouterr().err


def test_the_target_can_be_placed_anywhere_in_the_output_frame(tmp_path, stabilize_inputs, capsys):
    """Todo 1, eighth round: ``--target-at`` states where in the delivered frame it sits."""
    video, tracks, _ = stabilize_inputs
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "thirds.mp4"),
            "--channels", "none", "--target", "track:4", "--target-at", "top-left",
        ]
    ) == 0
    assert "(33%, 33%) of the output" in capsys.readouterr().out


def test_a_placement_outside_the_output_frame_is_refused(tmp_path, stabilize_inputs, capsys):
    video, tracks, _ = stabilize_inputs
    with pytest.raises(SystemExit):
        main(
            [
                "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "x.mp4"),
                "--target", "track:0", "--target-at", "1.4,0.5",
            ]
        )
    assert "fractions of the output frame" in capsys.readouterr().err


def test_a_smoothed_target_holds_the_smoothed_estimate(tmp_path, stabilize_inputs, capsys):
    """'smoothed' holds the smoothed estimate of that track, not the raw one."""
    video, tracks, _ = stabilize_inputs
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "soft.mp4"),
            "--target", "smoothed:4", "--smooth", "butterworth:cutoff_hz=1.0",
        ]
    ) == 0
    assert "holding the smoothed estimate of point 4" in capsys.readouterr().out


def test_the_delivery_resolution_scales_the_crop(tmp_path, stabilize_inputs):
    """``--width``/``--height`` are delivery size, not field of view."""
    from stable.video.source import PyAVVideoSource

    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "delivered.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--border", "crop", "--crop", "0.5", "--width", "640", "--height", "480", "--quiet",
        ]
    ) == 0
    with PyAVVideoSource(out) as decoded:
        assert (decoded.width, decoded.height) == (640, 480)


def test_the_crop_can_be_measured_over_the_rendered_range_only(tmp_path, stabilize_inputs, capsys):
    """Frames outside ``--frame-range`` stop dictating the crop of the ones inside."""
    video, tracks, clip = stabilize_inputs
    window = f"0:{max(clip.frame_count // 3, 2)}"

    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "whole.mp4"),
            "--target", "track:4", "--frame-range", window,
        ]
    ) == 0
    whole = _zoom_of(capsys.readouterr().out)

    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "fitted.mp4"),
            "--target", "track:4", "--frame-range", window, "--crop-to-range",
        ]
    ) == 0
    printed = capsys.readouterr().out
    assert "measured over" in printed
    assert _zoom_of(printed) < whole


def _zoom_of(printed: str) -> float:
    """Pull the ``zoom N.NNNN`` the border line reports out of the CLI's output."""
    match = re.search(r"zoom ([0-9.]+)", printed)
    assert match, printed
    return float(match.group(1))


def test_crop_scale_sets_the_output_size(tmp_path, stabilize_inputs):
    from stable.video.source import PyAVVideoSource

    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / "half.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--border", "crop", "--crop", "0.5", "--quiet",
        ]
    ) == 0
    with PyAVVideoSource(out) as decoded:
        assert decoded.width == pytest.approx(clip.width // 2, abs=2)
        assert decoded.height == pytest.approx(clip.height // 2, abs=2)


def test_a_crop_wider_than_the_coverage_warns_and_fills(tmp_path, stabilize_inputs, capsys):
    """Keeping the whole frame is allowed; the fill is what makes it usable."""
    from stable.video.source import PyAVVideoSource

    video, tracks, _ = stabilize_inputs
    out = str(tmp_path / "filled.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out,
            "--border", "crop", "--crop", "1.0", "--tripod",
            "--fill", "mirror", "--feather", "4",
        ]
    ) == 0
    assert "fill will be visible" in capsys.readouterr().out

    black = str(tmp_path / "black.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", black,
            "--border", "crop", "--crop", "1.0", "--tripod", "--quiet",
        ]
    ) == 0

    def letterbox_pixels(path: str) -> int:
        with PyAVVideoSource(path) as decoded:
            frame = decoded.get_frame(decoded.frame_count - 1)
        return int(np.count_nonzero(frame.max(axis=2) < 12))

    assert letterbox_pixels(black) > 200
    assert letterbox_pixels(out) < letterbox_pixels(black) / 5


@pytest.mark.parametrize("mode", ["solid", "stretch", "mirror", "blur"])
def test_every_fill_mode_renders(tmp_path, stabilize_inputs, mode):
    video, tracks, clip = stabilize_inputs
    out = str(tmp_path / f"{mode}.mp4")
    assert main(
        [
            "stabilize", video, "--tracks", tracks, "--out", out, "--quiet",
            "--border", "crop", "--crop", "1.0", "--tripod",
            "--fill", mode, "--fill-color", "#20c040", "--fill-tint", "0.25",
            "--fill-blur", "3", "--feather", "2",
        ]
    ) == 0
    assert frame_count(out) == clip.frame_count


def test_a_bad_fill_colour_is_rejected_by_argparse(tmp_path, stabilize_inputs, capsys):
    video, tracks, _ = stabilize_inputs
    with pytest.raises(SystemExit):
        main(
            [
                "stabilize", video, "--tracks", tracks, "--out", str(tmp_path / "x.mp4"),
                "--fill-color", "300,0,0",
            ]
        )
    assert "0-255" in capsys.readouterr().err
