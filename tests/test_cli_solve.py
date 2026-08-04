"""``stable solve`` and the argument parsing both Phase 3 commands share."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
from conftest import WarpedClipSource, handheld_transforms

from stable.cli.main import main
from stable.cli.motion import (
    build_smoothing_settings,
    load_tracks,
    parse_channel_filter_spec,
    parse_channels,
    parse_filter_spec,
)
from stable.core.errors import StableError
from stable.core.geometry import CHANNELS
from stable.core.types import TrackResult

GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0)]


@pytest.fixture
def tracks_file(tmp_path):
    """A tracks JSON file of a known handheld path, as ``stable track`` writes it."""
    from stable.cli.track import track_result_to_json_dict
    from stable.core.types import Query

    clip = WarpedClipSource(handheld_transforms(24, pivot=(160.0, 120.0)))
    result = clip.true_tracks(GRID)
    payload = track_result_to_json_dict(
        result,
        tracker="lk",
        video_path="clip.mp4",
        frame_range=(0, clip.frame_count),
        direction="bidir",
        queries=[Query(frame=0, x=x, y=y) for x, y in GRID],
    )
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path), clip


# -- specification parsing ---------------------------------------------------


def test_a_bare_filter_name_parses():
    """The simplest form: just the filter, all defaults."""
    assert parse_filter_spec("gaussian") == ("gaussian", {})


def test_filter_parameters_parse_and_validate():
    """Parameters are checked against the catalogue at parse time."""
    name, params = parse_filter_spec("butterworth:cutoff_hz=1.5,order=3")
    assert name == "butterworth"
    assert params == {"cutoff_hz": "1.5", "order": "3"}


@pytest.mark.parametrize(
    "spec",
    ["", "nosuchfilter", "butterworth:cutoff_hz", "butterworth:order=99", "gaussian:sigma=abc"],
)
def test_bad_filter_specifications_are_argparse_errors(spec):
    """A typo fails at the command line, before any decoding happens."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_filter_spec(spec)


def test_channel_filter_specifications_parse():
    """``channel:filter:params`` targets one curve."""
    assert parse_channel_filter_spec("r:gaussian:sigma=12") == ("r", "gaussian", {"sigma": "12"})
    assert parse_channel_filter_spec("s:tripod") == ("s", "tripod", {})


@pytest.mark.parametrize("spec", ["gaussian", "q:gaussian", "r:nosuchfilter"])
def test_bad_channel_filter_specifications_are_rejected(spec):
    """Both halves are validated."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_channel_filter_spec(spec)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("xyrs", ["x", "y", "r", "s"]),
        ("xy", ["x", "y"]),
        ("s,r", ["r", "s"]),
        ("SR", ["r", "s"]),
        ("none", []),
    ],
)
def test_channel_sets_parse_as_letters_or_lists(spec, expected):
    """Both ``xyr`` and ``x,y,r`` work, and the result is in canonical order."""
    assert parse_channels(spec) == expected


def test_an_unknown_channel_letter_is_rejected():
    """Only the four solved channels exist."""
    with pytest.raises(argparse.ArgumentTypeError, match="not channels"):
        parse_channels("xyz")


def test_smoothing_settings_are_built_from_the_arguments():
    """``--smooth`` applies to the enabled channels; ``--smooth-channel`` overrides."""
    args = argparse.Namespace(
        smooth=[("gaussian", {"sigma": 6.0})],
        smooth_channel=[("r", "tripod", {})],
        channels=["x", "y", "r"],
        strength=None,
        tripod=False,
    )
    settings = build_smoothing_settings(args)
    assert settings.active_channels == ["x", "y", "r"]
    assert settings.channel("x").filters[0].filter == "gaussian"
    assert settings.channel("r").filters[0].filter == "tripod"
    assert settings.channel("s").active is False


def test_the_tripod_flag_is_shorthand_for_the_tripod_filter():
    """``--tripod`` locks every enabled channel."""
    args = argparse.Namespace(
        smooth=None, smooth_channel=None, channels=list(CHANNELS), strength=None, tripod=True
    )
    settings = build_smoothing_settings(args)
    assert all(spec.filter == "tripod" for spec in settings.to_specs())


def test_strength_is_applied_to_every_active_channel():
    """One flag, one meaning, across the whole configuration."""
    args = argparse.Namespace(
        smooth=[("gaussian", {})],
        smooth_channel=None,
        channels=["x", "y"],
        strength=0.4,
        tripod=False,
    )
    settings = build_smoothing_settings(args)
    assert settings.channel("x").strength == 0.4
    assert settings.channel("y").strength == 0.4


def test_a_default_filter_is_used_only_when_nothing_was_asked_for():
    """``stable stabilize`` needs a default; ``stable solve`` does not."""
    empty = argparse.Namespace(
        smooth=None, smooth_channel=None, channels=list(CHANNELS), strength=None, tripod=False
    )
    assert build_smoothing_settings(empty, default=None).active_channels == []
    assert build_smoothing_settings(empty, default="gaussian:sigma=5").active_channels == list(
        CHANNELS
    )

    explicit = argparse.Namespace(
        smooth=[("moving_average", {})],
        smooth_channel=None,
        channels=list(CHANNELS),
        strength=None,
        tripod=False,
    )
    settings = build_smoothing_settings(explicit, default="gaussian:sigma=5")
    assert settings.channel("x").filters[0].filter == "moving_average"


# -- loading tracks ----------------------------------------------------------


def test_tracks_load_from_the_json_the_tracker_wrote(tracks_file):
    """``stable track --out tracks.json`` feeds straight into ``stable solve``."""
    path, clip = tracks_file
    loaded = load_tracks(path)
    assert loaded.n_points == len(GRID)
    assert loaded.n_frames == clip.frame_count


def test_tracks_load_from_npz(tmp_path):
    """The raw array format works too."""
    result = TrackResult.empty(2, 6)
    result.save_npz(tmp_path / "t.npz")
    assert load_tracks(tmp_path / "t.npz").n_frames == 6


@pytest.mark.parametrize(
    ("contents", "message"),
    [("not json at all", "not valid JSON"), ('{"hello": 1}', "does not look like")],
)
def test_malformed_track_files_are_reported_clearly(tmp_path, contents, message):
    """The user is told what is wrong with the file they named."""
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(StableError, match=message):
        load_tracks(path)


def test_a_missing_track_file_is_reported(tmp_path):
    """No traceback for a wrong path."""
    with pytest.raises(StableError, match="cannot read tracks"):
        load_tracks(tmp_path / "nope.json")


# -- the command -------------------------------------------------------------


def test_solve_writes_a_csv_of_the_curves(tmp_path, tracks_file, capsys):
    """The headline export: one row per frame of X, Y, R, S."""
    path, clip = tracks_file
    out = tmp_path / "curves.csv"
    assert main(["solve", path, "--out", str(out)]) == 0

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frame,x,y,r_rad,r_deg,s")
    assert len(lines) == clip.frame_count + 1
    assert "solved" in capsys.readouterr().out


def test_solve_writes_json_when_asked(tmp_path, tracks_file):
    """Same data, machine-readable."""
    path, clip = tracks_file
    out = tmp_path / "curves.json"
    assert main(["solve", path, "--out", str(out), "--quiet"]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["curves"]["x"]) == clip.frame_count
    assert data["summary"]["status_counts"]["solved"] == clip.frame_count


def test_solve_recovers_the_known_motion_through_the_cli(tmp_path, tracks_file):
    """End to end through argument parsing, the ground truth still comes back."""
    path, clip = tracks_file
    out = tmp_path / "curves.json"
    main(["solve", path, "--out", str(out), "--quiet", "--pivot", "160,120"])

    curves = json.loads(out.read_text(encoding="utf-8"))["curves"]
    for frame in range(clip.frame_count):
        truth = clip.relative_transform(frame)
        moved = truth.apply(np.asarray([160.0, 120.0]))
        assert curves["x"][frame] == pytest.approx(moved[0] - 160.0, abs=1e-3)
        assert curves["r"][frame] == pytest.approx(truth.rotation, abs=1e-5)


def test_solve_exports_smoothed_curves_when_smoothing_is_configured(tmp_path, tracks_file):
    """With ``--smooth`` the export carries raw and smoothed side by side."""
    path, _ = tracks_file
    out = tmp_path / "curves.csv"
    assert main(["solve", path, "--out", str(out), "--smooth", "gaussian:sigma=4", "--quiet"]) == 0
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith(
        "frame,raw_x,raw_y,raw_r,raw_s,smooth_x"
    )


def test_solve_honours_the_reference_frame(tmp_path, tracks_file):
    """The reference frame's own transform is the identity, wherever it is."""
    path, _ = tracks_file
    out = tmp_path / "curves.json"
    main(["solve", path, "--out", str(out), "--quiet", "--reference-frame", "9"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["reference_frame"] == 9
    assert data["curves"]["x"][9] == pytest.approx(0.0, abs=1e-6)
    assert data["curves"]["s"][9] == pytest.approx(1.0, abs=1e-9)


def test_solve_can_lock_degrees_of_freedom(tmp_path, tracks_file):
    """``--solve-channels`` constrains the fit itself."""
    path, _ = tracks_file
    out = tmp_path / "curves.json"
    main(["solve", path, "--out", str(out), "--quiet", "--solve-channels", "xy"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["curves"]["r"]) == {0.0}
    assert set(data["curves"]["s"]) == {1.0}


def test_solve_reports_a_bad_reference_frame(tmp_path, tracks_file, capsys):
    """A configuration mistake exits non-zero with an explanation."""
    path, _ = tracks_file
    assert main(["solve", path, "--out", str(tmp_path / "x.csv"), "--reference-frame", "999"]) == 1
    assert "outside the clip" in capsys.readouterr().err


def test_solve_reports_a_missing_tracks_file(tmp_path, capsys):
    """No traceback, just the problem."""
    assert main(["solve", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_solve_can_list_the_filters(tracks_file, capsys):
    """``--list-filters`` documents the catalogue without running anything."""
    path, _ = tracks_file
    assert main(["solve", path, "--out", "unused.csv", "--list-filters"]) == 0
    assert "butterworth" in capsys.readouterr().out


def test_the_filters_command_lists_the_catalogue(capsys):
    """Discoverability at the top level too, in both formats."""
    assert main(["filters"]) == 0
    assert "Savitzky-Golay" in capsys.readouterr().out
    assert main(["filters", "--json"]) == 0
    catalogue = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in catalogue} >= {"tripod", "spline"}
