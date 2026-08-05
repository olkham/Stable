"""End-to-end test of the ``stable track`` CLI subcommand against a real clip.

This is the Phase 1 acceptance criterion "track a point through a real test
clip end-to-end from CLI" — it goes through :func:`stable.cli.main.main`, a
real encoded video file on disk, and the real ``lk`` tracker plugin.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stable.cli.main import build_parser, main
from stable.cli.track import parse_point_spec, track_result_from_json_dict
from stable.core.types import Query


def test_parse_point_spec():
    query = parse_point_spec("120.5,80@3")
    assert query == Query(3, 120.5, 80.0)


def test_parse_point_spec_rejects_bad_input():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_point_spec("not-a-point")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_point_spec("1,2,3@4")


def test_track_subcommand_is_registered():
    parser = build_parser()
    assert parser.prog == "stable"
    # argparse raises SystemExit(2) for missing required args; "track" alone
    # is missing --point/--out, so this only proves the subcommand exists.
    with pytest.raises(SystemExit):
        parser.parse_args(["track"])


def test_track_end_to_end_from_the_cli_writes_json(tmp_path, tiny_clip):
    video_path, source = tiny_clip
    anchor_frame = 5
    x, y = source.true_position(anchor_frame)
    out_path = tmp_path / "tracks.json"

    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@{anchor_frame}",
            "--tracker",
            "lk",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert out_path.exists()

    with open(out_path, encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["tracker"] == "lk"
    assert data["direction"] == "bidir"
    assert data["frame_count"] == source.frame_count
    assert len(data["points"]) == 1
    entry = data["points"][0]
    assert entry["query"] == {"frame": anchor_frame, "x": x, "y": y}
    assert len(entry["points"]) == source.frame_count
    assert len(entry["confidence"]) == source.frame_count
    assert len(entry["visible"]) == source.frame_count

    # every frame should have tracked successfully on this clean synthetic clip
    assert all(cell is not None for cell in entry["points"])
    for frame, cell in enumerate(entry["points"]):
        tx, ty = source.true_position(frame)
        assert (cell[0] - tx) ** 2 + (cell[1] - ty) ** 2 < 4.0**2


def test_track_output_json_round_trips_to_a_track_result(tmp_path, tiny_clip):
    video_path, source = tiny_clip
    anchor_frame = 4
    x, y = source.true_position(anchor_frame)
    out_path = tmp_path / "tracks.json"

    main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@{anchor_frame}",
            "--tracker",
            "template",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    with open(out_path, encoding="utf-8") as handle:
        data = json.load(handle)
    result = track_result_from_json_dict(data)
    assert result.n_points == 1
    assert result.n_frames == source.frame_count
    assert np.isfinite(result.points[0, anchor_frame]).all()


def test_track_supports_multiple_points_and_npz_output(tmp_path, tiny_clip):
    video_path, source = tiny_clip
    x0, y0 = source.true_position(2)
    x1, y1 = source.true_position(7)
    out_path = tmp_path / "tracks.npz"

    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x0},{y0}@2",
            "--point",
            f"{x1},{y1}@7",
            "--tracker",
            "lk",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    from stable.core.types import TrackResult

    result = TrackResult.load_npz(out_path)
    assert result.n_points == 2
    assert result.n_frames == source.frame_count


def test_track_respects_an_explicit_frame_range_and_direction(tmp_path, tiny_clip):
    video_path, source = tiny_clip
    anchor_frame = 10
    x, y = source.true_position(anchor_frame)
    out_path = tmp_path / "tracks.json"

    main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@{anchor_frame}",
            "--tracker",
            "lk",
            "--frame-range",
            "0:15",
            "--direction",
            "fwd",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )

    with open(out_path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["frame_range"] == [0, 15]
    entry = data["points"][0]
    # fwd from frame 10: frames before the anchor stay null; the range caps at 15.
    assert all(cell is None for cell in entry["points"][:anchor_frame])
    assert entry["points"][anchor_frame] is not None
    assert all(cell is None for cell in entry["points"][15:])


def test_track_reports_an_unknown_tracker_cleanly(tmp_path, tiny_clip, capsys):
    video_path, source = tiny_clip
    out_path = tmp_path / "tracks.json"
    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            "1,1@0",
            "--tracker",
            "not_a_real_tracker",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )
    assert exit_code == 1
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "not_a_real_tracker" in captured.err


def test_track_reports_a_missing_video_cleanly(tmp_path, capsys):
    exit_code = main(
        [
            "track",
            str(tmp_path / "missing.mp4"),
            "--point",
            "1,1@0",
            "--tracker",
            "lk",
            "--out",
            str(tmp_path / "out.json"),
            "--quiet",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


# -- window and search sizes (--window / --search) ----------------------------
#
# The same per-point settings the browser exposes, reachable headlessly. The
# flags are stated in *pixels* whatever the chosen plugin calls its own window
# argument, so these tests check the translation as much as the plumbing.


def test_window_flag_reaches_the_plugin_in_its_own_units(tmp_path, tiny_clip, monkeypatch):
    """`--window 20` is LK's `win_size=41`, not `win_size=20`."""
    from stable.trackers.registry import create_tracker as real_create

    seen: dict[str, object] = {}

    def spy(name, **kwargs):
        seen.update(kwargs.get("options") or {})
        return real_create(name, **kwargs)

    monkeypatch.setattr("stable.cli.track.create_tracker", spy)

    video_path, source = tiny_clip
    x, y = source.true_position(0)
    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@0",
            "--tracker",
            "lk",
            "--window",
            "20",
            "--out",
            str(tmp_path / "tracks.json"),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert seen == {"win_size": 41}


def test_window_and_search_reach_a_tracker_that_takes_both(tmp_path, tiny_clip, monkeypatch):
    from stable.trackers.registry import create_tracker as real_create

    seen: dict[str, object] = {}

    def spy(name, **kwargs):
        seen.update(kwargs.get("options") or {})
        return real_create(name, **kwargs)

    monkeypatch.setattr("stable.cli.track.create_tracker", spy)

    video_path, source = tiny_clip
    x, y = source.true_position(0)
    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@0",
            "--tracker",
            "template",
            "--window",
            "8",
            "--search",
            "30",
            "--out",
            str(tmp_path / "tracks.json"),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert seen == {"template_radius": 8, "search_radius": 30}


def test_a_search_flag_a_tracker_cannot_honour_is_refused_with_the_reason(
    tmp_path, tiny_clip, capsys
):
    """Silently ignoring it would mean a run that is not what was asked for."""
    video_path, source = tiny_clip
    x, y = source.true_position(0)
    out_path = tmp_path / "tracks.json"
    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@0",
            "--tracker",
            "correlation",
            "--search",
            "30",
            "--out",
            str(out_path),
            "--quiet",
        ]
    )
    assert exit_code == 1
    assert not out_path.exists()
    err = capsys.readouterr().err
    assert "--search" in err and "never searches" in err


def test_an_out_of_range_window_is_refused(tmp_path, tiny_clip, capsys):
    video_path, source = tiny_clip
    x, y = source.true_position(0)
    exit_code = main(
        [
            "track",
            video_path,
            "--point",
            f"{x},{y}@0",
            "--tracker",
            "template",
            "--window",
            "500",
            "--out",
            str(tmp_path / "tracks.json"),
            "--quiet",
        ]
    )
    assert exit_code == 1
    assert "--window" in capsys.readouterr().err
