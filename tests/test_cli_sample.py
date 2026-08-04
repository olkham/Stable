"""Tests for the ``stable sample`` CLI subcommand.

Runs the real bundled clip (:mod:`stable.examples`) through the real
``track`` -> ``solve`` -> ``stabilize`` pipeline via :func:`stable.cli.main.main`
— this is also, in effect, the "headless integration on a bundled ~5s test
clip" PLAN.md's Phase 6 acceptance criterion asks for.
"""

from __future__ import annotations

import csv

import pytest

from stable.cli.main import main
from stable.examples import sample_metadata
from stable.video.source import PyAVVideoSource

pytestmark = pytest.mark.integration


def test_sample_writes_a_complete_project(tmp_path):
    out_dir = tmp_path / "sample-project"

    exit_code = main(["sample", str(out_dir), "--quiet"])

    assert exit_code == 0
    assert (out_dir / "clip.mp4").is_file()
    assert (out_dir / "tracks.json").is_file()
    assert (out_dir / "curves.csv").is_file()
    assert (out_dir / "stabilized.mp4").is_file()


def test_sample_clip_matches_its_bundled_metadata(tmp_path):
    out_dir = tmp_path / "sample-project"
    main(["sample", str(out_dir), "--quiet"])
    meta = sample_metadata()

    video = PyAVVideoSource(out_dir / "clip.mp4")
    try:
        assert video.frame_count == meta["frames"]
        assert video.width == meta["width"]
        assert video.height == meta["height"]
    finally:
        video.close()


def test_sample_tracks_all_three_points_through_the_whole_clip(tmp_path):
    out_dir = tmp_path / "sample-project"
    main(["sample", str(out_dir), "--quiet"])

    import json

    data = json.loads((out_dir / "tracks.json").read_text("utf-8"))
    assert len(data["points"]) == 3
    for entry in data["points"]:
        assert all(cell is not None for cell in entry["points"])


def test_sample_produces_a_genuine_xyrs_solve_with_small_residuals(tmp_path):
    out_dir = tmp_path / "sample-project"
    main(["sample", str(out_dir), "--quiet"])

    with open(out_dir / "curves.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 120  # matches the bundled clip's frame count
    assert all(row["status"] == "solved" for row in rows)
    assert all(int(row["n_inliers"]) >= 3 for row in rows)
    residuals = [float(row["residual_px"]) for row in rows]
    assert max(residuals) < 1.0  # a clean synthetic clip should solve almost exactly


def test_sample_stabilized_video_decodes_and_matches_frame_count(tmp_path):
    out_dir = tmp_path / "sample-project"
    main(["sample", str(out_dir), "--quiet"])

    video = PyAVVideoSource(out_dir / "stabilized.mp4")
    try:
        assert video.frame_count == sample_metadata()["frames"]
    finally:
        video.close()


def test_sample_respects_the_tracker_and_border_options(tmp_path):
    out_dir = tmp_path / "sample-project"
    exit_code = main(
        ["sample", str(out_dir), "--tracker", "template", "--border", "crop", "--quiet"]
    )
    assert exit_code == 0
    assert (out_dir / "stabilized.mp4").is_file()


def test_sample_defaults_to_a_stable_sample_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exit_code = main(["sample", "--quiet"])
    assert exit_code == 0
    assert (tmp_path / "stable-sample" / "stabilized.mp4").is_file()


def test_sample_prints_a_summary_unless_quiet(tmp_path, capsys):
    out_dir = tmp_path / "sample-project"
    main(["sample", str(out_dir)])
    out = capsys.readouterr().out
    assert "sample project written" in out
