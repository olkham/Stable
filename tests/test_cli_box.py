"""``stable track --box``: parsing, the corner-to-centre conversion, and sizes in the output.

The run itself uses a small fake plugin that declares ``tracks_boxes`` rather
than a deep tracker, so nothing here needs torch or weights.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from stable.cli.main import build_parser, main
from stable.cli.track import parse_box_spec, track_result_from_json_dict
from stable.core.types import TrackResult
from stable.trackers.dummy import DummyTracker
from stable.trackers.registry import register_tracker, registry


class FakeBoxTracker(DummyTracker):
    """The dummy tracker, declared box-capable, reporting each query's box as its size."""

    name = "fakebox"
    tracks_boxes = True

    def track(self, video, queries, frame_range, direction="bidir", progress=lambda *_: None):
        """Track like the dummy, then report each query's box on the frames it covered."""
        result = super().track(video, queries, frame_range, direction, progress)
        sizes = np.full((result.n_points, result.n_frames, 2), np.nan, dtype=np.float32)
        for index, query in enumerate(queries):
            if query.size is not None:
                sizes[index, np.isfinite(result.points[index]).all(axis=1)] = query.size
        return TrackResult(
            points=result.points,
            confidence=result.confidence,
            visible=result.visible,
            sizes=sizes,
        )


@pytest.fixture
def fake_box_tracker():
    """Register the fake box plugin for one test."""
    registry.discover()
    register_tracker(FakeBoxTracker, replace=True)
    yield
    registry.unregister("fakebox")


def test_box_spec_is_top_left_in_and_centre_out():
    query = parse_box_spec("100,60,40,30@7")

    assert (query.frame, query.x, query.y) == (7, 120.0, 75.0)
    assert query.size == (40.0, 30.0)


@pytest.mark.parametrize(
    "bad", ["100,60,40@0", "100,60,40,30", "a,b,c,d@0", "1,2,0,5@0", "1,2,3,-4@0", "1,2,3,4@-1"]
)
def test_malformed_boxes_are_rejected(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_box_spec(bad)


def test_box_and_point_options_share_one_ordered_list():
    args = build_parser().parse_args(
        ["track", "clip.mp4", "--point", "1,2@0", "--box", "10,10,4,6@3", "--out", "o.json"]
    )

    assert [q.has_box for q in args.points] == [False, True]
    assert args.points[1].position == (12.0, 13.0)


def test_neither_point_nor_box_is_a_usage_error(tiny_clip, tmp_path):
    path, _source = tiny_clip
    with pytest.raises(SystemExit) as raised:
        main(["track", path, "--out", str(tmp_path / "o.json")])
    assert raised.value.code == 2


def test_a_box_run_writes_sizes_to_json_and_they_round_trip(tmp_path, tiny_clip, fake_box_tracker):
    path, _source = tiny_clip
    out = tmp_path / "boxes.json"

    code = main(
        [
            "track",
            path,
            "--box",
            "20,10,14,10@3",
            "--tracker",
            "fakebox",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    entry = data["points"][0]
    assert entry["query"] == {"frame": 3, "x": 27.0, "y": 15.0, "w": 14.0, "h": 10.0}
    assert entry["sizes"][3] == [14.0, 10.0]
    restored = track_result_from_json_dict(data)
    assert restored.sizes is not None
    assert restored.sizes[0, 3].tolist() == [14.0, 10.0]


def test_a_box_run_writes_sizes_to_npz(tmp_path, tiny_clip, fake_box_tracker):
    path, _source = tiny_clip
    out = tmp_path / "boxes.npz"

    code = main(
        [
            "track",
            path,
            "--box",
            "20,10,14,10@3",
            "--tracker",
            "fakebox",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    assert code == 0
    assert TrackResult.load_npz(out).has_boxes


def test_a_point_only_run_has_no_sizes_in_its_json(tmp_path, tiny_clip):
    path, source = tiny_clip
    x, y = source.true_position(3)
    out = tmp_path / "points.json"

    code = main(
        ["track", path, "--point", f"{x},{y}@3", "--tracker", "lk", "--out", str(out), "--quiet"]
    )

    assert code == 0
    assert "sizes" not in json.loads(out.read_text(encoding="utf-8"))["points"][0]


def test_a_box_with_a_tracker_that_cannot_follow_boxes_is_an_error(
    tmp_path, tiny_clip, capsys, fake_box_tracker
):
    path, _source = tiny_clip
    out = tmp_path / "o.json"

    code = main(
        ["track", path, "--box", "20,10,14,10@3", "--tracker", "lk", "--out", str(out), "--quiet"]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "does not track boxes" in err
    assert "fakebox" in err
    assert not out.exists()
