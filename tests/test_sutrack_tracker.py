"""SUTrack as a Stable plugin: decoding, geometry, boxes, confidence, weights and wiring.

What is and is not verified here
--------------------------------
``tests/test_tracker_suite.py`` holds SUTrack to the contract every tracker
meets, through :class:`_sutrack_stub.SUTrackMarkerBackend`, which finds the
synthetic clip's marker in the crops the plugin cuts and answers in SUTrack's
map layout. This module covers what is particular to SUTrack: its decoder, its
crop geometry, the template boxes its encoder needs, box tracking, holding
through a loss, its calibration, its checkpoint and how it is offered. None of
it needs PyTorch or weights.

The network itself is ``tests/test_sutrack_net.py`` (needs PyTorch). A real
inference run is not part of the suite — the checkpoint is a 1 GB download —
but was done when the checkpoint was pinned: the port matches the authors' code
to within 2.4e-7 on the CPU, and the measurements behind every default here are
in :mod:`stable.trackers.sutrack`'s docstring.
"""

from __future__ import annotations

import math
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from _sutrack_stub import SUTrackMarkerBackend
from conftest import anchor_at

from stable.core.errors import ContractError, TrackerError
from stable.core.types import Query
from stable.core.video import SyntheticVideoSource
from stable.trackers.registry import (
    ENTRY_POINT_GROUP,
    TrackerRegistry,
    register_tracker,
)
from stable.trackers.siamese import (
    EXEMPLAR_SIZE,
    SiameseBackend,
    SiameseMaps,
    SiameseTrackerPlugin,
)
from stable.trackers.sutrack import (
    NETWORK_SCALE,
    PATCH,
    SUTRACK_T224_WEIGHTS,
    SUTrackMaps,
    SUTrackTracker,
    network_scale,
    template_box,
)
from stable.trackers.weights import registered_models

ROOT = Path(__file__).resolve().parents[1]


def plugin(**kwargs):
    backend = kwargs.pop("backend", None) or SUTrackMarkerBackend()
    return SUTrackTracker(backend=backend, **kwargs)


def _hide_module(monkeypatch, name: str) -> None:
    """Make ``import <name>`` fail even if it is already imported."""
    for cached in [n for n in list(sys.modules) if n == name or n.startswith(f"{name}.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setitem(sys.modules, name, None)


# -- geometry -----------------------------------------------------------------


def test_the_crops_are_resampled_to_sutracks_template_and_search_sizes():
    """112 / 127: twice the target box for the template, four times for the search."""
    assert NETWORK_SCALE == pytest.approx(112 / 127)
    assert network_scale(255) == pytest.approx(NETWORK_SCALE)
    assert 224 / network_scale(255) == pytest.approx(254.0)


def test_a_wider_search_is_shrunk_onto_the_trained_input():
    assert network_scale(511) * 511 == pytest.approx(network_scale(255) * 255)
    assert SUTrackTracker().score_size(511) == SUTrackTracker().score_size(255) == 14


def test_the_template_box_is_normalised_the_way_upstream_does_it():
    """transform_image_to_crop: the centre on pixel (side - 1) / 2, divided by side - 1."""
    box = template_box(56.0, 28.0)
    assert box == pytest.approx([(55.5 - 28.0) / 111, (55.5 - 14.0) / 111, 56 / 111, 28 / 111])


def test_the_default_search_is_the_trained_one():
    size, reach = SUTrackTracker().search_geometry()
    assert size == 255
    assert reach == pytest.approx(48 * EXEMPLAR_SIZE / 66.0)
    assert SUTrackTracker()._reach_of(255) == pytest.approx(6.5 * PATCH / NETWORK_SCALE)


def test_a_wider_search_radius_widens_the_search_until_it_reaches():
    tracker = SUTrackTracker(search_radius=100)
    size, reach = tracker.search_geometry()
    assert size > 255
    assert tracker._reach_of(size) >= reach
    assert tracker._reach_of(size - 16) < reach


def test_its_scale_gate_is_wider_than_the_base_classes():
    """A box held through an occlusion must be able to take the size it reappears at."""
    assert SUTrackTracker.scale_gate == 2.0 > SiameseTrackerPlugin.scale_gate


# -- the decoder -----------------------------------------------------------------


def _maps(score, boxes=None, *, stride=PATCH / NETWORK_SCALE, half=30.0):
    score = np.asarray(score, dtype=np.float32).reshape(1, 1, 14, 14)
    if boxes is None:
        boxes = np.full((1, 1, 4, 14, 14), half, dtype=np.float32)
    return SUTrackMaps(score=score, boxes=boxes, stride=stride)


def _grid(stride=PATCH / NETWORK_SCALE):
    return (np.arange(14) - 6.5) * stride


def test_the_box_is_read_from_the_peak_cell():
    score = np.full((14, 14), 0.05)
    score[5, 9] = 0.9
    boxes = np.zeros((1, 1, 4, 14, 14), dtype=np.float32)
    boxes[0, 0] = np.array([10.0, 20.0, 30.0, 40.0])[:, None, None]  # centre (+10, +10) off
    step = SUTrackTracker()._decode(_maps(score, boxes), 0, (60, 60), (0.0, 0.0), math.inf)
    grid = _grid()
    assert step.dx == pytest.approx(grid[9] + 10.0)
    assert step.dy == pytest.approx(grid[5] + 10.0)
    assert (step.width, step.height) == pytest.approx((40.0, 60.0))
    assert step.evidence == pytest.approx(0.9, abs=1e-6)


def test_the_hann_window_multiplies_the_score_as_upstream_applies_it():
    """A slightly stronger peak far out loses to one beside where the target is expected."""
    score = np.full((14, 14), 0.05)
    score[6, 7] = 0.80
    score[6, 13] = 0.85
    step = SUTrackTracker()._decode(_maps(score), 0, (60, 60), (0.0, 0.0), math.inf)
    assert step.dx == pytest.approx(_grid()[7])
    assert step.evidence == pytest.approx(0.80, abs=1e-6), "evidence is the unwindowed score"


def test_the_window_follows_where_the_target_is_expected():
    score = np.full((14, 14), 0.05)
    score[6, 7] = 0.80
    score[6, 12] = 0.80
    expected = (float(_grid()[12]), 0.0)
    step = SUTrackTracker()._decode(_maps(score), 0, (60, 60), expected, math.inf)
    assert step.dx == pytest.approx(_grid()[12])


def test_a_cell_beyond_the_reach_is_never_chosen():
    score = np.full((14, 14), 0.05)
    score[6, 12] = 0.99
    score[6, 7] = 0.10
    step = SUTrackTracker()._decode(_maps(score), 0, (60, 60), (0.0, 0.0), 30.0)
    assert abs(step.dx) <= 30.0 + PATCH / NETWORK_SCALE


def test_the_cell_size_comes_from_the_maps():
    """A wider search's cells are further apart in crop pixels; the backend says how far."""
    score = np.full((14, 14), 0.05)
    score[6, 10] = 0.9
    step = SUTrackTracker()._decode(_maps(score, stride=30.0), 0, (60, 60), (0.0, 0.0), math.inf)
    assert step.dx == pytest.approx(_grid(30.0)[10])
    plain = SiameseMaps(score=score.reshape(1, 1, 14, 14), boxes=np.full((1, 1, 4, 14, 14), 30.0))
    step = SUTrackTracker()._decode(plain, 0, (60, 60), (0.0, 0.0), math.inf)
    assert step.dx == pytest.approx(_grid()[10])


def test_a_box_takes_the_regressed_size_and_a_point_damps_it():
    score = np.full((14, 14), 0.5)
    tracker = SUTrackTracker(size_lr=0.25, box_size_lr=1.0)
    as_box = tracker._decode(_maps(score), 0, (60, 60), (0.0, 0.0), math.inf, box=True)
    as_point = tracker._decode(_maps(score), 0, (60, 60), (0.0, 0.0), math.inf)
    assert (as_box.rate, as_point.rate) == (1.0, 0.25)


def test_maps_without_boxes_are_refused():
    maps = SiameseMaps(score=np.zeros((1, 1, 14, 14)))
    with pytest.raises(TrackerError, match="box regression"):
        SUTrackTracker()._decode(maps, 0, (60, 60), (0.0, 0.0), math.inf)


# -- tracking through the plugin ---------------------------------------------------


def test_a_point_rides_on_the_box_rather_than_trusting_its_centre(video):
    """A network boxing something 9 crop pixels beside the patch must not move the point."""
    query = anchor_at(video, 0)
    result = plugin(backend=SUTrackMarkerBackend(bias=(9.0, -5.0))).track(
        video, [query], (0, video.frame_count), "fwd"
    )
    for frame in range(video.frame_count):
        tx, ty = video.true_position(frame)
        assert math.hypot(*(result.points[0, frame] - (tx, ty))) < 1.5, f"frame {frame}"


def test_the_encoder_is_told_each_templates_target_box(video):
    """A point's target is the centred square half the exemplar; a box keeps its shape."""
    backend = SUTrackMarkerBackend()
    plugin(backend=backend).track(
        video, [anchor_at(video, 0), Query(0, 30.0, 20.0, w=24.0, h=12.0)], (0, 4), "fwd"
    )
    sizes = backend.template_sizes[0]
    assert sizes[0] == pytest.approx((EXEMPLAR_SIZE / 2.0, EXEMPLAR_SIZE / 2.0))
    assert sizes[1, 0] / sizes[1, 1] == pytest.approx(2.0)
    assert len(backend.template_sizes) == backend.calls, "announced before every match"


def test_points_sharing_an_anchor_frame_share_each_network_call(video):
    backend = SUTrackMarkerBackend()
    plugin(backend=backend).track(
        video, [anchor_at(video, 0), Query(0, 20.0, 20.0), Query(0, 40.0, 30.0)], (0, 6), "fwd"
    )
    assert backend.calls == 6, "one anchor reading and one call per later frame"
    assert all(shape[0] == 3 for shape in backend.searches)


class ShapeShiftingBox(SyntheticVideoSource):
    """A white rectangle that drifts right while it widens and flattens."""

    def __init__(self, *, hidden: tuple[int, int] = (0, 0)) -> None:
        super().__init__(frame_count=30, width=200, height=140)
        self.hidden = hidden

    def true_box(self, frame: int) -> tuple[float, float, float, float]:
        return (70.0 + 1.5 * frame, 70.0, 24.0 + 0.6 * frame, 24.0 - 0.3 * frame)

    def get_frame(self, index: int) -> np.ndarray:
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        base = ((xs + ys) % 256).astype(np.uint8)
        frame = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 13, axis=0)], axis=-1)
        if self.hidden[0] <= index < self.hidden[1]:
            return frame
        cx, cy, w, h = self.true_box(index)
        inside = (np.abs(xs - cx) <= w / 2.0 - 0.5) & (np.abs(ys - cy) <= h / 2.0 - 0.5)
        frame[inside] = 255
        return frame

    def query(self, frame: int) -> Query:
        cx, cy, w, h = self.true_box(frame)
        return Query(frame, cx, cy, w=w, h=h)


def test_a_box_follows_an_object_that_changes_size_and_shape():
    video = ShapeShiftingBox()
    result = plugin(backend=SUTrackMarkerBackend(measure_box=True)).track(
        video, [video.query(0)], (0, 30)
    )
    assert result.has_boxes
    ratios = []
    for frame in range(30):
        cx, cy, w, h = video.true_box(frame)
        (x, y), (rw, rh) = result.points[0, frame], result.sizes[0, frame]
        assert math.hypot(x - cx, y - cy) < 1.5, f"frame {frame}"
        # The regressed size is taken as it is, so the stub's thresholded
        # measurement of a resampled rectangle comes through unsmoothed.
        assert 0.88 < rw / w < 1.12 and 0.88 < rh / h < 1.12, f"frame {frame}"
        ratios.append((rw / w, rh / h))
    assert np.mean(ratios, axis=0) == pytest.approx((1.0, 1.0), abs=0.03)
    assert result.sizes[0, -1, 0] / result.sizes[0, -1, 1] > 2.0  # it started square


def test_the_drawn_box_is_ground_truth_on_its_anchor_frame():
    video = ShapeShiftingBox()
    query = video.query(12)
    result = plugin(backend=SUTrackMarkerBackend(measure_box=True)).track(video, [query], (0, 30))
    assert tuple(result.points[0, 12]) == pytest.approx(query.position)
    assert tuple(result.sizes[0, 12]) == pytest.approx(query.size)


def test_a_lost_box_holds_its_place_and_its_size():
    video = ShapeShiftingBox(hidden=(14, 18))
    result = plugin(backend=SUTrackMarkerBackend(measure_box=True)).track(
        video, [video.query(0)], (0, 30), "fwd"
    )
    assert np.allclose(result.sizes[0, 14:18], result.sizes[0, 13])
    assert np.allclose(result.points[0, 14:18], result.points[0, 13])
    assert (result.confidence[0, 14:18] == 0.0).all()
    cx, cy, w, _ = video.true_box(29)
    assert math.hypot(*(result.points[0, 29] - (cx, cy))) < 1.5, "found again afterwards"
    assert result.sizes[0, 29, 0] / w == pytest.approx(1.0, abs=0.08)


def test_a_wider_box_search_window_reaches_inference():
    video = ShapeShiftingBox()
    backend = SUTrackMarkerBackend(measure_box=True)
    result = plugin(backend=backend, box_search_size=351).track(
        video, [video.query(0)], (0, 6), "fwd"
    )
    assert {shape[2] for shape in backend.searches} == {351}
    cx, cy, *_ = video.true_box(5)
    assert math.hypot(*(result.points[0, 5] - (cx, cy))) < 1.5


def test_a_lost_point_holds_and_is_found_again(occluded_video):
    query = Query(5, *occluded_video.true_position(5))
    result = plugin().track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")
    lo, hi = occluded_video.occlude_range
    assert np.allclose(result.points[0, lo:hi], result.points[0, lo - 1])
    assert (result.confidence[0, lo:hi] == 0.0).all()
    tx, ty = occluded_video.true_position(hi + 2)
    assert math.hypot(*(result.points[0, hi + 2] - (tx, ty))) < 1.5


def test_out_of_memory_mid_run_continues_on_the_cpu_with_rebuilt_templates(monkeypatch, video):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    backend = SUTrackMarkerBackend(oom_on=(3,))
    backend.device = "cuda"
    tracker = plugin(backend=backend)
    tracker.prepare("cuda")
    result = tracker.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")
    assert tracker.device == "cpu" and tracker.fell_back_to_cpu is True
    assert backend.embeds == 2, "the templates must be rebuilt on the new device"
    assert result.confidence[0, 1:].min() > 0.9


def test_backend_bugs_surface_as_tracker_errors(video):
    with pytest.raises(TrackerError, match="inference failed"):
        plugin(backend=SUTrackMarkerBackend(fail_on=(2,))).track(
            video, [anchor_at(video, 0)], (0, 10), "fwd"
        )


def test_a_backend_without_template_boxes_still_tracks(video):
    """Any SiameseBackend will do; telling it the template boxes is optional."""

    class Plain(SUTrackMarkerBackend):
        use_template_boxes = None  # type: ignore[assignment]

    result = plugin(backend=Plain()).track(video, [anchor_at(video, 0)], (0, 8), "fwd")
    tx, ty = video.true_position(7)
    assert math.hypot(*(result.points[0, 7] - (tx, ty))) < 1.5


# -- confidence ----------------------------------------------------------------------


def test_confidence_calibration_has_dead_zones_at_both_ends():
    tracker = SUTrackTracker()
    low, high = tracker.confidence_range
    confidence, visible = tracker.normalize_confidence(
        np.array([0.0, low, (low + high) / 2, high, 1.0])
    )
    assert confidence.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert visible.tolist() == [False, False, True, True, True]


def test_the_calibration_is_the_measured_one():
    """See the module docstring: covered steps topped out at 0.49, tracked ones sat near 0.9."""
    tracker = SUTrackTracker()
    assert tracker.confidence_range == (0.45, 0.72)
    assert tracker.lost_below == 0.45
    assert tracker.box_confidence_range == (0.35, 0.65)


def test_a_box_is_held_where_its_own_lower_calibration_starts():
    tracker = SUTrackTracker()
    low, high = tracker.box_confidence_range
    assert low < tracker.confidence_range[0]
    assert tracker.box_lost_below == low
    confidence, _ = tracker.normalize_confidence(np.array([low, (low + high) / 2]), box=True)
    assert confidence.tolist() == pytest.approx([0.0, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"template_radius": 0}, "template_radius"),
        ({"search_radius": -1}, "search_radius"),
        ({"confidence_range": (0.6, 0.6)}, "confidence_range"),
        ({"box_confidence_range": (0.6, 0.5)}, "box_confidence_range"),
        ({"box_search_size": 254}, "box_search_size"),
    ],
)
def test_arguments_are_validated(kwargs, message):
    with pytest.raises(ContractError, match=message):
        SUTrackTracker(**kwargs)


# -- weights, availability and how it is offered ------------------------------------------


def test_it_is_a_box_tracking_siamese_plugin_that_never_requires_a_gpu():
    assert issubclass(SUTrackTracker, SiameseTrackerPlugin)
    assert isinstance(SUTrackMarkerBackend(), SiameseBackend)
    assert SUTrackTracker.name == "sutrack"
    assert SUTrackTracker.tracks_boxes is True
    assert SUTrackTracker.requires_gpu is False


def test_the_checkpoint_is_pinned_mit_and_commercially_usable():
    spec = SUTrackTracker.model_spec
    assert spec is SUTRACK_T224_WEIGHTS
    assert spec.pinned
    assert spec.sha256 == "5815a447480fba64d67cfa6cb011f23bb5b89465685043292464ad283f91210d"
    assert spec.size_bytes == 1_044_947_583
    assert spec.license == SUTrackTracker.license == "MIT"
    assert spec.non_commercial is False
    assert spec in registered_models()


def test_the_checkpoint_comes_from_a_fixed_revision_of_the_authors_release():
    url = SUTRACK_T224_WEIGHTS.url
    assert url.startswith("https://huggingface.co/xche32/SUTrack/resolve/")
    assert "/resolve/main/" not in url, "a moving branch could change under a pinned digest"
    assert url.endswith("/sutrack_t224/SUTRACK_ep0180.pth.tar")


def test_the_package_declares_its_entry_point():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    entry = declared["project"]["entry-points"][ENTRY_POINT_GROUP]["sutrack"]
    assert entry == "stable.trackers.sutrack:SUTrackTracker"
    registry = TrackerRegistry()
    names = registry.discover(
        entry_points=[metadata.EntryPoint("sutrack", entry, ENTRY_POINT_GROUP)], strict=True
    )
    assert "sutrack" in names
    row = {r["name"]: r for r in registry.describe()}["sutrack"]
    assert row["tracks_boxes"] is True
    assert row["license"] == "MIT"


def test_without_torch_it_is_unavailable_and_says_why(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert SUTrackTracker.is_available() is False
    assert "PyTorch" in (SUTrackTracker.unavailable_reason() or "")


def test_without_opencv_it_asks_for_the_video_extra_not_for_model_code(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: object())
    _hide_module(monkeypatch, "cv2")
    reason = SUTrackTracker.unavailable_reason() or ""
    assert "OpenCV" in reason and "stable-tracker[video]" in reason


def test_building_the_backend_without_torch_says_so(monkeypatch):
    monkeypatch.setattr("stable.trackers.sutrack.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        SUTrackTracker()._build_backend("cpu", "weights.pth")


def test_the_pattern_and_search_sizes_are_offered_per_point_and_per_box():
    from stable.server.tracker_options import box_search_scale, tracker_settings, tracker_window

    register_tracker(SUTrackTracker)  # this checkout's, whatever was last pip-installed
    rows = {row["attribute"]: row for row in tracker_settings("sutrack")}
    assert rows["template_radius"]["value"] == 16
    assert rows["search_radius"]["value"] == 48
    assert "box_search_size" not in rows
    boxes = {row["attribute"]: row for row in tracker_settings("sutrack", box=True)}
    assert boxes["box_search_size"]["value"] == 255
    assert tracker_window("sutrack") == (16, "template_radius")
    # The server draws a box's search window as this multiple of its context
    # square; SUTrack's search really covers that much (254 of the 255 px crop).
    assert box_search_scale("sutrack") == pytest.approx(255 / 127)
