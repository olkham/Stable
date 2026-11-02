"""HiT: its decoder, its geometry on the Siamese pipeline, its calibration and its weights.

What is and is not verified here
--------------------------------
``tests/test_tracker_suite.py`` holds HiT to the contract every tracker meets,
through :class:`~hit_backends.HiTMarkerBackend`. This module covers what is
particular to it: reading a box out of the two corner maps, the appearance
evidence and its ratio to the anchor's own, the crop sizes HiT is fed, the
point riding on HiT's box, a box following HiT's box in full, holding through a
loss, the per-variant calibrations, the three checkpoints and the CPU fallback.

None of it needs PyTorch or a checkpoint. The network itself is
``tests/test_hit_net.py``; the real-weights checks (strict loading, parity with
the authors' code, tracking quality) were run when the checkpoints were pinned
and are summarised in the :mod:`stable.trackers.hit` module docstring.
"""

from __future__ import annotations

import logging
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from conftest import anchor_at
from hit_backends import HiTMarkerBackend, corner_logits

from stable.core.errors import ContractError, TrackerError
from stable.core.types import Query
from stable.core.video import SyntheticVideoSource
from stable.trackers.hit import (
    CALIBRATIONS,
    HIT_BASE,
    HIT_MODELS,
    HIT_SMALL,
    HIT_TINY,
    HiTMaps,
    HiTTracker,
    soft_argmax,
)
from stable.trackers.registry import register_tracker
from stable.trackers.siamese import SiameseBackend, SiameseMaps
from stable.trackers.weights import registered_models

CROP = 255
"""The search crop the plugin cuts by default, in its own pixels."""


def maps_for(
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    *,
    crop: int = CROP,
    similarity: float | np.ndarray | None = 0.8,
) -> HiTMaps:
    """One point's maps with its corners at the given network pixels."""
    sim = None
    if similarity is not None:
        sim = np.broadcast_to(np.asarray(similarity, dtype=np.float32), (1, 1, 16, 16)).copy()
    return HiTMaps(
        score=corner_logits(*top_left)[None, None],
        bottom_right=corner_logits(*bottom_right)[None, None],
        similarity=sim,
        crop_size=crop,
        network_size=256,
    )


def _hide_module(monkeypatch, name: str) -> None:
    """Make ``import <name>`` fail even if it is already imported."""
    for cached in [n for n in list(sys.modules) if n == name or n.startswith(f"{name}.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setitem(sys.modules, name, None)


# -- reading the corner maps ------------------------------------------------------


def test_a_corner_is_read_by_hits_soft_argmax():
    assert soft_argmax(corner_logits(100.3, 37.9), 16.0) == pytest.approx((100.3, 37.9))
    # Flat maps (no target) put each corner at the mean of the grid, 0..240.
    assert soft_argmax(np.zeros((16, 16)), 16.0) == pytest.approx((120.0, 120.0))


def test_the_box_is_read_in_the_search_crops_pixels_about_its_centre():
    """The network's pixel-centre convention puts the crop centre at 127.5."""
    step = HiTTracker()._decode(
        maps_for((100.0, 110.0), (160.0, 150.0)), 0, (60.0, 40.0), (0.0, 0.0), math.inf, box=True
    )
    scale = CROP / 256
    assert step.dx == pytest.approx((130.0 - 127.5) * scale)
    assert step.dy == pytest.approx((130.0 - 127.5) * scale)
    assert (step.width, step.height) == pytest.approx((60.0 * scale, 40.0 * scale))


def test_a_wider_search_window_is_read_back_at_its_own_scale():
    """HiT resamples any window to 256 px; the box comes back in the window's pixels."""
    step = HiTTracker()._decode(
        maps_for((40.0, 40.0), (104.0, 104.0), crop=383), 0, (96.0, 96.0), (0, 0), math.inf
    )
    assert step.width == pytest.approx(64.0 * 383 / 256)
    assert step.dx == pytest.approx((72.0 - 127.5) * 383 / 256)


def test_the_evidence_is_the_appearance_under_the_box():
    similarity = np.full((16, 16), 0.1, dtype=np.float32)
    # The box below is centred on network pixel 127.5, cell (127.5 + 0.5) / 16 = 8.
    similarity[8, 8] = 0.7
    step = HiTTracker()._decode(
        maps_for((97.5, 97.5), (157.5, 157.5), similarity=similarity), 0, (60, 60), (0, 0), math.inf
    )
    assert step.evidence == pytest.approx(0.7)


def test_negative_appearance_is_no_evidence_at_all():
    step = HiTTracker()._decode(
        maps_for((97.5, 97.5), (157.5, 157.5), similarity=-0.4), 0, (60, 60), (0, 0), math.inf
    )
    assert step.evidence == 0.0


def test_the_evidence_is_read_against_the_anchors_own():
    plugin = HiTTracker()
    maps = maps_for((97.5, 97.5), (157.5, 157.5), similarity=0.8)
    reference = plugin._decode(maps, 0, (60, 60), (0, 0), math.inf)
    weaker = plugin._decode(
        maps_for((97.5, 97.5), (157.5, 157.5), similarity=0.6), 0, (60, 60), (0, 0), 1e9
    )
    assert plugin.raw_confidence(weaker, reference) == pytest.approx(0.75)
    assert plugin.raw_confidence(reference, weaker) > 1.0
    flat = plugin._decode(maps_for((0, 0), (1, 1), similarity=0.0), 0, (60, 60), (0, 0), 1e9)
    assert plugin.raw_confidence(reference, flat) == 0.0


def test_a_box_further_than_the_reach_is_not_the_target():
    plugin = HiTTracker()
    far = maps_for((180.0, 100.0), (240.0, 160.0))  # centre 210, 130: ~82 px right
    assert plugin._decode(far, 0, (60, 60), (0.0, 0.0), 50.0).evidence == 0.0
    assert plugin._decode(far, 0, (60, 60), (0.0, 0.0), 100.0).evidence > 0.0
    assert plugin._decode(far, 0, (60, 60), (60.0, 0.0), 50.0).evidence > 0.0


def test_a_box_follows_hits_size_in_full_and_a_point_damped_and_gated():
    plugin = HiTTracker()
    maps = maps_for((97.5, 97.5), (157.5, 157.5))  # 60 network px, ~59.8 crop px
    assert plugin._decode(maps, 0, (60, 60), (0, 0), math.inf, box=True).rate == 1.0
    assert plugin._decode(maps, 0, (60, 60), (0, 0), math.inf).rate == pytest.approx(0.3)
    # A point's "box" that doubled since the anchor is not trusted to rescale it.
    assert plugin._decode(maps, 0, (30, 30), (0, 0), math.inf).rate == 0.0
    # A drawn box is never gated: HiT's box is the thing it was trained to get right.
    assert plugin._decode(maps, 0, (30, 30), (0, 0), math.inf, box=True).rate == 1.0
    assert HiTTracker.scale_gate == math.inf


def test_it_refuses_maps_that_are_not_hits():
    maps = SiameseMaps(score=np.zeros((1, 1, 16, 16)))
    with pytest.raises(TrackerError, match="corner maps"):
        HiTTracker()._decode(maps, 0, (60, 60), (0, 0), math.inf)


def test_hit_maps_validate_their_shapes():
    with pytest.raises(ContractError, match="both corner maps"):
        HiTMaps(score=np.zeros((1, 1, 16, 16)))
    with pytest.raises(ContractError, match="bottom-right"):
        HiTMaps(score=np.zeros((1, 1, 16, 16)), bottom_right=np.zeros((1, 1, 15, 15)))


# -- HiT's geometry on the Siamese pipeline ---------------------------------------


def test_hit_is_cut_the_crops_its_backend_resamples_to_128_and_256(video):
    backend = HiTMarkerBackend()
    HiTTracker(backend=backend).track(video, [anchor_at(video, 0)], (0, 3), "fwd")
    assert backend.exemplars == [(1, 127, 127, 3)]
    assert {shape[2:] for shape in backend.searches} == {(255, 255, 3)}
    assert {shape[1] for shape in backend.searches} == {1}, "one search scale"


def test_the_default_search_is_the_trained_one():
    assert HiTTracker().search_geometry()[0] == HiTTracker.trained_instance_size == 255


def test_a_wide_search_radius_grows_a_points_search_only_as_far_as_hit_tracks(caplog):
    caplog.set_level(logging.INFO, logger="stable.trackers")
    size, reach = HiTTracker(search_radius=60).search_geometry()
    assert 255 < size <= 415 and (size - 255) % 16 == 0
    capped, wanted = HiTTracker(search_radius=200).search_geometry()
    assert capped == HiTTracker.point_max_instance_size == 415
    assert HiTTracker()._reach_of(capped) < wanted
    assert "reaches" in caplog.text


def test_reach_spans_where_hit_was_trained_to_look():
    plugin = HiTTracker()
    # Trained with the target up to 1.75 target sizes (7/16 of the search) off centre.
    assert plugin._reach_of(255) == pytest.approx(104.0)
    assert plugin._reach_of(255) <= 255 * 7 / 16


@pytest.mark.parametrize("size", [255, 319, 415, 511])
def test_a_box_search_window_is_accepted_over_the_apps_whole_range(size):
    """A point keeps its stored settings when its tracker changes; none may break HiT."""
    assert HiTTracker(box_search_size=size).box_search_size == size


@pytest.mark.parametrize("size", [254, 256, 527, 319.5])
def test_invalid_box_search_windows_are_refused(size):
    with pytest.raises(ContractError, match="box_search_size"):
        HiTTracker(box_search_size=size)


def test_a_wider_box_search_window_reaches_inference():
    video = ShapeShiftingBox()
    backend = HiTMarkerBackend()
    result = HiTTracker(backend=backend, box_search_size=319).track(
        video, [video.query(0)], (0, 4), "fwd"
    )
    assert {shape[2] for shape in backend.searches} == {319}
    assert np.isfinite(result.sizes[0, :4]).all()


def test_the_point_rides_on_hits_box_rather_than_trusting_its_centre(video):
    """HiT boxes the object it thinks a patch belongs to: the point keeps its offset."""
    backend = HiTMarkerBackend(bias=(14.0, -9.0))
    result = HiTTracker(backend=backend).track(
        video, [anchor_at(video, 0)], (0, video.frame_count), "fwd"
    )
    for frame in range(video.frame_count):
        tx, ty = video.true_position(frame)
        px, py = result.points[0, frame]
        assert math.hypot(px - tx, py - ty) < 1.5, f"frame {frame}"


def test_a_lost_point_holds_where_it_was_last_seen_and_is_found_again(occluded_video):
    anchor = 5
    query = Query(anchor, *occluded_video.true_position(anchor))
    result = HiTTracker(backend=HiTMarkerBackend()).track(
        occluded_video, [query], (0, occluded_video.frame_count), "fwd"
    )
    lo, hi = occluded_video.occlude_range
    assert np.allclose(result.points[0, lo:hi], result.points[0, lo - 1])
    assert (result.confidence[0, lo:hi] == 0.0).all()
    tx, ty = occluded_video.true_position(occluded_video.frame_count - 1)
    px, py = result.points[0, -1]
    assert math.hypot(px - tx, py - ty) < 1.5


# -- boxes ---------------------------------------------------------------------------


class ShapeShiftingBox(SyntheticVideoSource):
    """A white rectangle that drifts right while it widens and flattens to an aspect of 2.7."""

    def __init__(self, *, hidden: tuple[int, int] = (0, 0)) -> None:
        super().__init__(frame_count=30, width=200, height=140)
        self.hidden = hidden

    def true_box(self, frame: int) -> tuple[float, float, float, float]:
        """``(cx, cy, w, h)`` of the rectangle at ``frame``, in pixel-edge units."""
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
    result = HiTTracker(backend=HiTMarkerBackend()).track(video, [video.query(0)], (0, 30))
    assert result.has_boxes
    for frame in range(30):
        cx, cy, w, h = video.true_box(frame)
        (x, y), (rw, rh) = result.points[0, frame], result.sizes[0, frame]
        assert math.hypot(x - cx, y - cy) < 1.0, f"frame {frame}"
        # Followed undamped, so the rectangle's whole-pixel rendering shows: a pixel
        # is 6% of its final height.
        assert 0.88 < rw / w < 1.12 and 0.88 < rh / h < 1.12, f"frame {frame}"
    final_w, final_h = result.sizes[0, -1]
    assert final_w / final_h > 2.4  # it started square
    assert result.confidence[0].min() > 0.9


def test_the_drawn_box_is_ground_truth_on_its_anchor_frame():
    video = ShapeShiftingBox()
    query = video.query(12)
    result = HiTTracker(backend=HiTMarkerBackend()).track(video, [query], (0, 30))
    assert tuple(result.points[0, 12]) == pytest.approx(query.position)
    assert tuple(result.sizes[0, 12]) == pytest.approx(query.size)


def test_a_lost_box_holds_its_place_and_its_size():
    video = ShapeShiftingBox(hidden=(14, 18))
    result = HiTTracker(backend=HiTMarkerBackend()).track(video, [video.query(0)], (0, 30), "fwd")
    assert np.allclose(result.sizes[0, 14:18], result.sizes[0, 13])
    assert np.allclose(result.points[0, 14:18], result.points[0, 13])
    assert (result.confidence[0, 14:18] == 0.0).all()
    assert result.confidence[0, 18:].min() > 0.9, "and it is found again"


def test_a_point_tracked_beside_a_box_is_tracked_as_it_would_be_alone(video):
    point = anchor_at(video, 0)
    box = Query(0, 30.0, 20.0, w=12.0, h=8.0)
    alone = HiTTracker(backend=HiTMarkerBackend()).track(video, [point], (0, video.frame_count))
    together = HiTTracker(backend=HiTMarkerBackend()).track(
        video, [box, point], (0, video.frame_count)
    )
    assert np.array_equal(together.points[1], alone.points[0])
    assert np.isnan(together.sizes[1]).all()
    assert np.isfinite(together.sizes[0]).all()


# -- calibration -------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(HIT_MODELS))
def test_each_variant_brings_its_measured_calibration(variant):
    plugin = HiTTracker(variant=variant)
    points, boxes = CALIBRATIONS[variant]
    assert plugin.confidence_range == points
    assert plugin.box_confidence_range == boxes
    assert plugin.lost_below == points[0] and plugin.box_lost_below == boxes[0]


def test_an_explicit_calibration_wins_over_the_variants():
    plugin = HiTTracker(confidence_range=(0.6, 0.8), box_confidence_range=(0.3, 0.7))
    assert plugin.confidence_range == (0.6, 0.8)
    assert plugin.box_confidence_range == (0.3, 0.7)


def test_confidence_calibration_has_dead_zones_at_both_ends():
    plugin = HiTTracker()
    low, high = plugin.confidence_range
    confidence, visible = plugin.normalize_confidence(
        np.array([0.0, low, (low + high) / 2, high, 2])
    )
    assert confidence.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert visible.tolist() == [False, False, True, True, True]


def test_a_box_is_held_where_its_own_lower_calibration_starts():
    """A deforming object scores lower against a fixed template while still tracked."""
    plugin = HiTTracker()
    low, high = plugin.box_confidence_range
    assert low < plugin.confidence_range[0]
    confidence, _ = plugin.normalize_confidence(np.array([low, (low + high) / 2]), box=True)
    assert confidence.tolist() == pytest.approx([0.0, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variant": "huge"}, "variant"),
        ({"size_lr": 1.5}, "size_lr"),
        ({"box_size_lr": -0.1}, "box_size_lr"),
        ({"template_radius": 0}, "template_radius"),
        ({"confidence_range": (0.6, 0.6)}, "confidence_range"),
    ],
)
def test_arguments_are_validated(kwargs, message):
    with pytest.raises(ContractError, match=message):
        HiTTracker(**kwargs)


# -- weights, lifecycle and availability ----------------------------------------------------


def test_each_variant_has_its_own_pinned_commercially_usable_checkpoint():
    assert set(HIT_MODELS) == {"base", "small", "tiny"}
    keys = {spec.key for spec in HIT_MODELS.values()}
    assert keys == {"hit_base", "hit_small", "hit_tiny"}
    for spec in HIT_MODELS.values():
        assert spec.pinned and spec.license == HiTTracker.license == "MIT"
        assert spec.non_commercial is False
        assert spec.size_bytes and spec.size_bytes > 100_000_000
        assert spec.url.startswith("https://drive.usercontent.google.com/download?id=")
        assert spec in registered_models()


def test_the_checkpoints_are_the_ones_that_were_verified():
    assert HIT_BASE.sha256 == "7d6ee218014c3dae93c04fff0df5284c5fd723d6fcf79d5210e83457bf242a00"
    assert HIT_SMALL.sha256 == "7f3d834d6d55eecf4ceea3f8c8cddf268a5bed8510678b25e5fc65b96855f475"
    assert HIT_TINY.sha256 == "021a7583e1a18d96683bdb69e85b678ea4c7cac5d2522b9ec90e7fdc1f6fa90b"
    assert (HIT_BASE.size_bytes, HIT_SMALL.size_bytes, HIT_TINY.size_bytes) == (
        511212333,
        137177837,
        118102065,
    )


def test_the_variant_chooses_the_checkpoint_and_base_is_the_default():
    assert HiTTracker.model_spec is HIT_BASE
    assert HiTTracker().variant == "base" and HiTTracker().model_spec is HIT_BASE
    assert HiTTracker(variant="tiny").model_spec is HIT_TINY


def test_the_package_advertises_the_plugin():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    entry_points = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["entry-points"]
    assert entry_points["stable.trackers"]["hit"] == "stable.trackers.hit:HiTTracker"


def test_it_tracks_boxes_and_never_requires_a_gpu():
    assert HiTTracker.tracks_boxes is True
    assert HiTTracker.requires_gpu is False
    assert isinstance(HiTMarkerBackend(), SiameseBackend)


def test_without_torch_it_is_unavailable_and_says_why(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert HiTTracker.is_available() is False
    assert "PyTorch" in (HiTTracker.unavailable_reason() or "")


def test_without_opencv_it_asks_for_the_video_extra_not_for_model_code(monkeypatch):
    """The network code ships with Stable; OpenCV is what is missing."""
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: object())
    _hide_module(monkeypatch, "cv2")
    reason = HiTTracker.unavailable_reason() or ""
    assert "OpenCV" in reason and "stable-tracker[video]" in reason


def test_building_the_backend_without_torch_says_so(monkeypatch):
    monkeypatch.setattr("stable.trackers.hit.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        HiTTracker()._build_backend("cpu", "weights.pth.tar")


def test_out_of_memory_mid_run_continues_on_the_cpu_with_rebuilt_templates(monkeypatch, video):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    backend = HiTMarkerBackend(oom_on=(3,))
    backend.device = "cuda"
    plugin = HiTTracker(backend=backend)
    plugin.prepare("cuda")

    result = plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")

    assert plugin.device == "cpu" and plugin.fell_back_to_cpu is True
    assert backend.embeds == 2, "the templates must be rebuilt on the new device"
    assert result.confidence[0, 1:].min() > 0.9


def test_backend_bugs_surface_as_tracker_errors(video):
    backend = HiTMarkerBackend(fail_on=(2,))
    with pytest.raises(TrackerError, match="inference failed"):
        HiTTracker(backend=backend).track(video, [anchor_at(video, 0)], (0, 10), "fwd")


# -- the operator's settings ------------------------------------------------------------------


@pytest.fixture
def registered():
    """HiT in the process-wide registry, whether or not the install has published it."""
    register_tracker(HiTTracker)


def test_points_offer_the_pattern_and_search_and_boxes_the_search_window(registered):
    from stable.server.tracker_options import box_search_scale, tracker_settings, tracker_window

    points = {row["attribute"]: row for row in tracker_settings("hit")}
    assert set(points) == {"template_radius", "search_radius"}
    assert tracker_window("hit") == (16, "template_radius")
    boxes = [row["attribute"] for row in tracker_settings("hit", box=True)]
    assert boxes == ["box_search_size"]
    # The window the viewer draws is the one HiT is really given.
    assert box_search_scale("hit") == pytest.approx(255 / 127)


def test_the_settings_reach_the_plugin_through_the_registry(registered):
    from stable.trackers.registry import create_tracker

    plugin = create_tracker(
        "hit",
        prepare=False,
        options={"template_radius": 24, "search_radius": 64, "box_search_size": 511, "win_size": 9},
    )
    assert (plugin.template_radius, plugin.search_radius, plugin.box_search_size) == (24, 64, 511)
