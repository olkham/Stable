"""Ocean and SiamDW: the Siamese pipeline, both decoders, and the ported networks.

What is and is not verified here
--------------------------------
`tests/test_tracker_suite.py` holds both plugins to the contract every tracker
meets, through :class:`~conftest.SiameseMarkerBackend`. This module covers what
is particular to them: the crop geometry, the anchor's own reading and the point
riding on it, the scale bookkeeping and its gate, holding through a loss, the
two decoders, the CPU fallback (which has to rebuild the templates), and — where
PyTorch is installed — the ported networks' shapes and parameter layout.

The ported networks themselves are ``tests/test_siamese_nets.py``, which needs
PyTorch. What is not covered anywhere is a real inference run: the checkpoints
are 10 MB and 104 MB downloads. Both were run for real when they were pinned
(see CONTRIBUTING.md, "Pinning a checkpoint"), and the networks were checked
against the authors' reference implementation to within 1e-5.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from conftest import SiameseMarkerBackend, anchor_at

from stable.core.errors import ContractError, TrackerError
from stable.core.types import Query
from stable.core.video import SyntheticVideoSource
from stable.server.tracker_options import tracker_search, tracker_settings, tracker_window
from stable.trackers.deep import ModelTrackerPlugin
from stable.trackers.ocean import OCEAN_OFFLINE, OceanTracker
from stable.trackers.siamdw import SIAMDW_RES22W, SiamDWTracker
from stable.trackers.siamese import (
    SiameseBackend,
    SiameseMaps,
    SiameseStep,
    SiameseTrackerPlugin,
    _Geometry,
    _Target,
    context_side,
    cosine_window,
    crop_square,
    sample_map,
)
from stable.trackers.weights import registered_models

PLUGINS = [OceanTracker, SiamDWTracker]


def plugin_for(cls, **kwargs):
    kind = "ocean" if cls is OceanTracker else "siamdw"
    backend = kwargs.pop("backend", None) or SiameseMarkerBackend(kind)
    return cls(backend=backend, **kwargs)


def _hide_module(monkeypatch, name: str) -> None:
    """Make ``import <name>`` fail even if it is already imported."""
    for cached in [n for n in list(sys.modules) if n == name or n.startswith(f"{name}.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setitem(sys.modules, name, None)


# -- geometry -----------------------------------------------------------------


def test_a_square_box_fills_half_of_its_exemplar():
    assert context_side(33, 33) == pytest.approx(66.0)


def test_the_exemplar_keeps_its_area_whatever_the_aspect():
    """SiamFC's rule pads by the same amount on both sides, then takes the mean."""
    wide = context_side(60, 20)
    assert wide == pytest.approx(math.sqrt((60 + 40) * (20 + 40)))


def test_crops_are_centred_to_the_sub_pixel():
    """The crop's centre pixel samples exactly the requested centre, unrounded."""
    ys, xs = np.mgrid[0:64, 0:64]
    frame = np.stack([xs * 3, ys * 3, np.zeros_like(xs)], axis=-1).astype(np.uint8)
    crop = crop_square(frame, (20.5, 30.25), 15.0, 15, np.zeros(3))
    # A linear ramp is resampled exactly by bilinear interpolation.
    assert crop[7, 7, 0] == pytest.approx(20.5 * 3, abs=1.0)
    assert crop[7, 7, 1] == pytest.approx(30.25 * 3, abs=1.0)
    # One output pixel per source pixel at side == size.
    assert crop[7, 8, 0] - crop[7, 7, 0] == pytest.approx(3, abs=1)


def test_crops_leaving_the_frame_are_padded_with_the_fill_colour():
    frame = np.full((20, 20, 3), 200, dtype=np.uint8)
    crop = crop_square(frame, (0.0, 0.0), 40.0, 40, np.array([10.0, 20.0, 30.0]))
    assert crop[0, 0].tolist() == [10, 20, 30]
    # Output pixel 38 samples source (18.5, 18.5): all four neighbours are inside.
    assert crop[-2, -2].tolist() == [200, 200, 200]


def test_a_centred_cosine_window_is_hanning():
    positions = (np.arange(25) - 12) * 8.0
    assert cosine_window(positions, 0.0, 12 * 8.0) == pytest.approx(np.hanning(25))


def test_the_window_follows_where_the_target_is_expected():
    positions = (np.arange(25) - 12) * 8.0
    window = cosine_window(positions, 16.0, 12 * 8.0)
    assert int(np.argmax(window)) == 14


def test_maps_are_sampled_bilinearly_and_clamped():
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert sample_map(values, 0.5, 0.5) == pytest.approx(1.5)
    assert sample_map(values, -4.0, 9.0) == pytest.approx(2.0)


def test_score_map_sizes_match_the_networks():
    assert OceanTracker().score_size(255) == 25
    assert OceanTracker().score_size(287) == 29
    assert SiamDWTracker().score_size(255) == 17
    assert SiamDWTracker().score_size(287) == 21


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_default_search_is_the_trained_one(cls):
    size, reach = cls().search_geometry()
    assert size == 255
    assert reach > 0


@pytest.mark.parametrize("cls", PLUGINS)
def test_a_wider_search_radius_grows_the_search_and_keeps_a_centre_cell(cls):
    plugin = cls(search_radius=80)
    size, reach = plugin.search_geometry()
    assert size > 255
    assert plugin.score_size(size) % 2 == 1
    assert (plugin.score_size(size) - 1) / 2 * plugin.total_stride >= reach


@pytest.mark.parametrize("cls", PLUGINS)
def test_an_unreachable_search_radius_is_capped_rather_than_unbounded(cls):
    size, _ = cls(template_radius=2, search_radius=256).search_geometry()
    assert size == cls.max_instance_size


# -- riding, scale and holding -----------------------------------------------------


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_point_rides_on_the_network_rather_than_trusting_its_centre(cls):
    """A network that reports the target 12 px off the patch still tracks it.

    Ocean does exactly this on natural images (it boxes the object it thinks
    the patch belongs to); reading every frame relative to the anchor's own
    answer cancels it.
    """
    video = SyntheticVideoSource(frame_count=24, width=64, height=48)
    kind = "ocean" if cls is OceanTracker else "siamdw"
    plugin = cls(backend=SiameseMarkerBackend(kind, bias=(12.0, -9.0)))
    result = plugin.track(video, [anchor_at(video, 6)], (0, video.frame_count))
    for frame in range(video.frame_count):
        error = np.hypot(*(result.points[0, frame] - video.true_position(frame)))
        assert error < 1.5, f"frame {frame}"


def _plugin_with_reference(width=60.0, height=60.0, dx=0.0, dy=0.0):
    plugin = OceanTracker(backend=SiameseMarkerBackend("ocean"))
    reference = SiameseStep(dx=dx, dy=dy, width=width, height=height, rate=0.0, evidence=1.0)
    geometry = _Geometry(instance_size=255, reach=100.0, anchor_scale=plugin.anchor_scale)
    return plugin, reference, geometry


def test_a_larger_box_grows_the_target_by_the_damped_rate():
    plugin, reference, geometry = _plugin_with_reference()
    target = _Target(100.0, 100.0)
    step = SiameseStep(dx=0.0, dy=0.0, width=66.0, height=66.0, rate=0.5, evidence=1.0)
    plugin._advance(target, step, reference, geometry.anchor_scale, geometry)
    assert target.scale == pytest.approx(1.05)
    assert (target.x, target.y) == pytest.approx((100.0, 100.0))


def test_a_box_of_another_shape_moves_the_point_but_not_its_scale():
    """Ocean flips between boxes of a patch; that must not walk the scale away."""
    plugin, reference, geometry = _plugin_with_reference()
    target = _Target(100.0, 100.0)
    step = SiameseStep(dx=8.0, dy=0.0, width=120.0, height=30.0, rate=1.0, evidence=1.0)
    plugin._advance(target, step, reference, geometry.anchor_scale, geometry)
    assert target.scale == pytest.approx(1.0)
    assert target.x == pytest.approx(100.0 + 8.0 / geometry.anchor_scale)


def test_the_scale_is_clamped():
    plugin, reference, geometry = _plugin_with_reference()
    target = _Target(100.0, 100.0, scale=4.9)
    # A target 4.9x its anchor size is searched in a crop 4.9x as wide.
    crop_scale = geometry.anchor_scale / 4.9
    step = SiameseStep(dx=0.0, dy=0.0, width=72.0, height=72.0, rate=1.0, evidence=1.0)
    plugin._advance(target, step, reference, crop_scale, geometry)
    assert target.scale == pytest.approx(plugin.scale_bounds[1])


def test_the_point_keeps_its_offset_from_what_the_network_tracks():
    """The reference reading sits 10 network px right of the point: it stays there."""
    plugin, reference, geometry = _plugin_with_reference(dx=10.0)
    target = _Target(100.0, 100.0)
    step = SiameseStep(dx=14.0, dy=0.0, width=60.0, height=60.0, rate=0.0, evidence=1.0)
    plugin._advance(target, step, reference, geometry.anchor_scale, geometry)
    assert target.x == pytest.approx(100.0 + 4.0 / geometry.anchor_scale)


@pytest.mark.parametrize("cls", PLUGINS)
def test_a_lost_point_holds_where_it_was_last_seen(cls, occluded_video):
    query = Query(5, *occluded_video.true_position(5))
    result = plugin_for(cls).track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")
    lo, hi = occluded_video.occlude_range
    held = result.points[0, lo:hi]
    assert np.allclose(held, result.points[0, lo - 1])
    assert not result.visible[0, lo:hi].any()
    assert (result.confidence[0, lo:hi] == 0.0).all()


@pytest.mark.parametrize("cls", PLUGINS)
def test_a_point_held_through_an_occlusion_is_found_again(cls, occluded_video):
    query = Query(5, *occluded_video.true_position(5))
    result = plugin_for(cls).track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")
    _, hi = occluded_video.occlude_range
    for frame in range(hi + 1, occluded_video.frame_count):
        error = np.hypot(*(result.points[0, frame] - occluded_video.true_position(frame)))
        assert error < 1.5, f"frame {frame}"
        assert result.confidence[0, frame] > 0.9


def test_a_weak_frame_can_be_followed_without_being_trusted(occluded_video):
    """lost_below below the calibration floor: confidence 0, but the point moves."""

    class Faint(SiameseMarkerBackend):
        def match(self, kernels, searches):
            maps = super().match(kernels, searches)
            similarity = np.where(maps.similarity > 0.5, 0.4, maps.similarity)
            return SiameseMaps(score=maps.score, boxes=maps.boxes, similarity=similarity)

    video = SyntheticVideoSource(frame_count=12, width=64, height=48)
    query = anchor_at(video, 0)
    trusting = OceanTracker(backend=Faint("ocean"), lost_below=0.3)
    holding = OceanTracker(backend=Faint("ocean"))
    moved = trusting.track(video, [query], (0, 12), "fwd")
    held = holding.track(video, [query], (0, 12), "fwd")
    assert (moved.confidence[0, 1:] == 0.0).all() and (held.confidence[0, 1:] == 0.0).all()
    assert np.hypot(*(moved.points[0, 11] - video.true_position(11))) < 1.5
    assert np.allclose(held.points[0, 1:], query.position)


@pytest.mark.parametrize("cls", PLUGINS)
def test_points_near_the_frame_edge_are_tracked_not_refused(cls):
    video = SyntheticVideoSource(frame_count=10, width=64, height=48)
    result = plugin_for(cls).track(video, [Query(0, 1.0, 1.0)], (0, 10))
    assert np.isfinite(result.points[0]).all()


@pytest.mark.parametrize("cls", PLUGINS)
def test_points_sharing_an_anchor_frame_share_each_network_call(cls, video):
    backend = SiameseMarkerBackend("ocean" if cls is OceanTracker else "siamdw")
    plugin = cls(backend=backend)
    queries = [anchor_at(video, 0), Query(0, 20.0, 20.0), Query(5, 30.0, 15.0)]
    plugin.track(video, queries, (0, video.frame_count), "fwd")
    assert {shape[0] for shape in backend.searches} == {2, 1}
    scales = 1 if cls is OceanTracker else 3
    assert {shape[1] for shape in backend.searches} == {scales}


@pytest.mark.parametrize("cls", PLUGINS)
def test_large_batches_are_split(cls, video, monkeypatch):
    monkeypatch.setattr(cls, "max_batch", 2)
    backend = SiameseMarkerBackend("ocean" if cls is OceanTracker else "siamdw")
    queries = [Query(0, 20.0 + i, 20.0) for i in range(5)]
    result = cls(backend=backend).track(video, queries, (0, 6), "fwd")
    assert max(shape[0] for shape in backend.searches) == 2
    assert result.n_points == 5


# -- the decoders ---------------------------------------------------------------


def _ocean_maps(cls_logits, boxes, similarity=None):
    score = np.asarray(cls_logits, dtype=np.float32)[None, None]
    return SiameseMaps(
        score=score,
        boxes=np.asarray(boxes, dtype=np.float32)[None, None],
        similarity=None if similarity is None else np.asarray(similarity)[None, None],
    )


def _uniform_boxes(size, half=30.0, shift=(0.0, 0.0)):
    grid = (np.arange(size) - size // 2) * 8.0
    gx, gy = np.meshgrid(grid, grid)
    cx, cy = shift
    return np.stack([gx - (cx - half), gy - (cy - half), (cx + half) - gx, (cy + half) - gy])


def test_ocean_reads_the_box_centre_from_the_chosen_cell():
    logits = np.full((25, 25), -6.0)
    logits[12, 14] = 6.0
    maps = _ocean_maps(logits, _uniform_boxes(25, shift=(13.5, -4.25)))
    step = OceanTracker()._decode(maps, 0, (60.0, 60.0), (0.0, 0.0), math.inf)
    assert (step.dx, step.dy) == pytest.approx((13.5, -4.25))
    assert (step.width, step.height) == pytest.approx((60.0, 60.0))


def test_ocean_never_chooses_a_cell_beyond_the_reach():
    logits = np.full((25, 25), -6.0)
    logits[12, 24] = 8.0  # the strongest cell, 96 network px right
    logits[12, 13] = 3.0  # a weaker one, 8 px right
    boxes = np.full((4, 25, 25), 30.0)  # every cell boxes itself, so dx names the cell
    plugin = OceanTracker(window_influence=0.0)
    unlimited = plugin._decode(_ocean_maps(logits, boxes), 0, (60.0, 60.0), (0.0, 0.0), math.inf)
    limited = plugin._decode(_ocean_maps(logits, boxes), 0, (60.0, 60.0), (0.0, 0.0), 20.0)
    assert unlimited.dx == pytest.approx(96.0)
    assert limited.dx == pytest.approx(8.0)


def test_ocean_prefers_a_box_shaped_like_the_expected_one():
    logits = np.full((25, 25), -6.0)
    logits[12, 11] = logits[12, 13] = 4.0
    boxes = _uniform_boxes(25)
    boxes[:, 12, 11] = [90.0, 15.0, 90.0, 15.0]  # wide and flat
    step = OceanTracker(penalty_k=0.5, window_influence=0.0)._decode(
        _ocean_maps(logits, boxes), 0, (60.0, 60.0), (0.0, 0.0), math.inf
    )
    assert step.width == pytest.approx(60.0)


def test_ocean_evidence_is_classifier_times_appearance():
    logits = np.zeros((25, 25))
    similarity = np.full((25, 25), 0.5)
    maps = _ocean_maps(logits, _uniform_boxes(25), similarity)
    step = OceanTracker()._decode(maps, 0, (60.0, 60.0), (0.0, 0.0), math.inf)
    assert step.evidence == pytest.approx(0.5 * 0.5)
    negative = _ocean_maps(logits, _uniform_boxes(25), -similarity)
    assert OceanTracker()._decode(negative, 0, (60.0, 60.0), (0.0, 0.0), math.inf).evidence == 0


def test_ocean_refuses_maps_without_boxes():
    maps = SiameseMaps(score=np.zeros((1, 1, 25, 25), dtype=np.float32))
    with pytest.raises(TrackerError, match="box"):
        OceanTracker()._decode(maps, 0, (60.0, 60.0), (0.0, 0.0), math.inf)


def _siamdw_maps(peaks, centre=(0.0, 0.0), size=17, sigma=8.0):
    grid = (np.arange(size) - size // 2) * 8.0
    gx, gy = np.meshgrid(grid, grid)
    bump = np.exp(-((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2) / (2 * sigma**2))
    score = np.stack([3.0 + (peak - 3.0) * bump for peak in peaks]).astype(np.float32)
    return SiameseMaps(score=score[None])


def test_siamdw_finds_the_peak_between_cells():
    """16x bicubic upsampling, read with the exact pixel-centre convention."""
    maps = _siamdw_maps([10.0, 4.0, 4.0], centre=(5.3, -2.6))
    step = SiamDWTracker(window_influence=0.0)._decode(maps, 0, (60.0, 60.0), (0.0, 0.0), math.inf)
    assert step.dx == pytest.approx(5.3, abs=0.5)
    assert step.dy == pytest.approx(-2.6, abs=0.5)


def test_siamdw_changes_scale_only_on_clearly_better_evidence():
    plugin = SiamDWTracker(window_influence=0.0)
    slightly = plugin._decode(_siamdw_maps([10.0, 4.0, 10.2]), 0, (60.0, 60.0), (0, 0), math.inf)
    clearly = plugin._decode(_siamdw_maps([10.0, 4.0, 12.0]), 0, (60.0, 60.0), (0, 0), math.inf)
    assert slightly.width == pytest.approx(60.0)
    assert clearly.width == pytest.approx(60.0 * plugin.scale_step)


def test_siamdw_reports_a_peak_found_at_another_scale_in_reference_pixels():
    plugin = SiamDWTracker(window_influence=0.0)
    step = plugin._decode(
        _siamdw_maps([4.0, 4.0, 12.0], centre=(16.0, 0.0)), 0, (60, 60), (0, 0), 1e9
    )
    assert step.dx == pytest.approx(16.0 * plugin.scale_step, abs=0.6)


def test_siamdw_never_chooses_a_peak_beyond_the_reach():
    plugin = SiamDWTracker(window_influence=0.0)
    far = _siamdw_maps([10.0, 4.0, 4.0], centre=(48.0, 0.0))
    step = plugin._decode(far, 0, (60.0, 60.0), (0.0, 0.0), 16.0)
    assert abs(step.dx) <= 16.0


def test_siamdw_evidence_is_relative_to_the_anchors_own_response():
    plugin = SiamDWTracker()
    reference = SiameseStep(0, 0, 60, 60, 0.0, evidence=10.0)
    assert plugin.raw_confidence(SiameseStep(0, 0, 60, 60, 0.0, 8.5), reference) == pytest.approx(
        0.85
    )
    flat = SiameseStep(0, 0, 60, 60, 0.0, evidence=0.0)
    assert plugin.raw_confidence(reference, flat) == 0.0


@pytest.mark.parametrize("cls", PLUGINS)
def test_confidence_calibration_has_dead_zones_at_both_ends(cls):
    plugin = cls()
    low, high = plugin.confidence_range
    confidence, visible = plugin.normalize_confidence(
        np.array([0.0, low, (low + high) / 2, high, 2])
    )
    assert confidence.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert visible.tolist() == [False, False, True, True, True]


# -- lifecycle, availability and weights -------------------------------------------


def test_the_siamese_plugins_share_the_model_lifecycle():
    assert issubclass(SiameseTrackerPlugin, ModelTrackerPlugin)
    assert isinstance(SiameseMarkerBackend("ocean"), SiameseBackend)


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_checkpoint_is_pinned_and_commercially_usable(cls):
    spec = cls.model_spec
    assert spec is not None and spec.pinned
    assert spec.license == cls.license == "MIT"
    assert spec.non_commercial is False
    assert spec.size_bytes and spec.size_bytes > 1_000_000
    assert spec in registered_models()


def test_the_checkpoints_are_the_ones_that_were_verified():
    assert OCEAN_OFFLINE.sha256 == (
        "16badb707871ba83a786f32fce5bdf7946254679cbc0051f32bf1f2c8e6cd805"
    )
    assert SIAMDW_RES22W.sha256 == (
        "1392552b0be47bef518219c47df8ccc7abbf1bc63c44ca5039e66c38ca9bedad"
    )


@pytest.mark.parametrize("cls", PLUGINS)
def test_it_never_requires_a_gpu(cls):
    assert cls.requires_gpu is False


@pytest.mark.parametrize("cls", PLUGINS)
def test_without_torch_it_is_unavailable_and_says_why(cls, monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert cls.is_available() is False
    assert "PyTorch" in (cls.unavailable_reason() or "")


@pytest.mark.parametrize("cls", PLUGINS)
def test_without_opencv_it_asks_for_the_video_extra_not_for_model_code(cls, monkeypatch):
    """The network code ships with Stable; OpenCV is what is missing."""
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: object())
    _hide_module(monkeypatch, "cv2")
    reason = cls.unavailable_reason() or ""
    assert "OpenCV" in reason and "stable-tracker[video]" in reason
    assert "model code" not in reason


@pytest.mark.parametrize("cls", PLUGINS)
def test_building_the_backend_without_torch_says_so(cls, monkeypatch):
    module = "stable.trackers.ocean" if cls is OceanTracker else "stable.trackers.siamdw"
    monkeypatch.setattr(f"{module}.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        cls()._build_backend("cpu", "weights.pth")


@pytest.mark.parametrize("cls", PLUGINS)
def test_out_of_memory_mid_run_continues_on_the_cpu_with_rebuilt_templates(cls, monkeypatch, video):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    backend = SiameseMarkerBackend("ocean" if cls is OceanTracker else "siamdw", oom_on=(3,))
    backend.device = "cuda"
    plugin = cls(backend=backend)
    plugin.prepare("cuda")

    result = plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")

    assert plugin.device == "cpu" and plugin.fell_back_to_cpu is True
    assert backend.embeds == 2, "the templates must be rebuilt on the new device"
    assert result.confidence[0, 1:].min() > 0.9


@pytest.mark.parametrize("cls", PLUGINS)
def test_out_of_memory_on_the_cpu_is_reported_clearly(cls, video):
    backend = SiameseMarkerBackend("ocean" if cls is OceanTracker else "siamdw", oom_on=(1,))
    backend.device = "cuda"
    plugin = cls(backend=backend)
    plugin.prepare("cpu")
    with pytest.raises(TrackerError, match="ran out of memory"):
        plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")


@pytest.mark.parametrize("cls", PLUGINS)
def test_backend_bugs_surface_as_tracker_errors(cls, video):
    backend = SiameseMarkerBackend("ocean" if cls is OceanTracker else "siamdw", fail_on=(2,))
    with pytest.raises(TrackerError, match="inference failed"):
        cls(backend=backend).track(video, [anchor_at(video, 0)], (0, 10), "fwd")


def test_a_backend_returning_the_wrong_batch_is_caught(video):
    class Short(SiameseMarkerBackend):
        def match(self, kernels, searches):
            maps = super().match(kernels, searches)
            return SiameseMaps(score=maps.score[:, :0], boxes=None)

    with pytest.raises(TrackerError, match="expected"):
        SiamDWTracker(backend=Short("siamdw")).track(video, [anchor_at(video, 0)], (0, 5))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"template_radius": 0}, "template_radius"),
        ({"search_radius": -1}, "search_radius"),
        ({"window_influence": 1.0}, "window_influence"),
        ({"confidence_range": (0.6, 0.6)}, "confidence_range"),
    ],
)
@pytest.mark.parametrize("cls", PLUGINS)
def test_arguments_are_validated(cls, kwargs, message):
    with pytest.raises(ContractError, match=message):
        cls(**kwargs)


def test_siamdw_scale_arguments_are_validated():
    with pytest.raises(ContractError, match="scale_step"):
        SiamDWTracker(scale_step=0.9)
    with pytest.raises(ContractError, match="response_upsample"):
        SiamDWTracker(response_upsample=0)


def test_maps_validate_their_shapes():
    with pytest.raises(ContractError, match="score"):
        SiameseMaps(score=np.zeros((1, 25, 25)))
    with pytest.raises(ContractError, match="box"):
        SiameseMaps(score=np.zeros((1, 1, 25, 25)), boxes=np.zeros((1, 1, 3, 25, 25)))
    with pytest.raises(ContractError, match="similarity"):
        SiameseMaps(score=np.zeros((1, 1, 25, 25)), similarity=np.zeros((1, 1, 24, 24)))


# -- the operator's settings ---------------------------------------------------------


@pytest.mark.parametrize("name", ["ocean", "siamdw"])
def test_the_pattern_and_search_sizes_are_offered_per_point(name):
    rows = {row["attribute"]: row for row in tracker_settings(name)}
    assert rows["template_radius"]["role"] == "window"
    assert rows["search_radius"]["role"] == "search"
    assert tracker_window(name) == (16, "template_radius")
    radius, attribute, _ = tracker_search(name, {"search_radius": 40})
    assert (radius, attribute) == (40, "search_radius")


def test_the_settings_reach_the_plugin_through_the_registry():
    from stable.trackers.registry import create_tracker

    plugin = create_tracker(
        "siamdw", prepare=False, options={"template_radius": 24, "search_radius": 64, "win_size": 9}
    )
    assert (plugin.template_radius, plugin.search_radius) == (24, 64)
