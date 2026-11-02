"""MixFormerV2 as a Stable plugin: its one-box reading, confidence, geometry and wiring.

What is and is not verified here
--------------------------------
``tests/test_tracker_suite.py`` holds both MixFormerV2 plugins to the contract
every tracker meets, through :class:`~conftest.MixFormerMarkerBackend`. This
module covers what is particular to them: how one regressed box per search crop
becomes a point or a box (riding, scale, the gate, the reach), the evidence and
its two calibrations, the search geometry the server's controls rely on, the
checkpoints and the entry points. Nothing here needs PyTorch or a download; the
network and the backend's tensor conventions are ``tests/test_mixformerv2_net.py``.

A real inference run is not part of the suite (the checkpoints are 70 MB and
240 MB). Both were run for real when they were pinned — parity with the
authors' code, and tracking measured on clips of known motion — as the module
docstring of :mod:`stable.trackers.mixformerv2` records.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEntryPoint, MixFormerMarkerBackend, anchor_at

from stable.core.errors import ContractError, TrackerError
from stable.core.types import Query
from stable.core.video import SyntheticVideoSource
from stable.server.tracker_options import (
    box_search_scale,
    tracker_search,
    tracker_settings,
    tracker_window,
)
from stable.trackers.mixformerv2 import (
    MIXFORMERV2_BASE_WEIGHTS,
    MIXFORMERV2_SMALL_WEIGHTS,
    MixFormerV2Maps,
    MixFormerV2SmallTracker,
    MixFormerV2Step,
    MixFormerV2Tracker,
)
from stable.trackers.registry import TrackerRegistry, registry
from stable.trackers.siamese import SiameseBackend, SiameseTrackerPlugin, _Seed, _Target
from stable.trackers.weights import registered_models

PLUGINS = [MixFormerV2Tracker, MixFormerV2SmallTracker]
ROOT = Path(__file__).resolve().parents[1]


def plugin_for(cls, **kwargs):
    backend = kwargs.pop("backend", None) or MixFormerMarkerBackend()
    return cls(backend=backend, **kwargs)


def one_box(x1, y1, x2, y2, *, logit=6.0, spread=3.0):
    """MixFormerV2-shaped maps for one point: a box relative to the crop centre."""
    return MixFormerV2Maps(
        score=np.full((1, 1, 1, 1), logit, dtype=np.float32),
        boxes=np.array([-x1, -y1, x2, y2], dtype=np.float32).reshape(1, 1, 4, 1, 1),
        spread=np.full((1, 1, 1, 1), spread, dtype=np.float32),
    )


@pytest.fixture
def registered():
    """Both plugins in the process-wide registry, as their entry points put them.

    The registry reads *installed* entry-point metadata, which lags
    ``pyproject.toml`` until the package is reinstalled; this makes the server
    tests independent of that.
    """
    registry.discover()
    added = [cls for cls in PLUGINS if cls.name not in registry]
    for cls in added:
        registry.register(cls)
    yield
    for cls in added:
        registry.unregister(cls.name)


# -- reading one box -----------------------------------------------------------------


def test_the_box_centre_and_size_come_from_the_one_regressed_box():
    step = MixFormerV2Tracker()._decode(
        one_box(-20.0, -40.0, 34.0, 10.0), 0, (54.0, 50.0), (0.0, 0.0), math.inf, box=True
    )
    assert (step.dx, step.dy) == pytest.approx((7.0, -15.0))
    assert (step.width, step.height) == pytest.approx((54.0, 50.0))
    assert step.evidence == pytest.approx(1 / (1 + math.exp(-6.0)))


def test_a_box_beyond_the_reach_is_no_evidence_at_all():
    """There is only one candidate, so a point told it cannot move that far holds."""
    plugin = MixFormerV2Tracker()
    far = one_box(60.0, -30.0, 120.0, 30.0)  # centred 90 px right
    assert plugin._decode(far, 0, (60.0, 60.0), (0.0, 0.0), 40.0).evidence == 0.0
    assert plugin._decode(far, 0, (60.0, 60.0), (80.0, 0.0), 40.0).evidence > 0.9


def test_maps_without_a_box_are_refused():
    from stable.trackers.siamese import SiameseMaps

    maps = SiameseMaps(score=np.zeros((1, 1, 1, 1), dtype=np.float32))
    with pytest.raises(TrackerError, match="box"):
        MixFormerV2Tracker()._decode(maps, 0, (60.0, 60.0), (0.0, 0.0), math.inf)


def test_maps_check_the_spread_shape():
    with pytest.raises(ContractError, match="spread"):
        MixFormerV2Maps(
            score=np.zeros((1, 1, 1, 1)),
            boxes=np.zeros((1, 1, 4, 1, 1)),
            spread=np.zeros((2, 1, 1, 1)),
        )


def test_a_box_that_changed_past_the_gate_is_pulled_back_to_it():
    """After an occlusion the object may be far bigger than the held box: grow toward it."""
    plugin = MixFormerV2Tracker()
    step = plugin._decode(
        one_box(-60.0, -60.0, 60.0, 60.0), 0, (60.0, 60.0), (0.0, 0.0), math.inf, box=True
    )
    assert step.width == pytest.approx(step.height)
    assert 1.25 < step.width / 60.0 < plugin.scale_gate
    unchanged = plugin._decode(
        one_box(-33.0, -30.0, 33.0, 30.0), 0, (60.0, 60.0), (0.0, 0.0), math.inf, box=True
    )
    assert (unchanged.width, unchanged.height) == pytest.approx((66.0, 60.0))


def test_a_points_box_is_left_to_the_scale_gate():
    """A patch's box that flips shape moves the point but must not rescale it."""
    step = MixFormerV2Tracker()._decode(
        one_box(-60.0, -15.0, 60.0, 15.0), 0, (60.0, 60.0), (0.0, 0.0), math.inf
    )
    assert (step.width, step.height) == pytest.approx((120.0, 30.0))


# -- confidence --------------------------------------------------------------------------


def test_a_points_evidence_is_score_times_sharpness_times_shape_agreement():
    plugin = MixFormerV2Tracker()
    reference = MixFormerV2Step(0, 0, 60, 60, 0.5, 0.99, score=0.99, spread=3.0)
    sharp = plugin._decode(one_box(-30, -30, 30, 30, spread=3.0), 0, (60, 60), (0, 0), math.inf)
    blurred = plugin._decode(one_box(-30, -30, 30, 30, spread=6.0), 0, (60, 60), (0, 0), math.inf)
    wider = plugin._decode(one_box(-45, -30, 45, 30, spread=3.0), 0, (60, 60), (0, 0), math.inf)
    score = 1 / (1 + math.exp(-6.0))
    assert plugin.raw_confidence(sharp, reference) == pytest.approx(score)
    assert plugin.raw_confidence(blurred, reference) == pytest.approx(score / 2)
    assert plugin.raw_confidence(wider, reference) == pytest.approx(score * (60 / 90) ** 2)
    sharper = plugin._decode(one_box(-30, -30, 30, 30, spread=1.0), 0, (60, 60), (0, 0), math.inf)
    assert plugin.raw_confidence(sharper, reference) == pytest.approx(score), "capped at 1"


def test_a_boxs_evidence_is_the_score_alone():
    """An object growing in frame widens its edge distributions without being lost."""
    plugin = MixFormerV2Tracker()
    reference = MixFormerV2Step(0, 0, 60, 60, 1.0, 0.99, score=0.99, spread=3.0, box=True)
    grown = plugin._decode(
        one_box(-40, -40, 40, 40, spread=9.0), 0, (60, 60), (0, 0), math.inf, box=True
    )
    assert plugin.raw_confidence(grown, reference) == pytest.approx(1 / (1 + math.exp(-6.0)))


def test_evidence_without_a_spread_is_the_score():
    from stable.trackers.siamese import SiameseStep

    plugin = MixFormerV2Tracker()
    plain = SiameseStep(0, 0, 60, 60, 0.5, evidence=0.8)
    assert plugin.raw_confidence(plain, plain) == pytest.approx(0.8)


@pytest.mark.parametrize("cls", PLUGINS)
def test_confidence_calibration_has_dead_zones_at_both_ends(cls):
    plugin = cls()
    for box in (False, True):
        low, high = plugin.box_confidence_range if box else plugin.confidence_range
        confidence, visible = plugin.normalize_confidence(
            np.array([0.0, low, (low + high) / 2, high, 1.0]), box=box
        )
        assert confidence.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
        assert visible.tolist() == [False, False, True, True, True]


@pytest.mark.parametrize("cls", PLUGINS)
def test_points_and_boxes_are_held_where_their_calibrations_start(cls):
    plugin = cls()
    assert plugin.lost_below == plugin.confidence_range[0]
    assert plugin.box_lost_below == plugin.box_confidence_range[0]


def test_the_calibrations_are_the_measured_ones():
    """See the module docstring for the distributions behind these numbers."""
    base, small = MixFormerV2Tracker(), MixFormerV2SmallTracker()
    assert base.confidence_range == (0.3, 0.7)
    assert small.confidence_range == (0.5, 0.8)
    assert base.box_confidence_range == small.box_confidence_range == (0.3, 0.75)


# -- tracking through the stub -----------------------------------------------------------


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_point_rides_on_the_box_rather_than_trusting_its_centre(cls):
    video = SyntheticVideoSource(frame_count=24, width=64, height=48)
    plugin = cls(backend=MixFormerMarkerBackend(bias=(12.0, -9.0)))
    result = plugin.track(video, [anchor_at(video, 6)], (0, video.frame_count))
    for frame in range(video.frame_count):
        error = np.hypot(*(result.points[0, frame] - video.true_position(frame)))
        assert error < 1.5, f"frame {frame}"


@pytest.mark.parametrize("cls", PLUGINS)
def test_a_lost_point_holds_and_is_found_again(cls, occluded_video):
    query = Query(5, *occluded_video.true_position(5))
    result = plugin_for(cls).track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")
    lo, hi = occluded_video.occlude_range
    assert np.allclose(result.points[0, lo:hi], result.points[0, lo - 1])
    assert (result.confidence[0, lo:hi] == 0.0).all()
    for frame in range(hi + 1, occluded_video.frame_count):
        error = np.hypot(*(result.points[0, frame] - occluded_video.true_position(frame)))
        assert error < 1.5, f"frame {frame}"
        assert result.confidence[0, frame] > 0.9


class GrowingBox(SyntheticVideoSource):
    """A white rectangle that drifts right, widens and flattens; hidden over ``hidden``."""

    def __init__(self, *, hidden: tuple[int, int] = (0, 0)) -> None:
        super().__init__(frame_count=30, width=220, height=150)
        self.hidden = hidden

    def true_box(self, frame: int) -> tuple[float, float, float, float]:
        return (70.0 + 1.5 * frame, 75.0, 24.0 + 1.2 * frame, 24.0 - 0.3 * frame)

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


class RectangleBackend(MixFormerMarkerBackend):
    """Boxes whatever white rectangle the crop shows: its real extent, as MixFormerV2 does."""

    def box_for(self, crop):
        ys, xs = np.nonzero(crop.min(axis=2) >= 250)
        if not len(xs):
            return None
        centre = (crop.shape[0] - 1) / 2.0
        return (
            xs.min() - 0.5 - centre,
            ys.min() - 0.5 - centre,
            xs.max() + 0.5 - centre,
            ys.max() + 0.5 - centre,
        )


def _box_error(video, result, frame):
    cx, cy, w, h = video.true_box(frame)
    (x, y), (rw, rh) = result.points[0, frame], result.sizes[0, frame]
    return math.hypot(x - cx, y - cy), rw / w, rh / h


@pytest.mark.parametrize("cls", PLUGINS)
def test_a_box_follows_an_object_that_changes_size_and_shape(cls):
    video = GrowingBox()
    result = cls(backend=RectangleBackend()).track(video, [video.query(0)], (0, 30))
    assert result.has_boxes
    # Undamped, unlike Ocean (MixFormerV2's box is taken as it comes, as upstream
    # does), so the fixture's whole-pixel rectangle shows through as a few percent.
    for frame in range(30):
        moved, along_w, along_h = _box_error(video, result, frame)
        assert moved < 1.5, f"frame {frame}"
        assert 0.9 < along_w < 1.1 and 0.9 < along_h < 1.1, f"frame {frame}"
    width, height = result.sizes[0, -1]
    assert width / height > 3.0  # it started square


def test_a_box_held_through_an_occlusion_grows_back_to_the_object():
    """The object kept growing while hidden; the gate must not freeze the held size."""
    video = GrowingBox(hidden=(8, 20))
    result = MixFormerV2Tracker(backend=RectangleBackend()).track(
        video, [video.query(0)], (0, 30), "fwd"
    )
    assert np.allclose(result.sizes[0, 8:20], result.sizes[0, 7])
    assert (result.confidence[0, 8:20] == 0.0).all()
    moved, along_w, along_h = _box_error(video, result, 29)
    assert moved < 1.5
    assert 0.9 < along_w < 1.1 and 0.9 < along_h < 1.1


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_drawn_box_is_ground_truth_on_its_anchor_frame(cls):
    video = GrowingBox()
    query = video.query(12)
    result = plugin_for(cls, backend=RectangleBackend()).track(video, [query], (0, 30))
    assert tuple(result.points[0, 12]) == pytest.approx(query.position)
    assert tuple(result.sizes[0, 12]) == pytest.approx(query.size)


@pytest.mark.parametrize("cls", PLUGINS)
def test_points_sharing_an_anchor_frame_share_each_network_call(cls, video):
    backend = MixFormerMarkerBackend()
    queries = [anchor_at(video, 0), Query(0, 20.0, 20.0), Query(5, 30.0, 15.0)]
    cls(backend=backend).track(video, queries, (0, video.frame_count), "fwd")
    assert {shape[0] for shape in backend.searches} == {2, 1}
    assert {shape[1] for shape in backend.searches} == {1}


@pytest.mark.parametrize("cls", PLUGINS)
def test_out_of_memory_mid_run_continues_on_the_cpu_with_rebuilt_templates(cls, monkeypatch, video):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    backend = MixFormerMarkerBackend(oom_on=(3,))
    backend.device = "cuda"
    plugin = cls(backend=backend)
    plugin.prepare("cuda")
    result = plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")
    assert plugin.device == "cpu" and plugin.fell_back_to_cpu is True
    assert backend.embeds == 2, "the templates must be rebuilt on the new device"
    assert result.confidence[0, 1:].min() > 0.9


# -- geometry the server's controls rely on --------------------------------------------


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_notional_score_map_reaches_until_the_target_leaves_the_crop(cls):
    plugin = cls()
    for size in range(255, 512, 16):
        cells = plugin.score_size(size)
        assert cells % 2 == 1
        reach = (cells - 1) / 2 * plugin.total_stride
        assert reach + plugin.exemplar_size / 4 <= size / 2 < reach + plugin.exemplar_size / 4 + 16


@pytest.mark.parametrize("cls", PLUGINS)
def test_the_default_point_search_covers_its_search_radius(cls):
    plugin = cls()
    size, reach = plugin.search_geometry()
    assert size == 271
    assert (plugin.score_size(size) - 1) / 2 * plugin.total_stride >= reach


@pytest.mark.parametrize(
    ("cls", "size"), [(MixFormerV2Tracker, 319), (MixFormerV2SmallTracker, 287)]
)
def test_a_box_is_searched_at_upstreams_benchmark_search_factor(cls, size):
    """319 px around a 127 px exemplar is upstream's 5.0x search for B; 287 its 4.5x for S."""
    video = GrowingBox()
    backend = RectangleBackend()
    cls(backend=backend).track(video, [video.query(0)], (0, 4), "fwd")
    assert {shape[2] for shape in backend.searches} == {size}


@pytest.mark.parametrize("cls", PLUGINS)
@pytest.mark.parametrize("size", [255, 271, 511])
def test_every_search_window_the_server_offers_is_accepted(cls, size):
    """The server validates 255-511 in steps of 16; all of it must build."""
    assert cls(box_search_size=size).box_search_size == size


@pytest.mark.parametrize("cls", PLUGINS)
@pytest.mark.parametrize("size", [239, 254, 256, 527, 319.5])
def test_invalid_box_search_windows_are_rejected(cls, size):
    with pytest.raises(ContractError, match="box_search_size"):
        cls(box_search_size=size)


def test_a_wider_box_search_keeps_the_template_and_covers_more_of_the_frame():
    query = Query(0, 100.0, 100.0, 40.0, 20.0)
    default, wider = MixFormerV2Tracker(), MixFormerV2Tracker(box_search_size=415)
    seed = default._seed(query, default.search_geometry())
    wide_seed = wider._seed(query, wider.search_geometry())
    assert wide_seed.anchor_scale == seed.anchor_scale
    target = _Target(query.x, query.y)
    ratio = wider._search_side(target, wide_seed) / default._search_side(target, seed)
    assert ratio == pytest.approx(415 / 319)


def test_the_point_keeps_its_offset_from_what_the_network_boxes():
    plugin = MixFormerV2Tracker()
    side = float(plugin.box_side)
    seed = _Seed(100.0, 100.0, side, side, box=False, instance_size=271, reach=100.0)
    reference = MixFormerV2Step(10.0, 0.0, 60.0, 60.0, 0.0, 1.0)
    target = _Target(100.0, 100.0)
    step = MixFormerV2Step(14.0, 0.0, 60.0, 60.0, 0.0, 1.0)
    plugin._advance(target, step, reference, seed.anchor_scale, seed)
    assert target.x == pytest.approx(100.0 + 4.0 / seed.anchor_scale)


# -- lifecycle, weights and wiring ----------------------------------------------------------


def test_both_plugins_are_siamese_box_trackers():
    for cls in PLUGINS:
        assert issubclass(cls, SiameseTrackerPlugin)
        assert cls.tracks_boxes and cls.requires_gpu is False
    assert isinstance(MixFormerMarkerBackend(), SiameseBackend)


@pytest.mark.parametrize(
    ("cls", "spec", "digest", "size"),
    [
        (
            MixFormerV2Tracker,
            MIXFORMERV2_BASE_WEIGHTS,
            "14a44d0c90ee856f3fe36af31c4f0af643cbc2ba47546de98c7162d3d704a228",
            240_205_177,
        ),
        (
            MixFormerV2SmallTracker,
            MIXFORMERV2_SMALL_WEIGHTS,
            "db94d947e2bf69918640aaf5dfce16a6dbed108fe942f43a8832324e5b6c569e",
            69_679_305,
        ),
    ],
)
def test_each_plugin_loads_its_own_verified_checkpoint(cls, spec, digest, size):
    assert cls.model_spec is spec
    assert spec.pinned and spec.sha256 == digest and spec.size_bytes == size
    assert spec.license == cls.license == "MIT"
    assert spec.non_commercial is False
    assert spec.url.startswith("https://drive.usercontent.google.com/download?id=")
    assert spec in registered_models()


@pytest.mark.parametrize("cls", PLUGINS)
def test_without_torch_it_is_unavailable_and_says_why(cls, monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert cls.is_available() is False
    assert "PyTorch" in (cls.unavailable_reason() or "")


@pytest.mark.parametrize("cls", PLUGINS)
@pytest.mark.parametrize(
    ("version", "available"), [("2.5.1", False), ("2.6.0", True), ("2.14.1+cu130", True)]
)
def test_a_pytorch_too_old_to_read_the_checkpoint_safely_is_reported_before_any_download(
    cls, monkeypatch, version, available
):
    from types import SimpleNamespace

    fake = SimpleNamespace(__version__=version)
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: fake)
    monkeypatch.setattr("stable.trackers.mixformerv2.torch_module", lambda: fake)
    assert cls.is_available() is available
    if not available:
        reason = cls.unavailable_reason() or ""
        assert "PyTorch 2.6" in reason and "torch>=2.6" in reason


@pytest.mark.parametrize("cls", PLUGINS)
def test_without_opencv_it_asks_for_the_video_extra(cls, monkeypatch):
    import sys

    for cached in [n for n in list(sys.modules) if n == "cv2" or n.startswith("cv2.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setitem(sys.modules, "cv2", None)
    reason = cls.unavailable_reason() or ""
    assert "OpenCV" in reason and "stable-tracker[video]" in reason


@pytest.mark.parametrize("cls", PLUGINS)
def test_building_the_backend_without_torch_says_so(cls, monkeypatch):
    monkeypatch.setattr("stable.trackers.mixformerv2.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        cls()._build_backend("cpu", "weights.pth.tar")


def test_the_entry_points_are_declared():
    """What `pip install` publishes; checked from the source, not the installed metadata."""
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = declared["project"]["entry-points"]["stable.trackers"]
    assert group["mixformerv2"] == "stable.trackers.mixformerv2:MixFormerV2Tracker"
    assert group["mixformerv2_small"] == "stable.trackers.mixformerv2:MixFormerV2SmallTracker"


def test_the_entry_points_load_and_describe_box_trackers():
    reg = TrackerRegistry()
    names = reg.discover(
        entry_points=[FakeEntryPoint(cls.name, cls) for cls in PLUGINS], strict=True
    )
    assert {"mixformerv2", "mixformerv2_small"} <= set(names)
    rows = {row["name"]: row for row in reg.describe()}
    for cls in PLUGINS:
        row = rows[cls.name]
        assert row["tracks_boxes"] is True and row["inputs"] == ["box", "point"]
        assert row["license"] == "MIT" and "MixFormerV2" in row["display_name"]


@pytest.mark.usefixtures("registered")
def test_the_server_offers_the_siamese_controls():
    assert tracker_window("mixformerv2") == (16, "template_radius")
    radius, attribute, _ = tracker_search("mixformerv2")
    assert (radius, attribute) == (48, "search_radius")
    box_rows = {row["attribute"]: row for row in tracker_settings("mixformerv2", box=True)}
    assert list(box_rows) == ["box_search_size"]
    assert box_rows["box_search_size"]["default"] == 319
    assert box_search_scale("mixformerv2") == pytest.approx(319 / 127)
    small = tracker_settings("mixformerv2_small", box=True)
    assert small[0]["default"] == 287


@pytest.mark.usefixtures("registered")
def test_the_model_manager_names_each_checkpoints_tracker():
    from stable.server.routes.models import _trackers_for

    assert _trackers_for("mixformerv2_base") == ["mixformerv2"]
    assert _trackers_for("mixformerv2_small") == ["mixformerv2_small"]
