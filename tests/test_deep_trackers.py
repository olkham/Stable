"""Tests for the deep-tracker pipeline: chunking, scaling, fallback, calibration.

Everything the plugins own but the neural network — which is exactly what can be
tested on a machine with no GPU, no torch and no checkpoints.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import numpy as np
import pytest
from conftest import EchoBackend, MarkerBackend, anchor_at

from stable.core.errors import ContractError, TrackerError, WeightsError
from stable.core.types import Query
from stable.trackers import cotracker3 as cotracker3_module
from stable.trackers import tapir as tapir_module
from stable.trackers.cotracker3 import CoTracker3Tracker
from stable.trackers.deep import (
    BackendPrediction,
    DeepTrackerPlugin,
    TrackerBackend,
    is_out_of_memory,
    resize_frames,
)
from stable.trackers.raft import RaftTracker, _pad_to_multiple, _sample_bilinear
from stable.trackers.tapir import TapirTracker
from stable.trackers.weights import ENV_MODEL_DIR, ModelSpec

PAYLOAD = b"stub weights" * 100
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _hide_module(monkeypatch, name: str) -> None:
    """Make ``import <name>`` (and its submodules) fail, however it is spelled.

    Setting one ``sys.modules`` entry to ``None`` is the documented way to make
    an import raise, but it only bites when the module is not *already*
    imported: ``from torchvision.models.optical_flow import raft_small``
    resolves straight out of the cache if an earlier test pulled it in. That is
    why the RAFT test passed alone and failed in the full run. Evicting the
    whole subtree first closes that gap; monkeypatch restores it afterwards.
    """
    for cached in [n for n in list(sys.modules) if n == name or n.startswith(f"{name}.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setitem(sys.modules, name, None)


class StubDeep(DeepTrackerPlugin):
    """A minimal concrete deep plugin whose "model" is a fake backend."""

    name = "stub_deep"
    display_name = "Stub Deep"
    requires_gpu = False
    license = "MIT"
    default_chunk_size = 6
    default_overlap = 2
    default_model_resolution = None

    def __init__(self, *, build_error: dict[str, Exception] | None = None, **kwargs: Any) -> None:
        """Create a stub plugin; ``build_error`` maps device -> exception to raise."""
        super().__init__(**kwargs)
        self.build_error = build_error or {}
        self.built_on: list[str] = []
        self.made: list[Any] = []

    def _require_implementation(self) -> None:
        """The stub's "model" is a plain NumPy fake, so it needs no torch."""

    def _build_backend(self, device: str, weights: Any) -> TrackerBackend:
        """Record the device, honour ``build_error``, and hand back a fake."""
        self.built_on.append(device)
        error = self.build_error.get(device)
        if error is not None:
            raise error
        backend = MarkerBackend()
        backend.device = device
        backend.weights = weights
        self.made.append(backend)
        return backend

    def normalize_confidence(self, prediction: BackendPrediction):
        """Pass the visibility head straight through."""
        confidence = np.clip(prediction.visibility, 0.0, 1.0).astype(np.float32)
        return confidence, confidence >= 0.5


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the model cache at a temp directory."""
    monkeypatch.setenv(ENV_MODEL_DIR, str(tmp_path / "models"))
    return tmp_path


@pytest.fixture
def served(tmp_path):
    """A ModelSpec served from a local file:// URL."""
    path = tmp_path / "served.bin"
    path.write_bytes(PAYLOAD)
    return ModelSpec(key="stub_model", url=path.as_uri(), sha256=DIGEST)


# -- out-of-memory classification -----------------------------------------


def test_out_of_memory_is_recognised_across_torch_versions():
    class OutOfMemoryError(RuntimeError):
        """Stand-in for torch.cuda.OutOfMemoryError."""

    assert is_out_of_memory(OutOfMemoryError("boom"))
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_out_of_memory(MemoryError())
    assert not is_out_of_memory(RuntimeError("shape mismatch"))
    assert not is_out_of_memory(ValueError("bad query"))


# -- device policy --------------------------------------------------------


def test_prepare_resolves_auto_to_a_real_device():
    plugin = StubDeep()
    plugin.prepare("auto")
    assert plugin.is_prepared and plugin.device in ("cpu", "cuda", "mps")


def test_prepare_is_idempotent():
    plugin = StubDeep()
    plugin.prepare("cpu")
    plugin.prepare("cpu")
    assert plugin.built_on == ["cpu"]  # weights loaded exactly once


def test_prepare_falls_back_to_cpu_when_the_gpu_will_not_load(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    plugin = StubDeep(build_error={"cuda": RuntimeError("CUDA error: no kernel image")})

    plugin.prepare("cuda")

    assert plugin.built_on == ["cuda", "cpu"]
    assert plugin.device == "cpu"
    assert plugin.fell_back_to_cpu is True


def test_a_gpu_failure_is_raised_when_fallback_is_disabled(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    plugin = StubDeep(
        allow_cpu_fallback=False, build_error={"cuda": RuntimeError("CUDA out of memory")}
    )
    with pytest.raises(TrackerError, match="could not load"):
        plugin.prepare("cuda")


def test_a_real_bug_during_load_is_not_mistaken_for_a_device_problem(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    plugin = StubDeep(build_error={"cuda": KeyError("missing state_dict key")})
    with pytest.raises(TrackerError, match="could not load"):
        plugin.prepare("cuda")
    assert plugin.built_on == ["cuda"]  # no pointless CPU retry


def test_out_of_memory_mid_run_continues_on_the_cpu(monkeypatch, video):
    """PLAN.md: GPU with automatic CPU fallback — a job must not die on OOM."""
    monkeypatch.setattr("stable.trackers.deep.select_device", lambda device: "cuda")
    backend = MarkerBackend(oom_on=(2,))
    backend.device = "cuda"
    plugin = StubDeep(backend=backend, chunk_size=6, overlap=2)
    plugin.prepare("cuda")
    assert plugin.device == "cuda"

    result = plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")

    assert plugin.device == "cpu" and plugin.fell_back_to_cpu is True
    assert backend.device == "cpu"
    assert np.isfinite(result.points[0, 0:]).all()  # the run completed regardless
    assert result.confidence[0, 5:].min() > 0.9


def test_out_of_memory_on_the_cpu_is_reported_clearly(video):
    backend = MarkerBackend(oom_on=(1,))
    backend.device = "cuda"  # the backend "is" on a GPU, the plugin thinks CPU
    plugin = StubDeep(backend=backend)
    plugin.prepare("cpu")
    with pytest.raises(TrackerError, match="ran out of memory"):
        plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")


def test_backend_bugs_surface_as_tracker_errors(video):
    plugin = StubDeep(backend=MarkerBackend(fail_on=(1,)))
    with pytest.raises(TrackerError, match="inference failed"):
        plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")


def test_a_backend_returning_the_wrong_shape_is_caught(video):
    class WrongShape:
        device = "cpu"

        def infer(self, frames, queries):
            length = frames.shape[0]
            return BackendPrediction(
                points=np.zeros((length + 1, queries.shape[0], 2), dtype=np.float32),
                visibility=np.ones((length + 1, queries.shape[0]), dtype=np.float32),
            )

        def to_device(self, device):
            self.device = device

        def release(self):
            pass

    plugin = StubDeep(backend=WrongShape())
    with pytest.raises(TrackerError, match="expected"):
        plugin.track(video, [anchor_at(video, 0)], (0, video.frame_count), "fwd")


def test_release_frees_the_backend_it_built():
    plugin = StubDeep()
    plugin.prepare("cpu")
    backend = plugin.made[0]
    plugin.release()
    assert backend.released is True
    assert not plugin.is_prepared


def test_release_leaves_an_injected_backend_alone():
    backend = MarkerBackend()
    plugin = StubDeep(backend=backend)
    plugin.prepare("cpu")
    plugin.release()
    assert backend.released is False  # the caller owns what the caller injected


# -- lazy, checksummed weights --------------------------------------------


def test_weights_are_fetched_on_prepare_not_before(cache, served):
    class Weighted(StubDeep):
        model_spec = served

    plugin = Weighted()
    assert Weighted.weights_downloaded() is False
    assert not plugin.is_prepared

    plugin.prepare("cpu")

    assert Weighted.weights_downloaded() is True
    assert plugin.made[0].weights.read_bytes() == PAYLOAD


def test_a_corrupt_checkpoint_stops_preparation(cache, served, tmp_path):
    class Weighted(StubDeep):
        model_spec = ModelSpec(key="stub_model", url=served.url, sha256="1" * 64)

    with pytest.raises(WeightsError, match="checksum mismatch"):
        Weighted().prepare("cpu")


def test_an_injected_backend_never_touches_the_weight_cache(cache):
    class Weighted(StubDeep):
        model_spec = ModelSpec(key="stub_model", url="https://example.invalid/x.pth")

    plugin = Weighted(backend=MarkerBackend())
    plugin.prepare("cpu")  # would raise if it tried to download
    assert plugin.is_prepared


# -- chunking, scaling and re-anchoring -----------------------------------


def test_frames_are_resized_and_coordinates_scaled_back(video):
    backend = EchoBackend()
    plugin = StubDeep(backend=backend, model_resolution=(96, 128), chunk_size=8, overlap=2)
    query = Query(0, 16.0, 12.0)

    result = plugin.track(video, [query], (0, 8), "fwd")

    assert backend.frame_sizes[0] == (96, 128)
    # 64x48 source -> 128x96 model space doubles both axes.
    assert backend.queries[0][0, 1] == pytest.approx(32.0)
    assert backend.queries[0][0, 2] == pytest.approx(24.0)
    # ...and the answer comes back in source pixels, not model pixels.
    assert result.points[0, 3] == pytest.approx((16.0, 12.0), abs=1e-3)


def test_native_resolution_skips_resizing(video):
    backend = EchoBackend()
    StubDeep(backend=backend, model_resolution=None).track(video, [Query(0, 5.0, 5.0)], (0, 4))
    assert backend.frame_sizes[0] == (video.height, video.width)


def test_windows_are_fed_in_run_order(video):
    class Recording(EchoBackend):
        def __init__(self):
            super().__init__()
            self.first_pixels = []

        def infer(self, frames, queries):
            self.first_pixels.append(frames[0].copy())
            return super().infer(frames, queries)

    backend = Recording()
    plugin = StubDeep(backend=backend, chunk_size=6, overlap=2)
    plugin.track(video, [anchor_at(video, 20)], (0, video.frame_count), "bwd")

    # A backward run must see the clip played backwards: window 0 starts on the
    # anchor frame, and each later window starts on an *earlier* frame.
    assert np.array_equal(backend.first_pixels[0], video.get_frame(20))
    assert np.array_equal(backend.first_pixels[1], video.get_frame(16))


def test_each_window_is_re_anchored_on_the_previous_one(video):
    backend = EchoBackend()
    plugin = StubDeep(backend=backend, chunk_size=6, overlap=2)
    plugin.track(video, [Query(0, 20.0, 15.0)], (0, video.frame_count), "fwd")

    assert len(backend.queries) > 1
    for query in backend.queries:
        assert query[0, 0] == 0.0, "a window's query always sits on its own first frame"
        # EchoBackend holds the point still, so every re-anchor must reproduce it.
        assert query[0, 1:3] == pytest.approx((20.0, 15.0))


def test_points_sharing_an_anchor_frame_are_batched(video):
    backend = MarkerBackend()
    plugin = StubDeep(backend=backend, chunk_size=48, overlap=0)
    queries = [Query(0, 20.0, 20.0), Query(0, 30.0, 15.0), Query(5, 25.0, 25.0)]

    plugin.track(video, queries, (0, video.frame_count), "fwd")

    # Two points on frame 0 go through together; the third has its own run.
    assert [q.shape[0] for q in backend.queries] == [2, 1]


def test_window_size_is_validated():
    with pytest.raises(ContractError, match="chunk_size"):
        StubDeep(chunk_size=1)
    with pytest.raises(ContractError, match="overlap"):
        StubDeep(chunk_size=8, overlap=8)
    with pytest.raises(ContractError, match="overlap"):
        StubDeep(chunk_size=8, overlap=-1)


class LazyHDSource:
    """A 1080p clip that generates frames on demand and counts live ones.

    Decoding a 30 s 1080p clip up front would be ~5.6 GB of uint8 before a
    single tensor exists, so the acceptance criterion "tracks without OOM on an
    8 GB GPU" is really a claim about never materialising more than one window.
    This source makes that claim measurable without a GPU.
    """

    def __init__(self, frame_count: int = 900) -> None:
        """Create a synthetic 1920x1080 clip of ``frame_count`` frames."""
        self._frame_count = frame_count
        self.handed_out = 0

    @property
    def frame_count(self) -> int:
        """Total frames."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width."""
        return 1920

    @property
    def height(self) -> int:
        """Frame height."""
        return 1080

    @property
    def fps(self) -> float:
        """Frame rate."""
        return 30.0

    def get_frame(self, index: int) -> np.ndarray:
        """Return a cheap 1080p frame."""
        self.handed_out += 1
        return np.full((1080, 1920, 3), index % 251, dtype=np.uint8)

    def iter_frames(self, start: int = 0, stop: int | None = None):
        """Yield frames in order."""
        for index in range(start, stop or self._frame_count):
            yield index, self.get_frame(index)


@pytest.mark.parametrize("plugin_class", [CoTracker3Tracker, TapirTracker])
def test_a_30s_1080p_clip_is_never_materialised_in_one_piece(plugin_class):
    """The bound behind "no OOM on an 8 GB GPU": one window in flight, downscaled."""

    class Sizing(EchoBackend):
        def __init__(self):
            super().__init__()
            self.window_bytes: list[int] = []

        def infer(self, frames, queries):
            self.window_bytes.append(frames.nbytes)
            return super().infer(frames, queries)

    backend = Sizing()
    plugin = plugin_class(backend=backend)  # default chunk size and resolution
    video = LazyHDSource(frame_count=900)  # 30 s at 30 fps

    result = plugin.track(video, [Query(0, 960.0, 540.0)], (0, 900), "fwd")

    assert np.isfinite(result.points[0]).all()
    assert len(backend.window_bytes) >= 900 // plugin.chunk_size

    height, width = plugin.model_resolution
    per_frame = height * width * 3
    assert max(backend.window_bytes) <= plugin.chunk_size * per_frame
    # The whole point: a window is megabytes, not gigabytes.
    assert max(backend.window_bytes) < 64 * 1024 * 1024
    assert max(backend.window_bytes) < 900 * 1080 * 1920 * 3 / 50


def test_resize_frames_round_trips_shape_and_dtype():
    frames = np.random.default_rng(0).integers(0, 255, (3, 20, 30, 3), dtype=np.uint8)
    out = resize_frames(frames, (10, 15))
    assert out.shape == (3, 10, 15, 3) and out.dtype == np.uint8
    assert resize_frames(frames, (20, 30)) is frames  # no-op when already correct


# -- CoTracker3 specifics -------------------------------------------------


def test_cotracker_accepts_two_and_three_headed_output():
    tracks = np.zeros((1, 4, 2, 2), dtype=np.float32)
    visibility = np.full((1, 4, 2), 0.8, dtype=np.float32)
    confidence = np.full((1, 4, 2), 0.5, dtype=np.float32)

    two = cotracker3_module._prediction_from_output((tracks, visibility))
    assert two.points.shape == (4, 2, 2) and two.confidence is None
    assert two.visibility == pytest.approx(0.8)

    three = cotracker3_module._prediction_from_output((tracks, visibility, confidence))
    assert three.confidence == pytest.approx(0.5)


def test_cotracker_rejects_output_it_does_not_understand():
    with pytest.raises(TrackerError, match="unexpected CoTracker output"):
        cotracker3_module._prediction_from_output({"tracks": np.zeros((1, 2, 1, 2))})


def test_cotracker_confidence_multiplies_its_two_heads():
    plugin = CoTracker3Tracker(backend=MarkerBackend(), confidence_range=(0.0, 1.0))
    prediction = BackendPrediction(
        points=np.zeros((1, 3, 2), dtype=np.float32),
        visibility=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        confidence=np.array([[1.0, 0.5, 1.0]], dtype=np.float32),
    )
    confidence, visible = plugin.normalize_confidence(prediction)
    assert confidence[0] == pytest.approx([1.0, 0.5, 0.0])
    assert visible[0].tolist() == [True, True, False]


def test_cotracker_calibration_has_dead_zones_at_both_ends():
    plugin = CoTracker3Tracker(backend=MarkerBackend())
    low, high = plugin.confidence_range
    raw = np.array([[0.0, low, (low + high) / 2, high, 1.0]], dtype=np.float32)
    prediction = BackendPrediction(points=np.zeros((1, 5, 2), dtype=np.float32), visibility=raw)
    confidence, _ = plugin.normalize_confidence(prediction)
    assert confidence[0, 0] == 0.0
    assert confidence[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert confidence[0, 2] == pytest.approx(0.5, abs=1e-6)
    assert confidence[0, 3] == pytest.approx(1.0) and confidence[0, 4] == 1.0


def test_cotracker_declares_its_non_commercial_licence():
    assert CoTracker3Tracker.license == "CC-BY-NC-4.0"
    assert CoTracker3Tracker.model_spec.non_commercial is True
    assert CoTracker3Tracker.default_model_resolution == (384, 512)


def test_cotracker_is_unavailable_without_torch(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert CoTracker3Tracker.is_available() is False


def test_cotracker_reports_a_missing_model_package(monkeypatch):
    monkeypatch.setattr("stable.trackers.cotracker3.torch_module", lambda: object())
    plugin = CoTracker3Tracker()
    with pytest.raises(TrackerError, match="cotracker package"):
        plugin._build_backend("cpu", "weights.pth")


def test_cotracker_reports_missing_torch(monkeypatch):
    monkeypatch.setattr("stable.trackers.cotracker3.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        CoTracker3Tracker()._build_backend("cpu", "weights.pth")


@pytest.mark.parametrize("plugin_class", [CoTracker3Tracker, TapirTracker, RaftTracker])
def test_a_missing_model_package_is_reported_before_any_download(plugin_class, cache, monkeypatch):
    """No user should wait for 400 MB only to learn a pip package is missing.

    The missing package is *simulated*: RAFT's lives in torchvision, which the
    `torch` extra now installs, so relying on it being absent would make this
    test pass or fail depending on the machine rather than on the code.
    """
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: object())
    monkeypatch.setattr(plugin_class, "_implementation_available", classmethod(lambda cls: False))
    plugin = plugin_class()
    with pytest.raises(TrackerError, match="model code"):
        plugin.prepare("cpu")
    assert "pip install" in plugin.install_hint
    assert plugin_class.weights_downloaded() is False


def test_missing_torch_is_reported_before_any_download(monkeypatch, cache):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        TapirTracker().prepare("cpu")


# -- TAPIR specifics ------------------------------------------------------


def test_tapir_converts_logits_and_transposes_to_frame_major():
    # TAPIR is point-major: [B, N, L, ...]; Stable's windows are frame-major.
    tracks = np.arange(2 * 3 * 2, dtype=np.float32).reshape(1, 2, 3, 2)
    occlusion = np.zeros((1, 2, 3), dtype=np.float32)  # sigmoid(0) = 0.5
    expected_dist = np.full((1, 2, 3), -np.inf, dtype=np.float32)  # sigmoid(-inf) = 0

    prediction = tapir_module._prediction_from_output(
        {"tracks": tracks, "occlusion": occlusion, "expected_dist": expected_dist}
    )

    assert prediction.points.shape == (3, 2, 2)
    assert prediction.points[1, 0] == pytest.approx(tracks[0, 0, 1])
    assert prediction.visibility == pytest.approx(0.5)
    assert prediction.confidence == pytest.approx(1.0)


def test_tapir_accepts_a_tuple_and_rejects_nonsense():
    tracks = np.zeros((1, 1, 2, 2), dtype=np.float32)
    logits = np.zeros((1, 1, 2), dtype=np.float32)
    assert tapir_module._prediction_from_output((tracks, logits, logits)).n_frames == 2

    with pytest.raises(TrackerError, match="missing key"):
        tapir_module._prediction_from_output({"tracks": tracks, "occlusion": logits})
    with pytest.raises(TrackerError, match="unexpected TAPIR output"):
        tapir_module._prediction_from_output("nope")


def test_tapir_confidence_follows_its_own_visibility_rule():
    plugin = TapirTracker(backend=MarkerBackend(), confidence_range=(0.0, 1.0))
    prediction = BackendPrediction(
        points=np.zeros((1, 3, 2), dtype=np.float32),
        visibility=np.array([[0.9, 0.9, 0.1]], dtype=np.float32),
        confidence=np.array([[0.9, 0.4, 0.9]], dtype=np.float32),
    )
    confidence, visible = plugin.normalize_confidence(prediction)
    assert confidence[0] == pytest.approx([0.81, 0.36, 0.09], abs=1e-6)
    assert visible[0].tolist() == [True, False, False]  # matches (1-occ)(1-dist) > 0.5


def test_tapir_declares_a_commercially_usable_licence():
    assert TapirTracker.license == "Apache-2.0"
    assert TapirTracker.model_spec.non_commercial is False
    assert TapirTracker.default_model_resolution == (256, 256)


def test_tapir_reports_a_missing_model_package(monkeypatch):
    monkeypatch.setattr("stable.trackers.tapir.torch_module", lambda: object())
    with pytest.raises(TrackerError, match="tapnet"):
        TapirTracker()._build_backend("cpu", "weights.pt")


def test_tapir_is_unavailable_without_torch(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert TapirTracker.is_available() is False


def test_availability_needs_both_torch_and_the_model_package(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: object())
    monkeypatch.setattr(TapirTracker, "_implementation_available", classmethod(lambda cls: True))
    assert TapirTracker.is_available() is True
    monkeypatch.setattr(TapirTracker, "_implementation_available", classmethod(lambda cls: False))
    assert TapirTracker.is_available() is False


# -- RAFT specifics ---------------------------------------------------------


def test_raft_declares_its_licence_and_defaults():
    assert RaftTracker.license == "BSD-3-Clause"
    assert RaftTracker.model_spec.non_commercial is False
    assert RaftTracker.default_model_resolution == (128, 128)


def test_raft_is_unavailable_without_torch(monkeypatch):
    monkeypatch.setattr("stable.trackers.deep.torch_module", lambda: None)
    assert RaftTracker.is_available() is False


def test_raft_reports_a_missing_model_package(monkeypatch):
    """Simulated, not assumed: torchvision is a declared dependency now.

    This used to pass only because torchvision happened not to be installed,
    so it started failing the moment the `torch` extra began pulling it in —
    the test was asserting a property of the machine, not of the code. Setting
    the module to None in sys.modules makes `import torchvision...` raise
    ImportError (documented CPython behaviour) whether or not it is present.
    """
    monkeypatch.setattr("stable.trackers.raft.torch_module", lambda: object())
    _hide_module(monkeypatch, "torchvision")
    with pytest.raises(TrackerError, match="torchvision"):
        RaftTracker()._build_backend("cpu", "weights.pt")


def test_raft_reports_missing_torch(monkeypatch):
    monkeypatch.setattr("stable.trackers.raft.torch_module", lambda: None)
    with pytest.raises(TrackerError, match="PyTorch"):
        RaftTracker()._build_backend("cpu", "weights.pt")


def test_raft_confidence_normalises_the_lone_flow_consistency_signal():
    # RAFT has no second head the way CoTracker3/TAPIR do; normalize_confidence
    # must still behave (combine_scores tolerates a single array).
    plugin = RaftTracker(backend=MarkerBackend(), confidence_range=(0.2, 0.8))
    prediction = BackendPrediction(
        points=np.zeros((1, 3, 2), dtype=np.float32),
        visibility=np.array([[0.9, 0.5, 0.1]], dtype=np.float32),
    )
    confidence, visible = plugin.normalize_confidence(prediction)
    assert confidence[0] == pytest.approx([1.0, 0.5, 0.0], abs=1e-6)
    # visible_threshold is "at or above" (see visibility_from_confidence), so
    # exactly 0.5 counts as visible.
    assert visible[0].tolist() == [True, True, False]


def test_pad_to_multiple_only_pads_when_it_must():
    import torch

    unpadded = torch.zeros(1, 3, 128, 128)
    assert _pad_to_multiple(unpadded, 8).shape == unpadded.shape

    ragged = torch.zeros(1, 3, 97, 131)
    padded = _pad_to_multiple(ragged, 8)
    assert padded.shape[-2] % 8 == 0
    assert padded.shape[-1] % 8 == 0
    assert padded.shape[-2] >= 97
    assert padded.shape[-1] >= 131


def test_sample_bilinear_interpolates_and_clamps_out_of_bounds_points():
    # A flow field that is a pure horizontal gradient: sampling at a half-pixel
    # x should average its two neighbouring columns exactly.
    flow = np.zeros((4, 4, 2), dtype=np.float64)
    flow[..., 0] = np.arange(4)[None, :]  # dx == column index
    samples = _sample_bilinear(flow, np.array([[1.5, 0.0], [-5.0, -5.0], [50.0, 50.0]]))
    assert samples[0, 0] == pytest.approx(1.5)
    # Out-of-bounds points are clamped into the field rather than raising.
    assert np.isfinite(samples).all()


# -- prediction contract --------------------------------------------------


def test_prediction_validates_its_shapes():
    points = np.zeros((4, 2, 2), dtype=np.float32)
    with pytest.raises(ContractError, match=r"\[L, N, 2\]"):
        BackendPrediction(points=np.zeros((4, 2)), visibility=np.zeros((4, 2)))
    with pytest.raises(ContractError, match="visibility"):
        BackendPrediction(points=points, visibility=np.zeros((4, 3)))
    with pytest.raises(ContractError, match="confidence"):
        BackendPrediction(points=points, visibility=np.zeros((4, 2)), confidence=np.zeros((3, 2)))
    prediction = BackendPrediction(points=points, visibility=np.zeros((4, 2)))
    assert (prediction.n_frames, prediction.n_points) == (4, 2)


def test_backends_satisfy_the_protocol():
    assert isinstance(MarkerBackend(), TrackerBackend)
    assert isinstance(EchoBackend(), TrackerBackend)
