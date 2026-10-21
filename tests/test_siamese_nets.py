"""The Ocean and SiamDW networks as ported from TracKit: shapes, layout, loading.

Needs PyTorch, so the whole module is skipped without it; everything that does
not is in ``tests/test_siamese_trackers.py``. The layout counts below were taken
from the real published checkpoints: they are what proves a download will load
into these networks without the download being part of the test run.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable.core.errors import TrackerError  # noqa: E402
from stable.trackers import siamese_nets  # noqa: E402
from stable.trackers.ocean import OceanTracker  # noqa: E402
from stable.trackers.siamdw import SiamDWTracker  # noqa: E402
from stable.trackers.siamese import TorchSiameseBackend  # noqa: E402


@pytest.mark.parametrize("size", [255, 271])
def test_ocean_maps_have_the_size_the_plugin_expects(size):
    network = siamese_nets.OceanNet().eval()
    with torch.inference_mode():
        kernel = network.embed(torch.zeros(2, 3, 127, 127))
        cls, reg, similarity = network.match(kernel, torch.zeros(2, 1, 3, size, size))
    cells = OceanTracker().score_size(size)
    assert tuple(kernel.shape) == (2, 256, 7, 7)
    assert tuple(cls.shape) == (2, 1, cells, cells)
    assert tuple(reg.shape) == (2, 1, 4, cells, cells)
    assert tuple(similarity.shape) == (2, 1, cells, cells)
    assert (reg > 0).all(), "edge distances come out of an exp"


@pytest.mark.parametrize("size", [255, 287])
def test_siamdw_maps_have_the_size_the_plugin_expects(size):
    network = siamese_nets.SiamDWNet().eval()
    with torch.inference_mode():
        kernel = network.embed(torch.zeros(2, 3, 127, 127))
        response, boxes, similarity = network.match(kernel, torch.zeros(2, 3, 3, size, size))
    cells = SiamDWTracker().score_size(size)
    assert tuple(kernel.shape) == (2, 512, 5, 5)
    assert tuple(response.shape) == (2, 3, cells, cells)
    assert boxes is None and similarity is None


@pytest.mark.parametrize(
    ("network", "parameters", "tensors"),
    # (parameters, tensors other than BatchNorm's num_batches_tracked) in the
    # published Ocean.pth and SiamDW.pth respectively.
    [("OceanNet", 25_869_136, 336), ("SiamDWNet", 2_458_048, 120)],
)
def test_the_networks_have_the_published_checkpoints_layout(network, parameters, tensors):
    built = getattr(siamese_nets, network)()
    assert sum(p.numel() for p in built.parameters()) == parameters
    named = [k for k in built.state_dict() if not k.endswith("num_batches_tracked")]
    assert len(named) == tensors


def test_published_spellings_are_renamed_onto_the_networks():
    renamed = siamese_nets.normalize_state_dict(
        {"state_dict": {"module.features.features.conv1.weight": 1, "head.bias": 2}}
    )
    assert renamed == {"backbone.conv1.weight": 1, "head.bias": 2}


def test_a_checkpoint_missing_only_batch_counters_loads(tmp_path):
    """SiamDW's release predates num_batches_tracked and spells the backbone the old way."""
    source = siamese_nets.SiamDWNet()
    state = {
        key.replace("backbone.", "features.features."): value
        for key, value in source.state_dict().items()
        if not key.endswith("num_batches_tracked")
    }
    path = tmp_path / "siamdw.pth"
    torch.save(state, path)
    loaded = siamese_nets.load_checkpoint(siamese_nets.SiamDWNet(), path)
    assert torch.equal(loaded.backbone.conv1.weight, source.backbone.conv1.weight)
    assert not loaded.training


def test_a_checkpoint_for_another_network_is_refused(tmp_path):
    path = tmp_path / "wrong.pth"
    torch.save(siamese_nets.SiamDWNet().state_dict(), path)
    with pytest.raises(TrackerError, match="does not match"):
        siamese_nets.load_checkpoint(siamese_nets.OceanNet(), path)


def test_each_search_is_correlated_with_its_own_template():
    search = torch.randn(2, 4, 9, 9)
    kernel = torch.randn(2, 4, 3, 3)
    batched = siamese_nets.xcorr_batched(search, kernel)
    for i in range(2):
        alone = torch.nn.functional.conv2d(search[i : i + 1], kernel[i : i + 1])[0, 0]
        assert torch.allclose(batched[i], alone, atol=1e-5)


def test_the_cosine_map_is_a_cosine():
    search = torch.randn(1, 4, 9, 9)
    kernel = search[:, :, 2:5, 3:6].clone()
    cosine = siamese_nets.cosine_map(search, kernel)
    assert cosine[0, 2, 3].item() == pytest.approx(1.0, abs=1e-5)
    assert cosine.abs().max().item() <= 1.0 + 1e-5


def test_cpu_calls_flush_denormals_and_put_the_mode_back():
    """SiamDW's weights are full of denormals: unflushed, a CPU frame took 77x longer."""
    from stable.trackers.siamese import _denormals_flushed

    supported = torch.set_flush_denormal(True)
    torch.set_flush_denormal(False)
    if not supported:
        pytest.skip("this CPU cannot flush denormals")
    tiny = torch.tensor([1e-40])
    with _denormals_flushed(torch, "cpu"):
        assert (tiny * 1.0).item() == 0.0
    assert (tiny * 1.0).item() != 0.0
    with _denormals_flushed(torch, "cuda"):
        assert (tiny * 1.0).item() != 0.0, "a GPU call leaves the CPU mode alone"


def test_the_backend_feeds_bgr_in_0_255():
    """TracKit's networks were trained on cv2.imread output: BGR, unnormalised."""

    class Recorder:
        seen = None

        def embed(self, tensor):
            Recorder.seen = tensor
            return tensor

        def match(self, kernel, search):
            return torch.zeros(search.shape[0], search.shape[1], 1, 1), None, None

        def to(self, device):
            return self

    backend = TorchSiameseBackend(Recorder(), "cpu")
    crop = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    crop[..., 0] = 200  # red, in Stable's RGB
    backend.embed(crop)
    assert tuple(Recorder.seen.shape) == (1, 3, 4, 4)
    assert Recorder.seen[0, 2].max().item() == 200.0  # red is the last channel in BGR
    assert Recorder.seen[0, 0].max().item() == 0.0
    maps = backend.match(None, np.zeros((2, 3, 4, 4, 3), dtype=np.uint8))
    assert maps.score.shape == (2, 3, 1, 1)
