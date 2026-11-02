"""The HiT network as ported from the authors' release: shapes, layout, loading.

Needs PyTorch, so the whole module is skipped without it; everything that does
not is in ``tests/test_hit_tracker.py``. The layout counts below were taken from
the three published checkpoints: they are what proves a download will load into
these networks strictly without the download being part of the test run. The
last test loads the real checkpoints when they happen to be cached.
"""

from __future__ import annotations

import pickle
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable.core.errors import TrackerError  # noqa: E402
from stable.trackers import hit_net  # noqa: E402
from stable.trackers.hit import HIT_MODELS, HiTBackend, HiTMaps  # noqa: E402
from stable.trackers.hit_net import HIT_CONFIGS, HiTConfig, HiTNet  # noqa: E402
from stable.trackers.weights import local_path  # noqa: E402

SMALL = HiTConfig(embed_dim=(32, 48, 64), key_dim=8, num_heads=(2, 3, 4), depth=(1, 1, 1))
"""A HiT a fraction of the published size: the same geometry, fast to build and run."""


def _randomised(network):
    """Give every batch norm non-trivial statistics, so folding them is really tested."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for module in network.modules():
            if isinstance(module, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d):
                size = module.num_features
                module.running_mean.copy_(torch.randn(size, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(size, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(size, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(size, generator=generator) * 0.1)
        for name, parameter in network.named_parameters():
            if name.endswith("attention_biases"):
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return network.eval()


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(1)
    return (
        torch.randn(batch, 3, 256, 256, generator=generator),
        torch.randn(batch, 3, 128, 128, generator=generator),
    )


@pytest.mark.parametrize(
    ("variant", "parameters", "tensors", "tables"),
    # (parameters, state_dict tensors, position-encoding tables) in the published
    # HiT_Base, HiT_Small and HiT_Tiny checkpoints.
    [("base", 42_220_250, 478, 14), ("small", 11_069_794, 478, 14), ("tiny", 9_628_530, 400, 11)],
)
def test_the_variants_have_the_published_checkpoints_layout(variant, parameters, tensors, tables):
    network = HiTNet(variant)
    state = network.state_dict()
    assert sum(p.numel() for p in network.parameters()) == parameters
    assert len(state) == tensors
    assert sum(k.endswith("attention_bias_idxs") for k in state) == tables


def test_the_dual_image_position_encoding_has_the_published_tables():
    """Shapes from HiT_Base: search and template tokens share one offset table per layer."""
    state = HiTNet("base").state_dict()
    expected = {
        "backbone.body.blocks.0.m.attention_biases": (6, 560),
        "backbone.body.blocks.0.m.attention_bias_idxs": (320, 320),
        "backbone.body.blocks.8.attention_biases": (12, 558),
        "backbone.body.blocks.8.attention_bias_idxs": (80, 320),
        "backbone.body.blocks.10.m.attention_biases": (9, 136),
        "backbone.body.blocks.18.attention_bias_idxs": (20, 80),
        "backbone.body.blocks.26.m.attention_biases": (12, 32),
    }
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape, key


def test_the_network_answers_in_the_shapes_the_plugin_reads():
    network = _randomised(HiTNet(SMALL))
    search, template = _inputs()
    with torch.inference_mode():
        kernel = network.embed(template)
        boxes, top_left, bottom_right, similarity = network.match(kernel, search)
    assert tuple(kernel.shape) == (2, 64, 32)
    assert tuple(boxes.shape) == (2, 4)
    assert tuple(top_left.shape) == tuple(bottom_right.shape) == (2, 16, 16)
    assert tuple(similarity.shape) == (2, 16, 16)
    assert similarity.abs().max() <= 1.0 + 1e-5
    assert torch.allclose(network(search, template), boxes)


def test_folding_the_batch_norms_changes_nothing():
    network = _randomised(HiTNet(SMALL))
    search, template = _inputs()
    with torch.inference_mode():
        before = network.match(network.embed(template), search)
        network.fuse()
        after = network.match(network.embed(template), search)
    assert not any(isinstance(m, hit_net.LinearBN | hit_net.ConvBN) for m in network.modules())
    for a, b in zip(before, after, strict=True):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_appearance_peaks_where_the_template_is():
    network = HiTNet(SMALL)
    generator = torch.Generator().manual_seed(2)
    tokens = torch.randn(1, 256 + 64, 8, generator=generator)
    search = tokens[0, :256].view(16, 16, 8)
    template = tokens[0, 256:].view(8, 8, 8)
    # Put the template's central 4x4 at search rows 5..8 and columns 9..12.
    search[5:9, 9:13] = template[2:6, 2:6]
    similarity = network.appearance(tokens)[0]
    row, col = divmod(int(similarity.argmax()), 16)
    # Cell (i, j) is the window of tokens i-2..i+1 and j-2..j+1.
    assert (row, col) == (7, 11)
    assert float(similarity[row, col]) == pytest.approx(1.0, abs=1e-5)


def test_a_checkpoint_round_trips_strictly(tmp_path):
    source = _randomised(HiTNet(SMALL))
    path = tmp_path / "hit.pth.tar"
    torch.save({"net": source.state_dict(), "epoch": 1500}, path)
    loaded = hit_net.load_checkpoint(HiTNet(SMALL), path)
    assert not loaded.training
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_a_checkpoint_of_another_shape_is_refused(tmp_path):
    path = tmp_path / "hit.pth.tar"
    torch.save({"net": HiTNet(SMALL).state_dict()}, path)
    with pytest.raises(TrackerError, match="does not match"):
        hit_net.load_checkpoint(HiTNet("tiny"), path)


def test_a_checkpoint_numbering_the_position_offsets_differently_is_refused(tmp_path):
    state = HiTNet(SMALL).state_dict()
    key = "backbone.body.blocks.0.m.attention_bias_idxs"
    state[key] = state[key].flip(0)
    path = tmp_path / "hit.pth.tar"
    torch.save({"net": state}, path)
    with pytest.raises(TrackerError, match="position-encoding"):
        hit_net.load_checkpoint(HiTNet(SMALL), path)


def _fake_training_module(monkeypatch, name: str, cls_name: str):
    """Define ``name.cls_name`` the way the authors' harness would, for pickling."""
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        package = ".".join(parts[:depth])
        if package not in sys.modules:
            monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    cls = type(cls_name, (), {"__module__": name})
    monkeypatch.setattr(sys.modules[name], cls_name, cls, raising=False)
    return cls


def test_the_authors_training_checkpoints_load_without_their_training_code(tmp_path, monkeypatch):
    """The real files pickle lib.train.admin objects beside the weights."""
    settings_cls = _fake_training_module(monkeypatch, "lib.train.admin.settings", "Settings")
    settings = settings_cls()
    settings.workspace_dir = "/home/somebody"
    state = HiTNet(SMALL).state_dict()
    path = tmp_path / "VT_ep1500.pth.tar"
    torch.save({"net": state, "settings": settings, "optimizer": {"state": {}}}, path)
    for package in ("lib.train.admin.settings", "lib.train.admin", "lib.train", "lib"):
        monkeypatch.delitem(sys.modules, package)

    loaded = hit_net.read_release_state_dict(path)

    assert set(loaded) == set(state)
    assert "lib" not in sys.modules, "the training code was never imported"


def test_a_checkpoint_naming_anything_else_is_refused(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ("this must never run",))

    path = tmp_path / "evil.pth.tar"
    torch.save({"net": {}, "payload": Payload()}, path)
    with pytest.raises(TrackerError, match="unexpected global"):
        hit_net.read_release_state_dict(path)


def test_a_file_that_is_no_checkpoint_is_refused(tmp_path):
    path = tmp_path / "notes.pth.tar"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(TrackerError):
        hit_net.read_release_state_dict(path)


def test_the_backend_feeds_hit_its_own_input_sizes(monkeypatch):
    seen: dict[str, tuple[int, ...]] = {}
    network = _randomised(HiTNet(SMALL))
    embed, match = network.embed, network.match
    monkeypatch.setattr(
        network, "embed", lambda t: seen.setdefault("template", tuple(t.shape)) and embed(t)
    )
    monkeypatch.setattr(
        network, "match", lambda k, s: seen.setdefault("search", tuple(s.shape)) and match(k, s)
    )
    backend = HiTBackend(network, "cpu")
    exemplars = np.random.default_rng(0).integers(0, 255, (2, 127, 127, 3), dtype=np.uint8)
    searches = np.random.default_rng(1).integers(0, 255, (2, 1, 319, 319, 3), dtype=np.uint8)
    maps = backend.match(backend.embed(exemplars), searches)
    assert seen == {"template": (2, 3, 128, 128), "search": (2, 3, 256, 256)}
    assert isinstance(maps, HiTMaps)
    assert maps.score.shape == maps.bottom_right.shape == maps.similarity.shape == (2, 1, 16, 16)
    assert (maps.crop_size, maps.network_size) == (319, 256)


@pytest.mark.parametrize("variant", sorted(HIT_CONFIGS))
def test_the_published_checkpoint_loads_strictly_when_it_is_cached(variant):
    path = local_path(HIT_MODELS[variant])
    if not path.is_file():
        pytest.skip(f"{path.name} is not downloaded")
    network = hit_net.load_checkpoint(HiTNet(variant), path).fuse()
    search, template = _inputs(1)
    with torch.inference_mode():
        boxes, *_ = network.match(network.embed(template), search)
    assert torch.isfinite(boxes).all()
