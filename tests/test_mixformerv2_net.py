"""The MixFormerV2 network as ported from MCG-NJU/MixFormerV2: shapes, attention, loading.

Needs PyTorch, so the whole module is skipped without it; the plugin is
``tests/test_mixformerv2_tracker.py``. The layout counts below were taken from
the real published checkpoints, and :func:`test_the_published_checkpoint_loads_strictly`
loads the real file when it is in the model cache (it is a 240 MB download, so
it is skipped otherwise). Numerical parity with the authors' own module
definitions was measured when the checkpoints were pinned: see the module
docstring of :mod:`stable.trackers.mixformerv2`.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable.core.errors import TrackerError  # noqa: E402
from stable.trackers import mixformerv2_net as net  # noqa: E402
from stable.trackers.mixformerv2 import (  # noqa: E402
    MIXFORMERV2_BASE_WEIGHTS,
    MIXFORMERV2_SMALL_WEIGHTS,
    MixFormerV2Backend,
)
from stable.trackers.weights import local_path  # noqa: E402

TINY = net.MixFormerV2Config(
    name="tiny",
    depth=2,
    mlp_ratio=2.0,
    template_size=32,
    search_size=64,
    embed_dim=24,
    num_heads=3,
    bins=16,
)


def tiny_network(seed: int = 0):
    torch.manual_seed(seed)
    network = net.MixFormerV2Net(TINY).eval()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0.0, 0.2)
    return network


def joint_forward(network, template, online, search):
    """MixFormer's asymmetric attention written the obvious way: one sequence, one mask."""
    backbone = network.backbone
    t = backbone.patch_embed(template) + backbone.pos_embed_t
    o = backbone.patch_embed(online) + backbone.pos_embed_t
    s = backbone.patch_embed(search) + backbone.pos_embed_s
    tokens = (backbone.reg_tokens + backbone.pos_embed_reg).expand(s.shape[0], -1, -1)
    x = torch.cat([t, o, s, tokens], dim=1)
    n_templates = t.shape[1] + o.shape[1]
    allowed = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool)
    allowed[:n_templates, n_templates:] = False  # templates never look at the search
    for block in backbone.blocks:
        attn = block.attn
        h = block.norm1(x)
        b, n, c = h.shape
        qkv = attn.qkv(h).reshape(b, n, 3, attn.num_heads, c // attn.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        weights = (q @ k.transpose(-2, -1)) * (c // attn.num_heads) ** -0.5
        weights = weights.masked_fill(~allowed, float("-inf")).softmax(-1)
        x = x + attn.proj((weights @ v).transpose(1, 2).reshape(b, n, c))
        x = x + block.mlp(block.norm2(x))
    tokens = x[:, -4:]
    boxes, _ = network.box_head(tokens)
    return boxes, network.score_head(tokens)


def test_the_split_forward_is_exactly_the_joint_one():
    """Running the templates once and the search per frame changes nothing."""
    network = tiny_network()
    template, online = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    search = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        split = network.match(network.embed(template, online), search)
        boxes, score = joint_forward(network, template, online, search)
    assert torch.allclose(split.boxes, boxes, atol=1e-5)
    assert torch.allclose(split.score, score, atol=1e-5)


def test_the_templates_never_see_the_search():
    network = tiny_network()
    template = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        first = network.embed(template)
        again = network.embed(template)
    for (k1, v1), (k2, v2) in zip(first, again, strict=True):
        assert torch.equal(k1, k2) and torch.equal(v1, v2)
    assert len(first) == TINY.depth
    assert tuple(first[0][0].shape) == (1, TINY.num_heads, 2 * TINY.template_cells**2, 8)


def test_one_template_defaults_to_being_both():
    """Upstream's state until its first online update: online template = template."""
    network = tiny_network()
    template, search = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        alone = network.match(network.embed(template), search)
        both = network(template, template, search)
    assert torch.equal(alone.boxes, both.boxes)


def test_each_search_is_matched_with_its_own_template():
    network = tiny_network()
    template, search = torch.randn(3, 3, 32, 32), torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        batched = network(template, template, search)
        for i in range(3):
            single = network(template[i : i + 1], template[i : i + 1], search[i : i + 1])
            assert torch.allclose(batched.boxes[i], single.boxes[0], atol=1e-6)
            assert torch.allclose(batched.score[i], single.score[0], atol=1e-6)


def test_outputs_have_the_documented_shapes_and_ranges():
    network = tiny_network()
    with torch.no_grad():
        out = network(
            torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64)
        )
    assert tuple(out.boxes.shape) == (2, 4)
    assert tuple(out.score.shape) == (2,)
    assert tuple(out.spread.shape) == (2, 4)
    assert (out.boxes >= 0).all() and (out.boxes <= (TINY.bins - 1) / TINY.bins).all()
    assert (out.spread >= 0).all()


def test_the_box_head_reads_the_tokens_in_upstreams_order():
    """Tokens are (left, right, top, bottom); the box comes out (x1, y1, x2, y2)."""

    class OneHot(torch.nn.Module):
        def forward(self, tokens):
            return torch.nn.functional.one_hot((tokens[..., 0] * 2).long(), 10).float() * 50.0

    head = net.BoxHead(4, 10)
    head.layers = OneHot()
    tokens = torch.arange(4.0).view(1, 4, 1).expand(1, 4, 4)  # token i holds the value i
    boxes, spread = head(tokens)
    # left -> bin 0, right -> bin 2, top -> bin 4, bottom -> bin 6
    assert boxes[0].tolist() == pytest.approx([0.0, 0.4, 0.2, 0.6], abs=1e-6)
    assert spread[0].tolist() == pytest.approx([0.0] * 4, abs=1e-6)


def test_the_score_head_averages_its_four_tokens():

    class First(torch.nn.Module):
        def forward(self, tokens):
            return tokens[..., :1]

    head = net.ScoreHead(4)
    head.layers = First()
    tokens = torch.tensor([[[1.0] * 4, [2.0] * 4, [3.0] * 4, [6.0] * 4]])
    assert head(tokens).tolist() == pytest.approx([3.0])


@pytest.mark.parametrize(
    ("config", "parameters", "tensors"),
    # Taken from the published mixformerv2_base.pth.tar / mixformerv2_small.pth.tar.
    [(net.MIXFORMERV2_BASE, 58_855_201, 114), (net.MIXFORMERV2_SMALL, 16_228_897, 66)],
)
def test_the_networks_have_the_published_checkpoints_layout(config, parameters, tensors):
    built = net.MixFormerV2Net(config)
    state = built.state_dict()
    assert len(state) == tensors
    assert sum(t.numel() for t in state.values()) == parameters
    assert tuple(state["backbone.pos_embed_s"].shape) == (1, config.search_cells**2, 768)
    assert tuple(state["box_head.layers.1.0.weight"].shape) == (96, 768)


def _pytracking_checkpoint(tmp_path, monkeypatch, state, foreign="lib.train.admin.stats"):
    """Save ``state`` the way MixFormerV2's trainer did: beside pickled trainer objects.

    The trainer's class only exists while the file is written, exactly as for a
    user loading the real checkpoint without the authors' code installed.
    """
    parts = foreign.split(".")
    names = [".".join(parts[: i + 1]) for i in range(len(parts))]
    modules = {name: types.ModuleType(name) for name in names}

    class AverageMeter:
        def __init__(self):
            self.avg, self.count = 0.5, 3

    AverageMeter.__module__ = foreign
    AverageMeter.__qualname__ = "AverageMeter"
    modules[foreign].AverageMeter = AverageMeter
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "checkpoint.pth.tar"
    torch.save({"epoch": 40, "net": state, "stats": {"train": AverageMeter()}}, path)
    for name in names:
        monkeypatch.delitem(sys.modules, name)
    return path


def test_a_training_checkpoint_loads_without_running_its_pickled_objects(tmp_path, monkeypatch):
    source = net.MixFormerV2Net(TINY)
    state = {f"module.{key}": value for key, value in source.state_dict().items()}
    path = _pytracking_checkpoint(tmp_path, monkeypatch, state)
    loaded = net.load_checkpoint(net.MixFormerV2Net(TINY), path)
    assert torch.equal(loaded.box_head.layers[1][0].weight, source.box_head.layers[1][0].weight)
    assert not loaded.training


def test_a_checkpoint_pickling_anything_else_is_refused(tmp_path, monkeypatch):
    path = _pytracking_checkpoint(
        tmp_path, monkeypatch, net.MixFormerV2Net(TINY).state_dict(), foreign="evil.payload"
    )
    with pytest.raises(TrackerError, match="could not read"):
        net.read_state_dict(path)


def test_a_checkpoint_for_the_other_variant_is_refused(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"net": net.MixFormerV2Net(replace(TINY, depth=3)).state_dict()}, path)
    with pytest.raises(TrackerError, match="does not match"):
        net.load_checkpoint(net.MixFormerV2Net(TINY), path)
    torch.save({"net": net.MixFormerV2Net(replace(TINY, bins=8)).state_dict()}, path)
    with pytest.raises(TrackerError, match="another shape"):
        net.load_checkpoint(net.MixFormerV2Net(TINY), path)


def test_a_file_without_a_state_dict_is_refused(tmp_path):
    path = tmp_path / "list.pth"
    torch.save([1, 2, 3], path)
    with pytest.raises(TrackerError, match="no MixFormerV2 state dict"):
        net.read_state_dict(path)


@pytest.mark.parametrize(
    ("spec", "config"),
    [
        (MIXFORMERV2_BASE_WEIGHTS, net.MIXFORMERV2_BASE),
        (MIXFORMERV2_SMALL_WEIGHTS, net.MIXFORMERV2_SMALL),
    ],
)
def test_the_published_checkpoint_loads_strictly(spec, config):
    """Every tensor in the real file has a home in the port, and vice versa."""
    path = local_path(spec)
    if not path.is_file():
        pytest.skip(f"{spec.key} is not in the model cache ({path})")
    network = net.load_checkpoint(net.MixFormerV2Net(config), path)
    with torch.inference_mode():
        out = network(
            torch.zeros(1, 3, config.template_size, config.template_size),
            torch.zeros(1, 3, config.template_size, config.template_size),
            torch.zeros(1, 3, config.search_size, config.search_size),
        )
    assert torch.isfinite(out.boxes).all() and torch.isfinite(out.score).all()


# -- the backend's conventions ---------------------------------------------------------


class Recorder:
    """A stand-in network that records what the backend feeds it."""

    config = TINY

    def __init__(self, box=(0.25, 0.5, 0.75, 0.625)):
        self.box = box
        self.templates = []
        self.searches = []
        self.memories = []

    def embed(self, template, online=None):
        self.templates.append(template)
        return [(template.new_zeros(template.shape[0], 1, 1, 1),) * 2]

    def match(self, memory, search):
        self.memories.append(memory)
        self.searches.append(search)
        batch = search.shape[0]
        return net.MixFormerV2Output(
            boxes=torch.tensor([self.box] * batch),
            score=torch.full((batch,), 2.0),
            spread=torch.full((batch, 4), 0.01),
        )

    def to(self, device):
        return self


def test_crops_are_resampled_to_the_networks_sizes_and_normalised_in_rgb():
    recorder = Recorder()
    backend = MixFormerV2Backend(recorder, "cpu")
    exemplar = np.zeros((1, 127, 127, 3), dtype=np.uint8)
    exemplar[..., 0] = 255  # pure red in Stable's RGB
    backend.embed(exemplar)
    template = recorder.templates[0]
    assert tuple(template.shape) == (1, 3, TINY.template_size, TINY.template_size)
    mean, std = net.IMAGENET_MEAN, net.IMAGENET_STD
    assert template[0, 0].mean().item() == pytest.approx((1.0 - mean[0]) / std[0], abs=1e-4)
    assert template[0, 2].mean().item() == pytest.approx((0.0 - mean[2]) / std[2], abs=1e-4)
    backend.match(None, np.zeros((2, 1, 271, 271, 3), dtype=np.uint8))
    assert tuple(recorder.searches[0].shape) == (2, 3, TINY.search_size, TINY.search_size)


def test_resampling_keeps_the_crop_centre_where_it_was():
    """A ramp through the crop centre still has its middle value at the network's centre."""
    recorder = Recorder()
    backend = MixFormerV2Backend(recorder, "cpu")
    ramp = np.tile(np.linspace(0, 254, 127).astype(np.uint8)[None, :, None], (127, 1, 3))
    backend.embed(ramp[None])
    row = recorder.templates[0][0, 0, TINY.template_size // 2]
    middle = (row[TINY.template_size // 2 - 1] + row[TINY.template_size // 2]) / 2
    assert middle.item() == pytest.approx(
        (127 / 255 - net.IMAGENET_MEAN[0]) / net.IMAGENET_STD[0], abs=0.02
    )


def test_the_box_comes_back_as_edge_distances_from_the_crop_centre_in_crop_pixels():
    backend = MixFormerV2Backend(Recorder(box=(0.25, 0.5, 0.75, 0.625)), "cpu")
    maps = backend.match(None, np.zeros((1, 1, 271, 271, 3), dtype=np.uint8))
    left, top, right, bottom = maps.boxes[0, 0, :, 0, 0]
    assert (left, top, right, bottom) == pytest.approx((0.25 * 271, 0.0, 0.25 * 271, 0.125 * 271))
    assert maps.score[0, 0, 0, 0] == pytest.approx(2.0)
    assert maps.spread[0, 0, 0, 0] == pytest.approx(0.01 * 271)


def test_several_search_scales_share_their_template():
    recorder = Recorder()
    backend = MixFormerV2Backend(recorder, "cpu")
    kernels = backend.embed(np.zeros((2, 127, 127, 3), dtype=np.uint8))
    maps = backend.match(kernels, np.zeros((2, 3, 255, 255, 3), dtype=np.uint8))
    assert maps.score.shape == (2, 3, 1, 1)
    assert recorder.memories[0][0][0].shape[0] == 6
