"""SUTrack's network as ported to plain PyTorch: shapes, layout, the language token, loading.

Needs PyTorch, so the whole module is skipped without it; everything that does
not is in ``tests/test_sutrack_tracker.py``. The layout counts below were taken
from the real published checkpoint, so they prove a download will load into this
network without the 1 GB download being part of the test run; the one test that
does use it skips unless the checkpoint is already cached.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable.core.errors import TrackerError  # noqa: E402
from stable.trackers import sutrack_net  # noqa: E402
from stable.trackers.sutrack import (  # noqa: E402
    SUTRACK_T224_WEIGHTS,
    SUTrackBackend,
    SUTrackMaps,
    network_scale,
    template_box,
)
from stable.trackers.sutrack_net import SUTrackConfig  # noqa: E402
from stable.trackers.weights import is_downloaded, local_path  # noqa: E402

TINY = SUTrackConfig(
    name="tiny",
    embed_dim=32,
    depth_stage1=1,
    depth_stage2=1,
    depth=2,
    num_heads=2,
    template_size=32,
    search_size=64,
    head_channels=16,
    text_width=8,
)
TINY_TWO_TEMPLATES = SUTrackConfig(**{**TINY.__dict__, "name": "tiny2", "num_templates": 2})


def _inputs(config, batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    count = config.num_templates
    templates = torch.randn(
        batch, count, 3, config.template_size, config.template_size, generator=generator
    )
    boxes = torch.tensor([0.25, 0.3, 0.5, 0.4]).expand(batch, count, 4)
    searches = torch.randn(batch, 3, config.search_size, config.search_size, generator=generator)
    return templates, boxes, searches


def _randomised(network, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for value in network.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
        for module in network.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_var.abs_().add_(0.5)
    return network.eval()


# -- shapes and batching ---------------------------------------------------------


@pytest.mark.parametrize("config", [TINY, TINY_TWO_TEMPLATES])
def test_the_head_returns_one_score_size_and_offset_map_per_search(config):
    network = _randomised(sutrack_net.build_sutrack(config))
    with torch.inference_mode():
        score, size, offset = network(*_inputs(config, 3))
    cells = config.search_size // 16
    assert tuple(score.shape) == (3, 1, cells, cells)
    assert tuple(size.shape) == (3, 2, cells, cells)
    assert tuple(offset.shape) == (3, 2, cells, cells)
    for squashed in (score, size):
        assert squashed.min() >= 1e-4 and squashed.max() <= 1 - 1e-4


@pytest.mark.parametrize("config", [TINY, TINY_TWO_TEMPLATES])
def test_each_search_is_scored_against_its_own_templates(config):
    network = _randomised(sutrack_net.build_sutrack(config))
    templates, boxes, searches = _inputs(config, 3)
    with torch.inference_mode():
        together = network(templates, boxes, searches)
        for i in range(3):
            alone = network(templates[i : i + 1], boxes[i : i + 1], searches[i : i + 1])
            for joint, single in zip(together, alone, strict=True):
                assert torch.allclose(joint[i : i + 1], single, atol=1e-5)


def test_template_tokens_are_typed_by_how_much_of_their_patch_the_target_covers():
    """Upstream's create_mask: pixel indices in [x0, x0 + w), averaged over each 16 px patch."""
    body = sutrack_net.build_sutrack(TINY).encoder.body
    with torch.no_grad():
        body.template_foreground_token.fill_(1.0)
        body.template_background_token.fill_(0.0)
        quarter = body.template_indication(torch.tensor([[0.25, 0.25, 0.5, 0.5]]), 32)
        corner = body.template_indication(torch.tensor([[0.0, 0.0, 0.5, 0.5]]), 32)
    # 32 px template, 2x2 tokens: a centred 16 px box covers a quarter of each.
    assert quarter[0, :, 0].tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert corner[0, :, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_a_search_of_another_size_is_refused():
    network = sutrack_net.build_sutrack(TINY)
    templates, boxes, _ = _inputs(TINY, 1)
    with pytest.raises(TrackerError, match="64 px search"):
        network(templates, boxes, torch.zeros(1, 3, 80, 80))


@pytest.mark.parametrize(
    ("config", "tensors", "parameters"),
    # Counted from the published SUTRACK_ep0180.pth.tar of each variant, after
    # the keys load_sutrack drops (CLIP, the unused text norms, the task head).
    [(sutrack_net.SUTRACK_T224, 377, 27_975_141), (sutrack_net.SUTRACK_B224, 661, 89_443_845)],
)
def test_the_network_has_the_published_checkpoints_layout(config, tensors, parameters):
    network = sutrack_net.build_sutrack(config)
    assert len(network.state_dict()) == tensors
    assert sum(p.numel() for p in network.parameters()) == parameters
    assert all(not key.startswith(sutrack_net.DROPPED_PREFIXES) for key in network.state_dict())


# -- the language token ------------------------------------------------------------


def _clip_text_tower(width=8, layers=2, context=5, vocabulary=11, seed=3):
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.3

    state = {
        "token_embedding.weight": rand(vocabulary, width),
        "positional_embedding": rand(context, width),
        "ln_final.weight": 1 + rand(width),
        "ln_final.bias": rand(width),
        "text_projection": rand(width, width).half(),  # CLIP keeps it in half precision
        "logit_scale": torch.tensor(4.6),
    }
    for layer in range(layers):
        prefix = f"transformer.resblocks.{layer}."
        state.update(
            {
                prefix + "ln_1.weight": 1 + rand(width),
                prefix + "ln_1.bias": rand(width),
                prefix + "attn.in_proj_weight": rand(3 * width, width).half(),
                prefix + "attn.in_proj_bias": rand(3 * width).half(),
                prefix + "attn.out_proj.weight": rand(width, width).half(),
                prefix + "attn.out_proj.bias": rand(width).half(),
                prefix + "ln_2.weight": 1 + rand(width),
                prefix + "ln_2.bias": rand(width),
                prefix + "mlp.c_fc.weight": rand(4 * width, width).half(),
                prefix + "mlp.c_fc.bias": rand(4 * width).half(),
                prefix + "mlp.c_proj.weight": rand(width, 4 * width).half(),
                prefix + "mlp.c_proj.bias": rand(width).half(),
            }
        )
    return state


def _encode_text_like_clip(state, tokens):
    """CLIP's encode_text, written out with torch's own MultiheadAttention and a causal mask."""
    width = state["positional_embedding"].shape[1]
    context = tokens.shape[0]
    f = {key: value.float() for key, value in state.items()}
    x = f["token_embedding.weight"][tokens] + f["positional_embedding"][:context]
    mask = torch.full((context, context), float("-inf")).triu(1)
    layer = 0
    while f"transformer.resblocks.{layer}.ln_1.weight" in f:
        p = f"transformer.resblocks.{layer}."
        attention = torch.nn.MultiheadAttention(width, num_heads=2)
        attention.in_proj_weight.data = f[p + "attn.in_proj_weight"]
        attention.in_proj_bias.data = f[p + "attn.in_proj_bias"]
        attention.out_proj.weight.data = f[p + "attn.out_proj.weight"]
        attention.out_proj.bias.data = f[p + "attn.out_proj.bias"]
        h = torch.nn.functional.layer_norm(x, (width,), f[p + "ln_1.weight"], f[p + "ln_1.bias"])
        x = x + attention(h[:, None], h[:, None], h[:, None], attn_mask=mask)[0][:, 0]
        h = torch.nn.functional.layer_norm(x, (width,), f[p + "ln_2.weight"], f[p + "ln_2.bias"])
        h = torch.nn.functional.linear(h, f[p + "mlp.c_fc.weight"], f[p + "mlp.c_fc.bias"])
        h = h * torch.sigmoid(1.702 * h)
        x = x + torch.nn.functional.linear(h, f[p + "mlp.c_proj.weight"], f[p + "mlp.c_proj.bias"])
        layer += 1
    x = torch.nn.functional.layer_norm(x, (width,), f["ln_final.weight"], f["ln_final.bias"])
    return x[tokens.argmax()] @ f["text_projection"]


def test_the_language_token_is_clips_embedding_of_an_empty_prompt():
    """Upstream's RGB path feeds CLIP an all-zero prompt; only position 0 matters."""
    clip = _clip_text_tower()
    reference = _encode_text_like_clip(clip, torch.zeros(5, dtype=torch.long))
    with torch.no_grad():
        ours = sutrack_net.empty_prompt_embedding(
            {"text_encoder.clip." + key: value for key, value in clip.items()}
        )
    assert torch.allclose(ours, reference, atol=1e-5)


def test_a_checkpoint_without_clip_is_not_a_sutrack_release():
    with pytest.raises(TrackerError, match="not a SUTrack release"):
        sutrack_net.empty_prompt_embedding({})


# -- loading ----------------------------------------------------------------------------


def _release(network, **extra):
    """A checkpoint laid out like upstream's: the state_dict under "net", CLIP and all."""
    state = dict(network.state_dict())
    state.update({"text_encoder.clip." + k: v for k, v in _clip_text_tower().items()})
    state["text_encoder.clip.visual.conv1.weight"] = torch.zeros(4, 3, 2, 2).half()
    state["text_encoder.norm1.weight"] = torch.ones(8)
    state["text_encoder.norm2.weight"] = torch.ones(32)
    state["task_decoder.layers.0.weight"] = torch.zeros(16, 32)
    state.update(extra)
    return {"net": state}


def test_a_release_loads_strictly_after_its_unused_parts_are_dropped(tmp_path):
    source = _randomised(sutrack_net.build_sutrack(TINY))
    path = tmp_path / "SUTRACK_ep0180.pth.tar"
    torch.save(_release(source), path)
    loaded = sutrack_net.load_sutrack(sutrack_net.build_sutrack(TINY), path)
    assert not loaded.training
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    expected = sutrack_net.empty_prompt_embedding(
        {"text_encoder.clip." + key: value for key, value in _clip_text_tower().items()}
    )
    assert torch.allclose(loaded.text_encoder.prompt, expected)
    assert "text_encoder.prompt" not in loaded.state_dict(), "computed, never stored"


@pytest.mark.parametrize(
    "damage",
    [
        lambda state: state.pop("encoder.body.cls_token"),
        lambda state: state.__setitem__("encoder.body.extra", torch.zeros(1)),
        lambda state: state.__setitem__("encoder.body.search_token", torch.zeros(7)),
    ],
    ids=["missing", "unexpected", "reshaped"],
)
def test_a_checkpoint_that_does_not_match_is_refused(tmp_path, damage):
    release = _release(sutrack_net.build_sutrack(TINY))
    damage(release["net"])
    path = tmp_path / "wrong.pth.tar"
    torch.save(release, path)
    with pytest.raises(TrackerError, match="does not match"):
        sutrack_net.load_sutrack(sutrack_net.build_sutrack(TINY), path)


# -- the backend --------------------------------------------------------------------------


class RecordingNetwork(torch.nn.Module):
    """Stands in for SUTrackNet: records its inputs, answers with fixed maps."""

    def __init__(self, offset=0.5, size=0.25, num_templates=1):
        super().__init__()
        self.config = SUTrackConfig(
            name="recording",
            embed_dim=8,
            depth_stage1=1,
            depth_stage2=1,
            depth=1,
            num_heads=1,
            num_templates=num_templates,
        )
        self.offset, self.size = offset, size
        self.seen = None

    def forward(self, templates, boxes, searches):
        self.seen = (templates, boxes, searches)
        batch = searches.shape[0]
        score = torch.full((batch, 1, 14, 14), 0.1)
        return (
            score,
            torch.full((batch, 2, 14, 14), self.size),
            torch.full((batch, 2, 14, 14), self.offset),
        )


def test_the_backend_feeds_imagenet_normalised_rgb_at_sutracks_sizes():
    network = RecordingNetwork(num_templates=2)
    backend = SUTrackBackend(network, "cpu")
    exemplar = np.zeros((2, 127, 127, 3), dtype=np.uint8)
    exemplar[..., 0] = 200  # red, in Stable's RGB
    kernels = backend.embed(exemplar)
    assert tuple(kernels.shape) == (2, 3, 112, 112)
    red = (200 / 255 - 0.485) / 0.229
    assert kernels[:, 0].mean().item() == pytest.approx(red, abs=1e-4)
    assert kernels[:, 1].mean().item() == pytest.approx(-0.456 / 0.224, abs=1e-4)
    backend.match(kernels, np.full((2, 1, 255, 255, 3), 128, dtype=np.uint8))
    templates, boxes, searches = network.seen
    assert tuple(templates.shape) == (2, 2, 3, 112, 112)
    assert tuple(boxes.shape) == (2, 2, 4)
    assert tuple(searches.shape) == (2, 3, 224, 224)
    assert searches[:, 2].mean().item() == pytest.approx((128 / 255 - 0.406) / 0.225, abs=1e-4)


def test_the_backend_marks_the_announced_template_boxes():
    network = RecordingNetwork()
    backend = SUTrackBackend(network, "cpu")
    kernels = backend.embed(np.zeros((2, 127, 127, 3), dtype=np.uint8))
    searches = np.zeros((2, 1, 255, 255, 3), dtype=np.uint8)
    backend.match(kernels, searches)
    assert network.seen[1][0, 0].tolist() == pytest.approx(template_box(56.0, 56.0).tolist())
    backend.use_template_boxes(np.array([[63.5, 63.5], [63.5, 31.75]]))
    backend.match(kernels, searches)
    assert network.seen[1][1, 0].tolist() == pytest.approx(template_box(56.0, 28.0).tolist())


@pytest.mark.parametrize("size", [255, 351])
def test_the_backend_reports_boxes_in_the_crops_pixels(size):
    """Upstream's decoding: (cell + offset) / 14 of the search, from its centre, in crop px."""
    network = RecordingNetwork(offset=0.25, size=0.3)
    backend = SUTrackBackend(network, "cpu")
    kernels = backend.embed(np.zeros((1, 127, 127, 3), dtype=np.uint8))
    maps = backend.match(kernels, np.zeros((1, 1, size, size, 3), dtype=np.uint8))
    assert isinstance(maps, SUTrackMaps)
    scale = network_scale(size)
    assert maps.stride == pytest.approx(16 / scale)
    left, top, right, bottom = maps.boxes[0, 0].astype(np.float64)
    grid = (np.arange(14) - 6.5) * maps.stride
    centre_x = grid[None, :] + (right - left) / 2
    expected = ((np.arange(14) + 0.25) / 14 - 0.5) * 223 / scale
    assert centre_x[3] == pytest.approx(expected, abs=1e-3)
    assert (left + right) == pytest.approx(np.full((14, 14), 0.3 * 223 / scale), abs=1e-3)
    assert (top + bottom)[0, 0] == pytest.approx(0.3 * 223 / scale, abs=1e-3)


# -- the published weights, where they are cached ------------------------------------------


@pytest.mark.skipif(
    not is_downloaded(SUTRACK_T224_WEIGHTS), reason="the SUTrack-T224 checkpoint is not cached"
)
def test_the_published_checkpoint_loads_and_finds_a_shifted_target():
    network = sutrack_net.load_sutrack(
        sutrack_net.build_sutrack(sutrack_net.SUTRACK_T224), local_path(SUTRACK_T224_WEIGHTS)
    )
    assert torch.isfinite(network.text_encoder.prompt).all()
    assert network.text_encoder.prompt.abs().max() > 0
    # A red disc on a grey ramp: the template centred on it, the search 24 px to its left.
    ys, xs = np.mgrid[0:400, 0:400]
    image = np.stack([xs * 0.3 + 60, ys * 0.3 + 60, np.full_like(xs, 90.0)], -1)
    image[(xs - 200) ** 2 + (ys - 200) ** 2 < 28**2] = (220, 40, 40)
    mean = torch.tensor(sutrack_net.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(sutrack_net.IMAGENET_STD).view(1, 3, 1, 1)

    def crop(cx, side):
        patch = image[200 - side // 2 : 200 + side // 2, cx - side // 2 : cx + side // 2]
        return (torch.from_numpy(patch).permute(2, 0, 1)[None].float() / 255 - mean) / std

    template = torch.nn.functional.interpolate(crop(200, 112), size=112)[:, None]
    search = crop(176, 224)
    box = torch.tensor([[[0.25, 0.25, 0.5, 0.5]]])
    with torch.inference_mode():
        score, size, offset = network(template, box, search)
    row, col = np.unravel_index(int(score[0, 0].argmax()), (14, 14))
    centre_x = (col + offset[0, 0, row, col].item()) / 14 * 223 - 111.5
    centre_y = (row + offset[0, 1, row, col].item()) / 14 * 223 - 111.5
    assert math.hypot(centre_x - 24.0, centre_y) < 3.0
    assert score.max().item() > 0.5
