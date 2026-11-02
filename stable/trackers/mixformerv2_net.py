"""The MixFormerV2 network in plain PyTorch, ported from MCG-NJU/MixFormerV2.

MixFormerV2 (Cui, Song, Wu & Wang, NeurIPS 2023) is a *one-stream* tracker: a
plain ViT reads the template and the search region together, and four learnt
**prediction tokens** ride along with the search tokens. After the last block
each token is turned into a probability distribution over 96 positions for one
edge of the target box (left, top, right, bottom), and the box is those four
distributions' means. A small MLP over the same four tokens predicts how much
the box can be trusted. There is no correlation, no score map and no
post-processing — which is what makes it fast.

This module rebuilds the inference half of the authors' online model
(``lib/models/mixformer2_vit/mixformer2_vit_online.py`` and ``head.py``) with
nothing beyond PyTorch: the upstream code needs ``timm`` and ``einops`` for a
handful of standard layers, and moves its box head to CUDA in its constructor.

What is kept exactly, because the published checkpoints depend on it:

* every module and parameter **name**, so a published ``net`` state dict loads
  with ``strict=True`` — no missing and no unexpected tensors (see
  :func:`load_checkpoint`);
* the **asymmetric mixed attention**: template tokens attend only to template
  tokens; search and prediction tokens attend to everything;
* the token order ``[template, online template, search, 4 prediction tokens]``,
  the prediction tokens' ``(left, right, top, bottom)`` order, and the heads'
  layer layout (LayerNorm ``eps`` 1e-6 in the ViT, 1e-5 in the box head);
* the **input convention** — RGB in ``[0, 1]``, normalised with the ImageNet
  mean and standard deviation — which
  :class:`stable.trackers.mixformerv2.MixFormerV2Backend` applies.

What is deliberately different:

* The network is **stateless and split in two**, the way upstream's own
  ``set_online``/``forward_test`` pair was meant to be used: :meth:`MixFormerV2Net.embed`
  runs the template tokens through every block once and returns their
  per-block keys and values; :meth:`MixFormerV2Net.match` then runs only the
  search and prediction tokens against them. Template tokens never attend to
  the search, so this is exact, and it is what lets one template serve every
  frame of a run at a fraction of the cost. Upstream's tracker runs the joint
  forward pass every frame instead; :meth:`MixFormerV2Net.forward` is that
  pass, kept for parity checks.
* The templates and searches are **batched**, one template pair per search.
* Attention is :func:`torch.nn.functional.scaled_dot_product_attention`, with
  the same ``1 / sqrt(head_dim)`` scale.
* Nothing is moved to CUDA implicitly, so the CPU fallback works.

Two published models are described by :data:`MIXFORMERV2_BASE` and
:data:`MIXFORMERV2_SMALL`. Their checkpoints are pytracking-style *training*
checkpoints that also pickle the trainer's statistics and settings objects;
:func:`load_checkpoint` reads them with ``weights_only=True`` and inert
stand-ins for those four classes, so no code from the file is ever run.

This module imports PyTorch at import time and is only ever imported lazily
by the plugin that needs it. Without PyTorch it still imports — every class is
then defined on a plain ``object`` base and cannot be built — so tooling that
walks the package (documentation, ``tests/test_public_api.py``) never needs the
optional dependency.

Licence
-------
Ported from MixFormerV2 (https://github.com/MCG-NJU/MixFormerV2), released
under the MIT licence. Its ViT layers follow ``timm``'s vision transformer
(Apache-2.0), whose module names they keep.

    MIT License

    Copyright (c) 2021 Multimedia Research

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.device import torch_module
from ..core.errors import TrackerError

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MixFormerV2Config",
    "MIXFORMERV2_BASE",
    "MIXFORMERV2_SMALL",
    "MixFormerV2Net",
    "MixFormerV2Backbone",
    "MixFormerV2Output",
    "BoxHead",
    "ScoreHead",
    "load_checkpoint",
    "read_state_dict",
]

torch: Any = torch_module()
if torch is not None:
    nn = torch.nn
    F = torch.nn.functional
    _Module: Any = torch.nn.Module
else:  # pragma: no cover - only without the 'torch' extra
    nn = F = None
    _Module = object

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Per-channel RGB mean the network's inputs are normalised by (in ``[0, 1]``)."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel RGB standard deviation the inputs are normalised by."""

PREDICTION_TOKENS = 4
"""Learnt tokens appended to the search: one per box edge."""

_TRAINING_GLOBALS = (
    "lib.train.admin.stats.AverageMeter",
    "lib.train.admin.stats.StatValue",
    "lib.train.admin.settings.Settings",
    "lib.train.admin.local.EnvironmentSettings",
)
"""Classes the published checkpoints pickle besides tensors: the trainer's
loss meters and settings. Nothing in them is needed to run the network."""


@dataclass(frozen=True)
class MixFormerV2Config:
    """The architecture of one published MixFormerV2 model.

    Attributes:
        name: Short name, used in messages.
        depth: Transformer blocks.
        mlp_ratio: Hidden width of each block's MLP, as a multiple of ``embed_dim``.
        template_size: Side of the template crop the network takes, in pixels.
        search_size: Side of the search crop, in pixels.
        embed_dim: Token width.
        num_heads: Attention heads per block.
        patch_size: Side of the square patch each token embeds.
        bins: Positions each edge's distribution is spread over, evenly across
            the search crop (upstream's ``FEAT_SZ``).
    """

    name: str
    depth: int
    mlp_ratio: float
    template_size: int
    search_size: int
    embed_dim: int = 768
    num_heads: int = 12
    patch_size: int = 16
    bins: int = 96

    @property
    def template_cells(self) -> int:
        """Patches per side of the template."""
        return self.template_size // self.patch_size

    @property
    def search_cells(self) -> int:
        """Patches per side of the search region."""
        return self.search_size // self.patch_size


MIXFORMERV2_BASE = MixFormerV2Config(
    name="MixFormerV2-B", depth=8, mlp_ratio=4.0, template_size=128, search_size=288
)
"""MixFormerV2-B: eight ViT-B blocks, 128 px templates, 288 px search
(upstream ``288_depth8_score``)."""

MIXFORMERV2_SMALL = MixFormerV2Config(
    name="MixFormerV2-S", depth=4, mlp_ratio=1.0, template_size=112, search_size=224
)
"""MixFormerV2-S: four ViT-B-wide blocks with a 1x MLP, 112 px templates, 224 px
search (upstream ``224_depth4_mlp1_score``)."""


@dataclass(frozen=True)
class MixFormerV2Output:
    """What one batch of searches produced.

    Attributes:
        boxes: ``[B, 4]`` ``(x1, y1, x2, y2)`` as fractions of the search crop's
            side; ``0.5`` is the crop centre.
        score: ``[B]`` the score head's logit that the box is on the target.
        spread: ``[B, 4]`` standard deviation of each edge's distribution, in
            the same units as ``boxes``: how unsure the network is of that edge.
    """

    boxes: Any
    score: Any
    spread: Any


class PatchEmbed(_Module):
    """A ``patch_size``-strided convolution turning an image into tokens.

    Args:
        patch_size: Side of each patch, in pixels.
        embed_dim: Token width.
    """

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        """Build the projection."""
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Any) -> Any:
        """Return ``[B, (H / p) * (W / p), embed_dim]`` tokens, row-major."""
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(_Module):
    """A block's two-layer GELU MLP (``timm``'s ``Mlp``, inference only).

    Args:
        dim: Token width.
        hidden: Hidden width.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        """Build the two layers."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Any) -> Any:
        """Apply the MLP."""
        return self.fc2(self.act(self.fc1(x)))


class Attention(_Module):
    """MixFormer's asymmetric mixed attention, split into its two halves.

    Template tokens attend only to template tokens, so they can be run once
    (:meth:`template`) and their keys and values kept; search tokens attend to
    the template's and their own (:meth:`search`).

    Args:
        dim: Token width.
        num_heads: Attention heads.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        """Build the joint query/key/value projection and the output projection."""
        super().__init__()
        self.num_heads = int(num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def _qkv(self, x: Any) -> Any:
        """Return ``[3, B, heads, N, head_dim]`` queries, keys and values."""
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        return qkv.permute(2, 0, 3, 1, 4)

    def _merge(self, x: Any) -> Any:
        """Return ``[B, heads, N, head_dim]`` as projected ``[B, N, dim]``."""
        batch, heads, tokens, width = x.shape
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, heads * width))

    def template(self, x: Any) -> tuple[Any, Any, Any]:
        """Self-attention among template tokens.

        Returns:
            ``(output, keys, values)``; the keys and values are what the search
            tokens attend to in this block.
        """
        q, k, v = self._qkv(x).unbind(0)
        return self._merge(F.scaled_dot_product_attention(q, k, v)), k, v

    def search(self, x: Any, keys: Any, values: Any) -> Any:
        """Search tokens attending to the template's keys and values and their own."""
        q, k, v = self._qkv(x).unbind(0)
        k = torch.cat([keys, k], dim=2)
        v = torch.cat([values, v], dim=2)
        return self._merge(F.scaled_dot_product_attention(q, k, v))


class Block(_Module):
    """One pre-norm transformer block.

    Args:
        dim: Token width.
        num_heads: Attention heads.
        mlp_ratio: MLP hidden width as a multiple of ``dim``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        """Build the norms, the attention and the MLP."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def template(self, x: Any) -> tuple[Any, Any, Any]:
        """Apply the block to template tokens; also return its keys and values."""
        out, keys, values = self.attn.template(self.norm1(x))
        x = x + out
        return x + self.mlp(self.norm2(x)), keys, values

    def search(self, x: Any, keys: Any, values: Any) -> Any:
        """Apply the block to search tokens, given the template's keys and values."""
        x = x + self.attn.search(self.norm1(x), keys, values)
        return x + self.mlp(self.norm2(x))


class MixFormerV2Backbone(_Module):
    """The ViT, its positional embeddings and the four prediction tokens.

    Args:
        config: The architecture.
    """

    def __init__(self, config: MixFormerV2Config) -> None:
        """Build the patch embedding, the blocks and the token parameters."""
        super().__init__()
        dim = config.embed_dim
        self.patch_embed = PatchEmbed(config.patch_size, dim)
        self.blocks = nn.Sequential(
            *[Block(dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth)]
        )
        # Fixed sine-cosine embeddings upstream; the checkpoints carry them.
        self.pos_embed_s = nn.Parameter(
            torch.zeros(1, config.search_cells**2, dim), requires_grad=False
        )
        self.pos_embed_t = nn.Parameter(
            torch.zeros(1, config.template_cells**2, dim), requires_grad=False
        )
        self.reg_tokens = nn.Parameter(torch.zeros(1, PREDICTION_TOKENS, dim))
        self.pos_embed_reg = nn.Parameter(torch.zeros(1, PREDICTION_TOKENS, dim))

    def embed(self, template: Any, online_template: Any) -> list[tuple[Any, Any]]:
        """Run both templates through every block.

        Args:
            template: ``[B, 3, t, t]`` normalised static templates.
            online_template: ``[B, 3, t, t]`` normalised online templates.

        Returns:
            Per block, the template tokens' ``(keys, values)``, each
            ``[B, heads, 2 * cells, head_dim]``.
        """
        x = torch.cat(
            [
                self.patch_embed(template) + self.pos_embed_t,
                self.patch_embed(online_template) + self.pos_embed_t,
            ],
            dim=1,
        )
        memory = []
        for block in self.blocks:
            x, keys, values = block.template(x)
            memory.append((keys, values))
        return memory

    def match(self, memory: list[tuple[Any, Any]], search: Any) -> Any:
        """Run the search and prediction tokens against the templates' ``memory``.

        Args:
            memory: What :meth:`embed` returned, one entry per search.
            search: ``[B, 3, s, s]`` normalised search crops.

        Returns:
            ``[B, 4, dim]`` the prediction tokens after the last block.
        """
        x = self.patch_embed(search) + self.pos_embed_s
        tokens = (self.reg_tokens + self.pos_embed_reg).expand(x.shape[0], -1, -1)
        x = torch.cat([x, tokens], dim=1)
        for block, (keys, values) in zip(self.blocks, memory, strict=True):
            x = block.search(x, keys, values)
        return x[:, -PREDICTION_TOKENS:]


class BoxHead(_Module):
    """Turns each prediction token into a distribution over one edge's position.

    Upstream's ``MlpHead`` with ``num_layers=2, norm=True``: a hidden layer and an
    output layer, both layer-normalised, shared by all four tokens.

    Args:
        dim: Token width.
        bins: Positions per distribution.
    """

    def __init__(self, dim: int, bins: int) -> None:
        """Build the shared MLP."""
        super().__init__()
        self.bins = int(bins)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU()),
            nn.Sequential(nn.Linear(dim, bins), nn.LayerNorm(bins)),
        )

    def forward(self, tokens: Any) -> tuple[Any, Any]:
        """Return the box and each edge's spread.

        Args:
            tokens: ``[B, 4, dim]`` prediction tokens, in upstream's
                ``(left, right, top, bottom)`` order.

        Returns:
            ``(boxes, spread)``: ``[B, 4]`` ``(x1, y1, x2, y2)`` — each edge's
            mean position — and ``[B, 4]`` each distribution's standard
            deviation, both as fractions of the search side.
        """
        left, right, top, bottom = tokens.unbind(dim=1)
        ordered = torch.stack([left, top, right, bottom], dim=1)
        probability = self.layers(ordered).softmax(dim=-1)
        positions = torch.arange(self.bins, device=tokens.device, dtype=probability.dtype)
        mean = (probability * positions).sum(-1)
        variance = (probability * (positions - mean[..., None]) ** 2).sum(-1)
        return mean / self.bins, variance.clamp_min(0).sqrt() / self.bins


class ScoreHead(_Module):
    """Upstream's ``MlpScoreDecoder``: one logit, averaged over the four tokens.

    Trained, in MixFormerV2's last stage, to say whether the predicted box is on
    the target.

    Args:
        dim: Token width.
    """

    def __init__(self, dim: int) -> None:
        """Build the two-layer MLP."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU()), nn.Linear(dim, 1)
        )

    def forward(self, tokens: Any) -> Any:
        """Return ``[B]`` logits for ``[B, 4, dim]`` prediction tokens."""
        return self.layers(tokens).mean(dim=1)[:, 0]


class MixFormerV2Net(_Module):
    """MixFormerV2 for inference: templates in once, searches in per frame.

    Args:
        config: Which published architecture to build.
    """

    def __init__(self, config: MixFormerV2Config = MIXFORMERV2_BASE) -> None:
        """Build the backbone and both heads with random weights."""
        super().__init__()
        self.config = config
        self.backbone = MixFormerV2Backbone(config)
        self.box_head = BoxHead(config.embed_dim, config.bins)
        self.score_head = ScoreHead(config.embed_dim)

    def embed(self, template: Any, online_template: Any | None = None) -> list[tuple[Any, Any]]:
        """Return the templates' per-block keys and values; see :meth:`match`.

        Args:
            template: ``[B, 3, t, t]`` normalised templates.
            online_template: Upstream's second, updated template; ``None`` uses
                ``template`` for it, which is upstream's own state until its
                first update.
        """
        return self.backbone.embed(
            template, template if online_template is None else online_template
        )

    def match(self, memory: list[tuple[Any, Any]], search: Any) -> MixFormerV2Output:
        """Locate each template in its search crop.

        Args:
            memory: :meth:`embed`'s output, one template pair per search.
            search: ``[B, 3, s, s]`` normalised search crops.

        Returns:
            The boxes, score logits and edge spreads.
        """
        tokens = self.backbone.match(memory, search)
        boxes, spread = self.box_head(tokens)
        return MixFormerV2Output(boxes=boxes, score=self.score_head(tokens), spread=spread)

    def forward(self, template: Any, online_template: Any, search: Any) -> MixFormerV2Output:
        """Run templates and searches in one go (upstream's per-frame pass)."""
        return self.match(self.embed(template, online_template), search)


def read_state_dict(path: str | Path) -> dict[str, Any]:
    """Return the network tensors from a published MixFormerV2 checkpoint.

    The files are pytracking training checkpoints: the weights sit under
    ``net``, beside the optimiser state and pickled trainer objects. They are
    read with ``weights_only=True``, the trainer's four bookkeeping classes
    replaced by an inert stand-in that keeps nothing, so loading never runs
    code from the file. A bare state dict loads too.

    Args:
        path: The checkpoint file.

    Returns:
        The tensors, with any ``module.`` prefix (from ``nn.DataParallel``)
        dropped.

    Raises:
        TrackerError: If PyTorch is missing or older than 2.6 (which first
            allowed a stand-in for a named class), or the file is not a
            checkpoint.
    """
    if torch is None:  # pragma: no cover - guarded by the plugin's availability check
        raise TrackerError("loading a MixFormerV2 checkpoint needs PyTorch")
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:  # pragma: no cover - PyTorch < 2.5 only
        raise TrackerError("loading a MixFormerV2 checkpoint safely needs PyTorch 2.6 or newer")
    try:
        with safe_globals([(_Discarded, name) for name in _TRAINING_GLOBALS]):
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TrackerError(
            f"could not read {Path(path).name} as a MixFormerV2 checkpoint ({exc}); "
            "loading one safely needs PyTorch 2.6 or newer"
        ) from exc
    tensors = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(tensors, dict):
        raise TrackerError(f"{Path(path).name} holds no MixFormerV2 state dict")
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in tensors.items()
    }


def load_checkpoint(network: Any, path: str | Path) -> Any:
    """Load a published checkpoint into ``network``, insisting on a complete match.

    Every tensor the network has must be in the file and every tensor in the
    file must be in the network, with the same shape: a file for the other
    variant, or for a different network, is refused rather than run with
    random weights.

    Args:
        network: A :class:`MixFormerV2Net`.
        path: The checkpoint file.

    Returns:
        ``network``, with the weights loaded and put in evaluation mode.

    Raises:
        TrackerError: If the checkpoint cannot be read or does not match.
    """
    state = read_state_dict(path)
    expected = network.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    reshaped = sorted(
        key
        for key in set(state) & set(expected)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if missing or unexpected or reshaped:
        raise TrackerError(
            f"{Path(path).name} does not match {network.config.name}: "
            f"{len(missing)} missing tensor(s) (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}), "
            f"{len(reshaped)} of another shape (e.g. {reshaped[:3]})"
        )
    network.load_state_dict(state, strict=True)
    return network.eval()


class _Discarded:
    """Stand-in for a pickled trainer object in a checkpoint: keeps nothing."""

    def __setstate__(self, state: Any) -> None:
        """Ignore the pickled state."""
