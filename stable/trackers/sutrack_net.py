"""SUTrack's network, as it runs on RGB video, in plain PyTorch.

SUTrack (Chen, Kang, Geng, Zhu, Liu, Wang & Lu, AAAI 2025, "SUTrack: Towards
Simple and Unified Single Object Tracking") is a *one-stream* tracker: the
template and the search region are cut into patch tokens and run through one
transformer together, so every layer can compare the two, and a centre head on
the search tokens says where the target is and how big. One set of weights
covers RGB, RGB-depth, RGB-thermal, RGB-event and RGB-language tracking; this
module ports only the path an RGB clip takes through it, from the authors' code
(https://github.com/chenxin-dlut/SUTrack, ``lib/models/sutrack``), so that
nothing beyond PyTorch is needed to run it — the upstream repository wants
``timm``, ``clip``, ``easydict`` and a research checkout.

What an RGB frame goes through, exactly as upstream evaluates LaSOT or GOT-10k:

* **Normalisation**: ImageNet mean and standard deviation, RGB order.
* **The auxiliary modality**: the patch embedding takes six channels, the second
  three for depth, thermal or events. Upstream's RGB path fills them with a
  second copy of the RGB image (``torch.cat((template, template), axis=1)``), and
  so does :meth:`SUTrackNet.forward`.
* **The encoder**: Fast-iTPN (a HiViT-style hierarchy): a 4x4 convolutional
  patch embedding, convolutional MLP blocks at 1/4 and 1/8 resolution with patch
  merging between them, then plain transformer blocks (SwiGLU MLPs with sub-LN,
  layer scale) over the 1/16 tokens of template and search *together*.
* **Token types**: learned embeddings mark each search token, and each template
  token as foreground or background according to how much of its 16x16 patch
  the target box covers.
* **The language token**: upstream's RGB trackers run with
  ``MULTI_MODAL_LANGUAGE`` on and no sentence, so CLIP's text encoder is fed an
  all-zero prompt and its projection is appended to the token sequence as one
  extra token. That token is the same for every frame of every clip. CLIP's
  causal mask means position 0 of the prompt attends only to itself, and its
  pooling reads position 0 (the ``argmax`` of an all-zero prompt), so the token
  is a fixed function of a dozen tensors of the checkpoint; :func:`load_sutrack`
  evaluates it once, at load, with no tokenizer, no CLIP package and no text
  encoder at inference (:func:`empty_prompt_embedding`).
* **The centre head**: three five-layer convolutional towers giving a centre
  score map (sigmoid), the box size (sigmoid, as a fraction of the search) and
  the sub-cell offset of the centre, per 16-pixel cell of the search region.

What is kept exactly, because the published checkpoints depend on it: every
module and parameter **name** of the encoder (``encoder.body.*``), the centre
head (``decoder.*``) and the language token's projection
(``text_encoder.text_proj.*``), so the authors' ``state_dict`` loads into them
with strict matching (see :func:`load_sutrack`), and every layer's geometry.

What is dropped, and why (the checkpoint keys :func:`load_sutrack` discards):

* ``text_encoder.clip.*`` — the whole of OpenAI's CLIP ViT-L/14, image tower
  included, which the training code kept as a submodule (it is most of the
  1 GB file). The image tower is never used by SUTrack; the text tower is used
  only to compute the constant empty-prompt token above, once, at load.
* ``text_encoder.norm1.*``, ``text_encoder.norm2.*`` — layer norms the upstream
  text encoder defines but never calls (both calls are commented out).
* ``task_decoder.*`` — the task-recognition classifier, an auxiliary training
  loss that says which modality a batch came from. Upstream's trackers never
  call it, and the RGB path's ``task_index`` is ``None``.

Deliberately different from upstream:

* The network is **stateless** and batched: :meth:`SUTrackNet.forward` takes a
  batch of searches, each with its own templates and template box, so one call
  serves every point sharing a frame.
* Attention uses :func:`torch.nn.functional.scaled_dot_product_attention`
  (same arithmetic in float32; on a GPU a fused kernel), not an explicit
  softmax over ``q @ k.T``.
* Only the trained search size is accepted. Upstream can resample the
  position embedding to another (its ``POS_TYPE: interpolate``), but measured
  with the real weights that broke the score map — at a 100 px search radius
  its peak on correctly tracked frames fell to a median 0.48, against 0.92
  with the same search shrunk onto the trained size — so
  :mod:`stable.trackers.sutrack` shrinks a wider search instead.

This module imports PyTorch at import time and is only ever imported lazily by
:mod:`stable.trackers.sutrack`. Without PyTorch it still imports — every class
is then defined on a plain ``object`` base and cannot be built — so tooling that
walks the package never needs the optional dependency.

Licence
-------
Ported from SUTrack (https://github.com/chenxin-dlut/SUTrack), whose code and
whose Hugging Face model card are released under the MIT licence. Its Fast-iTPN
encoder is itself from https://github.com/sunsmarterjie/iTPN (MIT, Copyright (c)
2023 University of Chinese Academy of Sciences, by Yunjie Tian):

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
    "SUTrackConfig",
    "SUTRACK_T224",
    "SUTRACK_B224",
    "DROPPED_PREFIXES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUTrackNet",
    "FastITPN",
    "CenterPredictor",
    "build_sutrack",
    "empty_prompt_embedding",
    "load_sutrack",
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
"""Per-channel mean SUTrack's inputs are normalised by (RGB, values in [0, 1])."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel standard deviation SUTrack's inputs are normalised by."""

DROPPED_PREFIXES = (
    "text_encoder.clip.",
    "text_encoder.norm1.",
    "text_encoder.norm2.",
    "task_decoder.",
)
"""Checkpoint keys :func:`load_sutrack` discards; see the module docstring for why."""

_CLIP = "text_encoder.clip."
_LAYER_NORM_EPS = 1e-6  # every norm in Fast-iTPN; CLIP's own use PyTorch's 1e-5


@dataclass(frozen=True)
class SUTrackConfig:
    """One published SUTrack variant: its encoder's shape and its input sizes.

    Attributes:
        name: Upstream's experiment name (``sutrack_<name>.yaml``).
        embed_dim: Token width at 1/16 resolution; the 1/4 and 1/8 stages run at
            a quarter and a half of it.
        depth_stage1: Convolutional MLP blocks at 1/4 resolution.
        depth_stage2: Convolutional MLP blocks at 1/8 resolution.
        depth: Transformer blocks over template and search tokens together.
        num_heads: Attention heads per transformer block.
        template_size: Template side in network pixels.
        search_size: Search side in network pixels, as trained.
        num_templates: Templates the encoder takes per search. Upstream fills
            them all with the first frame's template and, for the variants with
            two, replaces the second every 25 frames (see
            :mod:`stable.trackers.sutrack` for why Stable does not).
        patch_size: Network pixels per token side at 1/16 resolution.
        mlp_ratio: Hidden width of the transformer blocks' MLPs, per token width.
        bridge_mlp_ratio: The same for the 1/4 and 1/8 convolutional blocks.
        head_channels: Width of the centre head's first convolution.
        text_width: Width of CLIP ViT-L/14's text embedding, the input of the
            language token's projection.
    """

    name: str
    embed_dim: int
    depth_stage1: int
    depth_stage2: int
    depth: int
    num_heads: int
    template_size: int = 112
    search_size: int = 224
    num_templates: int = 1
    patch_size: int = 16
    mlp_ratio: float = 3.0
    bridge_mlp_ratio: float = 3.0
    head_channels: int = 256
    text_width: int = 768

    @property
    def search_cells(self) -> int:
        """Score-map side at the trained search size."""
        return self.search_size // self.patch_size

    @property
    def template_cells(self) -> int:
        """Template token-grid side."""
        return self.template_size // self.patch_size


SUTRACK_T224 = SUTrackConfig(
    name="sutrack_t224", embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6
)
"""SUTrack-T224: Fast-iTPN-T, one fixed template, 112 px template, 224 px search."""

SUTRACK_B224 = SUTrackConfig(
    name="sutrack_b224",
    embed_dim=512,
    depth_stage1=3,
    depth_stage2=3,
    depth=24,
    num_heads=8,
    num_templates=2,
)
"""SUTrack-B224: Fast-iTPN-B, two templates, 112 px template, 224 px search."""


def _layer_norm(dim: int) -> Any:
    """Return a Fast-iTPN layer norm (``eps=1e-6``)."""
    return nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)


def _channels_last_norm(norm: Any, x: Any) -> Any:
    """Apply a token-wise layer norm to a ``[B, C, H, W]`` map."""
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# -- Fast-iTPN, the encoder ------------------------------------------------------


class ConvPatchEmbed(_Module):
    """The 4x4, stride-4 convolution that cuts the image into 1/4-resolution tokens.

    Args:
        in_chans: Input channels: RGB plus the auxiliary modality, six.
        embed_dim: Output channels, a quarter of the encoder's token width.
    """

    def __init__(self, in_chans: int, embed_dim: int) -> None:
        """Create the embedding."""
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)

    def forward(self, x: Any) -> Any:
        """Embed ``[B, 6, H, W]`` into ``[B, C, H / 4, W / 4]``."""
        return self.proj(x)


class ConvMlp(_Module):
    """A 1x1-convolution MLP with GELU and a sub-layer norm on the hidden width."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Create the MLP for ``dim`` channels through ``hidden``."""
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.ffn_ln = _layer_norm(hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Any) -> Any:
        """Apply the MLP to a ``[B, C, H, W]`` map."""
        return self.fc2(_channels_last_norm(self.ffn_ln, self.act(self.fc1(x))))


class ConvMlpBlock(_Module):
    """A pre-norm residual :class:`ConvMlp` — Fast-iTPN's 1/4 and 1/8 stages."""

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        """Create the block."""
        super().__init__()
        self.norm2 = _layer_norm(dim)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Any) -> Any:
        """Apply the block to a ``[B, C, H, W]`` map."""
        return x + self.mlp(_channels_last_norm(self.norm2, x))


class ConvPatchMerge(_Module):
    """Halve the resolution and double the width: norm, then a 2x2 stride-2 convolution."""

    def __init__(self, dim: int) -> None:
        """Create the merge from ``dim`` to ``2 * dim`` channels."""
        super().__init__()
        self.norm = _layer_norm(dim)
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

    def forward(self, x: Any) -> Any:
        """Merge a ``[B, C, H, W]`` map into ``[B, 2C, H / 2, W / 2]``."""
        return self.reduction(_channels_last_norm(self.norm, x))


class SwiGLU(_Module):
    """The transformer blocks' MLP: ``w3(ln(silu(w1 x) * w2 x))``."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Create the MLP for ``dim`` channels through ``hidden``."""
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.ffn_ln = _layer_norm(hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: Any) -> Any:
        """Apply the MLP to ``[B, N, C]`` tokens."""
        return self.w3(self.ffn_ln(self.act(self.w1(x)) * self.w2(x)))


class Attention(_Module):
    """Multi-head self-attention with separate projections and biases on q and v only."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Create the attention for ``dim``-wide tokens."""
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Any) -> Any:
        """Attend over ``[B, N, C]`` tokens."""
        batch, count, width = x.shape

        def heads(value: Any) -> Any:
            return value.reshape(batch, count, self.num_heads, -1).transpose(1, 2)

        q = heads(F.linear(x, self.q_proj.weight, self.q_bias))
        k = heads(F.linear(x, self.k_proj.weight))
        v = heads(F.linear(x, self.v_proj.weight, self.v_bias))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, count, width))


class Block(_Module):
    """A pre-norm transformer block with layer scale (``gamma_1``, ``gamma_2``)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        """Create the block."""
        super().__init__()
        self.norm1 = _layer_norm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = _layer_norm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(torch.ones(dim))
        self.gamma_2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: Any) -> Any:
        """Apply the block to ``[B, N, C]`` tokens."""
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class FastITPN(_Module):
    """Fast-iTPN as SUTrack configures it: hierarchical stem, joint transformer.

    Args:
        config: The variant.

    Attributes:
        num_main_blocks: Transformer blocks at the end of :attr:`blocks`; the
            ones before them are the convolutional stem.
    """

    def __init__(self, config: SUTrackConfig) -> None:
        """Create an encoder with untrained weights."""
        super().__init__()
        dim = config.embed_dim
        self.config = config
        self.num_main_blocks = config.depth
        self.patch_size = config.patch_size
        self.num_patches_search = config.search_cells**2
        self.num_patches_template = config.template_cells**2
        self.patch_embed = ConvPatchEmbed(6, dim // 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_search + self.num_patches_template, dim)
        )
        self.template_background_token = nn.Parameter(torch.zeros(dim))
        self.template_foreground_token = nn.Parameter(torch.zeros(dim))
        self.search_token = nn.Parameter(torch.zeros(dim))
        blocks: list[Any] = [
            ConvMlpBlock(dim // 4, config.bridge_mlp_ratio) for _ in range(config.depth_stage1)
        ]
        blocks.append(ConvPatchMerge(dim // 4))
        blocks += [
            ConvMlpBlock(dim // 2, config.bridge_mlp_ratio) for _ in range(config.depth_stage2)
        ]
        blocks.append(ConvPatchMerge(dim // 2))
        blocks += [Block(dim, config.num_heads, config.mlp_ratio) for _ in range(config.depth)]
        self.blocks = nn.ModuleList(blocks)
        self.fc_norm = _layer_norm(dim)

    def stem(self, images: Any) -> Any:
        """Run the convolutional stem: ``[B, 6, H, W]`` to ``[B, H/16 * W/16, C]`` tokens."""
        x = self.patch_embed(images)
        for block in self.blocks[: -self.num_main_blocks]:
            x = block(x)
        return x.flatten(2).transpose(1, 2)

    def template_indication(self, boxes: Any, size: int) -> Any:
        """Return each template token's foreground/background type embedding.

        Upstream's ``create_mask``: a pixel is foreground when its index lies in
        ``[x0, x0 + w)`` (likewise in y) with the box in template pixels; each
        token weighs the two embeddings by the fraction of its patch inside.

        Args:
            boxes: ``[B, 4]`` normalised ``(x0, y0, w, h)`` of the target box in
                the template, as upstream's ``transform_image_to_crop`` makes it.
            size: Template side in pixels.

        Returns:
            ``[B, cells * cells, C]`` embeddings to add to the template tokens.
        """
        index = torch.arange(size, device=boxes.device, dtype=boxes.dtype)
        x0, y0 = boxes[:, 0:1] * size, boxes[:, 1:2] * size
        w, h = boxes[:, 2:3] * size, boxes[:, 3:4] * size
        inside_x = ((index >= x0) & (index < x0 + w)).to(boxes.dtype)
        inside_y = ((index >= y0) & (index < y0 + h)).to(boxes.dtype)
        mask = inside_y[:, :, None] * inside_x[:, None, :]
        patch = self.patch_size
        fraction = F.avg_pool2d(mask[:, None], patch, patch).flatten(1)[..., None]
        foreground = self.template_foreground_token.to(fraction.dtype)
        background = self.template_background_token.to(fraction.dtype)
        return foreground * fraction + background * (1.0 - fraction)

    def forward(self, templates: Any, boxes: Any, searches: Any, text: Any) -> Any:
        """Encode each search together with its templates and the language token.

        Args:
            templates: ``[B, T, 6, h, w]`` normalised templates.
            boxes: ``[B, T, 4]`` the target box in each template (see
                :meth:`template_indication`).
            searches: ``[B, 6, S, S]`` normalised searches at the trained size.
            text: ``[1, 1, C]`` the language token.

        Returns:
            ``[B, 1 + (S/16)^2 + T * (h/16)^2 + 1, C]`` tokens: class, search,
            templates, language.

        Raises:
            TrackerError: If the search is not the size the network was trained at.
        """
        batch, count = templates.shape[:2]
        size = templates.shape[-1]
        z = self.stem(templates.flatten(0, 1))
        x = self.stem(searches)
        if x.shape[1] != self.num_patches_search:
            raise TrackerError(
                f"SUTrack takes a {self.config.search_size} px search, got {searches.shape[-1]} px"
            )
        x = x + self.pos_embed[:, : self.num_patches_search] + self.search_token
        z = z + self.pos_embed[:, self.num_patches_search :]
        z = z + self.template_indication(boxes.flatten(0, 1), size)
        z = z.reshape(batch, count * z.shape[1], z.shape[2])
        tokens = torch.cat(
            [
                self.cls_token.expand(batch, -1, -1),
                x,
                z,
                text.expand(batch, -1, -1),
            ],
            dim=1,
        )
        for block in self.blocks[-self.num_main_blocks :]:
            tokens = block(tokens)
        return self.fc_norm(tokens)


class Encoder(_Module):
    """Upstream's wrapper around the encoder, kept so its parameter names match."""

    def __init__(self, config: SUTrackConfig) -> None:
        """Create the encoder."""
        super().__init__()
        self.body = FastITPN(config)


class TextToken(_Module):
    """The language token: CLIP's empty-prompt embedding, projected to token width.

    Attributes:
        prompt: ``[text_width]`` CLIP ViT-L/14's text embedding of an all-zero
            prompt; filled by :func:`load_sutrack` (see
            :func:`empty_prompt_embedding`). Not part of the ``state_dict``.
    """

    def __init__(self, config: SUTrackConfig) -> None:
        """Create the projection; the prompt starts at zero."""
        super().__init__()
        self.text_proj = nn.Linear(config.text_width, config.embed_dim)
        self.register_buffer("prompt", torch.zeros(config.text_width), persistent=False)

    def forward(self) -> Any:
        """Return the ``[1, 1, C]`` token."""
        return self.text_proj(self.prompt)[None, None]


# -- the centre head ----------------------------------------------------------------


def _conv_bn_relu(channels_in: int, channels_out: int) -> Any:
    """Return upstream's ``conv``: a 3x3 convolution, batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class CenterPredictor(_Module):
    """Upstream's centre head: score, offset and size towers over the search tokens."""

    def __init__(self, inplanes: int, channel: int) -> None:
        """Create the head for ``inplanes``-wide tokens."""
        super().__init__()
        widths = [inplanes, channel, channel // 2, channel // 4, channel // 8]
        for branch, outputs in (("ctr", 1), ("offset", 2), ("size", 2)):
            for index in range(4):
                setattr(
                    self,
                    f"conv{index + 1}_{branch}",
                    _conv_bn_relu(widths[index], widths[index + 1]),
                )
            setattr(self, f"conv5_{branch}", nn.Conv2d(widths[4], outputs, kernel_size=1))

    def _tower(self, branch: str, x: Any) -> Any:
        """Run one of the three towers."""
        for index in range(1, 6):
            x = getattr(self, f"conv{index}_{branch}")(x)
        return x

    def forward(self, x: Any) -> tuple[Any, Any, Any]:
        """Return ``(score, size, offset)`` maps for a ``[B, C, M, M]`` feature map.

        ``score`` (``[B, 1, M, M]``) and ``size`` (``[B, 2, M, M]``, width and
        height as fractions of the search side) are sigmoids clamped to
        ``[1e-4, 1 - 1e-4]``; ``offset`` (``[B, 2, M, M]``) is the raw sub-cell
        position of the centre, in cells from the cell's top-left corner.
        """

        def squash(value: Any) -> Any:
            return value.sigmoid().clamp(min=1e-4, max=1 - 1e-4)

        return (
            squash(self._tower("ctr", x)),
            squash(self._tower("size", x)),
            self._tower("offset", x),
        )


class SUTrackNet(_Module):
    """SUTrack on RGB input: encoder, language token and centre head.

    Args:
        config: The variant (:data:`SUTRACK_T224` or :data:`SUTRACK_B224`).
    """

    def __init__(self, config: SUTrackConfig) -> None:
        """Create the network with untrained weights; see :func:`load_sutrack`."""
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.text_encoder = TextToken(config)
        self.decoder = CenterPredictor(config.embed_dim, config.head_channels)

    def forward(self, templates: Any, boxes: Any, searches: Any) -> tuple[Any, Any, Any]:
        """Score each search against its templates.

        Args:
            templates: ``[B, T, 3, h, w]`` templates, ImageNet-normalised RGB.
                The auxiliary-modality channels are filled with a second copy,
                as upstream does for RGB video.
            boxes: ``[B, T, 4]`` normalised ``(x0, y0, w, h)`` of the target in
                each template.
            searches: ``[B, 3, S, S]`` searches, normalised likewise, at the
                trained ``search_size``.

        Returns:
            The centre head's ``(score, size, offset)`` maps, ``S / 16`` cells a
            side; see :meth:`CenterPredictor.forward`.
        """
        templates = torch.cat([templates, templates], dim=2)
        searches = torch.cat([searches, searches], dim=1)
        tokens = self.encoder.body(templates, boxes, searches, self.text_encoder())
        cells = searches.shape[-1] // self.config.patch_size
        search = tokens[:, 1 : 1 + cells * cells]
        feature = search.transpose(1, 2).reshape(search.shape[0], -1, cells, cells)
        return self.decoder(feature)


def build_sutrack(config: SUTrackConfig) -> Any:
    """Return an untrained :class:`SUTrackNet` for ``config``.

    Raises:
        TrackerError: If PyTorch is not installed.
    """
    if torch is None:  # pragma: no cover - guarded by the plugin's availability check
        raise TrackerError("SUTrack needs PyTorch")
    return SUTrackNet(config)


def empty_prompt_embedding(state: dict[str, Any]) -> Any:
    """Return CLIP's text embedding of an all-zero prompt, from a SUTrack checkpoint.

    Exactly ``clip.encode_text(zeros(1, 77))[0]``: CLIP's text transformer is
    causal, so prompt position 0 attends only to itself and every attention
    layer reduces to its value and output projections; and the embedding is
    read at the ``argmax`` of the prompt's token ids, which for an all-zero
    prompt is position 0. Computed in float32 from the stored weights (which the
    checkpoint holds in float16, as CLIP keeps them on a GPU).

    Args:
        state: The checkpoint's ``state_dict``, with its ``text_encoder.clip.*``
            tensors.

    Returns:
        A ``[width]`` float32 tensor.

    Raises:
        TrackerError: If the checkpoint carries no CLIP text tower.
    """

    def get(name: str) -> Any:
        try:
            return state[_CLIP + name].float()
        except KeyError:
            raise TrackerError(
                f"the checkpoint has no {_CLIP + name}; it is not a SUTrack release"
            ) from None

    def layer_norm(x: Any, prefix: str) -> Any:
        return F.layer_norm(x, x.shape[-1:], get(prefix + ".weight"), get(prefix + ".bias"))

    x = get("token_embedding.weight")[0] + get("positional_embedding")[0]
    width = x.shape[-1]
    layer = 0
    while _CLIP + f"transformer.resblocks.{layer}.ln_1.weight" in state:
        prefix = f"transformer.resblocks.{layer}."
        attention = layer_norm(x, prefix + "ln_1")
        value = F.linear(
            attention,
            get(prefix + "attn.in_proj_weight")[2 * width :],
            get(prefix + "attn.in_proj_bias")[2 * width :],
        )
        x = x + F.linear(
            value, get(prefix + "attn.out_proj.weight"), get(prefix + "attn.out_proj.bias")
        )
        hidden = F.linear(
            layer_norm(x, prefix + "ln_2"),
            get(prefix + "mlp.c_fc.weight"),
            get(prefix + "mlp.c_fc.bias"),
        )
        hidden = hidden * torch.sigmoid(1.702 * hidden)  # CLIP's QuickGELU
        x = x + F.linear(hidden, get(prefix + "mlp.c_proj.weight"), get(prefix + "mlp.c_proj.bias"))
        layer += 1
    if layer == 0:
        raise TrackerError("the checkpoint's CLIP text tower has no layers")
    return layer_norm(x, "ln_final") @ get("text_projection")


def load_sutrack(network: Any, path: str | Path) -> Any:
    """Load a published SUTrack checkpoint into ``network``, insisting on a complete match.

    The keys in :data:`DROPPED_PREFIXES` are discarded (see the module
    docstring); everything else must match the network exactly, with no
    missing or unexpected tensor, or the file is refused rather than tracked
    with half-random weights. The language token's CLIP embedding is computed
    from the dropped CLIP tensors first (:func:`empty_prompt_embedding`).

    Args:
        network: A :class:`SUTrackNet` of the checkpoint's variant.
        path: The checkpoint: upstream's ``SUTRACK_ep0180.pth.tar``, a dict
            with the ``state_dict`` under ``"net"`` (a bare ``state_dict`` is
            accepted too).

    Returns:
        ``network``, with the weights loaded and put in evaluation mode.

    Raises:
        TrackerError: If PyTorch is missing or the checkpoint does not match.
    """
    if torch is None:  # pragma: no cover - guarded by the plugin's availability check
        raise TrackerError("loading SUTrack needs PyTorch")
    # mmap: the file is mostly CLIP, which is read only where the prompt needs it.
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise TrackerError(f"{Path(path).name} holds no state_dict")
    kept = {key: value for key, value in state.items() if not key.startswith(DROPPED_PREFIXES)}
    expected = set(network.state_dict())
    missing = sorted(expected - set(kept))
    unexpected = sorted(set(kept) - expected)
    mismatched = sorted(
        key
        for key, value in network.state_dict().items()
        if key in kept and tuple(kept[key].shape) != tuple(value.shape)
    )
    if missing or unexpected or mismatched:
        raise TrackerError(
            f"{Path(path).name} does not match {network.config.name}: "
            f"{len(missing)} missing tensor(s) (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}), "
            f"{len(mismatched)} of another shape (e.g. {mismatched[:3]})"
        )
    prompt = empty_prompt_embedding(state)
    if tuple(prompt.shape) != tuple(network.text_encoder.prompt.shape):
        raise TrackerError(
            f"{Path(path).name}'s CLIP text width {tuple(prompt.shape)} does not match "
            f"{network.config.name}"
        )
    network.load_state_dict(kept, strict=True)
    with torch.no_grad():
        network.text_encoder.prompt.copy_(prompt)
    return network.eval()
