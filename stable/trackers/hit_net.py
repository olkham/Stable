"""The HiT network in plain PyTorch, ported from the authors' release.

HiT (Kang, Chen, Wang, Yan & Lu, ICCV 2023, "Exploring Lightweight Hierarchical
Vision Transformers for Efficient Visual Tracking") is three parts, all here:

* a **LeViT** backbone that reads the search region and the template *together*:
  both are cut into 16 px patches by a small convolutional stem, and every
  attention layer attends over the tokens of both images at once;
* the **Bridge Module**, which upsamples the backbone's last (4x4) stage twice
  and adds the two earlier stages back in, so the head sees 16x16 features with
  both the deep stage's semantics and the shallow stages' detail;
* a **corner head** (STARK's): two small convolution towers score every cell
  as the box's top-left and bottom-right corner, and a soft-argmax over each
  map gives the box.

The one idea specific to HiT is the **dual-image position encoding**. LeViT
encodes position as a learned bias per attention head and per spatial offset
between two tokens. HiT places the template's token grid on the same plane as
the search's, diagonally after it, so the same table also encodes "how far is
this template token from that search token": the index tables are built by
:func:`_dual_image_points` and :func:`_offset_index` exactly as the authors
build them, and the published checkpoints carry them too, so loading checks
that the two agree (:func:`load_checkpoint`).

What is kept exactly, because the published checkpoints depend on it:

* every module and parameter **name** (``backbone.body.blocks.3.m.qkv.c.weight``
  and so on), so the authors' ``state_dict`` loads with no missing or unexpected
  tensor;
* the **geometry**: a 128 px template and a 256 px search, 16 px patches, two
  attention-subsampling stages (16x16 to 8x8 to 4x4 search tokens), the search's
  tokens first, and the corner grid at the top-left of each 16 px cell;
* the **input convention**: RGB, scaled to ``[0, 1]`` and normalised with the
  ImageNet mean and standard deviation (the plugin's backend does that).

What is deliberately different:

* The network is **stateless and split in two**: :meth:`HiTNet.embed` runs the
  part of the template path that does not depend on the search (the patch
  stem), so a template is embedded once per run; :meth:`HiTNet.match` does the
  rest for a batch of searches, each with its own template.
* :meth:`HiTNet.match` returns the two corner maps' **logits** alongside the
  box, so the plugin can read how sure the network is; HiT has no score head of
  its own.
* Nothing is moved to CUDA implicitly (the authors' corner head calls
  ``.cuda()`` in its constructor), so the CPU fallback works, and the attention
  biases are gathered on every call rather than cached when ``eval()`` is
  called, so loading weights after switching modes cannot leave a stale cache.
* Folding the batch norms into the layers before them, which the authors' test
  harness does with ``replace_batchnorm``, is :meth:`HiTNet.fuse`.
* Only the configuration of the published weights is ported: the LeViT
  backbone with the ``"FB"`` neck and the ``"CORNER"`` head. The PVT backbone,
  the other necks and heads, the position-encoding ablations and DyHiT are not.

This module imports PyTorch at import time and is only ever imported lazily by
the plugin. Without PyTorch it still imports — every class is then defined on a
plain ``object`` base and cannot be built — so tooling that walks the package
never needs the optional dependency.

Licence
-------
Ported from HiT (https://github.com/kangben258/HiT), released under the MIT
licence:

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

HiT's backbone (``lib/models/HiT/levit.py``, here :class:`LeViT` and the layers
it is built from) is in turn modified from LeViT
(https://github.com/facebookresearch/LeViT), which carries this notice and is
licensed under the Apache License, Version 2.0:

    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
    License for the specific language governing permissions and limitations
    under the License.

The port to this module (stateless template/search split, no implicit CUDA,
the corner logits returned, untrained-configuration code removed) is Stable's
modification of those files.
"""

from __future__ import annotations

import io
import itertools
import pickle
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.device import torch_module
from ..core.errors import TrackerError

__all__ = [
    "HIT_CONFIGS",
    "HiTConfig",
    "HiTNet",
    "LeViT",
    "BridgeModule",
    "CornerHead",
    "load_checkpoint",
    "read_release_state_dict",
]

torch: Any = torch_module()
if torch is not None:
    nn = torch.nn
    F = torch.nn.functional
    _Module: Any = torch.nn.Module
    _Sequential: Any = torch.nn.Sequential
else:  # pragma: no cover - only without the 'torch' extra
    nn = F = None
    _Module = _Sequential = object


@dataclass(frozen=True)
class HiTConfig:
    """The shape of one HiT model.

    Attributes:
        embed_dim: Token width of each of the backbone's three stages.
        key_dim: Width of every attention head's query and key.
        num_heads: Attention heads in each stage.
        depth: Attention + MLP block pairs in each stage.
        template_size: Template side the model was trained at, in pixels.
        search_size: Search side the model was trained at, in pixels.
        patch_size: Pixels per token side after the convolutional stem.
        hidden_dim: Width of the Bridge Module's output, which the head reads.
        head_channels: Width of the corner head's first convolution.
    """

    embed_dim: tuple[int, int, int]
    key_dim: int
    num_heads: tuple[int, int, int]
    depth: tuple[int, int, int]
    template_size: int = 128
    search_size: int = 256
    patch_size: int = 16
    hidden_dim: int = 256
    head_channels: int = 256

    @property
    def feature_size(self) -> int:
        """Side of the corner head's maps: one cell per search token."""
        return self.search_size // self.patch_size


HIT_CONFIGS: dict[str, HiTConfig] = {
    # LeViT_384 under HiT-Base, LeViT_128 under HiT-Small, LeViT_128S under
    # HiT-Tiny: the authors' experiments/HiT/HiT_{Base,Small,Tiny}.yaml.
    "base": HiTConfig(embed_dim=(384, 512, 768), key_dim=32, num_heads=(6, 9, 12), depth=(4, 4, 4)),
    "small": HiTConfig(
        embed_dim=(128, 256, 384), key_dim=16, num_heads=(4, 8, 12), depth=(4, 4, 4)
    ),
    "tiny": HiTConfig(embed_dim=(128, 256, 384), key_dim=16, num_heads=(4, 6, 8), depth=(2, 3, 4)),
}
"""The published models' configurations, by variant name."""

_ATTN_RATIO = 2
"""Value width over key width in the stages' attention layers."""
_MLP_RATIO = 2
"""Hidden width over token width in the MLP blocks."""
_DOWN_ATTN_RATIO = 4
"""Value width over key width in the two subsampling attention layers."""
_STRIDE = 2
"""Token-grid subsampling between stages."""


class ConvBN(_Sequential):
    """A bias-free convolution and its batch norm (LeViT's ``Conv2d_BN``).

    Args:
        channels_in: Input channels.
        channels_out: Output channels.
        kernel: Kernel side.
        stride: Stride.
        padding: Zero padding on each side.
    """

    def __init__(
        self, channels_in: int, channels_out: int, kernel: int, stride: int, padding: int
    ) -> None:
        """Build the two layers under LeViT's names, ``c`` and ``bn``."""
        super().__init__()
        self.add_module(
            "c", nn.Conv2d(channels_in, channels_out, kernel, stride, padding, bias=False)
        )
        self.add_module("bn", nn.BatchNorm2d(channels_out))

    def fuse(self) -> Any:
        """Return one convolution computing the same as this pair in evaluation mode."""
        conv, bn = self.c, self.bn
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
        ).to(conv.weight.device)
        fused.weight.data.copy_(conv.weight * scale[:, None, None, None])
        fused.bias.data.copy_(bn.bias - bn.running_mean * scale)
        return fused


class LinearBN(_Sequential):
    """A bias-free linear layer over tokens and its batch norm (LeViT's ``Linear_BN``).

    Args:
        features_in: Input width.
        features_out: Output width.
    """

    def __init__(self, features_in: int, features_out: int) -> None:
        """Build the two layers under LeViT's names, ``c`` and ``bn``."""
        super().__init__()
        self.add_module("c", nn.Linear(features_in, features_out, bias=False))
        self.add_module("bn", nn.BatchNorm1d(features_out))

    def forward(self, x: Any) -> Any:
        """Apply the layer to ``[B, N, C]`` tokens, normalising over batch and tokens."""
        x = self.c(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)

    def fuse(self) -> Any:
        """Return one linear layer computing the same as this pair in evaluation mode."""
        linear, bn = self.c, self.bn
        scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
        fused = nn.Linear(linear.in_features, linear.out_features).to(linear.weight.device)
        fused.weight.data.copy_(linear.weight * scale[:, None])
        fused.bias.data.copy_(bn.bias - bn.running_mean * scale)
        return fused


class Residual(_Module):
    """``x + m(x)``; the stochastic depth used in training is not ported.

    Args:
        m: The residual branch.
    """

    def __init__(self, m: Any) -> None:
        """Wrap ``m`` under LeViT's name for it."""
        super().__init__()
        self.m = m

    def forward(self, x: Any) -> Any:
        """Add the branch to its input."""
        return x + self.m(x)


def _dual_image_points(resolution_x: int, resolution_z: int) -> list[tuple[int, int]]:
    """Return every token's position on HiT's shared plane, search first.

    The search's tokens sit at ``(0..rx-1, 0..rx-1)``; the template's continue
    diagonally after them, at ``(rx..rx+rz-1, rx..rx+rz-1)``. This is the
    dual-image position encoding: one bias table spans both images.
    """
    points = list(itertools.product(range(resolution_x), range(resolution_x)))
    band = range(resolution_x, resolution_x + resolution_z)
    return points + list(itertools.product(band, band))


def _offset_index(
    queries: Iterable[tuple[int, int]], keys: list[tuple[int, int]], stride: int = 1
) -> tuple[int, list[int]]:
    """Number every distinct ``|query * stride - key|`` offset in first-seen order.

    Returns:
        ``(distinct offsets, one index per (query, key) pair, row-major)`` —
        LeViT's ``attention_offsets`` and ``attention_bias_idxs``.
    """
    offsets: dict[tuple[int, int], int] = {}
    indices: list[int] = []
    for q in queries:
        for k in keys:
            offset = (abs(q[0] * stride - k[0]), abs(q[1] * stride - k[1]))
            indices.append(offsets.setdefault(offset, len(offsets)))
    return len(offsets), indices


class Attention(_Module):
    """Multi-head self-attention over both images' tokens, with LeViT's position biases.

    Args:
        dim: Token width.
        key_dim: Query/key width per head.
        num_heads: Heads.
        attn_ratio: Value width over ``key_dim``.
        resolution_x: Search tokens per side.
        resolution_z: Template tokens per side.
    """

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: int,
        resolution_x: int,
        resolution_z: int,
    ) -> None:
        """Build the projections and the dual-image bias table."""
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.qkv = LinearBN(dim, self.dh + key_dim * num_heads * 2)
        self.proj = nn.Sequential(nn.Hardswish(), LinearBN(self.dh, dim))
        points = _dual_image_points(resolution_x, resolution_z)
        count, indices = _offset_index(points, points)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, count))
        self.register_buffer(
            "attention_bias_idxs", torch.tensor(indices, dtype=torch.long).view(len(points), -1)
        )

    def forward(self, x: Any) -> Any:
        """Attend over ``[B, N, C]`` tokens."""
        batch, tokens, _ = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch, tokens, self.num_heads, -1)
            .split([self.key_dim, self.key_dim, self.d], dim=3)
        )
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.attention_biases[
            :, self.attention_bias_idxs
        ]
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, tokens, self.dh)
        return self.proj(x)


class Subsample(_Module):
    """Keep every ``stride``-th token of each image's grid, rows and columns.

    Args:
        stride: Subsampling step.
        resolution_x: Search tokens per side, before.
        resolution_z: Template tokens per side, before.
    """

    def __init__(self, stride: int, resolution_x: int, resolution_z: int) -> None:
        """Record the two grids' sides."""
        super().__init__()
        self.stride = stride
        self.resolution_x = resolution_x
        self.resolution_z = resolution_z

    def forward(self, x: Any) -> Any:
        """Subsample the search's grid and the template's, keeping them in that order."""
        batch, _, channels = x.shape
        rx, rz, s = self.resolution_x, self.resolution_z, self.stride
        search = x[:, : rx * rx].view(batch, rx, rx, channels)[:, ::s, ::s]
        template = x[:, rx * rx : rx * rx + rz * rz].view(batch, rz, rz, channels)[:, ::s, ::s]
        return torch.cat(
            (search.reshape(batch, -1, channels), template.reshape(batch, -1, channels)), dim=1
        )


class AttentionSubsample(_Module):
    """Attention whose queries are a subsampled grid: LeViT's shrinking stage transition.

    Args:
        in_dim: Token width in.
        out_dim: Token width out.
        key_dim: Query/key width per head.
        num_heads: Heads.
        attn_ratio: Value width over ``key_dim``.
        stride: Query subsampling step.
        resolution_x: Search tokens per side, before.
        resolution_z: Template tokens per side, before.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key_dim: int,
        num_heads: int,
        attn_ratio: int,
        stride: int,
        resolution_x: int,
        resolution_z: int,
    ) -> None:
        """Build the projections and the bias table between the two grids."""
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.kv = LinearBN(in_dim, self.dh + key_dim * num_heads)
        self.q = nn.Sequential(
            Subsample(stride, resolution_x, resolution_z), LinearBN(in_dim, key_dim * num_heads)
        )
        self.proj = nn.Sequential(nn.Hardswish(), LinearBN(self.dh, out_dim))
        points = _dual_image_points(resolution_x, resolution_z)
        smaller = _dual_image_points(
            (resolution_x - 1) // stride + 1, (resolution_z - 1) // stride + 1
        )
        count, indices = _offset_index(smaller, points, stride)
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, count))
        self.register_buffer(
            "attention_bias_idxs", torch.tensor(indices, dtype=torch.long).view(len(smaller), -1)
        )

    def forward(self, x: Any) -> Any:
        """Attend from the subsampled grid to the full one: ``[B, N, C]`` to ``[B, N', C']``."""
        batch, tokens, _ = x.shape
        k, v = (
            self.kv(x).view(batch, tokens, self.num_heads, -1).split([self.key_dim, self.d], dim=3)
        )
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        q = self.q(x).view(batch, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.attention_biases[
            :, self.attention_bias_idxs
        ]
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, -1, self.dh)
        return self.proj(x)


def _mlp(dim: int) -> Any:
    """Return LeViT's residual MLP block for tokens of width ``dim``."""
    hidden = dim * _MLP_RATIO
    return Residual(nn.Sequential(LinearBN(dim, hidden), nn.Hardswish(), LinearBN(hidden, dim)))


class LeViT(_Module):
    """HiT's backbone: LeViT over the search and template tokens together.

    Args:
        config: The model's shape.

    Attributes:
        fb_idx: Positions in ``blocks`` of the two subsampling layers; the
            Bridge Module reads the tokens just before each.
        num_patches_search: Search tokens after the last stage (16 at 256 px).
    """

    def __init__(self, config: HiTConfig) -> None:
        """Build the convolutional stem and the three stages."""
        super().__init__()
        dims = config.embed_dim
        act = nn.Hardswish
        self.patch_embed = nn.Sequential(
            ConvBN(3, dims[0] // 8, 3, 2, 1),
            act(),
            ConvBN(dims[0] // 8, dims[0] // 4, 3, 2, 1),
            act(),
            ConvBN(dims[0] // 4, dims[0] // 2, 3, 2, 1),
            act(),
            ConvBN(dims[0] // 2, dims[0], 3, 2, 1),
        )
        rx = config.search_size // config.patch_size
        rz = config.template_size // config.patch_size
        blocks: list[Any] = []
        fb_idx: list[int] = []
        for stage, (dim, heads, depth) in enumerate(
            zip(dims, config.num_heads, config.depth, strict=True)
        ):
            for _ in range(depth):
                blocks.append(Residual(Attention(dim, config.key_dim, heads, _ATTN_RATIO, rx, rz)))
                blocks.append(_mlp(dim))
            if stage < len(dims) - 1:
                fb_idx.append(len(blocks))
                blocks.append(
                    AttentionSubsample(
                        dim,
                        dims[stage + 1],
                        key_dim=config.key_dim,
                        num_heads=dim // config.key_dim,
                        attn_ratio=_DOWN_ATTN_RATIO,
                        stride=_STRIDE,
                        resolution_x=rx,
                        resolution_z=rz,
                    )
                )
                rx, rz = (rx - 1) // _STRIDE + 1, (rz - 1) // _STRIDE + 1
                blocks.append(_mlp(dims[stage + 1]))
        self.blocks = nn.Sequential(*blocks)
        self.fb_idx = fb_idx
        self.num_patches_search = rx * rx

    def tokens(self, image: Any) -> Any:
        """Return ``[B, H/16 * W/16, C]`` patch tokens for normalised ``[B, 3, H, W]`` images."""
        return self.patch_embed(image).flatten(2).transpose(1, 2)

    def forward(self, search_tokens: Any, template_tokens: Any) -> list[Any]:
        """Run the three stages over the search's tokens followed by the template's.

        Returns:
            ``[stage 1, stage 2, stage 3]`` tokens, the last with their mean
            prepended as a global token — HiT's ``forward_features`` output.
        """
        xz = torch.cat((search_tokens, template_tokens), dim=1)
        first, second = self.fb_idx
        xz1 = self.blocks[0:first](xz)
        xz2 = self.blocks[first:second](xz1)
        xz3 = self.blocks[second:](xz2)
        return [xz1, xz2, torch.cat((xz3.mean(1, keepdim=True), xz3), dim=1)]


class HiTBackbone(_Module):
    """Holds the backbone under the authors' name for it, ``backbone.body``.

    Args:
        config: The model's shape.
    """

    def __init__(self, config: HiTConfig) -> None:
        """Build the LeViT body."""
        super().__init__()
        self.body = LeViT(config)


class BridgeModule(_Module):
    """HiT's Bridge Module (the authors' ``NECK_FB``): three stages fused at 16x16.

    The last stage's 4x4 search tokens are upsampled by a stride-2 transposed
    convolution, the second stage's 8x8 tokens added, upsampled again and the
    first stage's 16x16 tokens added, then projected to ``hidden_dim``. The
    global token is projected alongside.

    Args:
        config: The model's shape.
        num_x: Search tokens in the last stage.
    """

    def __init__(self, config: HiTConfig, num_x: int) -> None:
        """Build the two upsampling steps and the two projections."""
        super().__init__()
        dims = config.embed_dim
        self.num_x = num_x
        self.num_layers = len(dims) - 1
        reverse = dims[::-1]
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(n, k, (_STRIDE, _STRIDE), (_STRIDE, _STRIDE)), nn.BatchNorm2d(k)
            )
            for n, k in zip(reverse[:-1], reverse[1:], strict=True)
        )
        self.proj1 = nn.Linear(dims[0], config.hidden_dim)
        self.proj2 = nn.Linear(dims[-1], config.hidden_dim)

    def forward(self, xz_list: list[Any]) -> tuple[Any, Any]:
        """Fuse the stages' search tokens.

        Returns:
            ``(global [B, hidden], search features [B, hidden, 16, 16])``.
        """
        last = len(xz_list) - 1
        maps = []
        for i, xz in enumerate(xz_list):
            x = xz[:, 1 : self.num_x + 1] if i == last else xz[:, : self.num_x * 4 ** (last - i)]
            batch, count, channels = x.shape
            side = int(round(count**0.5))
            maps.append(x.transpose(1, 2).reshape(batch, channels, side, side))
        x = maps[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x) + maps[-2 - i]
            if i < self.num_layers - 1:
                x = F.relu(x)
        features = self.proj1(x.flatten(2).transpose(1, 2))
        global_token = self.proj2(xz_list[-1][:, 0])
        batch, _, height, width = x.shape
        return global_token, features.transpose(1, 2).reshape(batch, -1, height, width)


def _conv_bn_relu(channels_in: int, channels_out: int) -> Any:
    """Return the corner head's 3x3 convolution, batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class CornerHead(_Module):
    """STARK's corner predictor: a score map per corner, read by soft-argmax.

    Args:
        inplanes: Feature width in.
        channel: Width of the first convolution; each next one halves it.
        feat_sz: Side of the feature map.
        stride: Search pixels per feature cell.

    Attributes:
        img_sz: The search side the coordinates are normalised by.
    """

    def __init__(self, inplanes: int, channel: int, feat_sz: int, stride: int) -> None:
        """Build the two convolution towers under the authors' names."""
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = feat_sz * stride
        for corner in ("tl", "br"):
            setattr(self, f"conv1_{corner}", _conv_bn_relu(inplanes, channel))
            setattr(self, f"conv2_{corner}", _conv_bn_relu(channel, channel // 2))
            setattr(self, f"conv3_{corner}", _conv_bn_relu(channel // 2, channel // 4))
            setattr(self, f"conv4_{corner}", _conv_bn_relu(channel // 4, channel // 8))
            setattr(self, f"conv5_{corner}", nn.Conv2d(channel // 8, 1, kernel_size=1))

    def score_maps(self, x: Any) -> tuple[Any, Any]:
        """Return the top-left and bottom-right logits, each ``[B, feat_sz, feat_sz]``."""
        out = []
        for corner in ("tl", "br"):
            y = x
            for depth in range(1, 6):
                y = getattr(self, f"conv{depth}_{corner}")(y)
            out.append(y[:, 0])
        return out[0], out[1]

    def soft_argmax(self, logits: Any) -> tuple[Any, Any]:
        """Return the expected ``(x, y)`` under ``softmax(logits)``, in search pixels.

        The grid is each cell's top-left pixel, ``0, 16, ... 240`` — the
        authors' ``indice * stride``.
        """
        batch = logits.shape[0]
        prob = logits.reshape(batch, -1).softmax(dim=1).view(batch, self.feat_sz, self.feat_sz)
        grid = torch.arange(self.feat_sz, device=logits.device, dtype=prob.dtype) * self.stride
        return (prob.sum(1) * grid).sum(1), (prob.sum(2) * grid).sum(1)

    def forward(self, x: Any) -> Any:
        """Return ``[B, 4]`` boxes, ``(x0, y0, x1, y1)`` as fractions of the search side."""
        top_left, bottom_right = self.score_maps(x)
        x0, y0 = self.soft_argmax(top_left)
        x1, y1 = self.soft_argmax(bottom_right)
        return torch.stack((x0, y0, x1, y1), dim=1) / self.img_sz


class HiTNet(_Module):
    """HiT: the LeViT backbone, the Bridge Module and the corner head.

    Args:
        config: A :class:`HiTConfig` or the name of a published one
            (``"base"``, ``"small"`` or ``"tiny"``; see :data:`HIT_CONFIGS`).

    Raises:
        TrackerError: For an unknown configuration name.
    """

    def __init__(self, config: HiTConfig | str = "base") -> None:
        """Build an untrained HiT of the given shape."""
        super().__init__()
        if isinstance(config, str):
            if config not in HIT_CONFIGS:
                raise TrackerError(
                    f"unknown HiT variant {config!r}; choose from {sorted(HIT_CONFIGS)}"
                )
            config = HIT_CONFIGS[config]
        self.config = config
        self.backbone = HiTBackbone(config)
        num_x = self.backbone.body.num_patches_search
        self.bottleneck = BridgeModule(config, num_x)
        self.box_head = CornerHead(
            config.hidden_dim, config.head_channels, config.feature_size, config.patch_size
        )

    def embed(self, template: Any) -> Any:
        """Return the template's patch tokens: everything that does not depend on the search.

        Args:
            template: ``[B, 3, 128, 128]`` normalised RGB.

        Returns:
            ``[B, 64, C]`` tokens, one batch row per template.
        """
        return self.backbone.body.tokens(template)

    def match(self, kernel: Any, search: Any) -> tuple[Any, Any, Any, Any]:
        """Find each template in its own search region.

        Args:
            kernel: ``[B, 64, C]`` tokens from :meth:`embed`.
            search: ``[B, 3, 256, 256]`` normalised RGB, one per template.

        Returns:
            ``(boxes, top_left, bottom_right, similarity)``: ``[B, 4]`` boxes as
            ``(cx, cy, w, h)`` fractions of the search side — exactly the
            authors' ``pred_boxes`` — each corner's ``[B, 16, 16]`` logits, and
            the ``[B, 16, 16]`` :meth:`appearance` map. The last is Stable's,
            not part of HiT's output; it reuses features the backbone computed
            anyway.
        """
        xz = self.backbone.body(self.backbone.body.tokens(search), kernel)
        global_token, features = self.bottleneck(xz)
        # The authors' "corner" head reads the search features weighted by
        # their dot product with the global token (HiT.forward_box_head).
        weight = torch.einsum("bchw,bc->bhw", features, global_token)
        top_left, bottom_right = self.box_head.score_maps(features * weight[:, None])
        x0, y0 = self.box_head.soft_argmax(top_left)
        x1, y1 = self.box_head.soft_argmax(bottom_right)
        side = float(self.box_head.img_sz)
        boxes = torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=1) / side
        return boxes, top_left, bottom_right, self.appearance(xz[0])

    def appearance(self, tokens: Any) -> Any:
        """Cosine between the template's target and every same-sized window of the search.

        Read from the first stage's tokens — 16 px apart, already compared with
        each other by that stage's attention but not yet reduced to the deep
        stages' "where is the object" — the template's central half (the
        target: the template is cut at twice its size) is correlated with every
        window of the search's token grid, normalised by both norms. A search
        window holding the target scores close to the template's own match; an
        occluder or a lookalike scores lower, which the corner maps alone do
        not tell apart (see :mod:`stable.trackers.hit`).

        Args:
            tokens: ``[B, Nx + Nz, C]`` first-stage tokens, search first.

        Returns:
            ``[B, Mx, Mx]`` similarities in ``[-1, 1]``. Cell ``(i, j)`` is the
            window centred on search pixel ``(16 j, 16 i)`` (pixel-edge
            coordinates), the corner grid's own positions.
        """
        cfg = self.config
        rx = cfg.search_size // cfg.patch_size
        rz = cfg.template_size // cfg.patch_size
        batch, _, channels = tokens.shape
        side = rz // 2
        start = (rz - side) // 2
        search = tokens[:, : rx * rx].transpose(1, 2).reshape(batch, channels, rx, rx)
        template = tokens[:, rx * rx : rx * rx + rz * rz].transpose(1, 2)
        kernel = template.reshape(batch, channels, rz, rz)[
            :, :, start : start + side, start : start + side
        ].contiguous()
        padded = F.pad(search, (side // 2, side - 1 - side // 2, side // 2, side - 1 - side // 2))
        dot = F.conv2d(padded.reshape(1, batch * channels, *padded.shape[2:]), kernel, groups=batch)
        ones = torch.ones((1, 1, side, side), dtype=padded.dtype, device=padded.device)
        energy = F.conv2d(padded.pow(2).sum(1, keepdim=True), ones)[:, 0]
        norms = energy.clamp_min(0).sqrt() * kernel.flatten(1).norm(dim=1)[:, None, None]
        return dot.reshape(batch, rx, rx) / norms.clamp_min(1e-6)

    def forward(self, search: Any, template: Any) -> Any:
        """Return ``[B, 4]`` ``(cx, cy, w, h)`` boxes for normalised search and template images."""
        return self.match(self.embed(template), search)[0]

    def fuse(self) -> HiTNet:
        """Fold every backbone batch norm into its layer, as the authors' test harness does.

        Call it after the weights are loaded and before inference; it changes
        the module tree, so a fused network no longer loads a checkpoint.

        Returns:
            ``self``.
        """
        _fuse_children(self.backbone.body)
        return self


def _fuse_children(module: Any) -> None:
    """Replace every :class:`ConvBN`/:class:`LinearBN` under ``module`` by its fused layer."""
    for name, child in module.named_children():
        if isinstance(child, ConvBN | LinearBN):
            setattr(module, name, child.fuse())
        else:
            _fuse_children(child)


class _Inert:
    """Stand-in for the training bookkeeping objects pickled beside the weights."""

    def __setstate__(self, state: Any) -> None:
        """Discard the pickled state: none of it is needed to track."""


_RELEASE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch", "device"),
}
"""The only callables a HiT checkpoint may name, besides storages and :class:`_Inert`."""

_TRAINING_MODULES = ("lib.train.admin.",)
"""Modules of the authors' training harness whose objects the checkpoints pickle."""


class _ReleaseUnpickler(pickle.Unpickler):
    """An allow-list unpickler for the authors' training checkpoints.

    The published files are whole training checkpoints: next to ``net`` they
    pickle the optimiser state and the harness's settings and statistics, as
    instances of classes from the authors' ``lib.train.admin`` package. That
    rules out ``torch.load(weights_only=True)`` and a plain unpickler would
    import (and run) whatever the file names, so this one resolves only the
    tensor-rebuilding helpers and turns every training-harness object into an
    inert placeholder. Anything else is refused.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Resolve an allowed global, or refuse the file."""
        if module.startswith(_TRAINING_MODULES):
            return _Inert
        if (module, name) in _RELEASE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"HiT checkpoint names an unexpected global {module}.{name}")


def _pickle_module() -> Any:
    """Return a ``pickle``-like module whose ``Unpickler`` is :class:`_ReleaseUnpickler`."""
    module = types.ModuleType("stable_hit_release_pickle")
    module.Unpickler = _ReleaseUnpickler  # type: ignore[attr-defined]
    module.load = lambda file, **kw: _ReleaseUnpickler(file, **kw).load()  # type: ignore[attr-defined]
    module.loads = lambda data, **kw: _ReleaseUnpickler(io.BytesIO(data), **kw).load()  # type: ignore[attr-defined]
    return module


def read_release_state_dict(path: str | Path) -> dict[str, Any]:
    """Return the network weights from one of the authors' checkpoint files.

    Args:
        path: A ``HiT_ep1500.pth.tar``-style training checkpoint (weights under
            ``net``), or a plain ``state_dict`` saved with ``torch.save``.

    Returns:
        The tensors, by parameter name.

    Raises:
        TrackerError: If PyTorch is missing or the file is not a checkpoint of
            the expected shape.
    """
    if torch is None:  # pragma: no cover - guarded by the plugin's availability check
        raise TrackerError("loading a HiT checkpoint needs PyTorch")
    try:
        # weights_only cannot be combined with a custom unpickler; the
        # allow-list above is the same protection, extended to the harness's
        # bookkeeping classes.
        payload = torch.load(
            str(path), map_location="cpu", pickle_module=_pickle_module(), weights_only=False
        )
    except Exception as exc:
        raise TrackerError(f"{Path(path).name} is not a readable HiT checkpoint: {exc}") from exc
    state = payload.get("net", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise TrackerError(f"{Path(path).name} holds no HiT state_dict")
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


def load_checkpoint(network: Any, path: str | Path) -> Any:
    """Load a published checkpoint into ``network``, insisting on an exact match.

    Every parameter and buffer must be present and none may be left over —
    ``load_state_dict(strict=True)``. That includes the dual-image position
    encoding's index tables, which the checkpoints carry: they are compared
    with the ones :class:`Attention` builds, so a port that numbered the
    offsets differently would be refused rather than track with every bias
    applied to the wrong token pair.

    Args:
        network: A :class:`HiTNet` of the checkpoint's variant, not yet fused.
        path: The checkpoint file.

    Returns:
        ``network``, with the weights loaded and put in evaluation mode.

    Raises:
        TrackerError: If PyTorch is missing or the checkpoint does not match.
    """
    state = read_release_state_dict(path)
    built = network.state_dict()
    missing = sorted(set(built) - set(state))
    unexpected = sorted(set(state) - set(built))
    if missing or unexpected:
        raise TrackerError(
            f"{Path(path).name} does not match {type(network).__name__}: "
            f"{len(missing)} missing tensor(s) (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    for key, value in state.items():
        if key.endswith("attention_bias_idxs") and not torch.equal(value, built[key]):
            raise TrackerError(f"{Path(path).name}: position-encoding table {key} differs")
    try:
        network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise TrackerError(
            f"{Path(path).name} does not match {type(network).__name__}: {exc}"
        ) from exc
    return network.eval()
