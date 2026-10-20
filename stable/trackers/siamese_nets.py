"""The Ocean and SiamDW networks in plain PyTorch, ported from TracKit.

These are the two architectures behind :mod:`stable.trackers.ocean` and
:mod:`stable.trackers.siamdw`, rewritten from the authors' own code so that
nothing beyond PyTorch is needed to run them: TracKit itself is a research
checkout (compiled CUDA extensions, TensorRT, benchmark tooling) that cannot be
pip-installed as a dependency, and the parts a tracker needs at inference time
are a few hundred lines of ordinary ``torch.nn``.

What is kept exactly, because the published checkpoints depend on it:

* every module and parameter **name**, so the authors' ``state_dict`` loads
  with no missing or unexpected tensors (see :func:`load_checkpoint`);
* every layer's **geometry** — kernel, stride, padding, dilation and the
  in-layer centre crops — so a 127 px exemplar and a 255 px search region still
  produce the 7x7 / 31x31 (Ocean) and 5x5 / 21x21 (SiamDW) feature maps the
  heads were trained on;
* the **input convention**: BGR channel order, values in ``[0, 255]``, no mean
  or standard-deviation normalisation (TracKit feeds ``cv2.imread`` output
  straight in). :class:`~stable.trackers.siamese.TorchSiameseBackend` does the
  conversion, so callers stay in Stable's uint8 RGB.

What is deliberately different:

* The networks are **stateless**: :meth:`OceanNet.embed` returns the template
  features instead of storing them on the module, so one loaded model can serve
  many points (and many runs) at once.
* The heads take a **batch** of searches, each correlated with its own
  template, plus several search scales per template (SiamDW's pyramid).
* Nothing is moved to CUDA implicitly (TracKit's Ocean head calls ``.cuda()``
  in its constructor), so the CPU fallback works.
* Only the configuration the published weights use is ported: Ocean without
  its deformable-convolution "object-aware" branch (the checkpoint has none),
  and SiamDW's CIResNet-22W backbone with the plain SiamFC correlation head.

This module imports PyTorch at import time and is only ever imported lazily
by the plugins that need it. Without PyTorch it still imports — every class is
then defined on a plain ``object`` base and cannot be built — so tooling that
walks the package (documentation, ``tests/test_public_api.py``) never needs the
optional dependency.

Licence
-------
Ported from TracKit (https://github.com/researchmm/TracKit), whose code and
whose README-linked models are released under the MIT licence:

    MIT License

    Copyright (c) 2019 eccv2020
    (source files: Copyright (c) Microsoft; written by Zhipeng Zhang and
    Houwen Peng)

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

from pathlib import Path
from typing import Any

from ..core.device import torch_module
from ..core.errors import TrackerError

__all__ = [
    "OCEAN_TEMPLATE_CROP",
    "OceanNet",
    "SiamDWNet",
    "ResNet50Dilated",
    "ResNet22W",
    "OceanHead",
    "load_checkpoint",
    "normalize_state_dict",
    "cosine_map",
    "xcorr_depthwise",
    "xcorr_batched",
]

torch: Any = torch_module()
if torch is not None:
    nn = torch.nn
    F = torch.nn.functional
    _Module: Any = torch.nn.Module
else:  # pragma: no cover - only without the 'torch' extra
    nn = F = None
    _Module = object

OCEAN_TEMPLATE_CROP = 4
"""Feature cells trimmed from each side of Ocean's 15x15 template map.

The exemplar's border cells mostly see the zero padding of the backbone's
convolutions rather than the target, so Ocean keeps only the central 7x7.
"""


def xcorr_depthwise(search: Any, kernel: Any) -> Any:
    """Correlate each search map with its own template, channel by channel.

    Args:
        search: ``[B, C, H, W]`` search features.
        kernel: ``[B, C, h, w]`` template features, one per search.

    Returns:
        ``[B, C, H - h + 1, W - w + 1]`` per-channel correlation maps.
    """
    batch, channels = kernel.size(0), kernel.size(1)
    flat = search.reshape(1, batch * channels, search.size(2), search.size(3))
    weight = kernel.reshape(batch * channels, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(flat, weight, groups=batch * channels)
    return out.reshape(batch, channels, out.size(2), out.size(3))


def xcorr_batched(search: Any, kernel: Any) -> Any:
    """Correlate each search map with its own template, summed over channels.

    The SiamFC correlation: one response map per (search, template) pair. A
    grouped convolution keeps every pair separate in one call, which is what
    TracKit's training path does and its single-target test path does not.

    Args:
        search: ``[B, C, H, W]`` search features.
        kernel: ``[B, C, h, w]`` template features, one per search.

    Returns:
        ``[B, H - h + 1, W - w + 1]`` response maps.
    """
    batch = kernel.size(0)
    flat = search.reshape(1, batch * search.size(1), search.size(2), search.size(3))
    out = F.conv2d(flat, kernel, groups=batch)
    return out.reshape(batch, out.size(2), out.size(3))


def cosine_map(search: Any, kernel: Any) -> Any:
    """Cosine similarity between each template and every window of its search map.

    :func:`xcorr_batched` divided by both vectors' norms: the template's, and
    each search window's, computed for every window at once by correlating the
    squared features with a box of ones.

    Args:
        search: ``[B, C, H, W]`` search features.
        kernel: ``[B, C, h, w]`` template features, one per search.

    Returns:
        ``[B, H - h + 1, W - w + 1]`` similarities in ``[-1, 1]``.
    """
    dot = xcorr_batched(search, kernel)
    box = torch.ones(
        (1, 1, kernel.size(2), kernel.size(3)), dtype=search.dtype, device=search.device
    )
    energy = F.conv2d(search.pow(2).sum(1, keepdim=True), box)[:, 0]
    norms = energy.clamp_min(0).sqrt() * kernel.flatten(1).norm(dim=1)[:, None, None]
    return dot / norms.clamp_min(1e-6)


# -- Ocean: ResNet-50 to stage 3, dilated ---------------------------------------


class Bottleneck(_Module):
    """The ResNet-50 bottleneck with the dilation rule Ocean was trained with.

    Args:
        inplanes: Input channels.
        planes: Bottleneck width; the block outputs ``planes * 4`` channels.
        stride: Stride of the 3x3 convolution.
        downsample: Projection for the residual branch, or ``None``.
        dilation: Dilation of the 3x3 convolution. A block that also projects
            its residual uses half of it, which is how a dilated stage enters
            without a seam.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Any = None,
        dilation: int = 1,
    ) -> None:
        """Build the three convolutions and their batch norms."""
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Any) -> Any:
        """Apply the block."""
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class ResNet50Dilated(_Module):
    """ResNet-50 truncated after stage 3, with stage 3 dilated instead of strided.

    Total stride 8 rather than 16, so a 255 px search region still yields a
    31x31 map — the resolution Ocean's box regression is defined on.
    """

    def __init__(self) -> None:
        """Build the stem and the three stages."""
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=1, dilation=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilation: int = 1) -> Any:
        """Return one stage: a projecting block followed by ``blocks - 1`` plain ones."""
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            if stride == 1 and dilation == 1:
                projection = nn.Conv2d(
                    self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False
                )
            else:
                inner = dilation // 2 if dilation > 1 else 1
                projection = nn.Conv2d(
                    self.inplanes,
                    planes * 4,
                    kernel_size=3,
                    stride=stride,
                    bias=False,
                    padding=inner if dilation > 1 else 0,
                    dilation=inner,
                )
            downsample = nn.Sequential(projection, nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        """Return the stage-3 features (1024 channels, stride 8)."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))


class ShrinkChannel(_Module):
    """The neck: a 1x1 projection from 1024 to 256 channels."""

    def __init__(self, in_channels: int = 1024, out_channels: int = 256) -> None:
        """Build the projection."""
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Any) -> Any:
        """Project ``x``."""
        return self.downsample(x)


def _conv_bn_relu(channels_in: int, channels_out: int, dilation: Any = 1) -> Any:
    """A padding-free 3x3 convolution, batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=3, bias=False, dilation=dilation),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class _Matrix(_Module):
    """Ocean's multi-dilation encoder ("matrix" in TracKit).

    Three receptive fields — square, tall and wide — computed for the template
    (``*_k``) and the search (``*_s``) alike, so the correlation that follows
    can match a target whose aspect ratio is not square.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the six encoders."""
        super().__init__()
        self.matrix11_k = _conv_bn_relu(in_channels, out_channels)
        self.matrix11_s = _conv_bn_relu(in_channels, out_channels)
        self.matrix12_k = _conv_bn_relu(out_channels, out_channels, dilation=(2, 1))
        self.matrix12_s = _conv_bn_relu(out_channels, out_channels, dilation=(2, 1))
        self.matrix21_k = _conv_bn_relu(out_channels, out_channels, dilation=(1, 2))
        self.matrix21_s = _conv_bn_relu(out_channels, out_channels, dilation=(1, 2))

    def forward(self, kernel: Any, search: Any) -> tuple[list[Any], list[Any]]:
        """Return the three template encodings and the three search encodings."""
        kernels = [self.matrix11_k(kernel), self.matrix12_k(kernel), self.matrix21_k(kernel)]
        searches = [self.matrix11_s(search), self.matrix12_s(search), self.matrix21_s(search)]
        return kernels, searches


class _GroupDW(_Module):
    """A learned, softmax-weighted sum of the three depthwise correlations."""

    def __init__(self) -> None:
        """Create the three mixing weights."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, kernels: list[Any], searches: list[Any]) -> Any:
        """Correlate each encoding pair and mix the results."""
        weight = F.softmax(self.weight, 0)
        return sum(
            weight[i] * xcorr_depthwise(search, kernel)
            for i, (kernel, search) in enumerate(zip(kernels, searches, strict=True))
        )


def _tower(channels: int, depth: int) -> Any:
    """``depth`` padded 3x3 convolutions with batch norm and ReLU."""
    layers: list[Any] = []
    for _ in range(depth):
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OceanHead(_Module):
    """Ocean's anchor-free head (``FCOSOcean`` in SOTS, ``box_tower`` in TracKit).

    Classification says, per score-map cell, whether the cell lies on the
    target; regression says, per cell, how far the target box's left, top,
    right and bottom edges are from it. A box is therefore available from every
    cell, not only the best-scoring one — which is what makes Ocean's answer
    sub-cell accurate.

    Args:
        in_channels: Channels of the neck's output.
        out_channels: Channels of the encodings and towers.
        towernum: Convolutions per tower.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256, towernum: int = 4) -> None:
        """Build the encoders, the two towers and the prediction layers."""
        super().__init__()
        self.cls_encode = _Matrix(in_channels, out_channels)
        self.reg_encode = _Matrix(in_channels, out_channels)
        self.cls_dw = _GroupDW()
        self.reg_dw = _GroupDW()
        self.bbox_tower = _tower(out_channels, towernum)
        self.cls_tower = _tower(out_channels, towernum)
        self.bbox_pred = nn.Conv2d(out_channels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1, 4, 1, 1))

    def forward(self, search: Any, kernel: Any) -> tuple[Any, Any]:
        """Return ``(cls [B, 1, S, S] logits, reg [B, 4, S, S] edge distances)``."""
        cls_z, cls_x = self.cls_encode(kernel, search)
        reg_z, reg_x = self.reg_encode(kernel, search)
        reg = torch.exp(
            self.adjust * self.bbox_pred(self.bbox_tower(self.reg_dw(reg_z, reg_x))) + self.bias
        )
        cls = 0.1 * self.cls_pred(self.cls_tower(self.cls_dw(cls_z, cls_x)))
        return cls, reg


class OceanNet(_Module):
    """Ocean (Zhang et al., ECCV 2020): backbone, neck and anchor-free head."""

    def __init__(self) -> None:
        """Build an untrained network; load weights with :func:`load_checkpoint`."""
        super().__init__()
        self.backbone = ResNet50Dilated()
        self.neck = ShrinkChannel(1024, 256)
        self.head = OceanHead(256, 256, towernum=4)

    def embed(self, exemplar: Any) -> Any:
        """Return template features for a batch of 127 px exemplars.

        Args:
            exemplar: ``[N, 3, 127, 127]`` BGR in ``[0, 255]``.

        Returns:
            ``[N, 256, 7, 7]`` — the central cells of the 15x15 map.
        """
        features = self.neck(self.backbone(exemplar))
        trim = OCEAN_TEMPLATE_CROP
        return features[:, :, trim:-trim, trim:-trim]

    def match(self, kernel: Any, search: Any) -> tuple[Any, Any, Any]:
        """Score every search crop against its point's template.

        Args:
            kernel: ``[N, 256, 7, 7]`` from :meth:`embed`.
            search: ``[N, S, 3, X, X]`` BGR search crops, ``S`` per template.

        Returns:
            ``(cls [N, S, M, M] logits, reg [N, S, 4, M, M], similarity
            [N, S, M, M])`` with ``M = (X - 127) / 8 + 9``. ``similarity`` is
            not part of Ocean: it is the cosine between the template features
            and the search features under each score cell (the same 7x7
            footprint, so the same grid), which
            :class:`~stable.trackers.ocean.OceanTracker` needs because Ocean's
            own classifier keeps firing on whatever is most target-like when the
            target itself is hidden.
        """
        count, scales = search.shape[0], search.shape[1]
        features = self.neck(self.backbone(search.flatten(0, 1)))
        kernels = kernel.repeat_interleave(scales, dim=0)
        cls, reg = self.head(features, kernels)
        similarity = cosine_map(features, kernels)
        size = cls.shape[-1]
        return (
            cls.reshape(count, scales, size, size),
            reg.reshape(count, scales, 4, size, size),
            similarity.reshape(count, scales, size, size),
        )


# -- SiamDW: CIResNet-22W ------------------------------------------------------------


class BottleneckCI(_Module):
    """SiamDW's cropping-inside-residual (CIR) unit, wide variant.

    The residual is added *before* the padded border is cropped away, so no
    cell of the output ever saw the zero padding — the guideline SiamDW is
    built on, since padding breaks the translation equivariance correlation
    relies on. The 3x3 convolution doubles the width ("W").

    Args:
        inplanes: Input channels.
        planes: Bottleneck width; the block outputs ``planes * 4`` channels.
        last_relu: Whether to apply the final ReLU (the backbone's last unit
            does not, so the correlation sees signed features).
        stride: Stride of the first convolution.
        downsample: Projection for the residual branch, or ``None``.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        last_relu: bool,
        stride: int = 1,
        downsample: Any = None,
    ) -> None:
        """Build the three convolutions and their batch norms."""
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes * 2, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.last_relu = bool(last_relu)

    def forward(self, x: Any) -> Any:
        """Apply the unit, then crop one cell from every side."""
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + residual
        if self.last_relu:
            out = self.relu(out)
        return out[:, :, 1:-1, 1:-1].contiguous()


class ResNet22W(_Module):
    """CIResNet-22W, the SiamDW backbone: padding-free features at stride 8."""

    def __init__(self) -> None:
        """Build the stem and the two stages."""
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, 3, last_relu=True)
        self.layer2 = self._make_layer(128, 4, last_relu=False, stride2pool=True)

    def _make_layer(
        self, planes: int, blocks: int, *, last_relu: bool, stride2pool: bool = False
    ) -> Any:
        """Return one stage; ``stride2pool`` downsamples with a pool after its first unit.

        The pool sits at index 1 of the stage, which is why the checkpoint has no
        ``layer2.1`` parameters.
        """
        downsample = None
        if self.inplanes != planes * BottleneckCI.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers: list[Any] = [BottleneckCI(self.inplanes, planes, True, downsample=downsample)]
        if stride2pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.inplanes = planes * 4
        for index in range(1, blocks):
            final = index == blocks - 1
            layers.append(BottleneckCI(self.inplanes, planes, last_relu if final else True))
        return nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        """Return the stage-2 features (512 channels, stride 8)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x[:, :, 2:-2, 2:-2].contiguous())
        return self.layer2(self.layer1(x))


class SiamDWNet(_Module):
    """SiamDW (Zhang & Peng, CVPR 2019): CIResNet-22W under a SiamFC correlation."""

    response_scale = 0.1
    """The fixed factor TracKit applies to the raw correlation (``Corr_Up``)."""

    def __init__(self) -> None:
        """Build an untrained network; load weights with :func:`load_checkpoint`."""
        super().__init__()
        self.backbone = ResNet22W()

    def embed(self, exemplar: Any) -> Any:
        """Return template features for a batch of 127 px exemplars.

        Args:
            exemplar: ``[N, 3, 127, 127]`` BGR in ``[0, 255]``.

        Returns:
            ``[N, 512, 5, 5]``.
        """
        return self.backbone(exemplar)

    def match(self, kernel: Any, search: Any) -> tuple[Any, None, None]:
        """Correlate every search crop with its point's template.

        Args:
            kernel: ``[N, 512, 5, 5]`` from :meth:`embed`.
            search: ``[N, S, 3, X, X]`` BGR search crops, ``S`` scales per template.

        Returns:
            ``(response [N, S, M, M], None, None)`` with ``M = (X - 127) / 8 + 1``.
            SiamFC regresses no box, and its response already *is* a
            similarity, so the other two elements are always ``None``.
        """
        count, scales = search.shape[0], search.shape[1]
        features = self.backbone(search.flatten(0, 1))
        response = xcorr_batched(features, kernel.repeat_interleave(scales, dim=0))
        size = response.shape[-1]
        return self.response_scale * response.reshape(count, scales, size, size), None, None


def normalize_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Rename a published checkpoint's tensors onto this module's layout.

    Two spellings exist in the wild: SOTS's (``backbone.*``, ``neck.*``,
    ``head.*``), which is this module's, and TracKit's older one, which nests
    the backbone as ``features.features.*`` — the SiamDW release uses it. A
    ``module.`` prefix (left by ``nn.DataParallel`` during training) is dropped
    too, exactly as TracKit's own loader does.

    Args:
        state: The checkpoint's tensors, or a training checkpoint holding them
            under ``state_dict``.

    Returns:
        A new mapping using this module's parameter names.
    """
    tensors = state.get("state_dict", state) if isinstance(state, dict) else state
    renamed: dict[str, Any] = {}
    for key, value in tensors.items():
        name = key[len("module.") :] if key.startswith("module.") else key
        if name.startswith("features.features."):
            name = "backbone." + name[len("features.features.") :]
        renamed[name] = value
    return renamed


def load_checkpoint(network: Any, path: str | Path) -> Any:
    """Load a published checkpoint into ``network``, insisting on a complete match.

    Every parameter and buffer must be present except ``num_batches_tracked``:
    the SiamDW release predates that counter, and it only matters for training.
    Anything else missing or unexpected means the file is not the network this
    module builds, and loading it anyway would track with random weights — so it
    is refused instead.

    Args:
        network: An :class:`OceanNet` or :class:`SiamDWNet`.
        path: The checkpoint file.

    Returns:
        ``network``, with the weights loaded and put in evaluation mode.

    Raises:
        TrackerError: If PyTorch is missing or the checkpoint does not match.
    """
    if torch is None:  # pragma: no cover - guarded by the plugins' availability check
        raise TrackerError("loading a Siamese tracker checkpoint needs PyTorch")
    state = normalize_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    expected = set(network.state_dict())
    missing = sorted(k for k in expected - set(state) if not k.endswith("num_batches_tracked"))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        raise TrackerError(
            f"{Path(path).name} does not match {type(network).__name__}: "
            f"{len(missing)} missing tensor(s) (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})"
        )
    network.load_state_dict(state, strict=False)
    return network.eval()
