"""HiT — a lightweight hierarchical vision transformer tracker, as a Stable plugin.

HiT (Kang, Chen, Wang, Yan & Lu, ICCV 2023) belongs to the generation after
Siamese correlation: one transformer reads the template and the search region
*together*, every attention layer comparing tokens of both, and a corner head
reads the box straight off the result. What HiT adds is speed. Its backbone is
LeViT — hierarchical, so most of its layers run on 8x8 and 4x4 token grids —
and a "Bridge Module" puts the detail of the early stages back before the
head, which is how it closes most of the accuracy gap to the large one-stream
trackers at a fraction of their cost (LaSOT AUC 64.6 for HiT-Base against 64.9
for TransT, at nearly three times its frame rate).

How it is used here
-------------------
* The network is the published one, ported to plain PyTorch in
  :mod:`stable.trackers.hit_net`: the checkpoints load with
  ``load_state_dict(strict=True)``, and on identical inputs the port, its
  batch norms folded as the plugin runs it, reproduces the authors' own module
  code to within 3e-4 search pixels on the box and 1e-4 on the corner logits
  (logits range over about 60), all three variants, on the GPU and the CPU.
  With cuDNN's default TF32 convolutions the GPU differs by up to 0.02 px.
* It is driven by :class:`~stable.trackers.siamese.SiameseTrackerPlugin`, like
  Ocean and SiamDW: exact crops, the search radius, the point riding on the
  network's box, holding through a loss. HiT's input sizes map onto that
  geometry exactly: its template is cut at twice the target and its search at
  four times, which for a square target is SiamFC's context and the 127/255
  crops the base class cuts; the backend resamples them to HiT's 128/256 (a
  0.4-0.8% scale change). For a box far from square the context is SiamFC's,
  up to 10% wider than HiT's at an aspect of 2.75, inside what HiT saw in
  training. Crops leaving the picture are padded with the frame's mean colour
  rather than HiT's black.
* The box is HiT's: the soft-argmax of each corner map, followed every frame
  as the authors' tracker does (``box_size_lr`` 1, and no gate on how far the
  shape may change — HiT's box is what it was trained to output; the gate that
  steadies Ocean held one stretching object's box at a stale shape). A point,
  whose "box" is HiT's guess at what object a patch belongs to, follows the
  size at ``size_lr`` 0.3 and not at all past a 1.3x change, as Ocean's does.
* ``box_search_size`` widens the box search, as for the other box trackers, but
  HiT's input is fixed at 256 px, so a wider window is the same work at a lower
  resolution. Up to 415 (the target then a third smaller than HiT was trained
  on) it still tracked a 70 px face moving 40 px a frame; from 447 it lost it,
  and on the composited clips below even 319 and 383 cost IoU (0.95 to 0.92
  and 0.90). The setting is accepted over the app's whole 255-511 range, since
  a point keeps its stored settings when its tracker changes, but the default
  255 — four times the target, which already kept that face — is the one to
  use. A point's search, which grows on its own with ``search_radius``, stops
  at 415.

Strengths and limits
--------------------
Measured with the real checkpoints over the kinds of clips
:mod:`stable.trackers.ocean` describes, with Ocean and SiamDW run on the same
clips for comparison:

* **Boxes are what it is for.** Over composited 60-frame clips — a face, a logo
  and a blob texture, each zooming 2x, shrinking to 0.45x, stretching from an
  aspect of 0.8 to 2.75 and "breathing", over a natural background — HiT-Base
  overlapped the truth by a mean IoU of 0.95 on the face and the blob texture
  (every clip 0.90 or better; Ocean 0.86, SiamDW 0.70 on the same eight). It
  lost the logo, a white card, when it shrank or stretched (IoU 0.19 and 0.44),
  exactly as the authors' own tracking loop does with the same network (0.17
  and 0.31), so the mean over all twelve is 0.79 (Ocean 0.80, SiamDW 0.70).
  With an occluder crossing the object as well it recovered on ten of the
  twelve (Ocean seven, SiamDW nine).
* **Points are serviceable on natural texture**: over eight 50-frame clips of
  known motion (handheld shake, a 30% zoom, a rotating pan and a 6 px/frame
  pan, over a photograph and over a dense blob texture; eight points each) it
  measured median 0.49 px and 90th percentile 1.2 px on the photograph (Ocean
  0.79 / 3.5, SiamDW 0.52 / 0.9) and 0.95 / 25 px on the blob texture, where
  every Siamese tracker jumps between lookalikes (Ocean 1.2 / 18, SiamDW
  0.7 / 28). It came through five of six occluder sweeps, at confidence of at
  most 0.17 while covered.
* **It is not a tracker for tiny features.** HiT sees the world in 16 px
  tokens. On the bundled 128x96 sample clip its points measured a median 1.1-1.3
  px against SiamDW's 0.54: use SiamDW, or ``lk``, for small detail.
* **Speed** (RTX 5090 shared with other jobs): about 4 ms a frame for a box and
  6 ms for eight points, almost all of it per-call overhead — the network
  itself takes 2-3 ms. On the CPU (4 threads): 26 ms a frame for a box.

Variants
--------
All three published models are supported (``variant``); each has its own
checkpoint and its own measured calibration (:data:`CALIBRATIONS`).
``"base"`` (LeViT-384, 42M parameters, a 511 MB download) is the default: it
is the most accurate in the authors' benchmarks and on the clips above for
points (``"small"`` measured a 90th percentile of 18 px on the photograph,
``"tiny"`` 3.2 px) and on the face and blob boxes, and on a GPU the three run
at the same speed. ``"small"`` (LeViT-128, 137 MB) and ``"tiny"`` (LeViT-128S,
118 MB) are for the CPU — 15 and 12 ms a frame for a box against Base's 26 —
and track boxes well (mean IoU 0.85 and 0.88 over all twelve clips; both did
better than Base on the logo) but points poorly.

Confidence
----------
HiT has no score head: the corner maps say where the corners are, not whether
the target is there, and measured on their own (peak probability, spread,
logit) they separated good from lost frames poorly (AUC 0.57-0.83). The
evidence is instead **appearance**: the cosine between the template's target
tokens and the search's tokens under the box found, read from the first stage
of HiT's own backbone (:meth:`~stable.trackers.hit_net.HiTNet.appearance`) —
an occluder or a lookalike scores lower. Like SiamDW's response it means
different things for different templates, so it is read **as a fraction of
the anchor frame's own** (:meth:`HiTTracker.raw_confidence`). For HiT-Base,
over the clips above:

* points: steps within 2 px of the truth measured median 0.98 (1st percentile
  0.89), steps with the point covered median 0.53 and at most 0.78;
  ``confidence_range`` ``(0.75, 0.9)`` maps 99% of the good steps to 0.9 or
  more and 94% of the covered ones to ``0``;
* boxes: frames overlapping the truth by IoU 0.9 or more measured median 0.81
  (5th percentile 0.59), frames under 0.5 median 0.46, frames with the object
  mostly covered median 0.53 — but up to 0.73 where the occluder was cut from
  the same texture as the object. ``box_confidence_range`` ``(0.5, 0.8)``,
  whose bottom is also where a box is held, zeroes 0.2% of the frames tracked
  at IoU 0.7 or better: the bottom is set by the lesson in
  :mod:`stable.trackers.ocean` (a correctly tracked frame read as ``0`` is a
  frame the solver must invent).

Weights
-------
The authors' training checkpoints (``HiT_{Base,Small,Tiny}/VT_ep1500.pth.tar``)
from the Google Drive folder the MIT-licensed ``kangben258/HiT`` README links.
They are whole training checkpoints — optimiser state and the harness's
settings beside the weights — so they are read with an allow-list unpickler
that never imports the authors' training code
(:func:`~stable.trackers.hit_net.read_release_state_dict`). Pinned like every
other checkpoint, from a real download that was loaded, matched against the
authors' module code and tracked with (CONTRIBUTING.md, "Pinning a
checkpoint"). The backbones were initialised from LeViT's ImageNet weights
(Apache-2.0) before training.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import ContractError, TrackerError
from .siamese import (
    SiameseBackend,
    SiameseMaps,
    SiameseStep,
    SiameseTrackerPlugin,
    _denormals_flushed,
    _numpy,
    _torch,
    sample_map,
)
from .weights import ModelSpec, register_model

__all__ = [
    "HiTTracker",
    "HiTBackend",
    "HiTMaps",
    "HIT_BASE",
    "HIT_SMALL",
    "HIT_TINY",
    "HIT_MODELS",
    "CALIBRATIONS",
    "soft_argmax",
]

logger = logging.getLogger(__name__)

_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

HIT_BASE = register_model(
    ModelSpec(
        key="hit_base",
        url=_DRIVE.format("1pK5Fo1-TNg0f_KbCndAXUDdprImb5zzt"),
        filename="HiT_Base.pth.tar",
        sha256="7d6ee218014c3dae93c04fff0df5284c5fd723d6fcf79d5210e83457bf242a00",
        size_bytes=511212333,
        license="MIT",
        non_commercial=False,
        notes=(
            "HiT-Base (Kang et al., ICCV 2023): LeViT-384 backbone, trained on LaSOT, "
            "GOT-10k, COCO and TrackingNet. The authors' training checkpoint "
            "(checkpoints/HiT_Base/VT_ep1500.pth.tar) from the Google Drive folder the "
            "MIT-licensed kangben258/HiT README links; the weights are 42.2M parameters, "
            "the rest of the file is optimiser state. Backbone initialised from LeViT "
            "(Apache-2.0)."
        ),
    )
)
"""The HiT-Base checkpoint."""

HIT_SMALL = register_model(
    ModelSpec(
        key="hit_small",
        url=_DRIVE.format("14I3VyqrRre6KThBBAqS2c27EkHmXM544"),
        filename="HiT_Small.pth.tar",
        sha256="7f3d834d6d55eecf4ceea3f8c8cddf268a5bed8510678b25e5fc65b96855f475",
        size_bytes=137177837,
        license="MIT",
        non_commercial=False,
        notes=(
            "HiT-Small (Kang et al., ICCV 2023): LeViT-128 backbone, 11.1M parameters. "
            "The authors' training checkpoint (checkpoints/HiT_Small/VT_ep1500.pth.tar) "
            "from the Google Drive folder the MIT-licensed kangben258/HiT README links."
        ),
    )
)
"""The HiT-Small checkpoint."""

HIT_TINY = register_model(
    ModelSpec(
        key="hit_tiny",
        url=_DRIVE.format("188aBm6sf-4RtXgFUZODYgRUc0oUtIZ14"),
        filename="HiT_Tiny.pth.tar",
        sha256="021a7583e1a18d96683bdb69e85b678ea4c7cac5d2522b9ec90e7fdc1f6fa90b",
        size_bytes=118102065,
        license="MIT",
        non_commercial=False,
        notes=(
            "HiT-Tiny (Kang et al., ICCV 2023): LeViT-128S backbone, 9.6M parameters. "
            "The authors' training checkpoint (checkpoints/HiT_Tiny/VT_ep1500.pth.tar) "
            "from the Google Drive folder the MIT-licensed kangben258/HiT README links."
        ),
    )
)
"""The HiT-Tiny checkpoint."""

HIT_MODELS: dict[str, ModelSpec] = {"base": HIT_BASE, "small": HIT_SMALL, "tiny": HIT_TINY}
"""Every published HiT checkpoint, by the variant name :class:`HiTTracker` takes."""

TEMPLATE_SIZE = 128
"""Template side HiT was trained on, in network pixels."""

SEARCH_SIZE = 256
"""Search side HiT was trained on, in network pixels."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Per-channel mean HiT normalises its RGB input by (after scaling to ``[0, 1]``)."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel standard deviation HiT normalises its RGB input by."""

CALIBRATIONS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "base": ((0.75, 0.9), (0.5, 0.8)),
    "small": ((0.85, 0.97), (0.65, 0.88)),
    "tiny": ((0.78, 0.93), (0.5, 0.8)),
}
"""Each variant's measured ``(confidence_range, box_confidence_range)``.

Fitted per variant because the appearance ratio runs higher on the smaller
backbones (HiT-Small scored covered points up to 0.89, Base up to 0.78); see
the module docstring for Base's distributions.
"""

REACH_FRACTION = 7.0 / 16.0
"""How far from the search centre, as a fraction of its side, HiT was trained to look.

Training jittered the target's centre uniformly up to 1.75 target sizes (half of
``CENTER_JITTER`` 3.5) from the centre of a search four target sizes across.
"""


@dataclass(frozen=True)
class HiTMaps(SiameseMaps):
    """HiT's raw output for one batch: its two corner maps and an appearance map.

    ``score`` holds the top-left corner's logits and :attr:`bottom_right` the
    bottom-right's, each ``[N, S, M, M]`` over HiT's 16x16 grid of search
    tokens. The box is their soft-argmax, which the plugin computes; nothing
    is decoded here, like every other :class:`~stable.trackers.siamese.SiameseMaps`.
    ``similarity``, when present, is
    :meth:`~stable.trackers.hit_net.HiTNet.appearance`: cell ``(i, j)`` is
    centred on corner-grid position ``(16 j, 16 i)`` in pixel-edge coordinates.

    Attributes:
        bottom_right: ``[N, S, M, M]`` bottom-right corner logits.
        crop_size: Side of the search crops the plugin cut, in its pixels.
        network_size: Side the crops were resampled to for the network.

    Raises:
        ContractError: On a shape mismatch, or a missing ``bottom_right``.
    """

    bottom_right: np.ndarray | None = None
    crop_size: int = SEARCH_SIZE
    network_size: int = SEARCH_SIZE

    def __post_init__(self) -> None:
        """Coerce and check the second corner map against the first (dataclass hook)."""
        super().__post_init__()
        if self.bottom_right is None:
            raise ContractError("HiT maps need both corner maps")
        bottom_right = np.asarray(self.bottom_right, dtype=np.float32)
        if bottom_right.shape != self.score.shape:
            raise ContractError(
                f"bottom-right maps must be {self.score.shape}, got {bottom_right.shape}"
            )
        object.__setattr__(self, "bottom_right", bottom_right)


def soft_argmax(logits: np.ndarray, stride: float) -> tuple[float, float]:
    """Return the expected ``(x, y)`` under ``softmax(logits)``: HiT's corner read-out.

    The grid is HiT's: each cell's top-left pixel, ``0, stride, 2 * stride, ...``.

    Args:
        logits: ``[M, M]`` corner logits.
        stride: Network pixels per cell.

    Returns:
        The expected corner, in network pixels.
    """
    values = np.asarray(logits, dtype=np.float64)
    prob = np.exp(values - values.max())
    prob /= prob.sum()
    grid = np.arange(values.shape[-1], dtype=np.float64) * stride
    return float(prob.sum(axis=0) @ grid), float(prob.sum(axis=1) @ grid)


class HiTBackend:
    """Runs a loaded :class:`~stable.trackers.hit_net.HiTNet` on the plugin's NumPy crops.

    Owns HiT's input convention: RGB scaled to ``[0, 1]`` and normalised with
    the ImageNet mean and deviation, the template resampled to 128 px and the
    search to 256 px (the plugin cuts them at 127 px and at the search side it
    chose; see :class:`HiTTracker`). On the CPU every call runs with denormal
    floats flushed to zero, as the other Siamese backends do.

    Args:
        network: A ``HiTNet`` with weights loaded, already on ``device``.
        device: The device string the network lives on.

    Attributes:
        device: Where the network currently is; updated by :meth:`to_device`.
    """

    def __init__(self, network: Any, device: str) -> None:
        """Wrap a loaded network."""
        self._network = network
        self.device = device

    def _normalised(self, crops: np.ndarray, size: int) -> Any:
        """Return ``[B, X, X, 3]`` uint8 RGB as HiT's ``[B, 3, size, size]`` input."""
        torch = _torch()
        tensor = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        if tensor.shape[-1] != size:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=tensor.shape[-1] > size,
            )
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def embed(self, exemplars: np.ndarray) -> Any:
        """Return the network's template tokens for ``exemplars``."""
        torch = _torch()
        with torch.inference_mode(), _denormals_flushed(torch, self.device):
            return self._network.embed(self._normalised(exemplars, TEMPLATE_SIZE))

    def match(self, kernels: Any, searches: np.ndarray) -> HiTMaps:
        """Run HiT on ``searches`` (``[N, S, X, X, 3]``), each against its template."""
        torch = _torch()
        count, scales, side = (int(v) for v in searches.shape[:3])
        flat = searches.reshape(count * scales, *searches.shape[2:])
        with torch.inference_mode(), _denormals_flushed(torch, self.device):
            batch = self._normalised(flat, SEARCH_SIZE)
            tokens = kernels if scales == 1 else kernels.repeat_interleave(scales, dim=0)
            _, top_left, bottom_right, similarity = self._network.match(tokens, batch)
            cells = int(top_left.shape[-1])
            return HiTMaps(
                score=_numpy(top_left).reshape(count, scales, cells, cells),
                similarity=_numpy(similarity).reshape(count, scales, cells, cells),
                bottom_right=_numpy(bottom_right).reshape(count, scales, cells, cells),
                crop_size=side,
                network_size=SEARCH_SIZE,
            )

    def to_device(self, device: str) -> None:
        """Move the network to ``device``."""
        self._network.to(device)
        self.device = device

    def release(self) -> None:
        """Drop the reference to the network so its memory can be reclaimed."""
        self._network = None


class HiTTracker(SiameseTrackerPlugin):
    """HiT as a Stable tracker plugin.

    Args:
        variant: Which published model to run: ``"base"`` (the default),
            ``"small"`` or ``"tiny"``; see the module docstring. Each has its
            own checkpoint (:data:`HIT_MODELS`) and calibration
            (:data:`CALIBRATIONS`).
        template_radius: Half-size of the target box, in source pixels.
        search_radius: How far the point may move between frames, in source
            pixels. The default is what the trained search reaches with the
            default box, with room to spare.
        size_lr: How fast a point's tracked size follows HiT's box. A point's
            "box" is HiT's guess at the object a patch belongs to, so it is
            damped, and ignored past a :attr:`point_scale_gate` change.
        box_size_lr: The same for a box the operator drew: ``1``, the authors'
            own tracker taking HiT's box as it comes.
        confidence_range: ``(low, high)`` calibration of the evidence (see
            :meth:`raw_confidence`); ``None`` for the variant's measured one.
        box_confidence_range: The same for a box, whose bottom is also where a
            box is held; ``None`` for the variant's measured one.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.siamese.SiameseTrackerPlugin`
            (``lost_below``, ``device``, ``backend``, ...). ``box_search_size``
            is accepted over the same 255-511 as for the other box trackers, but
            HiT resamples any window to its 256 px input, so a wider one reaches
            further at a lower resolution rather than at a higher cost, and
            past about 415 the target is too small for HiT to find; see the
            module docstring.

    Raises:
        ContractError: For an unknown ``variant`` or a rate outside ``[0, 1]``.

    Attributes:
        point_scale_gate: A point's size is not updated from a box that
            differs from the expected one by more than this factor (scale and
            aspect change multiplied). Boxes have no such gate
            (:attr:`scale_gate` is infinite): HiT's box is the one thing it was
            trained to get right.
        point_max_instance_size: The widest search a point's ``search_radius``
            grows to. Boxes may be given up to :attr:`max_instance_size` (the
            setting the app offers every box tracker); a point's search grows
            on its own, so it stops where HiT still tracked.
    """

    name = "hit"
    display_name = "HiT (hierarchical vision transformer)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "MIT"

    model_spec = HIT_BASE
    # The base class's crop geometry, in its units: a 127 px template and a 255
    # px search are HiT's 128 and 256 (twice and four times the target) to
    # within 0.8%. Searches grow in 16 px steps like the Siamese trackers'.
    exemplar_size = 127
    total_stride = 8
    trained_instance_size = 255
    max_instance_size = 511
    point_max_instance_size = 415
    scale_gate = math.inf
    point_scale_gate = 1.3

    def __init__(
        self,
        *,
        variant: str = "base",
        template_radius: int = 16,
        search_radius: int = 48,
        size_lr: float = 0.3,
        box_size_lr: float = 1.0,
        confidence_range: tuple[float, float] | None = None,
        box_confidence_range: tuple[float, float] | None = None,
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared HiT plugin; see the class docstring."""
        if variant not in HIT_MODELS:
            raise ContractError(
                f"unknown HiT variant {variant!r}; choose from {sorted(HIT_MODELS)}"
            )
        point_range, box_range = CALIBRATIONS[variant]
        kwargs.setdefault("window_influence", 0.0)
        super().__init__(
            template_radius=template_radius,
            search_radius=search_radius,
            confidence_range=point_range if confidence_range is None else confidence_range,
            box_confidence_range=box_range
            if box_confidence_range is None
            else box_confidence_range,
            visible_threshold=visible_threshold,
            **kwargs,
        )
        self.variant = variant
        self.model_spec = HIT_MODELS[variant]
        for label, rate in (("size_lr", size_lr), ("box_size_lr", box_size_lr)):
            if not 0.0 <= float(rate) <= 1.0:
                raise ContractError(f"{label} must lie in [0, 1], got {rate}")
        self.size_lr = float(size_lr)
        self.box_size_lr = float(box_size_lr)

    def search_geometry(self) -> tuple[int, float]:
        """Return a point's search side and reach, the side at most :attr:`point_max_instance_size`.

        See :meth:`~stable.trackers.siamese.SiameseTrackerPlugin.search_geometry`.
        HiT resamples every search to 256 px, so growing it buys reach with
        resolution; past the cap the target is too small for HiT to find, and
        the reach asked for is honoured only as far as the cap allows.
        """
        size, reach = super().search_geometry()
        if size > self.point_max_instance_size:
            size = self.point_max_instance_size
            logger.info(
                "%s: search_radius %d px needs a wider search than HiT can use; it reaches %.0f px",
                self.name,
                self.search_radius,
                self._reach_of(size) / self.anchor_scale,
            )
        return size, reach

    def score_size(self, instance_size: int) -> int:
        """Return the side of a grid at :attr:`total_stride` spanning HiT's reach.

        HiT has no score map to bound where it may answer; its soft-argmax can
        land anywhere in the search. What bounds it is training: the target was
        never further than :data:`REACH_FRACTION` of the search side from its
        centre. This is the odd grid at 8 px a cell that reaches that far, which
        is what :meth:`~stable.trackers.siamese.SiameseTrackerPlugin.search_geometry`
        needs to grow the search for a large ``search_radius``.
        """
        return 2 * int(int(instance_size) * REACH_FRACTION // self.total_stride) + 1

    def raw_confidence(self, step: SiameseStep, reference: SiameseStep) -> float:
        """Return the appearance match found, as a fraction of the anchor frame's own.

        See the module docstring: the cosine means different things for
        different templates, the ratio to what the template scores on its own
        frame does.
        """
        if reference.evidence <= 0.0:
            return 0.0
        return float(step.evidence / reference.evidence)

    def _build_backend(self, device: str, weights: Any) -> SiameseBackend:
        """Load the chosen HiT variant from ``weights`` onto ``device``.

        Raises:
            TrackerError: If PyTorch is missing or the checkpoint is not HiT's.
        """
        if torch_module() is None:
            raise TrackerError(
                "HiT needs PyTorch; install the 'torch' extra (`pip install stable-tracker[torch]`)"
            )
        from .hit_net import HiTNet, load_checkpoint  # noqa: PLC0415 - imports torch

        network = load_checkpoint(HiTNet(self.variant), weights).fuse().to(device)
        return HiTBackend(network, device)

    def _decode(
        self,
        maps: SiameseMaps,
        slot: int,
        expected_box: tuple[float, float],
        expected_centre: tuple[float, float],
        reach: float,
        *,
        box: bool = False,
    ) -> SiameseStep:
        """Read HiT's box for one point: the soft-argmax of each corner map.

        Args:
            maps: The batch's raw output; must be :class:`HiTMaps`.
            slot: This point's index in the batch.
            expected_box: The reference box's size in this crop. HiT regresses
                the box with no prior on it; a point's size is just not
                followed from a box too far from this (:attr:`point_scale_gate`).
            expected_centre: Where the reference box would be; a box further
                than ``reach`` from it is not the target.
            reach: How far from ``expected_centre`` the box may be.
            box: Whether this is a drawn box, whose size follows at
                ``box_size_lr`` rather than ``size_lr``.

        Returns:
            HiT's box in the search crop's pixels, the rate its size is
            followed at, and the evidence: the appearance cosine under the box
            (see :meth:`raw_confidence`), ``0`` beyond the reach, and ``1`` from
            a backend that reports no appearance map.

        Raises:
            TrackerError: If the backend returned something other than HiT maps.
        """
        if not isinstance(maps, HiTMaps) or maps.bottom_right is None:
            raise TrackerError(f"{self.name} backend returned no corner maps")
        cells = maps.score.shape[-1]
        stride = maps.network_size / cells
        x0, y0 = soft_argmax(maps.score[slot, 0], stride)
        x1, y1 = soft_argmax(maps.bottom_right[slot, 0], stride)
        scale = maps.crop_size / maps.network_size
        # Corners are in the network's pixel-centre convention, whose centre is
        # (size - 1) / 2: STARK's training labels put the crop centre there.
        middle = (maps.network_size - 1) / 2.0
        dx = ((x0 + x1) / 2.0 - middle) * scale
        dy = ((y0 + y1) / 2.0 - middle) * scale
        evidence = 1.0
        if maps.similarity is not None:
            # Appearance cells are centred on the corner grid's positions,
            # 16 px apart in pixel-edge coordinates (HiTNet.appearance).
            cell_x = ((x0 + x1) / 2.0 + 0.5) / stride
            cell_y = ((y0 + y1) / 2.0 + 0.5) / stride
            evidence = max(sample_map(maps.similarity[slot, 0], cell_x, cell_y), 0.0)
        if (
            math.isfinite(reach)
            and max(abs(dx - expected_centre[0]), abs(dy - expected_centre[1])) > reach
        ):
            evidence = 0.0
        width, height = max((x1 - x0) * scale, 1e-6), max((y1 - y0) * scale, 1e-6)
        rate = self.box_size_lr if box else self.size_lr
        if not box:
            along_x = width / max(expected_box[0], 1e-6)
            along_y = height / max(expected_box[1], 1e-6)
            measured = math.sqrt(along_x * along_y)
            aspect = along_x / along_y
            if max(measured, 1 / measured) * max(aspect, 1 / aspect) > self.point_scale_gate:
                rate = 0.0
        return SiameseStep(dx=dx, dy=dy, width=width, height=height, rate=rate, evidence=evidence)
