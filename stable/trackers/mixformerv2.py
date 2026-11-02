"""MixFormerV2 — a one-stream transformer tracker (MCG-NJU), as a Stable plugin.

MixFormerV2 (Cui, Song, Wu & Wang, NeurIPS 2023) does away with everything a
Siamese tracker is built from. There is no backbone run twice and no
correlation between a template and a search: one plain ViT reads both crops as
a single sequence of tokens, with four learnt *prediction tokens* riding along,
and after the last block each of those four is read as a probability
distribution over where one edge of the box lies. The box is their means; a
small MLP over the same tokens scores it. There is no score map, no anchor, no
window and no penalty — one box per frame, and how sure the network is of it.
It was distilled from a 12-layer teacher into 8 layers (MixFormerV2-B) and 4
(MixFormerV2-S), which is where its speed comes from.

How it is used here
-------------------
* The network is the published one, ported to plain PyTorch in
  :mod:`stable.trackers.mixformerv2_net` (no ``timm``, no ``einops``, nothing
  moved to CUDA behind the caller's back). Both checkpoints load with
  ``strict=True``, and on identical inputs the port reproduces the authors'
  own module definitions to 7e-7 of the search side (1.6e-4 px) on the box and
  9e-6 on the score logit, for both variants, on the GPU; the CPU path agrees
  with the authors' GPU run to 6e-6 and 8e-4. The templates' keys and values
  are computed once per anchor (:class:`MixFormerV2Backend`'s ``embed``) and
  only the search runs per frame — exact, because MixFormer's template tokens
  never attend to the search.
* It is driven by :class:`~stable.trackers.siamese.SiameseTrackerPlugin`, like
  Ocean and SiamDW: exact sub-pixel crops, the anchor's own reading that the
  point rides on, the search radius, scale, holding through a loss, batching.
  The base cuts a 127 px exemplar and an ``X`` px search at one scale, and the
  backend resamples them to the network's own 128/288 px (112/224 px for -S).
  So ``X / 127`` is the search factor relative to upstream's template factor of
  2: 255 px is a 4.0x search, 287 px upstream's configured 4.5x, 319 px the 5.0x
  its benchmark scripts run MixFormerV2-B at. Boxes are searched at 319 px
  (287 px for -S) by default, and every size the server's *Search window*
  control offers (255-511 px) is valid: MixFormerV2's cost does not grow with
  it, a larger window just sees more of the frame at a lower resolution. Points
  search from 255 px up, as far as ``search_radius`` needs; the default 48 px
  needs 271. The exemplar's context is SiamFC's rule rather than upstream's
  ``2 sqrt(wh)``: identical for a square box, 12% more context at an aspect of
  3, well inside the scale jitter MixFormerV2 was trained with.
* The one box is read as a 1x1 map (:class:`MixFormerV2Maps`). There is nothing
  for a cosine window to choose between, so ``search_radius`` is honoured the
  only way it can be: a box centred further away than that is no evidence at
  all, and the point holds.
* The point **rides** on the box, as on Ocean's: handed a patch, MixFormerV2
  boxes whatever object it takes the patch to belong to. A box is taken as the
  network gives it — ``box_size_lr`` 1, as upstream does — except that a change
  past :attr:`~stable.trackers.siamese.SiameseTrackerPlugin.scale_gate` in one
  frame is pulled back to the gate rather than ignored. That matters after an
  occlusion: the box is held at its old size while the object changes behind
  the occluder, and an ungated network answer that differs by more than the
  gate would otherwise never be taken up. A point's box is left to the gate as
  it is for Ocean.
* Upstream keeps a second, *online* template, replaced every few hundred frames
  (200 on LaSOT) by the best-scoring crop since the last replacement. Here both
  templates are the anchor's crop, which is upstream's own state until that
  first replacement. Stable's answer to an appearance change is another anchor
  (multi-anchor merging), not a template refreshed from the tracker's own
  output, which can lock onto an occluder; and a fixed template keeps forward
  and backward runs from one anchor symmetric and reproducible.

Strengths and limits
--------------------
Measured with the real checkpoint on 640x360 clips of known motion — a natural
photograph and a blob texture, each under handheld shake, a 30% zoom, a slow
rotating pan and a 6 px/frame pan, 50 frames and eight points each — and six
sweeps of a 90 px occluder (flat grey, or another photograph) across a point;
Ocean and SiamDW were run on the very same clips:

* On points it is the most accurate Siamese tracker here: median 0.27 px, 90th
  percentile 0.82 px, worst 2.4 px (photograph 0.22 / 0.80, blobs 0.32 / 0.83),
  against SiamDW's 0.50 / 0.89 and Ocean's 0.63 / 2.2 px (Ocean lost a point on
  the fast photographic pan).
* It came through all six occluder sweeps — worst error 0.2 px from six frames
  after the occluder had passed — reporting confidence ``0`` while covered.
* The bundled 128x96 sample clip is where it is weakest: median 1.1 px, 90th
  percentile 4.6 px, against SiamDW's 0.54 / 0.86. Its points are squares of 5
  to 9 px among others, and one of them has two neighbours inside its 33 px
  box: MixFormerV2 boxes the cluster, and the box wobbles 2-8 px as it decides
  which squares belong to it. The confidence says so — those frames score
  ``0`` (see Confidence) — and a ``template_radius`` that fits the feature (8)
  measured 0.38 / 2.7 px over the three points.

Boxes are what it was built for. Over twelve composited 60-frame clips — a
face, a logo and a blob texture, each zooming 2x, shrinking to 0.45x,
stretching from an aspect of 0.8 to 2.75, and "breathing" in and out of shape,
over a natural background — its box overlapped the truth by a mean IoU of 0.95
(10th percentile 0.89); on the same clips Ocean measured 0.90 (0.81) and SiamDW
0.68. With an occluder crossing each object as well it recovered on all twelve
(Ocean ten, SiamDW eight), mean IoU 0.88 including the covered frames.

Speed, one target, on an RTX 5090 shared with other jobs: 4.6 ms a frame
(about 220 fps; Ocean 3.7 ms), 16 ms for eight points in one call. On the CPU
(eight threads) 48 ms a frame, against Ocean's 65 ms. Peak GPU memory for one
target is a few hundred megabytes.

Confidence
----------
MixFormerV2's score head was trained, in the model's last stage, to say whether
the box is on the target, and on a clean track it is sure: 99% of steps within
2 px of the truth scored over 0.996. On its own it is not a usable confidence
for a point. With the point covered it still scored up to 0.67; worse, as the
occluder arrived the network boxed *it*, at scores of 0.4-0.96, and a point
following those frames was dragged off on four of the six sweeps. Two more
signals come from the same output:

* **Sharpness.** Each edge is a distribution over 96 positions; their standard
  deviation was 3-4 crop pixels on a clean patch and 12-57 with it covered.
  The evidence takes ``min(1, s_anchor / s)``, this frame's spread against the
  anchor's own.
* **Shape agreement.** For a patch, a box of a different shape from the one the
  anchor's reading would have now means the network is boxing something else:
  on the sample clip the frames more than 3 px off scored a median 0.995 on
  score and sharpness together. The evidence takes the product of the two
  sides' ratios (smaller over larger), squared.

For a point the evidence is the product of all three. Over the clips above it
measured a median 0.91 (1st percentile 0.63) on steps within 2 px of the truth
and at most 0.12 with the point covered; on the sample clip, a median 0.59 on
steps within 2 px and 0.25 on steps more than 3 px off. ``confidence_range``
of ``(0.3, 0.7)`` puts every covered step at ``0``, 95% of the steps within
2 px of the truth at ``1``, and the sample clip's wobbling point at ``0``.

For a box the evidence is the score head alone. The edges' spread grows with
the object — a box that doubles in size doubles it — so sharpness would read a
zoom as a loss, and a drawn box's shape is supposed to change. Correctly
tracked frames (IoU over 0.7) scored a median 0.997 (1st percentile 0.78),
frames with the object covered at most 0.24; ``box_confidence_range`` of
``(0.3, 0.75)`` separates them, and its bottom is where a box is held.

MixFormerV2-S
-------------
:class:`MixFormerV2SmallTracker` is the four-layer model with a 1x MLP: the box
tracker for a machine without a GPU, at 9.6 ms a frame on the same eight CPU
threads (1.6 ms on the GPU). On boxes it matched -B on the clips above (mean
IoU 0.95, 10th percentile 0.90; 0.87 and all twelve recovered with the
occluder). On points it is a poorer tracker — median 0.99 px, 90th percentile
3.2 px, worst 25 px, mostly from jumping between similar blobs — and its
confidence cannot tell those jumps apart (median evidence 0.76 on steps more
than 3 px off, 0.86 on steps within 2 px). Prefer ``mixformerv2`` or ``siamdw``
for points. Its point calibration, ``(0.5, 0.8)``, is the threshold at which it
came through all six occluder sweeps; its boxes share -B's.

Weights
-------
Both checkpoints come from the Google Drive folder the MIT-licensed
MCG-NJU/MixFormerV2 README links (``mixformerv2_base.pth.tar``,
``mixformerv2_small.pth.tar``: upstream's ``288_depth8_score`` and
``224_depth4_mlp1_score``, the online models with their score heads). They are
pytracking *training* checkpoints — the weights under ``net``, beside the
optimiser state and pickled trainer objects — and are read with
``weights_only=True`` and inert stand-ins for the trainer's four classes, so no
code in them runs; that needs PyTorch 2.6 or newer. The repository's MIT licence
is the only licence stated for them. Two things are worth knowing before
shipping work made with them: they were trained on LaSOT, GOT-10k, TrackingNet
and COCO, research datasets with terms of their own, and distilled from a
teacher whose ViT was initialised from Meta's MAE pre-training, released under
CC-BY-NC 4.0. Pinned like every other checkpoint, from a real download that was
loaded, checked against the authors' code, and tracked with (CONTRIBUTING.md,
"Pinning a checkpoint").
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import ContractError, TrackerError
from .confidence import sigmoid
from .siamese import SiameseBackend, SiameseMaps, SiameseStep, SiameseTrackerPlugin
from .weights import ModelSpec, register_model

__all__ = [
    "MIN_TORCH",
    "MixFormerV2Maps",
    "MixFormerV2Step",
    "MixFormerV2Tracker",
    "MixFormerV2SmallTracker",
    "MixFormerV2Backend",
    "MIXFORMERV2_BASE_WEIGHTS",
    "MIXFORMERV2_SMALL_WEIGHTS",
]

logger = logging.getLogger(__name__)

_NOTES = (
    "{model} (Cui et al., NeurIPS 2023), the online model with its score head, from the "
    "Google Drive folder the MIT-licensed MCG-NJU/MixFormerV2 README links. Trained on "
    "LaSOT, GOT-10k, TrackingNet and COCO, distilled from a MixViT teacher whose ViT was "
    "initialised from MAE (CC-BY-NC 4.0) pre-training."
)

MIXFORMERV2_BASE_WEIGHTS = register_model(
    ModelSpec(
        key="mixformerv2_base",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1povUjoidqk20PBVePzrwY7uu7B6fCvNE&export=download&confirm=t"
        ),
        filename="mixformerv2_base.pth.tar",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); 240,205,177 bytes.
        sha256="14a44d0c90ee856f3fe36af31c4f0af643cbc2ba47546de98c7162d3d704a228",
        size_bytes=240205177,
        license="MIT",
        non_commercial=False,
        notes=_NOTES.format(model="MixFormerV2-B"),
    )
)
"""The MixFormerV2-B checkpoint :class:`MixFormerV2Tracker` loads."""

MIXFORMERV2_SMALL_WEIGHTS = register_model(
    ModelSpec(
        key="mixformerv2_small",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1M962ncGsP2iAILIQkT4529MRmfIviLH9&export=download&confirm=t"
        ),
        filename="mixformerv2_small.pth.tar",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); 69,679,305 bytes.
        sha256="db94d947e2bf69918640aaf5dfce16a6dbed108fe942f43a8832324e5b6c569e",
        size_bytes=69679305,
        license="MIT",
        non_commercial=False,
        notes=_NOTES.format(model="MixFormerV2-S"),
    )
)
"""The MixFormerV2-S checkpoint :class:`MixFormerV2SmallTracker` loads."""


@dataclass(frozen=True)
class MixFormerV2Maps(SiameseMaps):
    """:class:`~stable.trackers.siamese.SiameseMaps` plus the box's uncertainty.

    Args:
        spread: ``[N, S, M, M]`` mean standard deviation of the four edge
            distributions, in search-crop pixels; ``None`` when unknown.
    """

    spread: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce and check the spread like the other maps (dataclass hook)."""
        super().__post_init__()
        spread = self.spread
        if spread is not None:
            spread = np.asarray(spread, dtype=np.float32)
            if spread.shape != self.score.shape:
                raise ContractError(f"spread maps must be {self.score.shape}, got {spread.shape}")
            object.__setattr__(self, "spread", spread)


@dataclass(frozen=True)
class MixFormerV2Step(SiameseStep):
    """:class:`~stable.trackers.siamese.SiameseStep` plus the box's uncertainty.

    Attributes:
        score: The score head's probability that the box is on the target.
        spread: Mean standard deviation of the edge distributions, in
            reference-crop pixels; ``nan`` when the backend reported none.
        box: Whether the reading is of a drawn box rather than a point's patch,
            which :meth:`MixFormerV2Tracker.raw_confidence` reads differently.
    """

    score: float = 1.0
    spread: float = math.nan
    box: bool = False


def _agreement(width: float, height: float, expected: tuple[float, float]) -> float:
    """Return how closely a ``width x height`` box matches ``expected``, in ``(0, 1]``.

    The product of each side's ratio, smaller over larger: ``1`` for the same
    box, ``0.5`` for one twice as wide.
    """
    expected_w, expected_h = max(expected[0], 1e-6), max(expected[1], 1e-6)
    return min(width / expected_w, expected_w / width) * min(
        height / expected_h, expected_h / height
    )


MIN_TORCH = (2, 6)
"""The first PyTorch whose ``weights_only`` loader accepts a stand-in for a named
class, which is what reading the published checkpoints safely needs."""


def _torch_reads_checkpoints() -> bool:
    """Whether the installed PyTorch is at least :data:`MIN_TORCH`."""
    torch = torch_module()
    found = re.match(r"(\d+)\.(\d+)", str(getattr(torch, "__version__", "")))
    return found is not None and (int(found[1]), int(found[2])) >= MIN_TORCH


def _torch() -> Any:
    """Return :mod:`torch`, or raise the error the plugin reports."""
    torch = torch_module()
    if torch is None:  # pragma: no cover - guarded by is_available()
        raise TrackerError("MixFormerV2 needs PyTorch; install the 'torch' extra")
    return torch


class MixFormerV2Backend:
    """Runs a :class:`~stable.trackers.mixformerv2_net.MixFormerV2Net` on NumPy crops.

    Owns the network's input and output conventions so nothing else has to:

    * Crops arrive at the Siamese base's sizes — a 127 px exemplar and a search
      of ``X`` px — and are resampled (bilinear, centre-preserving, anti-aliased
      when shrinking) to the network's own template and search sides, then
      scaled to ``[0, 1]`` and normalised with the ImageNet statistics, in RGB.
    * The network's box, ``(x1, y1, x2, y2)`` as fractions of the search side
      with ``0.5`` at the crop centre, is returned as Ocean-style distances from
      the crop centre to each edge, in the ``X`` px crop's pixels, on a 1x1 map.

    Args:
        network: A loaded ``MixFormerV2Net``, on ``device``, in evaluation mode.
        device: The device string the network lives on.

    Attributes:
        device: Where the network currently is; updated by :meth:`to_device`.
    """

    def __init__(self, network: Any, device: str) -> None:
        """Wrap a loaded network."""
        self._network = network
        self.device = device

    def _tensor(self, crops: np.ndarray, size: int) -> Any:
        """Return ``[B, H, W, 3]`` uint8 RGB as normalised ``[B, 3, size, size]`` floats."""
        torch = _torch()
        from .mixformerv2_net import IMAGENET_MEAN, IMAGENET_STD  # noqa: PLC0415 - imports torch

        tensor = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        if tensor.shape[-1] != size or tensor.shape[-2] != size:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=bool(tensor.shape[-1] > size),
            )
        mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
        return ((tensor - mean) / std).contiguous()

    def embed(self, exemplars: np.ndarray) -> Any:
        """Return the templates' per-block keys and values (the network's memory)."""
        torch = _torch()
        with torch.inference_mode():
            template = self._tensor(exemplars, self._network.config.template_size)
            return self._network.embed(template)

    def match(self, kernels: Any, searches: np.ndarray) -> SiameseMaps:
        """Run ``searches`` (``[N, S, X, X, 3]``) against their templates."""
        torch = _torch()
        count, scales, side = (int(v) for v in searches.shape[:3])
        flat = searches.reshape(count * scales, *searches.shape[2:])
        with torch.inference_mode():
            memory = kernels
            if scales > 1:
                memory = [
                    (keys.repeat_interleave(scales, 0), values.repeat_interleave(scales, 0))
                    for keys, values in kernels
                ]
            out = self._network.match(memory, self._tensor(flat, self._network.config.search_size))
            x1, y1, x2, y2 = out.boxes.double().unbind(-1)
            edges = torch.stack([0.5 - x1, 0.5 - y1, x2 - 0.5, y2 - 0.5], dim=-1) * side
            spread = out.spread.double().mean(-1) * side
            return MixFormerV2Maps(
                score=out.score.float().cpu().numpy().reshape(count, scales, 1, 1),
                boxes=edges.float().cpu().numpy().reshape(count, scales, 4, 1, 1),
                spread=spread.float().cpu().numpy().reshape(count, scales, 1, 1),
            )

    def to_device(self, device: str) -> None:
        """Move the network to ``device``."""
        self._network.to(device)
        self.device = device

    def release(self) -> None:
        """Drop the reference to the network so its memory can be reclaimed."""
        self._network = None


class MixFormerV2Tracker(SiameseTrackerPlugin):
    """MixFormerV2-B as a Stable tracker plugin.

    Args:
        template_radius: Half-size of the target box of a point, in source pixels.
        search_radius: How far a point may move between frames, in source
            pixels. The default needs a 271 px search with the default box.
        box_search_size: Search crop side for a box, in the base's network
            pixels (127 is the exemplar): 319 is upstream's 5.0x search for
            MixFormerV2-B. Anything from 255 to 511 in steps of 16 is valid.
        size_lr: How fast a point's scale follows the box the network puts
            round it, scaled further by the score each frame.
        box_size_lr: The same for a drawn box: ``1``, the box as the network
            gives it, as upstream takes it.
        confidence_range: ``(low, high)`` calibration of a point's evidence —
            score x sharpness x shape agreement; see the module docstring.
            ``low`` is also where a point is held.
        box_confidence_range: The same for a box, whose evidence is the score.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.siamese.SiameseTrackerPlugin`
            (``lost_below``, ``device``, ``backend``, ...). Its
            ``window_influence`` is accepted and has no effect: with one box per
            frame there is nothing for a motion prior to choose between.

    Attributes:
        variant: Which published model the plugin loads: ``"base"`` or ``"small"``.
        trained_instance_size: The smallest search, 255 px, a 4.0x search
            factor. MixFormerV2 always sees its own 288 px; this is how much of
            the frame that is cut from, at the least.
    """

    name = "mixformerv2"
    display_name = "MixFormerV2-B (one-stream transformer, MCG-NJU)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "MIT"

    model_spec = MIXFORMERV2_BASE_WEIGHTS
    variant = "base"
    trained_instance_size = 255
    max_instance_size = 511
    implementation_requirement = (
        "OpenCV for its template and search crops, and PyTorch 2.6 or newer to read "
        "its checkpoint safely"
    )
    install_hint = (
        "Install them with: pip install stable-tracker[video] and pip install -U "
        "'torch>=2.6' (see docs/windows-gpu-setup.md for the CUDA build)"
    )

    def __init__(
        self,
        *,
        template_radius: int = 16,
        search_radius: int = 48,
        box_search_size: int = 319,
        size_lr: float = 0.5,
        box_size_lr: float = 1.0,
        confidence_range: tuple[float, float] = (0.3, 0.7),
        box_confidence_range: tuple[float, float] = (0.3, 0.75),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared MixFormerV2 plugin; see the class docstring."""
        kwargs.setdefault("window_influence", 0.0)
        super().__init__(
            template_radius=template_radius,
            search_radius=search_radius,
            box_search_size=box_search_size,
            confidence_range=confidence_range,
            box_confidence_range=box_confidence_range,
            visible_threshold=visible_threshold,
            **kwargs,
        )
        self.size_lr = float(size_lr)
        self.box_size_lr = float(box_size_lr)

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether OpenCV imports and PyTorch can read the checkpoint safely.

        Checked before the weights are fetched, so an old PyTorch costs no
        download: see :data:`MIN_TORCH`.
        """
        return super()._implementation_available() and _torch_reads_checkpoints()

    def score_size(self, instance_size: int) -> int:
        """Return the side of the notional score map a search of ``instance_size`` has.

        MixFormerV2 has no score map: it regresses one box anywhere in its
        search crop. This is the map the Siamese base sizes point searches
        with — cells of :attr:`total_stride` out to where the target box (half
        the exemplar's side) still lies wholly inside the crop.
        """
        half = (int(instance_size) - self.exemplar_size / 2.0) / 2.0
        return 2 * int(math.floor(half / self.total_stride)) + 1

    def raw_confidence(self, step: SiameseStep, reference: SiameseStep) -> float:
        """Return the evidence that ``step`` found the target, before calibration.

        A box's evidence is the score head's. A point's is that, already times
        the shape agreement (see :meth:`_decode`), times its sharpness:
        ``min(1, s_anchor / s)``, the spread of the edge distributions in the
        anchor's own reading over this frame's. The module docstring has the
        measurements behind both.
        """
        evidence = float(step.evidence)
        if isinstance(step, MixFormerV2Step) and step.box:
            return evidence
        spread = getattr(step, "spread", math.nan)
        anchor = getattr(reference, "spread", math.nan)
        if math.isfinite(spread) and math.isfinite(anchor) and spread > 0.0:
            evidence *= min(1.0, anchor / spread)
        return evidence

    def _build_backend(self, device: str, weights: Any) -> SiameseBackend:
        """Load MixFormerV2 from ``weights`` onto ``device``.

        Raises:
            TrackerError: If PyTorch is missing or the checkpoint is not this variant's.
        """
        if torch_module() is None:
            raise TrackerError(
                "MixFormerV2 needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        from .mixformerv2_net import (  # noqa: PLC0415 - imports torch
            MIXFORMERV2_BASE,
            MIXFORMERV2_SMALL,
            MixFormerV2Net,
            load_checkpoint,
        )

        config = MIXFORMERV2_SMALL if self.variant == "small" else MIXFORMERV2_BASE
        network = load_checkpoint(MixFormerV2Net(config), weights).to(device)
        return MixFormerV2Backend(network, device)

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
        """Read MixFormerV2's one box for one point.

        Args:
            maps: The batch's raw output: 1x1 score logits and boxes, and the
                edges' spread when the backend reports it.
            slot: This point's index in the batch.
            expected_box: The anchor's reading's box, at the target's current
                scale, in this crop.
            expected_centre: Where the anchor's reading would be now.
            reach: How far from ``expected_centre`` the box may be centred.
            box: Whether this is a drawn box rather than a point's patch.

        Returns:
            The box, ``rate`` damped by the score, and as evidence the score —
            ``0`` beyond the reach, and for a point times the squared shape
            agreement with ``expected_box``. A drawn box that changed past the
            scale gate comes back at the gate.

        Raises:
            TrackerError: If the backend returned no box.
        """
        if maps.boxes is None:
            raise TrackerError(f"{self.name} backend returned no box regression")
        cell = maps.score.shape[-1] // 2
        score = float(sigmoid(maps.score[slot, 0, cell, cell]))
        left, top, right, bottom = (float(v) for v in maps.boxes[slot, 0, :, cell, cell])
        spread = math.nan
        if isinstance(maps, MixFormerV2Maps) and maps.spread is not None:
            spread = float(maps.spread[slot, 0, cell, cell])
        dx, dy = (right - left) / 2.0, (bottom - top) / 2.0
        evidence = score
        if math.isfinite(reach) and (
            abs(dx - expected_centre[0]) > reach or abs(dy - expected_centre[1]) > reach
        ):
            evidence = 0.0
        width, height = max(left + right, 1e-6), max(top + bottom, 1e-6)
        if box:
            width, height = self._within_gate(width, height, expected_box)
        else:
            evidence *= _agreement(width, height, expected_box) ** 2
        return MixFormerV2Step(
            dx=dx,
            dy=dy,
            width=width,
            height=height,
            rate=(self.box_size_lr if box else self.size_lr) * score,
            evidence=evidence,
            score=score,
            spread=spread,
            box=box,
        )

    def _within_gate(
        self, width: float, height: float, expected: tuple[float, float]
    ) -> tuple[float, float]:
        """Pull a box that changed past :attr:`scale_gate` back to just inside it.

        The change is measured as the base's gate measures it (scale change
        times aspect change); both are shrunk in proportion, so the box moves
        toward the network's answer by as much as one frame is allowed to.
        """
        expected_w, expected_h = max(expected[0], 1e-6), max(expected[1], 1e-6)
        along_x = math.log(width / expected_w)
        along_y = math.log(height / expected_h)
        change = abs(along_x + along_y) / 2.0 + abs(along_x - along_y)
        limit = 0.999 * math.log(self.scale_gate)
        if change <= limit:
            return width, height
        shrink = limit / change
        return expected_w * math.exp(along_x * shrink), expected_h * math.exp(along_y * shrink)


class MixFormerV2SmallTracker(MixFormerV2Tracker):
    """MixFormerV2-S: the four-layer model, a box tracker fast enough for a CPU.

    The same plugin as :class:`MixFormerV2Tracker` over the small checkpoint,
    with its own defaults; see the module docstring for what it is and is not
    good at.

    Args:
        box_search_size: 287, upstream's 4.5x search for MixFormerV2-S.
        confidence_range: ``(0.5, 0.8)``: its points need holding sooner than
            -B's to come through an occluder.
        box_confidence_range: As for -B.
        **kwargs: Forwarded to :class:`MixFormerV2Tracker`.
    """

    name = "mixformerv2_small"
    display_name = "MixFormerV2-S (one-stream transformer, MCG-NJU; fast on CPU)"
    model_spec = MIXFORMERV2_SMALL_WEIGHTS
    variant = "small"

    def __init__(
        self,
        *,
        box_search_size: int = 287,
        confidence_range: tuple[float, float] = (0.5, 0.8),
        box_confidence_range: tuple[float, float] = (0.3, 0.75),
        **kwargs: Any,
    ) -> None:
        """Create an unprepared MixFormerV2-S plugin; see the class docstring."""
        super().__init__(
            box_search_size=box_search_size,
            confidence_range=confidence_range,
            box_confidence_range=box_confidence_range,
            **kwargs,
        )
