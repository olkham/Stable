"""Ocean — object-aware, anchor-free Siamese tracking (TracKit), as a Stable plugin.

Ocean (Zhang, Peng, Fu, Li & Hu, ECCV 2020) is a Siamese tracker whose head is
anchor-free: rather than scoring a fixed set of preset boxes the way SiamRPN's
anchors do, every cell of its score map regresses the distance from itself to
each edge of the target's box, and a classification branch says which cells lie
on the target. The box comes from whichever cell the classifier trusts, not from
the nearest preset shape — which is why it is accurate where anchor-based
trackers are coarse, and why it holds on through large changes of aspect.

How it is used here
-------------------
* The network is the published one — ResNet-50 to stage 3, the 1x1 neck, and
  the multi-dilation head — ported to plain PyTorch in
  :mod:`stable.trackers.siamese_nets`. The deformable-convolution
  "object-aware" refinement branch is not, because the published checkpoint
  does not include it.
* The decoding is TracKit's: a penalty on boxes whose scale or aspect differ
  from what is expected, a cosine window favouring small motion, the best cell's
  box, and a damped size update weighted by how sure the classifier is. The
  hyper-parameters are the ones SOTS tuned this checkpoint to on OTB-2015, bar
  one: ``size_lr`` is 0.3, not 0.519. The box Ocean regresses around a patch is
  a far noisier scale signal than the box around an object, and the scale it
  drives is the search region's; together with the base class's isotropic,
  shape-gated scale (:attr:`~stable.trackers.siamese.SiameseTrackerPlugin.scale_gate`)
  that took the measured 90th-percentile error on handheld footage from 19 px to
  under 10.
* The point **rides** on Ocean's box (see :mod:`stable.trackers.siamese`). This
  is the network that needs it: handed a patch of a face it regresses the box
  of the collar or the cheek next to it, a few pixels off the patch, and
  reporting that box's centre measured 3.7 px median error on shifted natural
  images where riding on its motion measured 0.9 px.

Strengths and limits
--------------------
Measured with the real checkpoint over eight 40-60 frame clips of known motion
(handheld shake, a 30% zoom, a slow rotating pan and a 6 px/frame pan, each over
a natural photograph and over a blob texture; eight points each) and six sweeps
of an occluder across a point:

* On object-like features — blobs, markers, anything with a boundary Ocean can
  call an object — it is accurate (median 0.53 px, 90th percentile 1.1 px),
  through the zoom included.
* On smooth natural-image patches it is not: median 1.4 px but a 90th
  percentile of 12 px, because with no object in the patch its box wanders
  between neighbouring structures. SiamDW measured 0.9 / 2.5 px on the same
  patches and is the better choice for an arbitrary feature.
* It came through five of the six occluder sweeps, reporting confidence of at
  most 0.02 while covered (``lk`` came through none, reporting 1.0).

Boxes are what it was built for (see :mod:`stable.trackers.siamese`). Over
twelve composited 60-frame clips — a face, a logo and a blob texture, each
zooming 2x, shrinking to 0.45x, stretching from an aspect of 0.8 to 2.75, and
"breathing" in and out of shape, over a natural background — its box overlapped
the truth by a mean IoU of 0.92 (10th percentile 0.87), and with an occluder
crossing the object as well it recovered on eleven of the twelve. Three settings
differ from a point's. The size follows the box at TracKit's own
``box_size_lr`` of 0.519: the 0.3 that steadies a patch lagged an object
changing shape (mean IoU 0.86). A box is held only below raw evidence 0.3
(``box_lost_below``): the template is fixed, so a stretching object's evidence
against it fell under a point's threshold while it was still the best match, and
the held box was left behind on three of the twelve clips. And a box has its own
calibration, ``box_confidence_range`` ``(0.3, 0.6)``, which starts where the box
stops being held. The point calibration zeroed a fifth of a real shot it was
tracking correctly — a dog running through grass, 860 4K frames at 120 fps, the
box on the dog throughout, raw evidence median 0.55 and 5th percentile 0.38 —
and the solver, rightly refusing zero-confidence frames, invented the camera
motion across all 182 of them. On the composited clips correctly tracked frames
measured median 0.76 (5th percentile 0.53) and frames with the object covered
median 0.18 (90th percentile 0.42): from 0.3 up, 0.1% of the dog's frames and
74% of the covered ones read ``0``.

Confidence
----------
Ocean's classifier on its own is not a usable confidence. It was only ever
trained on pairs in which the target is present, so it keeps scoring whatever is
most target-like when it is not: measured on natural images it gave a median
0.86 with the target covered and 0.84 with the target outside the search region
altogether, against 0.99 when it was there. The evidence is therefore the
classifier **times** the cosine between the template's features and the search
features under the position found — Ocean's own backbone features, computed
alongside its head (:meth:`~stable.trackers.siamese_nets.OceanNet.match`). An
occluder still looks like *an* object, just not like this one. Over the clips
above that product separated steps within 2 px of the truth (5th percentile
0.59, median 0.71) from covered ones (median 0.25, highest 0.44), and ranked
steps over 3 px off below good ones with an AUC of 0.98; ``confidence_range``
of ``(0.45, 0.62)`` maps that onto the shared scale, with every covered step at
``0``.

Weights
-------
TracKit's own Google Drive links have gone; its README now points at the
authors' SOTS repository for models, and that is where this checkpoint comes
from (``Ocean.pth``, the offline ResNet-50 model). The code and the README that
designates it are MIT-licensed; the SOTS repository hosting the file carries no
licence of its own, which is worth knowing before shipping work made with it.
Pinned like every other checkpoint, from a real download that was loaded,
matched against the network, and tracked with (CONTRIBUTING.md, "Pinning a
checkpoint").
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import TrackerError
from .confidence import sigmoid
from .siamese import (
    SiameseBackend,
    SiameseMaps,
    SiameseStep,
    SiameseTrackerPlugin,
    TorchSiameseBackend,
    cosine_window,
    sample_map,
)
from .weights import ModelSpec, register_model

__all__ = ["OceanTracker", "OCEAN_OFFLINE"]

logger = logging.getLogger(__name__)

OCEAN_OFFLINE = register_model(
    ModelSpec(
        key="ocean_offline",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1lVEdXBi4DsfiAVohoYscQjV4TjFwcwXE&export=download&confirm=t"
        ),
        filename="Ocean.pth",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); 103,788,031 bytes.
        sha256="16badb707871ba83a786f32fce5bdf7946254679cbc0051f32bf1f2c8e6cd805",
        size_bytes=103788031,
        license="MIT",
        non_commercial=False,
        notes=(
            "Ocean (Zhang et al., ECCV 2020), offline ResNet-50 model, from the authors' "
            "SOTS model folder that TracKit's MIT-licensed README links as its model "
            "source. The SOTS repository itself carries no licence file."
        ),
    )
)
"""The Ocean checkpoint this plugin loads."""


def _change(ratio: np.ndarray) -> np.ndarray:
    """Return how far a ratio is from 1, symmetrically: ``max(r, 1 / r)``."""
    return np.maximum(ratio, 1.0 / ratio)


def _padded_size(width: np.ndarray | float, height: np.ndarray | float) -> np.ndarray:
    """Return the side of a box's context square — TracKit's notion of "size"."""
    pad = (np.asarray(width) + np.asarray(height)) * 0.5
    return np.sqrt((np.asarray(width) + pad) * (np.asarray(height) + pad))


class OceanTracker(SiameseTrackerPlugin):
    """Ocean as a Stable tracker plugin.

    Args:
        template_radius: Half-size of the target box, in source pixels.
        search_radius: How far the point may move between frames, in source
            pixels. The default is what the trained 255 px search reaches with
            the default box.
        penalty_k: Strength of the penalty on boxes whose scale or aspect differ
            from the expected one. Raise it for a feature that keeps its shape,
            lower it for one that genuinely deforms.
        window_influence: Weight of the cosine prior favouring small motion.
        size_lr: How fast the tracked size follows the regressed box, scaled
            further by the penalty and the classifier's certainty each frame.
        box_size_lr: ``size_lr`` for a box the operator drew: TracKit's own
            0.519, since that box *is* the object Ocean was trained to size.
        confidence_range: ``(low, high)`` calibration of classifier x feature
            cosine; see the module docstring for the distributions behind it.
        box_confidence_range: The same for a box, whose bottom is also where a
            box is held (``box_lost_below``); see the module docstring for why
            it sits below a point's.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.siamese.SiameseTrackerPlugin`
            (``lost_below``, ``device``, ``backend``, ...). Lowering
            ``lost_below`` to about 0.3 helps Ocean through fast pans over weak
            texture, at the price of following an occluder more often.
    """

    name = "ocean"
    display_name = "Ocean (anchor-free Siamese, TracKit)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "MIT"

    model_spec = OCEAN_OFFLINE

    def __init__(
        self,
        *,
        template_radius: int = 16,
        search_radius: int = 48,
        penalty_k: float = 0.109,
        window_influence: float = 0.2,
        size_lr: float = 0.3,
        box_size_lr: float = 0.519,
        confidence_range: tuple[float, float] = (0.45, 0.62),
        box_confidence_range: tuple[float, float] = (0.3, 0.6),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared Ocean plugin; see the class docstring."""
        super().__init__(
            template_radius=template_radius,
            search_radius=search_radius,
            window_influence=window_influence,
            confidence_range=confidence_range,
            visible_threshold=visible_threshold,
            box_confidence_range=box_confidence_range,
            **kwargs,
        )
        self.penalty_k = float(penalty_k)
        self.size_lr = float(size_lr)
        self.box_size_lr = float(box_size_lr)

    def score_size(self, instance_size: int) -> int:
        """Return Ocean's score-map side for a search crop of ``instance_size``.

        Eight cells more than a plain SiamFC correlation gives, because the
        template map is cropped from 15x15 to 7x7 before matching.
        """
        return (int(instance_size) - self.exemplar_size) // self.total_stride + 9

    def _build_backend(self, device: str, weights: Any) -> SiameseBackend:
        """Load Ocean from ``weights`` onto ``device``.

        Raises:
            TrackerError: If PyTorch is missing or the checkpoint is not Ocean's.
        """
        if torch_module() is None:
            raise TrackerError(
                "Ocean needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        from .siamese_nets import OceanNet, load_checkpoint  # noqa: PLC0415 - imports torch

        network = load_checkpoint(OceanNet(), weights).to(device)
        return TorchSiameseBackend(network, device)

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
        """Pick Ocean's box for one point: penalised, windowed, within reach.

        Args:
            maps: The batch's raw output; Ocean needs ``boxes`` and uses
                ``similarity`` when present.
            slot: This point's index in the batch.
            expected_box: The reference box's size in this crop, the target of
                the scale and aspect penalty.
            expected_centre: Where the reference box would be; the window and
                the reach are centred on it.
            reach: How far from ``expected_centre`` a cell may be chosen.
            box: Whether this is a drawn box, damped by ``box_size_lr``.

        Returns:
            The chosen cell's box, its damped size rate, and classifier x
            feature cosine as the evidence.

        Raises:
            TrackerError: If the backend returned no box maps.
        """
        if maps.boxes is None:
            raise TrackerError(f"{self.name} backend returned no box regression")
        logits = maps.score[slot, 0].astype(np.float64)
        left, top, right, bottom = maps.boxes[slot, 0].astype(np.float64)
        size = logits.shape[-1]
        stride = float(self.total_stride)
        grid = (np.arange(size) - size // 2) * stride
        grid_x, grid_y = np.meshgrid(grid, grid)
        width = np.maximum(left + right, 1e-6)
        height = np.maximum(top + bottom, 1e-6)
        centre_x = grid_x + (right - left) / 2.0
        centre_y = grid_y + (bottom - top) / 2.0

        cls = sigmoid(logits)
        expected_w, expected_h = max(expected_box[0], 1e-6), max(expected_box[1], 1e-6)
        scale_change = _change(_padded_size(width, height) / _padded_size(expected_w, expected_h))
        aspect_change = _change((expected_w / expected_h) / (width / height))
        penalty = np.exp(-(aspect_change * scale_change - 1.0) * self.penalty_k)

        half_span = (size - 1) / 2.0 * stride
        window = np.outer(
            cosine_window(grid, expected_centre[1], half_span),
            cosine_window(grid, expected_centre[0], half_span),
        )
        score = penalty * cls * (1.0 - self.window_influence) + window * self.window_influence
        if math.isfinite(reach):
            limit = reach + stride / 2.0
            outside = (np.abs(grid_x - expected_centre[0]) > limit) | (
                np.abs(grid_y - expected_centre[1]) > limit
            )
            if not outside.all():
                score = np.where(outside, -np.inf, score)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)

        found_x, found_y = float(centre_x[row, col]), float(centre_y[row, col])
        evidence = float(cls[row, col])
        if maps.similarity is not None:
            cosine = sample_map(
                maps.similarity[slot, 0], found_x / stride + size // 2, found_y / stride + size // 2
            )
            evidence *= max(cosine, 0.0)
        return SiameseStep(
            dx=found_x,
            dy=found_y,
            width=float(width[row, col]),
            height=float(height[row, col]),
            rate=float(
                penalty[row, col] * cls[row, col] * (self.box_size_lr if box else self.size_lr)
            ),
            evidence=evidence,
        )
