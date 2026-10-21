"""SiamDW — "deeper and wider" SiamFC (TracKit), as a Stable plugin.

SiamDW (Zhang & Peng, CVPR 2019) asks why Siamese trackers stopped improving
when given deeper backbones, and answers: padding. A padded convolution makes a
feature depend on where it sits in the crop, and a tracker whose whole method is
"slide the template over the search and see where it fits best" needs features
that do not. Its cropping-inside-residual units (CIR) crop away every cell the
padding touched, which lets a 22-layer, double-width ResNet (CIResNet-22W) sit
under the original SiamFC correlation and beat it handily.

How it is used here
-------------------
* The network is the published CIResNet-22W under SiamFC's plain correlation,
  ported to PyTorch in :mod:`stable.trackers.siamese_nets`.
* The decoding is SiamFC's, as TracKit does it: three search scales, the
  response upsampled 16x with bicubic interpolation so the peak lands between
  cells, a penalty on the two changed scales, a cosine window, and a damped
  scale update. Hyper-parameters are SOTS's OTB-2015 tuning of this checkpoint.
  Two details differ, both for sub-pixel accuracy: the peak's position is read
  with the resampling's exact pixel-centre convention (TracKit's is half an
  upsampled cell off), and it is converted back to source pixels at the scale it
  was *found* at rather than the previous one.
* The point rides on the correlation peak (see :mod:`stable.trackers.siamese`);
  SiamDW's peak is unbiased, so this changes nothing measurable, but it keeps
  the two Siamese plugins reading their networks the same way.

Strengths and limits
--------------------
Measured with the real checkpoint on the clips :mod:`stable.trackers.ocean`
describes (eight of known motion, eight points each, plus six occluder sweeps):
median error 0.65 px and 90th percentile 1.8 px overall, 0.92 / 2.5 px on smooth
natural-image patches, and it came through all six occluder sweeps (worst error
afterwards 2.5 px) reporting confidence ``0`` while covered. It is the more
reliable of the two Siamese plugins on arbitrary features. ``lk`` remains the
more precise on clean, unoccluded motion (0.19 px median on the same clips) and
came through none of the sweeps — the trade these trackers exist for.

It tracks boxes too, as SiamFC does: it searches scales, so a box keeps the
aspect it was drawn with. On the composited box clips described in
:mod:`stable.trackers.ocean` its box overlapped the truth by a mean IoU of 0.68:
close behind Ocean on zooms and shrinks, though lagging a 2x zoom by about 10%,
and lost on all three clips where the object stretched to over twice its
aspect. Ocean is the box tracker for anything that changes shape. A box has its
own calibration, ``box_confidence_range`` ``(0.4, 0.65)``, whose bottom is also
where the box is held: the point calibration zeroed — and held still — half the
frames of a real shot of a dog running through grass (response ratio median
0.55), while on the composited clips frames with the object covered never
exceeded 0.44 (90th percentile 0.34) and correctly tracked ones measured median
0.72 (1st percentile 0.56).

Confidence
----------
SiamFC's response is a similarity, not a probability, and its scale depends on
the template — measured peaks on matching content ranged from 7 to 15. So the
evidence is the response where the target was found **divided by the response
the template gets on its own anchor frame**: "how well does this frame match,
compared with how well the anchor matches itself". Tracking the clips above,
that ratio measured a median 0.87 (5th percentile 0.77) on steps within 2 px of
the truth, against a median 0.35 and a highest 0.52 on steps where the point
was covered. It runs lower where the features are only a few pixels across: on
the bundled 128x96 sample clip it measured a median 0.73 while tracking to
0.36 px. ``confidence_range`` of ``(0.55, 0.72)`` covers both — every covered
step maps to ``0``, the median sample-clip frame to ``1``.

Weights
-------
The GOT-10k-trained CIResNet-22W SiamFC. The file comes from the authors' SOTS
model folder (TracKit's own link has gone); its tensors are identical to the
``state_dict`` in ``SiamFCRes22W.pth`` linked from the MIT-licensed
``researchmm/SiamDW`` README, of which it is the weights alone.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import ContractError, TrackerError
from .siamese import (
    SiameseBackend,
    SiameseMaps,
    SiameseStep,
    SiameseTrackerPlugin,
    TorchSiameseBackend,
    cosine_window,
)
from .weights import ModelSpec, register_model

__all__ = ["SiamDWTracker", "SIAMDW_RES22W"]

logger = logging.getLogger(__name__)


def _cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Raises:
        TrackerError: If OpenCV is not installed.
    """
    try:
        import cv2  # noqa: PLC0415 - optional, resolved per call
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise TrackerError(
            "SiamDW needs opencv-python to read its response map; install the 'video' "
            "extra (`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


SIAMDW_RES22W = register_model(
    ModelSpec(
        key="siamdw_res22w",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1t0Z8Cul8fRbBcL0nxguFObTtcoqLkSVW&export=download&confirm=t"
        ),
        filename="SiamDW.pth",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); 9,920,295 bytes.
        sha256="1392552b0be47bef518219c47df8ccc7abbf1bc63c44ca5039e66c38ca9bedad",
        size_bytes=9920295,
        license="MIT",
        non_commercial=False,
        notes=(
            "SiamDW (Zhang & Peng, CVPR 2019): SiamFC with a CIResNet-22W backbone, "
            "trained on GOT-10k. Tensor-for-tensor the state_dict of SiamFCRes22W.pth "
            "from the MIT-licensed researchmm/SiamDW release; this copy, from the "
            "authors' SOTS model folder, holds the weights alone."
        ),
    )
)
"""The SiamDW checkpoint this plugin loads."""


class SiamDWTracker(SiameseTrackerPlugin):
    """SiamDW as a Stable tracker plugin.

    Args:
        template_radius: Half-size of the target box, in source pixels.
        search_radius: How far the point may move between frames, in source
            pixels. The default is what the trained 255 px search reaches with
            the default box.
        scale_step: Ratio between neighbouring search scales; three are searched
            per frame (``1 / step``, ``1`` and ``step``).
        scale_penalty: Factor applied to the changed scales' peaks, so the scale
            only changes when the evidence for it is clearly better.
        scale_lr: How far the target's scale moves toward the winning one each
            frame.
        window_influence: Weight of the cosine prior favouring small motion.
        response_upsample: Factor the 17x17 response is upsampled by before its
            peak is read; ``16`` puts it on a half-network-pixel grid.
        confidence_range: ``(low, high)`` calibration of the response ratio; see
            the module docstring for the distributions behind it.
        box_confidence_range: The same for a box, whose bottom is also where a
            box is held; see the module docstring.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.siamese.SiameseTrackerPlugin`
            (``lost_below``, ``device``, ``backend``, ...).

    Raises:
        ContractError: If ``scale_step`` is below 1 or ``response_upsample`` is
            not positive.
    """

    name = "siamdw"
    display_name = "SiamDW (CIResNet-22W SiamFC, TracKit)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "MIT"

    model_spec = SIAMDW_RES22W

    def __init__(
        self,
        *,
        template_radius: int = 16,
        search_radius: int = 32,
        scale_step: float = 1.089,
        scale_penalty: float = 0.954,
        scale_lr: float = 0.245,
        window_influence: float = 0.292,
        response_upsample: int = 16,
        confidence_range: tuple[float, float] = (0.55, 0.72),
        box_confidence_range: tuple[float, float] = (0.4, 0.65),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared SiamDW plugin; see the class docstring."""
        super().__init__(
            template_radius=template_radius,
            search_radius=search_radius,
            window_influence=window_influence,
            confidence_range=confidence_range,
            box_confidence_range=box_confidence_range,
            visible_threshold=visible_threshold,
            **kwargs,
        )
        if scale_step < 1.0:
            raise ContractError(f"scale_step must be >= 1, got {scale_step}")
        if int(response_upsample) < 1:
            raise ContractError(f"response_upsample must be >= 1, got {response_upsample}")
        self.scale_step = float(scale_step)
        self.scale_penalty = float(scale_penalty)
        self.scale_lr = float(scale_lr)
        self.response_upsample = int(response_upsample)
        # The reference scale first: the decoder reports in its pixels, and the
        # anchor's own reading is taken against it.
        self.search_scales = (1.0, 1.0 / self.scale_step, self.scale_step)

    def score_size(self, instance_size: int) -> int:
        """Return SiamFC's response side for a search crop of ``instance_size``."""
        return (int(instance_size) - self.exemplar_size) // self.total_stride + 1

    def raw_confidence(self, step: SiameseStep, reference: SiameseStep) -> float:
        """Return the response found, as a fraction of the anchor's own response.

        See the module docstring: a raw response means nothing across templates,
        a ratio to what the template scores on its own frame does.
        """
        if reference.evidence <= 0.0:
            return 0.0
        return float(step.evidence / reference.evidence)

    def _build_backend(self, device: str, weights: Any) -> SiameseBackend:
        """Load SiamDW from ``weights`` onto ``device``.

        Raises:
            TrackerError: If PyTorch is missing or the checkpoint is not SiamDW's.
        """
        if torch_module() is None:
            raise TrackerError(
                "SiamDW needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        from .siamese_nets import SiamDWNet, load_checkpoint  # noqa: PLC0415 - imports torch

        network = load_checkpoint(SiamDWNet(), weights).to(device)
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
        """Pick the best scale and its peak, SiamFC-style.

        Args:
            maps: The batch's raw responses, one per search scale.
            slot: This point's index in the batch.
            expected_box: The reference box in the reference crop; the box found
                is that, at the winning scale.
            expected_centre: Where the reference peak would be; the window and
                the reach are centred on it.
            reach: How far from ``expected_centre`` the peak may be.
            box: Unused: SiamFC's scale search is the same for a box and a point.

        Returns:
            The peak, in the reference crop's pixels, the winning scale's box,
            ``scale_lr``, and the raw response at the peak as the evidence.
        """
        cv2 = _cv2()
        responses = maps.score[slot]
        cells = responses.shape[-1]
        size = cells * self.response_upsample
        upsampled = cv2.resize(
            np.ascontiguousarray(np.moveaxis(responses, 0, -1)),
            (size, size),
            interpolation=cv2.INTER_CUBIC,
        )
        if upsampled.ndim == 2:  # OpenCV drops a single channel
            upsampled = upsampled[..., None]
        penalties = np.array([1.0 if k == 1.0 else self.scale_penalty for k in self.search_scales])
        peaks = upsampled.reshape(-1, upsampled.shape[-1]).max(axis=0) * penalties
        best = int(np.argmax(peaks))
        relative = self.search_scales[best]
        response = upsampled[..., best].astype(np.float64)

        # Exact pixel-centre mapping of the bicubic upsampling (OpenCV samples
        # source coordinate (u + 0.5) / factor - 0.5), in this scale's crop pixels.
        positions = (
            (np.arange(size) + 0.5) / self.response_upsample - 0.5 - (cells - 1) / 2.0
        ) * self.total_stride
        centre_x, centre_y = expected_centre[0] / relative, expected_centre[1] / relative
        half_span = (size - 1) / 2.0 * self.total_stride / self.response_upsample
        window = np.outer(
            cosine_window(positions, centre_y, half_span),
            cosine_window(positions, centre_x, half_span),
        )
        window /= max(float(window.sum()), 1e-12)
        shifted = response - response.min()
        total = float(shifted.sum())
        normalised = shifted / total if total > 0 else np.full_like(shifted, 1.0 / shifted.size)
        score = (1.0 - self.window_influence) * normalised + self.window_influence * window
        if math.isfinite(reach):
            limit = reach / relative
            outside = (np.abs(positions - centre_y)[:, None] > limit) | (
                np.abs(positions - centre_x)[None, :] > limit
            )
            if not outside.all():
                score = np.where(outside, -np.inf, score)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        return SiameseStep(
            dx=float(positions[col] * relative),
            dy=float(positions[row] * relative),
            width=float(expected_box[0] * relative),
            height=float(expected_box[1] * relative),
            rate=self.scale_lr,
            evidence=float(response[row, col]),
        )
