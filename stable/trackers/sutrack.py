"""SUTrack — a one-stream transformer tracker (AAAI 2025), as a Stable plugin.

SUTrack (Chen, Kang, Geng, Zhu, Liu, Wang & Lu, AAAI 2025) belongs to the
generation of trackers that dropped the Siamese split: instead of embedding the
template and the search region separately and correlating the two, it cuts both
into patch tokens and runs them through *one* transformer together, so every
layer can compare them, and a centre head on the search tokens says where the
target is and how big. It tops Ocean and SiamDW by a wide margin on the box
benchmarks (LaSOT AUC 69.6 for the T224 used here), and it does so on a 24M
parameter encoder that runs comfortably on a CPU.

How it is used here
-------------------
* The network is the published SUTrack-T224 — Fast-iTPN-T encoder, language
  token, centre head — ported to plain PyTorch in
  :mod:`stable.trackers.sutrack_net`, RGB path only. On the same inputs it
  reproduces the authors' own modules to within 2.4e-7 on the score map and
  5e-5 network pixels on the box on the CPU, and 1e-5 / 2e-3 px on a GPU, where
  upstream computes its language token through CLIP in half precision.
* It runs as a Siamese plugin (see :mod:`stable.trackers.siamese`): the
  template is cut once, at the anchor, and the point rides on the box SUTrack
  tracks. One fixed template is exactly how SUTrack-T224 itself is evaluated
  (``NUM_TEMPLATES: 1``), which is why it is the variant shipped: B224 and the
  larger ones add a second template that upstream replaces every 25 frames,
  which Stable deliberately does not do (every run starts from an
  operator-placed anchor). Measured in fixed-template mode, B224 was more
  precise (median 0.32 px against 0.42, box IoU 0.945 against 0.932) but
  recovered from fewer occlusions (10 of 12 boxes, 3 of 6 points) at twice the
  GPU time and three and a half times the CPU time; its network is ported and
  checked (:data:`~stable.trackers.sutrack_net.SUTRACK_B224`) but not offered.
* Crops are cut in the Siamese base class's frame and resampled by
  :data:`NETWORK_SCALE` (``112 / 127``), so a target's template is twice its
  box at 112 px and the trained 255 px search is four times it at 224 px —
  upstream's template and search factors. One difference: the context square
  is SiamFC's ``sqrt((w + p)(h + p))`` rather than upstream's ``2 sqrt(wh)``.
  They agree for a square box (every point) and differ by 12% of context at an
  aspect of 3; keeping SiamFC's keeps the search window the server draws for a
  box true for every Siamese plugin, and on boxes stretching to an aspect of
  2.75 the IoU stayed 0.89-0.95.
* The decoding is upstream's: the score map times its Hann window, the best
  cell, and that cell's offset and size — no penalty on scale or aspect. The
  window is centred where the anchor's own reading is expected rather than on
  the crop, as for Ocean. A box takes the regressed size as it is
  (``box_size_lr = 1``, as upstream); a point's scale is damped
  (``size_lr = 0.3``), which measured the same as ``1`` on the clips below and
  is the safer choice for a patch whose "object" is ambiguous.
* A wider search (``box_search_size``, or a ``search_radius`` beyond what 255 px
  reaches) is resampled *down* to the trained 224 px: the target then appears
  smaller than the template, which SUTrack's training (search scale jitter 0.5)
  covers. Zoomed out like this, a 351 px box search measured IoU 0.926 (0.932
  at 255), and a 100 px point search radius a median error of 0.74 px and a
  90th percentile of 1.6 px. The alternative, a larger network input with its
  position embedding resampled the way upstream's ``POS_TYPE: interpolate``
  does, broke the score map at that radius: its peak on correctly tracked
  frames fell to a median 0.48 (0.92 zoomed out) and the 90th-percentile error
  rose to 6.8 px.
* ``scale_gate`` is 2.0, not the base class's 1.3. SUTrack regresses the object
  rather than a patch beside it, and the gate's only measured effect was to
  freeze a box that had been held through an occlusion while the object
  changed size behind it: 10 of 12 such clips recovered at 1.3, 12 of 12 at 2.

Strengths and limits
--------------------
Measured with the real checkpoint on clips built the way
:mod:`stable.trackers.ocean` describes — eight 40-60 frame 480x360 clips of
known motion (handheld shake, a 30% zoom, a rotating pan, a 6 px/frame pan,
each over a natural photograph and over a blob texture; eight points each) and
six sweeps of an occluder across a point:

* Median error 0.42 px, 90th percentile 0.94 px (worst 2.5 px): 0.40 / 0.77 px
  on the natural photograph, 0.46 / 1.08 px on the blobs. Unlike Ocean it is as
  good on arbitrary natural-image patches as on objects.
* It came through five of the six occluder sweeps, reporting confidence ``0``
  while covered on all but one frame (0.15). The sixth — a static occluder over
  blobs panning 6 px a frame — held the point while the texture slid more than
  36 px under it, and it locked onto a neighbouring blob.
* On the bundled 128x96 sample clip (``stable sample``; ``stable-sample/clip.mp4``
  is the same file) its median error was 0.76 px over all ten blobs, 90th
  percentile 1.6 px — less precise than SiamDW there: the default 33 px box is a
  quarter of the frame's width, and one blob by the frame's edge strayed to
  7 px.

Boxes are what it was trained for. Over the twelve composited 60-frame clips of
:mod:`stable.trackers.ocean` (here a cartoon raccoon's face, a logo and a blob
texture, each zooming 2x, shrinking to 0.45x, stretching from an aspect of 0.8
to 2.75 and "breathing", over a photograph of a forest), its box overlapped the
truth by a mean IoU of 0.93 (10th percentile 0.86), and with a bar crossing and
fully covering the object it recovered on all twelve.

Confidence
----------
The evidence is the centre head's score — a sigmoid probability that the cell
holds the target's centre — at the chosen cell, before the window. Unlike
Ocean's classifier it falls when the target is covered: over the clips above,
steps within 2 px of the truth scored a median 0.88 (5th percentile 0.63, 1st
0.52) against a median 0.23 (90th percentile 0.43, highest 0.49) for steps where
the point was covered, which no other signal was needed to separate.
``confidence_range`` ``(0.45, 0.72)`` maps every covered step to ``0`` bar one
and the median frame of the sample clip (raw 0.73) to ``1``; its bottom is also
where a point is held, which is what brought the sweeps through — held only
below 0.2, one more of them followed the occluder off. A box's tracked frames
scored higher (median 0.93, 1st percentile 0.73) and covered ones lower (median
0.22, 90th percentile 0.32, highest 0.43), but a drawn object changes against
its fixed template in ways these clips do not, so ``box_confidence_range`` is
the lower ``(0.35, 0.65)``: a box is only held when it is clearly lost (see
:mod:`stable.trackers.ocean` for the shot that taught the base class this).

Speed
-----
6 ms a frame for one point on an RTX 5090 (12 ms for eight, batched; 240 MB of
GPU memory), and 22 ms a frame on eight threads of a Core Ultra 9 285K, where
Ocean took 62 ms.

Weights
-------
The authors' release on Hugging Face (``xche32/SUTrack``, MIT), pinned to a
revision. The file is upstream's training checkpoint, and 1 GB: nearly all of it
is OpenAI's CLIP ViT-L/14 (also MIT), which SUTrack only ever uses to produce
one constant token for RGB video; :func:`~stable.trackers.sutrack_net.load_sutrack`
computes that token once and keeps the 28M parameters that track.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import TrackerError
from .siamese import (
    EXEMPLAR_SIZE,
    SiameseBackend,
    SiameseMaps,
    SiameseStep,
    SiameseTrackerPlugin,
    cosine_window,
)
from .weights import ModelSpec, register_model

__all__ = [
    "SUTrackTracker",
    "SUTrackBackend",
    "SUTrackMaps",
    "SUTRACK_T224_WEIGHTS",
    "NETWORK_SCALE",
    "network_scale",
    "template_box",
]

logger = logging.getLogger(__name__)

_REVISION = "0a32608c19422aba4efffbaf05d0e227faedd8cf"
_HUB = f"https://huggingface.co/xche32/SUTrack/resolve/{_REVISION}/checkpoints/train/sutrack"

SUTRACK_T224_WEIGHTS = register_model(
    ModelSpec(
        key="sutrack_t224",
        url=f"{_HUB}/sutrack_t224/SUTRACK_ep0180.pth.tar",
        filename="sutrack_t224_ep0180.pth.tar",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); 1,044,947,583 bytes. It is also
        # the Hugging Face LFS digest of the file at this revision.
        sha256="5815a447480fba64d67cfa6cb011f23bb5b89465685043292464ad283f91210d",
        size_bytes=1044947583,
        license="MIT",
        non_commercial=False,
        notes=(
            "SUTrack-T224 (Chen et al., AAAI 2025), the authors' release on Hugging Face "
            "(xche32/SUTrack, MIT). The file is the training checkpoint: most of its 1 GB is "
            "OpenAI's CLIP ViT-L/14 (MIT), of which the tracker uses one constant token."
        ),
    )
)
"""The SUTrack-T224 checkpoint this plugin loads."""

TEMPLATE_SIDE = 112
"""Template side SUTrack-T224 was trained at, in network pixels."""

SEARCH_SIDE = 224
"""Search side SUTrack-T224 was trained at, in network pixels."""

PATCH = 16
"""Network pixels per score-map cell."""

TRAINED_INSTANCE = 255
"""The plugin's search side that maps onto :data:`SEARCH_SIDE` at :data:`NETWORK_SCALE`."""

NETWORK_SCALE = TEMPLATE_SIDE / EXEMPLAR_SIZE
"""Network pixels per plugin crop pixel at the trained search size.

The plugin cuts its crops in the Siamese base class's frame — a 127 px exemplar
over the target's context square, a search at the same scale, 255 px as
trained — and the backend resamples both by this one factor: the exemplar
becomes SUTrack's 112 px template (twice a square target), and the 255 px search
its 224 px one (four times it), keeping the scale relation SUTrack was trained on.
"""


def network_scale(instance_size: int) -> float:
    """Return network pixels per crop pixel for a search crop of ``instance_size``.

    :data:`NETWORK_SCALE` at the trained 255 px; a wider crop is shrunk onto the
    same 224 px input (see the module docstring for why).
    """
    return NETWORK_SCALE * TRAINED_INSTANCE / int(instance_size)


@dataclass(frozen=True)
class SUTrackMaps(SiameseMaps):
    """:class:`~stable.trackers.siamese.SiameseMaps` that carry their cell size.

    ``score`` is the centre head's score map (a probability, after its sigmoid)
    and ``boxes`` are each cell's box as distances from the cell's centre to
    the box's left, top, right and bottom edges — Ocean's layout — all in the
    plugin's search-crop pixels. SUTrack's cells are 16 network pixels apart,
    which is not a whole number of crop pixels and depends on the crop's side,
    so the maps say what it is.

    Args:
        stride: Crop pixels between neighbouring cells.
    """

    stride: float = PATCH / NETWORK_SCALE


def _cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Raises:
        TrackerError: If OpenCV is not installed.
    """
    try:
        import cv2  # noqa: PLC0415 - optional, resolved per call
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise TrackerError(
            "SUTrack needs opencv-python for its crops; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


def _torch() -> Any:
    """Return :mod:`torch`, or raise the error the plugin reports."""
    torch = torch_module()
    if torch is None:  # pragma: no cover - guarded by is_available()
        raise TrackerError("SUTrack needs PyTorch; install the 'torch' extra")
    return torch


def _resample(crops: np.ndarray, side: int, scale: float) -> np.ndarray:
    """Resample square crops about their centres by ``scale`` onto ``side`` pixels.

    Pixel centres map exactly: output ``u`` samples input
    ``(n - 1) / 2 + (u - (side - 1) / 2) / scale``. Where that falls a fraction
    of a pixel past the crop's edge, the edge is repeated.

    Args:
        crops: ``[B, n, n, 3]`` uint8.
        side: Output side.
        scale: Output pixels per input pixel.

    Returns:
        ``[B, side, side, 3]`` uint8.
    """
    cv2 = _cv2()
    size = crops.shape[1]
    shift = (side - 1) / 2.0 - scale * (size - 1) / 2.0
    matrix = np.array([[scale, 0.0, shift], [0.0, scale, shift]])
    return np.stack(
        [
            cv2.warpAffine(
                np.ascontiguousarray(crop),
                matrix,
                (side, side),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for crop in crops
        ]
    )


def template_box(width: float, height: float, side: int = TEMPLATE_SIDE) -> np.ndarray:
    """Return upstream's normalised ``(x0, y0, w, h)`` of a box centred in the template.

    Upstream's ``transform_image_to_crop(..., normalize=True)``: the box's centre
    sits on the template's centre pixel, ``(side - 1) / 2``, and everything is
    divided by ``side - 1``.

    Args:
        width: Box width in template pixels.
        height: Box height in template pixels.
        side: Template side.

    Returns:
        ``[4]`` float64.
    """
    span = side - 1.0
    centre = span / 2.0
    return np.array(
        [(centre - width / 2.0) / span, (centre - height / 2.0) / span, width / span, height / span]
    )


class SUTrackBackend:
    """Runs :class:`~stable.trackers.sutrack_net.SUTrackNet` on the plugin's crops.

    Owns SUTrack's input convention so nothing else has to: crops are resampled
    to its sizes (:func:`network_scale`) and normalised with ImageNet statistics
    in RGB order, and every network call carries each template's target box,
    which the encoder marks its foreground tokens with. SUTrack is one-stream —
    template and search are encoded together — so the "template" :meth:`embed`
    returns is just the normalised template image, and all the network's work
    happens in :meth:`match`.

    Args:
        network: A loaded ``SUTrackNet``, on ``device``, in evaluation mode.
        device: The device string the network lives on.

    Attributes:
        device: Where the network currently is; updated by :meth:`to_device`.
    """

    def __init__(self, network: Any, device: str) -> None:
        """Wrap a loaded network."""
        self._network = network
        self.device = device
        self._sizes: np.ndarray | None = None

    def use_template_boxes(self, sizes: np.ndarray) -> None:
        """Say how big each template's target is, for the matches that follow.

        Args:
            sizes: ``[N, 2]`` width and height of each point's target box in its
                127 px exemplar crop. Without it — or for a batch of another
                size — every target is taken to be the centred square half the
                exemplar's side, which is what a point's box always is.
        """
        self._sizes = np.asarray(sizes, dtype=np.float64).reshape(-1, 2)

    def _normalised(self, crops: np.ndarray) -> Any:
        """Return ``[B, n, n, 3]`` uint8 RGB as an ImageNet-normalised ``[B, 3, n, n]`` tensor."""
        torch = _torch()
        from .sutrack_net import IMAGENET_MEAN, IMAGENET_STD  # noqa: PLC0415 - imports torch

        tensor = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return ((tensor - mean) / std).contiguous()

    def embed(self, exemplars: np.ndarray) -> Any:
        """Return the normalised ``[N, 3, 112, 112]`` templates for ``exemplars``."""
        torch = _torch()
        side = int(self._network.config.template_size)
        with torch.inference_mode():
            return self._normalised(_resample(exemplars, side, side / exemplars.shape[1]))

    def _template_boxes(self, count: int) -> np.ndarray:
        """Return the ``[count, 4]`` normalised template boxes for this batch."""
        side = int(self._network.config.template_size)
        sizes = self._sizes
        if sizes is None or len(sizes) != count:
            sizes = np.full((count, 2), EXEMPLAR_SIZE / 2.0)
        factor = side / EXEMPLAR_SIZE
        return np.stack([template_box(w * factor, h * factor, side) for w, h in sizes])

    def match(self, kernels: Any, searches: np.ndarray) -> SiameseMaps:
        """Score ``searches`` (``[N, S, X, X, 3]``) against ``kernels``.

        Returns:
            :class:`SUTrackMaps` in the ``X`` px crop's pixels.
        """
        torch = _torch()
        count, scales, size = (int(v) for v in searches.shape[:3])
        side = int(self._network.config.search_size)
        scale = network_scale(size)
        templates_per_search = int(self._network.config.num_templates)
        boxes = np.repeat(self._template_boxes(count), scales, axis=0)
        with torch.inference_mode():
            search = self._normalised(
                _resample(searches.reshape(count * scales, *searches.shape[2:]), side, scale)
            )
            templates = kernels.repeat_interleave(scales, dim=0)[:, None]
            templates = templates.expand(-1, templates_per_search, -1, -1, -1)
            box = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
            box = box[:, None].expand(-1, templates_per_search, -1)
            score, extent, offset = (
                value.detach().double().cpu().numpy()
                for value in self._network(templates, box, search)
            )
        cells = score.shape[-1]
        span = side - 1.0
        # Upstream's training labels put a normalised 0.5 on the crop's centre
        # pixel and divide by side - 1; the cells are symmetric about that centre.
        positions = (np.arange(cells) - (cells - 1) / 2.0) * PATCH
        index = np.arange(cells, dtype=np.float64)
        centre_x = ((index[None, :] + offset[:, 0]) / cells - 0.5) * span
        centre_y = ((index[:, None] + offset[:, 1]) / cells - 0.5) * span
        width, height = extent[:, 0] * span, extent[:, 1] * span
        edges = np.stack(
            [
                positions[None, None, :] - (centre_x - width / 2.0),
                positions[None, :, None] - (centre_y - height / 2.0),
                (centre_x + width / 2.0) - positions[None, None, :],
                (centre_y + height / 2.0) - positions[None, :, None],
            ],
            axis=1,
        )
        return SUTrackMaps(
            score=score.reshape(count, scales, cells, cells),
            boxes=(edges / scale).reshape(count, scales, 4, cells, cells),
            stride=PATCH / scale,
        )

    def to_device(self, device: str) -> None:
        """Move the network to ``device``."""
        self._network.to(device)
        self.device = device

    def release(self) -> None:
        """Drop the reference to the network so its memory can be reclaimed."""
        self._network = None


class SUTrackTracker(SiameseTrackerPlugin):
    """SUTrack-T224 as a Stable tracker plugin.

    Args:
        template_radius: Half-size of the target box, in source pixels.
        search_radius: How far the point may move between frames, in source
            pixels. The default is well inside what the trained search reaches
            with the default box (61 px); beyond that the search is widened by
            shrinking it onto the network's input, which costs precision.
        size_lr: How far a point's tracked scale moves toward the size SUTrack
            regresses, each frame.
        box_size_lr: ``size_lr`` for a box the operator drew: ``1``, upstream's
            own rule of taking the regressed box as it is.
        confidence_range: ``(low, high)`` calibration of the centre score; see
            the module docstring for the distributions behind it.
        box_confidence_range: The same for a box, whose bottom is also where a
            box is held (``box_lost_below``); see the module docstring for why it
            sits below a point's.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.siamese.SiameseTrackerPlugin`
            (``lost_below``, ``box_search_size``, ``device``, ``backend``, ...).
            ``window_influence`` is accepted and has no effect: SUTrack's Hann
            window multiplies its score map at full strength, as upstream's does.
    """

    name = "sutrack"
    display_name = "SUTrack-T (one-stream transformer, AAAI 2025)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "MIT"

    model_spec = SUTRACK_T224_WEIGHTS
    scale_gate = 2.0

    def __init__(
        self,
        *,
        template_radius: int = 16,
        search_radius: int = 48,
        size_lr: float = 0.3,
        box_size_lr: float = 1.0,
        confidence_range: tuple[float, float] = (0.45, 0.72),
        box_confidence_range: tuple[float, float] = (0.35, 0.65),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared SUTrack plugin; see the class docstring."""
        super().__init__(
            template_radius=template_radius,
            search_radius=search_radius,
            confidence_range=confidence_range,
            box_confidence_range=box_confidence_range,
            visible_threshold=visible_threshold,
            **kwargs,
        )
        self.size_lr = float(size_lr)
        self.box_size_lr = float(box_size_lr)

    def score_size(self, instance_size: int) -> int:
        """Return SUTrack's score-map side: 14 cells, whatever the crop's side."""
        return SEARCH_SIDE // PATCH

    def _reach_of(self, instance_size: int) -> float:
        """How far, in crop pixels, the outermost score cell is from the centre."""
        cells = self.score_size(instance_size)
        return (cells - 1) / 2.0 * PATCH / network_scale(instance_size)

    def _build_backend(self, device: str, weights: Any) -> SiameseBackend:
        """Load SUTrack-T224 from ``weights`` onto ``device``.

        Raises:
            TrackerError: If PyTorch is missing or the checkpoint is not SUTrack's.
        """
        if torch_module() is None:
            raise TrackerError(
                "SUTrack needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        from .sutrack_net import (  # noqa: PLC0415 - imports torch
            SUTRACK_T224,
            build_sutrack,
            load_sutrack,
        )

        network = load_sutrack(build_sutrack(SUTRACK_T224), weights).to(device)
        return SUTrackBackend(network, device)

    def _match(self, anchor: Any, searches: np.ndarray) -> SiameseMaps:
        """Tell the backend each template's target box, then match as the base class does.

        SUTrack's encoder marks the template tokens the target covers, from its
        box: a point's is the ``2r + 1`` square, half the exemplar's side, but a
        drawn box can have any shape. Set before every call, so the templates
        rebuilt after a fall back to the CPU are marked the same way.
        """
        configure = getattr(self._backend, "use_template_boxes", None)
        if callable(configure):
            configure(
                np.array(
                    [
                        (seed.width * seed.anchor_scale, seed.height * seed.anchor_scale)
                        for seed in anchor.seeds
                    ]
                )
            )
        return super()._match(anchor, searches)

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
        """Pick SUTrack's box for one point: the windowed score's peak, within reach.

        Args:
            maps: The batch's raw output; SUTrack needs ``boxes``, and reads the
                cell size from :class:`SUTrackMaps` when the backend sends it.
            slot: This point's index in the batch.
            expected_box: Unused: SUTrack applies no scale or aspect penalty.
            expected_centre: Where the reference box would be; the Hann window
                and the reach are centred on it.
            reach: How far from ``expected_centre`` a cell may be chosen.
            box: Whether this is a drawn box, whose size follows at
                ``box_size_lr`` rather than ``size_lr``.

        Returns:
            The chosen cell's box, the size rate, and the centre score there,
            before the window, as the evidence.

        Raises:
            TrackerError: If the backend returned no box maps.
        """
        if maps.boxes is None:
            raise TrackerError(f"{self.name} backend returned no box regression")
        stride = float(getattr(maps, "stride", PATCH / NETWORK_SCALE))
        score = maps.score[slot, 0].astype(np.float64)
        left, top, right, bottom = maps.boxes[slot, 0].astype(np.float64)
        cells = score.shape[-1]
        grid = (np.arange(cells) - (cells - 1) / 2.0) * stride
        grid_x, grid_y = np.meshgrid(grid, grid)
        width = np.maximum(left + right, 1e-6)
        height = np.maximum(top + bottom, 1e-6)
        centre_x = grid_x + (right - left) / 2.0
        centre_y = grid_y + (bottom - top) / 2.0

        # Upstream's hann2d(centered=True): it would reach zero one cell past each edge.
        half_span = (cells + 1) / 2.0 * stride
        window = np.outer(
            cosine_window(grid, expected_centre[1], half_span),
            cosine_window(grid, expected_centre[0], half_span),
        )
        response = score * window
        if math.isfinite(reach):
            limit = reach + stride / 2.0
            outside = (np.abs(grid_x - expected_centre[0]) > limit) | (
                np.abs(grid_y - expected_centre[1]) > limit
            )
            if not outside.all():
                response = np.where(outside, -np.inf, response)
        row, col = np.unravel_index(int(np.argmax(response)), response.shape)
        return SiameseStep(
            dx=float(centre_x[row, col]),
            dy=float(centre_y[row, col]),
            width=float(width[row, col]),
            height=float(height[row, col]),
            rate=self.box_size_lr if box else self.size_lr,
            evidence=float(score[row, col]),
        )
