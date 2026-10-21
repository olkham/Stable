"""Siamese trackers as point trackers: a fixed template, searched for frame by frame.

Ocean (:mod:`stable.trackers.ocean`) and SiamDW (:mod:`stable.trackers.siamdw`)
are *single-object box* trackers. Given a box on the first frame they cut an
"exemplar" around it once, and on every later frame cut a larger "search"
region around where the target was last seen and ask the network where in that
region the exemplar is. Everything that turns that into a point tracker is the
same for both and lives here; each plugin supplies only its network
(:meth:`~stable.trackers.deep.ModelTrackerPlugin._build_backend`) and how to
read the network's output (:meth:`SiameseTrackerPlugin._decode`).

From a box to a point
---------------------
* **The pattern is the box.** A point's ``template_radius`` — the solid box the
  viewer draws — is the target: ``2r + 1`` pixels square, centred on the point.
* **The template is cut once, at the anchor, and never updated.** That is how
  both trackers were trained and benchmarked, and it is what multi-anchor
  merging already exploits: every run starts from an operator-placed anchor, so
  a long shot is never tracked from a template older than its nearest anchor.
* **The point rides on what the network tracks.** Both networks were trained on
  *objects*, and handed an arbitrary patch they find the object they think it
  belongs to: Ocean regresses a box hugging nearby structure (a collar, the
  edge of a face), centred a few pixels from the patch. Reporting that centre
  would quietly move the operator's point — the same trap
  :mod:`stable.trackers.sam3` avoids with its masks — so each run first asks the
  network about the **anchor frame itself**. Its answer there (where it puts the
  target, how big it thinks it is, how strongly it matches) is the reference,
  and every later frame is read *relative* to it: the point moves as that answer
  moves, and scales as it scales. On shifted natural images this took Ocean's
  median error from 3.7 px to 0.9 px; SiamDW, whose correlation peak is
  unbiased, is unaffected.
* **Scale adapts, only the position is reported.** Ocean regresses the box's
  size, SiamDW searches three scales; either way the search region keeps the
  target at the size the template saw it, so a feature the camera pushes in on
  is still matched like for like.

Box tracks
----------
A query that states a box (``Query.w``, ``Query.h`` — an object the operator
drew around) is tracked the way these networks were trained and benchmarked:
the exemplar is cut around that box rather than the ``2r + 1`` square, the
search is the trained 255 px with nothing but the cosine window limiting how far
it may move, and the box's two sides are tracked separately and reported as
:attr:`~stable.core.types.TrackResult.sizes`. The position is still the box's
centre, read relative to the anchor's own reading like a point's, so a network
that boxes an object a little off where the operator drew it does not move it.
SiamDW searches scales, not shapes, so its box keeps the aspect it was drawn
with; Ocean regresses each side and follows an object that changes shape.
Points and boxes can share a call: they are batched separately, because their
searches are cut at different sides.

Geometry
--------
The exemplar is the box plus context — SiamFC's rule, side
``sqrt((w + p)(h + p))`` with ``p = (w + h) / 2`` — resampled to 127 px. The
search region is the same scale, ``X`` px across (255 in training), so the
network sees the target at the size it learned it. ``search_radius`` (the
dashed box, "how far the point may move between frames") is honoured exactly:

* the search region grows past the trained 255 px — in steps that keep the
  score map's centre cell — until the score map reaches that far, up to
  :attr:`SiameseTrackerPlugin.max_instance_size`;
* candidates beyond it are masked out, so a distractor further away than the
  operator said the point can move is never chosen.

Both are fixed in network pixels at the anchor, so the search region scales
with the target afterwards, as the networks expect. Crops are an exact affine
resample — sub-pixel centred, never rounded to the pixel grid as TracKit's are —
padded with the anchor frame's mean colour where they leave the picture, so a
point near the edge is tracked rather than refused.

Losing the target
-----------------
Each frame's evidence is calibrated onto the shared confidence scale (see
:mod:`stable.trackers.confidence`). At or below ``lost_below`` the target is
treated as lost: the point holds where it was last seen, its scale is not
touched, and the next frame searches around the same spot, which is how a point
survives an occluder passing over it. The networks keep answering while the
target is hidden, so "lost" is a decision made here, not something they report —
and it is a separate threshold from confidence ``0``, because a frame too weak
to trust can still be the best guess of where a fast feature went. A box has
its own threshold, ``box_lost_below``: the template is fixed, so an object that
turns or stretches scores lower and lower against it while still being the best
match in sight.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.device import torch_module
from ..core.errors import ContractError, TrackerError
from ..core.types import Query, TrackResult
from ..core.video import VideoSource
from .base import Direction, ProgressCallback, noop_progress
from .confidence import normalize_scores, visibility_from_confidence
from .deep import ModelTrackerPlugin
from .progress import RunProgress, planned_steps

__all__ = [
    "EXEMPLAR_SIZE",
    "CONTEXT_AMOUNT",
    "TOTAL_STRIDE",
    "SiameseMaps",
    "SiameseStep",
    "SiameseBackend",
    "TorchSiameseBackend",
    "SiameseTrackerPlugin",
    "context_side",
    "crop_square",
    "cosine_window",
    "sample_map",
]

logger = logging.getLogger(__name__)

EXEMPLAR_SIZE = 127
"""Side, in network pixels, of the exemplar both networks were trained on."""

CONTEXT_AMOUNT = 0.5
"""Context added around the target box before the exemplar is cut (SiamFC's rule)."""

TOTAL_STRIDE = 8
"""Network pixels per score-map cell, for both networks."""


def _cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Raises:
        TrackerError: If OpenCV is not installed.
    """
    try:
        import cv2  # noqa: PLC0415 - optional, resolved per call
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise TrackerError(
            "the Siamese trackers need opencv-python for their crops; install the 'video' "
            "extra (`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


def context_side(width: float, height: float, amount: float = CONTEXT_AMOUNT) -> float:
    """Return the side of the square exemplar region around a ``width x height`` box.

    SiamFC's rule: grow each side by ``amount * (width + height)`` and take the
    geometric mean, so the exemplar has the same area whatever the box's aspect
    ratio. With the default amount a square box occupies exactly half of it.

    Args:
        width: Box width.
        height: Box height.
        amount: Context fraction; both networks were trained with ``0.5``.

    Returns:
        The exemplar side, in the box's units.
    """
    pad = amount * (width + height)
    return math.sqrt((width + pad) * (height + pad))


def crop_square(
    frame: np.ndarray,
    centre: tuple[float, float],
    side: float,
    size: int,
    fill: np.ndarray,
) -> np.ndarray:
    """Resample the ``side``-pixel square centred on ``centre`` to ``size x size``.

    Exact to the sub-pixel: output pixel ``u`` samples source coordinate
    ``centre + (u - (size - 1) / 2) * side / size``, with pixel centres on
    integers as everywhere in Stable. TracKit instead rounds the square's corner
    to the pixel grid and then assumes it did not, which is up to half a source
    pixel of error on every frame — invisible on a box benchmark, not on a point.

    Args:
        frame: ``[H, W, 3]`` uint8 RGB.
        centre: ``(x, y)`` the crop is centred on, in source pixels.
        side: Side of the cropped square, in source pixels.
        size: Side of the output, in pixels.
        fill: Colour for the parts of the square outside the frame.

    Returns:
        ``[size, size, 3]`` uint8 RGB.
    """
    cv2 = _cv2()
    scale = size / max(float(side), 1e-6)
    offset = (size - 1) / 2.0
    matrix = np.array(
        [[scale, 0.0, offset - scale * centre[0]], [0.0, scale, offset - scale * centre[1]]]
    )
    return cv2.warpAffine(
        np.ascontiguousarray(frame),
        matrix,
        (int(size), int(size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(float(v) for v in fill),
    )


def _calibration(name: str, value: tuple[float, float]) -> tuple[float, float]:
    """Return a ``(low, high)`` evidence calibration as floats.

    Raises:
        ContractError: Unless ``high > low``.
    """
    low, high = float(value[0]), float(value[1])
    if not high > low:
        raise ContractError(f"{name} needs high > low, got {value}")
    return (low, high)


def cosine_window(positions: np.ndarray, centre: float, half_span: float) -> np.ndarray:
    """Return a raised-cosine prior over ``positions``, peaking at ``centre``.

    ``0.5 + 0.5 cos(pi d / half_span)`` for a distance ``d`` within
    ``half_span``, and ``0`` beyond. Centred on the map, with ``half_span`` the
    distance to its outermost cell, this is exactly ``np.hanning`` — the window
    both trackers use; it is written out so it can be centred where the target
    is *expected*, which is the reference reading rather than the crop centre.

    Args:
        positions: Cell positions, in network pixels.
        centre: Where the window peaks.
        half_span: Distance at which it reaches zero.

    Returns:
        Window values in ``[0, 1]``, one per position.
    """
    distance = np.clip(np.abs(np.asarray(positions, dtype=np.float64) - centre) / half_span, 0, 1)
    return 0.5 + 0.5 * np.cos(np.pi * distance)


def sample_map(values: np.ndarray, x: float, y: float) -> float:
    """Bilinearly sample a 2-D map at fractional cell coordinates, clamped inside it.

    Args:
        values: ``[H, W]`` map.
        x: Column, in cells.
        y: Row, in cells.

    Returns:
        The interpolated value.
    """
    height, width = values.shape
    x = min(max(float(x), 0.0), width - 1.0)
    y = min(max(float(y), 0.0), height - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    top = values[y0, x0] * (1.0 - fx) + values[y0, x1] * fx
    bottom = values[y1, x0] * (1.0 - fx) + values[y1, x1] * fx
    return float(top * (1.0 - fy) + bottom * fy)


@dataclass(frozen=True)
class SiameseMaps:
    """One batch of raw network output, as NumPy.

    Deliberately raw, like :class:`~stable.trackers.deep.BackendPrediction`:
    what the numbers mean is the plugin's business, not the backend's.

    Args:
        score: ``[N, S, M, M]`` one score map per point and search scale —
            classification logits for Ocean, correlation responses for SiamDW.
        boxes: ``[N, S, 4, M, M]`` per-cell distances from the cell to the
            target box's left, top, right and bottom edges, in network pixels,
            for networks that regress a box; else ``None``.
        similarity: ``[N, S, M, M]`` appearance agreement between the template
            and the search under each cell, for networks whose ``score`` is not
            itself one; else ``None``.

    Raises:
        ContractError: On a shape mismatch between the arrays.
    """

    score: np.ndarray
    boxes: np.ndarray | None = None
    similarity: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce dtypes and check the shapes agree (dataclass hook)."""
        score = np.asarray(self.score, dtype=np.float32)
        if score.ndim != 4 or score.shape[2] != score.shape[3]:
            raise ContractError(f"score maps must be [N, S, M, M], got {score.shape}")
        boxes = self.boxes
        if boxes is not None:
            boxes = np.asarray(boxes, dtype=np.float32)
            expected = (*score.shape[:2], 4, *score.shape[2:])
            if boxes.shape != expected:
                raise ContractError(f"box maps must be {expected}, got {boxes.shape}")
        similarity = self.similarity
        if similarity is not None:
            similarity = np.asarray(similarity, dtype=np.float32)
            if similarity.shape != score.shape:
                raise ContractError(
                    f"similarity maps must be {score.shape}, got {similarity.shape}"
                )
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "similarity", similarity)

    @property
    def n_points(self) -> int:
        """Points ``N`` in the batch."""
        return int(self.score.shape[0])

    @property
    def n_scales(self) -> int:
        """Search scales ``S`` per point."""
        return int(self.score.shape[1])


@dataclass(frozen=True)
class SiameseStep:
    """What one point's search found, in the reference search crop's pixels.

    "Reference" means the crop cut at scale ``1`` of the run's search scales;
    a decoder that searched several converts back to it, so the plugin alone
    turns the answer into source pixels.

    Attributes:
        dx: Horizontal position of what the network tracked, relative to the
            crop centre.
        dy: Vertical position, likewise.
        width: Width of what the network tracked.
        height: Height, likewise.
        rate: How far the tracked size moves toward ``(width, height)`` this
            frame, in ``[0, 1]`` — the trackers' own damping.
        evidence: Raw, uncalibrated evidence that this is the target; see
            :meth:`SiameseTrackerPlugin.raw_confidence`.
    """

    dx: float
    dy: float
    width: float
    height: float
    rate: float
    evidence: float


@runtime_checkable
class SiameseBackend(Protocol):
    """The model seam for template/search networks: NumPy crops in, maps out.

    Implementations own a loaded network. They are created by
    :meth:`~stable.trackers.deep.ModelTrackerPlugin._build_backend`, or injected
    into a plugin's constructor, which is how the pipeline is tested without
    torch or weights (``tests/conftest.py``'s ``SiameseMarkerBackend``).
    """

    device: str

    def embed(self, exemplars: np.ndarray) -> Any:
        """Turn exemplar crops into templates.

        Args:
            exemplars: ``[N, 127, 127, 3]`` uint8 RGB, one per point.

        Returns:
            An opaque batch of templates for :meth:`match`, valid until the
            backend changes device.
        """
        ...

    def match(self, kernels: Any, searches: np.ndarray) -> SiameseMaps:
        """Score each point's search crops against its template.

        Args:
            kernels: What :meth:`embed` returned for these points.
            searches: ``[N, S, X, X, 3]`` uint8 RGB, ``S`` scales per point.

        Returns:
            The batch's :class:`SiameseMaps`.
        """
        ...

    def to_device(self, device: str) -> None:
        """Move the loaded model onto ``device`` (used by the CPU fallback)."""
        ...

    def release(self) -> None:
        """Free the model and any device memory. Must be safe to call twice."""
        ...


def _torch() -> Any:
    """Return :mod:`torch`, or raise the error the plugins report."""
    torch = torch_module()
    if torch is None:  # pragma: no cover - guarded by is_available()
        raise TrackerError("the Siamese trackers need PyTorch; install the 'torch' extra")
    return torch


def _numpy(value: Any) -> np.ndarray:
    """Return a tensor as a detached float32 CPU array."""
    return value.detach().float().cpu().numpy()


@contextmanager
def _denormals_flushed(torch: Any, device: str) -> Iterator[None]:
    """Flush denormal floats to zero for one CPU network call, then turn it back off.

    The SiamDW release carries 715,493 denormal values (below ``1.2e-38``) across
    116 of its tensors, and more arise in its activations. x86 CPUs handle those
    in microcode, which made one frame of SiamDW on the CPU take 2.8 s instead
    of 36 ms. Flushing them changes nothing that could matter numerically.

    PyTorch has no getter for the mode, so this restores the default (off) —
    the only mode Stable otherwise runs in. GPUs need none of this.
    """
    if not str(device).startswith("cpu"):
        yield
        return
    enabled = bool(torch.set_flush_denormal(True))
    try:
        yield
    finally:
        if enabled:
            torch.set_flush_denormal(False)


class TorchSiameseBackend:
    """Runs one of :mod:`stable.trackers.siamese_nets`' networks on NumPy crops.

    Owns the networks' input convention so nothing else has to: TracKit feeds
    ``cv2.imread`` output straight in, so crops go in **BGR**, as floats in
    ``[0, 255]``, with no normalisation. On the CPU every call runs with
    denormal floats flushed to zero (see :func:`_denormals_flushed`).

    Args:
        network: An ``OceanNet`` or ``SiamDWNet`` with weights loaded, already
            on ``device`` and in evaluation mode.
        device: The device string the network lives on.

    Attributes:
        device: Where the network currently is; updated by :meth:`to_device`.
    """

    def __init__(self, network: Any, device: str) -> None:
        """Wrap a loaded network."""
        self._network = network
        self.device = device

    def _tensor(self, crops: np.ndarray) -> Any:
        """Return ``[B, H, W, 3]`` uint8 RGB as a ``[B, 3, H, W]`` float BGR tensor."""
        torch = _torch()
        tensor = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device)
        return tensor.flip(-1).permute(0, 3, 1, 2).float().contiguous()

    def embed(self, exemplars: np.ndarray) -> Any:
        """Return the network's template features for ``exemplars``."""
        torch = _torch()
        with torch.inference_mode(), _denormals_flushed(torch, self.device):
            return self._network.embed(self._tensor(exemplars))

    def match(self, kernels: Any, searches: np.ndarray) -> SiameseMaps:
        """Score ``searches`` (``[N, S, X, X, 3]``) against ``kernels``."""
        torch = _torch()
        count, scales = int(searches.shape[0]), int(searches.shape[1])
        flat = searches.reshape(count * scales, *searches.shape[2:])
        with torch.inference_mode(), _denormals_flushed(torch, self.device):
            batch = self._tensor(flat)
            score, boxes, similarity = self._network.match(
                kernels, batch.reshape(count, scales, *batch.shape[1:])
            )
            return SiameseMaps(
                score=_numpy(score),
                boxes=None if boxes is None else _numpy(boxes),
                similarity=None if similarity is None else _numpy(similarity),
            )

    def to_device(self, device: str) -> None:
        """Move the network to ``device``."""
        self._network.to(device)
        self.device = device

    def release(self) -> None:
        """Drop the reference to the network so its memory can be reclaimed."""
        self._network = None


@dataclass(frozen=True)
class _Seed:
    """One query as its search sees it, fixed at the anchor.

    Attributes:
        x: Anchor position (the box centre), in source pixels.
        y: Likewise.
        width: The target box at the anchor, in source pixels: the query's own
            box, or the ``2r + 1`` square of a point.
        height: Likewise.
        box: Whether the query stated its box. A box's two sides are tracked
            and reported; a point's scale stays isotropic and is not reported.
        instance_size: Search crop side, in network pixels.
        reach: How far a candidate may be from where the target is expected,
            in network pixels.
    """

    x: float
    y: float
    width: float
    height: float
    box: bool
    instance_size: int
    reach: float

    @property
    def anchor_scale(self) -> float:
        """Network pixels per source pixel in the crops cut at the anchor."""
        return EXEMPLAR_SIZE / context_side(self.width, self.height)


@dataclass
class _Anchor:
    """What one batch of points learned from its anchor frame.

    Attributes:
        exemplars: The exemplar crops, kept so the templates can be rebuilt if
            the model moves device mid-run.
        kernels: The backend's templates for them.
        references: Each point's self-match — the network's reading of the
            anchor frame — which every later frame is read relative to.
        seeds: The batch's queries.
        fill: Padding colour for crops leaving the frame.
    """

    exemplars: np.ndarray
    kernels: Any
    references: list[SiameseStep]
    seeds: list[_Seed]
    fill: np.ndarray


@dataclass
class _Target:
    """One point between frames: where it is and how much its target has scaled.

    ``scale_x`` and ``scale_y`` are the target box's sides against the anchor's;
    they differ only for a box query, whose aspect is tracked.
    """

    x: float
    y: float
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class _Sink:
    """Where one ``track()`` call's results go, and who hears about progress."""

    points: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray
    sizes: np.ndarray | None
    reporter: RunProgress


class SiameseTrackerPlugin(ModelTrackerPlugin[SiameseBackend]):
    """Base class for Siamese single-object trackers driven as point trackers.

    Subclasses set the class attributes below and implement
    :meth:`~stable.trackers.deep.ModelTrackerPlugin._build_backend`,
    :meth:`score_size` and :meth:`_decode`; see the module docstring for what
    this class does around them.

    Args:
        template_radius: Half-size of the target box, in source pixels; the box
            is ``2r + 1`` square and centred on the point. It wants to cover the
            feature: these networks match the box *and* half as much context
            again around it.
        search_radius: How far the point may move between frames, in source
            pixels at the anchor's scale. See the module docstring.
        window_influence: Weight of the cosine prior favouring small motion, in
            ``[0, 1)``. Higher is steadier and slower to follow a fast feature.
        confidence_range: ``(low, high)`` calibration of the raw evidence (see
            :meth:`raw_confidence`) onto the shared confidence scale.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        lost_below: Raw evidence at or below which the target is treated as
            lost and the point held; ``None`` means the bottom of
            ``confidence_range``. It can sit lower than that: a frame can be too
            weak to *trust* (confidence 0, so the solver ignores it) and still
            be the network's best guess of where a fast feature went. Holding on
            every weak frame is how a point falls behind a fast pan and never
            catches up.
        box_confidence_range: ``confidence_range`` for a query with a box;
            ``None`` means the same. An object the operator drew changes
            appearance as it turns or deforms, and a fixed template's evidence
            falls with it while the tracker is still on it, so a box wants a
            calibration of its own (see each plugin's module docstring).
        box_lost_below: ``lost_below`` for a query with a box; ``None`` means
            the bottom of ``box_confidence_range``. That default is the rule
            for boxes: a frame the tracker trusts enough to move the box on is a
            frame the solver should be allowed to use, and one it does not is
            one the solver should not.
        device: Preferred device; see
            :class:`~stable.trackers.deep.ModelTrackerPlugin`.
        allow_cpu_fallback: See :class:`~stable.trackers.deep.ModelTrackerPlugin`.
        backend: A ready :class:`SiameseBackend`, for tests and other runtimes.
        allow_download: See :class:`~stable.trackers.deep.ModelTrackerPlugin`.

    Raises:
        ContractError: If a size or calibration argument is out of range.

    Attributes:
        exemplar_size: Exemplar side, in network pixels.
        total_stride: Network pixels per score-map cell.
        trained_instance_size: The search side the weights were trained at. A
            search is never cut smaller: below it the network sees less context
            than it learned to use.
        max_instance_size: The largest search side, bounding the per-frame cost
            of a large ``search_radius`` (it grows with its square). A radius the
            score map cannot reach even at this size is honoured only as far
            as it can be.
        search_scales: Relative sizes of the search crops cut per frame; the
            first ``1.0`` is the reference. ``(1.0,)`` cuts one.
        max_batch: Points matched together in one network call.
        scale_bounds: How far the target may shrink or grow from its size at
            the anchor before it is clamped.
        scale_gate: A reading whose box differs from the expected one by more
            than this factor — scale and aspect change multiplied, as in
            TracKit's penalty — still moves the point but is not trusted to
            rescale it. A point's scale is isotropic: it reports no box, so
            tracking aspect would only feed back noise. Without either, Ocean's
            box regression of an arbitrary patch (which flips between shapes
            from frame to frame) walked the scale to its lower clamp within 60
            frames of handheld footage; see :mod:`stable.trackers.ocean`. A box
            query's sides are tracked separately, still behind the gate.
        tracks_boxes: Always ``True``: a query with a box is tracked as that
            object, and its size reported (see the module docstring).
    """

    tracks_boxes = True
    exemplar_size: int = EXEMPLAR_SIZE
    total_stride: int = TOTAL_STRIDE
    trained_instance_size: int = 255
    max_instance_size: int = 511
    search_scales: tuple[float, ...] = (1.0,)
    max_batch: int = 16
    scale_bounds: tuple[float, float] = (0.2, 5.0)
    scale_gate: float = 1.3
    implementation_requirement = "OpenCV for its template and search crops"
    install_hint = "Install it with: pip install stable-tracker[video]"

    def __init__(
        self,
        *,
        template_radius: int = 16,
        search_radius: int = 32,
        window_influence: float = 0.3,
        confidence_range: tuple[float, float] = (0.2, 0.8),
        visible_threshold: float = 0.5,
        lost_below: float | None = None,
        box_confidence_range: tuple[float, float] | None = None,
        box_lost_below: float | None = None,
        device: str | None = None,
        allow_cpu_fallback: bool = True,
        backend: SiameseBackend | None = None,
        allow_download: bool = True,
    ) -> None:
        """Create an unprepared Siamese tracker; see the class docstring."""
        super().__init__(
            device=device,
            allow_cpu_fallback=allow_cpu_fallback,
            backend=backend,
            allow_download=allow_download,
        )
        self.template_radius = int(template_radius)
        self.search_radius = int(search_radius)
        if self.template_radius < 1:
            raise ContractError(f"template_radius must be >= 1, got {self.template_radius}")
        if self.search_radius < 0:
            raise ContractError(f"search_radius must be >= 0, got {self.search_radius}")
        self.window_influence = float(window_influence)
        if not 0.0 <= self.window_influence < 1.0:
            raise ContractError(f"window_influence must lie in [0, 1), got {self.window_influence}")
        self.confidence_range = _calibration("confidence_range", confidence_range)
        self.box_confidence_range = (
            self.confidence_range
            if box_confidence_range is None
            else _calibration("box_confidence_range", box_confidence_range)
        )
        self.visible_threshold = float(visible_threshold)
        self.lost_below = self.confidence_range[0] if lost_below is None else float(lost_below)
        self.box_lost_below = (
            self.box_confidence_range[0] if box_lost_below is None else float(box_lost_below)
        )

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether OpenCV, which cuts the crops, is importable.

        The network code itself ships with Stable
        (:mod:`stable.trackers.siamese_nets`), so this is the one prerequisite
        beyond PyTorch.
        """
        try:
            import cv2  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    # -- what a family supplies --------------------------------------------

    @abstractmethod
    def score_size(self, instance_size: int) -> int:
        """Return the side of the score map a search crop of ``instance_size`` yields.

        Args:
            instance_size: Search crop side, in network pixels.

        Returns:
            Score cells per side.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Read one point's maps: where is the target, how big, how sure?

        Everything is in the reference search crop's pixels (see
        :class:`SiameseStep`).

        Args:
            maps: The batch's raw network output.
            slot: This point's index in the batch.
            expected_box: How big the reference reading's box would be in this
                crop if nothing had changed.
            expected_centre: Where it would be, relative to the crop centre.
                Priors (the cosine window) centre here, not on the crop.
            reach: How far from ``expected_centre`` a candidate may be;
                ``math.inf`` for the anchor's own reading.
            box: Whether the query is a box the operator drew, rather than a
                point's patch; a decoder may damp its size differently for one.

        Returns:
            What the search found.
        """
        raise NotImplementedError

    def raw_confidence(self, step: SiameseStep, reference: SiameseStep) -> float:
        """Return the evidence that ``step`` found the target, before calibration.

        The default trusts the decoder's own evidence. A network whose raw
        scores mean different things for different templates overrides this to
        read them against ``reference``, the anchor frame's own reading.

        Args:
            step: This frame's reading.
            reference: The anchor frame's reading of the same template.

        Returns:
            A score that :meth:`normalize_confidence` maps through
            ``confidence_range``.
        """
        return float(step.evidence)

    def normalize_confidence(
        self, raw: np.ndarray, *, box: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map raw evidence onto the shared ``[0, 1]`` confidence.

        Args:
            raw: Scores from :meth:`raw_confidence`.
            box: Calibrate with ``box_confidence_range``, for box queries.

        Returns:
            ``(confidence float32, visible bool)``, each shaped like ``raw``.
        """
        low, high = self.box_confidence_range if box else self.confidence_range
        confidence = normalize_scores(raw, low=low, high=high)
        return confidence, visibility_from_confidence(confidence, self.visible_threshold)

    # -- geometry ---------------------------------------------------------------

    @property
    def box_side(self) -> int:
        """Side of the target box at the anchor, in source pixels."""
        return 2 * self.template_radius + 1

    @property
    def anchor_scale(self) -> float:
        """Network pixels per source pixel in the crops cut at an anchor."""
        return self.exemplar_size / context_side(self.box_side, self.box_side)

    def search_geometry(self) -> tuple[int, float]:
        """Return the search crop side and the reach ``search_radius`` asks for.

        Returns:
            ``(instance_size, reach)``: the crop side in network pixels — the
            trained size, grown in steps that keep a centre cell until the score
            map spans ``reach`` or :attr:`max_instance_size` is met — and
            ``reach``, ``search_radius`` in network pixels at the anchor.
        """
        scale = self.anchor_scale
        reach = self.search_radius * scale
        size = self.trained_instance_size
        step = 2 * self.total_stride
        while self._reach_of(size) < reach and size + step <= self.max_instance_size:
            size += step
        if self._reach_of(size) < reach:
            logger.info(
                "%s: search_radius %d px needs a larger search than %d px; it reaches %.0f px",
                self.name,
                self.search_radius,
                self.max_instance_size,
                self._reach_of(size) / scale,
            )
        return size, reach

    def _reach_of(self, instance_size: int) -> float:
        """How far, in network pixels, the outermost score cell is from the centre."""
        return (self.score_size(instance_size) - 1) / 2.0 * self.total_stride

    # -- the contract -------------------------------------------------------------

    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Track every query point through ``frame_range``. One run, no merging.

        Points sharing an anchor frame are tracked together, one network call
        per frame for all of them (up to :attr:`max_batch` at a time).

        Args:
            video: Clip to read frames from.
            queries: Anchor positions, one per point; each must lie inside
                ``frame_range``. A position near or past the frame edge is fine:
                the crops are padded. A query with a box is tracked as a box.
            frame_range: ``(start, end)`` half-open absolute frame range.
            direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
            progress: Progress reporter, called once per frame stepped.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``, in query order; with ``sizes`` if any
            query has a box (``NaN`` rows for the points among them).

        Raises:
            TrackerError: If the range or direction is invalid, a query lies
                outside the range, OpenCV is missing, or the model fails
                (including running out of memory with no fallback left).
        """
        n_frames = int(video.frame_count)
        start, end = int(frame_range[0]), int(frame_range[1])
        if direction not in ("fwd", "bwd", "bidir"):
            raise TrackerError(f"unknown direction {direction!r}")
        if end <= start:
            raise TrackerError(f"frame_range must be non-empty, got [{start}, {end})")
        if start < 0 or end > n_frames:
            raise TrackerError(f"frame_range [{start}, {end}) is outside the clip (T={n_frames})")
        if not queries:
            return TrackResult.empty(0, n_frames)
        for query in queries:
            if not start <= query.frame < end:
                raise TrackerError(
                    f"query frame {query.frame} must lie inside frame_range [{start}, {end})"
                )
        _cv2()  # before the weights load: a missing OpenCV should not cost a download
        if not self.is_prepared:
            self.prepare(self.requested_device or "auto")
        if self._backend is None:  # pragma: no cover - prepare() guarantees this
            raise TrackerError(f"{self.name} has no loaded model; call prepare() first")

        point_geometry = self.search_geometry()
        seeds = [self._seed(query, point_geometry) for query in queries]
        boxes = any(seed.box for seed in seeds)
        sink = _Sink(
            points=np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32),
            confidence=np.zeros((len(queries), n_frames), dtype=np.float32),
            visible=np.zeros((len(queries), n_frames), dtype=bool),
            sizes=np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32) if boxes else None,
            reporter=RunProgress(progress, total=planned_steps(queries, (start, end), direction)),
        )

        # Crops are stacked into one network call, so a batch shares its frame
        # and its search side; boxes and points search at different sides.
        groups: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        for index, (query, seed) in enumerate(zip(queries, seeds, strict=True)):
            groups.setdefault((query.frame, seed.instance_size), []).append(index)

        for (anchor_frame, _), members in groups.items():
            frame = self._read(video, anchor_frame)
            fill = frame.reshape(-1, 3).mean(axis=0)
            runs: list[range] = []
            if direction in ("fwd", "bidir") and end - anchor_frame > 1:
                runs.append(range(anchor_frame + 1, end))
            if direction in ("bwd", "bidir") and anchor_frame > start:
                runs.append(range(anchor_frame - 1, start - 1, -1))
            for first in range(0, len(members), self.max_batch):
                batch = members[first : first + self.max_batch]
                anchor = self._read_anchor(frame, [seeds[i] for i in batch], fill)
                for frames in runs:
                    self._follow(video, anchor, batch, frames, sink)

        # The anchor is ground truth: never let a model's own reading of its
        # query frame override the position (or box) the operator placed there.
        for index, query in enumerate(queries):
            sink.points[index, query.frame] = query.position
            sink.confidence[index, query.frame] = 1.0
            sink.visible[index, query.frame] = True
            if sink.sizes is not None and query.size is not None:
                sink.sizes[index, query.frame] = query.size

        progress(1.0, "done")
        return TrackResult(
            points=sink.points, confidence=sink.confidence, visible=sink.visible, sizes=sink.sizes
        )

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _read(video: VideoSource, index: int) -> np.ndarray:
        """Return frame ``index``, without copying it where the source allows.

        Raises:
            TrackerError: If the source does not hand back ``[H, W, 3]`` RGB.
        """
        view = getattr(video, "get_frame_view", None)
        frame = np.asarray(view(index) if callable(view) else video.get_frame(index))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise TrackerError(f"video frames must be [H, W, 3] RGB, got {frame.shape}")
        return frame

    def _seed(self, query: Query, point_geometry: tuple[int, float]) -> _Seed:
        """Return how ``query`` is searched for.

        A point is the ``2r + 1`` box searched as ``search_radius`` asks (see
        :meth:`search_geometry`). A box is searched the way the networks were
        trained and benchmarked: the trained search side, around the box the
        operator drew, with nothing beyond the cosine window to limit how far it
        may move — ``search_radius`` describes a point's patch, not an object.
        """
        if query.size is None:
            side = float(self.box_side)
            instance_size, reach = point_geometry
            return _Seed(query.x, query.y, side, side, False, instance_size, reach)
        width, height = query.size
        return _Seed(query.x, query.y, width, height, True, self.trained_instance_size, math.inf)

    def _search_side(self, target: _Target, seed: _Seed) -> float:
        """Return the reference search crop's side, in source pixels, at ``target``'s scale."""
        side = context_side(seed.width * target.scale_x, seed.height * target.scale_y)
        return side * seed.instance_size / self.exemplar_size

    def _searches(self, frame: np.ndarray, targets: list[_Target], anchor: _Anchor) -> np.ndarray:
        """Cut every target's search crops: ``[N, S, X, X, 3]``."""
        return np.stack(
            [
                np.stack(
                    [
                        crop_square(
                            frame,
                            (t.x, t.y),
                            self._search_side(t, seed) * k,
                            seed.instance_size,
                            anchor.fill,
                        )
                        for k in self.search_scales
                    ]
                )
                for t, seed in zip(targets, anchor.seeds, strict=True)
            ]
        )

    def _read_anchor(self, frame: np.ndarray, seeds: list[_Seed], fill: np.ndarray) -> _Anchor:
        """Cut the templates and take the network's reading of the anchor frame."""
        exemplars = np.stack(
            [
                crop_square(
                    frame,
                    (seed.x, seed.y),
                    context_side(seed.width, seed.height),
                    self.exemplar_size,
                    fill,
                )
                for seed in seeds
            ]
        )
        anchor = _Anchor(
            exemplars=exemplars,
            kernels=self._embed(exemplars),
            references=[],
            seeds=list(seeds),
            fill=fill,
        )
        targets = [_Target(seed.x, seed.y) for seed in seeds]
        maps = self._match(anchor, self._searches(frame, targets, anchor))
        anchor.references = [
            self._decode(
                maps,
                slot,
                (seed.width * seed.anchor_scale, seed.height * seed.anchor_scale),
                (0.0, 0.0),
                math.inf,
                box=seed.box,
            )
            for slot, seed in enumerate(seeds)
        ]
        return anchor

    def _follow(
        self,
        video: VideoSource,
        anchor: _Anchor,
        members: list[int],
        frames: range,
        sink: _Sink,
    ) -> None:
        """Track one batch through ``frames``, in order, from its anchor."""
        targets = [_Target(seed.x, seed.y) for seed in anchor.seeds]
        for index in frames:
            frame = self._read(video, index)
            maps = self._match(anchor, self._searches(frame, targets, anchor))
            scales = [
                seed.instance_size / self._search_side(t, seed)
                for t, seed in zip(targets, anchor.seeds, strict=True)
            ]
            steps = [
                self._decode(
                    maps,
                    slot,
                    self._expected(reference, target, scale, seed, box=True),
                    self._expected(reference, target, scale, seed, box=False),
                    seed.reach,
                    box=seed.box,
                )
                for slot, (target, reference, scale, seed) in enumerate(
                    zip(targets, anchor.references, scales, anchor.seeds, strict=True)
                )
            ]
            raw = np.array(
                [self.raw_confidence(s, r) for s, r in zip(steps, anchor.references, strict=True)],
                dtype=np.float64,
            )
            scores, seen = self.normalize_confidence(raw)
            boxed = np.array([seed.box for seed in anchor.seeds])
            if boxed.any():
                box_scores, box_seen = self.normalize_confidence(raw, box=True)
                scores = np.where(boxed, box_scores, scores)
                seen = np.where(boxed, box_seen, seen)
            height, width = frame.shape[:2]
            for slot, point in enumerate(members):
                target, seed = targets[slot], anchor.seeds[slot]
                if raw[slot] > (self.box_lost_below if seed.box else self.lost_below):
                    self._advance(target, steps[slot], anchor.references[slot], scales[slot], seed)
                    target.x = min(max(target.x, 0.0), width - 1.0)
                    target.y = min(max(target.y, 0.0), height - 1.0)
                sink.points[point, index] = (target.x, target.y)
                sink.confidence[point, index] = scores[slot]
                sink.visible[point, index] = bool(seen[slot])
                if seed.box and sink.sizes is not None:
                    sink.sizes[point, index] = (
                        seed.width * target.scale_x,
                        seed.height * target.scale_y,
                    )
            sink.reporter.step(len(members))

    @staticmethod
    def _expected(
        reference: SiameseStep,
        target: _Target,
        scale: float,
        seed: _Seed,
        *,
        box: bool,
    ) -> tuple[float, float]:
        """Where (``box=False``) or how big (``box=True``) the reference reading is now.

        The anchor's own reading, in this frame's search crop: grown with the
        target's scale and converted to this crop's pixels. With no motion and no
        scale change, this is exactly what the network should answer again.
        """
        factor_x = target.scale_x * scale / seed.anchor_scale
        factor_y = target.scale_y * scale / seed.anchor_scale
        if box:
            return (reference.width * factor_x, reference.height * factor_y)
        return (reference.dx * factor_x, reference.dy * factor_y)

    def _advance(
        self,
        target: _Target,
        step: SiameseStep,
        reference: SiameseStep,
        scale: float,
        seed: _Seed,
    ) -> None:
        """Move ``target`` by what ``step`` says changed since the reference reading.

        The tracked thing's size, against the reference reading's at the target's
        current scale, updates that scale (damped by ``step.rate``, and not at
        all past :attr:`scale_gate`) — each side on its own for a box, both by
        their geometric mean for a point; the tracked thing's centre is converted
        to source pixels; and the point is put back at the reference offset from
        that centre, grown or shrunk with the target — the "riding" the module
        docstring describes.
        """
        expected_w, expected_h = self._expected(reference, target, scale, seed, box=True)
        expected_w, expected_h = max(expected_w, 1e-6), max(expected_h, 1e-6)
        width, height = max(step.width, 1e-6), max(step.height, 1e-6)
        along_x, along_y = width / expected_w, height / expected_h
        measured = math.sqrt(along_x * along_y)
        aspect = along_x / along_y
        change = max(measured, 1.0 / measured) * max(aspect, 1.0 / aspect)
        rate = min(max(float(step.rate), 0.0), 1.0) if change <= self.scale_gate else 0.0
        if not seed.box:
            along_x = along_y = measured
        low, high = self.scale_bounds
        target.scale_x = min(max(target.scale_x * (1.0 - rate + rate * along_x), low), high)
        target.scale_y = min(max(target.scale_y * (1.0 - rate + rate * along_y), low), high)
        centre_x = target.x + step.dx / scale
        centre_y = target.y + step.dy / scale
        target.x = centre_x - reference.dx * target.scale_x / seed.anchor_scale
        target.y = centre_y - reference.dy * target.scale_y / seed.anchor_scale

    def _embed(self, exemplars: np.ndarray) -> Any:
        """Build templates, absorbing a GPU out-of-memory failure."""
        backend = self._backend
        assert backend is not None  # noqa: S101 - track() checks this
        return self._call_with_cpu_fallback(
            lambda: backend.embed(exemplars),
            what=f"{len(exemplars)} template(s)",
            advice="Track fewer points at once.",
        )

    def _match(self, anchor: _Anchor, searches: np.ndarray) -> SiameseMaps:
        """Run one frame's searches, rebuilding the templates if the model moves."""
        backend = self._backend
        assert backend is not None  # noqa: S101 - track() checks this

        def rebuild() -> None:
            anchor.kernels = backend.embed(anchor.exemplars)

        maps = self._call_with_cpu_fallback(
            lambda: backend.match(anchor.kernels, searches),
            what=(
                f"{searches.shape[0]} point(s) x {searches.shape[1]} search crop(s) of "
                f"{searches.shape[2]} px"
            ),
            advice="Lower search_radius or track fewer points at once.",
            on_cpu=rebuild,
        )
        if maps.n_points != searches.shape[0] or maps.n_scales != searches.shape[1]:
            raise TrackerError(
                f"{self.name} backend returned maps for {maps.n_points}x{maps.n_scales} "
                f"searches, expected {searches.shape[0]}x{searches.shape[1]}"
            )
        return maps

    def __repr__(self) -> str:
        """Return a debug representation naming the device and the box size."""
        return (
            f"<{type(self).__name__} name={self.name!r} device={self.device!r} "
            f"box={self.box_side}px search={self.search_radius}px>"
        )
