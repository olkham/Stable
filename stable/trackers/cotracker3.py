"""CoTracker3 — Meta's transformer point tracker, as a Stable plugin.

CoTracker3 (Karaev et al., 2024) tracks a set of query points jointly through a
window of frames with a transformer that lets the points support each other,
which is why it holds on through occlusion and texture-less regions far better
than optical flow. It has two heads besides the positions: *visibility* (is the
point occluded?) and *confidence* (is this prediction actually near the truth?),
and Stable multiplies them into the one confidence scale the solver reads —
see :mod:`stable.trackers.confidence`.

Licensing (important)
---------------------
The CoTracker **weights are CC-BY-NC-4.0 — non-commercial**. They are never
bundled and never downloaded silently: fetching them requires an explicit
opt-in (``STABLE_ACCEPT_NONCOMMERCIAL=1`` or ``accept_noncommercial=True``), and
:attr:`CoTracker3Tracker.license` says so, so the UI can label the tracker.
Stable's own code stays MIT. If you need a commercially usable deep tracker,
use :mod:`stable.trackers.tapir` (Apache-2.0).

Getting the model code
----------------------
The checkpoint is only the weights; the architecture lives in Meta's
``cotracker`` package::

    pip install "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"

:meth:`CoTracker3Tracker.is_available` reports False until that (and torch) is
importable, so the UI greys the tracker out instead of failing mid-job.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import TrackerError
from .confidence import combine_scores, normalize_scores, visibility_from_confidence
from .deep import BackendPrediction, DeepTrackerPlugin, TrackerBackend
from .weights import ModelSpec, register_model

__all__ = ["CoTracker3Tracker", "COTRACKER3_OFFLINE"]

logger = logging.getLogger(__name__)

COTRACKER3_OFFLINE = register_model(
    ModelSpec(
        key="cotracker3_offline",
        url="https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth",
        filename="cotracker3_scaled_offline.pth",
        # Pinned from a real download verified against a real inference run (see
        # CONTRIBUTING.md "Pinning a checkpoint"); ~97 MiB, 101890938 bytes.
        sha256="2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
        size_bytes=101890938,
        license="CC-BY-NC-4.0",
        non_commercial=True,
        notes=(
            "CoTracker3 offline (scaled) checkpoint from Meta's official release. "
            "Non-commercial weights; Stable never bundles them."
        ),
    )
)
"""The CoTracker3 offline checkpoint this plugin loads."""


class CoTrackerBackend:
    """Runs a loaded CoTracker3 model on one window of frames.

    Args:
        model: A ``CoTrackerPredictor`` (or anything with the same call
            signature) already on ``device``.
        device: The device string the model currently lives on.

    Attributes:
        device: Where the model currently is; updated by :meth:`to_device`.
    """

    def __init__(self, model: Any, device: str) -> None:
        """Wrap an already-loaded predictor."""
        self._model = model
        self.device = device

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Track ``queries`` through ``frames`` with CoTracker3.

        Args:
            frames: ``[L, H, W, 3]`` uint8 RGB in run order.
            queries: ``[N, 3]`` rows of ``(local_frame, x, y)``.

        Returns:
            The window's :class:`~stable.trackers.deep.BackendPrediction`.

        Raises:
            TrackerError: If the model returns something unrecognisable.
        """
        torch = torch_module()
        if torch is None:  # pragma: no cover - guarded by is_available()
            raise TrackerError("CoTracker3 needs PyTorch; install the 'torch' extra")
        video = (
            torch.from_numpy(np.ascontiguousarray(frames))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self.device)
        )
        query_tensor = torch.from_numpy(np.ascontiguousarray(queries)).float()[None].to(self.device)
        with torch.inference_mode():
            output = self._model(video, queries=query_tensor)
        return _prediction_from_output(output)

    def to_device(self, device: str) -> None:
        """Move the underlying model to ``device``."""
        mover = getattr(self._model, "to", None)
        if mover is not None:
            mover(device)
        self.device = device

    def release(self) -> None:
        """Drop the reference to the model so its memory can be reclaimed."""
        self._model = None


def _prediction_from_output(output: Any) -> BackendPrediction:
    """Normalise CoTracker's return value into a :class:`BackendPrediction`.

    Releases of CoTracker have returned ``(tracks, visibility)`` and
    ``(tracks, visibility, confidence)``, with visibility either a probability
    or an already-thresholded boolean. All of those are accepted here rather
    than pinning Stable to one upstream revision.

    Args:
        output: Whatever the predictor returned.

    Returns:
        The window's prediction, batch dimension removed.

    Raises:
        TrackerError: If the output is not a 2- or 3-tuple of tensors.
    """
    if not isinstance(output, tuple | list) or not 2 <= len(output) <= 3:
        raise TrackerError(
            f"unexpected CoTracker output {type(output).__name__}; expected a 2- or 3-tuple of "
            "(tracks, visibility[, confidence])"
        )
    arrays = [_to_numpy(item) for item in output]
    tracks = arrays[0]
    visibility = arrays[1]
    confidence = arrays[2] if len(arrays) == 3 else None
    if tracks.ndim == 4:  # [B, L, N, 2] -> [L, N, 2]
        tracks = tracks[0]
    if visibility.ndim == 3:  # [B, L, N] -> [L, N]
        visibility = visibility[0]
    if confidence is not None and confidence.ndim == 3:
        confidence = confidence[0]
    return BackendPrediction(
        points=tracks.astype(np.float32),
        visibility=visibility.astype(np.float32),
        confidence=None if confidence is None else confidence.astype(np.float32),
    )


def _to_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a detached CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class CoTracker3Tracker(DeepTrackerPlugin):
    """CoTracker3 as a Stable tracker plugin.

    Args:
        confidence_range: ``(low, high)`` calibration for the combined
            visibility x confidence score — below ``low`` the model is treated
            as carrying no information, above ``high`` as fully reliable.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible.
        **kwargs: Forwarded to :class:`~stable.trackers.deep.DeepTrackerPlugin`
            (``chunk_size``, ``overlap``, ``model_resolution``, ``device``,
            ``backend``, ...).
    """

    name = "cotracker3"
    display_name = "CoTracker3 (Meta)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "CC-BY-NC-4.0"

    model_spec = COTRACKER3_OFFLINE
    install_hint = (
        'Install it with: pip install "cotracker @ '
        'git+https://github.com/facebookresearch/co-tracker.git"'
    )
    # 48 frames at 384x512 is ~1.5 GB of activations for the offline model, which
    # leaves room on an 8 GB card for the weights and a 1080p decode buffer.
    default_chunk_size = 48
    default_overlap = 8
    default_model_resolution = (384, 512)

    def __init__(
        self,
        *,
        confidence_range: tuple[float, float] = (0.10, 0.95),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared CoTracker3 plugin; see the class docstring."""
        super().__init__(**kwargs)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))
        self.visible_threshold = float(visible_threshold)

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether Meta's ``cotracker`` package is importable."""
        try:
            import cotracker  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _build_backend(self, device: str, weights: Any) -> TrackerBackend:
        """Load the CoTracker3 predictor from ``weights`` onto ``device``.

        Raises:
            TrackerError: If the ``cotracker`` package is missing or the
                checkpoint will not load.
        """
        torch = torch_module()
        if torch is None:
            raise TrackerError(
                "CoTracker3 needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        try:
            from cotracker.predictor import CoTrackerPredictor  # noqa: PLC0415
        except Exception as exc:
            raise TrackerError(
                "CoTracker3 needs Meta's cotracker package for the model code: "
                'pip install "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"'
            ) from exc
        model = CoTrackerPredictor(checkpoint=str(weights), offline=True, v2=False)
        model = model.to(device)
        model.eval()
        return CoTrackerBackend(model, device)

    def normalize_confidence(self, prediction: BackendPrediction) -> tuple[np.ndarray, np.ndarray]:
        """Combine CoTracker3's visibility and confidence heads into ``[0, 1]``.

        The two heads answer different questions — "can you see it?" and "is
        this where it is?" — and both must hold for a sample to be usable, so
        they multiply. The product is then stretched onto the shared scale with
        :attr:`confidence_range`.

        Args:
            prediction: The backend's raw window output.

        Returns:
            ``(confidence [L, N] float32, visible [L, N] bool)``.
        """
        raw = combine_scores(prediction.visibility, prediction.confidence)
        low, high = self.confidence_range
        confidence = normalize_scores(raw, low=low, high=high)
        visible = visibility_from_confidence(confidence, self.visible_threshold)
        return confidence, visible
