"""TAPIR / BootsTAP — DeepMind's point tracker, as a Stable plugin.

TAPIR (Doersch et al., 2023) tracks each query point by matching it against a
cost volume over the whole clip and then iteratively refining that estimate, and
BootsTAPIR adds large-scale self-supervised bootstrapping on real footage. The
practical difference from CoTracker3 is that TAPIR treats points independently,
which makes it excellent on isolated, fast-moving features and cheaper per point,
where CoTracker3's joint attention helps most when several points move together.

Both are worth having, and the point of the plugin contract is that the user can
try one and then the other without anything downstream noticing.

Licensing
---------
BootsTAPIR weights are **Apache-2.0** — commercially usable, unlike CoTracker's.
That makes this the default deep tracker to recommend.

Getting the model code
----------------------
Weights come from DeepMind's release; the PyTorch architecture lives in the
``tapnet`` package::

    pip install "tapnet @ git+https://github.com/google-deepmind/tapnet.git"

:meth:`TapirTracker.is_available` reports False until that (and torch) is
importable.

Confidence
----------
TAPIR emits two logits per sample: ``occlusion`` and ``expected_dist`` (is the
prediction further than the training threshold from the truth?). The library's
own visibility rule is ``(1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist))
> 0.5``; Stable keeps that product as the *continuous* confidence, which is
exactly the shared meaning "probability this position is accurate", and only
thresholds it for the boolean ``visible`` flag.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import TrackerError
from .confidence import (
    combine_scores,
    normalize_scores,
    sigmoid,
    visibility_from_confidence,
)
from .deep import BackendPrediction, DeepTrackerPlugin, TrackerBackend
from .weights import ModelSpec, register_model

__all__ = ["TapirTracker", "BOOTSTAPIR"]

logger = logging.getLogger(__name__)

BOOTSTAPIR = register_model(
    ModelSpec(
        key="bootstapir",
        url="https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
        filename="bootstapir_checkpoint_v2.pt",
        sha256=None,  # unpinned: see CONTRIBUTING.md "Pinning a checkpoint"
        license="Apache-2.0",
        non_commercial=False,
        notes="BootsTAPIR v2 PyTorch checkpoint from DeepMind's tapnet release.",
    )
)
"""The BootsTAPIR checkpoint this plugin loads."""


class TapirBackend:
    """Runs a loaded TAPIR model on one window of frames.

    Args:
        model: A ``tapnet.torch.tapir_model.TAPIR`` instance already on
            ``device``.
        device: The device string the model currently lives on.

    Attributes:
        device: Where the model currently is; updated by :meth:`to_device`.
    """

    def __init__(self, model: Any, device: str) -> None:
        """Wrap an already-loaded TAPIR model."""
        self._model = model
        self.device = device

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Track ``queries`` through ``frames`` with TAPIR.

        TAPIR wants frames scaled to ``[-1, 1]`` and queries as ``(t, y, x)``,
        both of which are handled here so the rest of Stable can stay in uint8
        RGB and ``(t, x, y)``.

        Args:
            frames: ``[L, H, W, 3]`` uint8 RGB in run order.
            queries: ``[N, 3]`` rows of ``(local_frame, x, y)``.

        Returns:
            The window's :class:`~stable.trackers.deep.BackendPrediction`, with
            raw occlusion/expected-distance logits already turned into
            probabilities.

        Raises:
            TrackerError: If the model returns something unrecognisable.
        """
        torch = torch_module()
        if torch is None:  # pragma: no cover - guarded by is_available()
            raise TrackerError("TAPIR needs PyTorch; install the 'torch' extra")
        video = torch.from_numpy(np.ascontiguousarray(frames)).float().to(self.device)
        video = (video / 255.0) * 2.0 - 1.0
        points = np.stack([queries[:, 0], queries[:, 2], queries[:, 1]], axis=1)
        query_tensor = torch.from_numpy(np.ascontiguousarray(points)).float().to(self.device)
        with torch.inference_mode():
            output = self._model(video[None], query_tensor[None])
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
    """Turn TAPIR's output dict into a :class:`BackendPrediction`.

    TAPIR returns ``tracks`` ``[B, N, L, 2]`` plus ``occlusion`` and
    ``expected_dist`` logits ``[B, N, L]``; Stable's window convention is
    frame-major ``[L, N, ...]``, so the axes are transposed here. The two logits
    become probabilities: ``visibility = 1 - sigmoid(occlusion)`` and
    ``confidence = 1 - sigmoid(expected_dist)``.

    Args:
        output: The model's return value (a mapping, or a tuple in that order).

    Returns:
        The window's prediction.

    Raises:
        TrackerError: If the expected keys are absent.
    """
    if isinstance(output, dict):
        try:
            tracks = _to_numpy(output["tracks"])
            occlusion = _to_numpy(output["occlusion"])
            expected_dist = _to_numpy(output["expected_dist"])
        except KeyError as exc:
            raise TrackerError(
                f"TAPIR output is missing key {exc.args[0]!r}; expected tracks, occlusion "
                "and expected_dist"
            ) from exc
    elif isinstance(output, tuple | list) and len(output) == 3:
        tracks, occlusion, expected_dist = (_to_numpy(item) for item in output)
    else:
        raise TrackerError(
            f"unexpected TAPIR output {type(output).__name__}; expected a dict with "
            "tracks/occlusion/expected_dist"
        )
    if tracks.ndim == 4:  # [B, N, L, 2] -> [N, L, 2]
        tracks = tracks[0]
    if occlusion.ndim == 3:
        occlusion = occlusion[0]
    if expected_dist.ndim == 3:
        expected_dist = expected_dist[0]
    return BackendPrediction(
        points=np.transpose(tracks, (1, 0, 2)).astype(np.float32),
        visibility=(1.0 - sigmoid(occlusion)).T.astype(np.float32),
        confidence=(1.0 - sigmoid(expected_dist)).T.astype(np.float32),
    )


def _to_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a detached CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class TapirTracker(DeepTrackerPlugin):
    """BootsTAPIR as a Stable tracker plugin.

    Args:
        confidence_range: ``(low, high)`` calibration for the combined
            ``(1 - occlusion) x (1 - expected_dist)`` score. TAPIR's
            expected-distance head is deliberately conservative, so the usable
            band starts a little higher than CoTracker3's.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible — ``0.5`` reproduces TAPIR's own visibility rule.
        **kwargs: Forwarded to :class:`~stable.trackers.deep.DeepTrackerPlugin`.
    """

    name = "tapir"
    display_name = "TAPIR / BootsTAP (DeepMind)"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "Apache-2.0"

    model_spec = BOOTSTAPIR
    install_hint = (
        'Install it with: pip install "tapnet @ git+https://github.com/google-deepmind/tapnet.git"'
    )
    # TAPIR builds a cost volume per point, so windows are shorter than
    # CoTracker3's; 256x256 is the resolution BootsTAPIR was trained at.
    default_chunk_size = 32
    default_overlap = 8
    default_model_resolution = (256, 256)

    def __init__(
        self,
        *,
        confidence_range: tuple[float, float] = (0.15, 0.90),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared TAPIR plugin; see the class docstring."""
        super().__init__(**kwargs)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))
        self.visible_threshold = float(visible_threshold)

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether DeepMind's ``tapnet`` PyTorch model code is importable."""
        try:
            from tapnet.torch import tapir_model  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _build_backend(self, device: str, weights: Any) -> TrackerBackend:
        """Load BootsTAPIR from ``weights`` onto ``device``.

        Raises:
            TrackerError: If ``tapnet`` is missing or the checkpoint will not
                load.
        """
        torch = torch_module()
        if torch is None:
            raise TrackerError(
                "TAPIR needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        try:
            from tapnet.torch import tapir_model  # noqa: PLC0415
        except Exception as exc:
            raise TrackerError(
                "TAPIR needs DeepMind's tapnet package for the model code: "
                'pip install "tapnet @ git+https://github.com/google-deepmind/tapnet.git"'
            ) from exc
        model = tapir_model.TAPIR(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        return TapirBackend(model, device)

    def normalize_confidence(self, prediction: BackendPrediction) -> tuple[np.ndarray, np.ndarray]:
        """Combine TAPIR's visibility and accuracy probabilities into ``[0, 1]``.

        Args:
            prediction: The backend's raw window output, whose ``visibility``
                and ``confidence`` are already ``1 - sigmoid(logit)``.

        Returns:
            ``(confidence [L, N] float32, visible [L, N] bool)``.
        """
        raw = combine_scores(prediction.visibility, prediction.confidence)
        low, high = self.confidence_range
        confidence = normalize_scores(raw, low=low, high=high)
        visible = visibility_from_confidence(confidence, self.visible_threshold)
        return confidence, visible
