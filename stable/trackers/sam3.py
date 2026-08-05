"""SAM 3 — segment the thing, then let the point ride on it (todo 27).

Every other tracker here answers "where did this *pixel neighbourhood* go?".
That question has no answer while the subject is behind something else, which is
why occlusion is where they all fail: LK integrates flow into the occluder,
template matching latches onto whatever the occluder looks like, and feature
matching (:mod:`stable.trackers.features`) at least says "I don't know" but
cannot say when the subject is back.

SAM 3 answers a different question — "where is this *object*?" — and its video
model carries a memory of the object across frames, so it can report both that
the object is gone and that it has returned. The point the operator placed then
rides along with the object's mask:

#. the query point is added to a video inference session as an object prompt on
   the run's first frame (``add_inputs_to_inference_session``);
#. ``propagate_in_video_iterator`` walks the window, maintaining the model's own
   memory of that object — the propagation is the model's, not a mask chained
   by hand frame to frame;
#. the tracked position is the query point **carried by the mask's motion**,
   not the mask's centroid — the operator picked a corner of a sign, not the
   middle of the sign, and a centroid would quietly move their point.

``object_score_logits`` is the model's own "is the object present?" head, and it
is what makes occlusion legible: it collapses when the subject is hidden, which
becomes a low confidence and ``visible=False`` on the shared scale, and it
recovers when the subject does — the mask picking the object back up, rather
than a search window that has long since drifted.

Weights
-------
This is the one deep plugin with no :class:`~stable.trackers.weights.ModelSpec`.
The checkpoint is not a single file at a URL — it is a Hugging Face repo
(``facebook/sam3``) fetched and cached by ``transformers`` itself, which does its
own integrity checking. Rather than mirror the file layout (and pin hashes that
would go stale on the next revision), this plugin defers to that cache and says
so; :meth:`Sam3Tracker.weights_downloaded` therefore reports what the HF cache
holds rather than what Stable's own model manager has.

Status
------
The tracking pipeline around it — windowing, re-anchoring, stitching, confidence
normalisation, CPU fallback — is exercised by ``tests/test_tracker_suite.py``
through the injected-backend seam, exactly as CoTracker3 and TAPIR are. The
model call in :class:`Sam3Backend` itself is written against the documented
``Sam3TrackerVideoModel``/``Sam3TrackerVideoProcessor`` API and has **not** been run
against the real checkpoint here (it is a gated, multi-gigabyte download); treat
that method as unverified until someone runs it on real weights.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.device import torch_module
from ..core.errors import TrackerError
from .confidence import combine_scores, normalize_scores, sigmoid, visibility_from_confidence
from .deep import BackendPrediction, DeepTrackerPlugin, TrackerBackend

__all__ = ["Sam3Tracker", "Sam3Backend", "DEFAULT_CHECKPOINT"]

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "facebook/sam3"
"""The Hugging Face repo the plugin loads by default."""


def _centroids(masks: np.ndarray) -> np.ndarray:
    """Return each mask's centre of mass, or ``NaN`` where a mask is empty.

    Args:
        masks: ``[N, H, W]`` boolean or float masks.

    Returns:
        ``[N, 2]`` ``(x, y)`` centroids in mask pixels.
    """
    out = np.full((masks.shape[0], 2), np.nan, dtype=np.float64)
    for index, mask in enumerate(masks):
        binary = np.asarray(mask) > 0
        if not binary.any():
            continue
        ys, xs = np.nonzero(binary)
        out[index] = (xs.mean(), ys.mean())
    return out


class Sam3Backend:
    """Runs SAM 3's promptable segmentation over one window of frames.

    Args:
        model: A loaded ``Sam3TrackerVideoModel`` already on ``device``.
        processor: Its matching ``Sam3TrackerVideoProcessor``.
        device: The device string the model currently lives on.
        mask_threshold: Logit above which a mask pixel counts as the object.

    Attributes:
        device: Where the model currently is; updated by :meth:`to_device`.
    """

    def __init__(
        self, model: Any, processor: Any, device: str, *, mask_threshold: float = 0.0
    ) -> None:
        """Wrap an already-loaded SAM 3 model and its processor."""
        self._model = model
        self._processor = processor
        self.device = device
        self.mask_threshold = float(mask_threshold)

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Propagate each query's object through ``frames``.

        Args:
            frames: ``[L, H, W, 3]`` uint8 RGB in run order.
            queries: ``[N, 3]`` rows of ``(local_frame, x, y)``; by the
                :class:`~stable.trackers.deep.DeepTrackerPlugin` convention
                ``local_frame`` is always 0, so every object is prompted on the
                window's first frame and carried forward from there. A backward
                run is handed the clip already reversed, which is why nothing
                here needs a ``reverse`` flag.

        Returns:
            The window's :class:`~stable.trackers.deep.BackendPrediction`.

        Raises:
            TrackerError: If PyTorch is unavailable or the model returns
                something unrecognisable.
        """
        torch = torch_module()
        if torch is None:  # pragma: no cover - guarded by is_available()
            raise TrackerError("SAM 3 needs PyTorch; install the 'torch' extra")

        length = int(frames.shape[0])
        n_points = int(queries.shape[0])
        points = np.full((length, n_points, 2), np.nan, dtype=np.float32)
        presence = np.zeros((length, n_points), dtype=np.float32)
        quality = np.ones((length, n_points), dtype=np.float32)

        # Where each object sat when it was prompted: the point rides on the
        # mask's *motion*, so this is the origin every later frame is measured
        # against rather than a centroid the operator never chose.
        anchors = np.asarray(queries[:, 1:3], dtype=np.float64)

        session = self._processor.init_video_session(
            video=[np.ascontiguousarray(frame) for frame in frames],
            inference_device=self.device,
        )
        for index in range(n_points):
            self._processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=index,
                input_points=[[[[float(anchors[index][0]), float(anchors[index][1])]]]],
                input_labels=[[[1]]],
            )

        origin: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for output in self._model.propagate_in_video_iterator(
                inference_session=session, start_frame_idx=0
            ):
                frame_index = int(output.frame_idx)
                if not 0 <= frame_index < length:  # pragma: no cover - defensive
                    continue
                masks = self._processor.post_process_masks(
                    output.pred_masks.cpu(),
                    [session.video_height, session.video_width],
                    binarize=True,
                )[0]
                mask_array = np.asarray(masks)
                if mask_array.ndim == 4:  # [objects, 1, H, W]
                    mask_array = mask_array[:, 0]
                elif mask_array.ndim == 2:  # a single object comes back unbatched
                    mask_array = mask_array[None]

                centres = _centroids(mask_array)
                objectness = sigmoid(_to_numpy(output.object_score_logits).reshape(-1))
                object_ids = list(output.object_ids or range(len(centres)))

                for slot, object_id in enumerate(object_ids):
                    point = int(object_id)
                    if not 0 <= point < n_points or slot >= len(centres):
                        continue  # pragma: no cover - defensive
                    centre = centres[slot]
                    if not np.isfinite(centre).all():
                        continue  # the object is not in this frame: leave a gap
                    if point not in origin:
                        origin[point] = centre
                    points[frame_index, point] = anchors[point] + (centre - origin[point])
                    if slot < len(objectness):
                        presence[frame_index, point] = float(objectness[slot])

        return BackendPrediction(points=points, visibility=presence, confidence=quality)

    def to_device(self, device: str) -> None:
        """Move the underlying model to ``device``."""
        mover = getattr(self._model, "to", None)
        if mover is not None:
            mover(device)
        self.device = device

    def release(self) -> None:
        """Drop references to the model so its memory can be reclaimed."""
        self._model = None
        self._processor = None


def _to_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a detached CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class Sam3Tracker(DeepTrackerPlugin):
    """SAM 3 as a Stable tracker plugin, for subjects that get occluded.

    Args:
        checkpoint: Hugging Face repo id to load.
        confidence_range: ``(low, high)`` calibration for the combined
            presence x mask-quality score.
        visible_threshold: Normalised confidence at or above which the point
            counts as visible. Occlusion is the thing this plugin is for, so
            this is the threshold that decides "hidden" versus "here".
        **kwargs: Forwarded to :class:`~stable.trackers.deep.DeepTrackerPlugin`.
    """

    name = "sam3"
    display_name = "SAM 3 (Meta) — occlusion-aware"
    requires_gpu = False  # prefers CUDA, falls back to CPU internally
    license = "SAM-3 licence (see the model card)"

    # No ModelSpec: the checkpoint is a Hugging Face repo that `transformers`
    # fetches and verifies itself. See the module docstring.
    model_spec = None
    install_hint = (
        "Install it with: pip install 'stable-tracker[sam3]' "
        "(transformers >= 5.14, which is where Sam3Tracker landed)"
    )
    # Segmentation is heavy per frame, so windows are short; SAM 3 resizes
    # internally, so frames go in at native resolution.
    default_chunk_size = 16
    default_overlap = 4
    default_model_resolution = None

    def __init__(
        self,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        confidence_range: tuple[float, float] = (0.2, 0.9),
        visible_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Create an unprepared SAM 3 plugin; see the class docstring."""
        super().__init__(**kwargs)
        self.checkpoint = str(checkpoint)
        self.confidence_range = (float(confidence_range[0]), float(confidence_range[1]))
        self.visible_threshold = float(visible_threshold)

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether a ``transformers`` with SAM 3's tracker classes is importable."""
        try:
            import transformers  # noqa: PLC0415
        except Exception:
            return False
        return hasattr(transformers, "Sam3TrackerVideoModel") and hasattr(
            transformers, "Sam3TrackerVideoProcessor"
        )

    @classmethod
    def weights_downloaded(cls) -> bool:
        """Whether the checkpoint is already in the Hugging Face cache.

        Stable's own model manager does not own this one (see the module
        docstring), so this asks ``huggingface_hub`` instead of Stable's cache.
        A missing hub package means "not downloaded", never an error.
        """
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except Exception:
            return False
        try:
            snapshot_download(DEFAULT_CHECKPOINT, local_files_only=True)
        except Exception:
            return False
        return True

    def _build_backend(self, device: str, weights: Any) -> TrackerBackend:
        """Load SAM 3 onto ``device``.

        Args:
            device: Resolved device string.
            weights: Always ``None`` — this plugin has no ``ModelSpec``; the
                checkpoint comes from the Hugging Face cache.

        Raises:
            TrackerError: If PyTorch or a SAM 3-capable transformers is missing,
                or the checkpoint will not load.
        """
        torch = torch_module()
        if torch is None:
            raise TrackerError(
                "SAM 3 needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        try:
            from transformers import (  # noqa: PLC0415
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
            )
        except Exception as exc:
            raise TrackerError(
                "SAM 3 needs transformers >= 5.14 for the model code: "
                "pip install 'stable-tracker[sam3]'"
            ) from exc
        try:
            model = Sam3TrackerVideoModel.from_pretrained(self.checkpoint)
            processor = Sam3TrackerVideoProcessor.from_pretrained(self.checkpoint)
        except Exception as exc:
            raise TrackerError(
                f"could not load SAM 3 checkpoint {self.checkpoint!r}: {exc}. "
                "It is a gated repo — accept the licence on its model page and "
                "log in with `huggingface-cli login` first."
            ) from exc
        # `.to(device)` is typed for PreTrainedModel-to-PreTrainedModel moves in
        # transformers' stubs; the device-string overload is the documented one.
        model = model.to(device)  # type: ignore[arg-type]
        model.eval()
        return Sam3Backend(model, processor, device)

    def normalize_confidence(self, prediction: BackendPrediction) -> tuple[np.ndarray, np.ndarray]:
        """Combine "is the object here?" with "is the mask any good?".

        Args:
            prediction: The backend's raw window output, whose ``visibility`` is
                the presence probability and whose ``confidence`` is the mask
                IoU estimate.

        Returns:
            ``(confidence [L, N] float32, visible [L, N] bool)``.
        """
        raw = combine_scores(prediction.visibility, prediction.confidence)
        low, high = self.confidence_range
        confidence = normalize_scores(raw, low=low, high=high)
        visible = visibility_from_confidence(confidence, self.visible_threshold)
        return confidence, visible
