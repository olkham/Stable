"""Shared machinery for deep-learning tracker plugins (CoTracker3, TAPIR, ...).

A neural point tracker plugin is mostly *not* the neural network. It is: pick a
device and survive not getting the one you wanted; make sure the weights are on
disk; cut the clip into windows the GPU can hold; feed each window at the model's
input resolution and map coordinates back; stitch the windows; and turn the
model's scores into the project's one confidence scale. All of that is identical
between CoTracker3 and TAPIR, so it lives here and each plugin supplies only:

* a :class:`ModelSpec` (where its checkpoint comes from),
* :meth:`DeepTrackerPlugin._build_backend` (how to load it), and
* :meth:`DeepTrackerPlugin.normalize_confidence` (what its scores mean).

The model itself sits behind :class:`TrackerBackend`, a three-method seam taking
plain NumPy in and out. That keeps torch out of the pipeline code, lets the whole
chunk/stitch/timeline path be tested on a machine with no GPU and no torch by
injecting a fake backend, and means a future runtime (ONNX, TensorRT, a remote
worker) is a new backend rather than a new plugin.

Device policy
-------------
``prepare()`` resolves ``auto`` to CUDA when it is there and CPU when it is not.
If the weights will not load on the GPU, or a window blows out GPU memory
mid-run, the plugin moves to CPU and carries on — a slow result beats a crashed
job, and :attr:`DeepTrackerPlugin.fell_back_to_cpu` records that it happened so
the UI can say so. Out-of-memory on CPU is a real failure and is reported as one.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.device import select_device, torch_module
from ..core.errors import ContractError, StableError, TrackerError
from ..core.types import Query, TrackResult
from ..core.video import VideoSource
from .base import Direction, ProgressCallback, TrackerPlugin, noop_progress
from .chunking import Chunk, ChunkStitcher, plan_chunks
from .weights import ModelSpec, ensure_weights, is_downloaded, local_path

__all__ = [
    "BackendPrediction",
    "TrackerBackend",
    "DeepTrackerPlugin",
    "is_out_of_memory",
    "resize_frames",
]

logger = logging.getLogger(__name__)

_EMPTY_FRAME = np.zeros((0, 0, 3), dtype=np.uint8)
"""Placeholder filling a window slot until its frame is decoded."""


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether an exception is an out-of-memory failure from a compute backend.

    Matches ``torch.cuda.OutOfMemoryError`` (by name, so torch need not be
    imported), the ``RuntimeError`` older torch builds raise instead, and plain
    :class:`MemoryError`.

    Args:
        exc: The exception to classify.

    Returns:
        True if the failure is "not enough memory" rather than a real bug.
    """
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ in ("OutOfMemoryError", "OutOfMemoryError_"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a stack of RGB frames to ``size``.

    Uses OpenCV when it is installed (``INTER_AREA`` down, ``INTER_LINEAR`` up)
    and falls back to torch's bilinear interpolation, so the deep plugins do not
    add OpenCV to their dependency set just to rescale.

    Args:
        frames: ``[L, H, W, 3]`` uint8 RGB.
        size: Target ``(height, width)``.

    Returns:
        ``[L, size[0], size[1], 3]`` uint8 RGB. The input is returned unchanged
        when it already has the requested size.

    Raises:
        TrackerError: If neither OpenCV nor torch is importable.
    """
    height, width = int(size[0]), int(size[1])
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    try:
        import cv2  # noqa: PLC0415 - optional, resolved per call
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    if cv2 is not None:
        shrinking = height * width < frames.shape[1] * frames.shape[2]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return np.stack(
            [cv2.resize(frame, (width, height), interpolation=interpolation) for frame in frames]
        )
    torch = torch_module()
    if torch is None:  # pragma: no cover - needs an env with neither library
        raise TrackerError(
            "resizing frames needs either opencv-python or torch; install the "
            "'video' or 'torch' extra"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    resized = torch.nn.functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


@dataclass(frozen=True)
class BackendPrediction:
    """One window of raw model output, in model-input pixel coordinates.

    Deliberately *raw*: the backend does no calibration, because what a score
    means is a property of the model and belongs in the plugin (see
    :meth:`DeepTrackerPlugin.normalize_confidence`).

    Args:
        points: ``[L, N, 2]`` positions for the window's ``L`` frames and ``N``
            query points, in the pixel space of the frames that were passed in.
        visibility: ``[L, N]`` probability that the point is visible (not
            occluded, in frame), in ``[0, 1]``.
        confidence: ``[L, N]`` optional second head — the model's own estimate
            that the prediction is accurate — or ``None`` for models without one.

    Raises:
        ContractError: On a shape mismatch between the three arrays.
    """

    points: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce dtypes and check the shapes agree (dataclass hook)."""
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 3 or points.shape[2] != 2:
            raise ContractError(f"prediction points must be [L, N, 2], got {points.shape}")
        expected = points.shape[:2]
        visibility = np.asarray(self.visibility, dtype=np.float32)
        if visibility.shape != expected:
            raise ContractError(f"prediction visibility must be {expected}, got {visibility.shape}")
        confidence = self.confidence
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != expected:
                raise ContractError(
                    f"prediction confidence must be {expected}, got {confidence.shape}"
                )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "confidence", confidence)

    @property
    def n_frames(self) -> int:
        """Window length ``L``."""
        return int(self.points.shape[0])

    @property
    def n_points(self) -> int:
        """Number of points ``N`` in the window."""
        return int(self.points.shape[1])


@runtime_checkable
class TrackerBackend(Protocol):
    """The model seam: NumPy in, :class:`BackendPrediction` out.

    Implementations own a loaded network. They are created by
    :meth:`DeepTrackerPlugin._build_backend` and may also be injected into a
    plugin's constructor, which is how the pipeline is tested without weights.
    """

    device: str

    def infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Track ``queries`` through one window of frames.

        Args:
            frames: ``[L, H, W, 3]`` uint8 RGB, in run order — for a backward
                run this is the clip played backwards, so a backend never needs
                to know about direction.
            queries: ``[N, 3]`` float32 rows of ``(local_frame, x, y)`` in the
                pixel space of ``frames``.

        Returns:
            A :class:`BackendPrediction` covering all ``L`` frames.
        """
        ...

    def to_device(self, device: str) -> None:
        """Move the loaded model onto ``device`` (used by the CPU fallback)."""
        ...

    def release(self) -> None:
        """Free the model and any device memory. Must be safe to call twice."""
        ...


class DeepTrackerPlugin(TrackerPlugin):
    """Base class implementing chunked, resolution-scaled, resilient tracking.

    Subclasses set the class attributes below and implement
    :meth:`_build_backend` and :meth:`normalize_confidence`.

    Args:
        chunk_size: Frames per inference window; ``None`` uses
            :attr:`default_chunk_size`. Lower it if a GPU is tight on memory.
        overlap: Frames shared between adjacent windows, crossfaded to hide the
            seam; ``None`` uses :attr:`default_overlap`.
        model_resolution: ``(height, width)`` the frames are resized to before
            inference, or ``None`` to feed native resolution. This is the main
            reason a 1080p clip fits in 8 GB.
        device: Preferred device string; ``None``/``"auto"`` picks CUDA when
            available. Resolved in :meth:`prepare`.
        allow_cpu_fallback: Whether an out-of-memory failure on the GPU should
            silently continue on the CPU.
        backend: A ready-made :class:`TrackerBackend` to use instead of loading
            weights. Intended for tests and alternative runtimes.
        allow_download: Passed to :func:`~stable.trackers.weights.ensure_weights`;
            False makes :meth:`prepare` fail rather than hit the network.

    Attributes:
        model_spec: The checkpoint this plugin needs, or ``None`` if it has none.
        default_chunk_size: Window length that fits a typical 8 GB GPU.
        default_overlap: Default crossfade length between windows.
        default_model_resolution: Input size the model was trained at.
        install_hint: How to install the package holding the model code, shown
            when it is missing.
    """

    model_spec: ModelSpec | None = None
    install_hint: str = ""
    default_chunk_size: int = 48
    default_overlap: int = 8
    default_model_resolution: tuple[int, int] | None = None

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        model_resolution: tuple[int, int] | None | str = "default",
        device: str | None = None,
        allow_cpu_fallback: bool = True,
        backend: TrackerBackend | None = None,
        allow_download: bool = True,
    ) -> None:
        """Create an unprepared deep tracker; see the class docstring."""
        super().__init__()
        self.chunk_size = int(chunk_size if chunk_size is not None else self.default_chunk_size)
        self.overlap = int(overlap if overlap is not None else self.default_overlap)
        if self.chunk_size < 2:
            raise ContractError(f"chunk_size must be >= 2, got {self.chunk_size}")
        if not 0 <= self.overlap < self.chunk_size:
            raise ContractError(
                f"overlap must lie in [0, chunk_size), got {self.overlap} with "
                f"chunk_size={self.chunk_size}"
            )
        self.model_resolution: tuple[int, int] | None = (
            self.default_model_resolution if isinstance(model_resolution, str) else model_resolution
        )
        self.requested_device = device
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.allow_download = bool(allow_download)
        self.fell_back_to_cpu = False
        self._backend: TrackerBackend | None = backend
        self._injected_backend = backend

    # -- availability ----------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether torch and the model's implementation package are importable.

        Never touches the network or the weight cache — the UI calls this to
        decide whether to grey out a tracker, and that must be instant.
        """
        if torch_module() is None:
            return False
        return cls._implementation_available()

    @classmethod
    def _implementation_available(cls) -> bool:
        """Whether the third-party package holding the model code is importable."""
        return True

    def _require_implementation(self) -> None:
        """Fail fast if this plugin cannot possibly run in this environment.

        Called before the weights are fetched: discovering a missing package
        *after* pulling several hundred megabytes would be an unkind way to
        learn about it.

        Raises:
            TrackerError: If PyTorch or the model's implementation package is
                not importable, with the command that fixes it.
        """
        if torch_module() is None:
            raise TrackerError(
                f"{self.name} needs PyTorch; install the 'torch' extra "
                "(`pip install stable-tracker[torch]`)"
            )
        if not self._implementation_available():
            raise TrackerError(
                f"{self.name} needs its model code, which Stable does not vendor. "
                f"{self.install_hint}".strip()
            )

    @classmethod
    def weights_downloaded(cls) -> bool:
        """Whether this plugin's checkpoint is already cached and verified."""
        return cls.model_spec is None or is_downloaded(cls.model_spec)

    @classmethod
    def weights_path(cls) -> str | None:
        """Return where this plugin's checkpoint is (or would be) cached."""
        return None if cls.model_spec is None else str(local_path(cls.model_spec))

    # -- lifecycle -------------------------------------------------------

    def prepare(self, device: str = "auto", progress: ProgressCallback = noop_progress) -> None:
        """Resolve the device, fetch the weights if needed, and load the model.

        Idempotent: re-preparing on the same device is a no-op, so the registry
        preparing a plugin it just created costs nothing.

        Args:
            device: Preferred device; ``"auto"`` picks CUDA when available.
            progress: Reporter for the (potentially long) download and load.

        Raises:
            TrackerError: If PyTorch or the model's implementation package is
                missing, the weights cannot be fetched or verified, or the model
                will not load.
        """
        resolved = select_device(device if device else self.requested_device)
        if self._prepared and self._backend is not None and self.device == resolved:
            return

        if self._injected_backend is not None:
            self._backend = self._injected_backend
            self.device = resolved
            self._prepared = True
            return

        self._require_implementation()
        weights = None
        if self.model_spec is not None:
            weights = ensure_weights(
                self.model_spec, progress=progress, allow_download=self.allow_download
            )
        progress(0.9, f"loading {self.name} on {resolved}")
        try:
            self._backend = self._build_backend(resolved, weights)
        except Exception as exc:
            if resolved != "cpu" and self.allow_cpu_fallback and _is_device_failure(exc):
                logger.warning(
                    "%s could not load on %s (%s); falling back to CPU", self.name, resolved, exc
                )
                self.fell_back_to_cpu = True
                resolved = "cpu"
                self._backend = self._build_backend(resolved, weights)
            elif isinstance(exc, StableError):
                raise
            else:
                raise TrackerError(f"could not load {self.name} weights: {exc}") from exc
        self.device = resolved
        self._prepared = True
        progress(1.0, f"{self.name} ready on {resolved}")

    def release(self) -> None:
        """Free the model and empty the CUDA cache. Safe to call twice."""
        if self._backend is not None and self._backend is not self._injected_backend:
            try:
                self._backend.release()
            except Exception as exc:  # pragma: no cover - backend-specific
                logger.debug("%s backend release failed: %s", self.name, exc)
        self._backend = None
        _empty_cuda_cache()
        super().release()

    @abstractmethod
    def _build_backend(self, device: str, weights: Any) -> TrackerBackend:
        """Load the model and return a backend bound to ``device``.

        Args:
            device: Resolved device string.
            weights: Path to the verified checkpoint, or ``None`` if the plugin
                declares no :attr:`model_spec`.

        Returns:
            A ready :class:`TrackerBackend`.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize_confidence(self, prediction: BackendPrediction) -> tuple[np.ndarray, np.ndarray]:
        """Map one window's raw scores onto the shared ``[0, 1]`` confidence.

        Args:
            prediction: The backend's raw output for a window.

        Returns:
            ``(confidence [L, N] float32 in [0, 1], visible [L, N] bool)``. See
            :mod:`stable.trackers.confidence` for the shared meaning and helpers.
        """
        raise NotImplementedError

    # -- the contract ----------------------------------------------------

    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Track every query point through ``frame_range``. One run, no merging.

        Points sharing an anchor frame are tracked in a single batched pass —
        the usual case, and where these models are much faster than tracking
        points one at a time.

        Args:
            video: Clip to read frames from.
            queries: Anchor positions, one per point; each must lie inside
                ``frame_range``.
            frame_range: ``(start, end)`` half-open absolute frame range.
            direction: ``"fwd"``, ``"bwd"`` or ``"bidir"``.
            progress: Progress reporter, driven per inference window.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``, in query order.

        Raises:
            TrackerError: If the range or direction is invalid, a query lies
                outside the range, or the model fails (including running out of
                memory with no fallback left).
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
        if not self.is_prepared:
            self.prepare(self.requested_device or "auto")
        if self._backend is None:  # pragma: no cover - prepare() guarantees this
            raise TrackerError(f"{self.name} has no loaded model; call prepare() first")

        points = np.full((len(queries), n_frames, 2), np.nan, dtype=np.float32)
        confidence = np.zeros((len(queries), n_frames), dtype=np.float32)
        visible = np.zeros((len(queries), n_frames), dtype=bool)

        groups: OrderedDict[int, list[int]] = OrderedDict()
        for index, query in enumerate(queries):
            groups.setdefault(query.frame, []).append(index)

        tasks: list[tuple[int, list[int], int, int]] = []
        for anchor_frame, indices in groups.items():
            if direction in ("fwd", "bidir") and end - anchor_frame > 1:
                tasks.append((anchor_frame, indices, 1, end - anchor_frame))
            if direction in ("bwd", "bidir") and anchor_frame - start > 0:
                tasks.append((anchor_frame, indices, -1, anchor_frame - start + 1))

        total_chunks = sum(
            len(plan_chunks(length, self.chunk_size, self.overlap)) for _, _, _, length in tasks
        )
        done_chunks = 0

        for anchor_frame, indices, sign, length in tasks:
            seeds = np.array([queries[i].position for i in indices], dtype=np.float64)
            stitcher = ChunkStitcher(length, len(indices))
            plan = plan_chunks(length, self.chunk_size, self.overlap)
            for position, chunk in enumerate(plan):
                absolute = anchor_frame + sign * np.arange(chunk.start, chunk.end)
                frames, scale = self._load_window(video, absolute)
                model_queries = np.zeros((len(indices), 3), dtype=np.float32)
                model_queries[:, 1] = seeds[:, 0] * scale[0]
                model_queries[:, 2] = seeds[:, 1] * scale[1]

                prediction = self._infer(frames, model_queries)
                if prediction.n_frames != chunk.n_frames or prediction.n_points != len(indices):
                    raise TrackerError(
                        f"{self.name} backend returned {prediction.n_frames}x"
                        f"{prediction.n_points}, expected {chunk.n_frames}x{len(indices)}"
                    )
                window_points = prediction.points.astype(np.float64)
                window_points[..., 0] /= scale[0]
                window_points[..., 1] /= scale[1]
                window_conf, window_visible = self.normalize_confidence(prediction)
                stitcher.add(chunk, window_points, window_conf, window_visible)

                following = plan[position + 1] if position + 1 < len(plan) else None
                seeds = _next_seeds(window_points, chunk, following, seeds)
                done_chunks += 1
                progress(
                    done_chunks / max(total_chunks, 1),
                    f"{self.name}: window {done_chunks}/{total_chunks}",
                )

            run_points, run_conf, run_visible = stitcher.result()
            absolute = anchor_frame + sign * np.arange(length)
            for slot, point_index in enumerate(indices):
                points[point_index, absolute] = run_points[:, slot]
                confidence[point_index, absolute] = run_conf[:, slot]
                visible[point_index, absolute] = run_visible[:, slot]

        # The anchor is ground truth: never let a model's own prediction for its
        # query frame override the position the user placed there.
        for index, query in enumerate(queries):
            points[index, query.frame] = query.position
            confidence[index, query.frame] = 1.0
            visible[index, query.frame] = True

        progress(1.0, "done")
        return TrackResult(points=points, confidence=confidence, visible=visible)

    # -- internals -------------------------------------------------------

    def _load_window(
        self, video: VideoSource, absolute: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Fetch and resize the frames for one window.

        Two details matter for memory and speed, both invisible to the backend:

        * Frames are **decoded in ascending order** and only then arranged into
          run order. Video decode is sequential, so pulling a backward window
          back-to-front costs one seek per frame; pulling it front-to-back costs
          one seek per window.
        * Each frame is resized as it arrives rather than after stacking, so the
          window never exists at full resolution — the difference between ~35 MB
          and ~300 MB for 48 frames of 1080p.

        Args:
            video: Clip to read from.
            absolute: Absolute frame indices in run order (descending for a
                backward run).

        Returns:
            ``(frames [L, H, W, 3] uint8, scale (sx, sy))`` where ``scale``
            converts source pixels to model pixels.

        Raises:
            TrackerError: If the source does not hand back ``[H, W, 3]`` RGB.
        """
        loaded: list[np.ndarray] = [_EMPTY_FRAME] * len(absolute)
        source_shape: tuple[int, int] | None = None
        for slot in np.argsort(absolute, kind="stable"):
            frame = np.asarray(video.get_frame(int(absolute[slot])))
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise TrackerError(f"video frames must be [H, W, 3] RGB, got {frame.shape}")
            source_shape = (frame.shape[0], frame.shape[1])
            if self.model_resolution is not None:
                frame = resize_frames(frame[None], self.model_resolution)[0]
            loaded[slot] = frame

        frames = np.ascontiguousarray(np.stack(loaded))
        scale = (1.0, 1.0)
        if self.model_resolution is not None and source_shape is not None:
            target_h, target_w = self.model_resolution
            scale = (target_w / source_shape[1], target_h / source_shape[0])
        return frames, scale

    def _infer(self, frames: np.ndarray, queries: np.ndarray) -> BackendPrediction:
        """Run one window through the backend, absorbing out-of-memory failures.

        On GPU OOM the model is moved to the CPU and the window is retried
        there; the rest of the run then stays on the CPU rather than thrashing
        back and forth. OOM on the CPU is fatal.

        Raises:
            TrackerError: If inference fails for any reason other than a
                recoverable OOM.
        """
        backend = self._backend
        assert backend is not None  # noqa: S101 - track() checks this
        try:
            return backend.infer(frames, queries)
        except Exception as exc:
            if not is_out_of_memory(exc):
                if isinstance(exc, StableError):
                    raise
                raise TrackerError(f"{self.name} inference failed: {exc}") from exc
            if self.device == "cpu" or not self.allow_cpu_fallback:
                raise TrackerError(
                    f"{self.name} ran out of memory on {self.device} "
                    f"(window of {frames.shape[0]} frames at "
                    f"{frames.shape[2]}x{frames.shape[1]}). Reduce chunk_size or "
                    "model_resolution."
                ) from exc
            logger.warning(
                "%s ran out of memory on %s; retrying this window on the CPU",
                self.name,
                self.device,
            )
            _empty_cuda_cache()
            try:
                backend.to_device("cpu")
            except Exception as move_exc:  # pragma: no cover - backend-specific
                raise TrackerError(
                    f"{self.name} ran out of memory and could not fall back to CPU: {move_exc}"
                ) from move_exc
            self.device = "cpu"
            self.fell_back_to_cpu = True
            _empty_cuda_cache()
            try:
                return backend.infer(frames, queries)
            except Exception as retry_exc:
                raise TrackerError(
                    f"{self.name} inference failed after falling back to CPU: {retry_exc}"
                ) from retry_exc

    def __repr__(self) -> str:
        """Return a debug representation naming the device and window size."""
        return (
            f"<{type(self).__name__} name={self.name!r} device={self.device!r} "
            f"chunk={self.chunk_size} overlap={self.overlap}>"
        )


def _next_seeds(
    window_points: np.ndarray, chunk: Chunk, following: Chunk | None, previous: np.ndarray
) -> np.ndarray:
    """Return the query positions to re-anchor the next window on.

    The next window starts inside this one (that is what the overlap is for), so
    its first frame's position is already predicted here. A non-finite
    prediction there falls back to the previous seed rather than poisoning the
    rest of the run with ``NaN``.
    """
    if following is None:
        return previous
    # With zero overlap the next window starts one past this one, so the closest
    # thing to a re-anchor is this window's last prediction.
    source = min(max(following.start, chunk.start), chunk.end - 1)
    candidate = window_points[source - chunk.start]
    good = np.isfinite(candidate).all(axis=1)
    return np.where(good[:, None], candidate, previous)


def _is_device_failure(exc: BaseException) -> bool:
    """Whether a model-load failure looks like "this device will not work"."""
    if is_out_of_memory(exc):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("cuda", "cudnn", "no kernel image", "device-side", "gpu", "mps")
    )


def _empty_cuda_cache() -> None:
    """Return GPU memory to the driver, if torch and a GPU are present."""
    torch = torch_module()
    if torch is None:
        return
    try:  # pragma: no cover - only meaningful with a real GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - driver-dependent
        logger.debug("could not empty the CUDA cache: %s", exc)
