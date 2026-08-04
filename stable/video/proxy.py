"""Half/quarter-resolution JPEG proxy generation and disk-backed serving.

A UI scrubbing a 1080p timeline should not decode full-resolution video on
every frame move. The pattern here is a one-time cost, then a cheap one:

#. :func:`generate_proxies` decodes the source clip **once**, resizes every
   frame, and writes it to disk as a JPEG. This is the only step that touches
   the real decoder.
#. :class:`ProxyStore` serves frames from those JPEGs. It has no reference to
   the source video at all — structurally, it cannot decode anything — so
   scrubbing a cached clip is disk I/O, never a video decode.

Getting that one-time cost down
-------------------------------
Generation used to be a single loop doing decode → resize → JPEG-encode → write
for one frame at a time, which on a 32-core machine keeps about one core busy.
Two changes fix that, and both are invisible to callers:

* **Encoding is fanned out to a thread pool.** ``cv2.resize``, ``cv2.imencode``
  and the file write all release the GIL, so while the main thread pulls the
  next frame out of the decoder, several workers are compressing the previous
  ones. The pool is bounded so the in-flight frames cannot grow without limit.
* **Every tier comes out of one decode pass.**
  :func:`generate_all_proxies` used to call :func:`generate_proxies` per tier,
  decoding the whole clip once for each. Now the clip is decoded once and each
  decoded frame is resized to every requested tier.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..core.errors import VideoError
from ..core.video import VideoSource
from ..trackers.base import ProgressCallback, noop_progress

__all__ = [
    "PROXY_SCALES",
    "ProxyManifest",
    "generate_proxies",
    "generate_all_proxies",
    "ProxyStore",
    "default_proxy_workers",
]

PROXY_SCALES: dict[str, float] = {"half": 0.5, "quarter": 0.25}
"""Named proxy resolution tiers Stable generates by convention."""

MANIFEST_FILENAME = "proxy.json"
"""Name of the JSON sidecar file written next to a tier's proxy frames."""

_MAX_PROXY_WORKERS = 8
"""Ceiling on JPEG-encoding threads.

Past this the job is bound by the single decoder feeding it, and every extra
worker is another full-resolution frame held in memory waiting for a turn.
"""


def default_proxy_workers() -> int:
    """Return how many JPEG-encoding threads proxy generation should use.

    Returns:
        Between 1 and :data:`_MAX_PROXY_WORKERS`, scaled off the CPU count.
    """
    cpus = os.cpu_count() or 1
    return max(1, min(_MAX_PROXY_WORKERS, cpus - 1))


def _cv2() -> Any:
    """Import and return :mod:`cv2`, lazily.

    Returns:
        The imported ``cv2`` module.

    Raises:
        VideoError: If OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise VideoError(
            "proxy generation requires opencv-python; install the 'video' extra "
            "(`pip install stable-tracker[video]`)"
        ) from exc
    return cv2


def _frame_filename(index: int) -> str:
    """Return the on-disk filename used for proxy frame ``index``."""
    return f"{index:06d}.jpg"


@dataclass(frozen=True)
class ProxyManifest:
    """On-disk description of one proxy tier, alongside its JPEG frames.

    Args:
        count: Number of proxy frames written (equals the source clip's
            ``frame_count``).
        width: Proxy frame width in pixels.
        height: Proxy frame height in pixels.
        scale: Resolution scale relative to the source, e.g. ``0.5``.
        fps: Source frame rate, carried through for transport controls.
        quality: JPEG quality (0-100) used when the proxies were written.
    """

    count: int
    width: int
    height: int
    scale: float
    fps: float
    quality: int = 90

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict written as the manifest file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProxyManifest:
        """Rebuild a manifest from :meth:`to_dict` output.

        Raises:
            VideoError: If a required key is missing.
        """
        try:
            return cls(
                count=int(data["count"]),
                width=int(data["width"]),
                height=int(data["height"]),
                scale=float(data["scale"]),
                fps=float(data["fps"]),
                quality=int(data.get("quality", 90)),
            )
        except KeyError as exc:
            raise VideoError(f"proxy manifest is missing key {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class _Tier:
    """One resolution being written during a generation pass."""

    name: str
    scale: float
    directory: str
    width: int
    height: int


def _plan_tier(video: VideoSource, name: str, scale: float, directory: str) -> _Tier:
    """Validate a tier's scale and compute its output size, creating its directory."""
    if not 0.0 < scale <= 1.0:
        raise VideoError(f"scale must lie in (0, 1], got {scale}")
    os.makedirs(directory, exist_ok=True)
    return _Tier(
        name=name,
        scale=scale,
        directory=directory,
        width=max(1, round(video.width * scale)),
        height=max(1, round(video.height * scale)),
    )


def _write_frame(cv2: Any, tier: _Tier, index: int, frame: np.ndarray, quality: int) -> None:
    """Resize one frame for one tier and write it as a JPEG.

    Runs on a worker thread; every OpenCV call in here releases the GIL, which
    is what lets several of these overlap each other and the decoder.

    Raises:
        VideoError: If the file cannot be written.
    """
    if (frame.shape[1], frame.shape[0]) == (tier.width, tier.height):
        resized = frame
    else:
        resized = cv2.resize(frame, (tier.width, tier.height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    path = os.path.join(tier.directory, _frame_filename(index))
    if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise VideoError(f"failed to write proxy frame {index} to {path!r}")


def _source_frames(video: VideoSource) -> Iterator[tuple[int, np.ndarray]]:
    """Iterate the whole clip, avoiding a per-frame copy where the source allows it.

    :meth:`stable.video.source.PyAVVideoSource.iter_frame_views` hands back the
    cache's own read-only arrays. Nothing here mutates a frame — every OpenCV
    call allocates its own output — so skipping that copy is free speed.
    """
    views = getattr(video, "iter_frame_views", None)
    if callable(views):
        return iter(views())
    return iter(video.iter_frames())


def _write_manifests(
    tiers: Iterable[_Tier], *, count: int, fps: float, quality: int
) -> dict[str, ProxyManifest]:
    """Write each tier's ``proxy.json`` and return the manifests by tier name."""
    manifests: dict[str, ProxyManifest] = {}
    for tier in tiers:
        manifest = ProxyManifest(
            count=count,
            width=tier.width,
            height=tier.height,
            scale=tier.scale,
            fps=fps,
            quality=quality,
        )
        with open(os.path.join(tier.directory, MANIFEST_FILENAME), "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
        manifests[tier.name] = manifest
    return manifests


def _generate(
    video: VideoSource,
    tiers: list[_Tier],
    *,
    quality: int,
    workers: int,
    progress: ProgressCallback,
) -> dict[str, ProxyManifest]:
    """Decode the clip once and write every tier, encoding on a thread pool.

    Raises:
        VideoError: If OpenCV is unavailable or a frame fails to write.
    """
    cv2 = _cv2()
    total = video.frame_count
    if not tiers:
        return {}
    if workers <= 1:
        for index, frame in _source_frames(video):
            for tier in tiers:
                _write_frame(cv2, tier, index, frame, quality)
            progress((index + 1) / total if total else 1.0, f"proxy {index + 1}/{total}")
        return _write_manifests(tiers, count=total, fps=video.fps, quality=quality)

    # Keep only a couple of frames per worker in flight: a full-resolution frame
    # is several megabytes and the decoder can produce them far faster than they
    # can be compressed.
    in_flight: list[Future[None]] = []
    limit = workers * 2
    done = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stable-proxy") as pool:
        for index, frame in _source_frames(video):
            for tier in tiers:
                in_flight.append(pool.submit(_write_frame, cv2, tier, index, frame, quality))
            while len(in_flight) >= limit * len(tiers):
                in_flight.pop(0).result()
            done += 1
            progress(done / total if total else 1.0, f"proxy {done}/{total}")
        for future in in_flight:
            future.result()
    return _write_manifests(tiers, count=total, fps=video.fps, quality=quality)


def generate_proxies(
    video: VideoSource,
    out_dir: str | os.PathLike[str],
    *,
    scale: float = 0.5,
    quality: int = 90,
    workers: int | None = None,
    progress: ProgressCallback = noop_progress,
) -> ProxyManifest:
    """Decode ``video`` once and write one JPEG proxy frame per frame to ``out_dir``.

    Args:
        video: Source clip to decode. Any :class:`~stable.core.video.VideoSource`
            works, including a real :class:`~stable.video.source.PyAVVideoSource`.
        out_dir: Destination directory; created if missing. A manifest file
            (``proxy.json``) is written alongside the frame images.
        scale: Resolution scale relative to the source, e.g. ``0.5`` for a
            half-res proxy or ``0.25`` for quarter-res. See :data:`PROXY_SCALES`.
        quality: JPEG quality in ``[0, 100]`` passed to the encoder.
        workers: Threads used to resize and encode frames; ``None`` picks a
            sensible number for this machine, ``1`` keeps everything on the
            calling thread.
        progress: Progress reporter, called after every frame.

    Returns:
        The :class:`ProxyManifest` describing what was written (also saved to
        ``out_dir/proxy.json`` so a later :class:`ProxyStore` can load it).

    Raises:
        VideoError: If OpenCV is unavailable, a frame fails to write, or
            ``scale``/``quality`` is out of range.
    """
    if not 0 <= quality <= 100:
        raise VideoError(f"quality must lie in [0, 100], got {quality}")
    tier = _plan_tier(video, "proxy", scale, os.fspath(out_dir))
    manifests = _generate(
        video,
        [tier],
        quality=quality,
        workers=default_proxy_workers() if workers is None else max(1, int(workers)),
        progress=progress,
    )
    return manifests["proxy"]


def generate_all_proxies(
    video: VideoSource,
    base_dir: str | os.PathLike[str],
    *,
    tiers: dict[str, float] | None = None,
    quality: int = 90,
    workers: int | None = None,
    progress: ProgressCallback = noop_progress,
) -> dict[str, ProxyManifest]:
    """Generate every named proxy tier into its own subdirectory of ``base_dir``.

    All tiers come out of a **single** decode pass — each decoded frame is
    resized once per tier — so adding the quarter-res tier costs a downscale per
    frame rather than a second trip through the decoder.

    Args:
        video: Source clip to decode.
        base_dir: Parent directory; each tier is written to ``base_dir/<name>``.
        tiers: Mapping of tier name to scale; defaults to :data:`PROXY_SCALES`
            (``half`` and ``quarter``).
        quality: JPEG quality passed to the encoder for every tier.
        workers: Threads used to resize and encode frames; see
            :func:`generate_proxies`.
        progress: Progress reporter; called with the overall fraction across
            all tiers combined.

    Returns:
        Mapping of tier name to its :class:`ProxyManifest`.

    Raises:
        VideoError: If OpenCV is unavailable, a frame fails to write, or a
            tier's scale or the quality is out of range.
    """
    if not 0 <= quality <= 100:
        raise VideoError(f"quality must lie in [0, 100], got {quality}")
    spec = PROXY_SCALES if tiers is None else tiers
    base_dir = os.fspath(base_dir)
    planned = [
        _plan_tier(video, name, scale, os.path.join(base_dir, name))
        for name, scale in spec.items()
    ]
    return _generate(
        video,
        planned,
        quality=quality,
        workers=default_proxy_workers() if workers is None else max(1, int(workers)),
        progress=progress,
    )


class ProxyStore:
    """Serves proxy frames from disk only — no reference to the source video.

    This is what makes timeline scrubbing I/O-bound rather than decode-bound:
    once :func:`generate_proxies` has run, a :class:`ProxyStore` never touches
    a video decoder again, even if the original file is deleted or moved.

    Args:
        directory: A directory previously populated by :func:`generate_proxies`.

    Raises:
        VideoError: If the directory has no ``proxy.json`` manifest.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        """Load the manifest written by :func:`generate_proxies`; see the class docstring."""
        self._directory = os.fspath(directory)
        manifest_path = os.path.join(self._directory, MANIFEST_FILENAME)
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise VideoError(f"no proxy manifest at {manifest_path!r}: {exc}") from exc
        self.manifest = ProxyManifest.from_dict(data)

    @property
    def frame_count(self) -> int:
        """Number of proxy frames available."""
        return self.manifest.count

    def frame_path(self, index: int) -> str:
        """Return the on-disk path of proxy frame ``index``.

        Args:
            index: Absolute frame index.

        Returns:
            The JPEG file path (not guaranteed to exist if the directory was
            tampered with after generation).

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < self.manifest.count:
            raise IndexError(f"proxy frame {index} out of range for count={self.manifest.count}")
        return os.path.join(self._directory, _frame_filename(index))

    def get_frame(self, index: int) -> np.ndarray:
        """Read proxy frame ``index`` from disk as ``uint8`` RGB ``(H, W, 3)``.

        This method never touches the source video — it is a plain image
        read, which is what keeps scrubbing I/O-bound.

        Args:
            index: Absolute frame index.

        Returns:
            The proxy frame.

        Raises:
            IndexError: If ``index`` is out of range.
            VideoError: If the JPEG file is missing or unreadable.
        """
        path = self.frame_path(index)
        cv2 = _cv2()
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise VideoError(f"could not read proxy frame {path!r}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        """Number of proxy frames available (same as :attr:`frame_count`)."""
        return self.frame_count

    def __repr__(self) -> str:
        """Return a short debug representation naming the directory and size."""
        m = self.manifest
        return f"<ProxyStore directory={self._directory!r} {m.width}x{m.height} count={m.count}>"
