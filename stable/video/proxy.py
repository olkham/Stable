"""Half/quarter-resolution JPEG proxy generation and disk-backed serving.

A UI scrubbing a 1080p timeline should not decode full-resolution video on
every frame move. The pattern here is a one-time cost, then a cheap one:

#. :func:`generate_proxies` decodes the source clip **once**, resizes every
   frame, and writes it to disk as a JPEG. This is the only step that touches
   the real decoder.
#. :class:`ProxyStore` serves frames from those JPEGs. It has no reference to
   the source video at all — structurally, it cannot decode anything — so
   scrubbing a cached clip is disk I/O, never a video decode.
"""

from __future__ import annotations

import json
import os
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
]

PROXY_SCALES: dict[str, float] = {"half": 0.5, "quarter": 0.25}
"""Named proxy resolution tiers Stable generates by convention."""

MANIFEST_FILENAME = "proxy.json"
"""Name of the JSON sidecar file written next to a tier's proxy frames."""


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


def generate_proxies(
    video: VideoSource,
    out_dir: str | os.PathLike[str],
    *,
    scale: float = 0.5,
    quality: int = 90,
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
        progress: Progress reporter, called after every frame.

    Returns:
        The :class:`ProxyManifest` describing what was written (also saved to
        ``out_dir/proxy.json`` so a later :class:`ProxyStore` can load it).

    Raises:
        VideoError: If OpenCV is unavailable, a frame fails to write, or
            ``scale``/``quality`` is out of range.
    """
    if not 0.0 < scale <= 1.0:
        raise VideoError(f"scale must lie in (0, 1], got {scale}")
    if not 0 <= quality <= 100:
        raise VideoError(f"quality must lie in [0, 100], got {quality}")
    cv2 = _cv2()
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    width = max(1, round(video.width * scale))
    height = max(1, round(video.height * scale))
    total = video.frame_count
    same_size = (width, height) == (video.width, video.height)

    for index, frame in video.iter_frames():
        if same_size:
            resized = frame
        else:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        path = os.path.join(out_dir, _frame_filename(index))
        if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise VideoError(f"failed to write proxy frame {index} to {path!r}")
        progress((index + 1) / total if total else 1.0, f"proxy {index + 1}/{total}")

    manifest = ProxyManifest(
        count=total, width=width, height=height, scale=scale, fps=video.fps, quality=quality
    )
    with open(os.path.join(out_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
    return manifest


def generate_all_proxies(
    video: VideoSource,
    base_dir: str | os.PathLike[str],
    *,
    tiers: dict[str, float] | None = None,
    quality: int = 90,
    progress: ProgressCallback = noop_progress,
) -> dict[str, ProxyManifest]:
    """Generate every named proxy tier into its own subdirectory of ``base_dir``.

    Args:
        video: Source clip to decode.
        base_dir: Parent directory; each tier is written to ``base_dir/<name>``.
        tiers: Mapping of tier name to scale; defaults to :data:`PROXY_SCALES`
            (``half`` and ``quarter``).
        quality: JPEG quality passed to :func:`generate_proxies` for every tier.
        progress: Progress reporter; called with the overall fraction across
            all tiers combined.

    Returns:
        Mapping of tier name to its :class:`ProxyManifest`.
    """
    tiers = PROXY_SCALES if tiers is None else tiers
    base_dir = os.fspath(base_dir)
    manifests: dict[str, ProxyManifest] = {}
    names = list(tiers)
    for tier_index, name in enumerate(names):

        def _tier_progress(
            fraction: float, message: str = "", _i: int = tier_index, _name: str = name
        ) -> None:
            progress((_i + fraction) / len(names), f"{_name}: {message}")

        manifests[name] = generate_proxies(
            video,
            os.path.join(base_dir, name),
            scale=tiers[name],
            quality=quality,
            progress=_tier_progress,
        )
    return manifests


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
