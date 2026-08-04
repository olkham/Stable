"""Video I/O: PyAV decode, frame cache, proxy generation, FFmpeg export.

Phase 1 fills this in against the contract fixed in Phase 0:
:class:`~stable.core.video.VideoSource`. Nothing else in the codebase may
import PyAV or OpenCV directly — trackers, the solver and the server all go
through :class:`~stable.core.video.VideoSource` and the helpers re-exported
here.

``av`` and ``opencv-python`` (the ``video`` extra) are imported lazily inside
each function/class, so merely importing this package never requires them —
only calling into the real decode/encode/proxy paths does.
"""

from __future__ import annotations

from .encode import CODECS, export_video
from .proxy import PROXY_SCALES, ProxyManifest, ProxyStore, generate_all_proxies, generate_proxies
from .source import PyAVVideoSource

__all__ = [
    "PyAVVideoSource",
    "CODECS",
    "export_video",
    "PROXY_SCALES",
    "ProxyManifest",
    "ProxyStore",
    "generate_proxies",
    "generate_all_proxies",
]
