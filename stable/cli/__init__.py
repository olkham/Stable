"""Headless command line: ``stable ...``.

Introspection commands (``trackers``, ``devices``, ``filters``) prove packaging
and plugin discovery work on a fresh install; ``track`` runs a tracker plugin,
``solve`` turns tracks into X/Y/R/S curves, and ``stabilize`` renders the
stabilised clip. Phase 6 adds ``serve``.
"""

from __future__ import annotations

from .main import main

__all__ = ["main"]
