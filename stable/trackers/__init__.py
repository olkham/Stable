"""Tracker plugins: the contract, the registry, and multi-anchor merging.

Adding a tracker never means editing this package — publish a pip package that
advertises the ``stable.trackers`` entry-point group and Stable will find it.
See :mod:`stable.trackers.registry`.
"""

from __future__ import annotations

from .base import Direction, ProgressCallback, TrackerPlugin, noop_progress
from .chunking import Chunk, ChunkStitcher, chunk_weights, plan_chunks
from .confidence import (
    combine_scores,
    normalize_scores,
    sanitize_scores,
    visibility_from_confidence,
)
from .cotracker3 import CoTracker3Tracker
from .deep import BackendPrediction, DeepTrackerPlugin, ModelTrackerPlugin, TrackerBackend
from .dummy import DummyTracker
from .lk import LKTracker
from .merge import (
    MergeConfig,
    MergeReport,
    MergeResult,
    RunSpec,
    TrackRun,
    merge_anchor_runs,
    plan_anchor_runs,
)
from .registry import (
    ENTRY_POINT_GROUP,
    TrackerRegistry,
    available_trackers,
    create_tracker,
    describe_trackers,
    get_tracker,
    register_tracker,
    registry,
)
from .siamese import SiameseBackend, SiameseMaps, SiameseStep, SiameseTrackerPlugin
from .tapir import TapirTracker
from .template import TemplateTracker
from .weights import (
    ModelSpec,
    describe_models,
    ensure_weights,
    register_model,
    registered_models,
)

__all__ = [
    "TrackerPlugin",
    "Direction",
    "ProgressCallback",
    "noop_progress",
    "DummyTracker",
    "LKTracker",
    "TemplateTracker",
    "CoTracker3Tracker",
    "TapirTracker",
    # deep-tracker machinery (shared by every model plugin)
    "ModelTrackerPlugin",
    "DeepTrackerPlugin",
    "TrackerBackend",
    "BackendPrediction",
    "SiameseTrackerPlugin",
    "SiameseBackend",
    "SiameseMaps",
    "SiameseStep",
    "Chunk",
    "ChunkStitcher",
    "plan_chunks",
    "chunk_weights",
    "normalize_scores",
    "combine_scores",
    "sanitize_scores",
    "visibility_from_confidence",
    # model weights
    "ModelSpec",
    "ensure_weights",
    "register_model",
    "registered_models",
    "describe_models",
    "TrackerRegistry",
    "ENTRY_POINT_GROUP",
    "registry",
    "register_tracker",
    "get_tracker",
    "create_tracker",
    "available_trackers",
    "describe_trackers",
    "RunSpec",
    "TrackRun",
    "MergeConfig",
    "MergeReport",
    "MergeResult",
    "plan_anchor_runs",
    "merge_anchor_runs",
]
