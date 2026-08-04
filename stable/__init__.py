"""Stable — open-source point tracking and 2D stabilization.

Track user-defined points with swappable deep-learning or classic trackers,
solve X/Y/Rotation/Scale motion, edit the tracks non-destructively, and export
tracking data or a stabilized video.

This top level re-exports the Phase 0 contracts every other subsystem builds on::

    from stable import Query, TrackResult, PointTrack, SimilarityTransform
    from stable import Project, TrackerPlugin, get_tracker

Subpackages
-----------
``core``       data model, project file, geometry, device selection
``video``      decode, frame cache, proxies, encode (Phase 1)
``trackers``   plugin registry, multi-anchor merging, implementations
``solver``     tracks to per-frame similarity, with RANSAC
``smoothing``  filters over the X, Y, R, S curves
``render``     inverse-warp stabilization and export
``server``     FastAPI app, job queue, websocket progress (Phase 4)
``cli``        headless track / solve / stabilize / export
"""

from __future__ import annotations

from importlib import metadata

from .core.corrections import (
    Correction,
    DeleteRangeCorrection,
    GapFillCorrection,
    OffsetCorrection,
    ReplaceCorrection,
    RetrackCorrection,
    apply_corrections,
)
from .core.device import available_devices, device_info, select_device
from .core.errors import (
    ContractError,
    DeviceError,
    ProjectError,
    RenderError,
    SchemaVersionError,
    SmoothingError,
    SolverError,
    StableError,
    TrackerError,
    TrackerNotFoundError,
    VideoError,
    WeightsError,
)
from .core.geometry import CHANNELS, SimilarityTransform
from .core.project import SCHEMA_VERSION, PointSpec, Project
from .core.types import PointTrack, Query, TrackResult
from .core.video import VideoSource
from .render import (
    OutputGeometry,
    StabilizationPlan,
    coverage_scale,
    iter_stabilized_frames,
    plan_output,
    render_stabilized,
    stabilize,
    warp_frame,
)
from .smoothing import (
    ChannelSmoothing,
    SmoothedMotion,
    SmoothingSettings,
    describe_filters,
    smooth_curve,
    smooth_motion,
)
from .solver import (
    MotionSolve,
    RansacConfig,
    SolveConfig,
    SolveConstraints,
    SolveStatus,
    solve_motion,
    solve_similarity,
    solve_similarity_ransac,
)
from .trackers.base import TrackerPlugin
from .trackers.cotracker3 import CoTracker3Tracker
from .trackers.deep import DeepTrackerPlugin, TrackerBackend
from .trackers.lk import LKTracker
from .trackers.merge import (
    MergeConfig,
    MergeReport,
    MergeResult,
    RunSpec,
    TrackRun,
    merge_anchor_runs,
    plan_anchor_runs,
)
from .trackers.registry import (
    available_trackers,
    create_tracker,
    describe_trackers,
    get_tracker,
    register_tracker,
    registry,
)
from .trackers.tapir import TapirTracker
from .trackers.template import TemplateTracker
from .trackers.weights import ModelSpec, describe_models, ensure_weights, registered_models
from .video import (
    CODECS,
    PROXY_SCALES,
    ProxyManifest,
    ProxyStore,
    PyAVVideoSource,
    export_video,
    generate_all_proxies,
    generate_proxies,
)

try:
    __version__ = metadata.version("stable-tracker")
except metadata.PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # core types
    "Query",
    "TrackResult",
    "PointTrack",
    "SimilarityTransform",
    "CHANNELS",
    "VideoSource",
    # corrections
    "Correction",
    "OffsetCorrection",
    "ReplaceCorrection",
    "GapFillCorrection",
    "DeleteRangeCorrection",
    "RetrackCorrection",
    "apply_corrections",
    # project
    "Project",
    "PointSpec",
    "SCHEMA_VERSION",
    # trackers
    "TrackerPlugin",
    "LKTracker",
    "TemplateTracker",
    "CoTracker3Tracker",
    "TapirTracker",
    "DeepTrackerPlugin",
    "TrackerBackend",
    "registry",
    "register_tracker",
    "get_tracker",
    "create_tracker",
    "available_trackers",
    "describe_trackers",
    # multi-anchor merging
    "RunSpec",
    "TrackRun",
    "MergeConfig",
    "MergeReport",
    "MergeResult",
    "plan_anchor_runs",
    "merge_anchor_runs",
    # model weights (Phase 2)
    "ModelSpec",
    "ensure_weights",
    "registered_models",
    "describe_models",
    # video I/O (Phase 1)
    "PyAVVideoSource",
    "CODECS",
    "export_video",
    "PROXY_SCALES",
    "ProxyManifest",
    "ProxyStore",
    "generate_proxies",
    "generate_all_proxies",
    # solver (Phase 3)
    "SolveConfig",
    "SolveConstraints",
    "SolveStatus",
    "MotionSolve",
    "RansacConfig",
    "solve_motion",
    "solve_similarity",
    "solve_similarity_ransac",
    # smoothing (Phase 3)
    "SmoothingSettings",
    "ChannelSmoothing",
    "SmoothedMotion",
    "smooth_motion",
    "smooth_curve",
    "describe_filters",
    # render (Phase 3)
    "StabilizationPlan",
    "OutputGeometry",
    "plan_output",
    "coverage_scale",
    "warp_frame",
    "iter_stabilized_frames",
    "render_stabilized",
    "stabilize",
    # device
    "select_device",
    "available_devices",
    "device_info",
    # errors
    "StableError",
    "ContractError",
    "ProjectError",
    "SchemaVersionError",
    "TrackerError",
    "TrackerNotFoundError",
    "WeightsError",
    "DeviceError",
    "VideoError",
    "SolverError",
    "SmoothingError",
    "RenderError",
]
