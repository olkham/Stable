"""Core contracts: data model, project file, geometry and device selection.

Everything in this package is dependency-light (NumPy only) and is imported by
every other subsystem. Changing anything here is a contract change and must be
reflected in ``PLAN.md``.
"""

from __future__ import annotations

from .corrections import (
    CORRECTION_TYPES,
    Correction,
    DeleteRangeCorrection,
    GapFillCorrection,
    OffsetCorrection,
    ReplaceCorrection,
    RetrackCorrection,
    apply_corrections,
    correction_from_dict,
)
from .device import available_devices, device_info, select_device
from .errors import (
    ContractError,
    DeviceError,
    ProjectError,
    SchemaVersionError,
    StableError,
    TrackerError,
    TrackerNotFoundError,
    VideoError,
)
from .geometry import CHANNELS, SimilarityTransform
from .project import (
    SCHEMA_VERSION,
    PointSpec,
    Project,
    RenderSettings,
    SmoothingSpec,
    SolveSettings,
    VideoRef,
    migrate,
    register_migration,
)
from .types import PointTrack, Query, TrackResult
from .video import SyntheticVideoSource, VideoSource

__all__ = [
    "Query",
    "TrackResult",
    "PointTrack",
    "SimilarityTransform",
    "CHANNELS",
    "VideoSource",
    "SyntheticVideoSource",
    "Correction",
    "OffsetCorrection",
    "ReplaceCorrection",
    "GapFillCorrection",
    "DeleteRangeCorrection",
    "RetrackCorrection",
    "CORRECTION_TYPES",
    "correction_from_dict",
    "apply_corrections",
    "Project",
    "PointSpec",
    "VideoRef",
    "SolveSettings",
    "SmoothingSpec",
    "RenderSettings",
    "SCHEMA_VERSION",
    "migrate",
    "register_migration",
    "select_device",
    "available_devices",
    "device_info",
    "StableError",
    "ContractError",
    "ProjectError",
    "SchemaVersionError",
    "TrackerError",
    "TrackerNotFoundError",
    "DeviceError",
    "VideoError",
]
