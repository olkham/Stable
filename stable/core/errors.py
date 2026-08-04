"""Exception types shared by every Stable subsystem.

All errors raised deliberately by Stable derive from :class:`StableError`, so a
host application (CLI, server, plugin) can catch that one base class and be sure
it caught "a Stable problem" rather than an arbitrary Python failure.
"""

from __future__ import annotations

__all__ = [
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


class StableError(Exception):
    """Base class for every error raised on purpose by Stable."""


class ContractError(StableError):
    """A core data contract was violated (bad array shape, dtype, or range).

    Raised by the validating constructors in :mod:`stable.core.types` and
    :mod:`stable.core.geometry`. Seeing this almost always means a tracker
    plugin returned something that does not match the documented contract.
    """


class ProjectError(StableError):
    """A project file could not be read, written, or interpreted."""


class SchemaVersionError(ProjectError):
    """A project file declares a schema version this build cannot handle.

    Raised when the file is newer than :data:`stable.core.project.SCHEMA_VERSION`
    or when no migration path exists from the file's version to the current one.
    """


class TrackerError(StableError):
    """A tracker plugin failed, or was used in a way its contract forbids."""


class TrackerNotFoundError(TrackerError):
    """No tracker plugin is registered under the requested name."""


class WeightsError(TrackerError):
    """Model weights could not be located, downloaded, or verified.

    Raised by :mod:`stable.trackers.weights` — a failed download, a checksum
    mismatch, an unpinned checkpoint without an explicit opt-in, or a
    non-commercial checkpoint the user has not accepted the licence for. It is a
    :class:`TrackerError` because from a caller's point of view "the model is
    not usable" is the same failure whether the weights or the code were the
    problem.
    """


class DeviceError(StableError):
    """A compute device was requested explicitly but is not available."""


class VideoError(StableError):
    """Video decode, proxy generation, or encode/export failed.

    Raised by :mod:`stable.video` (the Phase 1 PyAV/OpenCV-backed implementation
    of the :class:`~stable.core.video.VideoSource` contract) and by the classic
    CV tracker plugins when an optional dependency (``av``, ``opencv-python``)
    required for real video I/O is missing.
    """


class SolverError(StableError):
    """The motion solver could not turn the given tracks into a transform.

    Raised by :mod:`stable.solver` for inputs it cannot fit at all — no
    correspondences, malformed point arrays, non-finite weights, or a reference
    frame in which not a single point is usable. Frames that merely *degrade*
    (one point, low confidence, no usable point at all) never raise: they are
    reported through :class:`~stable.solver.motion.SolveStatus` instead.
    """


class SmoothingError(StableError):
    """A smoothing filter was misconfigured or cannot run on this curve.

    Raised by :mod:`stable.smoothing` for an unknown filter name, an unknown or
    out-of-range parameter, or a curve too short for the requested filter.
    """


class RenderError(StableError):
    """The stabilised render could not be planned or written.

    Raised by :mod:`stable.render` when the warped frames share no usable common
    area, when a border policy is incompatible with the requested output, or
    when OpenCV is missing.
    """
