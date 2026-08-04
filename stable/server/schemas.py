"""Pydantic request bodies for the FastAPI routes.

Response bodies are plain ``dict`` (built in :mod:`stable.server.serialize` and
the route modules) rather than response models — the shapes mirror
:mod:`stable.core.project`'s own ``to_dict`` methods closely enough that a
second, parallel schema would only drift from it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..trackers.merge import DEFAULT_ANCHOR_TOLERANCE_PX

__all__ = [
    "PointCreate",
    "AnchorCreate",
    "PointUpdate",
    "TrackRequest",
    "SolveSettingsUpdate",
    "SmoothingFilterIn",
    "SmoothingUpdate",
    "RenderSettingsUpdate",
    "ExportVideoRequest",
    "ProjectRenameRequest",
    "CorrectionIn",
    "CorrectionUpdate",
    "RetrackRequest",
    "ModelDownloadRequest",
]

Direction = Literal["fwd", "bwd", "bidir"]


class PointCreate(BaseModel):
    """Body for ``POST /api/projects/{id}/points``: place a new point's first anchor."""

    frame: int
    x: float
    y: float
    tracker: str | None = None


class AnchorCreate(BaseModel):
    """Body for ``POST /api/projects/{id}/points/{point_id}/anchors``: add an anchor pin."""

    frame: int
    x: float
    y: float


class PointUpdate(BaseModel):
    """Body for ``PATCH /api/projects/{id}/points/{point_id}``."""

    tracker: str | None = None
    enabled_for_solve: bool | None = None


class TrackRequest(BaseModel):
    """Body for ``POST /api/projects/{id}/points/{point_id}/track``.

    ``anchor_tolerance_px`` is how far a run may land from one of the point's
    keyframes before it is stopped there and the keyframe reported as
    unreached; see :func:`stable.trackers.merge.enforce_anchor_agreement`.
    """

    tracker: str | None = None
    direction: Direction = "bidir"
    device: str | None = None
    anchor_tolerance_px: float = Field(default=DEFAULT_ANCHOR_TOLERANCE_PX, gt=0)


class SolveSettingsUpdate(BaseModel):
    """Body for ``PUT /api/projects/{id}/solve/settings``."""

    reference_frame: int | None = None
    channels: dict[str, bool] | None = None


class SmoothingFilterIn(BaseModel):
    """One entry of a ``PUT /api/projects/{id}/smoothing`` filter list."""

    channel: Literal["x", "y", "r", "s"]
    filter: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SmoothingUpdate(BaseModel):
    """Body for ``PUT /api/projects/{id}/smoothing``: the whole ordered filter list."""

    filters: list[SmoothingFilterIn] = Field(default_factory=list)


class RenderSettingsUpdate(BaseModel):
    """Body for ``PUT /api/projects/{id}/render``."""

    border: str | None = None
    codec: str | None = None
    crf: int | None = None


class ExportVideoRequest(BaseModel):
    """Body for ``POST /api/projects/{id}/export/video``."""

    filename: str = "stabilized.mp4"
    codec: str = "h264"
    crf: int = 18


class ProjectRenameRequest(BaseModel):
    """Body for ``PATCH /api/projects/{id}``."""

    name: str


class CorrectionIn(BaseModel):
    """Body for ``POST /api/projects/{id}/points/{point_id}/corrections``.

    Deliberately open: the body *is* a correction's project-file form, so the
    typed validation lives in :func:`stable.core.corrections.correction_from_dict`
    — one definition of what an edit is, rather than a Pydantic mirror of it that
    can drift. Only ``kind`` is required here; everything else is that type's own
    fields, checked by the dataclass that will store them.
    """

    model_config = ConfigDict(extra="allow")

    kind: str

    def payload(self) -> dict[str, Any]:
        """Return the dict to hand to ``correction_from_dict``."""
        return self.model_dump()


class CorrectionUpdate(BaseModel):
    """Body for ``PATCH .../corrections/{correction_id}``: toggle an edit on or off."""

    enabled: bool | None = None


class RetrackRequest(BaseModel):
    """Body for ``POST /api/projects/{id}/points/{point_id}/retrack``.

    A retrack is a *bounded* run, not a whole-clip one. ``x``/``y`` may be
    omitted to start from wherever the point already is at ``frame``, which is
    what the panel's "track on to the next keyframe" buttons send.
    """

    frame: int
    x: float | None = None
    y: float | None = None
    direction: Direction = "fwd"
    tracker: str | None = None
    device: str | None = None
    max_frames: int | None = Field(default=None, gt=0)
    stop_at_keyframe: bool = False
    anchor_tolerance_px: float = Field(default=DEFAULT_ANCHOR_TOLERANCE_PX, gt=0)


class ModelDownloadRequest(BaseModel):
    """Body for ``POST /api/models/{key}/download``.

    Both flags default to False: a checkpoint with no pinned checksum, or one
    licensed non-commercial, fails closed (see
    :mod:`stable.trackers.weights`) unless explicitly accepted here — a
    checked box in the UI, exactly like the same-named ``ensure_weights``
    arguments, rather than a process-wide environment variable.
    """

    accept_noncommercial: bool = False
    allow_unpinned: bool = False
