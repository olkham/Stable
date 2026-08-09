"""Pydantic request bodies for the FastAPI routes.

Response bodies are plain ``dict`` (built in :mod:`stable.server.serialize` and
the route modules) rather than response models — the shapes mirror
:mod:`stable.core.project`'s own ``to_dict`` methods closely enough that a
second, parallel schema would only drift from it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

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
    "ClipRangeUpdate",
    "ProjectRenameRequest",
    "CorrectionIn",
    "CorrectionUpdate",
    "RetrackRequest",
    "ModelDownloadRequest",
    "TemplateOptionsUpdate",
]

Direction = Literal["fwd", "bwd", "bidir"]


class TemplateOptionsUpdate(BaseModel):
    """Body for ``PUT .../points/{id}/patches``: one point's tracker sizes.

    Every field is optional *and* nullable, and the two mean different things:
    omitted leaves the stored value alone, while an explicit ``null`` clears it
    so the tracker's own default applies again (todo 14). Range checking lives
    in :mod:`stable.server.tracker_options` beside the storage, not here, so the
    CLI and the HTTP layer cannot disagree about what is valid.

    ``template_radius`` and ``search_radius`` are named because they were the
    first two, but any argument in
    :data:`stable.server.tracker_options.SETTINGS` may be sent — LK's
    ``win_size``, mean-shift's ``window_radius``, and so on — under the plugin's
    own name. Hence ``extra="allow"``: the settable set is a property of the
    *tracker*, which this model cannot see, so the route checks the keys against
    the point's own plugin and rejects one it would not accept.
    """

    model_config = ConfigDict(extra="allow")

    template_radius: int | None = None
    search_radius: int | None = None


class PointCreate(BaseModel):
    """Body for ``POST /api/projects/{id}/points``: place a new point's first anchor."""

    frame: int
    x: float
    y: float
    tracker: str | None = None


class AnchorCreate(BaseModel):
    """Body for ``POST /api/projects/{id}/points/{point_id}/anchors``: add an anchor."""

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
    anchors before it is stopped there and the anchor reported as unreached;
    see :func:`stable.trackers.merge.enforce_anchor_agreement`.
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
    """Body for ``POST /api/projects/{id}/export/video``.

    ``filename`` defaults to ``None``, meaning "name it for me": the route
    stamps the time so two exports of the same project are two files. It used to
    default to ``stabilized.mp4``, which silently overwrote the previous render —
    a re-export with different settings destroyed the version you were comparing
    it against.
    """

    filename: str | None = None
    codec: str = "h264"
    crf: int = 18


class ClipRangeUpdate(BaseModel):
    """Body for ``PUT /api/projects/{id}/range``: the In/Out markers (todo 1).

    Both fields are optional *and* nullable, and the two differ: omitted leaves
    that marker where it is (dragging one handle), while an explicit ``null`` on
    both clears the range back to the whole clip. Range checking lives in
    :mod:`stable.server.clip_range` beside the storage.
    """

    in_: int | None = Field(default=None, alias="in")
    out: int | None = None

    model_config = ConfigDict(populate_by_name=True)


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

    A directional run from one frame, not a whole-clip one. ``x``/``y`` may be
    omitted to start from wherever the point already is at ``frame``, which is
    what the panel's "track on to the next anchor" buttons send.

    ``stop_at_keyframe`` is the previous name of ``stop_at_anchor`` and is still
    accepted: the two words were the same thing (todo 6, seventh round), and a
    browser tab left open across the rename must not start sending a bound the
    server silently ignores.
    """

    model_config = ConfigDict(populate_by_name=True)

    frame: int
    x: float | None = None
    y: float | None = None
    direction: Direction = "fwd"
    tracker: str | None = None
    device: str | None = None
    max_frames: int | None = Field(default=None, gt=0)
    stop_at_anchor: bool = Field(
        default=False,
        validation_alias=AliasChoices("stop_at_anchor", "stop_at_keyframe"),
    )
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
