"""The project file: schema v1, load/save, and the migration chain.

The on-disk form is exactly the JSON documented in ``PLAN.md``::

    { "version": 1,
      "video": { "path": "...", "fps": 0, "frames": 0, "w": 0, "h": 0 },
      "points": [ { "id": "p1", "queries": [{"frame": 0, "x": 0, "y": 0}],
                    "tracker": "cotracker3", "enabled_for_solve": true,
                    "raw": "tracks/p1.npz",
                    "corrections": [ ... ],
                    "color": "#5b9bd5" } ],
      "solve": { "reference_frame": 0, "channels": {"x": true, "y": true, "r": true, "s": true} },
      "smoothing": [ { "channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0} } ],
      "render": { "border": "auto_zoom", "codec": "h264", "crf": 18 } }

Rules
-----
* ``version`` is an integer that only ever increases. Reading a file older than
  :data:`SCHEMA_VERSION` runs it through :func:`migrate`; reading a newer one
  raises :class:`~stable.core.errors.SchemaVersionError` rather than guessing.
* Every path inside the file (``video.path``, ``points[].raw``,
  ``corrections[].source``) is stored **relative to the project file** when it
  is inside the project directory, so projects stay portable. Use
  :meth:`Project.resolve_path` to turn one back into an absolute path.
* Unknown keys are ignored on load — a v1 build must not crash on a file
  written by a slightly newer build of the same schema version.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .corrections import Correction, correction_from_dict
from .errors import ProjectError, SchemaVersionError
from .geometry import CHANNELS
from .types import Query

__all__ = [
    "SCHEMA_VERSION",
    "Project",
    "VideoRef",
    "PointSpec",
    "SolveSettings",
    "SmoothingSpec",
    "RenderSettings",
    "migrate",
    "register_migration",
    "MIGRATIONS",
    "BORDER_POLICIES",
]

SCHEMA_VERSION = 1
"""Schema version written by this build."""

BORDER_POLICIES: tuple[str, ...] = ("crop", "auto_zoom", "transparent")
"""Border handling policies for the stabilised render (Phase 3)."""

_HEX_COLOR = re.compile(r"#[0-9a-f]{6}")
"""The only colour form a point may carry: what an HTML colour input produces."""

Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""A function upgrading a project dict by exactly one schema version."""

MIGRATIONS: dict[int, Migration] = {}
"""Registered migrations, keyed by the version they upgrade *from*."""

_T = TypeVar("_T")

_SHARING_RETRY_DELAYS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2)
"""Seconds to wait between attempts on a file another handle is briefly holding.

Windows refuses to rename over a file that any other handle has open, and can
refuse to open one that is part-way through being replaced — both surface as
``PermissionError``. The server reads the project file from many threads at once
(request handlers, jobs, the frame stream once per frame), each for well under a
millisecond, so a save that collides with one of those reads only has to wait
for it. About 0.6 s in all: long enough to outlast any read, short enough that a
file that really is locked still fails promptly.
"""

_RETRY_SHARING_VIOLATIONS = os.name == "nt"
"""Whether a ``PermissionError`` on the project file can be transient here.

Only on Windows. Elsewhere a rename over an open file succeeds and a
``PermissionError`` means the permissions really are wrong, so retrying would
only delay the error.
"""


def _retry_sharing_violations(operation: Callable[[], _T], path: str) -> _T:
    """Run a file operation, retrying while another handle briefly holds the file.

    Args:
        operation: The open or rename to attempt.
        path: The file it acts on. Windows reports a directory there as
            ``PermissionError`` too, but that is a mistake rather than a
            moment's contention, so it fails at once instead of after every
            retry.

    Returns:
        Whatever ``operation`` returns, from the first attempt that succeeds.

    Raises:
        PermissionError: From the last attempt, if the file stays held.
    """
    if _RETRY_SHARING_VIOLATIONS and not os.path.isdir(path):
        for delay in _SHARING_RETRY_DELAYS:
            try:
                return operation()
            except PermissionError:
                time.sleep(delay)
    return operation()


def register_migration(from_version: int, fn: Migration) -> None:
    """Register a one-step schema migration.

    Args:
        from_version: The version the migration reads. It must produce
            ``from_version + 1``.
        fn: Callable taking and returning a project dict.

    Raises:
        ProjectError: If a migration is already registered for that version.
    """
    if from_version in MIGRATIONS:
        raise ProjectError(f"a migration from version {from_version} is already registered")
    MIGRATIONS[from_version] = fn


def migrate(data: dict[str, Any], *, target: int = SCHEMA_VERSION) -> dict[str, Any]:
    """Upgrade a raw project dict to ``target`` schema version.

    Args:
        data: Parsed project JSON. Not modified.
        target: Version to upgrade to; defaults to :data:`SCHEMA_VERSION`.

    Returns:
        A new dict at version ``target``.

    Raises:
        ProjectError: If ``version`` is missing or not an integer.
        SchemaVersionError: If the file is newer than ``target``, or no
            migration exists for some step along the way.
    """
    if "version" not in data:
        raise ProjectError("project file has no 'version' key")
    version = data["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectError(f"project 'version' must be an integer, got {version!r}")
    if version > target:
        raise SchemaVersionError(
            f"project file is version {version} but this build understands up to {target}; "
            "upgrade stable-tracker"
        )
    current = dict(data)
    while version < target:
        step = MIGRATIONS.get(version)
        if step is None:
            raise SchemaVersionError(
                f"no migration registered from schema version {version} to {version + 1}"
            )
        current = step(current)
        version += 1
        current["version"] = version
    return current


@dataclass
class VideoRef:
    """The clip a project is built on.

    Args:
        path: Path to the video file, relative to the project file when possible.
        fps: Frame rate.
        frames: Total frame count ``T``; also the length of every track array.
        w: Frame width in pixels.
        h: Frame height in pixels.
    """

    path: str = ""
    fps: float = 0.0
    frames: int = 0
    w: int = 0
    h: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file dict for this clip reference."""
        return {"path": self.path, "fps": self.fps, "frames": self.frames, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoRef:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        return cls(
            path=str(data.get("path", "")),
            fps=float(data.get("fps", 0.0)),
            frames=int(data.get("frames", 0)),
            w=int(data.get("w", 0)),
            h=int(data.get("h", 0)),
        )


@dataclass
class PointSpec:
    """One tracked point: its anchors, its tracker, its raw data and its edits.

    Args:
        id: Project-unique identifier (e.g. ``"p1"``).
        queries: The point's anchors. Several across the clip is the normal
            case — see the multi-anchor section of ``PLAN.md``.
        tracker: Registered tracker plugin name.
        enabled_for_solve: Whether the solver may use this point.
        raw: Path to the point's ``.npz`` raw track, relative to the project
            file; ``None`` until the point has been tracked.
        corrections: The non-destructive edit layer, applied in order.
        color: The colour the UI draws this point in, as ``#rrggbb``, or
            ``None`` to let it pick one from its palette. The palette has eight
            entries and a project may have more points than that, so two tracks
            can end up the same colour with no way to tell them apart on the
            frame; this is the operator's answer to that, and it belongs with
            the project rather than in one browser tab.
        derived_from: The ids this point's path was averaged from, or empty for
            an ordinary tracked point. A **composite** track is not a tracked
            one: nothing chose a tracker for it, no window was matched, no run
            produced it, and the anchor it carries is a formality. The UI has to
            know that to stop offering the controls for all of those (todo 2,
            tenth round, second pass), and it has to know it *after a reload* —
            which is why this is on the point rather than in the response that
            created it, where it started out.

            Stale ids are kept rather than pruned when a source is deleted: "made
            from p1 and p2" is what happened, and the path baked from them does
            not change when one of them goes. The panel draws a missing source as
            plain text instead of a link.
    """

    id: str
    queries: list[Query] = field(default_factory=list)
    tracker: str = "lk"
    enabled_for_solve: bool = True
    raw: str | None = None
    corrections: list[Correction] = field(default_factory=list)
    color: str | None = None
    derived_from: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalise and validate :attr:`color` (dataclass hook).

        Raises:
            ProjectError: If the colour is not ``#rrggbb``.
        """
        if self.color is None:
            return
        color = str(self.color).strip().lower()
        if not _HEX_COLOR.fullmatch(color):
            raise ProjectError(f"point colour must be #rrggbb, got {self.color!r}")
        self.color = color

    @property
    def is_box(self) -> bool:
        """Whether this is a **box track**: one whose anchors state boxes.

        Position is still the box centre; the box adds a per-frame size to the
        track and needs a tracker with ``tracks_boxes``.
        """
        return any(q.has_box for q in self.queries)

    @property
    def is_composite(self) -> bool:
        """Whether this point's path was averaged from other tracks."""
        return bool(self.derived_from)

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file dict for this point."""
        return {
            "id": self.id,
            "queries": [q.to_dict() for q in self.queries],
            "tracker": self.tracker,
            "enabled_for_solve": self.enabled_for_solve,
            "raw": self.raw,
            "corrections": [c.to_dict() for c in self.corrections],
            "color": self.color,
            "derived_from": list(self.derived_from),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointSpec:
        """Rebuild from :meth:`to_dict` output.

        Raises:
            ProjectError: If the point has no ``id``.
        """
        if "id" not in data:
            raise ProjectError("point entry is missing 'id'")
        return cls(
            id=str(data["id"]),
            queries=[Query.from_dict(q) for q in data.get("queries", [])],
            tracker=str(data.get("tracker", "lk")),
            enabled_for_solve=bool(data.get("enabled_for_solve", True)),
            raw=data.get("raw"),
            corrections=[correction_from_dict(c) for c in data.get("corrections", [])],
            # Absent on every project written before colours existed, which is
            # the same as "use the palette" — no migration needed.
            color=data.get("color") or None,
            # Likewise: absent means "an ordinary tracked point", which is what
            # every point written before composites existed was.
            derived_from=[str(source) for source in data.get("derived_from") or []],
        )

    def sorted_queries(self) -> list[Query]:
        """Return the anchors ordered by frame (the order merging expects)."""
        return sorted(self.queries, key=lambda q: q.frame)


@dataclass
class SolveSettings:
    """How the motion solver turns tracks into ``X, Y, R, S`` curves.

    Args:
        reference_frame: Frame the solve is relative to (identity transform there).
        channels: Which channels are solved/applied, keyed by ``x``, ``y``, ``r``, ``s``.
    """

    reference_frame: int = 0
    channels: dict[str, bool] = field(
        default_factory=lambda: {"x": True, "y": True, "r": True, "s": True}
    )

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file dict for the solve settings."""
        return {
            "reference_frame": self.reference_frame,
            "channels": {c: bool(self.channels.get(c, True)) for c in CHANNELS},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolveSettings:
        """Rebuild from :meth:`to_dict` output; missing channels default to on."""
        raw = data.get("channels", {}) or {}
        return cls(
            reference_frame=int(data.get("reference_frame", 0)),
            channels={c: bool(raw.get(c, True)) for c in CHANNELS},
        )


@dataclass
class SmoothingSpec:
    """One smoothing filter applied to one channel.

    The list is ordered; filters compose in order on the same channel.

    Args:
        channel: One of ``x``, ``y``, ``r``, ``s``.
        filter: Filter name (``moving_average``, ``gaussian``, ``butterworth``,
            ``savgol``, ``spline``, ``tripod``) — the catalogue lands in Phase 3.
        params: Filter-specific parameters, e.g. ``{"cutoff_hz": 1.0}``.
        enabled: Whether the filter is active.
    """

    channel: str
    filter: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the channel name (dataclass hook)."""
        if self.channel not in CHANNELS:
            raise ProjectError(f"unknown smoothing channel {self.channel!r}; expected {CHANNELS}")

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file dict for this filter."""
        return {
            "channel": self.channel,
            "filter": self.filter,
            "params": dict(self.params),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmoothingSpec:
        """Rebuild from :meth:`to_dict` output.

        Raises:
            ProjectError: If ``channel`` or ``filter`` is missing.
        """
        try:
            return cls(
                channel=str(data["channel"]),
                filter=str(data["filter"]),
                params=dict(data.get("params", {})),
                enabled=bool(data.get("enabled", True)),
            )
        except KeyError as exc:
            raise ProjectError(f"smoothing entry is missing key {exc.args[0]!r}") from exc


@dataclass
class RenderSettings:
    """Export settings for the stabilised render.

    Args:
        border: One of :data:`BORDER_POLICIES`.
        codec: FFmpeg codec name.
        crf: Constant rate factor (quality); lower is better.
    """

    border: str = "auto_zoom"
    codec: str = "h264"
    crf: int = 18

    def __post_init__(self) -> None:
        """Validate the border policy (dataclass hook)."""
        if self.border not in BORDER_POLICIES:
            raise ProjectError(f"unknown border policy {self.border!r}; expected {BORDER_POLICIES}")

    def to_dict(self) -> dict[str, Any]:
        """Return the project-file dict for the render settings."""
        return {"border": self.border, "codec": self.codec, "crf": self.crf}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderSettings:
        """Rebuild from :meth:`to_dict` output; missing keys fall back to defaults."""
        return cls(
            border=str(data.get("border", "auto_zoom")),
            codec=str(data.get("codec", "h264")),
            crf=int(data.get("crf", 18)),
        )


@dataclass
class Project:
    """An in-memory Stable project — the root of the versioned schema.

    Args:
        video: The clip being tracked.
        points: Tracked points, each with anchors, raw data and corrections.
        solve: Motion solver settings.
        smoothing: Ordered list of per-channel smoothing filters.
        render: Export settings.
        version: Schema version; always :data:`SCHEMA_VERSION` for new projects.
        path: Absolute path of the file this project was loaded from or last
            saved to. Not serialised; it anchors relative path resolution.
    """

    video: VideoRef = field(default_factory=VideoRef)
    points: list[PointSpec] = field(default_factory=list)
    solve: SolveSettings = field(default_factory=SolveSettings)
    smoothing: list[SmoothingSpec] = field(default_factory=list)
    render: RenderSettings = field(default_factory=RenderSettings)
    version: int = SCHEMA_VERSION
    path: str | None = field(default=None, compare=False)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the complete project-file dict (JSON-safe)."""
        return {
            "version": self.version,
            "video": self.video.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "solve": self.solve.to_dict(),
            "smoothing": [s.to_dict() for s in self.smoothing],
            "render": self.render.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, path: str | None = None) -> Project:
        """Rebuild a project from a raw dict, migrating it if necessary.

        Args:
            data: Parsed project JSON at any supported schema version.
            path: Absolute path the dict came from, used for relative paths.

        Returns:
            The migrated, validated :class:`Project`.

        Raises:
            ProjectError: If the structure is invalid.
            SchemaVersionError: If the version cannot be migrated to current.
        """
        if not isinstance(data, dict):
            raise ProjectError(f"project must be a JSON object, got {type(data).__name__}")
        data = migrate(data)
        project = cls(
            video=VideoRef.from_dict(data.get("video", {}) or {}),
            points=[PointSpec.from_dict(p) for p in data.get("points", [])],
            solve=SolveSettings.from_dict(data.get("solve", {}) or {}),
            smoothing=[SmoothingSpec.from_dict(s) for s in data.get("smoothing", [])],
            render=RenderSettings.from_dict(data.get("render", {}) or {}),
            version=SCHEMA_VERSION,
            path=path,
        )
        project.validate()
        return project

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the project serialised as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str, *, path: str | None = None) -> Project:
        """Parse a project from a JSON string.

        Raises:
            ProjectError: If the text is not valid JSON.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectError(f"invalid project JSON: {exc}") from exc
        return cls.from_dict(data, path=path)

    def save(self, path: str | os.PathLike[str] | None = None) -> str:
        """Write the project to disk atomically.

        The file is written to a temporary sibling and then moved into place, so
        an interrupted save can never truncate an existing project. On Windows
        the move waits out a reader that has the file open at that instant
        rather than failing with "Access is denied" (see
        :data:`_SHARING_RETRY_DELAYS`).

        Args:
            path: Destination; defaults to :attr:`path` (the file it came from).

        Returns:
            The absolute path written.

        Raises:
            ProjectError: If no path is available or the project is invalid.
        """
        target = os.path.abspath(os.fspath(path)) if path is not None else self.path
        if not target:
            raise ProjectError("no path given and this project has no path to save back to")
        self.validate()
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())
            handle.write("\n")
        _retry_sharing_violations(lambda: os.replace(tmp, target), target)
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Project:
        """Read, migrate and validate a project file.

        Args:
            path: Path to a ``.stable.json`` project file.

        Returns:
            The loaded :class:`Project`, with :attr:`path` set.

        Raises:
            ProjectError: If the file is missing or malformed.
            SchemaVersionError: If it is too new to read.
        """
        target = os.path.abspath(os.fspath(path))

        def read() -> str:
            with open(target, encoding="utf-8") as handle:
                return handle.read()

        try:
            # A read that lands while a save is replacing the file can be
            # refused on Windows; it is a moment, not a missing project.
            text = _retry_sharing_violations(read, target)
        except OSError as exc:
            raise ProjectError(f"cannot read project {target}: {exc}") from exc
        return cls.from_json(text, path=target)

    # -- helpers ---------------------------------------------------------

    @property
    def directory(self) -> str:
        """Directory the project file lives in; the CWD if it was never saved."""
        return os.path.dirname(self.path) if self.path else os.getcwd()

    def resolve_path(self, relative: str | None) -> str | None:
        """Turn a project-relative path into an absolute one.

        Args:
            relative: A path from the project file, or ``None``.

        Returns:
            The absolute path, or ``None`` if ``relative`` was ``None``.
            Already-absolute inputs are returned unchanged.
        """
        if relative is None:
            return None
        if os.path.isabs(relative):
            return relative
        return os.path.normpath(os.path.join(self.directory, relative))

    def relative_path(self, absolute: str) -> str:
        """Turn an absolute path into a project-relative one where possible.

        Args:
            absolute: Any filesystem path.

        Returns:
            A path relative to the project directory, using forward slashes;
            the original absolute path if it lives on another drive or above the
            project directory.
        """
        try:
            rel = os.path.relpath(absolute, self.directory)
        except ValueError:
            return absolute
        if rel.startswith(".."):
            return absolute
        return rel.replace(os.sep, "/")

    def point(self, point_id: str) -> PointSpec:
        """Return the point with the given id.

        Raises:
            ProjectError: If no such point exists.
        """
        for spec in self.points:
            if spec.id == point_id:
                return spec
        raise ProjectError(f"no point with id {point_id!r}")

    def new_point_id(self) -> str:
        """Return a ``pN`` identifier that this project has not used before.

        Deliberately *not* the lowest free number. Point ids key files on disk
        (``tracks/p1.npz``, its report, its retrack runs) and caches in the
        browser, so recycling one meant a newly placed point inherited the dead
        one's track — which is exactly how it was reported: "when I delete a
        point and add a new one back it restores the previous track".

        Deleting every artefact is the real fix (see
        :func:`stable.server.routes.points.delete_point`); counting past the
        highest id in use is the belt to its braces, and it costs nothing — the
        first point is still ``p1``, and the numbering only skips where
        something *was*. The counter is derived rather than stored, so emptying
        a project does start again at ``p1``; that is safe precisely because
        deleting a point now takes its files with it.
        """
        used = {p.id for p in self.points}
        index = max((self._point_number(p.id) for p in self.points), default=0) + 1
        while f"p{index}" in used:  # pragma: no cover - only a hand-edited file
            index += 1
        return f"p{index}"

    @staticmethod
    def _point_number(point_id: str) -> int:
        """Return the ``N`` in a ``pN`` id, or 0 for an id in any other shape."""
        if not point_id.startswith("p"):
            return 0
        try:
            return int(point_id[1:])
        except ValueError:
            return 0

    def validate(self) -> None:
        """Check cross-field invariants that per-field validation cannot see.

        Raises:
            ProjectError: On duplicate point ids, a negative reference frame, an
                anchor beyond the end of the clip, or a bad schema version.
        """
        if self.version != SCHEMA_VERSION:
            raise ProjectError(
                f"in-memory project must be at version {SCHEMA_VERSION}, got {self.version}"
            )
        seen: set[str] = set()
        for spec in self.points:
            if spec.id in seen:
                raise ProjectError(f"duplicate point id {spec.id!r}")
            seen.add(spec.id)
            if self.video.frames:
                for query in spec.queries:
                    if query.frame >= self.video.frames:
                        raise ProjectError(
                            f"point {spec.id!r} has an anchor at frame {query.frame} "
                            f"but the clip has {self.video.frames} frames"
                        )
        if self.solve.reference_frame < 0:
            raise ProjectError("solve.reference_frame must be >= 0")
        if self.video.frames and self.solve.reference_frame >= self.video.frames:
            raise ProjectError(
                f"solve.reference_frame {self.solve.reference_frame} is past the end of the clip"
            )
