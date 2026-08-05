"""On-disk project storage: create/list/load/save/delete, plus runtime caches.

Layout on disk, under :func:`stable.server.config.workspace_root`::

    <root>/<project_id>/
      project.stable.json   # stable.core.project.Project — the versioned schema
      meta.json              # server-only bookkeeping (name, created_at, ...)
      video/source.<ext>     # the uploaded clip, copied so the project is self-contained
      proxies/<tier>/...     # stable.video.proxy.generate_proxies output
      tracks/<point_id>.npz  # stable.core.types.PointTrack.save_npz output
      tracks/<point_id>.report.json  # merge diagnostics (confidence heat-strip source)
      tracks/<point_id>.<correction_id>.npz  # a "retrack from here" run (Phase 5)
      solve.json              # cached raw MotionSolve (server.motion output)
      smoothing.json          # cached smoothed curves (server.motion output)
      history.json            # undo/redo stack over the corrections layer
      renders/<file>.mp4      # exported stabilized video

A single process-wide lock serialises every read-modify-write of a project's
JSON file. This is a local, single-user tool — one lock is simpler and safer
than per-project locking, and the operations it guards (touching a small JSON
file, not video frames) are fast enough that it is never a bottleneck.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.errors import ProjectError, StableError
from ..core.project import Project
from ..video.proxy import ProxyStore
from ..video.source import PyAVVideoSource
from .config import workspace_root

__all__ = ["ProjectNotFoundError", "ProjectMeta", "Workspace"]

PROJECT_FILENAME = "project.stable.json"
META_FILENAME = "meta.json"


class ProjectNotFoundError(StableError):
    """No project is stored under the given id."""


@dataclass
class ProjectMeta:
    """Server-only bookkeeping that is not part of the versioned project schema.

    Args:
        id: Short opaque identifier, also the project's directory name.
        name: Human-readable project name shown in the UI.
        created_at: Unix timestamp the project was created.
        original_filename: The uploaded video's original filename, for display.
    """

    id: str
    name: str
    created_at: float
    original_filename: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict written as ``meta.json``."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "original_filename": self.original_filename,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMeta:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            created_at=float(data.get("created_at", 0.0)),
            original_filename=str(data.get("original_filename", "")),
        )


class Workspace:
    """Owns every project directory and the runtime objects opened against it.

    Args:
        root: Directory projects are stored under; defaults to
            :func:`stable.server.config.workspace_root`.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Create a workspace rooted at ``root`` (or the default), creating it."""
        self.root = root or workspace_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._video_cache: dict[str, PyAVVideoSource] = {}
        self._proxy_cache: dict[tuple[str, str], ProxyStore] = {}

    # -- paths -------------------------------------------------------------

    def project_dir(self, project_id: str) -> Path:
        """Return the directory a project's files live in."""
        return self.root / project_id

    def _project_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / PROJECT_FILENAME

    def _meta_file(self, project_id: str) -> Path:
        return self.project_dir(project_id) / META_FILENAME

    def video_dir(self, project_id: str) -> Path:
        """Return the directory the source video is copied into."""
        return self.project_dir(project_id) / "video"

    def proxies_dir(self, project_id: str) -> Path:
        """Return the directory proxy tiers are generated into."""
        return self.project_dir(project_id) / "proxies"

    def tracks_dir(self, project_id: str) -> Path:
        """Return the directory per-point raw tracks are saved into."""
        return self.project_dir(project_id) / "tracks"

    def renders_dir(self, project_id: str) -> Path:
        """Return the directory exported videos are written into."""
        return self.project_dir(project_id) / "renders"

    def _require_dir(self, project_id: str) -> Path:
        directory = self.project_dir(project_id)
        if not directory.is_dir():
            raise ProjectNotFoundError(f"no project with id {project_id!r}")
        return directory

    # -- CRUD ----------------------------------------------------------------

    def create_project(
        self, *, video_bytes: bytes, original_filename: str, name: str | None = None
    ) -> tuple[ProjectMeta, Project]:
        """Create a new project from an uploaded video file's raw bytes.

        Copies the video into the project directory (so the project is
        self-contained), reads its metadata with :class:`PyAVVideoSource`, and
        writes an initial (pointless) project file with no tracked points yet.

        Args:
            video_bytes: The raw bytes of the uploaded video file.
            original_filename: The client's filename, used for the extension
                and for display.
            name: Display name; defaults to the filename's stem.

        Returns:
            ``(meta, project)`` for the newly created project.

        Raises:
            ProjectError: If the uploaded bytes cannot be decoded as video.
        """
        project_id = uuid4().hex[:12]
        directory = self.project_dir(project_id)
        directory.mkdir(parents=True, exist_ok=False)
        video_dir = self.video_dir(project_id)
        video_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_filename).suffix or ".mp4"
        video_path = video_dir / f"source{suffix}"
        video_path.write_bytes(video_bytes)

        try:
            source = PyAVVideoSource(video_path)
            try:
                frames, w, h, fps = source.frame_count, source.width, source.height, source.fps
            finally:
                source.close()
        except StableError:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        except Exception as exc:  # pragma: no cover - defensive: PyAV surprises
            shutil.rmtree(directory, ignore_errors=True)
            raise ProjectError(f"could not read uploaded video: {exc}") from exc

        project = Project()
        project.video.path = project.relative_path(str(video_path))
        project.video.frames = frames
        project.video.w = w
        project.video.h = h
        project.video.fps = fps
        project.path = str(self._project_file(project_id))
        project.save()

        meta = ProjectMeta(
            id=project_id,
            name=name or Path(original_filename).stem or project_id,
            created_at=time.time(),
            original_filename=original_filename,
        )
        self._meta_file(project_id).write_text(
            json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
        )
        return meta, project

    def list_projects(self) -> list[ProjectMeta]:
        """Return every stored project's metadata, newest first."""
        out: list[ProjectMeta] = []
        for child in self.root.iterdir():
            meta_file = child / META_FILENAME
            if meta_file.is_file():
                try:
                    out.append(ProjectMeta.from_dict(json.loads(meta_file.read_text("utf-8"))))
                except (OSError, ValueError, KeyError):  # pragma: no cover - corrupt project
                    continue
        out.sort(key=lambda m: m.created_at, reverse=True)
        return out

    def load_meta(self, project_id: str) -> ProjectMeta:
        """Return a project's server-only metadata.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        self._require_dir(project_id)
        meta_file = self._meta_file(project_id)
        try:
            return ProjectMeta.from_dict(json.loads(meta_file.read_text("utf-8")))
        except OSError as exc:
            raise ProjectNotFoundError(f"no project with id {project_id!r}") from exc

    def load_project(self, project_id: str) -> Project:
        """Load and return a project's current state from disk.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        self._require_dir(project_id)
        try:
            return Project.load(self._project_file(project_id))
        except ProjectError as exc:
            raise ProjectNotFoundError(f"no project with id {project_id!r}: {exc}") from exc

    def save_project(self, project_id: str, project: Project) -> None:
        """Persist ``project`` back to disk at its canonical path."""
        self._require_dir(project_id)
        project.save(self._project_file(project_id))

    def update_project(self, project_id: str, mutate: Callable[[Project], None]) -> Project:
        """Read-modify-write a project under the workspace's global lock.

        Args:
            project_id: The project to update.
            mutate: Called with the loaded :class:`Project`; mutate it in place.

        Returns:
            The saved project.

        Raises:
            ProjectNotFoundError: If no such project exists.
            ProjectError: If the mutation leaves the project invalid.
        """
        with self._lock:
            project = self.load_project(project_id)
            mutate(project)
            project.validate()
            self.save_project(project_id, project)
            return project

    def rename_project(self, project_id: str, name: str) -> ProjectMeta:
        """Update a project's display name.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        with self._lock:
            meta = self.load_meta(project_id)
            meta.name = name
            self._meta_file(project_id).write_text(
                json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
            )
            return meta

    def delete_project(self, project_id: str) -> None:
        """Delete a project directory and evict any cached runtime objects.

        The cached video source is *closed*, not merely dropped: it holds an
        open handle on the clip inside the directory being removed, and Windows
        refuses to delete an open file — so evicting it alone would leave the
        video behind and ``ignore_errors=True`` would hide that it had.
        """
        with self._lock:
            source = self._video_cache.pop(project_id, None)
            if source is not None:
                source.close()
            for key in [k for k in self._proxy_cache if k[0] == project_id]:
                self._proxy_cache.pop(key, None)
            directory = self._require_dir(project_id)
            shutil.rmtree(directory, ignore_errors=True)

    # -- runtime objects -----------------------------------------------------

    def get_video_source(self, project_id: str) -> PyAVVideoSource:
        """Return a cached, open :class:`PyAVVideoSource` for a project's clip.

        The source is decoded once and reused across every job and frame
        request against this project, rather than re-indexing the whole clip
        (an ``O(frames)`` pass) on every call.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        with self._lock:
            cached = self._video_cache.get(project_id)
            if cached is not None:
                return cached
            project = self.load_project(project_id)
            path = project.resolve_path(project.video.path)
            if not path:
                raise ProjectError(f"project {project_id!r} has no video path")
            source = PyAVVideoSource(path)
            self._video_cache[project_id] = source
            return source

    def get_proxy_store(self, project_id: str, tier: str = "half") -> ProxyStore:
        """Return a cached :class:`ProxyStore` for one proxy tier of a project.

        Raises:
            ProjectNotFoundError: If no such project exists.
            stable.core.errors.VideoError: If that tier has not been generated yet.
        """
        with self._lock:
            key = (project_id, tier)
            cached = self._proxy_cache.get(key)
            if cached is not None:
                return cached
            self._require_dir(project_id)
            store = ProxyStore(self.proxies_dir(project_id) / tier)
            self._proxy_cache[key] = store
            return store

    def has_proxy(self, project_id: str, tier: str = "half") -> bool:
        """Return whether a proxy tier has already been generated."""
        manifest = self.proxies_dir(project_id) / tier / "proxy.json"
        return manifest.is_file()

    def close(self) -> None:
        """Close every cached video source. Called on server shutdown."""
        with self._lock:
            for source in self._video_cache.values():
                source.close()
            self._video_cache.clear()
            self._proxy_cache.clear()


_workspace: Workspace | None = None
_workspace_lock = threading.Lock()


def get_workspace() -> Workspace:
    """Return the process-wide :class:`Workspace` singleton, creating it lazily."""
    global _workspace
    with _workspace_lock:
        if _workspace is None:
            _workspace = Workspace()
        return _workspace


def reset_workspace(root: Path | None = None) -> Workspace:
    """Replace the process-wide workspace (tests use this for isolation)."""
    global _workspace
    with _workspace_lock:
        if _workspace is not None:
            _workspace.close()
        _workspace = Workspace(root)
        return _workspace
