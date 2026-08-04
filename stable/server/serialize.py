"""JSON shapes shared by more than one route module.

Kept separate from ``routes/`` so both ``routes/projects.py`` and
``routes/points.py`` can build the same point summary without importing each
other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.project import PointSpec, Project
from ..core.types import PointTrack
from .storage import Workspace
from .tracking import load_report, report_path_for

__all__ = ["point_summary", "project_detail"]

_EMPTY_STATS: dict[str, Any] = {
    "has_track": False,
    "tracked_frames": 0,
    "mean_confidence": 0.0,
    "worst_frames": [],
}


def _load_track_stats(project: Project, point: PointSpec) -> dict[str, Any]:
    """Return the lightweight (non-array) stats used in list/detail responses."""
    if not point.raw:
        return dict(_EMPTY_STATS)
    path = project.resolve_path(point.raw)
    try:
        track = PointTrack.load_npz(path) if path else None
    except (OSError, ValueError):  # pragma: no cover - corrupt/missing npz
        track = None
    if track is None:
        return dict(_EMPTY_STATS)
    valid = track.valid
    conf = track.confidence
    worst: list[int] = []
    if path:
        report = load_report(report_path_for(Path(path)))
        if report:
            disagreement = report.get("disagreement") or []
            ranked = sorted(
                (i for i, v in enumerate(disagreement) if v is not None),
                key=lambda i: disagreement[i],
                reverse=True,
            )
            worst = ranked[:5]
    return {
        "has_track": True,
        "tracked_frames": int(valid.sum()),
        "mean_confidence": float(conf[valid].mean()) if valid.any() else 0.0,
        "worst_frames": worst,
    }


def point_summary(project: Project, point: PointSpec) -> dict[str, Any]:
    """Return a point's list/detail summary (no per-frame arrays)."""
    return {
        "id": point.id,
        "queries": [q.to_dict() for q in point.sorted_queries()],
        "tracker": point.tracker,
        "enabled_for_solve": point.enabled_for_solve,
        **_load_track_stats(project, point),
    }


def project_detail(workspace: Workspace, project_id: str) -> dict[str, Any]:
    """Return the full detail payload for ``GET /api/projects/{id}``."""
    meta = workspace.load_meta(project_id)
    project = workspace.load_project(project_id)
    proxy: dict[str, Any] = {}
    for tier_name in ("half", "quarter"):
        if workspace.has_proxy(project_id, tier_name):
            store = workspace.get_proxy_store(project_id, tier_name)
            proxy[tier_name] = {
                "ready": True,
                "width": store.manifest.width,
                "height": store.manifest.height,
                "count": store.manifest.count,
                "fps": store.manifest.fps,
            }
        else:
            proxy[tier_name] = {"ready": False}

    return {
        "id": meta.id,
        "name": meta.name,
        "created_at": meta.created_at,
        "original_filename": meta.original_filename,
        "video": project.video.to_dict(),
        "points": [point_summary(project, p) for p in project.points],
        "solve": project.solve.to_dict(),
        "smoothing": [s.to_dict() for s in project.smoothing],
        "render": project.render.to_dict(),
        "proxy": proxy,
    }
