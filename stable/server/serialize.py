"""JSON shapes shared by more than one route module.

Kept separate from ``routes/`` so both ``routes/projects.py`` and
``routes/points.py`` can build the same point summary without importing each
other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import StableError
from ..core.project import PointSpec, Project
from .storage import Workspace
from .tracker_options import load_tracker_options, tracker_search, tracker_window
from .tracking import anchors_changed, load_display_track, load_report, report_path_for

__all__ = ["point_summary", "project_detail"]

_EMPTY_STATS: dict[str, Any] = {
    "has_track": False,
    "tracked_frames": 0,
    "mean_confidence": 0.0,
    "worst_frames": [],
    "anchors_changed": False,
}


def _load_track_stats(project: Project, point: PointSpec) -> dict[str, Any]:
    """Return the lightweight (non-array) stats used in list/detail responses.

    Measured on the **displayed** track — raw plus the correction layer — which
    is what the canvas draws, the solver consumes and the export writes. Reading
    the raw ``.npz`` alone under-reported every point whose coverage comes from
    corrections: a point tracked only by a directional run from one anchor has
    an empty baseline (see :func:`stable.server.tracking.run_point_retrack`) and
    used to report "0 frames · 0% confidence" over a track plainly drawn on the
    frame. ``worst_frames`` stays a property of the *run*, so it still comes
    from the merge report.
    """
    if not point.raw:
        return dict(_EMPTY_STATS)
    path = project.resolve_path(point.raw)
    try:
        track = load_display_track(project, point) if path else None
    except (OSError, ValueError, StableError):  # pragma: no cover - corrupt/missing npz
        track = None
    if track is None:
        return dict(_EMPTY_STATS)
    valid = track.valid
    conf = track.confidence
    worst: list[int] = []
    report = None
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
        # Whether the anchors have moved on since this track was computed, so
        # the card can offer a re-track instead of quietly showing a stale one.
        "anchors_changed": anchors_changed(point, report),
    }


def point_summary(
    project: Project, point: PointSpec, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a point's list/detail summary (no per-frame arrays).

    Args:
        project: The point's project.
        point: The point to describe.
        options: The point's stored tracker settings. Omit it and they are read
            from the project's own directory — correct for a caller summarising
            one point, wasteful for a caller summarising a list, which reads the
            sidecar once and passes each point's share (``{}`` included, so a
            point with no overrides does not trigger a second read).
    """
    if options is None:
        options = load_tracker_options(Path(project.directory)).get(point.id, {})
    radius, radius_source = tracker_window(point.tracker, options)
    search_radius, search_source, _note = tracker_search(point.tracker, options)
    return {
        "id": point.id,
        "queries": [q.to_dict() for q in point.sorted_queries()],
        "tracker": point.tracker,
        "enabled_for_solve": point.enabled_for_solve,
        # The regions this point's tracker matches and searches, in source
        # pixels. Carried on every point so the canvas can draw the pattern and
        # search boxes for all of them without a request per point — see
        # :mod:`stable.server.tracker_options` for where the numbers come from.
        "window": {
            "radius": radius,
            "attribute": radius_source,
            "search_radius": search_radius,
            "search_attribute": search_source,
        },
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

    # One sidecar read for the whole list rather than one per point.
    stored = load_tracker_options(workspace.project_dir(project_id))
    return {
        "id": meta.id,
        "name": meta.name,
        "created_at": meta.created_at,
        "original_filename": meta.original_filename,
        "video": project.video.to_dict(),
        "points": [point_summary(project, p, stored.get(p.id)) for p in project.points],
        "solve": project.solve.to_dict(),
        "smoothing": [s.to_dict() for s in project.smoothing],
        "render": project.render.to_dict(),
        "proxy": proxy,
    }
