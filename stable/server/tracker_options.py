"""Per-point tracker settings, and the template patch they describe.

Todo 14 asks for two things about template matching: let the operator choose the
template size, and let them *see* the patch that size produces at each of their
keyframes. Both live here.

Where the settings are kept
---------------------------
In ``tracker_options.json``, a sidecar in the project directory beside
``solve.json`` / ``smoothing.json`` / ``render_options.json``, keyed by point id.
The v1 project schema stays exactly as it is — the same choice
:mod:`stable.server.motion` made for the render options, and for the same
reason: this is a per-project preference, not part of the document a migration
would have to carry forward.

Settings are stored per *point*, not per tracker, because that is the unit the
operator is thinking about ("this corner needs a bigger patch than that one"),
and they are filtered against the plugin's own constructor when a run starts
(:meth:`stable.trackers.registry.TrackerRegistry.create`), so switching a point
from ``template`` to ``lk`` and back neither errors nor loses the setting.

Why the patch is rendered here and not stored
---------------------------------------------
A template patch is a pure function of (clip, frame, position, radius), all of
which the server already has. Caching it would mean invalidating it whenever an
anchor moves — so it is simply cropped on demand from the video source, which is
already open and cached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ProjectError
from ..core.project import PointSpec

__all__ = [
    "TRACKER_OPTIONS_CACHE",
    "TEMPLATE_RADIUS_RANGE",
    "WINDOW_ATTRIBUTES",
    "load_tracker_options",
    "point_options",
    "save_point_options",
    "template_patch",
    "patch_bounds",
    "tracker_window",
]

TRACKER_OPTIONS_CACHE = "tracker_options.json"
"""File name of a project's per-point tracker settings sidecar."""

TEMPLATE_RADIUS_RANGE = (2, 128)
"""Bounds on the template half-size, in source pixels.

Below 2 there is not enough signal for normalised cross-correlation to mean
anything; above 128 the patch stops being a feature and starts being the shot.
"""

SEARCH_RADIUS_RANGE = (2, 256)
"""Bounds on how far beyond the template to search each frame."""


def options_path(project_dir: Path) -> Path:
    """Return where a project's tracker-options sidecar lives."""
    return Path(project_dir) / TRACKER_OPTIONS_CACHE


def load_tracker_options(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Return every point's stored tracker options, keyed by point id.

    A missing or unreadable sidecar yields ``{}`` — every plugin's own defaults
    then apply, which is exactly the behaviour before this existed.
    """
    path = options_path(project_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar only
        return {}
    if not isinstance(payload, dict):  # pragma: no cover - hand-edited sidecar
        return {}
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def point_options(project_dir: Path, point_id: str) -> dict[str, Any]:
    """Return one point's stored tracker options (``{}`` if it has none)."""
    return load_tracker_options(project_dir).get(point_id, {})


def _validate(options: dict[str, Any]) -> dict[str, Any]:
    """Coerce and range-check the options this server knows about.

    Unknown keys pass through untouched: a plugin added later can store its own
    settings here without this module needing to learn about them, and
    :meth:`~stable.trackers.registry.TrackerRegistry.create` drops any the
    plugin does not accept.

    Raises:
        ProjectError: If a known option is not an integer in range.
    """
    checked = dict(options)
    for key, (low, high) in (
        ("template_radius", TEMPLATE_RADIUS_RANGE),
        ("search_radius", SEARCH_RADIUS_RANGE),
    ):
        if key not in checked or checked[key] is None:
            checked.pop(key, None)
            continue
        try:
            value = int(checked[key])
        except (TypeError, ValueError):
            raise ProjectError(f"{key} must be an integer, got {checked[key]!r}") from None
        if not low <= value <= high:
            raise ProjectError(f"{key} must lie in [{low}, {high}], got {value}")
        checked[key] = value
    return checked


def save_point_options(project_dir: Path, point_id: str, options: dict[str, Any]) -> dict[str, Any]:
    """Merge ``options`` into one point's settings and persist them.

    Args:
        project_dir: The project's directory.
        point_id: Which point the settings belong to.
        options: Keys to set. A key whose value is ``None`` is *removed*, which
            is how the UI restores a plugin's own default.

    Returns:
        The point's settings after the merge.

    Raises:
        ProjectError: If a known option is out of range.
    """
    checked = _validate(options)
    stored = load_tracker_options(project_dir)
    current = dict(stored.get(point_id, {}))
    for key, value in options.items():
        if value is None:
            current.pop(key, None)
        elif key in checked:
            current[key] = checked[key]
    if current:
        stored[point_id] = current
    else:
        stored.pop(point_id, None)
    options_path(project_dir).write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return current


def patch_bounds(
    position: tuple[float, float], radius: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Return the ``(x0, y0, x1, y1)`` patch box, or ``None`` if it leaves the frame.

    Deliberately the same "must fit entirely" rule
    :meth:`stable.trackers.template.TemplateTracker._extract_patch` applies, so
    a patch the panel cannot show is exactly a patch the tracker would refuse to
    build — the thumbnail grid is a preview of the real thing, not an
    approximation of it.
    """
    cx, cy = int(round(position[0])), int(round(position[1]))
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return x0, y0, x1, y1


def template_patch(
    video: Any, frame: int, position: tuple[float, float], radius: int
) -> np.ndarray | None:
    """Crop the template patch a tracker would build at ``position`` on ``frame``.

    Args:
        video: An open video source.
        frame: Absolute frame index.
        position: Patch centre in source pixels.
        radius: Template half-size; the patch is ``2 * radius + 1`` square.

    Returns:
        The ``uint8`` RGB patch, or ``None`` if it would fall outside the frame.
    """
    if not 0 <= frame < video.frame_count:
        return None
    box = patch_bounds(position, radius, video.width, video.height)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(video.get_frame_view(frame)[y0:y1, x0:x1])


def effective_radius(project_dir: Path, point: PointSpec, default: int) -> int:
    """Return the template radius in force for a point, stored or default."""
    stored = point_options(project_dir, point.id).get("template_radius")
    try:
        return int(stored) if stored is not None else int(default)
    except (TypeError, ValueError):  # pragma: no cover - hand-edited sidecar
        return int(default)


WINDOW_ATTRIBUTES: tuple[tuple[str, float], ...] = (
    ("template_radius", 1.0),
    ("patch_radius", 1.0),
    ("window_radius", 1.0),
    ("win_size", 0.5),
)
"""Constructor arguments that describe the region a tracker looks at, and the
factor turning each into a *radius*.

Every classic-CV tracker in this project matches a small neighbourhood of the
point, but they do not agree on what to call it — ``template_radius`` (template,
kalman, particle), ``patch_radius`` (correlation), ``window_radius`` (meanshift)
or ``win_size`` (lk, klt, which state a full side length, hence the 0.5). Rather
than teach the panel about each plugin, the effective radius is read from
whichever of these the plugin declares, in this order. A tracker declaring none
of them has no local window at all — the deep trackers see whole frames — and
:func:`tracker_window` says so instead of inventing a number.
"""

DEFAULT_PREVIEW_RADIUS = 16
"""Patch half-size previewed for a tracker that has no window of its own."""


def tracker_window(
    tracker_name: str, options: dict[str, Any] | None = None
) -> tuple[int, str | None]:
    """Return the half-size of the region ``tracker_name`` actually matches.

    Args:
        tracker_name: Registered plugin name.
        options: The point's stored overrides, which win over the plugin's
            defaults exactly as they do when a run is created.

    Returns:
        ``(radius, attribute)`` — the radius in source pixels and the
        constructor argument it came from, or ``(DEFAULT_PREVIEW_RADIUS, None)``
        when the tracker has no local window (or cannot be inspected, e.g. a
        plugin whose optional dependency is missing). The radius is still
        useful in that case: it is the size of the *preview*, which is honest
        as long as the caller reports that it is not the tracker's own.
    """
    import inspect  # noqa: PLC0415 - only needed on this path

    from ..trackers.registry import get_tracker  # noqa: PLC0415 - avoids an import cycle

    try:
        plugin_cls = get_tracker(tracker_name)
        parameters = inspect.signature(plugin_cls.__init__).parameters
    except Exception:  # noqa: BLE001 - an unknown tracker just has no window
        return DEFAULT_PREVIEW_RADIUS, None

    stored = options or {}
    for name, factor in WINDOW_ATTRIBUTES:
        if name not in parameters:
            continue
        raw = stored.get(name, parameters[name].default)
        if raw is inspect.Parameter.empty or raw is None:
            continue
        try:
            radius = int(round(float(raw) * factor))
        except (TypeError, ValueError):  # pragma: no cover - hand-edited sidecar
            continue
        return max(1, radius), name
    return DEFAULT_PREVIEW_RADIUS, None
