"""Per-point tracker settings, and the template patch they describe.

Todo 14 asks for two things about template matching: let the operator choose the
template size, and let them *see* the patch that size produces at each of their
anchors. Both live here.

Every tracker's window, not only the template's
-----------------------------------------------
The first version of this module let the size be *set* only where the plugin
called it ``template_radius``, and merely *reported* the number everywhere else.
That is the wrong split: "the thing I am tracking is bigger than 33 px" is a
statement about the subject, not about which plugin happens to be selected, and
every classic-CV tracker here matches a neighbourhood whose size it takes as a
constructor argument. :data:`SETTINGS` is the one table that says which of those
arguments are the operator's to set, in what units, and within what range; the
API, the panel and the validation all read it, so a control is offered exactly
where changing it would change the next run.

Two roles are distinguished, the pair a Resolve user would recognise as the
pattern window and the search area:

``window``
    The region whose pixels are matched — the "pattern". Every classic-CV
    tracker has one.
``search``
    How far from the last known position that pattern is looked for. Only the
    trackers that perform an explicit spatial search have one (``template``,
    ``kalman``, and the Siamese ``ocean`` and ``siamdw``, whose network scores a
    whole search region); phase correlation measures a shift without
    searching at all, and LK's reach comes from its pyramid depth instead —
    :func:`tracker_search` reports that reach and where it came from rather than
    inventing a control that would do nothing.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ProjectError
from ..core.project import PointSpec

__all__ = [
    "TRACKER_OPTIONS_CACHE",
    "TEMPLATE_RADIUS_RANGE",
    "SEARCH_RADIUS_RANGE",
    "SETTINGS",
    "WINDOW_ATTRIBUTES",
    "TrackerSetting",
    "constructor_parameters",
    "load_tracker_options",
    "point_options",
    "save_point_options",
    "setting_for",
    "template_patch",
    "patch_bounds",
    "tracker_search",
    "tracker_settings",
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


@dataclass(frozen=True)
class TrackerSetting:
    """One constructor argument the operator is allowed to set, per point.

    Attributes:
        attribute: The plugin constructor argument this sets, which is also the
            key it is stored under and the name the range error names.
        role: ``"window"`` (the pattern being matched) or ``"search"`` (how far
            it is looked for). The panel groups controls by this.
        label: Human label for the control.
        radius_factor: Multiplier turning the stored value into a *radius* in
            source pixels — ``1.0`` for the arguments that already are one,
            ``0.5`` for those stating a full side length (LK's ``win_size``).
            ``0.0`` marks a value that is not a length at all (``max_level``).
        minimum: Smallest accepted value, in the argument's own units.
        maximum: Largest accepted value, in the argument's own units.
        step: Increment the UI should move in — ``2`` for the side lengths that
            want to stay odd so the patch has a centre pixel.
        help: One sentence saying what raising it buys and what it costs.
    """

    attribute: str
    role: str
    label: str
    radius_factor: float
    minimum: int
    maximum: int
    step: int = 1
    help: str = ""

    @property
    def unit(self) -> str:
        """How to read a value: a half-size, a full side, or neither."""
        if self.radius_factor == 1.0:
            return "radius"
        if self.radius_factor == 0.5:
            return "side"
        return "count"

    def radius(self, value: float) -> int:
        """Return ``value`` as a radius in source pixels (0 when it is not one)."""
        return max(0, int(round(float(value) * self.radius_factor)))

    def from_radius(self, radius: int) -> int:
        """Return the value in this argument's own units for a radius in pixels.

        The inverse of :meth:`radius`, so a caller can talk in pixels — which is
        what the operator measures — whatever the plugin happens to call its
        window. A full side length comes back odd, since a patch needs a centre
        pixel.

        Raises:
            ProjectError: If this setting is not a length at all.
        """
        if self.radius_factor == 0.0:
            raise ProjectError(f"{self.attribute} is not a size in pixels")
        if self.radius_factor == 0.5:
            return 2 * int(radius) + 1
        return int(round(int(radius) / self.radius_factor))


SETTINGS: tuple[TrackerSetting, ...] = (
    TrackerSetting(
        attribute="template_radius",
        role="window",
        label="Pattern size",
        radius_factor=1.0,
        minimum=TEMPLATE_RADIUS_RANGE[0],
        maximum=TEMPLATE_RADIUS_RANGE[1],
        help="Half-size of the matched patch. Raise it for a large or low-contrast "
        "feature; a patch bigger than the feature starts tracking the background too.",
    ),
    TrackerSetting(
        attribute="patch_radius",
        role="window",
        label="Pattern size",
        radius_factor=1.0,
        minimum=TEMPLATE_RADIUS_RANGE[0],
        maximum=TEMPLATE_RADIUS_RANGE[1],
        help="Half-size of the correlated patch. It also bounds the per-frame shift "
        "this tracker can measure, so raise it for fast motion.",
    ),
    TrackerSetting(
        attribute="window_radius",
        role="window",
        label="Pattern size",
        radius_factor=1.0,
        minimum=TEMPLATE_RADIUS_RANGE[0],
        maximum=TEMPLATE_RADIUS_RANGE[1],
        help="Half-size of the mean-shift window. It should cover the feature and "
        "little else — the histogram inside it is what is being followed.",
    ),
    TrackerSetting(
        attribute="win_size",
        role="window",
        label="Pattern size",
        radius_factor=0.5,
        minimum=5,
        maximum=255,
        step=2,
        help="Full side of the optical-flow window. Larger is more robust to noise "
        "and less able to follow fine detail near a motion boundary.",
    ),
    TrackerSetting(
        attribute="search_radius",
        role="search",
        label="Search area",
        radius_factor=1.0,
        minimum=SEARCH_RADIUS_RANGE[0],
        maximum=SEARCH_RADIUS_RANGE[1],
        help="How far beyond the pattern to look each frame. It has to exceed the "
        "fastest per-frame motion; every extra pixel costs search time.",
    ),
    TrackerSetting(
        attribute="max_level",
        role="search",
        label="Pyramid levels",
        radius_factor=0.0,
        minimum=0,
        maximum=6,
        help="Optical flow has no search box: its reach is the window scaled up "
        "through this many pyramid levels, so each level doubles the motion it can "
        "follow — and coarsens what it can distinguish.",
    ),
)
"""Every per-point setting this server knows how to validate and describe.

Keyed by the plugin *constructor argument*, because that is what actually
reaches the tracker (:meth:`stable.trackers.registry.TrackerRegistry.create`
filters options against the constructor). A plugin is offered exactly the
settings it declares — no mapping table per tracker, and no control that would
be silently dropped.
"""


def setting_for(attribute: str) -> TrackerSetting | None:
    """Return the :class:`TrackerSetting` for a constructor argument, if known."""
    for setting in SETTINGS:
        if setting.attribute == attribute:
            return setting
    return None


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
    for setting in SETTINGS:
        key = setting.attribute
        if key not in checked or checked[key] is None:
            checked.pop(key, None)
            continue
        try:
            value = int(checked[key])
        except (TypeError, ValueError):
            raise ProjectError(f"{key} must be an integer, got {checked[key]!r}") from None
        if not setting.minimum <= value <= setting.maximum:
            raise ProjectError(
                f"{key} must lie in [{setting.minimum}, {setting.maximum}], got {value}"
            )
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


def forget_point(project_dir: Path, point_id: str) -> bool:
    """Drop a deleted point's stored settings from the sidecar.

    A point's settings are keyed by its id, so leaving them behind means the
    next point to be given that id silently inherits a window size nobody chose
    for it — the same class of bug as its track file outliving it.

    Returns:
        True if there was anything to remove.
    """
    stored = load_tracker_options(project_dir)
    if point_id not in stored:
        return False
    stored.pop(point_id)
    options_path(project_dir).write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return True


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
    view = getattr(video, "get_frame_view", None)
    frame_pixels = view(frame) if callable(view) else video.get_frame(frame)
    return np.ascontiguousarray(frame_pixels[y0:y1, x0:x1])


def proxy_patch(
    store: Any, frame: int, position: tuple[float, float], radius: int
) -> np.ndarray | None:
    """Crop the same patch from a **proxy** frame instead of the source clip.

    Scrubbing is where this matters. A thumbnail is drawn at 72 px whatever its
    pixel size, so serving it from the half-resolution proxy costs a JPEG read
    from disk rather than a full-resolution random-access seek that queues
    behind the decoder's lock — the difference between a smooth scrub and the
    UI stalling with a dozen seeks outstanding.

    The *availability* rule stays in source pixels (:func:`patch_bounds`, the
    tracker's own edge rule); only the pixels come from the proxy. So a patch
    the grid marks ✕ is still exactly one the tracker would refuse.

    Args:
        store: A :class:`stable.video.proxy.ProxyStore`.
        frame: Absolute frame index.
        position: Patch centre in **source** pixels.
        radius: Half-size in **source** pixels.

    Returns:
        The ``uint8`` RGB patch at proxy scale, or ``None`` when the proxy
        cannot supply it (out of range, or the scaled patch leaves the frame),
        in which case the caller falls back to the source.
    """
    scale = float(getattr(store.manifest, "scale", 1.0) or 1.0)
    if not 0 <= frame < store.frame_count:
        return None
    scaled_radius = max(1, int(round(radius * scale)))
    centre = (position[0] * scale, position[1] * scale)
    box = patch_bounds(centre, scaled_radius, store.manifest.width, store.manifest.height)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    try:
        pixels = store.get_frame(frame)
    except Exception:  # noqa: BLE001 - a missing proxy frame just means "use the source"
        return None
    return np.ascontiguousarray(pixels[y0:y1, x0:x1])


def effective_radius(project_dir: Path, point: PointSpec, default: int) -> int:
    """Return the template radius in force for a point, stored or default."""
    stored = point_options(project_dir, point.id).get("template_radius")
    try:
        return int(stored) if stored is not None else int(default)
    except (TypeError, ValueError):  # pragma: no cover - hand-edited sidecar
        return int(default)


WINDOW_ATTRIBUTES: tuple[tuple[str, float], ...] = tuple(
    (setting.attribute, setting.radius_factor) for setting in SETTINGS if setting.role == "window"
)
"""Constructor arguments that describe the region a tracker looks at, and the
factor turning each into a *radius*.

Every classic-CV tracker in this project matches a small neighbourhood of the
point, but they do not agree on what to call it — ``template_radius`` (template,
kalman, particle, and the target box of ocean and siamdw), ``patch_radius``
(correlation), ``window_radius`` (meanshift)
or ``win_size`` (lk, klt, which state a full side length, hence the 0.5). Rather
than teach the panel about each plugin, the effective radius is read from
whichever of these the plugin declares, in this order. A tracker declaring none
of them has no local window at all — the deep trackers see whole frames — and
:func:`tracker_window` says so instead of inventing a number.

Derived from :data:`SETTINGS` rather than written out again, so a window
argument can only ever be added in one place.
"""

DEFAULT_PREVIEW_RADIUS = 16
"""Patch half-size previewed for a tracker that has no window of its own."""


def _parameters(tracker_name: str) -> dict[str, Any]:
    """Return ``tracker_name``'s effective constructor arguments and defaults.

    Deliberately the *same* answer
    :func:`stable.trackers.registry.constructor_parameters` gives, because that
    is the function :meth:`~stable.trackers.registry.TrackerRegistry.create`
    filters a point's stored options through. A control offered here that the
    filter would then drop is a setting that appears to work and does nothing —
    so the panel and the run must never disagree about what a plugin accepts.

    Returns:
        ``{name: default}``, or ``{}`` for an unknown or uninspectable plugin —
        which callers read as "no settings", never as an error.
    """
    from ..trackers.registry import (  # noqa: PLC0415 - avoids an import cycle
        constructor_parameters as parameters_of,
    )
    from ..trackers.registry import get_tracker

    try:
        plugin_cls = get_tracker(tracker_name)
    except Exception:  # noqa: BLE001 - an unknown tracker simply has no settings
        return {}
    return parameters_of(plugin_cls)


def constructor_parameters(tracker_name: str) -> dict[str, Any]:
    """Return ``{argument: default}`` for everything ``tracker_name`` accepts."""
    return dict(_parameters(tracker_name))


def _effective(tracker_name: str, setting: TrackerSetting, stored: dict[str, Any]) -> int | None:
    """Return a setting's value for a point: its override, else the plugin default."""
    parameters = _parameters(tracker_name)
    if setting.attribute not in parameters:
        return None
    raw = stored.get(setting.attribute, parameters[setting.attribute])
    if raw is None:
        return None
    try:
        return int(round(float(raw)))
    except (TypeError, ValueError):  # pragma: no cover - hand-edited sidecar
        return None


def tracker_settings(
    tracker_name: str, options: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Describe every per-point setting ``tracker_name`` actually accepts.

    This is what makes the panel tracker-agnostic: it renders one control per
    row returned here, in this order, and knows nothing about which plugin is
    selected.

    Args:
        tracker_name: Registered plugin name.
        options: The point's stored overrides, which win over the plugin's own
            defaults exactly as they do when a run is created.

    Returns:
        One row per setting, each carrying the current ``value``, the plugin's
        ``default``, the accepted range, the ``radius`` in source pixels the
        value corresponds to (``0`` when it is not a length), and whether the
        point has overridden it.
    """
    stored = options or {}
    rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        value = _effective(tracker_name, setting, stored)
        if value is None:
            continue
        rows.append(
            {
                "attribute": setting.attribute,
                "role": setting.role,
                "label": setting.label,
                "value": value,
                "default": _effective(tracker_name, setting, {}),
                "minimum": setting.minimum,
                "maximum": setting.maximum,
                "step": setting.step,
                "unit": setting.unit,
                "radius": setting.radius(value),
                "help": setting.help,
                "stored": setting.attribute in stored,
            }
        )
    return rows


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
    stored = options or {}
    for setting in SETTINGS:
        if setting.role != "window":
            continue
        value = _effective(tracker_name, setting, stored)
        if value is None:
            continue
        return max(1, setting.radius(value)), setting.attribute
    return DEFAULT_PREVIEW_RADIUS, None


def tracker_search(
    tracker_name: str, options: dict[str, Any] | None = None
) -> tuple[int | None, str | None, str]:
    """Return how far ``tracker_name`` looks for its pattern each frame.

    The Resolve-style "search area", answered honestly per algorithm rather than
    invented for the ones that do not have one:

    * an explicit ``search_radius`` (``template``, ``kalman``, ``ocean``,
      ``siamdw``) is reported as it is stored — the box really is scanned;
    * optical flow (``lk``, ``klt``) has no box. Its reach is the window carried
      up the pyramid, ``win_size / 2 * 2 ** max_level``, so that is what is
      reported, attributed to ``max_level``;
    * everything else gets ``None`` and a sentence saying why, which the panel
      shows in place of a control.

    Returns:
        ``(radius, attribute, note)``. ``radius`` is in source pixels and
        measured from the point, like every other radius here; ``attribute`` is
        the setting that governs it, or ``None`` when nothing does.
    """
    stored = options or {}
    parameters = _parameters(tracker_name)

    explicit = setting_for("search_radius")
    if explicit is not None:
        value = _effective(tracker_name, explicit, stored)
        if value is not None:
            return value, explicit.attribute, "Scanned around the last known position each frame."

    levels = setting_for("max_level")
    window = setting_for("win_size")
    if levels is not None and window is not None and "max_level" in parameters:
        max_level = _effective(tracker_name, levels, stored)
        win_size = _effective(tracker_name, window, stored)
        if max_level is not None and win_size is not None:
            reach = int(round(window.radius(win_size) * (2**max_level)))
            return (
                reach,
                levels.attribute,
                "Optical flow does not scan a box: this is how far the pattern can move "
                "between frames and still be found, from the pyramid depth.",
            )

    if "patch_radius" in parameters:
        return (
            None,
            None,
            "Phase correlation measures the shift directly and never searches a wider "
            "area — the pattern size itself bounds the motion it can recover.",
        )
    return None, None, f"{tracker_name} has no search window of its own."
