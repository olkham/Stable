"""Tracker discovery: built-ins plus third-party plugins from entry points.

A tracker ships as an ordinary pip package that advertises the
``stable.trackers`` entry-point group::

    # in the plugin's pyproject.toml
    [project.entry-points."stable.trackers"]
    my_tracker = "my_package.tracker:MyTracker"

Installing that package is the entire integration step — Stable picks it up on
next launch, it appears in the UI's tracker picker, and project files can name
it. Nothing in the solver, renderer or UI needs to change.

A broken third-party plugin must never take the app down, so
:meth:`TrackerRegistry.discover` logs and skips entry points that fail to
import, unless the caller asks for ``strict=True`` (used by tests and CI).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from importlib import metadata
from typing import Any

from ..core.device import select_device
from ..core.errors import TrackerError, TrackerNotFoundError
from .base import TrackerPlugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "TrackerRegistry",
    "registry",
    "register_tracker",
    "get_tracker",
    "create_tracker",
    "available_trackers",
    "describe_trackers",
]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "stable.trackers"
"""The entry-point group third-party trackers advertise themselves under."""


class TrackerRegistry:
    """A name to :class:`~stable.trackers.base.TrackerPlugin` class mapping.

    The process-wide instance is :data:`registry`; separate instances are handy
    in tests for isolation.

    Args:
        group: Entry-point group to search during :meth:`discover`.
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        """Create an empty registry that will search ``group`` on discovery."""
        self.group = group
        self._plugins: dict[str, type[TrackerPlugin]] = {}
        self._discovered = False

    # -- registration ----------------------------------------------------

    def register(
        self, plugin: type[TrackerPlugin], *, replace: bool = False
    ) -> type[TrackerPlugin]:
        """Add a plugin class to the registry.

        Registering the *same* class twice is a no-op, so a plugin that is both
        a built-in and an advertised entry point registers cleanly.

        Args:
            plugin: A concrete :class:`TrackerPlugin` subclass with a non-empty
                ``name``.
            replace: Allow overriding a different plugin already using that name.

        Returns:
            ``plugin`` unchanged, so this can be used as a class decorator.

        Raises:
            TrackerError: If ``plugin`` is not a concrete ``TrackerPlugin``
                subclass, has no ``name``, or collides with a different class
                and ``replace`` is False.
        """
        if not (inspect.isclass(plugin) and issubclass(plugin, TrackerPlugin)):
            raise TrackerError(f"{plugin!r} is not a TrackerPlugin subclass")
        if inspect.isabstract(plugin):
            raise TrackerError(f"{plugin.__name__} is abstract; implement track()")
        name = getattr(plugin, "name", "")
        if not name or not isinstance(name, str):
            raise TrackerError(f"{plugin.__name__} must define a non-empty 'name' class attribute")
        existing = self._plugins.get(name)
        if existing is not None and existing is not plugin and not replace:
            raise TrackerError(
                f"tracker name {name!r} is already registered by {existing.__module__}."
                f"{existing.__qualname__}"
            )
        self._plugins[name] = plugin
        return plugin

    def unregister(self, name: str) -> None:
        """Remove a plugin by name; unknown names are ignored."""
        self._plugins.pop(name, None)

    # -- discovery -------------------------------------------------------

    def discover(
        self,
        *,
        entry_points: Iterable[Any] | None = None,
        strict: bool = False,
        force: bool = False,
    ) -> list[str]:
        """Load built-ins and every advertised entry point.

        Args:
            entry_points: Iterable of objects with ``.name`` and ``.load()``.
                Defaults to the installed distributions' ``stable.trackers``
                group. Injecting a list makes discovery testable without
                installing packages.
            strict: Re-raise plugin import errors instead of logging them.
            force: Re-run discovery even if it already ran.

        Returns:
            The sorted names of every registered tracker after discovery.

        Raises:
            TrackerError: Only when ``strict`` is set and an entry point fails
                to load or register.
        """
        if self._discovered and not force and entry_points is None:
            return self.names()
        self._register_builtins()
        if entry_points is None:
            try:
                entry_points = metadata.entry_points(group=self.group)
            except Exception as exc:  # pragma: no cover - broken metadata only
                if strict:
                    raise TrackerError(f"cannot read entry points: {exc}") from exc
                logger.warning("could not read %s entry points: %s", self.group, exc)
                entry_points = []
        for entry in entry_points:
            try:
                self.register(entry.load())
            except Exception as exc:
                if strict:
                    raise TrackerError(f"tracker entry point {entry.name!r} failed: {exc}") from exc
                logger.warning("skipping tracker entry point %r: %s", entry.name, exc)
        self._discovered = True
        return self.names()

    def _register_builtins(self) -> None:
        """Register the trackers that ship inside this package."""
        from .dummy import DummyTracker  # noqa: PLC0415 - avoids an import cycle

        self.register(DummyTracker)

    # -- lookup ----------------------------------------------------------

    def get(self, name: str) -> type[TrackerPlugin]:
        """Return the plugin class registered under ``name``.

        Discovery runs automatically on first use.

        Args:
            name: The tracker's machine name, as stored in project files.

        Returns:
            The plugin class (not an instance).

        Raises:
            TrackerNotFoundError: If no plugin uses that name.
        """
        self.discover()
        try:
            return self._plugins[name]
        except KeyError:
            raise TrackerNotFoundError(
                f"no tracker named {name!r}; available: {self.names()}"
            ) from None

    def create(
        self,
        name: str,
        *,
        device: str | None = None,
        prepare: bool = True,
        options: dict[str, Any] | None = None,
    ) -> TrackerPlugin:
        """Instantiate a tracker and (by default) prepare it on a device.

        Args:
            name: The tracker's machine name.
            device: Device string; ``None`` selects automatically
                (``cuda`` when available, else ``cpu``).
            prepare: Call :meth:`~stable.trackers.base.TrackerPlugin.prepare`.
            options: Constructor keyword arguments for the plugin, e.g.
                ``{"template_radius": 20}`` — how a per-point tracker setting
                reaches the plugin that honours it (todo 14). Keys the plugin
                does not accept are **ignored**, not an error: options are
                stored per point and follow it if its tracker is changed, and a
                stale key from a previous tracker must not stop it running.

        Returns:
            A ready-to-use plugin instance.

        Raises:
            TrackerNotFoundError: If no plugin uses that name.
            TrackerError: If the plugin needs a GPU and none is available.
        """
        plugin_cls = self.get(name)
        resolved = select_device(device)
        if plugin_cls.requires_gpu and resolved == "cpu":
            raise TrackerError(f"tracker {name!r} requires a GPU but no CUDA device is available")
        plugin = plugin_cls(**self._accepted_options(plugin_cls, options))
        if prepare:
            plugin.prepare(resolved)
        return plugin

    @staticmethod
    def _accepted_options(
        plugin_cls: type[TrackerPlugin], options: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Return the subset of ``options`` that ``plugin_cls.__init__`` accepts."""
        if not options:
            return {}
        try:
            parameters = inspect.signature(plugin_cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            return {}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(options)
        return {key: value for key, value in options.items() if key in parameters}

    def names(self) -> list[str]:
        """Return every registered tracker name, sorted."""
        return sorted(self._plugins)

    def items(self) -> list[tuple[str, type[TrackerPlugin]]]:
        """Return ``(name, class)`` pairs for every registered tracker, sorted."""
        return sorted(self._plugins.items())

    def describe(self) -> list[dict[str, Any]]:
        """Return UI-ready metadata for every registered tracker.

        Returns:
            One dict per tracker with keys ``name``, ``display_name``,
            ``requires_gpu``, ``license`` and ``available``.
        """
        self.discover()
        return [
            {
                "name": plugin.name,
                "display_name": plugin.display_name or plugin.name,
                "requires_gpu": bool(plugin.requires_gpu),
                "license": plugin.license,
                "available": plugin.is_available(),
            }
            for _, plugin in self.items()
        ]

    def __contains__(self, name: object) -> bool:
        """Whether a tracker with this name is registered (no discovery)."""
        return name in self._plugins

    def __len__(self) -> int:
        """Number of registered trackers (no discovery)."""
        return len(self._plugins)

    def __repr__(self) -> str:
        """Return a short debug representation listing registered names."""
        return f"<TrackerRegistry group={self.group!r} trackers={self.names()}>"


registry = TrackerRegistry()
"""The process-wide tracker registry."""


def register_tracker(plugin: type[TrackerPlugin], *, replace: bool = False) -> type[TrackerPlugin]:
    """Register a plugin in the process-wide :data:`registry`."""
    return registry.register(plugin, replace=replace)


def get_tracker(name: str) -> type[TrackerPlugin]:
    """Return the plugin class named ``name`` from the process-wide registry."""
    return registry.get(name)


def create_tracker(
    name: str,
    *,
    device: str | None = None,
    prepare: bool = True,
    options: dict[str, Any] | None = None,
) -> TrackerPlugin:
    """Instantiate a prepared tracker from the process-wide registry."""
    return registry.create(name, device=device, prepare=prepare, options=options)


def available_trackers() -> list[str]:
    """Return every discoverable tracker name, sorted."""
    return registry.discover()


def describe_trackers() -> list[dict[str, Any]]:
    """Return UI-ready metadata for every discoverable tracker."""
    return registry.describe()
