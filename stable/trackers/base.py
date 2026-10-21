"""The tracker plugin contract — the swappability core of the project.

Every tracker, from OpenCV Lucas-Kanade to CoTracker3, implements
:class:`TrackerPlugin` and returns a :class:`~stable.core.types.TrackResult`.
Nothing downstream (solver, smoothing, render, UI) knows which plugin produced a
track, which is what makes swapping in the next SOTA model a packaging problem
rather than a code change.

Contract rules a plugin must honour
-----------------------------------
#. ``track`` returns arrays covering the **whole clip timeline** ``T =
   video.frame_count``, indexed by absolute frame number. Frames outside
   ``frame_range`` are ``NaN`` / confidence ``0`` / ``visible False``.
#. Point order in the result matches the order of ``queries``.
#. ``confidence`` is normalised to ``[0, 1]`` *by the plugin*, so the solver can
   weight all trackers identically.
#. ``track`` performs **one run** — a single pass from the given queries. It
   never merges multiple anchors; :mod:`stable.trackers.merge` does that,
   tracker-agnostically.
#. A plugin that sets ``tracks_boxes`` starts from the anchor's box
   (``Query.w``/``Query.h``, ``Query.position`` being its centre) and reports
   ``TrackResult.sizes``. One that does not ignores the box, tracks the centre
   and returns no sizes; the merge layer copes with either.
#. ``progress`` is called with a fraction in ``[0, 1]`` often enough for a UI
   progress bar, and the plugin must tolerate it being a no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol

from ..core.types import Query, TrackResult
from ..core.video import VideoSource

__all__ = ["TrackerPlugin", "Direction", "ProgressCallback", "noop_progress"]

Direction = Literal["fwd", "bwd", "bidir"]
"""Direction of a tracking run relative to the query frame."""


class ProgressCallback(Protocol):
    """Callable a plugin uses to report progress.

    Args:
        fraction: Completion in ``[0, 1]``.
        message: Optional short human-readable status.
    """

    def __call__(self, fraction: float, message: str = "") -> None:
        """Report progress; implementations must not raise."""
        ...


def noop_progress(fraction: float, message: str = "") -> None:
    """A :class:`ProgressCallback` that discards its arguments.

    Used as the default so plugins can always call ``progress(...)`` without a
    None check.

    Args:
        fraction: Ignored.
        message: Ignored.
    """


class TrackerPlugin(ABC):
    """Base class for every point tracker.

    Subclasses set the four class attributes below and implement :meth:`track`.
    They are discovered through the ``stable.trackers`` entry-point group; see
    :mod:`stable.trackers.registry`.

    Attributes:
        name: Machine name, stable across versions — this is what project files
            store in ``points[].tracker``.
        display_name: Human label for the UI's tracker picker.
        requires_gpu: True if the plugin is unusable without a GPU. Plugins that
            merely *prefer* one set this False and fall back internally.
        license: SPDX-ish licence of the plugin **and its weights**. Non-commercial
            models (e.g. CoTracker's ``CC-BY-NC-4.0``) must say so here; the UI
            surfaces it and never bundles such weights.
        tracks_boxes: True if the plugin starts from an anchor's box
            (``Query.w``/``Query.h``) and reports ``TrackResult.sizes``. A plugin
            that does not ignores the box and tracks the centre
            (``Query.position``), returning no sizes.
    """

    name: str = ""
    display_name: str = ""
    requires_gpu: bool = False
    license: str = "MIT"
    tracks_boxes: bool = False

    def __init__(self) -> None:
        """Create an unprepared plugin instance (no weights loaded yet)."""
        self.device: str = "cpu"
        self._prepared: bool = False

    # -- lifecycle -------------------------------------------------------

    def prepare(self, device: str) -> None:
        """Load weights and move the model onto ``device``.

        Called once before the first :meth:`track` call. The default
        implementation just records the device, which is all a classic-CV
        plugin needs. Deep-learning plugins override it to load weights (and
        should be idempotent — re-preparing the same device must be cheap).

        Args:
            device: A device string from
                :func:`stable.core.device.select_device`, e.g. ``"cuda"``.
        """
        self.device = device
        self._prepared = True

    def release(self) -> None:
        """Free heavy resources (weights, GPU memory). Safe to call twice."""
        self._prepared = False

    @property
    def is_prepared(self) -> bool:
        """Whether :meth:`prepare` has run since the last :meth:`release`."""
        return self._prepared

    @classmethod
    def is_available(cls) -> bool:
        """Whether this plugin can run in the current environment.

        Override to report missing optional dependencies or weights. The UI uses
        this to grey out a tracker instead of failing at run time.

        Returns:
            True by default.
        """
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        """Why :meth:`is_available` said no, and what would fix it.

        A greyed-out tracker with no explanation is indistinguishable from a
        broken one: the operator cannot tell whether to install something, wait
        for a download, or pick a different plugin. Overriding this turns "not
        available" into an instruction.

        Keep it to one sentence, name the command where there is one, and do not
        touch the network — this is called to render a list.

        Returns:
            ``None`` when the plugin is available (the default, so a plugin that
            is always available need not override it), otherwise a short
            human-readable reason.
        """
        return None

    # -- the one method that matters -------------------------------------

    @abstractmethod
    def track(
        self,
        video: VideoSource,
        queries: list[Query],
        frame_range: tuple[int, int],
        direction: Direction = "bidir",
        progress: ProgressCallback = noop_progress,
    ) -> TrackResult:
        """Track every query point across ``frame_range``. One run, no merging.

        Args:
            video: Clip to read frames from.
            queries: Anchor positions to track, one per point. When several
                queries share a point, the caller has already split them into
                separate runs — see :mod:`stable.trackers.merge`.
            frame_range: ``(start, end)`` inclusive-exclusive absolute frame
                range to cover, clipped to the clip length by the caller.
            direction: ``"fwd"`` tracks forward from each query frame, ``"bwd"``
                backward, ``"bidir"`` both.
            progress: Progress reporter; see :class:`ProgressCallback`.

        Returns:
            A :class:`~stable.core.types.TrackResult` with ``N = len(queries)``
            and ``T = video.frame_count``, in query order.

        Raises:
            TrackerError: If tracking cannot run (bad range, unloaded weights).
        """
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------

    def __repr__(self) -> str:
        """Return a short debug representation naming the plugin and device."""
        return f"<{type(self).__name__} name={self.name!r} device={self.device!r}>"
