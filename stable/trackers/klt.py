"""KLT (Kanade-Lucas-Tomasi) — pyramidal LK with Shi-Tomasi corner re-detection.

Todo 27 asks for a KLT tracker, while noting that :mod:`stable.trackers.lk` is
already pyramidal Lucas-Kanade and warning against shipping a near-duplicate.
The two are not the same algorithm, though the difference is easy to miss
because "KLT" and "LK" get used interchangeably in casual writing:

* ``lk`` (:class:`~stable.trackers.lk.LKTracker`) is pure optical flow: it
  takes whatever pixel the user (or the previous frame) placed a point at and
  tracks *that*, with no opinion on whether it is a good pixel to track.
* **KLT proper** — the combination Tomasi & Kanade actually described — adds
  the other half: periodically re-verify the tracked position against the
  Shi-Tomasi minimum-eigenvalue "cornerness" criterion, and re-seed tracking
  from the strongest nearby corner when one exists close by. A point sitting
  on a smooth gradient or a repeating texture (not a true corner) drifts
  under pure LK in a way re-detection actively corrects; a point already
  sitting on a strong corner is left alone (the search only *drifts toward* a
  better corner, it does not run away to an unrelated one — see
  ``redetect_max_shift`` on :class:`~stable.trackers.lk.LKTracker`).

Rather than duplicating the pyramidal LK stepping loop, forward-backward
confidence check, direction handling and contract validation, this is a thin
subclass of :class:`~stable.trackers.lk.LKTracker` that simply turns its
optional re-detection parameters on by default and registers under its own
name — the substantive feature (``LKTracker._redetect_corner`` and the
re-detection step in ``LKTracker._run_direction``) lives once, in ``lk.py``,
and is exercised by both plugins.
"""

from __future__ import annotations

from typing import Any

from .lk import LKTracker

__all__ = ["KLTTracker"]


class KLTTracker(LKTracker):
    """Pyramidal LK with Shi-Tomasi corner re-detection turned on by default.

    Every argument is forwarded to :class:`~stable.trackers.lk.LKTracker`; only
    the re-detection defaults differ (enabled here, disabled there) and the
    registered ``name``/``display_name`` differ, so the two show up as
    distinct, independently selectable trackers in the UI's picker and in
    project files.

    Args:
        redetect_interval: Frames between corner re-detection attempts;
            defaults to every 5 frames here (``0``, disabled, is ``lk``'s
            default).
        **kwargs: Forwarded to :class:`~stable.trackers.lk.LKTracker`
            (``win_size``, ``max_level``, ``fb_threshold``,
            ``redetect_search_radius``, ``redetect_quality_level``,
            ``redetect_min_distance``, ``redetect_max_shift``, ...).
    """

    name = "klt"
    display_name = "KLT (Kanade-Lucas-Tomasi)"
    requires_gpu = False
    license = "MIT"

    def __init__(self, *, redetect_interval: int = 5, **kwargs: Any) -> None:
        """Create a KLT tracker; see the class docstring for the arguments."""
        super().__init__(redetect_interval=redetect_interval, **kwargs)
