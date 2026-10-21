"""Health check, tracker/filter catalogues and device info — no project required."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ... import __version__
from ...core.device import device_info
from ...core.errors import StableError
from ...core.project import BORDER_POLICIES
from ...smoothing.filters import describe_filters
from ...trackers.registry import describe_trackers, get_tracker, tracker_inputs
from ..tracker_options import (
    box_search_scale,
    check_tracker_options,
    tracker_search,
    tracker_settings,
    tracker_window,
)

__all__ = ["router"]

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe for the frontend's initial connection check."""
    return {"status": "ok", "version": __version__}


@router.get("/trackers")
def trackers() -> list[dict[str, Any]]:
    """List every discoverable tracker plugin, for the tracker picker.

    Each entry carries ``available`` (False greys the option out in the UI
    rather than letting a track job fail on a missing dependency or GPU).
    """
    return describe_trackers()


@router.get("/trackers/{name}/settings")
def tracker_settings_preview(name: str, request: Request, box: bool = False) -> dict[str, Any]:
    """Describe a tracker's settings before any track uses it.

    A new track is set up tracker first: the operator picks the tracker, sizes
    its pattern and search area, and only then marks the video. Every other
    description of these settings belongs to an existing point, so this one
    answers for a tracker alone, for whatever values the sliders are at.

    Query parameters are candidate values under the settings' own argument
    names (``?template_radius=24&search_radius=60``); any left out take the
    plugin's default, exactly as they would in a run.

    Returns:
        ``tracker``; ``inputs``, what a track can be started from (preferred
        first); ``settings``, one row per control, shaped as the patch panel's;
        and ``window`` / ``search``, the regions those values describe in
        source pixels from the point — what the viewer previews under the
        cursor before the point is placed. ``search.radius`` is ``None`` for a
        tracker with no search area, with ``note`` saying why.

    Raises:
        HTTPException: 404 for an unknown tracker; 400 for a setting it does
            not take or a value out of range.
    """
    try:
        plugin = get_tracker(name)
    except StableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        candidate = dict(request.query_params)
        candidate.pop("box", None)
        options = check_tracker_options(name, candidate)
    except StableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    radius, attribute = tracker_window(name, options)
    search, search_attribute, note = tracker_search(name, options)
    if box:
        radius, attribute = 0, "box"
        search, search_attribute = None, None
        note = "Searched each frame around the box's last position, scaled to its size."
    return {
        "tracker": name,
        "input": "box" if box else "point",
        "inputs": tracker_inputs(plugin),
        "settings": tracker_settings(name, options, box=box),
        "window": {"radius": radius, "attribute": attribute},
        "search": {
            "radius": search,
            "attribute": search_attribute,
            "note": note,
            "box_scale": box_search_scale(name, options) if box else None,
        },
    }


@router.get("/filters")
def filters() -> list[dict[str, Any]]:
    """List every smoothing filter with its parameter schema.

    The smoothing panel and the curve editor build their controls from this
    rather than hardcoding parameter names, so a filter added to
    :data:`stable.smoothing.filters.FILTERS` appears in the UI with the right
    inputs, defaults and bounds — and a rename can never silently desync the
    two halves of the app.
    """
    return describe_filters()


@router.get("/border-policies")
def border_policies() -> list[str]:
    """List the stabilized render's border policies, for the export dialog."""
    return list(BORDER_POLICIES)


@router.get("/devices")
def devices() -> dict[str, Any]:
    """Report the compute devices this server can use."""
    return device_info()
