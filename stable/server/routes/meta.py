"""Health check, tracker/filter catalogues and device info — no project required."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ... import __version__
from ...core.device import device_info
from ...core.project import BORDER_POLICIES
from ...smoothing.filters import describe_filters
from ...trackers.registry import describe_trackers

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
