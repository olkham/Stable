"""Compute device selection: ``cuda`` when available, ``cpu`` otherwise.

Stack rule from ``PLAN.md``: *CUDA when available, CPU fallback always*. So the
default behaviour of :func:`select_device` never raises — a missing GPU, a
missing PyTorch install or a broken driver all degrade to ``"cpu"``. Callers
that genuinely require a GPU pass ``strict=True`` and handle
:class:`~stable.core.errors.DeviceError`.

PyTorch is an optional dependency (``pip install stable-tracker[torch]``); this
module imports it lazily so the classic-CV path never pays for it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .errors import DeviceError

__all__ = ["select_device", "available_devices", "device_info", "torch_module"]

logger = logging.getLogger(__name__)

_ENV_OVERRIDE = "STABLE_DEVICE"
"""Environment variable that overrides automatic device selection."""


def torch_module() -> Any | None:
    """Return the imported :mod:`torch` module, or ``None`` if unavailable.

    Returns:
        The module object, or ``None`` when PyTorch is not installed or fails to
        import (a broken CUDA build, for instance).
    """
    try:
        import torch  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # pragma: no cover - depends on the environment
        logger.debug("torch unavailable: %s", exc)
        return None
    return torch


def _cuda_available(torch: Any) -> bool:
    """Return whether torch reports a usable CUDA device."""
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver-dependent
        logger.debug("cuda probe failed: %s", exc)
        return False


def _mps_available(torch: Any) -> bool:
    """Return whether torch reports a usable Apple MPS device."""
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:  # pragma: no cover - platform-dependent
        logger.debug("mps probe failed: %s", exc)
        return False


def available_devices() -> list[str]:
    """List the device strings usable in this process, best first.

    Returns:
        e.g. ``["cuda", "cpu"]`` on a GPU box, ``["cpu"]`` without PyTorch.
    """
    devices: list[str] = []
    torch = torch_module()
    if torch is not None:
        if _cuda_available(torch):
            devices.append("cuda")
        if _mps_available(torch):
            devices.append("mps")
    devices.append("cpu")
    return devices


def select_device(preferred: str | None = None, *, strict: bool = False) -> str:
    """Pick the device to run a tracker on.

    Resolution order: explicit ``preferred`` argument, then the ``STABLE_DEVICE``
    environment variable, then automatic (``cuda`` if available, else ``cpu``).

    Args:
        preferred: A device string such as ``"cuda"``, ``"cuda:1"``, ``"cpu"`` or
            ``"auto"``. ``None`` or ``"auto"`` means automatic selection.
        strict: When True, an unavailable explicit request raises instead of
            falling back to ``"cpu"``.

    Returns:
        A device string safe to pass to ``torch.device`` and to
        :meth:`~stable.trackers.base.TrackerPlugin.prepare`.

    Raises:
        DeviceError: If ``strict`` is set and the requested device is
            unavailable, or if the device string is not recognised.
    """
    requested = preferred if preferred is not None else os.environ.get(_ENV_OVERRIDE)
    if requested is None or requested.strip().lower() in ("", "auto"):
        return available_devices()[0]

    requested = requested.strip().lower()
    family, _, index = requested.partition(":")
    if family not in ("cpu", "cuda", "mps"):
        raise DeviceError(
            f"unknown device {requested!r}; expected 'cpu', 'cuda[:N]', 'mps' or 'auto'"
        )
    if index and not index.isdigit():
        raise DeviceError(f"malformed device index in {requested!r}")

    if family == "cpu":
        return "cpu"

    torch = torch_module()
    ok = torch is not None and (
        (family == "cuda" and _cuda_available(torch)) or (family == "mps" and _mps_available(torch))
    )
    if ok and family == "cuda" and index and torch is not None:
        count = int(torch.cuda.device_count())
        if int(index) >= count:
            ok = False
            if strict:
                raise DeviceError(f"cuda:{index} requested but only {count} CUDA device(s) present")
    if ok:
        return requested
    if strict:
        raise DeviceError(
            f"device {requested!r} was requested but is not available "
            f"(usable devices: {available_devices()})"
        )
    logger.warning("device %r unavailable, falling back to cpu", requested)
    return "cpu"


def device_info() -> dict[str, Any]:
    """Return a diagnostics dict about the compute environment.

    Returns:
        Keys: ``torch`` (version or ``None``), ``cuda`` (bool),
        ``cuda_devices`` (list of names), ``mps`` (bool), ``selected``
        (what :func:`select_device` would return now).
    """
    torch = torch_module()
    info: dict[str, Any] = {
        "torch": getattr(torch, "__version__", None),
        "cuda": False,
        "cuda_devices": [],
        "mps": False,
    }
    if torch is not None:
        info["cuda"] = _cuda_available(torch)
        info["mps"] = _mps_available(torch)
        if info["cuda"]:
            try:  # pragma: no cover - requires a GPU
                info["cuda_devices"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
            except Exception as exc:  # pragma: no cover - driver-dependent
                logger.debug("could not enumerate CUDA devices: %s", exc)
    info["selected"] = select_device()
    return info
