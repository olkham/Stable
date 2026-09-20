"""Where Stable keeps its data: projects, model checkpoints, caches.

Historically every root was hard-coded under the user's home directory
(``~/.stable/...``). That is the wrong drive on a very common setup: the tool
is installed on a big data disk precisely *because* projects are huge (proxies
and renders of a single clip run to gigabytes), while the home directory sits
on a small system SSD that then fills up.

So the default root now follows **the installation the process is running
from**, and falls back to the home directory only when there is nowhere better:

#. ``$STABLE_HOME``, if set — the explicit answer always wins.
#. The source checkout / editable install this package lives in (the directory
   holding ``pyproject.toml`` next to ``stable/``), as ``<checkout>/.stable``.
   This is what ``install.sh`` / ``install.bat`` produce, so it is the normal
   case for anyone who installed by cloning.
#. The active virtual environment (``<sys.prefix>/.stable``), for a wheel
   installed into a project-local venv — still "next to the app", still off
   the system drive if the venv is.
#. ``~/.stable`` — a system-wide ``pip install``, where writing next to the
   package would mean writing into ``site-packages``.

The per-store overrides (``$STABLE_WORKSPACE_DIR``, ``$STABLE_MODEL_DIR``)
still take precedence over all of this; see :mod:`stable.server.config` and
:mod:`stable.trackers.weights`. Nothing here creates a directory — the module
that owns a store creates it when it first needs it.

A store left behind under ``~/.stable`` by an older version is moved into the
new root on the next launch; that is :mod:`stable.core.migrate_storage`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["ENV_HOME", "LEGACY_HOME", "checkout_root", "data_root", "legacy_root"]

ENV_HOME = "STABLE_HOME"
"""Environment variable overriding the whole data root (projects, models, caches)."""

LEGACY_HOME = Path("~/.stable")
"""Where every store lived before the root followed the installation."""

_DIR_NAME = ".stable"
_CHECKOUT_MARKER = "pyproject.toml"


def _writable(path: Path) -> bool:
    """Return whether ``path`` (or its nearest existing parent) can be written to.

    Args:
        path: The candidate directory, which need not exist yet.

    Returns:
        Whether the first existing directory at or above ``path`` is writable.
        Best-effort: Windows reports directory permissions only through the
        read-only flag, so a genuinely unwritable location there is caught when
        the store is created rather than here.
    """
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return os.access(candidate, os.W_OK)
    return False


def checkout_root() -> Path | None:
    """Return the source checkout this package is installed from, if it is one.

    Returns:
        The directory holding ``pyproject.toml`` alongside the ``stable``
        package — a git clone or an ``pip install -e`` — or ``None`` when the
        package was installed from a wheel into ``site-packages``.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / _CHECKOUT_MARKER).is_file() else None


def _venv_root() -> Path | None:
    """Return the active virtual environment's directory, if there is one.

    Returns:
        ``sys.prefix`` when it differs from ``sys.base_prefix`` (the standard
        way to detect a venv), else ``None``.
    """
    prefix = Path(sys.prefix).resolve()
    return prefix if prefix != Path(sys.base_prefix).resolve() else None


def legacy_root() -> Path:
    """Return the pre-relocation data root, ``~/.stable``."""
    return LEGACY_HOME.expanduser()


def data_root() -> Path:
    """Return the directory every Stable store lives under.

    Resolution order is the one in the module docstring: ``$STABLE_HOME``, the
    source checkout, the active virtualenv, then ``~/.stable``. A candidate
    that is not writable is skipped rather than returned, so an app installed
    read-only (a system package, a mounted image) still works.

    Returns:
        The root directory. It is not created here.
    """
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    for base in (checkout_root(), _venv_root()):
        if base is not None:
            candidate = base / _DIR_NAME
            if _writable(candidate):
                return candidate
    return legacy_root()
