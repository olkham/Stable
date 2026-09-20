"""Server-wide configuration: where projects and their derived data live.

The server is a single local process (see the module docstring in
``stable/server/__init__.py``), so configuration is deliberately tiny: one
directory tree under :func:`stable.core.paths.data_root` — which follows the
installation, not the user's home directory, because projects are gigabytes
and the app is normally installed on the drive that has room for them —
overridable for tests and Docker via an environment variable, the same pattern
:mod:`stable.trackers.weights` uses for the model cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.paths import data_root

__all__ = ["ENV_WORKSPACE_DIR", "WORKSPACE_DIR_NAME", "workspace_root"]

ENV_WORKSPACE_DIR = "STABLE_WORKSPACE_DIR"
"""Environment variable overriding the workspace root (projects live under it)."""

WORKSPACE_DIR_NAME = "projects"
"""The workspace's sub-directory name inside the data root."""


def workspace_root() -> Path:
    """Return the directory server projects are stored under.

    Returns:
        ``$STABLE_WORKSPACE_DIR`` when set, else ``projects/`` inside
        :func:`stable.core.paths.data_root`. Not created here —
        :class:`stable.server.storage.Workspace` creates it lazily.
    """
    override = os.environ.get(ENV_WORKSPACE_DIR)
    if override:
        return Path(override).expanduser()
    return data_root() / WORKSPACE_DIR_NAME
