"""Server-wide configuration: where projects and their derived data live.

The server is a single local process (see the module docstring in
``stable/server/__init__.py``), so configuration is deliberately tiny: one
directory tree under the user's home, overridable for tests and Docker via an
environment variable — the same pattern :mod:`stable.trackers.weights` uses
for the model cache.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ENV_WORKSPACE_DIR", "workspace_root"]

ENV_WORKSPACE_DIR = "STABLE_WORKSPACE_DIR"
"""Environment variable overriding the workspace root (projects live under it)."""

_DEFAULT_ROOT = Path("~/.stable/projects")


def workspace_root() -> Path:
    """Return the directory server projects are stored under.

    Returns:
        ``$STABLE_WORKSPACE_DIR`` when set, else ``~/.stable/projects``. Not
        created here — :class:`stable.server.storage.Workspace` creates it lazily.
    """
    override = os.environ.get(ENV_WORKSPACE_DIR)
    return Path(override).expanduser() if override else _DEFAULT_ROOT.expanduser()
