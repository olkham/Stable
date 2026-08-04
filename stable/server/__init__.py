"""Local FastAPI server: project CRUD, frame serving, job queue, progress.

A single local process the user launches (``python -m stable.server`` today;
``stable serve`` in Phase 6). It never talks to a remote service. See
:func:`create_app` for the application factory and ``PLAN.md``'s Phase 4
section for the full feature list.

Submodules:

* :mod:`stable.server.storage` — on-disk project layout (CRUD, video/proxy caches).
* :mod:`stable.server.jobs` — the background job queue (track/solve/render/export).
* :mod:`stable.server.tracking` — the real multi-anchor tracking pipeline as a job.
* :mod:`stable.server.motion` — solve/smooth/render glue over ``stable.solver``,
  ``stable.smoothing`` and ``stable.render``, plus the on-disk curve cache.
* :mod:`stable.server.history` — the undo/redo stack over the corrections layer.
* :mod:`stable.server.routes` — the FastAPI routers, one module per resource.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
