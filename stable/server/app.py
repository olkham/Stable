"""The FastAPI application: project CRUD, frame serving, jobs, WebSocket progress.

``create_app()`` is the single factory used both by tests (via
``fastapi.testclient.TestClient``) and by :mod:`stable.server.__main__` (via
``uvicorn``). Every ``StableError`` raised by the tracking/video/core stack is
translated into a clean HTTP response here rather than a raw 500, so a broken
tracker plugin or a missing proxy tier surfaces as a real error message in the
UI instead of an opaque stack trace.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..core.errors import StableError
from .routes import edits, export, meta, models, points, projects, solve, ws
from .storage import ProjectNotFoundError, get_workspace

__all__ = ["create_app", "web_dist_dir", "DEV_SERVER_ORIGINS"]

DEV_SERVER_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
"""The only cross-origin callers allowed: the Vite dev server (``web/vite.config.ts``)."""

_PACKAGED_WEB_DIST = Path(__file__).resolve().parent / "webdist"
"""Where a packaged build of the frontend lives inside the installed package.

Populated by ``scripts/build_frontend.py`` before the wheel is built (see
``pyproject.toml``'s package-data) so a real ``pip install`` ships a working
UI with no Node/npm on the end user's machine — the Stack decision is "single
local process the user runs", not "...and also installs a JS toolchain".
"""

_SOURCE_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
"""Where ``cd web && npm run build`` puts its output in a source checkout."""


def web_dist_dir() -> Path | None:
    """Return the built frontend's directory, preferring the packaged copy.

    Returns:
        :data:`_PACKAGED_WEB_DIST` if it holds a build (the common case for a
        ``pip install``), else :data:`_SOURCE_WEB_DIST` if *that* holds one
        (a repo checkout that ran the Vite build itself), else ``None``.
    """
    for candidate in (_PACKAGED_WEB_DIST, _SOURCE_WEB_DIST):
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _register_error_handlers(app: FastAPI) -> None:
    """Map Stable's own exception hierarchy onto HTTP status codes.

    Registered from most specific to least: Starlette walks each exception's
    MRO and uses the first handler it finds, so :class:`ProjectNotFoundError`
    gets its own 404 while every other :class:`StableError` falls through to
    the generic 400 below it.
    """

    @app.exception_handler(ProjectNotFoundError)
    async def _not_found(_request: Request, exc: ProjectNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(StableError)
    async def _bad_request(_request: Request, exc: StableError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Close every cached video source on shutdown (startup needs nothing)."""
    try:
        yield
    finally:
        get_workspace().close()


def create_app() -> FastAPI:
    """Build and return the Stable FastAPI application.

    Returns:
        A fully configured app: CORS limited to the Vite dev server, every REST
        router mounted under ``/api``, the job-progress WebSocket at
        ``/ws/projects/{project_id}`` and ``/ws/models``, and — if a built
        frontend can be found (see :func:`web_dist_dir`) — that build served
        as static files at ``/`` so the whole tool is one process, matching
        the Stack decision "single local process the user runs".
    """
    app = FastAPI(title="Stable", version=__version__, lifespan=_lifespan)

    # The frontend reaches the API through Vite's dev-server proxy or through
    # the static mount below — same-origin either way, so CORS is never what
    # makes the UI work. It used to allow every origin, which let any web page
    # the operator happened to visit drive this server (delete projects, start
    # jobs) and read its answers: "localhost only, no credentials" is exactly
    # what a drive-by request does not need. Only the dev server's own origins
    # are allowed, for anything run against it directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_SERVER_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_error_handlers(app)

    app.include_router(meta.router, prefix="/api")
    app.include_router(projects.router, prefix="/api")
    app.include_router(points.router, prefix="/api")
    app.include_router(edits.router, prefix="/api")
    app.include_router(solve.router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(ws.router)

    dist = web_dist_dir()
    if dist is not None:
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    else:

        @app.get("/")
        def _no_frontend() -> dict[str, Any]:
            return {
                "status": "ok",
                "detail": (
                    "no built frontend found; run `stable serve` (which builds it) or "
                    "`cd web && npm install && npm run build`, or use the Vite dev "
                    "server (`cd web && npm run dev`) and its URL instead"
                ),
            }

    return app
