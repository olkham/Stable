"""The ``stable serve`` subcommand: build the frontend, serve it, open a browser.

Headless-to-browser entry point for::

    stable serve
    stable serve --port 8080 --no-browser

This is the one-command launch PLAN.md's Phase 6 promises — the polished
successor to :mod:`stable.server.__main__` (which Phase 4 added only as a
thin dev/test runner, deliberately leaving this command for Phase 6). It:

#. Looks for an already-built frontend via
   :func:`stable.server.app.web_dist_dir`.
#. If none is found *and* a ``web/`` source checkout with a ``package.json``
   sits next to the installed package (a repo checkout or editable install),
   builds it with ``npm install``/``npm run build``. A real ``pip install``
   from a wheel ships the frontend already built (see ``pyproject.toml``'s
   package-data and ``scripts/build_frontend.py``), so this path is a source-
   checkout convenience — an end user with no Node.js installed never needs it.
#. Starts uvicorn and, once the port answers, opens the default browser.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

__all__ = ["add_serve_parser", "run_serve"]


def add_serve_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Attach the ``serve`` subcommand to ``subparsers``.

    Args:
        subparsers: The action object returned by ``ArgumentParser.add_subparsers``.

    Returns:
        The new subcommand's parser (for tests that want to inspect it).
    """
    parser = subparsers.add_parser(
        "serve",
        help="build (if needed) and serve the web UI, opening a browser",
        description=(
            "Build the frontend if it is not already built, start the local server, "
            "and open it in the default browser -- the whole tool as one command."
        ),
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port to bind (default: %(default)s)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser automatically"
    )
    parser.add_argument(
        "--no-build", action="store_true", help="never attempt to build the frontend"
    )
    parser.add_argument(
        "--reload", action="store_true", help="auto-reload on source changes (development)"
    )
    return parser


def _source_web_dir() -> Path | None:
    """Return the ``web/`` source checkout's directory, if this looks like one.

    Returns:
        The directory, if it holds a ``package.json`` (a repo checkout or
        editable install sitting next to ``stable/``); else ``None``.
    """
    candidate = Path(__file__).resolve().parents[2] / "web"
    return candidate if (candidate / "package.json").is_file() else None


def _build_frontend(web_dir: Path) -> bool:
    """Run ``npm install`` (if needed) then ``npm run build`` in ``web_dir``.

    Args:
        web_dir: The frontend's source directory.

    Returns:
        Whether the build succeeded.
    """
    npm = shutil.which("npm")
    if npm is None:
        print(
            "error: no built frontend found and `npm` is not on PATH, so it cannot be "
            f"built automatically. Install Node.js, then run:\n"
            f"  cd {web_dir} && npm install && npm run build",
            file=sys.stderr,
        )
        return False
    if not (web_dir / "node_modules").is_dir():
        print(f"installing frontend dependencies ({web_dir})...")
        if subprocess.run([npm, "install"], cwd=str(web_dir)).returncode != 0:
            return False
    print("building frontend...")
    return subprocess.run([npm, "run", "build"], cwd=str(web_dir)).returncode == 0


def _wait_until_serving(url: str, *, timeout: float = 20.0) -> bool:
    """Poll ``url`` until it answers or ``timeout`` elapses.

    Args:
        url: The URL to poll.
        timeout: How long to keep trying, in seconds.

    Returns:
        Whether the server answered before the timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1.0)  # noqa: S310 - localhost only
            return True
        except OSError:
            time.sleep(0.2)
    return False


def run_serve(args: argparse.Namespace) -> int:
    """Execute the ``serve`` subcommand: build, serve, open a browser.

    Args:
        args: Parsed arguments from :func:`add_serve_parser`.

    Returns:
        Process exit code: ``0`` on a normal shutdown (Ctrl+C), ``1`` if the
        'server' extra is missing.
    """
    try:
        import uvicorn  # noqa: PLC0415 - optional 'server' extra
    except ImportError:
        print(
            "error: the local server needs the 'server' extra: "
            'pip install "stable-tracker[server]"',
            file=sys.stderr,
        )
        return 1

    from ..server.app import create_app, web_dist_dir  # noqa: PLC0415 - optional extra

    if web_dist_dir() is None and not args.no_build:
        web_dir = _source_web_dir()
        if web_dir is not None:
            if not _build_frontend(web_dir):
                print(
                    "warning: frontend build failed; continuing with the API only",
                    file=sys.stderr,
                )
        else:
            print(
                "warning: no built frontend found and no web/ source checkout nearby; "
                "the API will run without a UI. See README.md.",
                file=sys.stderr,
            )

    url = f"http://{args.host}:{args.port}/"
    print(f"starting Stable at {url}  (Ctrl+C to stop)")

    if not args.no_browser:

        def _open_when_ready() -> None:
            if _wait_until_serving(url):
                webbrowser.open(url)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    if args.reload:
        uvicorn.run(
            "stable.server.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
