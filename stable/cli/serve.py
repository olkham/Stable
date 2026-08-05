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

Binding, and why this module does it itself
-------------------------------------------
``stable serve`` used to print its URL and *then* hand the port to uvicorn,
which meant the one line the user is told to click was printed before anything
had been bound — and was simply wrong if the bind then failed. On Windows the
default port 8000 is routinely held by ``iphlpsvc`` (Microsoft's IP Helper
service reserves ranges around it), so "wrong" is the common case, not the
exotic one, and what the user saw was a correct-looking URL followed by a
traceback.

So the socket is bound *here*, before anything is printed: the requested port
first, then the next few, then whatever the OS hands out. Uvicorn is then given
the already-bound socket to serve on, which is exactly what it does with its
own internally-bound one. Nothing is printed until that succeeded, and what is
printed is the port actually held — not the port that was asked for.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

__all__ = ["add_serve_parser", "run_serve"]

PORT_SCAN_RANGE = 10
"""How many ports past the requested one to try before asking the OS to choose.

Enough to step over a couple of neighbouring services (or another copy of
Stable) while keeping the port predictable; past that, a port the OS picks is
more useful than a tenth guess.
"""


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


def _bind_listener(host: str, port: int) -> tuple[socket.socket, int]:
    """Bind a listening socket on ``host``, falling back to a free port.

    Tries the requested port, then the next :data:`PORT_SCAN_RANGE`, then port
    ``0`` (any free port the OS chooses). The socket is returned still bound and
    listening, so the port cannot be taken by anything else between choosing it
    and serving on it.

    Args:
        host: Interface to bind.
        port: Preferred port; ``0`` asks the OS to choose immediately.

    Returns:
        ``(socket, port)`` — the listening socket and the port it actually holds.

    Raises:
        OSError: If even an OS-assigned port could not be bound, which means
            something is wrong with the interface rather than with the port.
    """
    candidates = [port, *range(port + 1, port + 1 + PORT_SCAN_RANGE), 0] if port else [0]
    last_error: OSError | None = None
    for candidate in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Deliberately *not* SO_REUSEADDR: on Linux it would let this bind
            # succeed while another process is still listening on the same port,
            # which would turn "the port is busy" into two servers splitting the
            # user's requests. Windows' SO_REUSEADDR is worse still (it steals
            # the port outright).
            sock.bind((host, candidate))
            sock.listen(2048)
        except OSError as exc:
            sock.close()
            last_error = exc
            continue
        return sock, sock.getsockname()[1]
    assert last_error is not None  # noqa: S101 - the loop always tries port 0
    raise last_error


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
        'server' extra is missing or no port could be bound at all.
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

    try:
        listener, port = _bind_listener(args.host, args.port)
    except OSError as exc:
        print(f"error: cannot listen on {args.host}: {exc}", file=sys.stderr)
        return 1
    if port != args.port:
        print(
            f"note: port {args.port} is in use; serving on {port} instead",
            file=sys.stderr,
        )

    url = f"http://{args.host}:{port}/"
    print(f"Stable is running at {url}  (Ctrl+C to stop)")

    if not args.no_browser:

        def _open_when_ready() -> None:
            if _wait_until_serving(url):
                webbrowser.open(url)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    if args.reload:
        # The reload supervisor runs the app in child processes that inherit
        # their own socket, so this one is handed back before uvicorn rebinds
        # the port we just proved is free.
        listener.close()
        uvicorn.run(
            "stable.server.app:create_app",
            factory=True,
            host=args.host,
            port=port,
            reload=True,
        )
    else:
        config = uvicorn.Config(create_app(), host=args.host, port=port)
        try:
            uvicorn.Server(config).run(sockets=[listener])
        finally:
            listener.close()
    return 0
