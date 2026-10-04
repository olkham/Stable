"""Dev/test entry point: ``python -m stable.server``.

Phase 6 owns the polished ``stable serve`` CLI command (model manager UI,
opening the browser automatically, packaging). This is the minimal runner
Phase 4 needs to actually exercise the server — during development and from
the ``run`` workflow — without reaching into ``stable/cli``.
"""

from __future__ import annotations

import argparse


def main() -> int:
    """Run the Stable server with uvicorn."""
    parser = argparse.ArgumentParser(prog="python -m stable.server")
    # Loopback, like `stable serve`: the server has no login, so being reachable
    # from the network is something to ask for with `--host 0.0.0.0`. It is
    # also what the Vite dev server proxies to (web/vite.config.ts).
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on source changes")
    args = parser.parse_args()

    import uvicorn  # noqa: PLC0415 - optional 'server' extra

    from ..core.migrate_storage import migrate_legacy_storage  # noqa: PLC0415

    # Same one-time relocation `stable serve` does -- this runner opens the
    # same workspace, so it has to look in the same place for old projects.
    migrate_legacy_storage()

    if args.reload:
        uvicorn.run(
            "stable.server.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        from .app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
