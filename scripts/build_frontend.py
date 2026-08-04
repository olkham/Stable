"""Build the frontend and copy it into the package as ``stable/server/webdist``.

Run before building a wheel/sdist for release (and by CI's frontend job)::

    python scripts/build_frontend.py

This is what makes ``pip install stable-tracker`` ship a working UI with no
Node.js on the end user's machine: :func:`stable.server.app.web_dist_dir`
looks for exactly this directory first, ahead of the ``web/dist`` a source
checkout's own ``npm run build`` produces (see that function's docstring).
``stable serve`` also runs the equivalent of this script on demand when
running from a source checkout with no build yet (see
``stable/cli/serve.py``); this standalone script exists for the packaging
step, where "build once, then ship the wheel" is what we want rather than
"build again on every install".
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
DIST_DIR = WEB_DIR / "dist"
TARGET_DIR = REPO_ROOT / "stable" / "server" / "webdist"


def main() -> int:
    """Run ``npm ci``/``npm install`` + ``npm run build``, then copy the output in."""
    npm = shutil.which("npm")
    if npm is None:
        print("error: `npm` is not on PATH; install Node.js to build the frontend", file=sys.stderr)
        return 1

    if (WEB_DIR / "package-lock.json").is_file():
        install_cmd = [npm, "ci"]
    else:
        install_cmd = [npm, "install"]
    if subprocess.run(install_cmd, cwd=str(WEB_DIR)).returncode != 0:
        return 1
    if subprocess.run([npm, "run", "build"], cwd=str(WEB_DIR)).returncode != 0:
        return 1

    if not (DIST_DIR / "index.html").is_file():
        print(f"error: {DIST_DIR} has no index.html after the build", file=sys.stderr)
        return 1

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    shutil.copytree(DIST_DIR, TARGET_DIR)
    print(f"copied {DIST_DIR} -> {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
