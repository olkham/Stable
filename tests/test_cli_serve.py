"""Tests for the ``stable serve`` CLI subcommand.

``run_serve``'s last step is a blocking ``uvicorn.run(...)`` call — the same
shape :mod:`stable.server.__main__` already has and does not unit-test either
(there is no good way to "run a server forever" inside a test). What *is*
tested here: every decision made before that call (frontend discovery,
auto-build, the URL, the browser-opening thread) and the two build helpers in
isolation, by monkeypatching ``uvicorn.run`` itself to a no-op that records
its arguments instead of blocking.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from stable.cli.serve import (
    _build_frontend,
    _source_web_dir,
    _wait_until_serving,
    add_serve_parser,
    run_serve,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(host="127.0.0.1", port=8123, no_browser=True, no_build=True, reload=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def fake_uvicorn(monkeypatch):
    """Install a stand-in ``uvicorn`` module that records how it was called."""
    calls: dict = {}

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls["kwargs"] = kwargs

    module = types.SimpleNamespace(run=fake_run)
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return calls


def test_serve_subcommand_is_registered():
    import argparse as ap

    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_serve_parser(subparsers)
    args = parser.parse_args(["serve", "--port", "9999", "--no-browser"])
    assert args.port == 9999
    assert args.no_browser is True
    assert args.host == "127.0.0.1"  # default


def test_run_serve_starts_uvicorn_with_the_requested_host_and_port(
    fake_uvicorn, monkeypatch, tmp_path
):
    # Pretend a frontend is already built, so no build is attempted.
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)
    rc = run_serve(_args(host="0.0.0.0", port=8123))
    assert rc == 0
    assert fake_uvicorn["kwargs"]["host"] == "0.0.0.0"
    assert fake_uvicorn["kwargs"]["port"] == 8123


def test_run_serve_uses_the_reload_form_when_requested(fake_uvicorn, monkeypatch, tmp_path):
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)
    rc = run_serve(_args(reload=True))
    assert rc == 0
    assert fake_uvicorn["app"] == "stable.server.app:create_app"
    assert fake_uvicorn["kwargs"]["factory"] is True
    assert fake_uvicorn["kwargs"]["reload"] is True


def test_run_serve_opens_a_browser_unless_disabled(fake_uvicorn, monkeypatch, tmp_path):
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)
    opened = []
    monkeypatch.setattr("stable.cli.serve._wait_until_serving", lambda url, **kw: True)
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    run_serve(_args(no_browser=False))
    # the browser is opened from a background thread; give it a moment
    import time

    for _ in range(50):
        if opened:
            break
        time.sleep(0.02)
    assert opened == ["http://127.0.0.1:8123/"]


def test_run_serve_reports_a_missing_server_extra_cleanly(monkeypatch, capsys):
    # Setting a module to None in sys.modules makes `import uvicorn` raise
    # ImportError (documented CPython behaviour) without needing a real
    # missing-package environment.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    rc = run_serve(_args())
    assert rc == 1
    assert "server" in capsys.readouterr().err


def test_run_serve_warns_but_continues_when_no_frontend_is_found(
    fake_uvicorn, monkeypatch, capsys
):
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: None)
    monkeypatch.setattr("stable.cli.serve._source_web_dir", lambda: None)
    rc = run_serve(_args(no_build=False))
    assert rc == 0
    assert "no built frontend" in capsys.readouterr().err


def test_run_serve_builds_the_frontend_when_a_source_checkout_is_found(
    fake_uvicorn, monkeypatch, tmp_path
):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "package.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: None)
    monkeypatch.setattr("stable.cli.serve._source_web_dir", lambda: web_dir)
    built = []
    monkeypatch.setattr("stable.cli.serve._build_frontend", lambda d: built.append(d) or True)

    rc = run_serve(_args(no_build=False))
    assert rc == 0
    assert built == [web_dir]


def test_no_build_flag_skips_the_frontend_build_entirely(fake_uvicorn, monkeypatch):
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: None)
    called = []
    monkeypatch.setattr("stable.cli.serve._source_web_dir", lambda: called.append(1))
    rc = run_serve(_args(no_build=True))
    assert rc == 0
    assert called == []  # never even looked for a source checkout


# -- helper functions ---------------------------------------------------------


def test_source_web_dir_finds_the_repo_checkout():
    # This test itself runs from the repo checkout, so web/package.json exists.
    web_dir = _source_web_dir()
    assert web_dir is not None
    assert (web_dir / "package.json").is_file()


def test_build_frontend_reports_a_clear_error_when_npm_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert _build_frontend(tmp_path) is False
    assert "npm" in capsys.readouterr().err


def test_build_frontend_runs_install_then_build(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr("subprocess.run", fake_run)
    assert _build_frontend(tmp_path) is True
    assert calls == [
        (["/usr/bin/npm", "install"], str(tmp_path)),
        (["/usr/bin/npm", "run", "build"], str(tmp_path)),
    ]


def test_build_frontend_skips_install_when_node_modules_exists(monkeypatch, tmp_path):
    (tmp_path / "node_modules").mkdir()
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr("subprocess.run", fake_run)
    assert _build_frontend(tmp_path) is True
    assert calls == [["/usr/bin/npm", "run", "build"]]


def test_wait_until_serving_returns_false_on_timeout():
    assert _wait_until_serving("http://127.0.0.1:1/nope", timeout=0.2) is False
