"""Tests for the ``stable serve`` CLI subcommand.

``run_serve``'s last step is a blocking call into uvicorn — the same shape
:mod:`stable.server.__main__` already has and does not unit-test either (there
is no good way to "run a server forever" inside a test). What *is* tested here:
every decision made before that call (frontend discovery, auto-build, which
port was bound, what URL the user was told, the browser-opening thread) and the
build and bind helpers in isolation, by replacing uvicorn with a no-op that
records its arguments instead of blocking.

:func:`stable.cli.serve._bind_listener` is the exception: it is tested against
real sockets, because "does this actually take a free port when the requested
one is held" is not a question a fake can answer.
"""

from __future__ import annotations

import argparse
import socket
import sys
import types

import pytest

from stable.cli.serve import (
    _bind_listener,
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


class _FakeListener:
    """Stands in for the bound socket, so tests never touch a real port."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_uvicorn(monkeypatch):
    """Install a stand-in ``uvicorn`` and a stand-in bind, recording both.

    ``run_serve`` serves on a socket it bound itself (see the module docstring),
    so the normal path goes through ``uvicorn.Config``/``uvicorn.Server`` rather
    than ``uvicorn.run`` — which is reserved for ``--reload``. Both are faked
    here into one ``calls`` dict so a test can assert on the host/port whichever
    path it exercised.
    """
    calls: dict = {"listener": _FakeListener()}

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls["kwargs"] = kwargs

    class FakeConfig:
        def __init__(self, app, **kwargs):
            calls["app"] = app
            calls["kwargs"] = kwargs

    class FakeServer:
        def __init__(self, config):
            calls["config"] = config

        def run(self, sockets=None):
            calls["sockets"] = sockets

    module = types.SimpleNamespace(run=fake_run, Config=FakeConfig, Server=FakeServer)
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    monkeypatch.setattr(
        "stable.cli.serve._bind_listener",
        lambda host, port: (calls["listener"], calls.get("bound_port", port)),
    )
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


def test_run_serve_serves_on_the_socket_it_bound_itself(fake_uvicorn, monkeypatch, tmp_path):
    """The bind happens here, not inside uvicorn — that is what makes the
    printed URL trustworthy, so it is asserted rather than assumed."""
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)
    run_serve(_args())
    assert fake_uvicorn["sockets"] == [fake_uvicorn["listener"]]
    assert fake_uvicorn["listener"].closed is True  # released on shutdown


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


def test_run_serve_announces_the_port_it_actually_bound(
    fake_uvicorn, monkeypatch, tmp_path, capsys
):
    """Todo: fall back to a free port, and say so.

    On this project's own Windows box port 8000 is held by ``iphlpsvc``, so the
    fallback is the everyday path, and the URL printed has to be the one that
    works — not the one that was asked for.
    """
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)
    fake_uvicorn["bound_port"] = 8131  # what _bind_listener settled on

    rc = run_serve(_args(port=8123))
    assert rc == 0
    out = capsys.readouterr()
    assert "http://127.0.0.1:8131/" in out.out
    assert "8123 is in use" in out.err
    assert "http://127.0.0.1:8123/" not in out.out  # never advertises the dead port
    assert fake_uvicorn["kwargs"]["port"] == 8131


def test_run_serve_prints_nothing_clickable_when_no_port_could_be_bound(
    fake_uvicorn, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr("stable.server.app.web_dist_dir", lambda: tmp_path)

    def refuse(host, port):
        raise OSError("interface is down")

    monkeypatch.setattr("stable.cli.serve._bind_listener", refuse)
    rc = run_serve(_args())
    assert rc == 1
    out = capsys.readouterr()
    assert "http://" not in out.out
    assert "interface is down" in out.err


def test_run_serve_reports_a_missing_server_extra_cleanly(monkeypatch, capsys):
    # Setting a module to None in sys.modules makes `import uvicorn` raise
    # ImportError (documented CPython behaviour) without needing a real
    # missing-package environment.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    rc = run_serve(_args())
    assert rc == 1
    assert "server" in capsys.readouterr().err


def test_run_serve_warns_but_continues_when_no_frontend_is_found(fake_uvicorn, monkeypatch, capsys):
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


# -- binding ------------------------------------------------------------------


def test_bind_listener_takes_the_requested_port_when_it_is_free():
    sock, port = _bind_listener("127.0.0.1", 0)  # 0: let the OS pick a free one
    try:
        assert port == sock.getsockname()[1]
        assert port > 0
    finally:
        sock.close()


def test_bind_listener_steps_past_a_port_that_is_already_taken():
    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(1)
    taken = squatter.getsockname()[1]
    try:
        sock, port = _bind_listener("127.0.0.1", taken)
        try:
            assert port != taken
            assert port in range(taken + 1, taken + 1 + 10)
        finally:
            sock.close()
    finally:
        squatter.close()


def test_bind_listener_falls_back_to_an_os_assigned_port_when_the_whole_range_is_taken(
    monkeypatch,
):
    """With every candidate refused but the last, the OS-assigned port wins.

    Squatting on eleven consecutive real ports is not something a test can
    arrange reliably, so the refusals are injected instead — the point under
    test is the *order* of the candidates, ending at port 0.
    """
    real_bind = socket.socket.bind
    attempted: list[int] = []

    def picky_bind(self, address):
        attempted.append(address[1])
        if address[1] != 0:
            raise OSError("address in use")
        return real_bind(self, address)

    monkeypatch.setattr(socket.socket, "bind", picky_bind)
    sock, port = _bind_listener("127.0.0.1", 8123)
    try:
        assert attempted == [8123, *range(8124, 8134), 0]
        assert port not in (0, 8123)
    finally:
        sock.close()


def test_bind_listener_reraises_when_even_an_os_assigned_port_fails(monkeypatch):
    def always_refuse(self, address):
        raise OSError("interface is down")

    monkeypatch.setattr(socket.socket, "bind", always_refuse)
    with pytest.raises(OSError, match="interface is down"):
        _bind_listener("127.0.0.1", 8123)
