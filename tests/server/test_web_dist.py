"""Tests for :func:`stable.server.app.web_dist_dir`'s frontend-resolution order.

Phase 6 packages a pre-built frontend snapshot inside the installed package
(``stable/server/webdist``, populated by ``scripts/build_frontend.py``) so a
real ``pip install`` needs no Node.js — but a source checkout that ran its own
``cd web && npm run build`` should still work. Both candidate directories are
monkeypatched here rather than touched on disk, so this runs regardless of
whether either build actually happened to exist in the checkout under test.
"""

from __future__ import annotations

from pathlib import Path

import stable.server.app as app_module


def _write_index(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text("<html></html>", encoding="utf-8")


def test_prefers_the_packaged_build_over_the_source_checkout(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"
    source = tmp_path / "source"
    _write_index(packaged)
    _write_index(source)
    monkeypatch.setattr(app_module, "_PACKAGED_WEB_DIST", packaged)
    monkeypatch.setattr(app_module, "_SOURCE_WEB_DIST", source)

    assert app_module.web_dist_dir() == packaged


def test_falls_back_to_the_source_checkout_build(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"  # never created
    source = tmp_path / "source"
    _write_index(source)
    monkeypatch.setattr(app_module, "_PACKAGED_WEB_DIST", packaged)
    monkeypatch.setattr(app_module, "_SOURCE_WEB_DIST", source)

    assert app_module.web_dist_dir() == source


def test_returns_none_when_neither_build_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_PACKAGED_WEB_DIST", tmp_path / "packaged")
    monkeypatch.setattr(app_module, "_SOURCE_WEB_DIST", tmp_path / "source")

    assert app_module.web_dist_dir() is None


def test_a_directory_without_index_html_does_not_count(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "assets").mkdir()  # build directory exists but is incomplete
    monkeypatch.setattr(app_module, "_PACKAGED_WEB_DIST", packaged)
    monkeypatch.setattr(app_module, "_SOURCE_WEB_DIST", tmp_path / "source")

    assert app_module.web_dist_dir() is None


def test_the_repo_ships_a_packaged_frontend_snapshot():
    """Guards against the committed stable/server/webdist snapshot going stale/missing.

    Regenerate it with `python scripts/build_frontend.py` after a frontend
    change; this only checks that *something* is there and it is a real Vite
    build, not that it is up to date with the current web/src/.
    """
    packaged = Path(__file__).resolve().parents[2] / "stable" / "server" / "webdist"
    index = packaged / "index.html"
    assert index.is_file(), (
        f"{packaged} has no index.html -- run `python scripts/build_frontend.py` "
        "so `pip install` ships a working UI without Node.js"
    )
    assert list(packaged.glob("assets/*.js")), "packaged build has no JS bundle"
