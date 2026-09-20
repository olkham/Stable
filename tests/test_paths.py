"""Where the data root resolves to, and the one-time move out of ``~/.stable``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from stable.core import migrate_storage, paths
from stable.core.migrate_storage import (
    MIN_HEADROOM_BYTES,
    migrate_legacy_storage,
    planned_moves,
)
from stable.core.paths import ENV_HOME, data_root
from stable.server.config import ENV_WORKSPACE_DIR, workspace_root
from stable.trackers.weights import ENV_MODEL_DIR, model_root


@pytest.fixture
def home(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the legacy root at a temp directory so no real ``~/.stable`` is touched."""
    legacy = Path(tmp_path) / "home" / ".stable"  # type: ignore[arg-type]
    monkeypatch.setattr(paths, "LEGACY_HOME", legacy)
    monkeypatch.delenv(ENV_HOME, raising=False)
    monkeypatch.delenv(ENV_WORKSPACE_DIR, raising=False)
    monkeypatch.delenv(ENV_MODEL_DIR, raising=False)
    return legacy


def seed(root: Path, *names: str, size: int = 16) -> Path:
    """Create ``root/<name>/data.bin`` for each name and return ``root``."""
    for name in names:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data.bin").write_bytes(b"x" * size)
    return root


# -- the safety net --------------------------------------------------------


def test_no_test_can_reach_the_real_legacy_home():
    """The autouse guard in ``conftest.py`` is load-bearing: migration *moves* data.

    ``run_serve`` migrates for real, and ``tests/test_cli_serve.py`` calls it
    for real. Without the guard, running the suite relocates whatever is in the
    developer's own ``~/.stable`` — which is exactly what happened once.
    """
    assert paths.legacy_root() != Path("~/.stable").expanduser()
    assert not paths.legacy_root().exists()


# -- resolving the root ----------------------------------------------------


def test_the_root_follows_the_checkout_the_package_is_installed_from(home: Path):
    """A source checkout keeps its data next to itself, not in the home directory."""
    root = paths.checkout_root()
    assert root is not None, "the test suite runs from a checkout"
    assert data_root() == root / ".stable"
    assert data_root() != home


def test_both_stores_hang_off_the_one_root(home: Path):
    assert workspace_root() == data_root() / "projects"
    assert model_root() == data_root() / "models"


def test_stable_home_overrides_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "elsewhere"))
    assert data_root() == tmp_path / "elsewhere"
    assert workspace_root() == tmp_path / "elsewhere" / "projects"
    assert model_root() == tmp_path / "elsewhere" / "models"


def test_the_per_store_overrides_still_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "root"))
    monkeypatch.setenv(ENV_WORKSPACE_DIR, str(tmp_path / "ws"))
    monkeypatch.setenv(ENV_MODEL_DIR, str(tmp_path / "weights"))
    assert workspace_root() == tmp_path / "ws"
    assert model_root() == tmp_path / "weights"


def test_a_wheel_install_falls_back_to_the_venv_then_the_home_directory(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With no checkout to sit next to, the venv is used, and failing that, home."""
    monkeypatch.setattr(paths, "checkout_root", lambda: None)

    venv = tmp_path / "venv"
    venv.mkdir()
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    assert data_root() == venv / ".stable"

    monkeypatch.setattr(sys, "base_prefix", str(venv))  # no virtualenv active
    assert data_root() == home


# -- planning the move -----------------------------------------------------


def test_nothing_to_do_when_there_is_no_legacy_root(home: Path):
    assert planned_moves() == []
    assert migrate_legacy_storage() == []


def test_nothing_to_do_when_the_legacy_store_is_empty(home: Path, tmp_path: Path):
    (home / "projects").mkdir(parents=True)
    assert planned_moves({"projects": tmp_path / "new"}) == []


def test_nothing_to_do_when_the_legacy_root_is_already_the_destination(home: Path):
    seed(home / "projects", "abc")
    assert planned_moves({"projects": home / "projects"}) == []


# -- doing the move --------------------------------------------------------


def test_projects_and_models_are_moved_and_the_old_tree_is_removed(home: Path, tmp_path: Path):
    seed(home / "projects", "abc", "def")
    seed(home / "models", "cotracker3")
    destinations = {
        "projects": tmp_path / "new" / "projects",
        "models": tmp_path / "new" / "models",
    }

    lines: list[str] = []
    moves = migrate_legacy_storage(destinations, log=lines.append)

    assert {move.store for move in moves} == {"projects", "models"}
    assert (destinations["projects"] / "abc" / "data.bin").read_bytes() == b"x" * 16
    assert (destinations["projects"] / "def").is_dir()
    assert (destinations["models"] / "cotracker3").is_dir()
    assert not home.exists(), "the emptied legacy tree is cleaned up"
    assert any("abc" in line for line in lines), lines


def test_a_second_launch_does_nothing(home: Path, tmp_path: Path):
    seed(home / "projects", "abc")
    destinations = {"projects": tmp_path / "new" / "projects"}
    migrate_legacy_storage(destinations, log=lambda _: None)

    lines: list[str] = []
    assert migrate_legacy_storage(destinations, log=lines.append) == []
    assert lines == []


def test_a_name_the_destination_already_has_is_never_overwritten(home: Path, tmp_path: Path):
    seed(home / "projects", "abc", "def")
    destination = tmp_path / "new" / "projects"
    seed(destination, "abc", size=99)

    migrate_legacy_storage({"projects": destination}, log=lambda _: None)

    assert (destination / "abc" / "data.bin").read_bytes() == b"x" * 99, "kept the newer copy"
    assert (destination / "def").is_dir(), "the rest still moved"
    assert (home / "projects" / "abc").is_dir(), "the conflicting one stays put"


def test_a_failed_copy_leaves_the_source_intact_and_no_rubble(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seed(home / "projects", "abc", "def")
    destination = tmp_path / "new" / "projects"

    def explode(source: str, target: str) -> None:
        if source.endswith("abc"):
            Path(target).mkdir(parents=True)
            (Path(target) / "half.bin").write_bytes(b"partial")
            raise OSError("disk went away")
        Path(target).mkdir(parents=True)

    monkeypatch.setattr(migrate_storage.shutil, "move", explode)

    lines: list[str] = []
    migrate_legacy_storage({"projects": destination}, log=lines.append)

    assert not (destination / "abc").exists(), "the half-written copy is discarded"
    assert (home / "projects" / "abc" / "data.bin").is_file(), "the original is untouched"
    assert any("disk went away" in line for line in lines), lines


def test_a_destination_with_no_room_is_skipped_rather_than_half_filled(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seed(home / "projects", "abc")
    destination = tmp_path / "new" / "projects"
    monkeypatch.setattr(migrate_storage, "_free_space", lambda _: MIN_HEADROOM_BYTES // 2)

    lines: list[str] = []
    migrate_legacy_storage({"projects": destination}, log=lines.append)

    assert not destination.exists()
    assert (home / "projects" / "abc" / "data.bin").is_file()
    assert any("free" in line for line in lines), lines


def test_migration_never_raises_out_of_the_startup_path(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A broken move is reported, not propagated -- the app still has to start."""
    seed(home / "projects", "abc")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("everything is on fire")

    monkeypatch.setattr(migrate_storage, "_entry_size", explode)
    migrate_legacy_storage({"projects": tmp_path / "new"}, log=lambda _: None)
    assert (home / "projects" / "abc").is_dir()
