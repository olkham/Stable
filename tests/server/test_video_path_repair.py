"""The source clip is referenced relatively, and a stale absolute path self-heals.

A project is meant to be self-contained: the clip is copied inside it and the
project file points at it *relatively*, so the whole directory can be moved to
another drive and still work. It did not, for two reasons that this module
pins down — the path was written before the project knew where it lived, and
nothing repaired the projects that had already been written that way.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from stable.core.project import Project
from stable.server.storage import Workspace


def project_json(workspace: Workspace, project_id: str) -> dict:
    """Return the raw on-disk project dict, bypassing any load-time repair."""
    path = workspace.project_dir(project_id) / "project.stable.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_video_path(workspace: Workspace, project_id: str, value: str) -> None:
    """Rewrite the stored video path, as an older version would have left it."""
    path = workspace.project_dir(project_id) / "project.stable.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["video"]["path"] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture
def project(workspace: Workspace, tiny_clip) -> str:
    """Create a project from the tiny clip and return its id.

    Built through :class:`Workspace` rather than the REST API on purpose: an
    upload also enqueues a proxy job, whose open handle on ``source.mp4`` stops
    Windows moving or deleting the directory these tests need to move around.
    Storage behaviour is what is under test here, so the HTTP layer is noise.
    """
    meta, _ = workspace.create_project(
        video_bytes=Path(tiny_clip[0]).read_bytes(), original_filename="clip.mp4"
    )
    return meta.id


# -- the bug: an absolute path was stored -----------------------------------


def test_a_new_project_stores_its_clip_relatively(workspace: Workspace, project: str):
    stored = project_json(workspace, project)["video"]["path"]
    assert stored == "video/source.mp4"
    assert not os.path.isabs(stored)


def test_the_whole_project_directory_can_be_moved_to_another_root(
    workspace: Workspace, project: str, tmp_path: Path
):
    """What the storage relocation actually needs: a project that survives a move."""
    moved_root = tmp_path / "moved-workspace"
    moved_root.mkdir()
    shutil.move(str(workspace.project_dir(project)), str(moved_root / project))

    relocated = Workspace(moved_root)
    loaded = relocated.load_project(project)
    resolved = loaded.resolve_path(loaded.video.path)

    assert resolved is not None and os.path.isfile(resolved)
    assert Path(resolved).parent.parent == moved_root / project
    source = relocated.get_video_source(project)
    try:
        assert source.frame_count > 0
    finally:
        source.close()


# -- the repair: projects written by the older version ----------------------


def test_an_absolute_path_from_a_drive_that_no_longer_exists_is_repaired(
    workspace: Workspace, project: str
):
    write_video_path(workspace, project, r"C:\Users\someone\.stable\projects\x\video\source.mp4")

    loaded = workspace.load_project(project)

    assert loaded.video.path == "video/source.mp4"
    assert project_json(workspace, project)["video"]["path"] == "video/source.mp4", "persisted"
    resolved = loaded.resolve_path(loaded.video.path)
    assert resolved is not None and os.path.isfile(resolved)


def test_an_absolute_path_that_still_resolves_inside_the_project_is_made_relative(
    workspace: Workspace, project: str
):
    """The path works today but breaks on the next move — repair it anyway."""
    actual = workspace.video_dir(project) / "source.mp4"
    write_video_path(workspace, project, str(actual))

    assert workspace.load_project(project).video.path == "video/source.mp4"


def test_a_clip_deliberately_kept_outside_the_project_is_left_alone(
    workspace: Workspace, project: str, tmp_path: Path
):
    """Only clips inside the project directory are ours to rewrite."""
    external = tmp_path / "elsewhere" / "shot.mp4"
    external.parent.mkdir(parents=True)
    shutil.copyfile(workspace.video_dir(project) / "source.mp4", external)
    write_video_path(workspace, project, str(external))

    assert workspace.load_project(project).video.path == str(external)


def test_a_project_with_no_clip_at_all_reports_the_original_failure(
    workspace: Workspace, project: str
):
    """Nothing to repair with: don't invent a path, let the real error surface."""
    missing = r"C:\Users\someone\.stable\projects\x\video\source.mp4"
    write_video_path(workspace, project, missing)
    shutil.rmtree(workspace.video_dir(project))

    assert workspace.load_project(project).video.path == missing


def test_loading_an_already_repaired_project_rewrites_nothing(workspace: Workspace, project: str):
    """The repair is a one-off, not a write on every single load."""
    path = workspace.project_dir(project) / "project.stable.json"
    workspace.load_project(project)
    before = path.stat().st_mtime_ns

    for _ in range(3):
        workspace.load_project(project)
    assert path.stat().st_mtime_ns == before


def test_the_repair_does_not_disturb_the_rest_of_the_project(workspace: Workspace, project: str):
    write_video_path(workspace, project, r"C:\gone\video\source.mp4")
    before = project_json(workspace, project)

    workspace.load_project(project)

    after = project_json(workspace, project)
    assert after["video"]["path"] != before["video"]["path"]
    for key in ("version", "solve", "smoothing", "render", "points"):
        assert after.get(key) == before.get(key), key
    assert Project.load(workspace.project_dir(project) / "project.stable.json").validate() is None
