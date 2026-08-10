"""Contract tests for the versioned project file: round-trip, migrate, validate."""

from __future__ import annotations

import json

import pytest

from stable.core.corrections import GapFillCorrection, OffsetCorrection
from stable.core.errors import ProjectError, SchemaVersionError
from stable.core.project import (
    SCHEMA_VERSION,
    PointSpec,
    Project,
    RenderSettings,
    SmoothingSpec,
    SolveSettings,
    VideoRef,
    migrate,
    register_migration,
)
from stable.core.types import Query

# The example straight out of PLAN.md (JSON comment stripped).
PLAN_EXAMPLE = """
{ "version": 1,
  "video": { "path": "clip.mp4", "fps": 24, "frames": 120, "w": 1920, "h": 1080 },
  "points": [ { "id": "p1", "queries": [{"frame": 0, "x": 0, "y": 0}],
                "tracker": "cotracker3", "enabled_for_solve": true,
                "raw": "tracks/p1.npz",
                "corrections": [] } ],
  "solve": { "reference_frame": 0, "channels": {"x": true, "y": true, "r": true, "s": true} },
  "smoothing": [ { "channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0} } ],
  "render": { "border": "auto_zoom", "codec": "h264", "crf": 18 } }
"""


def full_project() -> Project:
    """A project exercising every part of the schema."""
    return Project(
        video=VideoRef(path="clip.mp4", fps=23.976, frames=120, w=1920, h=1080),
        points=[
            PointSpec(
                id="p1",
                queries=[Query(0, 100.0, 200.0), Query(60, 140.5, 190.25)],
                tracker="cotracker3",
                enabled_for_solve=True,
                raw="tracks/p1.npz",
                corrections=[
                    OffsetCorrection(frame=30, dx=1.0, dy=-1.0, falloff_before=5, falloff_after=5),
                    GapFillCorrection(start=70, end=75, method="linear", confidence=0.3),
                ],
            ),
            PointSpec(
                id="p2",
                queries=[Query(10, 5.0, 6.0)],
                tracker="lk",
                enabled_for_solve=False,
                color="#ff8800",
            ),
        ],
        solve=SolveSettings(
            reference_frame=12,
            channels={"x": True, "y": True, "r": False, "s": True},
        ),
        smoothing=[
            SmoothingSpec(channel="x", filter="butterworth", params={"cutoff_hz": 1.0}),
            SmoothingSpec(channel="r", filter="tripod", params={}, enabled=False),
        ],
        render=RenderSettings(border="crop", codec="prores", crf=12),
    )


def test_schema_round_trip_in_memory():
    project = full_project()
    assert Project.from_dict(project.to_dict()) == project


def test_schema_round_trip_on_disk(tmp_path):
    project = full_project()
    path = project.save(tmp_path / "shot.stable.json")
    loaded = Project.load(path)

    assert loaded == project
    assert loaded.path == str(tmp_path / "shot.stable.json")
    assert loaded.to_dict() == project.to_dict()
    # Anchors, corrections and their ids all survive verbatim.
    assert loaded.point("p1").queries == project.point("p1").queries
    assert loaded.point("p1").corrections == project.point("p1").corrections
    # The file on disk is plain, readable JSON.
    assert json.loads((tmp_path / "shot.stable.json").read_text(encoding="utf-8"))["version"] == 1


def test_saving_twice_is_atomic_and_leaves_no_temp_file(tmp_path):
    project = full_project()
    target = tmp_path / "shot.stable.json"
    project.save(target)
    project.save()  # saves back to the remembered path
    assert not (tmp_path / "shot.stable.json.tmp").exists()
    assert Project.load(target) == project


def test_plan_example_parses():
    project = Project.from_json(PLAN_EXAMPLE)
    assert project.version == SCHEMA_VERSION
    assert project.video.frames == 120
    assert project.point("p1").tracker == "cotracker3"
    assert project.point("p1").queries == [Query(0, 0.0, 0.0)]
    assert project.solve.channels == {"x": True, "y": True, "r": True, "s": True}
    assert project.smoothing[0].params == {"cutoff_hz": 1.0}
    assert project.render.border == "auto_zoom"


def test_a_points_colour_survives_the_round_trip_and_defaults_to_the_palette(tmp_path):
    project = full_project()
    # A point with no colour of its own says so, rather than inventing one: the
    # UI's palette is the answer, and it needs to know it may give one.
    assert project.point("p1").color is None
    assert project.point("p2").color == "#ff8800"

    loaded = Project.load(project.save(tmp_path / "shot.stable.json"))
    assert loaded.point("p1").color is None
    assert loaded.point("p2").color == "#ff8800"

    # Written before colours existed: absent is the same as "use the palette".
    legacy = full_project().to_dict()
    for point in legacy["points"]:
        point.pop("color")
    assert [p.color for p in Project.from_dict(legacy).points] == [None, None]


def test_a_composite_remembers_what_it_was_made_from(tmp_path):
    """Provenance is stored, not just returned (todo 2, tenth round, second pass).

    It began as a key on the response that created the point, which was enough
    to say "made from p1 + p2" once and lost it on the next reload — and the
    panel needs it *every* time it draws the card, to know it is looking at a
    composite rather than a tracked track.
    """
    project = full_project()
    assert project.point("p1").derived_from == []
    assert project.point("p1").is_composite is False

    project.points.append(PointSpec(id="p9", derived_from=["p1", "p2"]))
    assert project.point("p9").is_composite is True

    loaded = Project.load(project.save(tmp_path / "shot.stable.json"))
    assert loaded.point("p9").derived_from == ["p1", "p2"]
    assert loaded.point("p1").derived_from == []

    # Written before composites existed: absent is "an ordinary tracked point",
    # which is what every point in such a file is.
    legacy = full_project().to_dict()
    for point in legacy["points"]:
        point.pop("derived_from")
    assert [p.derived_from for p in Project.from_dict(legacy).points] == [[], []]


def test_a_colour_the_ui_cannot_draw_is_refused():
    # Not a matter of taste: `#rrggbb` is what an HTML colour input produces and
    # what every canvas here strokes with, so anything else would be stored and
    # then silently ignored.
    for bad in ("red", "#fff", "rgb(1,2,3)", "#gggggg"):
        with pytest.raises(ProjectError, match="colour"):
            PointSpec(id="p1", color=bad)
    # Case and surrounding space are the operator's typing, not a different
    # colour, so they are normalised rather than refused.
    assert PointSpec(id="p1", color="  #FF8800 ").color == "#ff8800"


def test_defaults_are_sane():
    project = Project()
    assert project.version == SCHEMA_VERSION
    assert project.solve.channels == {"x": True, "y": True, "r": True, "s": True}
    assert project.render.border == "auto_zoom"
    assert project.new_point_id() == "p1"


def test_unknown_keys_are_ignored():
    data = full_project().to_dict()
    data["future_section"] = {"hello": 1}
    data["points"][0]["future_field"] = True
    assert Project.from_dict(data).point("p1").id == "p1"


def test_future_versions_are_refused():
    data = full_project().to_dict()
    data["version"] = SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionError):
        Project.from_dict(data)


def test_missing_or_malformed_version():
    with pytest.raises(ProjectError):
        migrate({"points": []})
    with pytest.raises(ProjectError):
        migrate({"version": "1"})


def test_migration_chain_runs(clean_migrations):
    def v0_to_v1(data: dict) -> dict:
        upgraded = dict(data)
        upgraded["points"] = [{"id": pid} for pid in data.get("point_ids", [])]
        upgraded.pop("point_ids", None)
        return upgraded

    register_migration(0, v0_to_v1)
    legacy = {"version": 0, "point_ids": ["p1", "p2"], "video": {"path": "a.mp4", "frames": 10}}
    project = Project.from_dict(legacy)
    assert project.version == SCHEMA_VERSION
    assert [p.id for p in project.points] == ["p1", "p2"]

    with pytest.raises(ProjectError):
        register_migration(0, v0_to_v1)


def test_missing_migration_is_an_error(clean_migrations):
    with pytest.raises(SchemaVersionError):
        migrate({"version": 0})


def test_validate_catches_cross_field_problems():
    project = full_project()
    project.points[1].id = "p1"
    with pytest.raises(ProjectError, match="duplicate point id"):
        project.validate()

    project = full_project()
    project.points[0].queries.append(Query(999, 0.0, 0.0))
    with pytest.raises(ProjectError, match="anchor at frame"):
        project.validate()

    project = full_project()
    project.solve.reference_frame = 500
    with pytest.raises(ProjectError, match="reference_frame"):
        project.validate()

    project = full_project()
    project.solve.reference_frame = -1
    with pytest.raises(ProjectError):
        project.validate()


def test_invalid_enum_values_are_rejected():
    with pytest.raises(ProjectError):
        RenderSettings(border="feathered")
    with pytest.raises(ProjectError):
        SmoothingSpec(channel="z", filter="gaussian")
    with pytest.raises(ProjectError):
        SmoothingSpec.from_dict({"channel": "x"})
    with pytest.raises(ProjectError):
        PointSpec.from_dict({"tracker": "lk"})


def test_bad_json_and_missing_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ProjectError):
        Project.load(bad)
    with pytest.raises(ProjectError):
        Project.load(tmp_path / "nope.json")
    with pytest.raises(ProjectError):
        Project.from_dict([1, 2, 3])
    with pytest.raises(ProjectError):
        Project().save()  # no path known


def test_path_helpers(tmp_path):
    project = full_project()
    project.save(tmp_path / "shot.stable.json")
    assert project.directory == str(tmp_path)
    assert project.resolve_path("tracks/p1.npz") == str(tmp_path / "tracks" / "p1.npz")
    assert project.resolve_path(None) is None
    absolute = str(tmp_path / "tracks" / "p2.npz")
    assert project.relative_path(absolute) == "tracks/p2.npz"
    assert project.resolve_path(absolute) == absolute


def test_point_lookup_and_ids():
    project = full_project()
    assert project.point("p2").tracker == "lk"
    with pytest.raises(ProjectError):
        project.point("nope")
    assert project.new_point_id() == "p3"


def test_sorted_queries():
    spec = PointSpec(id="p1", queries=[Query(40, 0, 0), Query(0, 0, 0), Query(12, 0, 0)])
    assert [q.frame for q in spec.sorted_queries()] == [0, 12, 40]
