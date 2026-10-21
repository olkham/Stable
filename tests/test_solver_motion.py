"""The per-clip solve: curve assembly, graceful degradation and exports."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import WarpedClipSource, handheld_transforms

from stable.core.errors import SolverError
from stable.core.geometry import SimilarityTransform
from stable.core.project import SolveSettings
from stable.core.types import PointTrack, TrackResult
from stable.solver import (
    MotionSolve,
    SolveConfig,
    SolveStatus,
    solve_frame,
    solve_motion,
)

PIVOT = (160.0, 120.0)

TOL = 1e-4
"""Ground-truth tolerance.

``TrackResult`` stores positions as float32 (a locked Phase 0 decision), so a
position near 300 px carries about 2e-5 px of quantisation before the solver
sees it. Anything tighter than this would be testing float32, not the solve;
anything looser would stop being "exact recovery".
"""


def tracks_from(transforms, points, *, confidence: float = 1.0) -> TrackResult:
    """Return exact tracks of ``points`` under a known per-frame transform list."""
    source = np.asarray(points, dtype=np.float64)
    stacked = np.stack([np.asarray(t.apply(source)) for t in transforms], axis=1)
    return TrackResult(
        points=stacked,
        confidence=np.full(stacked.shape[:2], confidence, dtype=np.float32),
        visible=np.ones(stacked.shape[:2], dtype=bool),
    )


GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0), (160.0, 120.0)]


@pytest.fixture
def known_motion():
    """A 30-frame known camera path plus the exact tracks it produces."""
    transforms = handheld_transforms(30, pivot=PIVOT)
    return transforms, tracks_from(transforms, GRID)


# -- ground truth ------------------------------------------------------------


def test_recovers_the_known_transform_of_every_frame(known_motion):
    """The solve reproduces the similarity that generated the tracks."""
    transforms, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    for frame, truth in enumerate(transforms):
        assert solve.transform_at(frame).allclose(truth, atol=TOL)


def test_curves_are_reported_about_the_pivot(known_motion):
    """X and Y are the pivot's displacement; R and S are the raw rotation/scale."""
    transforms, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    assert solve.pivot == PIVOT
    for frame, truth in enumerate(transforms):
        moved = truth.apply(np.asarray(PIVOT))
        assert solve.curve("x")[frame] == pytest.approx(moved[0] - PIVOT[0], abs=TOL)
        assert solve.curve("y")[frame] == pytest.approx(moved[1] - PIVOT[1], abs=TOL)
        assert solve.curve("r")[frame] == pytest.approx(truth.rotation, abs=TOL)
        assert solve.curve("s")[frame] == pytest.approx(truth.scale, abs=TOL)


def test_the_reference_frame_solves_to_the_identity(known_motion):
    """Whatever the reference frame, its own transform is the identity."""
    _, tracks = known_motion
    for reference in (0, 7, 29):
        solve = solve_motion(
            tracks, config=SolveConfig(reference_frame=reference), frame_size=(320, 240)
        )
        assert solve.transform_at(reference).is_identity(atol=1e-9)
        assert solve.curve("s")[reference] == pytest.approx(1.0)


def test_the_solve_is_relative_to_the_reference_frame(known_motion):
    """Moving the reference frame re-bases every transform, exactly."""
    transforms, tracks = known_motion
    solve = solve_motion(tracks, config=SolveConfig(reference_frame=10), frame_size=(320, 240))
    inverse = transforms[10].inverse()
    for frame, truth in enumerate(transforms):
        assert solve.transform_at(frame).allclose(truth.compose(inverse), atol=TOL)


def test_rotation_is_unwrapped_across_the_pi_boundary():
    """A shot rotating through +/-pi gives a continuous curve, not a 2pi step."""
    angles = np.linspace(math.pi - 0.4, math.pi + 0.4, 25)
    transforms = [SimilarityTransform.about_pivot(PIVOT, rotation=a) for a in angles]
    solve = solve_motion(tracks_from(transforms, GRID), frame_size=(320, 240))
    steps = np.diff(solve.curve("r"))
    assert np.abs(steps).max() < 0.1
    assert np.all(steps > 0.0)


def test_a_wrong_point_is_rejected_without_moving_the_curve(known_motion):
    """One track that slid onto something else must not bend the solved motion."""
    transforms, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[1, 10:] += (35.0, -28.0)  # point 1 jumps mid-clip and stays wrong
    broken = TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible)

    solve = solve_motion(broken, frame_size=(320, 240))
    for frame, truth in enumerate(transforms):
        assert solve.transform_at(frame).allclose(truth, atol=1e-4)
    assert (solve.n_inliers[10:] == 4).all()


# -- weighting ---------------------------------------------------------------


def test_confidence_weights_the_fit(known_motion):
    """A low-confidence point that disagrees is outvoted rather than obeyed."""
    transforms, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    confidence = np.array(tracks.confidence, dtype=np.float32)
    points[2] += (12.0, 9.0)
    confidence[2] = 0.06  # above the gate, but barely trusted

    solve = solve_motion(
        TrackResult(points=points, confidence=confidence, visible=tracks.visible),
        frame_size=(320, 240),
    )
    for frame, truth in enumerate(transforms):
        assert solve.transform_at(frame).allclose(truth, atol=1e-3)


def test_low_confidence_samples_are_gated_out(known_motion):
    """Samples below ``min_confidence`` are not correspondences at all."""
    _, tracks = known_motion
    confidence = np.array(tracks.confidence, dtype=np.float32)
    confidence[3:, 5:9] = 0.0
    solve = solve_motion(
        TrackResult(points=tracks.points, confidence=confidence, visible=tracks.visible),
        frame_size=(320, 240),
    )
    assert (solve.n_points[5:9] == 3).all()
    assert (solve.n_points[:5] == 5).all()


# -- graceful degradation ----------------------------------------------------


def test_one_point_degrades_to_translation_and_is_flagged():
    """A single track gives X and Y only, and the frames say so."""
    transforms = handheld_transforms(20, pivot=PIVOT)
    tracks = tracks_from(transforms, [(100.0, 90.0)])
    solve = solve_motion(tracks, frame_size=(320, 240))

    assert (solve.status == int(SolveStatus.TRANSLATION_ONLY)).all()
    assert solve.flagged.all()
    assert np.allclose(solve.curve("r"), 0.0)
    assert np.allclose(solve.curve("s"), 1.0)
    # The translation it does report is the real motion of that point.
    moved = transforms[7].apply(np.asarray([[100.0, 90.0]]))[0]
    assert solve.transform_at(7).tx == pytest.approx(moved[0] - 100.0, abs=TOL)


def test_one_point_is_a_full_solve_when_only_x_and_y_were_asked_for():
    """Rotation and scale switched off: nothing requested is missing, nothing flagged.

    Flagging every frame of such a solve is what hid the frames that really
    were not measured — the interpolated ones — among the ones that were.
    """
    transforms = handheld_transforms(20, pivot=PIVOT)
    tracks = tracks_from(transforms, [(100.0, 90.0)])
    channels = {"x": True, "y": True, "r": False, "s": False}
    solve = solve_motion(tracks, frame_size=(320, 240), config=SolveConfig(channels=channels))

    assert (solve.status == int(SolveStatus.SOLVED)).all()
    assert not solve.flagged.any()


def test_a_hole_in_the_middle_is_interpolated_and_flagged(known_motion):
    """Frames with nothing usable are filled from their solved neighbours."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[:, 12:16] = np.nan
    solve = solve_motion(
        TrackResult(
            points=points,
            confidence=tracks.confidence,
            visible=tracks.visible,
        ),
        frame_size=(320, 240),
    )

    assert (solve.status[12:16] == int(SolveStatus.INTERPOLATED)).all()
    assert solve.flagged[12:16].all()
    assert not solve.flagged[:12].any()
    assert np.isfinite(solve.curve("x")).all()
    # The filled values sit between their neighbours rather than being reset.
    inside = solve.curve("x")[12:16]
    lo, hi = sorted((solve.curve("x")[11], solve.curve("x")[16]))
    assert ((inside >= lo - 1e-9) & (inside <= hi + 1e-9)).all()


def test_a_hole_at_the_clip_edge_holds_the_nearest_value(known_motion):
    """Off the end there is nothing to interpolate between, so the value is held."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[:, :3] = np.nan
    points[:, -2:] = np.nan
    solve = solve_motion(
        TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible),
        config=SolveConfig(reference_frame=5),
        frame_size=(320, 240),
    )

    assert (solve.status[:3] == int(SolveStatus.HELD)).all()
    assert (solve.status[-2:] == int(SolveStatus.HELD)).all()
    assert np.allclose(solve.curve("x")[:3], solve.curve("x")[3])
    assert np.allclose(solve.curve("y")[-2:], solve.curve("y")[-3])


def test_gap_filling_can_be_turned_off(known_motion):
    """With interpolation off, unsolved frames stay at the identity."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[:, 12:16] = np.nan
    solve = solve_motion(
        TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible),
        config=SolveConfig(interpolate_gaps=False),
        frame_size=(320, 240),
    )
    assert np.allclose(solve.curve("x")[12:16], 0.0)
    assert np.allclose(solve.curve("s")[12:16], 1.0)
    assert solve.flagged[12:16].all()


def test_a_clip_that_never_solves_returns_the_identity_everywhere():
    """Nothing to go on anywhere is still an answer, marked as such."""
    points = np.full((2, 8, 2), np.nan, dtype=np.float32)
    points[:, 0] = [[10.0, 10.0], [90.0, 40.0]]  # only the reference frame is known
    confidence = np.zeros((2, 8), dtype=np.float32)
    confidence[:, 0] = 1.0
    solve = solve_motion(
        TrackResult(points=points, confidence=confidence, visible=np.ones((2, 8), bool)),
        frame_size=(320, 240),
    )
    # Frame 0 solves (against itself); everything after it is held.
    assert solve.status_at(0) is SolveStatus.SOLVED
    assert (solve.status[1:] == int(SolveStatus.HELD)).all()
    assert np.allclose(solve.curve("x"), 0.0)
    assert np.allclose(solve.curve("s"), 1.0)


def test_no_usable_point_at_all_is_the_identity_status():
    """When even the reference cannot be reached, every frame reports IDENTITY."""
    tracks = [PointTrack.empty(6)]
    with pytest.raises(SolverError, match="no point is usable in reference frame"):
        solve_motion(tracks)

    # Reaching _assemble with no solved frame at all is the IDENTITY case.
    solve = MotionSolve(
        curves={"x": np.zeros(4), "y": np.zeros(4), "r": np.zeros(4), "s": np.ones(4)},
        residual=np.full(4, np.nan),
        n_points=np.zeros(4, np.int32),
        n_inliers=np.zeros(4, np.int32),
        status=np.full(4, int(SolveStatus.IDENTITY), np.int8),
        reference_frame=0,
        pivot=(0.0, 0.0),
    )
    assert solve.flagged.all()
    assert solve.summary()["status_counts"]["identity"] == 4


def test_a_high_residual_flags_an_otherwise_solved_frame(known_motion):
    """Frames that fit badly are surfaced even though they did solve."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    # Push every point a different way on one frame: no similarity explains it.
    points[:, 9] += np.array([[9.0, 0.0], [-9.0, 0.0], [0.0, 9.0], [0.0, -9.0], [7.0, 7.0]])
    solve = solve_motion(
        TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible),
        frame_size=(320, 240),
    )
    assert solve.status_at(9) is SolveStatus.SOLVED
    assert solve.residual[9] > solve.max_residual_px
    assert solve.flagged[9]
    assert not solve.flagged[8]


# -- channel gating ----------------------------------------------------------


@pytest.mark.parametrize("channel", ["r", "s"])
def test_disabling_rotation_or_scale_constrains_the_fit(known_motion, channel):
    """A locked channel is held at identity and the rest is refitted around it."""
    _, tracks = known_motion
    channels = dict.fromkeys("xyrs", True)
    channels[channel] = False
    solve = solve_motion(
        tracks, config=SolveConfig(channels=channels), frame_size=(320, 240)
    )
    expected = 0.0 if channel == "r" else 1.0
    assert np.allclose(solve.curve(channel), expected)


@pytest.mark.parametrize("channel", ["x", "y"])
def test_disabling_a_translation_channel_zeroes_its_curve(known_motion, channel):
    """X or Y off means the render never moves the frame along that axis."""
    _, tracks = known_motion
    channels = dict.fromkeys("xyrs", True)
    channels[channel] = False
    solve = solve_motion(tracks, config=SolveConfig(channels=channels), frame_size=(320, 240))
    other = "y" if channel == "x" else "x"
    assert np.allclose(solve.curve(channel), 0.0)
    assert np.abs(solve.curve(other)).max() > 1.0


def test_config_can_be_built_from_project_solve_settings():
    """The project file's ``solve`` block maps straight onto a solve config."""
    settings = SolveSettings(
        reference_frame=4, channels={"x": True, "y": True, "r": False, "s": True}
    )
    config = SolveConfig.from_settings(settings, min_confidence=0.2)
    assert config.reference_frame == 4
    assert config.constraints.rotate is False
    assert config.constraints.scale is True
    assert config.min_confidence == 0.2


# -- pivot -------------------------------------------------------------------


def test_the_pivot_defaults_to_the_frame_centre(known_motion):
    """Given the clip size, X and Y describe how far the shot moved."""
    _, tracks = known_motion
    assert solve_motion(tracks, frame_size=(1920, 1080)).pivot == (960.0, 540.0)


def test_the_pivot_falls_back_to_the_point_centroid(known_motion):
    """Without a clip size the points themselves define a sensible origin."""
    _, tracks = known_motion
    solve = solve_motion(tracks)
    expected = np.asarray(GRID, dtype=np.float64).mean(axis=0)
    assert solve.pivot == pytest.approx(tuple(expected))


def test_an_explicit_pivot_is_honoured(known_motion):
    """The pivot is a reporting choice, so the transforms are unaffected by it."""
    transforms, tracks = known_motion
    solve = solve_motion(
        tracks, config=SolveConfig(pivot=(0.0, 0.0)), frame_size=(320, 240)
    )
    assert solve.pivot == (0.0, 0.0)
    for frame, truth in enumerate(transforms):
        assert solve.transform_at(frame).allclose(truth, atol=TOL)
    assert solve.curve("x")[5] == pytest.approx(transforms[5].tx, abs=TOL)


# -- container and exports ---------------------------------------------------


def test_arrays_are_read_only(known_motion):
    """A solve is immutable, like the raw tracks it came from."""
    _, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    with pytest.raises(ValueError):
        solve.curve("x")[0] = 5.0
    with pytest.raises(ValueError):
        solve.residual[0] = 5.0


def test_unknown_channel_is_rejected(known_motion):
    """Asking for a channel that does not exist is an error, not a KeyError."""
    _, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    with pytest.raises(SolverError, match="unknown channel"):
        solve.curve("z")


def test_transform_at_bounds(known_motion):
    """Frame indices outside the clip raise IndexError."""
    _, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    assert len(solve) == 30
    with pytest.raises(IndexError):
        solve.transform_at(30)
    assert len(solve.transforms()) == 30


def test_csv_export_round_trips_the_curves(tmp_path, known_motion):
    """The CSV holds every frame, both rotation units, and the solve quality."""
    _, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    path = solve.save(tmp_path / "curves.csv")

    rows = [line.split(",") for line in open(path, encoding="utf-8").read().splitlines()]
    header, body = rows[0], rows[1:]
    assert header[:6] == ["frame", "x", "y", "r_rad", "r_deg", "s"]
    assert len(body) == solve.n_frames
    for frame, row in enumerate(body):
        assert int(row[0]) == frame
        assert float(row[1]) == pytest.approx(solve.curve("x")[frame], abs=1e-5)
        assert float(row[4]) == pytest.approx(math.degrees(solve.curve("r")[frame]), abs=1e-5)
        assert row[9] == SolveStatus(int(solve.status[frame])).label


def test_json_export_is_machine_readable(tmp_path, known_motion):
    """The JSON export carries the curves, the flags and the summary."""
    _, tracks = known_motion
    solve = solve_motion(tracks, frame_size=(320, 240))
    data = json.loads(open(solve.save(tmp_path / "curves.json"), encoding="utf-8").read())

    assert data["version"] == 1
    assert data["pivot"] == [160.0, 120.0]
    assert len(data["curves"]["x"]) == solve.n_frames
    assert set(data["status"]) == {"solved"}
    assert data["summary"]["status_counts"]["solved"] == solve.n_frames


def test_unsolved_frames_export_a_null_residual(known_motion):
    """A residual that does not exist is ``null``, not a fake zero."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[:, 5] = np.nan
    solve = solve_motion(
        TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible),
        frame_size=(320, 240),
    )
    assert solve.to_dict()["residual"][5] is None
    assert solve.to_csv().splitlines()[6].split(",")[6] == ""


# -- inputs ------------------------------------------------------------------


def test_accepts_a_list_of_point_tracks(known_motion):
    """A sequence of PointTrack is as good as a TrackResult."""
    _, tracks = known_motion
    from_result = solve_motion(tracks, frame_size=(320, 240))
    from_list = solve_motion(tracks.tracks(), frame_size=(320, 240))
    assert np.allclose(from_result.curve("x"), from_list.curve("x"))


@pytest.mark.parametrize(
    ("tracks", "message"),
    [
        ([], "at least one track"),
        ([PointTrack.empty(4), PointTrack.empty(6)], "same T"),
        (["not a track"], "sequence of PointTrack"),
    ],
)
def test_bad_track_input_is_rejected(tracks, message):
    """Malformed input is a solver error with an explanation."""
    with pytest.raises(SolverError, match=message):
        solve_motion(tracks)


def test_a_reference_frame_past_the_clip_is_rejected(known_motion):
    """A reference frame that does not exist is a configuration mistake."""
    _, tracks = known_motion
    with pytest.raises(SolverError, match="outside the clip"):
        solve_motion(tracks, config=SolveConfig(reference_frame=99))


def test_solve_frame_matches_the_full_solve(known_motion):
    """The single-frame preview cannot disagree with the clip-wide solve."""
    _, tracks = known_motion
    config = SolveConfig(reference_frame=2)
    solve = solve_motion(tracks, config=config, frame_size=(320, 240))
    for frame in (0, 2, 17):
        single = solve_frame(tracks, frame, config=config)
        assert single.transform is not None
        assert single.transform.allclose(
            solve.transform_at(frame).compose(SimilarityTransform.identity()), atol=1e-6
        )
        assert single.n_points == solve.n_points[frame]


def test_solve_frame_reports_an_unsolvable_frame_rather_than_raising(known_motion):
    """Preview of a frame with nothing usable returns no transform, not an error."""
    _, tracks = known_motion
    points = np.array(tracks.points, dtype=np.float64)
    points[:, 4] = np.nan
    broken = TrackResult(points=points, confidence=tracks.confidence, visible=tracks.visible)
    assert solve_frame(broken, 4).transform is None


def test_progress_is_reported(known_motion):
    """The solver drives a progress callback and always finishes at 1.0."""
    _, tracks = known_motion
    seen: list[float] = []
    solve_motion(tracks, frame_size=(320, 240), progress=lambda f, m="": seen.append(f))
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_solving_the_pixels_of_a_warped_clip(warped_clip: WarpedClipSource):
    """Tracks read off a rendered clip solve back to the motion that rendered it."""
    features = warped_clip.feature_points(6)
    solve = solve_motion(
        warped_clip.true_tracks(features),
        frame_size=(warped_clip.width, warped_clip.height),
    )
    for frame in range(warped_clip.frame_count):
        assert solve.transform_at(frame).allclose(warped_clip.relative_transform(frame), atol=TOL)
