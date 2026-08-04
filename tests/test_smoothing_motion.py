"""Smoothing a solve, and the stabilisation warp that falls out of it."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import handheld_transforms

from stable.core.errors import SmoothingError
from stable.core.geometry import CHANNELS
from stable.core.types import TrackResult
from stable.smoothing import MIN_SCALE, SmoothedMotion, SmoothingSettings, smooth_motion
from stable.solver import MotionSolve, SolveConfig, solve_motion

PIVOT = (160.0, 120.0)
GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0)]


def build_solve(n_frames: int = 60, **kwargs):
    """Return a solve of a known handheld path through a fixed point grid."""
    transforms = handheld_transforms(n_frames, pivot=PIVOT, **kwargs)
    source = np.asarray(GRID, dtype=np.float64)
    points = np.stack([np.asarray(t.apply(source)) for t in transforms], axis=1)
    tracks = TrackResult(
        points=points,
        confidence=np.ones(points.shape[:2], dtype=np.float32),
        visible=np.ones(points.shape[:2], dtype=bool),
    )
    return transforms, solve_motion(tracks, frame_size=(320, 240))


@pytest.fixture
def solved():
    """A 60-frame solve of a handheld path."""
    return build_solve()[1]


def roughness(curve: np.ndarray) -> float:
    """Return total absolute second difference — how jagged a curve is."""
    return float(np.abs(np.diff(np.asarray(curve), 2)).sum())


# -- the smoothed curves -----------------------------------------------------


def test_no_settings_means_no_smoothing_and_an_identity_warp(solved):
    """The default is to change nothing, so nothing is stabilised."""
    motion = smooth_motion(solved, fps=30.0)
    for channel in CHANNELS:
        assert np.allclose(motion.curve(channel), solved.curve(channel))
    for frame in range(solved.n_frames):
        assert motion.stabilization_transform(frame).is_identity(atol=1e-9)


def test_smoothing_reduces_the_jitter_in_every_channel(solved):
    """The point of the exercise: smoother curves than the solve found."""
    motion = smooth_motion(
        solved, SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}), fps=30.0
    )
    for channel in CHANNELS:
        assert roughness(motion.curve(channel)) < roughness(solved.curve(channel))


def test_a_disabled_channel_is_left_exactly_alone(solved):
    """"Don't stabilise rotation" must be exact, not approximately exact."""
    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0})
    settings.set_enabled("r", False)
    motion = smooth_motion(solved, settings, fps=30.0)

    assert np.array_equal(motion.curve("r"), solved.curve("r"))
    for frame in range(solved.n_frames):
        assert motion.stabilization_transform(frame).rotation == pytest.approx(0.0, abs=1e-12)


def test_channels_are_smoothed_independently(solved):
    """Configuring one channel does not disturb the others."""
    settings = SmoothingSettings.uniform("gaussian", {"sigma": 8.0}, channels=["x"])
    motion = smooth_motion(solved, settings, fps=30.0)
    assert not np.allclose(motion.curve("x"), solved.curve("x"))
    for channel in ("y", "r", "s"):
        assert np.array_equal(motion.curve(channel), solved.curve(channel))


def test_strength_scales_how_far_the_frame_is_pushed(solved):
    """Half strength moves the frame half as far as full strength."""
    full = smooth_motion(
        solved, SmoothingSettings.uniform("gaussian", {"sigma": 10.0}), fps=30.0
    )
    half = smooth_motion(
        solved,
        SmoothingSettings.uniform("gaussian", {"sigma": 10.0, "strength": 0.5}),
        fps=30.0,
    )
    assert np.allclose(half.correction("x"), 0.5 * full.correction("x"), atol=1e-9)
    assert np.abs(half.correction("x")).max() < np.abs(full.correction("x")).max()


def test_the_correction_curve_is_smoothed_minus_raw(solved):
    """What the render will undo, and the curve editor's before/after difference."""
    motion = smooth_motion(
        solved, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}), fps=30.0
    )
    for channel in CHANNELS:
        assert np.allclose(
            motion.correction(channel), motion.curve(channel) - solved.curve(channel)
        )


def test_the_scale_curve_is_clamped_above_zero():
    """A smoothed scale of zero would make the warp non-invertible; it cannot happen.

    A hard zoom-out gives a scale curve steep enough that a heavy spline fit
    (which tends to a straight line) extrapolates through zero at the end of the
    clip. The clamp is what stops that becoming an un-renderable frame.
    """
    frames = 60
    steep = 20.0 / (np.arange(frames, dtype=np.float64) + 1.0)
    solve = MotionSolve(
        curves={
            "x": np.zeros(frames),
            "y": np.zeros(frames),
            "r": np.zeros(frames),
            "s": steep,
        },
        residual=np.zeros(frames),
        n_points=np.full(frames, 4, np.int32),
        n_inliers=np.full(frames, 4, np.int32),
        status=np.zeros(frames, np.int8),
        reference_frame=0,
        pivot=PIVOT,
    )
    settings = SmoothingSettings.uniform("spline", {"lam": 1e12}, channels=["s"])
    motion = smooth_motion(solve, settings, fps=30.0)

    assert (motion.curve("s") >= MIN_SCALE).all()
    # Every frame still yields an invertible warp, which is the point of the clamp.
    for frame in range(frames):
        assert motion.stabilization_transform(frame).scale != 0.0


def test_frame_rate_changes_what_a_frequency_filter_does(solved):
    """A 1 Hz cutoff means different things at 24 and 60 fps, and must."""
    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0})
    slow = smooth_motion(solved, settings, fps=24.0)
    fast = smooth_motion(solved, settings, fps=60.0)
    assert not np.allclose(slow.curve("x"), fast.curve("x"))
    assert roughness(fast.curve("x")) < roughness(slow.curve("x"))


# -- the stabilisation warp --------------------------------------------------


def test_the_warp_is_smoothed_composed_with_the_inverse_raw(solved):
    """``smoothed(t) . raw(t)^-1``, exactly as the plan specifies."""
    motion = smooth_motion(
        solved, SmoothingSettings.uniform("gaussian", {"sigma": 8.0}), fps=30.0
    )
    for frame in (0, 13, 41, 59):
        expected = motion.transform_at(frame).compose(solved.transform_at(frame).inverse())
        assert motion.stabilization_transform(frame).allclose(expected, atol=1e-12)


def test_the_warp_undoes_the_motion_it_removed(solved):
    """Applying the warp to a tracked point lands it on the smoothed path."""
    motion = smooth_motion(
        solved, SmoothingSettings.uniform("gaussian", {"sigma": 8.0}), fps=30.0
    )
    probe = np.asarray([PIVOT])
    for frame in (5, 25, 50):
        raw_position = solved.transform_at(frame).apply(probe)
        warped = motion.stabilization_transform(frame).apply(raw_position)
        assert np.allclose(warped, motion.transform_at(frame).apply(probe), atol=1e-9)


def test_tripod_locks_the_frame_to_the_reference():
    """Every channel on tripod makes the warp exactly ``raw(t)^-1``."""
    transforms, solve = build_solve(40)
    motion = smooth_motion(solve, SmoothingSettings.tripod(), fps=30.0)

    for frame in range(solve.n_frames):
        assert motion.transform_at(frame).is_identity(atol=1e-9)
        assert motion.stabilization_transform(frame).allclose(
            solve.transform_at(frame).inverse(), atol=1e-9
        )
    # A point that moved with the camera comes back to where it started.
    probe = np.asarray([[100.0, 80.0]])
    for frame in (0, 17, 39):
        moved = transforms[frame].apply(probe)
        assert np.allclose(motion.stabilization_transform(frame).apply(moved), probe, atol=1e-3)


def test_tripod_on_a_non_zero_reference_frame_locks_to_that_frame():
    """The lock is to the reference frame, not to frame zero."""
    _, solve = build_solve(30)
    resolved = solve_motion(
        TrackResult(
            points=np.stack(
                [
                    np.stack([solve.transform_at(f).apply(np.asarray(GRID))[i] for f in range(30)])
                    for i in range(len(GRID))
                ]
            ),
            confidence=np.ones((len(GRID), 30), dtype=np.float32),
            visible=np.ones((len(GRID), 30), dtype=bool),
        ),
        config=SolveConfig(reference_frame=12),
        frame_size=(320, 240),
    )
    motion = smooth_motion(resolved, SmoothingSettings.tripod(), fps=30.0)
    assert motion.reference_frame == 12
    assert motion.stabilization_transform(12).is_identity(atol=1e-6)


def test_transforms_and_bounds(solved):
    """Every frame has a warp, and out-of-range frames raise."""
    motion = smooth_motion(solved, SmoothingSettings.tripod(), fps=30.0)
    assert len(motion) == solved.n_frames
    assert len(motion.stabilization_transforms()) == solved.n_frames
    with pytest.raises(IndexError):
        motion.transform_at(solved.n_frames)


def test_the_pivot_and_reference_come_from_the_solve(solved):
    """Smoothing never re-bases the motion; it only changes its shape."""
    motion = smooth_motion(solved, fps=30.0)
    assert motion.pivot == solved.pivot
    assert motion.reference_frame == solved.reference_frame


# -- container and exports ---------------------------------------------------


def test_curves_are_read_only(solved):
    """Smoothed curves are immutable, like everything else in the pipeline."""
    motion = smooth_motion(solved, SmoothingSettings.tripod(), fps=30.0)
    with pytest.raises(ValueError):
        motion.curve("x")[0] = 1.0


def test_unknown_channel_is_rejected(solved):
    """A bad channel name is a smoothing error, not a KeyError."""
    motion = smooth_motion(solved, fps=30.0)
    with pytest.raises(SmoothingError, match="unknown channel"):
        motion.curve("q")


def test_malformed_curves_are_rejected(solved):
    """A curve of the wrong length would silently misalign the whole render."""
    good = {c: np.zeros(solved.n_frames) for c in CHANNELS}
    with pytest.raises(SmoothingError, match="missing curve"):
        SmoothedMotion(solved, {"x": good["x"]}, SmoothingSettings(), 30.0)
    with pytest.raises(SmoothingError, match="must have shape"):
        SmoothedMotion(solved, {**good, "r": np.zeros(3)}, SmoothingSettings(), 30.0)


def test_csv_export_puts_raw_and_smoothed_side_by_side(tmp_path, solved):
    """One file a user can open and see exactly what the render will do."""
    motion = smooth_motion(
        solved, SmoothingSettings.uniform("gaussian", {"sigma": 8.0}), fps=30.0
    )
    lines = open(motion.save(tmp_path / "curves.csv"), encoding="utf-8").read().splitlines()
    header = lines[0].split(",")

    assert header[:9] == [
        "frame", "raw_x", "raw_y", "raw_r", "raw_s", "smooth_x", "smooth_y", "smooth_r", "smooth_s"
    ]
    assert len(lines) == solved.n_frames + 1
    row = lines[6].split(",")
    assert float(row[1]) == pytest.approx(solved.curve("x")[5], abs=1e-6)
    assert float(row[5]) == pytest.approx(motion.curve("x")[5], abs=1e-6)


def test_json_export_carries_the_solve_and_the_settings(tmp_path, solved):
    """Enough to reconstruct what was done, not just what came out."""
    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 0.9})
    motion = smooth_motion(solved, settings, fps=30.0)
    data = json.loads(open(motion.save(tmp_path / "curves.json"), encoding="utf-8").read())

    assert data["fps"] == 30.0
    assert data["solve"]["reference_frame"] == 0
    assert data["smoothing"]["x"]["filters"][0]["filter"] == "butterworth"
    assert len(data["smoothed"]["x"]) == solved.n_frames


def test_repr_names_the_active_channels(solved):
    """Debug output says what is actually being stabilised."""
    settings = SmoothingSettings.uniform("gaussian", channels=["x", "r"])
    assert "x,r" in repr(smooth_motion(solved, settings, fps=30.0))
    assert "none" in repr(smooth_motion(solved, fps=30.0))


# -- interaction with degraded solves ----------------------------------------


def test_smoothing_works_across_an_interpolated_hole():
    """A filled gap is a normal part of the curve by the time smoothing sees it."""
    transforms = handheld_transforms(40, pivot=PIVOT)
    source = np.asarray(GRID, dtype=np.float64)
    points = np.stack([np.asarray(t.apply(source)) for t in transforms], axis=1)
    points[:, 15:20] = np.nan
    solve = solve_motion(
        TrackResult(
            points=points,
            confidence=np.ones(points.shape[:2], dtype=np.float32),
            visible=np.ones(points.shape[:2], dtype=bool),
        ),
        frame_size=(320, 240),
    )
    motion = smooth_motion(
        solve, SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}), fps=30.0
    )
    for channel in CHANNELS:
        assert np.isfinite(motion.curve(channel)).all()
    assert roughness(motion.curve("x")) < roughness(solve.curve("x"))


def test_a_one_point_solve_still_stabilises_translation():
    """Degrading to X/Y does not stop the render working, it just does less."""
    transforms = handheld_transforms(30, pivot=PIVOT)
    points = np.stack(
        [np.asarray(t.apply(np.asarray([[120.0, 100.0]])))[0] for t in transforms]
    )[None]
    solve = solve_motion(
        TrackResult(
            points=points,
            confidence=np.ones((1, 30), dtype=np.float32),
            visible=np.ones((1, 30), dtype=bool),
        ),
        frame_size=(320, 240),
    )
    motion = smooth_motion(solve, SmoothingSettings.tripod(), fps=30.0)
    warps = motion.stabilization_transforms()
    assert all(w.rotation == pytest.approx(0.0) for w in warps)
    assert all(w.scale == pytest.approx(1.0) for w in warps)
    assert max(abs(w.tx) for w in warps) > 1.0
