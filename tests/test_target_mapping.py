"""Mapping tracked points onto chosen target positions (todo 5).

Pinning *one* point can only translate the picture: a shot that rotates or zooms
still rotates and zooms around the pinned feature. Stating where **two** points
should end up is a different request, and the transform that satisfies it is
exactly the rotation and scale the shot needs — which is what the report asked
for ("this would automatically mean we compensate for scale and rotation in
order to map the two tracked points to two target points").

These tests are about that claim, not about the plumbing: a clip whose camera
rotates and zooms by known amounts must come out with both mapped points sitting
on their targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import RenderError
from stable.core.geometry import SimilarityTransform
from stable.render.options import StabilizationTarget, TargetPair
from stable.render.stabilize import StabilizationPlan
from stable.smoothing.channels import SmoothingSettings
from stable.smoothing.motion import SmoothedMotion
from stable.solver.motion import MotionSolve

FRAMES = 24
SIZE = (320, 240)
PIVOT = (160.0, 120.0)


def _identity_motion(frames: int = FRAMES) -> SmoothedMotion:
    """A solve with no motion at all, so the base warp is the identity.

    That isolates the mapping: anything the plan does to the picture here is the
    target mapping's doing, not the smoothing's.
    """
    flat = {
        "x": np.zeros(frames),
        "y": np.zeros(frames),
        "r": np.zeros(frames),
        "s": np.ones(frames),
    }
    solve = MotionSolve(
        curves=dict(flat),
        residual=np.zeros(frames),
        n_points=np.full(frames, 4, np.int32),
        n_inliers=np.full(frames, 4, np.int32),
        status=np.zeros(frames, np.int8),
        reference_frame=0,
        pivot=PIVOT,
    )
    return SmoothedMotion(
        solve=solve,
        curves=dict(flat),
        settings=SmoothingSettings(),  # nothing enabled: the base warp is the identity
        fps=24.0,
    )


def _moving_paths(transform_at) -> tuple[np.ndarray, np.ndarray]:
    """Two features carried through the clip by ``transform_at(frame)``."""
    start = np.array([[100.0, 100.0], [220.0, 140.0]])
    paths = np.stack(
        [
            np.stack([transform_at(f).apply(start[k]) for f in range(FRAMES)])
            for k in range(len(start))
        ]
    )
    return paths, start


def test_one_pair_puts_the_point_exactly_where_it_was_asked_to_go():
    paths, _start = _moving_paths(lambda f: SimilarityTransform(tx=2.0 * f, ty=-1.0 * f))
    # Where it sat on frame 0 — far enough for the drift to matter, close enough
    # that an automatic crop still finds common area to keep.
    destination = paths[:1, 0]

    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=paths[:1],
        mapped_destinations=destination,
    )

    for frame in range(FRAMES):
        landed = plan.transforms[frame].apply(paths[0, frame])
        assert landed == pytest.approx(destination[0], abs=1e-6)


def test_one_pair_only_translates_it_never_rotates_or_scales():
    """A single correspondence cannot observe rotation or scale, and must not invent one."""
    paths, _start = _moving_paths(lambda f: SimilarityTransform(rotation=0.01 * f, scale=1.0))
    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=paths[:1],
        mapped_destinations=np.array([[160.0, 120.0]]),
    )
    for transform in plan.transforms:
        assert transform.rotation == pytest.approx(0.0, abs=1e-12)
        assert transform.scale == pytest.approx(1.0, abs=1e-12)


def test_two_pairs_undo_a_rotating_zooming_camera():
    """The headline claim: two points onto two targets removes rotation and scale."""
    rotation_per_frame = 0.02
    scale_per_frame = 1.004

    def camera(frame: int) -> SimilarityTransform:
        about_centre = SimilarityTransform(
            rotation=rotation_per_frame * frame, scale=scale_per_frame**frame
        )
        # Rotate/scale about the frame centre, the way a real camera move reads.
        offset = np.asarray(PIVOT) - about_centre.apply(np.asarray(PIVOT))
        return SimilarityTransform(
            tx=float(offset[0]),
            ty=float(offset[1]),
            rotation=about_centre.rotation,
            scale=about_centre.scale,
        ).compose(about_centre.__class__())  # keep the composed form explicit

    paths, start = _moving_paths(camera)
    # Ask for each feature back where it started on frame 0.
    destinations = np.stack([paths[0, 0], paths[1, 0]])

    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=paths,
        mapped_destinations=destinations,
    )

    worst = 0.0
    for frame in range(FRAMES):
        for k in range(2):
            landed = plan.transforms[frame].apply(paths[k, frame])
            worst = max(worst, float(np.hypot(*(landed - destinations[k]))))
    assert worst < 1e-6, f"mapped points missed their targets by up to {worst:.3g}px"

    # ...and it did so by rotating and scaling, not merely shifting.
    assert abs(plan.transforms[-1].rotation) > 0.1
    assert abs(plan.transforms[-1].scale - 1.0) > 0.01


def test_three_pairs_that_disagree_are_least_squares_not_a_failure():
    """A similarity cannot satisfy three arbitrary pairs; it must fit, not refuse."""
    paths, _start = _moving_paths(lambda f: SimilarityTransform())
    third = np.stack([np.array([200.0, 200.0])] * FRAMES)[None]
    sources = np.concatenate([paths, third])
    # Destinations that no similarity can hit exactly (a sheared arrangement).
    destinations = np.array([[100.0, 100.0], [220.0, 140.0], [150.0, 260.0]])

    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=sources,
        mapped_destinations=destinations,
    )

    landed = np.stack([plan.transforms[0].apply(sources[k, 0]) for k in range(3)])
    errors = np.hypot(*(landed - destinations).T)
    assert errors.max() > 1e-3, "this arrangement should not be exactly satisfiable"
    # A least-squares fit spreads the error rather than abandoning any one pair.
    assert errors.max() < 60.0


def test_a_mapping_needs_both_halves_of_every_pair():
    paths, _start = _moving_paths(lambda f: SimilarityTransform())
    with pytest.raises(RenderError, match="every pair needs both halves"):
        StabilizationPlan.build(
            _identity_motion(),
            source_size=SIZE,
            border="crop",
            mapped_sources=paths,
            mapped_destinations=np.array([[10.0, 10.0]]),
        )


def test_a_mapping_over_a_different_clip_is_refused():
    paths, _start = _moving_paths(lambda f: SimilarityTransform())
    with pytest.raises(RenderError, match="different clip"):
        StabilizationPlan.build(
            _identity_motion(frames=FRAMES + 10),
            source_size=SIZE,
            border="crop",
            mapped_sources=paths,
            mapped_destinations=np.stack([paths[0, 0], paths[1, 0]]),
        )


def test_a_gap_in_a_mapped_track_is_held_not_lurched_through():
    """Gaps are held, the same rule the single-target lock uses."""
    paths, _start = _moving_paths(lambda f: SimilarityTransform(tx=1.0 * f))
    holed = paths.copy()
    holed[0, 5:8] = np.nan

    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=holed,
        mapped_destinations=np.stack([paths[0, 0], paths[1, 0]]),
    )
    for transform in plan.transforms:
        assert np.isfinite([transform.tx, transform.ty, transform.rotation, transform.scale]).all()


# -- the options model ---------------------------------------------------------


def test_a_map_target_needs_at_least_one_pair():
    with pytest.raises(RenderError, match="at least one"):
        StabilizationTarget(mode="map")


def test_a_point_cannot_be_mapped_to_two_places():
    with pytest.raises(RenderError, match="duplicates"):
        StabilizationTarget(
            mode="map",
            pairs=(TargetPair("p1", 10.0, 10.0), TargetPair("p1", 20.0, 20.0)),
        )


def test_the_target_round_trips_through_its_dict_form():
    target = StabilizationTarget(
        mode="map", pairs=(TargetPair("p1", 10.0, 20.0), TargetPair("p2", 30.0, 40.0))
    )
    restored = StabilizationTarget.from_dict(target.to_dict())
    assert restored == target
    assert restored.uses_rotation_and_scale is True
    single = StabilizationTarget(mode="map", pairs=(TargetPair("p1", 1.0, 2.0),))
    assert single.uses_rotation_and_scale is False


def test_a_frame_with_no_confidence_still_gets_a_transform():
    """The live bug: outside the tracked span every pair scores zero.

    Confidence weights the fit, but a frame where *nothing* has confidence has
    no information to weigh — not a reason to refuse. Before this, planning a
    mapped export over a clip with an In/Out range raised "weights sum to zero"
    and the whole render failed.
    """
    paths, _start = _moving_paths(lambda f: SimilarityTransform(tx=1.0 * f))
    destinations = np.stack([paths[0, 0], paths[1, 0]])
    weights = np.ones((2, FRAMES))
    weights[:, 10:15] = 0.0  # untracked frames

    plan = StabilizationPlan.build(
        _identity_motion(),
        source_size=SIZE,
        border="crop",
        mapped_sources=paths,
        mapped_destinations=destinations,
        mapped_weights=weights,
    )

    assert len(plan.transforms) == FRAMES
    for frame in range(10, 15):
        transform = plan.transforms[frame]
        assert np.isfinite([transform.tx, transform.ty, transform.rotation, transform.scale]).all()
        # Equal weights still satisfy both pairs on those frames.
        for k in range(2):
            landed = transform.apply(paths[k, frame])
            assert landed == pytest.approx(destinations[k], abs=1e-6)
