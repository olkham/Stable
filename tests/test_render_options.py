"""Todo 17's render options at the library level: target, crop and border fill.

The server and CLI tests prove these reach the operator; these prove the maths
and the contracts underneath — including the one that matters most, that a
stabilisation target is a translation *composed onto*
``SmoothedMotion.stabilization_transform`` rather than a transform rebuilt from
the (pivot-relative) channel values.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import WarpedClipSource, handheld_transforms

from stable.core.errors import RenderError
from stable.core.geometry import SimilarityTransform
from stable.render import (
    BorderFill,
    CropSpec,
    RenderOptions,
    StabilizationPlan,
    StabilizationTarget,
    average_positions,
    coverage_scale,
    hold_gaps,
    plan_output,
    warp_frame,
)
from stable.smoothing import SmoothingSettings, smooth_motion
from stable.solver import solve_motion

pytest.importorskip("cv2")

GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0), (160.0, 120.0)]
WIDTH, HEIGHT = 200, 100


@pytest.fixture(scope="module")
def clip() -> WarpedClipSource:
    """A 30-frame textured clip moved by a known handheld path."""
    return WarpedClipSource(handheld_transforms(30, pivot=(160.0, 120.0), amplitude=9.0))


def motion_for(source: WarpedClipSource, settings: SmoothingSettings | None = None):
    """Solve and smooth a clip from its own ground-truth tracks."""
    solve = solve_motion(source.true_tracks(GRID), frame_size=(source.width, source.height))
    return smooth_motion(solve, settings or SmoothingSettings(), fps=source.fps)


def truth_path(source: WarpedClipSource, point: tuple[float, float]) -> np.ndarray:
    """Return where ``point`` really is on every frame of the clip."""
    return np.asarray(
        [source.transform_at(f).apply(np.asarray(point)) for f in range(source.frame_count)],
        dtype=np.float64,
    )


# -- option value types ---------------------------------------------------------


def test_options_round_trip_through_json():
    options = RenderOptions(
        target=StabilizationTarget(mode="point", point_id="p3", anchor_frame=7),
        crop=CropSpec(scale=0.75, centre="target"),
        fill=BorderFill(mode="blur", color=(10, 20, 30), blur=4.0, tint=0.5, feather=2.0),
    )
    assert RenderOptions.from_dict(options.to_dict()) == options
    # Anything absent takes its default, so an older project renders as before.
    assert RenderOptions.from_dict({}) == RenderOptions()
    assert RenderOptions.from_dict(None).target.mode == "frame"


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"mode": "middle"}, "unknown stabilisation target"),
        ({"mode": "point"}, "needs a point_id"),
        ({"mode": "centroid", "anchor_frame": -2}, "anchor_frame"),
    ],
)
def test_bad_targets_are_rejected_with_a_reason(kwargs, fragment):
    with pytest.raises(RenderError, match=fragment):
        StabilizationTarget(**kwargs)


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5, float("nan")])
def test_a_crop_scale_outside_the_unit_interval_is_rejected(scale):
    with pytest.raises(RenderError, match="crop scale"):
        CropSpec(scale=scale)


def test_an_unknown_fill_mode_or_out_of_range_number_is_rejected():
    with pytest.raises(RenderError, match="unknown border fill"):
        BorderFill(mode="feathered")
    with pytest.raises(RenderError, match="tint"):
        BorderFill(tint=1.5)
    with pytest.raises(RenderError, match="feather"):
        BorderFill(feather=-1.0)


def test_only_the_plain_solid_fill_keeps_the_single_warp_fast_path():
    assert BorderFill().is_plain
    assert BorderFill(mode="solid", color=(255, 0, 0)).is_plain
    assert not BorderFill(mode="mirror").is_plain
    assert not BorderFill(mode="solid", feather=3.0).is_plain
    assert not BorderFill(mode="solid", tint=0.4).is_plain


# -- resolving a target to positions ----------------------------------------------


def test_average_positions_ignores_untracked_frames():
    a = np.array([[0.0, 0.0], [10.0, 10.0], [np.nan, np.nan]])
    b = np.array([[4.0, 8.0], [np.nan, np.nan], [6.0, 2.0]])
    averaged = average_positions([a, b])
    assert averaged[0] == pytest.approx([2.0, 4.0])
    assert averaged[1] == pytest.approx([10.0, 10.0])  # only `a` had this frame
    assert averaged[2] == pytest.approx([6.0, 2.0])


def test_average_positions_leaves_a_frame_nobody_covered_as_a_gap():
    a = np.array([[np.nan, np.nan], [1.0, 1.0]])
    assert np.isnan(average_positions([a])[0]).all()
    with pytest.raises(RenderError, match="at least one tracked point"):
        average_positions([])


def test_hold_gaps_holds_the_nearest_valid_sample_in_both_directions():
    path = np.array(
        [[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan], [5.0, 6.0], [np.nan, np.nan]]
    )
    held = hold_gaps(path)
    assert not np.isnan(held).any()
    assert held[0] == pytest.approx([1.0, 2.0])  # filled backwards from frame 1
    assert held[2] == pytest.approx([1.0, 2.0])  # held forwards from frame 1
    assert held[4] == pytest.approx([5.0, 6.0])


def test_a_target_that_was_never_tracked_says_so():
    with pytest.raises(RenderError, match="no tracked position anywhere"):
        hold_gaps(np.full((6, 2), np.nan))


# -- the target lock ------------------------------------------------------------


def test_a_target_pins_its_point_with_no_smoothing_at_all(clip):
    """The heart of todo 17: stabilisation without configuring a filter."""
    motion = motion_for(clip)  # no filters => the base warp is the identity
    plain = StabilizationPlan.build(motion, source_size=(clip.width, clip.height))
    assert plain.is_no_op

    feature = GRID[4]
    plan = StabilizationPlan.build(
        motion,
        source_size=(clip.width, clip.height),
        target_positions=truth_path(clip, feature),
    )
    assert not plan.is_no_op
    landed = np.asarray(
        [plan.output_transform(f).apply(np.asarray(clip.transform_at(f).apply(feature)))
         for f in range(plan.n_frames)]
    )
    assert np.ptp(landed, axis=0).max() < 1e-6


def test_the_target_lock_composes_onto_the_sanctioned_warp(clip):
    """Pivot safety: the extra term is a pure translation, nothing is rebuilt.

    Rebuilding a transform from the solve's channel values (the mistake Phase 3
    recorded) would change the rotation and scale too. Asserting they survive
    untouched is what makes that impossible to reintroduce here.
    """
    motion = motion_for(clip, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}))
    base = motion.stabilization_transforms()
    plan = StabilizationPlan.build(
        motion,
        source_size=(clip.width, clip.height),
        target_positions=truth_path(clip, GRID[0]),
        border="transparent",
    )
    moved = 0.0
    for frame, warp in enumerate(plan.transforms):
        assert warp.rotation == pytest.approx(base[frame].rotation, abs=1e-12)
        assert warp.scale == pytest.approx(base[frame].scale, abs=1e-12)
        moved = max(moved, abs(warp.tx - base[frame].tx) + abs(warp.ty - base[frame].ty))
    assert moved > 1.0, "the target should have had something left to correct"


def test_the_anchor_frame_chooses_which_framing_is_held(clip):
    motion = motion_for(clip)
    positions = truth_path(clip, GRID[4])
    for anchor in (0, 15, 29):
        plan = StabilizationPlan.build(
            motion,
            source_size=(clip.width, clip.height),
            target_positions=positions,
            anchor_frame=anchor,
            border="transparent",
        )
        # At its own anchor frame the render must not move the picture at all.
        assert plan.transforms[anchor].is_identity(atol=1e-9)


def test_a_target_from_a_different_clip_is_refused(clip):
    motion = motion_for(clip)
    with pytest.raises(RenderError, match="different clip"):
        StabilizationPlan.build(
            motion, source_size=(clip.width, clip.height), target_positions=np.zeros((4, 2))
        )
    with pytest.raises(RenderError, match="anchor frame"):
        StabilizationPlan.build(
            motion,
            source_size=(clip.width, clip.height),
            target_positions=truth_path(clip, GRID[0]),
            anchor_frame=999,
        )


def test_max_displacement_measures_stabilisation_not_zoom(clip):
    """A plan that only zooms is still a no-op as far as motion goes."""
    motion = motion_for(clip)
    plan = StabilizationPlan.build(
        motion, source_size=(clip.width, clip.height), crop=CropSpec(scale=0.5)
    )
    assert plan.geometry.zoom == pytest.approx(2.0)
    assert plan.max_displacement < 1e-9
    assert plan.is_no_op


# -- crop geometry ---------------------------------------------------------------


def test_an_auto_crop_still_gives_exactly_the_phase_3_geometry():
    shifted = [SimilarityTransform(tx=10.0), SimilarityTransform(tx=-10.0)]
    legacy = plan_output(shifted, WIDTH, HEIGHT, "auto_zoom")
    explicit = plan_output(shifted, WIDTH, HEIGHT, "auto_zoom", crop=CropSpec())
    assert legacy.zoom == pytest.approx(explicit.zoom)
    assert legacy.fit.allclose(explicit.fit)
    assert legacy.crop_scale == pytest.approx(legacy.coverage)
    assert not legacy.gaps


def test_an_explicit_crop_overrides_the_measured_coverage():
    shifted = [SimilarityTransform(tx=10.0), SimilarityTransform(tx=-10.0)]
    tight = plan_output(shifted, WIDTH, HEIGHT, "crop", crop=CropSpec(scale=0.5))
    assert (tight.width, tight.height) == (100, 50)
    assert tight.crop_scale == pytest.approx(0.5)
    assert not tight.gaps  # cropping harder than needed cannot expose anything

    loose = plan_output(shifted, WIDTH, HEIGHT, "crop", crop=CropSpec(scale=1.0))
    assert (loose.width, loose.height) == (WIDTH, HEIGHT)
    assert loose.coverage < 1.0
    assert loose.gaps


def test_an_explicit_crop_can_render_a_shot_with_no_common_area():
    """Auto has nothing to crop to; an explicit crop plus a fill still delivers."""
    apart = [SimilarityTransform(tx=-500.0), SimilarityTransform(tx=500.0)]
    with pytest.raises(RenderError, match="share no common area"):
        plan_output(apart, WIDTH, HEIGHT, "auto_zoom")
    rescued = plan_output(apart, WIDTH, HEIGHT, "crop", crop=CropSpec(scale=1.0))
    assert rescued.gaps
    assert (rescued.width, rescued.height) == (WIDTH, HEIGHT)


def test_the_crop_can_be_centred_somewhere_other_than_the_frame_centre():
    still = [SimilarityTransform.identity()]
    off = plan_output(still, WIDTH, HEIGHT, "crop", centre=(60.0, 30.0), crop=CropSpec(scale=0.5))
    assert off.centre == (60.0, 30.0)
    # The requested centre lands in the middle of the output rectangle.
    assert off.fit.apply(np.array([60.0, 30.0])) == pytest.approx([off.width / 2, off.height / 2])


def test_coverage_measured_about_an_off_centre_point_is_smaller():
    shifted = [SimilarityTransform(tx=10.0), SimilarityTransform(tx=-10.0)]
    centred = coverage_scale(shifted, WIDTH, HEIGHT)
    off = coverage_scale(shifted, WIDTH, HEIGHT, centre=(WIDTH / 2 + 30.0, HEIGHT / 2))
    assert off < centred


def test_crop_around_the_target_needs_a_target(clip):
    """With no target to pin, "centre on the target" falls back to the frame."""
    motion = motion_for(clip, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}))
    plan = StabilizationPlan.build(
        motion, source_size=(clip.width, clip.height), crop=CropSpec(centre="target")
    )
    assert plan.geometry.centre == (clip.width / 2.0, clip.height / 2.0)

    pinned = StabilizationPlan.build(
        motion,
        source_size=(clip.width, clip.height),
        target_positions=truth_path(clip, GRID[0]),
        crop=CropSpec(scale=0.6, centre="target"),
    )
    assert pinned.geometry.centre != (clip.width / 2.0, clip.height / 2.0)


# -- border fill pixels -------------------------------------------------------------


def texture() -> np.ndarray:
    """A frame whose every pixel is bright, so a fill is unmistakable."""
    rng = np.random.default_rng(3)
    return rng.integers(120, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def black_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that are (near) black — i.e. letterbox, not picture."""
    return float(np.count_nonzero(image.max(axis=2) < 12)) / image[:, :, 0].size


@pytest.mark.parametrize("mode", ["stretch", "mirror", "blur"])
def test_every_non_solid_fill_removes_the_black_edge(mode):
    shifted = SimilarityTransform(tx=40.0, ty=20.0)
    letterboxed = warp_frame(texture(), shifted, border="crop")
    assert black_fraction(letterboxed) > 0.2

    filled = warp_frame(texture(), shifted, border="crop", fill=BorderFill(mode=mode))
    assert black_fraction(filled) == 0.0
    # The picture itself is untouched: only the exposed area changed.
    assert np.array_equal(letterboxed[30:80, 60:180], filled[30:80, 60:180])


def test_a_solid_fill_uses_its_colour_and_stays_one_warp():
    shifted = SimilarityTransform(tx=40.0)
    filled = warp_frame(
        texture(), shifted, border="crop", fill=BorderFill(mode="solid", color=(200, 10, 10))
    )
    assert tuple(int(v) for v in filled[HEIGHT // 2, 5]) == (200, 10, 10)


def test_tint_and_feather_combine_with_any_plate():
    shifted = SimilarityTransform(tx=40.0)
    mirrored = warp_frame(texture(), shifted, border="crop", fill=BorderFill(mode="mirror"))
    tinted = warp_frame(
        texture(),
        shifted,
        border="crop",
        fill=BorderFill(mode="mirror", color=(0, 0, 0), tint=0.8, feather=4.0),
    )
    # 80% toward black is much darker out there, but still not a hard black edge.
    edge = slice(0, 20)
    assert tinted[:, edge].mean() < mirrored[:, edge].mean() / 2.0
    assert black_fraction(tinted) == 0.0


def test_a_fill_is_ignored_where_alpha_already_answers_the_question():
    shifted = SimilarityTransform(tx=40.0)
    rgba = warp_frame(
        texture(), shifted, border="transparent", fill=BorderFill(mode="mirror")
    )
    assert rgba.shape[2] == 4
    assert rgba[HEIGHT // 2, 5, 3] == 0


def test_the_plan_carries_the_fill_into_every_rendered_frame(clip):
    from stable.render import iter_stabilized_frames, stabilized_frame  # noqa: PLC0415

    motion = motion_for(clip, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}))

    def plan_with(fill: BorderFill | None) -> StabilizationPlan:
        return StabilizationPlan.build(
            motion,
            source_size=(clip.width, clip.height),
            border="crop",
            crop=CropSpec(scale=1.0),
            fill=fill,
        )

    filled, letterboxed = plan_with(BorderFill(mode="mirror")), plan_with(None)
    assert filled.geometry.gaps
    assert filled.to_dict()["fill"]["mode"] == "mirror"

    # This clip's own texture is dark, so "no black at all" is the wrong test:
    # what matters is that the letterbox the fill was asked to remove is gone.
    window = (10, 14)
    baseline = [
        black_fraction(f) for f in iter_stabilized_frames(clip, letterboxed, frame_range=window)
    ]
    assert min(baseline) > 0.005, baseline
    for index, frame in enumerate(iter_stabilized_frames(clip, filled, frame_range=window)):
        assert black_fraction(frame) < baseline[index] / 4.0, (index, baseline[index])
    assert black_fraction(stabilized_frame(clip, filled, 12)) < min(baseline) / 4.0
