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
    ExportRange,
    OutputResolution,
    Placement,
    RenderOptions,
    StabilizationPlan,
    StabilizationTarget,
    TargetPoint,
    average_positions,
    coverage_scale,
    hold_gaps,
    plan_output,
    window_size,
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
        target=StabilizationTarget(
            points=(
                TargetPoint(id="t1", place=Placement(1 / 3, 0.25), point_ids=("p3",)),
                TargetPoint(id="t2", place=Placement(0.7, 0.6), source="smoothed"),
            )
        ),
        crop=CropSpec(scale=0.75, limit_to_range=True),
        fill=BorderFill(mode="blur", color=(10, 20, 30), blur=4.0, tint=0.5, feather=2.0),
        export_range=ExportRange(start=12, end=200),
        resolution=OutputResolution(1920, 1080),
    )
    assert RenderOptions.from_dict(options.to_dict()) == options
    # Anything absent takes its default: no marks, which is the smoothing warp.
    assert RenderOptions.from_dict({}) == RenderOptions()
    assert RenderOptions.from_dict(None).target.points == ()


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # `frame` held nothing, so it becomes no marks at all.
        ({"mode": "frame"}, []),
        ({"mode": "point", "point_id": "p2"}, [(("p2",), "raw")]),
        ({"mode": "track", "point_id": "p2"}, [(("p2",), "raw")]),
        ({"mode": "smoothed_track", "point_id": "p2"}, [(("p2",), "smoothed")]),
        # `centroid` / `average` named no point and meant "all of them", which
        # is exactly what an empty track list means now.
        ({"mode": "centroid"}, [((), "raw")]),
        ({"mode": "selected", "point_ids": ["p1", "p3"]}, [(("p1", "p3"), "raw")]),
        (
            {"mode": "map", "pairs": [{"point_id": "p1", "u": 0.2, "v": 0.3}]},
            [(("p1",), "raw")],
        ),
    ],
)
def test_sidecars_from_before_the_target_list_still_load(stored, expected):
    """Three generations of stored target must not become junk on read."""
    target = StabilizationTarget.from_dict(stored)
    assert [(p.point_ids, p.source) for p in target.points] == expected
    # Whatever they migrate to, the ids have to be usable and distinct.
    assert len({p.id for p in target.points}) == len(target.points)


def test_a_legacy_map_mark_is_read_as_a_fraction_of_the_frame():
    """Marks used to be pixels before the crop fit; they are output fractions now.

    The clip size is what converts them, and it is only available when the
    caller has the project — so the conversion is offered rather than assumed,
    and a mark read without it goes to the centre rather than being taken for a
    fraction it plainly is not.
    """
    stored = {"mode": "map", "pairs": [{"point_id": "p1", "x": 480.0, "y": 270.0}]}
    converted = StabilizationTarget.from_dict(stored, (1920, 1080))
    assert converted.points[0].place.fraction == pytest.approx((0.25, 0.25))
    blind = StabilizationTarget.from_dict(stored)
    assert blind.points[0].place.fraction == (0.5, 0.5)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"source": "guessed"}, "unknown target source"),
        ({"point_ids": ("p1", "p1")}, "only be mapped once"),
        ({"weight": 0.0}, "weight must be positive"),
        ({"id": ""}, "needs an id"),
    ],
)
def test_bad_target_points_are_rejected_with_a_reason(kwargs, fragment):
    with pytest.raises(RenderError, match=fragment):
        TargetPoint(**kwargs)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_a_placement_outside_the_output_frame_is_rejected(value):
    with pytest.raises(RenderError, match="placement"):
        Placement(u=value)


def test_the_output_resolution_scales_the_crop_or_follows_it():
    assert OutputResolution().is_source
    assert OutputResolution().resolve((640, 360)) == (640, 360)
    assert OutputResolution(1920, 1080).resolve((640, 360)) == (1920, 1080)
    # One side alone keeps the shape and changes the size.
    assert OutputResolution(width=1280).resolve((640, 360)) == (1280, 720)
    assert OutputResolution(height=720).resolve((640, 360)) == (1280, 720)
    # Both sides set is also a statement about the crop window's shape.
    assert OutputResolution(1080, 1080).aspect == (1080.0, 1080.0)
    assert OutputResolution(width=1080).aspect is None
    with pytest.raises(RenderError, match="at least 2 pixels"):
        OutputResolution(width=1)


def test_the_crop_window_takes_the_delivered_aspect_ratio():
    """A square delivery crops the biggest square, rather than squeezing 16:9."""
    assert window_size(1920, 1080, None) == (1920.0, 1080.0)
    assert window_size(1920, 1080, (1.0, 1.0)) == (1080.0, 1080.0)
    assert window_size(1920, 1080, (16.0, 9.0)) == (1920.0, 1080.0)
    assert window_size(1080, 1920, (16.0, 9.0)) == (1080.0, 607.5)


def test_an_export_range_must_be_non_empty_and_may_follow_the_markers():
    assert not ExportRange().is_custom
    assert ExportRange().resolve(500, (10, 40)) == (10, 40)
    # Either end alone overrides that end and leaves the other following.
    assert ExportRange(start=25).resolve(500, (10, 40)) == (25, 40)
    assert ExportRange(end=60).resolve(500, (10, 40)) == (10, 60)
    # A range from a longer clip is clamped rather than rejected.
    assert ExportRange(start=900, end=950).resolve(100, (0, 100)) == (99, 100)
    with pytest.raises(RenderError, match="end after it starts"):
        ExportRange(start=40, end=40)
    with pytest.raises(RenderError, match="frame 0 or later"):
        ExportRange(start=-3)


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


# -- the target fit --------------------------------------------------------------


def smoothed_path(motion, source, point):
    """The dashed trail: a point's path after the smoothing's own warp."""
    warps = motion.stabilization_transforms()
    return np.asarray(
        [warps[f].apply(np.asarray(source.transform_at(f).apply(point)))
         for f in range(source.frame_count)],
        dtype=np.float64,
    )


def held(motion, clip, paths, marks, **kwargs):
    """Plan a render holding ``paths`` on ``marks``."""
    return StabilizationPlan.build(
        motion,
        source_size=(clip.width, clip.height),
        sources=np.stack(paths),
        placements=np.asarray(marks, dtype=np.float64),
        **kwargs,
    )


def test_a_target_pins_its_point_with_no_smoothing_at_all(clip):
    """Stabilisation without configuring a filter: the fit *is* the warp."""
    motion = motion_for(clip)  # no filters => the smoothing warp is the identity
    plain = StabilizationPlan.build(motion, source_size=(clip.width, clip.height))
    assert plain.is_no_op

    feature = GRID[4]
    plan = held(motion, clip, [truth_path(clip, feature)], [(0.5, 0.5)])
    assert not plan.is_no_op
    landed = np.asarray(
        [plan.output_transform(f).apply(np.asarray(clip.transform_at(f).apply(feature)))
         for f in range(plan.n_frames)]
    )
    assert np.ptp(landed, axis=0).max() < 1e-6


@pytest.mark.parametrize("place", [(0.5, 0.5), (1 / 3, 0.25)])
def test_a_smoothed_target_sticks_to_the_dashed_trail(clip, place):
    """The whole reason a target names a source.

    A ``raw`` mark holds the solid trail on the mark; a ``smoothed`` one holds
    the dashed trail there. Both must be *exact*, on every frame — which they
    can only be because the fit is the entire stabilisation, with no smoothing
    warp composed underneath moving the chosen path somewhere else first.
    """
    motion = motion_for(clip, SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}))
    raw = truth_path(clip, GRID[4])
    dashed = smoothed_path(motion, clip, GRID[4])
    # The two trails have to actually differ, or this proves nothing.
    assert np.abs(raw - dashed).max() > 1.0

    for name, path in (("raw", raw), ("smoothed", dashed)):
        plan = held(motion, clip, [path], [place])
        mark = np.array([place[0] * plan.geometry.width, place[1] * plan.geometry.height])
        landed = np.asarray(
            [plan.output_transform(f).apply(path[f]) for f in range(plan.n_frames)]
        )
        assert np.abs(landed - mark).max() < 1e-6, f"the {name} trail left its mark"


def test_the_fit_is_the_whole_warp_so_a_smoothed_target_is_gentler(clip):
    """Holding the dashed trail leaves the shot moving; holding the solid one does not.

    Both are legitimate asks, and the difference is the reason the control
    exists: a raw mark takes out everything the tracker saw, a smoothed one
    takes out what the smoothed estimate saw and leaves the rest.
    """
    motion = motion_for(clip, SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}))
    feature = GRID[4]
    raw = truth_path(clip, feature)
    dashed = smoothed_path(motion, clip, feature)

    def feature_spread(path):
        plan = held(motion, clip, [path], [(0.5, 0.5)], border="transparent")
        landed = np.asarray(
            [plan.output_transform(f).apply(raw[f]) for f in range(plan.n_frames)]
        )
        return float(np.ptp(landed, axis=0).max())

    assert feature_spread(raw) < 1e-6
    assert feature_spread(dashed) > 1.0


def test_a_target_from_a_different_clip_is_refused(clip):
    motion = motion_for(clip)
    with pytest.raises(RenderError, match="different clip"):
        held(motion, clip, [np.zeros((4, 2))], [(0.5, 0.5)])


def test_with_no_marks_the_plan_is_the_smoothing_warp(clip):
    """The fallback, and what a project renders as before a target is placed."""
    motion = motion_for(clip, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}))
    base = motion.stabilization_transforms()
    plan = StabilizationPlan.build(
        motion, source_size=(clip.width, clip.height), border="transparent"
    )
    for frame, warp in enumerate(plan.transforms):
        assert warp.allclose(base[frame])


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


def test_the_crop_is_pinned_to_the_origin_it_is_given():
    still = [SimilarityTransform.identity()]
    off = plan_output(still, WIDTH, HEIGHT, "crop", origin=(60.0, 30.0), crop=CropSpec(scale=0.5))
    assert off.centre == (60.0, 30.0)
    # With the default placement the origin lands in the middle of the output.
    assert off.fit.apply(np.array([60.0, 30.0])) == pytest.approx([off.width / 2, off.height / 2])


@pytest.mark.parametrize("border", ["crop", "auto_zoom"])
@pytest.mark.parametrize("placement", [(0.5, 0.5), (1 / 3, 1 / 3), (0.2, 0.85), (0.0, 1.0)])
def test_the_origin_lands_exactly_on_its_placement_in_the_output(border, placement):
    """The identity the whole output-space vocabulary rests on (todo 1, eighth round).

    A target's position is stated as a fraction of the *output* frame, and the
    only reason that can be honoured exactly is that the crop/zoom fit is built
    to satisfy ``fit(origin) == placement × output size``. If this drifts, every
    mark the operator places is quietly wrong by the crop.
    """
    shifted = [SimilarityTransform(tx=6.0, ty=-4.0), SimilarityTransform(tx=-6.0, ty=4.0)]
    origin = (WIDTH / 2 + 12.0, HEIGHT / 2 - 5.0)
    geometry = plan_output(shifted, WIDTH, HEIGHT, border, origin=origin, placement=placement)
    landed = geometry.fit.apply(np.array(origin))
    assert landed == pytest.approx(
        [placement[0] * geometry.width, placement[1] * geometry.height], abs=1e-9
    )
    assert geometry.placement == pytest.approx(placement)


def test_coverage_measured_about_an_off_centre_point_is_smaller():
    shifted = [SimilarityTransform(tx=10.0), SimilarityTransform(tx=-10.0)]
    centred = coverage_scale(shifted, WIDTH, HEIGHT)
    off = coverage_scale(shifted, WIDTH, HEIGHT, origin=(WIDTH / 2 + 30.0, HEIGHT / 2))
    assert off < centred


def test_a_placement_off_the_centre_costs_crop():
    """Composing on a thirds line is not free — the rectangle grows one-sided."""
    shifted = [SimilarityTransform(tx=10.0), SimilarityTransform(tx=-10.0)]
    centred = coverage_scale(shifted, WIDTH, HEIGHT)
    thirds = coverage_scale(shifted, WIDTH, HEIGHT, placement=(1 / 3, 0.5))
    assert thirds < centred


def test_the_crop_can_be_measured_over_a_sub_range_only():
    """Todo 1, eighth round: frames that are not being written get no vote on the crop.

    The whole clip's answer is limited by the worst frame in it. An export that
    writes only the calm half should be croppable as if the violent half were
    not there — because, as far as the file is concerned, it is not.
    """
    calm = [SimilarityTransform(tx=2.0), SimilarityTransform(tx=-2.0)]
    violent = [SimilarityTransform(tx=40.0), SimilarityTransform(tx=-40.0)]
    whole = plan_output(calm + violent, WIDTH, HEIGHT, "auto_zoom")
    windowed = plan_output(calm + violent, WIDTH, HEIGHT, "auto_zoom", coverage_range=(0, 2))
    assert windowed.coverage > whole.coverage
    assert windowed.measured_frames == 2
    assert whole.measured_frames == 4


def test_an_empty_or_disjoint_crop_range_is_refused():
    still = [SimilarityTransform.identity(), SimilarityTransform.identity()]
    with pytest.raises(RenderError, match="non-empty"):
        plan_output(still, WIDTH, HEIGHT, "crop", coverage_range=(3, 3))
    with pytest.raises(RenderError, match="does not overlap"):
        plan_output(still, WIDTH, HEIGHT, "crop", coverage_range=(9, 12))


def test_the_crop_is_pinned_to_the_marks_centroid(clip):
    """The output rectangle is grown around where the marks put the picture.

    Under the fit the origin is the marks' own destination, not the feature's
    position — the fit is what put the feature there. So a mark at the centre
    leaves the origin at the frame centre, and moving it moves the rectangle
    with it, which is what makes the placement identity hold at every crop.
    """
    motion = motion_for(clip, SmoothingSettings.uniform("gaussian", {"sigma": 6.0}))
    plain = StabilizationPlan.build(motion, source_size=(clip.width, clip.height))
    assert plain.geometry.centre == (clip.width / 2.0, clip.height / 2.0)

    path = truth_path(clip, GRID[0])
    centred = held(motion, clip, [path], [(0.5, 0.5)], crop=CropSpec(scale=0.6))
    assert centred.geometry.centre == (clip.width / 2.0, clip.height / 2.0)

    off = held(motion, clip, [path], [(0.25, 0.75)], crop=CropSpec(scale=0.6))
    assert off.geometry.centre == pytest.approx((clip.width * 0.25, clip.height * 0.75))
    # ...and the mark still lands exactly on its fraction of the output.
    landed = off.geometry.fit.apply(np.asarray(off.geometry.centre))
    assert landed == pytest.approx(
        [0.25 * off.geometry.width, 0.75 * off.geometry.height], abs=1e-9
    )
