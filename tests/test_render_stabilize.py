"""Planning and running a stabilised render: no gaps, real files, audio intact."""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import WarpedClipSource

from stable.core.errors import RenderError, VideoError
from stable.core.geometry import SimilarityTransform
from stable.core.project import RenderSettings
from stable.core.types import TrackResult
from stable.render import (
    StabilizationPlan,
    iter_stabilized_frames,
    render_stabilized,
    stabilize,
    stabilized_frame,
    write_frame_sequence,
)
from stable.smoothing import SmoothingSettings, smooth_motion
from stable.solver import solve_motion

pytest.importorskip("cv2")

GRID = [(60.0, 50.0), (250.0, 60.0), (240.0, 200.0), (70.0, 190.0)]


def motion_for(clip: WarpedClipSource, settings: SmoothingSettings | None = None):
    """Solve and smooth a warped clip, using its own ground-truth tracks."""
    solve = solve_motion(
        clip.true_tracks(GRID), frame_size=(clip.width, clip.height)
    )
    return smooth_motion(
        solve,
        settings or SmoothingSettings.uniform("gaussian", {"sigma": 8.0}),
        fps=clip.fps,
    )


@pytest.fixture
def plan(flat_clip: WarpedClipSource) -> StabilizationPlan:
    """An auto-zoom plan over the all-white handheld clip."""
    return StabilizationPlan.build(
        motion_for(flat_clip),
        source_size=(flat_clip.width, flat_clip.height),
        border="auto_zoom",
    )


# -- planning ----------------------------------------------------------------


def test_a_plan_covers_every_frame(plan, flat_clip):
    """One output transform per frame, decided before any pixel is read."""
    assert plan.n_frames == flat_clip.frame_count
    assert plan.source_size == (flat_clip.width, flat_clip.height)
    assert plan.border == "auto_zoom"
    assert plan.has_alpha is False


def test_the_output_transform_folds_the_fit_into_the_warp(plan):
    """One affine per frame, so the frame is resampled exactly once."""
    for frame in (0, 10, 29):
        expected = plan.geometry.fit.compose(plan.transforms[frame])
        assert plan.output_transform(frame).allclose(expected, atol=1e-12)


def test_out_of_range_frames_raise(plan):
    """A plan knows how long it is."""
    with pytest.raises(IndexError):
        plan.output_transform(plan.n_frames)


def test_a_plan_can_be_built_from_project_render_settings(flat_clip):
    """The project file's border policy drives the plan."""
    built = StabilizationPlan.from_settings(
        motion_for(flat_clip),
        RenderSettings(border="crop"),
        source_size=(flat_clip.width, flat_clip.height),
    )
    assert built.border == "crop"
    assert built.geometry.width < flat_clip.width


def test_a_plan_summarises_itself(plan):
    """Enough for a UI or a log line, without the per-frame matrices."""
    data = plan.to_dict()
    assert data["frames"] == plan.n_frames
    assert data["geometry"]["border"] == "auto_zoom"
    assert data["interpolation"] == "linear"


def test_no_smoothing_means_a_plan_that_does_nothing(flat_clip):
    """With smoothed == raw the warp is the identity and no zoom is needed."""
    plan = StabilizationPlan.build(
        motion_for(flat_clip, SmoothingSettings()),
        source_size=(flat_clip.width, flat_clip.height),
    )
    assert plan.geometry.zoom == 1.0
    assert all(t.is_identity(atol=1e-9) for t in plan.transforms)


# -- the acceptance property: no border gaps ---------------------------------


def test_auto_zoom_leaves_no_border_gap_in_any_frame(flat_clip, plan):
    """The headline guarantee, checked on every pixel of every frame.

    The clip renders pure white, so a pixel that is not white can only be
    uncovered area. Auto-zoom promises there is none.
    """
    for frame in iter_stabilized_frames(flat_clip, plan):
        assert frame.shape == (flat_clip.height, flat_clip.width, 3)
        assert frame.min() > 0, "auto_zoom left an uncovered pixel"


def test_crop_leaves_no_border_gap_either(flat_clip):
    """Same guarantee, at the cost of resolution rather than magnification."""
    cropped = StabilizationPlan.build(
        motion_for(flat_clip), source_size=(flat_clip.width, flat_clip.height), border="crop"
    )
    assert (cropped.geometry.width, cropped.geometry.height) < (
        flat_clip.width,
        flat_clip.height,
    )
    for frame in iter_stabilized_frames(flat_clip, cropped):
        assert frame.shape[:2] == (cropped.geometry.height, cropped.geometry.width)
        assert frame.min() > 0


def test_the_auto_zoom_is_minimal(flat_clip, plan):
    """Zoom any less and a gap appears — so this really is the smallest that works."""
    assert plan.geometry.zoom > 1.0
    slack = SimilarityTransform.about_pivot(
        (flat_clip.width / 2.0, flat_clip.height / 2.0), scale=plan.geometry.zoom * 0.97
    )
    gaps = 0
    for frame in range(plan.n_frames):
        warped = stabilized_frame(
            flat_clip,
            StabilizationPlan(
                transforms=plan.transforms,
                geometry=type(plan.geometry)(
                    width=flat_clip.width,
                    height=flat_clip.height,
                    fit=slack,
                    coverage=plan.geometry.coverage,
                    zoom=plan.geometry.zoom * 0.97,
                    border="auto_zoom",
                ),
                source_size=plan.source_size,
            ),
            frame,
        )
        gaps += int(warped.min() == 0)
    assert gaps > 0, "a smaller zoom would still have covered every frame"


def test_transparent_shows_exactly_where_the_gaps_are(flat_clip):
    """The policy that keeps the gaps also has to make them visible."""
    transparent = StabilizationPlan.build(
        motion_for(flat_clip),
        source_size=(flat_clip.width, flat_clip.height),
        border="transparent",
    )
    assert transparent.has_alpha
    frames = list(iter_stabilized_frames(flat_clip, transparent))
    assert all(frame.shape[2] == 4 for frame in frames)
    assert any((frame[:, :, 3] == 0).any() for frame in frames)


def test_a_capped_zoom_is_allowed_to_leave_gaps(flat_clip):
    """``max_zoom`` trades coverage for framing, on purpose and on the record."""
    capped = StabilizationPlan.build(
        motion_for(flat_clip),
        source_size=(flat_clip.width, flat_clip.height),
        border="auto_zoom",
        max_zoom=1.001,
    )
    assert capped.geometry.clamped
    assert any(frame.min() == 0 for frame in iter_stabilized_frames(flat_clip, capped))


# -- iterating frames --------------------------------------------------------


def test_frames_can_be_rendered_over_a_sub_range(flat_clip, plan):
    """Rendering part of a clip still uses the whole clip's plan."""
    frames = list(iter_stabilized_frames(flat_clip, plan, frame_range=(5, 9)))
    assert len(frames) == 4
    assert np.array_equal(frames[0], stabilized_frame(flat_clip, plan, 5))


def test_progress_reaches_one(flat_clip, plan):
    """Progress is reported per frame and finishes."""
    seen: list[float] = []
    list(iter_stabilized_frames(flat_clip, plan, progress=lambda f, m="": seen.append(f)))
    assert seen[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("frame_range", "message"),
    [((5, 5), "non-empty"), ((-1, 4), "outside the planned range"), ((0, 999), "outside")],
)
def test_bad_frame_ranges_are_rejected(flat_clip, plan, frame_range, message):
    """An impossible range fails before any decoding happens."""
    with pytest.raises(RenderError, match=message):
        list(iter_stabilized_frames(flat_clip, plan, frame_range=frame_range))


def test_a_plan_longer_than_the_clip_is_rejected(flat_clip):
    """Rendering past the end of the source is caught, not left to the decoder."""
    longer = StabilizationPlan(
        transforms=tuple([SimilarityTransform.identity()] * (flat_clip.frame_count + 5)),
        geometry=StabilizationPlan.build(
            motion_for(flat_clip), source_size=(flat_clip.width, flat_clip.height)
        ).geometry,
        source_size=(flat_clip.width, flat_clip.height),
    )
    with pytest.raises(RenderError, match="runs past the clip"):
        list(iter_stabilized_frames(flat_clip, longer))


# -- writing files -----------------------------------------------------------


def test_rendering_to_a_video_file(tmp_path, warped_clip, tiny_clip):
    """The whole pipeline lands in a playable file of the right length."""
    from stable.video.source import PyAVVideoSource

    plan = StabilizationPlan.build(
        motion_for(warped_clip), source_size=(warped_clip.width, warped_clip.height)
    )
    out = str(tmp_path / "stable.mp4")
    written = render_stabilized(warped_clip, plan, out, fps=warped_clip.fps)

    assert os.path.exists(written)
    with PyAVVideoSource(written) as decoded:
        assert decoded.frame_count == warped_clip.frame_count
        assert (decoded.width, decoded.height) == (warped_clip.width, warped_clip.height)


def test_audio_survives_the_render(tmp_path, tiny_clip):
    """The source clip's audio track is copied into the stabilised output."""
    import av

    from stable.video.source import PyAVVideoSource

    path, source = tiny_clip
    with av.open(path) as container:
        assert container.streams.audio, "the fixture clip must have audio to prove this"

    with PyAVVideoSource(path) as video:
        # One tracked point is enough: this test is about the encode, not the solve.
        points = np.zeros((1, video.frame_count, 2), dtype=np.float32)
        for frame in range(video.frame_count):
            points[0, frame] = source.true_position(frame)
        tracks = TrackResult(
            points=points,
            confidence=np.ones((1, video.frame_count), dtype=np.float32),
            visible=np.ones((1, video.frame_count), dtype=bool),
        )
        motion = smooth_motion(
            solve_motion(tracks, frame_size=(video.width, video.height)),
            SmoothingSettings.uniform("gaussian", {"sigma": 6.0}, channels=["x", "y"]),
            fps=video.fps,
        )
        plan = StabilizationPlan.build(motion, source_size=(video.width, video.height))
        out = str(tmp_path / "with_audio.mp4")
        render_stabilized(video, plan, out, audio_source=path)

    with av.open(out) as container:
        assert container.streams.audio, "audio did not survive the stabilised export"
        assert container.streams.video


def test_rendering_to_a_directory_writes_a_png_sequence(tmp_path, flat_clip, plan):
    """An extension-less destination means an image sequence."""
    out = tmp_path / "frames"
    written = render_stabilized(flat_clip, plan, out, frame_range=(0, 4))
    files = sorted(os.listdir(written))
    assert files == [f"frame_{i:06d}.png" for i in range(4)]


def test_transparent_frames_reach_a_png_sequence_with_their_alpha(tmp_path, flat_clip):
    """PNG is the only output here that can carry the alpha, and it does."""
    import cv2

    transparent = StabilizationPlan.build(
        motion_for(flat_clip),
        source_size=(flat_clip.width, flat_clip.height),
        border="transparent",
    )
    directory = tmp_path / "rgba"
    render_stabilized(flat_clip, transparent, directory)
    images = [
        cv2.imread(str(directory / name), cv2.IMREAD_UNCHANGED)
        for name in sorted(os.listdir(directory))
    ]
    assert all(image.shape[2] == 4 for image in images)
    # Frames the stabiliser barely moved have no gap; the ones it moved most do.
    assert any((image[:, :, 3] == 0).any() for image in images)


def test_transparent_into_a_video_file_refuses_and_explains(tmp_path, flat_clip):
    """Silently dropping the alpha would be the worst possible outcome."""
    transparent = StabilizationPlan.build(
        motion_for(flat_clip),
        source_size=(flat_clip.width, flat_clip.height),
        border="transparent",
    )
    with pytest.raises(RenderError, match="no alpha channel"):
        render_stabilized(flat_clip, transparent, tmp_path / "nope.mp4")


def test_an_empty_frame_sequence_is_an_error(tmp_path):
    """Writing nothing is a bug somewhere upstream, not a successful render."""
    with pytest.raises(RenderError, match="no frames"):
        write_frame_sequence([], tmp_path / "empty")


def test_frame_sequences_reject_a_frame_of_the_wrong_shape(tmp_path):
    """Only RGB and RGBA can be written."""
    with pytest.raises(RenderError, match="RGB or RGBA"):
        write_frame_sequence([np.zeros((4, 4), dtype=np.uint8)], tmp_path / "bad")


def test_an_unknown_codec_is_rejected(tmp_path, flat_clip, plan):
    """The encoder's own validation is reached, not swallowed."""
    with pytest.raises(VideoError, match="unknown codec"):
        render_stabilized(flat_clip, plan, tmp_path / "x.mp4", codec="prores")


# -- the convenience wrapper -------------------------------------------------


def test_stabilize_plans_and_renders_in_one_call(tmp_path, warped_clip):
    """What the CLI uses; it hands back the plan so the zoom can be reported."""
    written, plan = stabilize(
        warped_clip,
        motion_for(warped_clip),
        tmp_path / "one_call.mp4",
        border="crop",
        fps=warped_clip.fps,
    )
    assert os.path.exists(written)
    assert plan.border == "crop"
    assert plan.geometry.zoom == 1.0


def test_stabilize_rejects_an_unknown_border(tmp_path, warped_clip):
    """Caught before any work is done."""
    with pytest.raises(RenderError, match="unknown border policy"):
        stabilize(warped_clip, motion_for(warped_clip), tmp_path / "x.mp4", border="pillar")


def test_a_still_clip_is_returned_essentially_untouched(tmp_path):
    """No motion to remove means no crop, no zoom and no visible change."""
    still = WarpedClipSource([SimilarityTransform.identity()] * 12)
    plan = StabilizationPlan.build(
        motion_for(still, SmoothingSettings.tripod()),
        source_size=(still.width, still.height),
    )
    assert plan.geometry.zoom == 1.0
    for index, frame in enumerate(iter_stabilized_frames(still, plan)):
        assert np.array_equal(frame, still.get_frame(index))
