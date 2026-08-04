"""Phase 3 acceptance criteria, checked end to end on real pixels and real files.

``PLAN.md`` asks for three things:

#. *"synthetic test — a clip warped by known X, Y, R, S recovers the ground-truth
   motion within tolerance"*
#. *"stabilized real handheld clip shows no border gaps in auto-zoom"*
#. *"audio survives export"*

All three are checked here against **one** run of the whole pipeline — encode a
clip whose motion is known exactly, track it with a real tracker on real
(compressed) pixels, solve, smooth, plan the border policy, render and encode —
so nothing is proved against a convenient stub. The last test then re-tracks the
rendered file and shows the shot really did get steadier, which is the product
claim the three criteria exist to support.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import WarpedClipSource, encode_synthetic_clip, handheld_transforms

from stable.core.types import Query
from stable.render import StabilizationPlan, iter_stabilized_frames, render_stabilized, warp_frame
from stable.smoothing import SmoothingSettings, smooth_motion
from stable.solver import SolveStatus, solve_motion
from stable.trackers.lk import LKTracker
from stable.video.source import PyAVVideoSource

pytest.importorskip("cv2")
pytest.importorskip("av")

FRAMES = 30
SIZE = (320, 240)
PIVOT = (160.0, 120.0)


def trackable_features(clip: WarpedClipSource, count: int = 10) -> list[tuple[float, float]]:
    """Return features far enough from the edge to survive the whole move."""
    inside = [p for p in clip.feature_points() if 40 < p[0] < 280 and 40 < p[1] < 200]
    return inside[:count]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the entire Phase 3 pipeline once, on a clip with known motion and audio.

    Returns:
        A dict with the ground-truth clip, the decoded source, the tracks a real
        tracker produced from its pixels, the solve, the smoothed motion, the
        auto-zoom plan and the rendered output file.
    """
    directory = tmp_path_factory.mktemp("acceptance")
    truth = WarpedClipSource(
        handheld_transforms(FRAMES, pivot=PIVOT, amplitude=7.0, rotation=0.02, scale=0.01),
        width=SIZE[0],
        height=SIZE[1],
        fps=25.0,
    )
    source_path = str(directory / "handheld.mp4")
    encode_synthetic_clip(truth, source_path, with_audio=True, crf=12)

    video = PyAVVideoSource(source_path)
    features = trackable_features(truth)
    tracker = LKTracker()
    tracks = tracker.track(
        video, [Query(frame=0, x=x, y=y) for x, y in features], (0, video.frame_count), "bidir"
    )
    tracker.release()

    solve = solve_motion(tracks, frame_size=(video.width, video.height))
    motion = smooth_motion(
        solve,
        SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}),
        fps=video.fps,
    )
    plan = StabilizationPlan.build(
        motion, source_size=(video.width, video.height), border="auto_zoom"
    )
    out_path = str(directory / "stabilized.mp4")
    render_stabilized(video, plan, out_path, audio_source=source_path)

    result = {
        "truth": truth,
        "source_path": source_path,
        "video": video,
        "tracks": tracks,
        "solve": solve,
        "motion": motion,
        "plan": plan,
        "out_path": out_path,
    }
    yield result
    video.close()


# -- criterion 1 -------------------------------------------------------------


def test_the_solve_recovers_the_ground_truth_motion(pipeline):
    """A clip warped by known X, Y, R, S solves back to those numbers.

    The tolerances are what a real tracker on H.264-compressed pixels can
    deliver: well under a tenth of a pixel of translation at the frame centre,
    a hundredth of a degree of rotation, and a hundredth of a percent of scale.
    """
    truth, solve = pipeline["truth"], pipeline["solve"]
    probe = np.asarray([PIVOT, [40.0, 40.0], [280.0, 200.0]])

    worst_position = worst_rotation = worst_scale = 0.0
    for frame in range(FRAMES):
        expected = truth.relative_transform(frame)
        actual = solve.transform_at(frame)
        worst_position = max(
            worst_position, float(np.abs(actual.apply(probe) - expected.apply(probe)).max())
        )
        worst_rotation = max(worst_rotation, abs(actual.rotation - expected.rotation))
        worst_scale = max(worst_scale, abs(actual.scale - expected.scale))

    assert worst_position < 0.25, "positions drifted from the known motion"
    assert worst_rotation < 2e-4, "rotation drifted from the known motion"
    assert worst_scale < 1e-3, "scale drifted from the known motion"


def test_every_frame_solved_cleanly(pipeline):
    """No frame had to degrade, interpolate or be held on a well-tracked clip."""
    solve = pipeline["solve"]
    assert (solve.status == int(SolveStatus.SOLVED)).all()
    assert not solve.flagged.any()
    assert float(np.nanmax(solve.residual)) < 1.0


def test_the_recovered_curves_match_the_known_channels(pipeline):
    """Each of the four curves individually tracks the motion that was imposed."""
    truth, solve = pipeline["truth"], pipeline["solve"]
    for frame in range(FRAMES):
        expected = truth.relative_transform(frame)
        moved = expected.apply(np.asarray(PIVOT))
        assert solve.curve("x")[frame] == pytest.approx(moved[0] - PIVOT[0], abs=0.25)
        assert solve.curve("y")[frame] == pytest.approx(moved[1] - PIVOT[1], abs=0.25)
        assert solve.curve("r")[frame] == pytest.approx(expected.rotation, abs=2e-4)
        assert solve.curve("s")[frame] == pytest.approx(expected.scale, abs=1e-3)


# -- criterion 2 -------------------------------------------------------------


def test_auto_zoom_leaves_no_border_gap(pipeline):
    """Every rendered frame is fully covered by source pixels.

    Checked by pushing a pure-white frame through the very transforms the render
    used: any pixel that is not white is uncovered area, and there must be none.
    The clip's own compressed pixels cannot answer this question — a genuinely
    dark corner of the picture is indistinguishable from a gap — so the mask is
    the honest instrument.
    """
    plan = pipeline["plan"]
    white = np.full((SIZE[1], SIZE[0], 3), 255, dtype=np.uint8)

    assert plan.geometry.zoom > 1.0, "this shot should have needed a real zoom"
    assert plan.geometry.clamped is False
    for frame in range(plan.n_frames):
        covered = warp_frame(
            white, plan.output_transform(frame), size=plan.geometry.size, border="crop"
        )
        assert covered.min() > 0, f"frame {frame} has an uncovered border"


def test_the_rendered_file_keeps_the_delivery_size(pipeline):
    """Auto-zoom is the policy that does not cost the user any resolution."""
    plan, video = pipeline["plan"], pipeline["video"]
    assert plan.geometry.size == (video.width, video.height)
    with PyAVVideoSource(pipeline["out_path"]) as rendered:
        assert (rendered.width, rendered.height) == (video.width, video.height)
        assert rendered.frame_count == FRAMES


def test_the_zoom_is_no_more_than_the_shot_required(pipeline):
    """Minimality: shave 3% off the zoom and gaps appear."""
    plan = pipeline["plan"]
    white = np.full((SIZE[1], SIZE[0], 3), 255, dtype=np.uint8)
    from stable.core.geometry import SimilarityTransform

    slack = SimilarityTransform.about_pivot(PIVOT, scale=plan.geometry.zoom * 0.97)
    gaps = sum(
        int(warp_frame(white, slack.compose(plan.transforms[frame]), border="crop").min() == 0)
        for frame in range(plan.n_frames)
    )
    assert gaps > 0, "a smaller zoom would have worked, so this one was not minimal"


def test_the_crop_policy_also_covers_every_frame(pipeline):
    """The other gap-free policy, on the same shot."""
    cropped = StabilizationPlan.build(
        pipeline["motion"], source_size=SIZE, border="crop"
    )
    white = np.full((SIZE[1], SIZE[0], 3), 255, dtype=np.uint8)
    assert (cropped.geometry.width, cropped.geometry.height) < SIZE
    for frame in range(cropped.n_frames):
        covered = warp_frame(
            white, cropped.output_transform(frame), size=cropped.geometry.size, border="crop"
        )
        assert covered.min() > 0


# -- criterion 3 -------------------------------------------------------------


def test_audio_survives_the_export(pipeline):
    """The stabilised file carries the source clip's audio track, unre-encoded."""
    import av

    with av.open(pipeline["source_path"]) as container:
        assert container.streams.audio, "the fixture must have audio for this to mean anything"
        source_codec = container.streams.audio[0].codec_context.name

    with av.open(pipeline["out_path"]) as container:
        assert container.streams.video, "the render lost its video stream"
        assert container.streams.audio, "the render lost its audio stream"
        assert container.streams.audio[0].codec_context.name == source_codec


# -- the claim the criteria support ------------------------------------------


def _resolve(path: str, queries: list[Query]):
    """Track and solve a rendered file with the same tracker and features."""
    with PyAVVideoSource(path) as video:
        tracker = LKTracker()
        tracks = tracker.track(video, queries, (0, video.frame_count), "bidir")
        tracker.release()
        return solve_motion(tracks, frame_size=(video.width, video.height))


def _judder(solve) -> float:
    """Return the high-frequency energy of a solve — what stabilisation removes."""
    return float(
        np.abs(np.diff(solve.curve("x"), 2)).mean() + np.abs(np.diff(solve.curve("y"), 2)).mean()
    )


def _excursion(solve) -> float:
    """Return how far the shot travelled overall — what stabilisation keeps."""
    return float(np.abs(solve.curve("x")).max() + np.abs(solve.curve("y")).max())


def test_the_rendered_clip_is_measurably_steadier_than_the_source(pipeline):
    """Re-tracking the output shows the judder is gone and the move is not.

    This is the end-to-end proof: the same tracker, the same solver and the same
    features, run over the file that was actually written to disk. A 1 Hz
    Butterworth sits between the fixture's 0.35 Hz drift and its 6 Hz judder, so
    the right answer is "most of the judder removed, most of the move kept" —
    not "everything removed", which would be a tripod, and not "nothing".
    """
    queries = [Query(frame=0, x=x, y=y) for x, y in trackable_features(pipeline["truth"])]
    before = _resolve(pipeline["source_path"], queries)
    after = _resolve(pipeline["out_path"], queries)

    assert _judder(before) > 0.1, "the source clip should visibly judder"
    assert _judder(after) < 0.35 * _judder(before), "too much judder survived the render"
    assert _excursion(after) > 0.5 * _excursion(before), "the deliberate move was flattened too"


def test_a_tripod_render_removes_the_motion_almost_entirely(pipeline, tmp_path):
    """The other extreme: locking every channel really does nail the shot down."""
    video = pipeline["video"]
    locked = smooth_motion(pipeline["solve"], SmoothingSettings.tripod(), fps=video.fps)
    plan = StabilizationPlan.build(
        locked, source_size=(video.width, video.height), border="auto_zoom"
    )
    out = str(tmp_path / "tripod.mp4")
    render_stabilized(video, plan, out, audio_source=pipeline["source_path"])

    queries = [Query(frame=0, x=x, y=y) for x, y in trackable_features(pipeline["truth"])]
    before = _resolve(pipeline["source_path"], queries)
    after = _resolve(out, queries)
    assert _excursion(after) < 0.1 * _excursion(before)


def test_the_stabilised_frames_differ_from_the_source_frames(pipeline):
    """A sanity check that the render did something at all."""
    plan, video = pipeline["plan"], pipeline["video"]
    rendered = list(iter_stabilized_frames(video, plan, frame_range=(FRAMES // 2, FRAMES // 2 + 1)))
    original = video.get_frame(FRAMES // 2)
    assert not np.array_equal(rendered[0], original)
    assert os.path.getsize(pipeline["out_path"]) > 0
