"""The one integration suite every tracker plugin must pass, classic or SOTA.

Phase 2's headline acceptance criterion is *"both plugins pass the same
integration suite as lk"*. Rather than paraphrasing ``lk``'s tests for each new
model, this module parametrises one suite over every shipped plugin — so the
swappability claim in ``PLAN.md`` is enforced by construction: anything that
fails here is not a drop-in tracker.

CoTracker3 and TAPIR run against :class:`~conftest.MarkerBackend`, a fake
backend that finds the synthetic clip's marker in the pixels it is handed. That
keeps the suite green on a machine with no GPU, no torch and no checkpoints
while still exercising everything the *plugin* owns — window planning, frame
loading, coordinate scaling, re-anchoring, stitching, confidence normalisation
and timeline assembly. Only the neural network itself is stubbed. Ocean and
SiamDW do the same through :class:`~conftest.SiameseMarkerBackend`, which finds
the marker in the template and search crops they cut.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import MarkerBackend, SiameseMarkerBackend, anchor_at

from stable.core.errors import TrackerError
from stable.core.types import Query, TrackResult
from stable.trackers.correlation import CorrelationTracker
from stable.trackers.cotracker3 import CoTracker3Tracker
from stable.trackers.features import OrbTracker, SiftTracker
from stable.trackers.kalman import KalmanTracker
from stable.trackers.klt import KLTTracker
from stable.trackers.lk import LKTracker
from stable.trackers.meanshift import MeanShiftTracker
from stable.trackers.ocean import OceanTracker
from stable.trackers.particle import ParticleFilterTracker
from stable.trackers.raft import RaftTracker
from stable.trackers.registry import available_trackers, describe_trackers, get_tracker
from stable.trackers.sam3 import Sam3Tracker
from stable.trackers.siamdw import SiamDWTracker
from stable.trackers.tapir import TapirTracker
from stable.trackers.template import TemplateTracker

# Small windows with a real overlap so the multi-chunk path is the *default* in
# every test below, not a special case one test remembers to cover.
DEEP_KWARGS = {"chunk_size": 6, "overlap": 2, "model_resolution": None}


def make_plugin(name: str):
    """Build a ready-to-track plugin instance by tracker name."""
    if name == "lk":
        return LKTracker()
    if name == "klt":
        return KLTTracker()
    if name == "template":
        return TemplateTracker()
    if name == "correlation":
        return CorrelationTracker()
    if name == "kalman":
        return KalmanTracker()
    if name == "particle":
        return ParticleFilterTracker()
    if name == "meanshift":
        return MeanShiftTracker()
    if name == "sift":
        return SiftTracker()
    if name == "orb":
        return OrbTracker()
    if name == "cotracker3":
        return CoTracker3Tracker(backend=MarkerBackend(), **DEEP_KWARGS)
    if name == "tapir":
        return TapirTracker(backend=MarkerBackend(), **DEEP_KWARGS)
    if name == "raft":
        return RaftTracker(backend=MarkerBackend(), **DEEP_KWARGS)
    if name == "sam3":
        return Sam3Tracker(backend=MarkerBackend(), **DEEP_KWARGS)
    if name == "ocean":
        return OceanTracker(backend=SiameseMarkerBackend("ocean"))
    if name == "siamdw":
        return SiamDWTracker(backend=SiameseMarkerBackend("siamdw"))
    raise AssertionError(f"unknown tracker {name!r}")


TRACKERS = [
    "lk",
    "klt",
    "template",
    "correlation",
    "kalman",
    "particle",
    "meanshift",
    "cotracker3",
    "tapir",
    "raft",
    "sam3",
    "ocean",
    "siamdw",
]
# The windowed plugins; Ocean and SiamDW never cut a run into windows (see
# stable/trackers/siamese.py), so the chunking criterion below is not theirs.
DEEP = ["cotracker3", "tapir", "raft", "sam3"]

# The feature matchers are held to the same *registry* contract here, but
# their behavioural contract lives in tests/test_feature_trackers.py: this
# module's clip is a smooth gradient plus one bright square, and SIFT's DoG
# finds a single keypoint in the whole frame. Asserting tracking accuracy on
# material a tracker cannot see would test the fixture, not the plugin.
REGISTRY_TRACKERS = [*TRACKERS, "sift", "orb"]


@pytest.fixture(params=TRACKERS)
def plugin(request):
    """Every shipped tracker, one at a time."""
    return make_plugin(request.param)


# -- registry contract ----------------------------------------------------


@pytest.mark.parametrize("name", REGISTRY_TRACKERS)
def test_every_tracker_is_discoverable_and_described(name):
    plugin_class = get_tracker(name)
    assert name in available_trackers()
    row = {r["name"]: r for r in describe_trackers()}[name]
    assert row["display_name"] and row["license"]
    assert isinstance(row["available"], bool)
    assert plugin_class.name == name
    # Deep trackers prefer a GPU but must never *require* one: PLAN.md's
    # acceptance criterion is that CPU fallback completes, slow but correct.
    assert plugin_class.requires_gpu is False


@pytest.mark.parametrize("name", REGISTRY_TRACKERS)
def test_lifecycle_is_uniform(name):
    plugin = make_plugin(name)
    assert not plugin.is_prepared
    plugin.prepare("cpu")
    assert plugin.is_prepared and plugin.device == "cpu"
    plugin.release()
    assert not plugin.is_prepared
    assert name in repr(plugin)


# -- the shared behavioural contract --------------------------------------


def test_result_spans_the_whole_clip_timeline(plugin, video):
    result = plugin.track(video, [anchor_at(video, 4)], (2, 8))

    assert isinstance(result, TrackResult)
    assert result.n_points == 1
    assert result.n_frames == video.frame_count
    assert np.isnan(result.points[0, :2]).all()
    assert np.isnan(result.points[0, 8:]).all()
    assert np.isfinite(result.points[0, 2:8]).all()


def test_points_come_back_in_query_order(plugin, video):
    # Kept clear of the frame edge: template's default patch needs a 13px margin.
    queries = [anchor_at(video, 0), Query(0, 20.0, 20.0), Query(0, 30.0, 15.0)]
    result = plugin.track(video, queries, (0, video.frame_count))

    assert result.n_points == 3
    for index, query in enumerate(queries):
        assert result.points[index, 0] == pytest.approx(query.position, abs=1.5)


def test_tracks_the_synthetic_marker_accurately(plugin, video):
    result = plugin.track(video, [anchor_at(video, 6)], (0, video.frame_count), "bidir")

    for frame in range(video.frame_count):
        px, py = result.points[0, frame]
        assert np.isfinite(px)
        tx, ty = video.true_position(frame)
        assert (px - tx) ** 2 + (py - ty) ** 2 < 1.5**2, f"frame {frame}"


def test_directions_cover_the_expected_frames(plugin, video):
    query = anchor_at(video, 10)

    fwd = plugin.track(video, [query], (0, video.frame_count), "fwd")
    assert np.isnan(fwd.points[0, :10]).all()
    assert np.isfinite(fwd.points[0, 10:]).all()

    bwd = plugin.track(video, [query], (0, video.frame_count), "bwd")
    assert np.isfinite(bwd.points[0, :11]).all()
    assert np.isnan(bwd.points[0, 11:]).all()

    bidir = plugin.track(video, [query], (0, video.frame_count), "bidir")
    assert np.isfinite(bidir.points[0]).all()


def test_the_anchor_frame_holds_the_users_position_exactly(plugin, video):
    query = Query(frame=7, x=31.25, y=22.75)
    result = plugin.track(video, [query], (0, video.frame_count))

    assert result.points[0, 7] == pytest.approx((31.25, 22.75), abs=1e-4)
    assert result.confidence[0, 7] == pytest.approx(1.0)
    assert bool(result.visible[0, 7])


def test_confidence_obeys_the_shared_scale(plugin, video):
    """Normalised to [0, 1] and high on a clean track — for every plugin."""
    result = plugin.track(video, [anchor_at(video, 6)], (0, video.frame_count))
    confidence = result.confidence[0]

    assert np.isfinite(confidence).all()
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0
    assert confidence.min() > 0.9, "a perfectly tracked marker must score near 1"
    assert result.visible[0].all()


def test_confidence_collapses_when_the_marker_is_occluded(plugin, occluded_video):
    anchor_frame = 5
    query = Query(anchor_frame, *occluded_video.true_position(anchor_frame))
    result = plugin.track(occluded_video, [query], (0, occluded_video.frame_count), "fwd")

    lo, hi = occluded_video.occlude_range
    clean = result.confidence[0, anchor_frame:lo]
    occluded = result.confidence[0, lo:hi]
    assert clean.min() > 0.9
    assert occluded.max() < clean.min()


def test_empty_query_list_yields_an_empty_result(plugin, video):
    result = plugin.track(video, [], (0, 5))
    assert result.n_points == 0
    assert result.n_frames == video.frame_count


def test_query_must_lie_inside_frame_range(plugin, video):
    with pytest.raises(TrackerError, match="frame_range"):
        plugin.track(video, [Query(2, 20.0, 20.0)], (5, 10))


def test_invalid_arguments_raise_tracker_errors(plugin, video):
    query = Query(0, 20.0, 20.0)
    with pytest.raises(TrackerError, match="non-empty"):
        plugin.track(video, [query], (5, 5))
    with pytest.raises(TrackerError, match="outside the clip"):
        plugin.track(video, [query], (0, 999))
    with pytest.raises(TrackerError, match="unknown direction"):
        plugin.track(video, [query], (0, 5), "sideways")


def test_progress_is_reported_and_ends_at_one(plugin, video):
    seen: list[float] = []
    plugin.track(
        video,
        [anchor_at(video, 0), anchor_at(video, 1)],
        (0, video.frame_count),
        progress=lambda fraction, message="": seen.append(fraction),
    )
    assert seen
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_progress_moves_while_a_single_query_run_is_in_flight(plugin, video):
    """todo 4 (second round): "running 0%" for the whole of a long track.

    The test above ran two queries, so a plugin reporting once *per query* still
    produced a rising sequence and passed. Server-side tracking gives every run
    exactly one query
    (:func:`stable.server.tracking.run_point_tracking` calls
    ``plugin.track(video, [spec.anchor], ...)``), which turned that same code
    into a single report, fired when the run was already over — a progress bar
    that sat at zero while the frame counter beside it climbed.

    So the number that matters is how often progress is heard from *before* the
    final ``1.0``, on one query. That is what this asserts.
    """
    seen: list[float] = []
    plugin.track(
        video,
        [anchor_at(video, 0)],
        (0, video.frame_count),
        progress=lambda fraction, message="": seen.append(fraction),
    )

    during = seen[:-1]
    assert len(during) > 1, "a run must report progress while it is running, not only at the end"
    assert during == sorted(during)
    assert during[-1] > 0.0
    assert seen[-1] == pytest.approx(1.0)


def test_output_is_deterministic(plugin, video):
    query = anchor_at(video, 3)
    first = plugin.track(video, [query], (0, video.frame_count))
    second = make_plugin(type(plugin).name).track(video, [query], (0, video.frame_count))
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.confidence, second.confidence)


def test_results_are_immutable(plugin, video):
    result = plugin.track(video, [anchor_at(video, 2)], (0, video.frame_count))
    with pytest.raises(ValueError):
        result.points[0, 0, 0] = 1.0


# -- the deep plugins must survive a long clip in small windows ------------


@pytest.mark.parametrize("name", DEEP)
def test_a_long_clip_is_tracked_in_many_windows_without_seams(name, tiny_clip_factory):
    """The chunking acceptance criterion, in miniature.

    A 120-frame clip through 6-frame windows is 20+ inference calls per run —
    the same code path a 30 s 1080p clip takes through 48-frame windows, minus
    the GPU. What matters is that the seams do not show: the stitched path stays
    on the marker and its per-frame velocity never jumps.
    """
    from stable.video.source import PyAVVideoSource

    path, truth = tiny_clip_factory(frame_count=120, period=60.0, radius=12.0)
    plugin_class = type(make_plugin(name))
    backend = MarkerBackend()
    chunked = plugin_class(backend=backend, chunk_size=6, overlap=2, model_resolution=None)
    whole = plugin_class(backend=MarkerBackend(), chunk_size=200, overlap=0, model_resolution=None)

    with PyAVVideoSource(path) as video:
        anchor = Query(0, *truth.true_position(0))
        frame_range = (0, video.frame_count)
        result = chunked.track(video, [anchor], frame_range, "fwd")
        reference = whole.track(video, [anchor], frame_range, "fwd")

    assert backend.calls > 20, "the run must actually be chunked"
    assert max(window[0] for window in backend.windows) <= 6

    points = result.points[0]
    assert np.isfinite(points).all()
    errors = np.hypot(
        points[:, 0] - [truth.true_position(f)[0] for f in range(120)],
        points[:, 1] - [truth.true_position(f)[1] for f in range(120)],
    )
    assert errors.max() < 2.0, "stitched track drifted off the marker"

    # The point of overlap stitching: cutting a run into 30 windows must give
    # back what one window would have, seams and re-anchoring included.
    assert np.abs(points - reference.points[0]).max() < 0.5
    assert result.confidence[0].min() > 0.9
