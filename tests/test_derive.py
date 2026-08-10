"""Deriving one track from several (todo 2, tenth round).

The operation is the per-frame mean: on frame ``f``, the midpoint of the line
joining the sources' positions on frame ``f``. Most of these tests exist because
the first implementation did something else — it read each source at matching
*progress* along its own path — and that time-warps any source covering fewer
frames, so the derived path led and lagged the very features it should have sat
between. :func:`test_the_mean_never_leads_or_lags_a_source` is that bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.derive import (
    DEFAULT_SPAN_MODE,
    SPAN_MODES,
    covered_span,
    mean_track,
)
from stable.core.errors import ContractError
from stable.core.types import PointTrack

CLIP = 100


def path(start: int, end: int, x0: float, x1: float, y: float) -> PointTrack:
    """A clip-length track covering ``[start, end]`` and moving x0 → x1 over it."""
    points = np.full((CLIP, 2), np.nan, dtype=np.float32)
    confidence = np.zeros(CLIP, dtype=np.float32)
    visible = np.zeros(CLIP, dtype=bool)
    span = end - start + 1
    points[start : end + 1, 0] = np.linspace(x0, x1, span)
    points[start : end + 1, 1] = y
    confidence[start : end + 1] = 1.0
    visible[start : end + 1] = True
    return PointTrack(points=points, confidence=confidence, visible=visible)


def test_a_track_is_measured_by_the_frames_it_covers_not_by_the_clip() -> None:
    """Every track's array is clip-length; only the covered span says where it is."""
    assert covered_span(path(10, 39, 0.0, 100.0, 5.0)) == (10, 39)
    assert covered_span(PointTrack.empty(CLIP)) is None


def test_the_mean_is_the_midpoint_of_the_two_sources_on_every_frame() -> None:
    """The operation, stated plainly: halfway along the line joining them."""
    left = path(0, CLIP - 1, 0.0, 200.0, 10.0)
    right = path(0, CLIP - 1, 100.0, 100.0, 50.0)

    derived = mean_track([left, right])
    for frame in (0, 25, 50, 75, CLIP - 1):
        assert derived.points[frame, 0] == pytest.approx(
            (left.points[frame, 0] + right.points[frame, 0]) / 2, abs=1e-3
        )
        assert derived.points[frame, 1] == pytest.approx(30.0, abs=1e-3)


def test_the_mean_never_leads_or_lags_a_source() -> None:
    """The reported bug: a shorter source must not be replayed at another speed.

    Two features moving *together* — same motion, one 40 pixels to the right —
    with one tracked over fewer frames than the other. Their midpoint is by
    construction that same motion, offset 20 pixels, on every frame they share.
    Reading either source at anything other than frame ``f`` for output frame
    ``f`` bends that, which is exactly what leads and lags.
    """
    motion = np.sin(np.linspace(0.0, 4.0, CLIP)) * 30.0 + np.linspace(0.0, 60.0, CLIP)

    def moving(start: int, end: int, offset: float) -> PointTrack:
        points = np.full((CLIP, 2), np.nan, dtype=np.float32)
        confidence = np.zeros(CLIP, dtype=np.float32)
        visible = np.zeros(CLIP, dtype=bool)
        points[start : end + 1, 0] = motion[start : end + 1] + offset
        points[start : end + 1, 1] = 40.0
        confidence[start : end + 1] = 1.0
        visible[start : end + 1] = True
        return PointTrack(points=points, confidence=confidence, visible=visible)

    long_source = moving(0, CLIP - 1, 0.0)
    short_source = moving(0, 59, 40.0)

    derived = mean_track([long_source, short_source], "overlap")
    shared = np.arange(0, 60)
    assert derived.points[shared, 0] == pytest.approx(motion[shared] + 20.0, abs=1e-3)
    # And it stays exactly midway between the two on every one of those frames —
    # never ahead of one and behind the other.
    gap_to_long = derived.points[shared, 0] - long_source.points[shared, 0]
    gap_to_short = short_source.points[shared, 0] - derived.points[shared, 0]
    assert gap_to_long == pytest.approx(gap_to_short, abs=1e-3)


def test_more_than_two_sources_average_to_their_centroid() -> None:
    tracks = [
        path(0, CLIP - 1, 0.0, 0.0, 0.0),
        path(0, CLIP - 1, 30.0, 30.0, 0.0),
        path(0, CLIP - 1, 60.0, 60.0, 90.0),
    ]
    derived = mean_track(tracks)
    assert derived.points[10] == pytest.approx([30.0, 30.0], abs=1e-3)


def test_overlap_keeps_only_the_frames_every_source_covers() -> None:
    """The default: every sample of the result is a true mean of all of them."""
    early = path(0, 59, 0.0, 100.0, 10.0)
    late = path(40, 99, 200.0, 300.0, 20.0)

    derived = mean_track([early, late], "overlap")
    assert covered_span(derived) == (40, 59)
    assert np.isnan(derived.points[:40]).all()
    assert np.isnan(derived.points[60:]).all()
    assert derived.points[50, 0] == pytest.approx(
        (early.points[50, 0] + late.points[50, 0]) / 2, abs=1e-3
    )


def test_union_keeps_every_frame_any_source_covers() -> None:
    """More coverage, and on the frames only one source reaches it *is* that one."""
    early = path(0, 59, 0.0, 100.0, 10.0)
    late = path(40, 99, 200.0, 300.0, 20.0)

    derived = mean_track([early, late], "union")
    assert covered_span(derived) == (0, 99)
    assert derived.points[10] == pytest.approx(early.points[10], abs=1e-3)
    assert derived.points[80] == pytest.approx(late.points[80], abs=1e-3)
    assert derived.points[50, 0] == pytest.approx(
        (early.points[50, 0] + late.points[50, 0]) / 2, abs=1e-3
    )


def test_sources_that_never_share_a_frame_overlap_into_nothing() -> None:
    """Not an error here — a fact about the tracks, for the caller to report."""
    derived = mean_track([path(0, 20, 0.0, 1.0, 1.0), path(50, 70, 0.0, 1.0, 1.0)], "overlap")
    assert covered_span(derived) is None
    assert not derived.valid.any()


def test_a_gap_in_one_source_is_a_gap_in_the_overlap() -> None:
    """No sample is invented: the midpoint of one known point is not a midpoint."""
    whole = path(0, CLIP - 1, 0.0, 100.0, 10.0)
    holed_points = np.array(whole.points, copy=True)
    holed_points[30:36] = np.nan
    holed = whole.replace(points=holed_points)

    derived = mean_track([whole, holed], "overlap")
    assert np.isnan(derived.points[30:36]).all()
    assert np.isfinite(derived.points[36:]).all()

    # Under `union` those frames fall back to the source that still has them.
    joined = mean_track([whole, holed], "union")
    assert joined.points[30:36, 0] == pytest.approx(
        np.asarray(whole.points[30:36, 0], dtype=np.float64), abs=1e-3
    )


def test_averaging_identical_paths_returns_them_unchanged() -> None:
    """The sanity floor: the mean of a thing with itself is the thing."""
    track = path(5, 44, 0.0, 80.0, 12.0)
    derived = mean_track([track, track])
    assert derived.points[5:45] == pytest.approx(
        np.asarray(track.points[5:45], dtype=np.float64), abs=1e-3
    )
    assert covered_span(derived) == (5, 44)


def test_confidence_is_averaged_over_the_sources_present() -> None:
    strong = path(0, CLIP - 1, 0.0, 10.0, 1.0)
    weak = strong.replace(confidence=np.full(CLIP, 0.4, dtype=np.float32))
    derived = mean_track([strong, weak])
    assert derived.confidence[10] == pytest.approx(0.7, abs=1e-6)


def test_the_modes_are_the_documented_ones() -> None:
    assert set(SPAN_MODES) == {"overlap", "union"}
    assert DEFAULT_SPAN_MODE == "overlap"


def test_the_invalid_inputs_are_refused_by_name() -> None:
    track = path(0, 9, 0.0, 10.0, 1.0)
    with pytest.raises(ContractError):
        mean_track([])
    with pytest.raises(ContractError):
        mean_track([track], "halfway")
    with pytest.raises(ContractError):
        # Two tracks from different clips are not two tracks of one thing.
        mean_track([track, PointTrack.empty(CLIP + 5)])
