"""The ``smooth`` correction: filtering one track's own path (todo 1, tenth round).

The Smoothing panel's four channels are the *camera* motion the solver recovers
from every solve-enabled track at once. This is the other thing — one point's
trajectory, smoothed on its own — so these tests are about what that does to one
:class:`~stable.core.types.PointTrack` and, just as much, about what it leaves
alone: the gaps, the confidence, and every frame outside the range.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.corrections import (
    DEFAULT_SMOOTH_BLEND,
    DEFAULT_SMOOTH_FILTER,
    SmoothCorrection,
    apply_corrections,
    correction_from_dict,
    corrections_anchored_at,
    ground_truth_anchors,
)
from stable.core.errors import ContractError, SmoothingError
from stable.core.types import PointTrack

FRAMES = 80


def jittery_track(*, gap: tuple[int, int] | None = None, seed: int = 0) -> PointTrack:
    """A straight move with pixel noise on it, optionally with a hole in it."""
    rng = np.random.default_rng(seed)
    path = np.stack([np.linspace(0.0, 160.0, FRAMES), np.full(FRAMES, 90.0)], axis=1)
    points = (path + rng.normal(0.0, 3.0, size=(FRAMES, 2))).astype(np.float32)
    visible = np.ones(FRAMES, dtype=bool)
    if gap is not None:
        points[gap[0] : gap[1] + 1] = np.nan
        visible[gap[0] : gap[1] + 1] = False
    return PointTrack(
        points=points, confidence=np.full(FRAMES, 0.8, dtype=np.float32), visible=visible
    )


def wobble(track: PointTrack, start: int, end: int) -> float:
    """Frame-to-frame movement of the vertical channel — the thing being removed."""
    column = np.asarray(track.points[start : end + 1, 1], dtype=np.float64)
    return float(np.nanstd(np.diff(column)))


def test_smoothing_a_track_removes_its_frame_to_frame_jitter() -> None:
    track = jittery_track()
    smoothed = apply_corrections(
        track,
        [SmoothCorrection(start=0, end=FRAMES - 1, filter="gaussian", params={"sigma": 6.0})],
        fps=30.0,
    )
    assert wobble(smoothed, 0, FRAMES - 1) < wobble(track, 0, FRAMES - 1) / 4


def test_a_range_leaves_every_frame_outside_it_untouched() -> None:
    """The whole point of a range: smooth the bad section, not the good one."""
    track = jittery_track()
    smoothed = apply_corrections(
        track,
        [SmoothCorrection(start=30, end=50, filter="gaussian", params={"sigma": 5.0}, blend=5)],
        fps=30.0,
    )
    # Well clear of the cross-fade at either edge, the path is bit-identical.
    assert np.array_equal(smoothed.points[:20], track.points[:20])
    assert np.array_equal(smoothed.points[60:], track.points[60:])
    assert wobble(smoothed, 35, 45) < wobble(track, 35, 45)


def test_smoothing_never_invents_a_position_in_a_gap() -> None:
    """A filter needs a gap-free curve; the track must not inherit the filling.

    ``apply_filter`` interpolates non-finite samples away before it runs — it has
    to, since one ``NaN`` poisons a global operation — so the danger is writing
    that interpolation back. Filling a gap is ``gap_fill``'s job, and it stays a
    separate, visible edit.
    """
    track = jittery_track(gap=(20, 26))
    smoothed = apply_corrections(
        track,
        [SmoothCorrection(start=0, end=FRAMES - 1, params={"sigma": 4.0})],
        fps=30.0,
    )
    assert np.isnan(smoothed.points[20:27]).all()
    assert np.isfinite(smoothed.points[27:]).all()


def test_smoothing_leaves_confidence_and_visibility_alone() -> None:
    track = jittery_track(gap=(10, 12))
    smoothed = apply_corrections(track, [SmoothCorrection(start=0, end=FRAMES - 1)], fps=30.0)
    assert np.array_equal(smoothed.confidence, track.confidence)
    assert np.array_equal(smoothed.visible, track.visible)


def test_a_disabled_smooth_changes_nothing() -> None:
    track = jittery_track()
    smoothed = apply_corrections(
        track, [SmoothCorrection(start=0, end=FRAMES - 1, enabled=False)], fps=30.0
    )
    assert np.array_equal(smoothed.points, track.points)


def test_the_frame_rate_reaches_a_filter_that_works_in_hertz() -> None:
    """A Butterworth cutoff is in Hz, so the same correction differs by clip."""
    track = jittery_track()
    correction = SmoothCorrection(
        start=0, end=FRAMES - 1, filter="butterworth", params={"cutoff_hz": 2.0}
    )
    slow = apply_corrections(track, [correction], fps=12.0)
    fast = apply_corrections(track, [correction], fps=60.0)
    assert not np.allclose(slow.points, fast.points)
    # 2 Hz of a 12 fps clip is a far larger share of its bandwidth than 2 Hz of
    # a 60 fps one, so the higher frame rate is cut harder.
    assert wobble(fast, 0, FRAMES - 1) < wobble(slow, 0, FRAMES - 1)


def test_a_cutoff_at_or_above_nyquist_is_refused_rather_than_ignored() -> None:
    track = jittery_track()
    with pytest.raises(SmoothingError):
        apply_corrections(
            track,
            [
                SmoothCorrection(
                    start=0, end=FRAMES - 1, filter="butterworth", params={"cutoff_hz": 30.0}
                )
            ],
            fps=30.0,
        )


def test_an_unknown_parameter_is_refused_by_the_catalogue_that_owns_it() -> None:
    track = jittery_track()
    with pytest.raises(SmoothingError):
        apply_corrections(
            track,
            [SmoothCorrection(start=0, end=10, filter="gaussian", params={"windwo": 5})],
            fps=30.0,
        )


def test_the_range_and_the_blend_are_validated_on_construction() -> None:
    with pytest.raises(ContractError):
        SmoothCorrection(start=-1, end=10)
    with pytest.raises(ContractError):
        SmoothCorrection(start=10, end=4)
    with pytest.raises(ContractError):
        SmoothCorrection(start=0, end=10, blend=-1)


def test_it_round_trips_through_the_project_file_form() -> None:
    correction = SmoothCorrection(
        start=4, end=44, filter="savgol", params={"window": 11, "polyorder": 2}, blend=3
    )
    data = correction.to_dict()
    assert data["kind"] == "smooth"
    assert correction_from_dict(data) == correction
    # `params` is copied out rather than shared, so a stored correction cannot be
    # mutated through the dict that was written for it.
    data["params"]["window"] = 99
    assert correction.params["window"] == 11


def test_the_defaults_are_the_documented_ones() -> None:
    correction = SmoothCorrection(start=0, end=1)
    assert correction.filter == DEFAULT_SMOOTH_FILTER
    assert correction.blend == DEFAULT_SMOOTH_BLEND
    assert correction.params == {}


def test_smoothing_states_no_position_so_it_is_not_an_anchor() -> None:
    """It moves the path; it does not say where the feature *is* on any frame."""
    corrections = [SmoothCorrection(start=0, end=FRAMES - 1)]
    assert ground_truth_anchors(corrections) == []
    assert corrections_anchored_at(corrections, 0) == []
