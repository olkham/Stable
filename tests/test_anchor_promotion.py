"""Promotion of manual corrections to keyframes, and bounding a run.

The pure halves of todos 11, 12 and 20:
:func:`stable.core.corrections.ground_truth_anchors` decides *which* edits count
as the operator stating a position, and
:func:`stable.server.tracking.bounded_frame_range` decides how far a run started
from one of them should go.
"""

from __future__ import annotations

import pytest

from stable.core.corrections import (
    DeleteRangeCorrection,
    GapFillCorrection,
    OffsetCorrection,
    ReplaceCorrection,
    RetrackCorrection,
    ground_truth_anchors,
)
from stable.core.errors import ProjectError
from stable.core.project import PointSpec
from stable.core.types import Query
from stable.server.tracking import bounded_frame_range, effective_anchors

# -- which corrections are ground truth ------------------------------------------


def test_absolute_edits_promote_to_anchors() -> None:
    anchors = ground_truth_anchors(
        [
            ReplaceCorrection(frame=12, x=3.0, y=4.0),
            RetrackCorrection(frame=30, x=9.0, y=8.0, direction="fwd"),
        ]
    )
    assert [(q.frame, q.x, q.y) for q in anchors] == [(12, 3.0, 4.0), (30, 9.0, 8.0)]


def test_edits_that_state_no_position_do_not_promote() -> None:
    """A relative nudge, a gap fill and a deletion are not statements of position."""
    assert (
        ground_truth_anchors(
            [
                OffsetCorrection(frame=5, dx=2.0, dy=1.0),
                GapFillCorrection(start=10, end=14),
                DeleteRangeCorrection(start=20, end=22),
            ]
        )
        == []
    )


def test_a_disabled_edit_retires_the_anchor_it_implied() -> None:
    assert ground_truth_anchors([ReplaceCorrection(frame=8, x=1.0, y=1.0, enabled=False)]) == []


def test_the_last_edit_on_a_frame_wins() -> None:
    anchors = ground_truth_anchors(
        [
            ReplaceCorrection(frame=6, x=1.0, y=1.0),
            ReplaceCorrection(frame=6, x=7.0, y=7.0),
        ]
    )
    assert [(q.frame, q.x, q.y) for q in anchors] == [(6, 7.0, 7.0)]


def test_promoted_anchors_come_back_in_frame_order() -> None:
    anchors = ground_truth_anchors(
        [
            ReplaceCorrection(frame=30, x=0.0, y=0.0),
            ReplaceCorrection(frame=2, x=0.0, y=0.0),
            ReplaceCorrection(frame=17, x=0.0, y=0.0),
        ]
    )
    assert [q.frame for q in anchors] == [2, 17, 30]


# -- a point's effective ground truth ---------------------------------------------


def test_effective_anchors_unions_pins_and_promotions() -> None:
    point = PointSpec(
        id="p1",
        queries=[Query(frame=0, x=1.0, y=1.0), Query(frame=40, x=2.0, y=2.0)],
        corrections=[ReplaceCorrection(frame=20, x=5.0, y=6.0)],
    )
    assert [q.frame for q in effective_anchors(point)] == [0, 20, 40]


def test_a_correction_outranks_a_pin_on_the_same_frame() -> None:
    """The correction is the later statement about where the feature is."""
    point = PointSpec(
        id="p1",
        queries=[Query(frame=10, x=1.0, y=1.0)],
        corrections=[ReplaceCorrection(frame=10, x=99.0, y=98.0)],
    )
    assert [(q.frame, q.x, q.y) for q in effective_anchors(point)] == [(10, 99.0, 98.0)]


# -- how far a bounded run goes ----------------------------------------------------


def test_an_unbounded_run_covers_everything_in_its_direction() -> None:
    assert bounded_frame_range("fwd", 10, 100) == (10, 100)
    assert bounded_frame_range("bwd", 10, 100) == (0, 11)
    assert bounded_frame_range("bidir", 10, 100) == (0, 100)


def test_a_run_can_stop_at_the_neighbouring_keyframe() -> None:
    anchors = [Query(frame=5, x=0.0, y=0.0), Query(frame=40, x=0.0, y=0.0)]
    assert bounded_frame_range("fwd", 10, 100, anchors=anchors, stop_at_keyframe=True) == (10, 41)
    assert bounded_frame_range("bwd", 10, 100, anchors=anchors, stop_at_keyframe=True) == (5, 11)
    assert bounded_frame_range("bidir", 10, 100, anchors=anchors, stop_at_keyframe=True) == (5, 41)


def test_stopping_at_a_keyframe_uses_the_nearest_one() -> None:
    anchors = [Query(frame=f, x=0.0, y=0.0) for f in (2, 6, 30, 60)]
    assert bounded_frame_range("bidir", 10, 100, anchors=anchors, stop_at_keyframe=True) == (6, 31)


def test_stopping_at_a_keyframe_with_none_that_way_is_a_no_op() -> None:
    anchors = [Query(frame=10, x=0.0, y=0.0)]
    assert bounded_frame_range("bidir", 10, 100, anchors=anchors, stop_at_keyframe=True) == (0, 100)


def test_a_run_can_be_limited_to_a_frame_count() -> None:
    assert bounded_frame_range("fwd", 10, 100, max_frames=5) == (10, 16)
    assert bounded_frame_range("bwd", 10, 100, max_frames=5) == (5, 11)
    assert bounded_frame_range("bidir", 10, 100, max_frames=5) == (5, 16)


def test_the_tighter_of_the_two_bounds_wins() -> None:
    anchors = [Query(frame=13, x=0.0, y=0.0)]
    assert bounded_frame_range(
        "fwd", 10, 100, anchors=anchors, max_frames=20, stop_at_keyframe=True
    ) == (10, 14)
    assert bounded_frame_range(
        "fwd", 10, 100, anchors=anchors, max_frames=1, stop_at_keyframe=True
    ) == (10, 12)


def test_bounds_never_leave_the_clip() -> None:
    assert bounded_frame_range("bidir", 2, 20, max_frames=50) == (0, 20)
    assert bounded_frame_range("fwd", 19, 20, max_frames=50) == (19, 20)


def test_a_run_always_contains_its_own_start_frame() -> None:
    for direction in ("fwd", "bwd", "bidir"):
        start, end = bounded_frame_range(direction, 7, 30, max_frames=1)
        assert start <= 7 < end


def test_a_non_positive_frame_limit_is_rejected() -> None:
    with pytest.raises(ProjectError):
        bounded_frame_range("fwd", 5, 30, max_frames=0)
