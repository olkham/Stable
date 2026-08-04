"""``enforce_anchor_agreement``: a run stops at a keyframe it disagrees with.

The tracker-agnostic half of todo 13. The merge has always *ended* on every
anchor exactly; this is the rule that stops a run continuing past one it has
already lost, and that names the frame and the pixel distance so the disagreement
can be reported rather than silently swallowed.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import ContractError
from stable.core.types import PointTrack, Query
from stable.trackers.merge import (
    DEFAULT_ANCHOR_TOLERANCE_PX,
    AnchorViolation,
    enforce_anchor_agreement,
)


def straight_track(n_frames: int = 30, *, slope: float = 1.0) -> PointTrack:
    """A fully-covered, confident run moving one pixel per frame in x."""
    frames = np.arange(n_frames, dtype=np.float64)
    return PointTrack(
        points=np.stack([slope * frames, np.full(n_frames, 5.0)], axis=1),
        confidence=np.full(n_frames, 0.8),
        visible=np.ones(n_frames, dtype=bool),
    )


def test_an_agreeing_anchor_leaves_the_run_untouched() -> None:
    track = straight_track()
    anchors = [Query(frame=0, x=0.0, y=5.0), Query(frame=20, x=21.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=0, direction="fwd")

    assert violations == []
    assert guarded is track
    assert guarded.valid.all()


def test_a_disagreeing_anchor_truncates_the_run_from_that_frame_on() -> None:
    track = straight_track()
    anchors = [Query(frame=0, x=0.0, y=5.0), Query(frame=20, x=200.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=0, direction="fwd")

    assert [v.frame for v in violations] == [20]
    assert violations[0].distance == pytest.approx(180.0)
    assert violations[0].tracked == pytest.approx((20.0, 5.0))
    assert violations[0].anchor == pytest.approx((200.0, 5.0))
    assert violations[0].origin == 0
    assert violations[0].direction == "fwd"

    # Everything up to the disagreement survives; the anchor frame and beyond do not.
    assert guarded.valid[:20].all()
    assert not guarded.valid[20:].any()
    assert guarded.confidence[20:].max() == 0.0
    assert not guarded.visible[20:].any()
    # The input is untouched: raw tracker output stays immutable.
    assert track.valid.all()


def test_a_backward_run_is_truncated_the_other_way() -> None:
    track = straight_track()
    anchors = [Query(frame=8, x=400.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=25, direction="bwd")

    assert [v.frame for v in violations] == [8]
    assert violations[0].direction == "bwd"
    assert not guarded.valid[:9].any()
    assert guarded.valid[9:].all()


def test_a_bidirectional_run_is_guarded_in_both_directions() -> None:
    track = straight_track()
    anchors = [Query(frame=4, x=99.0, y=5.0), Query(frame=26, x=-99.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=15, direction="bidir")

    assert sorted(v.frame for v in violations) == [4, 26]
    assert sorted(v.direction for v in violations) == ["bwd", "fwd"]
    assert not guarded.valid[:5].any()
    assert guarded.valid[5:26].all()
    assert not guarded.valid[26:].any()


def test_the_run_stops_at_the_first_disagreement_not_the_worst() -> None:
    """Frames after the first breach are derived from a position already wrong."""
    track = straight_track()
    anchors = [Query(frame=10, x=30.0, y=5.0), Query(frame=20, x=1000.0, y=5.0)]

    _guarded, violations = enforce_anchor_agreement(track, anchors, origin=0, direction="fwd")

    assert [v.frame for v in violations] == [10]


def test_agreeing_keyframes_are_walked_through_to_reach_a_later_one() -> None:
    track = straight_track()
    anchors = [
        Query(frame=5, x=5.0, y=5.0),
        Query(frame=10, x=10.0, y=5.0),
        Query(frame=15, x=55.0, y=5.0),
    ]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=0, direction="fwd")

    assert [v.frame for v in violations] == [15]
    assert guarded.valid[:15].all()


def test_the_runs_own_anchor_is_never_a_violation() -> None:
    """A run cannot disagree with the position it was started from."""
    track = straight_track()
    anchors = [Query(frame=7, x=900.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=7, direction="bidir")

    assert violations == []
    assert guarded.valid.all()


def test_frames_the_run_never_reached_are_not_disagreements() -> None:
    points = np.asarray(straight_track().points, dtype=np.float64).copy()
    points[12:] = np.nan
    track = PointTrack(
        points=points,
        confidence=np.concatenate([np.full(12, 0.8), np.zeros(18)]),
        visible=np.concatenate([np.ones(12, dtype=bool), np.zeros(18, dtype=bool)]),
    )
    anchors = [Query(frame=20, x=999.0, y=5.0)]

    guarded, violations = enforce_anchor_agreement(track, anchors, origin=0, direction="fwd")

    assert violations == []
    assert guarded.valid[:12].all()


def test_tolerance_is_the_dial_between_noise_and_a_lost_feature() -> None:
    track = straight_track()
    anchors = [Query(frame=20, x=24.0, y=5.0)]  # 4 px out

    _loose, none_found = enforce_anchor_agreement(
        track, anchors, origin=0, direction="fwd", tolerance_px=DEFAULT_ANCHOR_TOLERANCE_PX
    )
    _tight, found = enforce_anchor_agreement(
        track, anchors, origin=0, direction="fwd", tolerance_px=1.0
    )

    assert none_found == []
    assert [v.frame for v in found] == [20]


def test_no_anchors_to_guard_is_a_no_op() -> None:
    track = straight_track()
    guarded, violations = enforce_anchor_agreement(track, [], origin=3, direction="bidir")
    assert guarded is track
    assert violations == []


def test_bad_arguments_are_rejected() -> None:
    track = straight_track()
    anchors = [Query(frame=5, x=0.0, y=0.0)]
    with pytest.raises(ContractError):
        enforce_anchor_agreement(track, anchors, origin=0, direction="fwd", tolerance_px=0.0)
    with pytest.raises(ContractError):
        enforce_anchor_agreement(track, anchors, origin=0, direction="sideways")


def test_a_violation_serialises_for_the_wire() -> None:
    violation = AnchorViolation(
        frame=12,
        anchor=(1.0, 2.0),
        tracked=(3.0, 4.0),
        distance=2.828,
        origin=0,
        direction="fwd",
    )
    assert violation.to_dict() == {
        "frame": 12,
        "anchor": [1.0, 2.0],
        "tracked": [3.0, 4.0],
        "distance": 2.828,
        "origin": 0,
        "direction": "fwd",
    }
