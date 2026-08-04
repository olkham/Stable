"""Tests for windowed inference planning and overlap stitching."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import ContractError
from stable.trackers.chunking import Chunk, ChunkStitcher, chunk_weights, plan_chunks

# -- planning ------------------------------------------------------------


def test_a_short_run_needs_one_window():
    chunks = plan_chunks(10, 48, 8)
    assert [(c.start, c.end) for c in chunks] == [(0, 10)]
    assert chunks[0].blend_in == 0 and chunks[0].blend_out == 0


def test_windows_overlap_by_the_requested_amount():
    chunks = plan_chunks(100, 40, 10)
    assert [(c.start, c.end) for c in chunks] == [(0, 40), (30, 70), (60, 100)]
    assert [c.blend_in for c in chunks] == [0, 10, 10]
    assert [c.blend_out for c in chunks] == [10, 10, 0]


def test_windows_cover_every_frame_exactly_once_or_more():
    for length in (1, 2, 7, 33, 64, 65, 200):
        for chunk_size, overlap in ((8, 0), (8, 3), (16, 8), (48, 8), (5, 4)):
            chunks = plan_chunks(length, chunk_size, overlap)
            covered = np.zeros(length, dtype=int)
            for chunk in chunks:
                covered[chunk.start : chunk.end] += 1
            assert covered.min() >= 1, (length, chunk_size, overlap)
            assert chunks[0].start == 0
            assert chunks[-1].end == length
            assert all(c.n_frames <= chunk_size for c in chunks)


def test_planning_always_makes_progress_even_with_maximal_overlap():
    chunks = plan_chunks(50, 4, 99)  # overlap is clamped to chunk_size - 1
    assert len(chunks) == 47
    assert all(b.start > a.start for a, b in zip(chunks, chunks[1:], strict=False))


def test_zero_length_run_plans_nothing():
    assert plan_chunks(0, 16, 4) == []


def test_planning_validates_its_arguments():
    with pytest.raises(ContractError, match="length"):
        plan_chunks(-1, 16, 4)
    with pytest.raises(ContractError, match="chunk_size"):
        plan_chunks(10, 1, 0)
    with pytest.raises(ContractError, match="overlap"):
        plan_chunks(10, 16, -1)


def test_chunk_validation():
    with pytest.raises(ContractError, match="non-empty"):
        Chunk(5, 5)
    with pytest.raises(ContractError, match="start"):
        Chunk(-1, 4)
    with pytest.raises(ContractError, match=">= 0"):
        Chunk(0, 4, blend_in=-1)
    with pytest.raises(ContractError, match="exceed"):
        Chunk(0, 4, blend_out=9)
    assert Chunk(2, 6).indices().tolist() == [2, 3, 4, 5]


# -- weighting -----------------------------------------------------------


def test_a_lone_window_has_uniform_weight():
    assert chunk_weights(Chunk(0, 5)).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_ramps_rise_and_fall_across_the_overlaps():
    weights = chunk_weights(Chunk(0, 6, blend_in=2, blend_out=2))
    assert weights[0] < weights[1] < weights[2]
    assert weights[3] > weights[4] > weights[5]
    assert weights.max() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_heavily_overlapped_short_window_degrades_to_a_tent():
    weights = chunk_weights(Chunk(0, 6, blend_in=4, blend_out=4))
    assert (weights > 0).all() and weights.max() <= 1.0
    assert weights[:3].tolist() == sorted(weights[:3])
    assert weights[3:].tolist() == sorted(weights[3:], reverse=True)


# -- stitching -----------------------------------------------------------


def _constant(chunk: Chunk, value: float, n_points: int = 1, confidence: float = 1.0):
    points = np.full((chunk.n_frames, n_points, 2), value, dtype=np.float64)
    conf = np.full((chunk.n_frames, n_points), confidence)
    visible = np.ones((chunk.n_frames, n_points), dtype=bool)
    return points, conf, visible


def test_a_single_window_passes_straight_through():
    chunk = Chunk(0, 4)
    stitcher = ChunkStitcher(4, 1)
    stitcher.add(chunk, *_constant(chunk, 3.0))
    points, confidence, visible = stitcher.result()
    assert points[:, 0, 0] == pytest.approx([3.0] * 4)
    assert confidence[:, 0] == pytest.approx([1.0] * 4)
    assert visible.all()


def test_overlap_crossfades_monotonically_between_windows():
    plan = plan_chunks(16, 10, 4)
    stitcher = ChunkStitcher(16, 1)
    for index, chunk in enumerate(plan):
        stitcher.add(chunk, *_constant(chunk, 0.0 if index == 0 else 10.0))
    points, _, _ = stitcher.result()
    blended = points[plan[1].start : plan[0].end, 0, 0]
    assert 0.0 < blended.min() and blended.max() < 10.0
    assert blended.tolist() == sorted(blended.tolist())  # smooth handover, no jump


def test_stitching_is_independent_of_insertion_order():
    plan = plan_chunks(20, 8, 3)
    forward = ChunkStitcher(20, 1)
    backward = ChunkStitcher(20, 1)
    for index, chunk in enumerate(plan):
        forward.add(chunk, *_constant(chunk, float(index)))
    for index, chunk in reversed(list(enumerate(plan))):
        backward.add(chunk, *_constant(chunk, float(index)))
    assert np.allclose(forward.result()[0], backward.result()[0])


def test_a_confident_window_outvotes_an_unsure_one():
    plan = plan_chunks(16, 10, 4)
    stitcher = ChunkStitcher(16, 1)
    stitcher.add(plan[0], *_constant(plan[0], 0.0, confidence=0.02))
    stitcher.add(plan[1], *_constant(plan[1], 10.0, confidence=1.0))
    points, _, _ = stitcher.result()
    overlap = points[plan[1].start : plan[0].end, 0, 0]
    assert overlap.min() > 5.0  # the confident window dominates the shared frames


def test_uncovered_frames_stay_nan():
    stitcher = ChunkStitcher(6, 1)
    chunk = Chunk(0, 3)
    stitcher.add(chunk, *_constant(chunk, 1.0))
    points, confidence, visible = stitcher.result()
    assert np.isnan(points[3:]).all()
    assert confidence[3:].tolist() == [[0.0], [0.0], [0.0]]
    assert not visible[3:].any()


def test_non_finite_samples_are_ignored_not_averaged():
    chunk_a, chunk_b = plan_chunks(16, 10, 4)
    stitcher = ChunkStitcher(16, 1)
    points_a, conf_a, vis_a = _constant(chunk_a, 5.0)
    points_a[-2:] = np.nan
    stitcher.add(chunk_a, points_a, conf_a, vis_a)
    stitcher.add(chunk_b, *_constant(chunk_b, 9.0))
    points, _, _ = stitcher.result()
    assert points[chunk_a.end - 1, 0, 0] == pytest.approx(9.0)
    assert np.isfinite(points).all()


def test_visibility_is_a_weighted_vote():
    plan = plan_chunks(16, 10, 4)
    stitcher = ChunkStitcher(16, 1)
    points, conf, visible = _constant(plan[0], 1.0)
    stitcher.add(plan[0], points, conf, np.zeros_like(visible))
    stitcher.add(plan[1], *_constant(plan[1], 1.0))
    _, _, merged_visible = stitcher.result()
    assert not merged_visible[0, 0]
    assert merged_visible[-1, 0]


def test_stitcher_validates_shapes():
    stitcher = ChunkStitcher(10, 2)
    chunk = Chunk(0, 4)
    with pytest.raises(ContractError, match="points"):
        stitcher.add(chunk, np.zeros((4, 1, 2)), np.zeros((4, 2)), np.zeros((4, 2), bool))
    with pytest.raises(ContractError, match="confidence"):
        stitcher.add(chunk, np.zeros((4, 2, 2)), np.zeros((4, 1)), np.zeros((4, 2), bool))
    with pytest.raises(ContractError, match="visible"):
        stitcher.add(chunk, np.zeros((4, 2, 2)), np.zeros((4, 2)), np.zeros((4, 1), bool))
    with pytest.raises(ContractError, match="exceeds length"):
        stitcher.add(Chunk(8, 20), np.zeros((12, 2, 2)), np.zeros((12, 2)), np.zeros((12, 2), bool))
    with pytest.raises(ContractError, match="non-negative"):
        ChunkStitcher(-1, 1)


def test_confidence_is_clamped_into_the_contract():
    chunk = Chunk(0, 3)
    stitcher = ChunkStitcher(3, 1)
    points, conf, visible = _constant(chunk, 1.0)
    conf[:] = [[5.0], [np.nan], [-2.0]]
    stitcher.add(chunk, points, conf, visible)
    _, merged_conf, _ = stitcher.result()
    assert merged_conf.min() >= 0.0 and merged_conf.max() <= 1.0
