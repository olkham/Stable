"""Tests for the shared confidence-normalisation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import ContractError
from stable.trackers.confidence import (
    combine_scores,
    normalize_scores,
    sanitize_scores,
    sigmoid,
    visibility_from_confidence,
)


def test_sigmoid_matches_the_definition_and_survives_extremes():
    values = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    out = sigmoid(values)
    assert out[2] == pytest.approx(0.5)
    assert out[1] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)
    assert np.isfinite(out).all()  # no overflow warning-turned-nan at +-800


def test_sanitize_clamps_anything_into_the_contract():
    out = sanitize_scores(np.array([np.nan, -np.inf, np.inf, -3.0, 0.25, 7.0]))
    assert out.tolist() == [0.0, 0.0, 1.0, 0.0, 0.25, 1.0]
    assert out.dtype == np.float32


def test_normalize_is_a_clamped_linear_stretch():
    raw = np.array([0.0, 0.1, 0.3, 0.5, 0.9, 1.0])
    out = normalize_scores(raw, low=0.1, high=0.9)
    assert out[0] == 0.0 and out[1] == 0.0  # dead zone: no information
    assert out[4] == pytest.approx(1.0) and out[5] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.25)
    assert out[3] == pytest.approx(0.5)
    assert out.dtype == np.float32


def test_normalize_is_monotonic_and_bounded():
    raw = np.linspace(-1.0, 2.0, 50)
    out = normalize_scores(raw, low=0.15, high=0.9)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-7)


def test_gamma_biases_the_middle_without_moving_the_ends():
    raw = np.array([0.0, 0.5, 1.0])
    strict = normalize_scores(raw, gamma=2.0)
    lenient = normalize_scores(raw, gamma=0.5)
    assert strict[1] < 0.5 < lenient[1]
    assert strict[0] == lenient[0] == 0.0
    assert strict[2] == lenient[2] == 1.0


def test_normalize_validates_its_range():
    with pytest.raises(ContractError, match="high > low"):
        normalize_scores(np.zeros(3), low=0.9, high=0.1)
    with pytest.raises(ContractError, match="gamma"):
        normalize_scores(np.zeros(3), gamma=0.0)


def test_combining_independent_signals_multiplies_them():
    visibility = np.array([1.0, 0.9, 0.0])
    accuracy = np.array([1.0, 0.8, 1.0])
    assert combine_scores(visibility, accuracy) == pytest.approx([1.0, 0.72, 0.0])


def test_combining_skips_absent_heads_and_clips_factors():
    assert combine_scores(np.array([0.5]), None) == pytest.approx([0.5])
    assert combine_scores(np.array([2.0]), np.array([-1.0])) == pytest.approx([0.0])
    with pytest.raises(ContractError, match="at least one"):
        combine_scores(None, None)


def test_visibility_thresholds_confidence():
    confidence = np.array([0.0, 0.49, 0.5, 1.0])
    assert visibility_from_confidence(confidence).tolist() == [False, False, True, True]
    assert visibility_from_confidence(confidence, 0.99).tolist() == [False, False, False, True]
    with pytest.raises(ContractError, match="threshold"):
        visibility_from_confidence(confidence, 1.5)
