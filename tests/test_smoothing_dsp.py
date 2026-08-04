"""The pure-NumPy DSP kernels, checked against theory and (if present) SciPy.

The reference implementations live in SciPy, which this project deliberately
does not depend on. Where SciPy happens to be installed these tests assert exact
agreement with it; where it is not, the same properties are checked directly
against the closed-form theory, so the suite is meaningful either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable.core.errors import SmoothingError
from stable.smoothing import dsp

try:  # pragma: no cover - depends on the environment
    from scipy import signal as _sps
except ImportError:  # pragma: no cover - the supported no-SciPy environment
    _sps = None

requires_scipy = pytest.mark.skipif(_sps is None, reason="SciPy is not installed")


def ramp(length: int = 200, slope: float = 0.7) -> np.ndarray:
    """Return a straight line — the signal a camera pan looks like locally."""
    return slope * np.arange(length, dtype=np.float64) + 3.0


def noisy_walk(length: int = 300, seed: int = 0) -> np.ndarray:
    """Return a smooth drift with jitter on top — the shape of handheld motion."""
    rng = np.random.default_rng(seed)
    base = 20.0 * np.sin(np.linspace(0.0, 3.0, length))
    return base + rng.normal(scale=1.5, size=length)


# -- odd extension -----------------------------------------------------------


def test_odd_extension_continues_the_slope():
    """Odd reflection extrapolates a ramp instead of folding it back."""
    values = ramp(10)
    extended = dsp.odd_extension(values, 4)
    assert extended.shape[0] == 18
    assert np.allclose(np.diff(extended), 0.7)


def test_odd_extension_clips_the_pad_to_the_signal():
    """Asking for more padding than there is signal is clamped, not an error."""
    assert dsp.odd_extension(np.arange(4.0), 99).shape[0] == 4 + 2 * 3
    assert dsp.odd_extension(np.arange(4.0), 0).tolist() == [0.0, 1.0, 2.0, 3.0]


# -- Butterworth -------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_butterworth_has_unit_dc_gain(order):
    """A constant curve must survive the filter untouched — a gain error is a zoom error."""
    sos = dsp.butter_lowpass_sos(order, 1.0, 30.0)
    constant = np.full(200, 4.25)
    assert np.allclose(dsp.sosfiltfilt(sos, constant), 4.25, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_butterworth_magnitude_matches_the_closed_form(order):
    """The response follows ``1/sqrt(1 + (w/wc)^2n)`` of the analogue prototype."""
    fps, cutoff = 50.0, 5.0
    sos = dsp.butter_lowpass_sos(order, cutoff, fps)
    for probe_hz in (0.5, 2.0, 5.0, 12.0):
        # Measure the gain by filtering a pure tone (forward only, so the
        # magnitude is the single-pass response). RMS rather than peak: at ten
        # samples per period the peak sample can miss the true crest by 5%.
        n = 4000
        t = np.arange(n) / fps
        tone = np.sin(2.0 * math.pi * probe_hz * t)
        steady = slice(n // 2, None)
        filtered = dsp.sosfilt(sos, tone)[steady]
        gain = float(np.sqrt(np.mean(filtered**2)) / np.sqrt(np.mean(tone[steady] ** 2)))
        # Digital Butterworth: warp the probe the same way the design warped
        # the cutoff, then evaluate the analogue magnitude.
        warp = math.tan(math.pi * probe_hz / fps) / math.tan(math.pi * cutoff / fps)
        expected = 1.0 / math.sqrt(1.0 + warp ** (2 * order))
        assert gain == pytest.approx(expected, rel=0.02)


def test_butterworth_is_zero_phase_when_run_both_ways():
    """filtfilt must not delay the signal; a lag would slide the whole shot late."""
    fps = 30.0
    n = 600
    tone = np.sin(2.0 * math.pi * 0.5 * np.arange(n) / fps)
    sos = dsp.butter_lowpass_sos(3, 2.0, fps)

    zero_phase = dsp.sosfiltfilt(sos, tone)
    causal = dsp.sosfilt(sos, tone)
    middle = slice(200, 400)
    # The zero-phase result lines up with the input; the causal one visibly lags.
    assert np.abs(zero_phase[middle] - tone[middle]).max() < 0.02
    assert np.abs(causal[middle] - tone[middle]).max() > 0.1


def test_butterworth_removes_the_fast_component_and_keeps_the_slow_one():
    """The point of the filter: judder out, camera move in."""
    fps, n = 30.0, 600
    t = np.arange(n) / fps
    slow = 10.0 * np.sin(2.0 * math.pi * 0.2 * t)
    fast = 2.0 * np.sin(2.0 * math.pi * 6.0 * t)
    sos = dsp.butter_lowpass_sos(4, 1.0, fps)

    filtered = dsp.sosfiltfilt(sos, slow + fast)
    middle = slice(100, 500)
    assert np.abs(filtered[middle] - slow[middle]).max() < 0.3


def test_short_signals_pass_through_untouched():
    """Two samples cannot be filtered, so they are returned as they came."""
    sos = dsp.butter_lowpass_sos(2, 1.0, 30.0)
    assert dsp.sosfiltfilt(sos, np.array([1.0, 9.0])).tolist() == [1.0, 9.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"order": 0, "cutoff_hz": 1.0, "fps": 30.0}, "order"),
        ({"order": 2, "cutoff_hz": 1.0, "fps": 0.0}, "fps"),
        ({"order": 2, "cutoff_hz": 20.0, "fps": 30.0}, "Nyquist"),
        ({"order": 2, "cutoff_hz": 0.0, "fps": 30.0}, "Nyquist"),
    ],
)
def test_invalid_butterworth_design_is_rejected(kwargs, message):
    """Impossible filter designs fail with an explanation, not a nan-filled curve."""
    with pytest.raises(SmoothingError, match=message):
        dsp.butter_lowpass_sos(**kwargs)


@requires_scipy
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(("cutoff", "fps"), [(1.0, 30.0), (0.5, 24.0), (5.0, 60.0)])
def test_butterworth_design_matches_scipy(order, cutoff, fps):
    """The hand-rolled design is the same filter SciPy's ``butter`` produces."""
    mine = dsp.butter_lowpass_sos(order, cutoff, fps)
    reference = _sps.butter(order, cutoff / (fps / 2), btype="low", output="sos")
    _, ours = _sps.sosfreqz(mine, worN=512)
    _, theirs = _sps.sosfreqz(reference, worN=512)
    assert np.abs(np.abs(ours) - np.abs(theirs)).max() < 1e-9


@requires_scipy
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
def test_filtfilt_matches_scipy(order):
    """Same padding, same initial conditions, same numbers."""
    values = noisy_walk(300)
    mine = dsp.sosfiltfilt(dsp.butter_lowpass_sos(order, 1.0, 30.0), values)
    theirs = _sps.sosfiltfilt(_sps.butter(order, 1.0 / 15.0, output="sos"), values)
    assert np.abs(mine - theirs).max() < 1e-9


# -- Savitzky-Golay ----------------------------------------------------------


def test_savgol_reproduces_a_polynomial_exactly():
    """A cubic fit leaves a cubic alone, edges included — that is the whole idea."""
    t = np.arange(60, dtype=np.float64)
    cubic = 0.001 * t**3 - 0.05 * t**2 + 2.0 * t + 7.0
    assert np.allclose(dsp.savgol_filter(cubic, 21, 3), cubic, atol=1e-6)


def test_savgol_smooths_noise_but_keeps_the_trend():
    """Noise falls away while the underlying move is preserved at both ends."""
    t = np.arange(200, dtype=np.float64)
    trend = 0.05 * t**2
    rng = np.random.default_rng(1)
    smoothed = dsp.savgol_filter(trend + rng.normal(scale=2.0, size=t.size), 31, 2)
    assert np.abs(smoothed - trend).max() < 2.0


def test_savgol_window_is_clipped_to_the_curve():
    """A window longer than the clip is shrunk rather than refused."""
    values = ramp(9)
    assert np.allclose(dsp.savgol_filter(values, 101, 1), values, atol=1e-9)


def test_savgol_rejects_an_impossible_polyorder():
    """A polynomial degree the window cannot support is a configuration error."""
    with pytest.raises(SmoothingError, match="polyorder"):
        dsp.savgol_filter(ramp(6), 5, 9)
    with pytest.raises(SmoothingError, match="polyorder"):
        dsp.savgol_coeffs(5, 5, 2)


@requires_scipy
@pytest.mark.parametrize(("window", "polyorder"), [(11, 2), (21, 3), (7, 1), (9, 4)])
def test_savgol_matches_scipy(window, polyorder):
    """Including the interpolated edges, which is the fiddly part."""
    values = noisy_walk(200, seed=4)
    mine = dsp.savgol_filter(values, window, polyorder)
    theirs = _sps.savgol_filter(values, window, polyorder, mode="interp")
    assert np.abs(mine - theirs).max() < 1e-9


# -- Gaussian ----------------------------------------------------------------


def test_gaussian_kernel_is_normalised_and_symmetric():
    """A kernel that does not sum to 1 would change the curve's level."""
    kernel = dsp.gaussian_kernel(3.0)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel[::-1])
    assert kernel.shape[0] == 2 * 12 + 1


def test_gaussian_kernel_radius_can_be_capped():
    """A very wide filter on a short curve is truncated, not refused."""
    assert dsp.gaussian_kernel(50.0, max_radius=4).shape[0] == 9
    assert dsp.gaussian_kernel(50.0, max_radius=0).shape[0] == 3


def test_gaussian_kernel_rejects_a_non_positive_sigma():
    """Zero width is not a filter."""
    with pytest.raises(SmoothingError, match="sigma"):
        dsp.gaussian_kernel(0.0)


# -- smoothing spline --------------------------------------------------------


def test_spline_with_zero_penalty_is_the_identity():
    """No penalty means no smoothing, exactly."""
    values = noisy_walk(50)
    assert np.allclose(dsp.smoothing_spline(values, 0.0), values)


def test_spline_smooths_more_as_the_penalty_grows():
    """Roughness falls monotonically with lambda."""
    values = noisy_walk(200, seed=2)

    def roughness(curve: np.ndarray) -> float:
        return float(np.abs(np.diff(curve, 2)).sum())

    scores = [roughness(dsp.smoothing_spline(values, lam)) for lam in (0.0, 1.0, 100.0, 1e5)]
    assert scores == sorted(scores, reverse=True)


def test_spline_approaches_a_straight_line_for_a_huge_penalty():
    """The infinite-penalty limit of a second-difference penalty is a line."""
    values = noisy_walk(120, seed=6)
    fitted = dsp.smoothing_spline(values, 1e12)
    assert np.abs(np.diff(fitted, 2)).max() < 1e-4


def test_spline_preserves_a_straight_line_at_any_penalty():
    """A ramp has no second difference to penalise, so it survives untouched."""
    values = ramp(80)
    assert np.allclose(dsp.smoothing_spline(values, 500.0), values, atol=1e-6)


def test_spline_handles_curves_too_short_to_fit():
    """Fewer than four samples define no cubic, so they pass through."""
    assert dsp.smoothing_spline(np.array([1.0, 2.0, 3.0]), 10.0).tolist() == [1.0, 2.0, 3.0]


def test_spline_rejects_a_negative_penalty():
    """A negative penalty would amplify roughness rather than remove it."""
    with pytest.raises(SmoothingError, match="lam"):
        dsp.smoothing_spline(ramp(20), -1.0)


@requires_scipy
@pytest.mark.parametrize("lam", [0.5, 10.0, 1000.0])
def test_spline_matches_scipy(lam):
    """Same Reinsch formulation, same answer as ``make_smoothing_spline``."""
    from scipy.interpolate import make_smoothing_spline

    values = noisy_walk(300, seed=1)
    knots = np.arange(values.size, dtype=np.float64)
    mine = dsp.smoothing_spline(values, lam)
    theirs = make_smoothing_spline(knots, values, lam=lam)(knots)
    assert np.abs(mine - theirs).max() < 1e-8


# -- banded linear algebra ---------------------------------------------------


def test_banded_cholesky_solves_the_same_system_as_a_dense_one():
    """The O(T) banded solver agrees with a dense reference solve."""
    size = 30
    diagonal, first, second = 6.0, -2.0, 0.5
    dense = (
        np.diag(np.full(size, diagonal))
        + np.diag(np.full(size - 1, first), 1)
        + np.diag(np.full(size - 1, first), -1)
        + np.diag(np.full(size - 2, second), 2)
        + np.diag(np.full(size - 2, second), -2)
    )
    band = np.zeros((3, size))
    band[0, :] = diagonal
    band[1, : size - 1] = first
    band[2, : size - 2] = second

    rhs = np.random.default_rng(0).normal(size=size)
    assert np.allclose(dsp.cho_solve_banded(dsp.cholesky_banded(band), rhs.copy()),
                       np.linalg.solve(dense, rhs))


def test_banded_cholesky_rejects_an_indefinite_matrix():
    """A matrix that is not positive definite has no Cholesky factor."""
    band = np.zeros((2, 4))
    band[0, :] = -1.0
    with pytest.raises(SmoothingError, match="positive definite"):
        dsp.cholesky_banded(band)
