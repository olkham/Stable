"""The filter catalogue: schema, behaviour of each filter, and the strength blend."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable.core.errors import SmoothingError
from stable.smoothing import (
    FILTERS,
    FilterContext,
    FilterParam,
    apply_filter,
    describe_filters,
    filter_names,
    get_filter,
)

CONTEXT = FilterContext(fps=30.0, reference_frame=0, channel="x", n_frames=240)

EVERY_FILTER = sorted(FILTERS)


def jittery(length: int = 240, seed: int = 0) -> np.ndarray:
    """Return a slow camera move with fast jitter on top."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 30.0
    return 20.0 * np.sin(2.0 * math.pi * 0.2 * t) + rng.normal(scale=1.2, size=length)


def roughness(curve: np.ndarray) -> float:
    """Return total absolute second difference — how jagged a curve is."""
    return float(np.abs(np.diff(np.asarray(curve), 2)).sum())


# -- the catalogue -----------------------------------------------------------


def test_the_catalogue_holds_exactly_the_six_filters_the_plan_names():
    """Moving average, Gaussian, Butterworth, Savitzky-Golay, spline and tripod."""
    assert set(filter_names()) == {
        "moving_average",
        "gaussian",
        "butterworth",
        "savgol",
        "spline",
        "tripod",
    }


def test_every_filter_publishes_a_usable_schema():
    """A UI can build a control for every parameter without knowing the filter."""
    for entry in describe_filters():
        assert entry["name"] in FILTERS
        assert entry["display_name"] and entry["description"]
        names = [param["name"] for param in entry["params"]]
        assert "strength" in names, "strength is universal"
        assert len(names) == len(set(names))
        for param in entry["params"]:
            assert param["kind"] in ("int", "float")
            assert isinstance(param["default"], int | float)


def test_unknown_filter_names_are_rejected_with_the_alternatives():
    """The error tells the user what they could have written instead."""
    with pytest.raises(SmoothingError, match="unknown smoothing filter"):
        get_filter("bilateral")
    with pytest.raises(SmoothingError, match="butterworth"):
        get_filter("bilateral")


def test_unknown_parameters_are_an_error_not_a_silent_no_op():
    """A typo'd cutoff must not look like a filter that simply does nothing."""
    with pytest.raises(SmoothingError, match="cuttoff_hz"):
        apply_filter("butterworth", jittery(), CONTEXT, {"cuttoff_hz": 1.0})


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("butterworth", {"order": 0}, "order"),
        ("butterworth", {"order": 99}, "order"),
        ("moving_average", {"window": 0}, "window"),
        ("gaussian", {"sigma": -1.0}, "sigma"),
        ("spline", {"lam": -5.0}, "lam"),
        ("gaussian", {"strength": 1.5}, "strength"),
        ("gaussian", {"strength": "loads"}, "must be a number"),
        ("gaussian", {"sigma": float("nan")}, "finite"),
    ],
)
def test_out_of_range_parameters_are_rejected(name, params, message):
    """Bounds come from the schema, so the message names the offending parameter."""
    with pytest.raises(SmoothingError, match=message):
        apply_filter(name, jittery(), CONTEXT, params)


def test_parameters_fall_back_to_their_documented_defaults():
    """Omitting a parameter uses the default the schema advertises."""
    coerced = get_filter("butterworth").coerce({})
    assert coerced == {"cutoff_hz": 1.0, "order": 2, "strength": 1.0}


def test_int_parameters_are_coerced_to_int():
    """A UI or CLI passing "21" or 21.0 for a window still gets an integer."""
    assert get_filter("savgol").coerce({"window": "21"})["window"] == 21
    assert isinstance(get_filter("savgol").coerce({"window": 21.0})["window"], int)


# -- behaviour shared by every filter ----------------------------------------


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_every_filter_preserves_length_and_finiteness(name):
    """Whatever the filter, the curve keeps its shape and stays usable."""
    values = jittery()
    result = apply_filter(name, values, CONTEXT)
    assert result.shape == values.shape
    assert np.isfinite(result).all()


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_every_filter_leaves_a_constant_curve_alone(name):
    """A still camera must stay still: no filter may invent motion."""
    constant = np.full(240, 7.5)
    assert np.allclose(apply_filter(name, constant, CONTEXT), 7.5, atol=1e-9)


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_strength_zero_is_an_exact_bypass(name):
    """Strength 0 must return the raw curve bit for bit, so it truly disables."""
    values = jittery()
    assert np.array_equal(apply_filter(name, values, CONTEXT, {"strength": 0.0}), values)


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_strength_blends_linearly_between_raw_and_filtered(name):
    """Half strength is exactly halfway between the raw and the smoothed curve."""
    values = jittery()
    full = apply_filter(name, values, CONTEXT, {"strength": 1.0})
    half = apply_filter(name, values, CONTEXT, {"strength": 0.5})
    assert np.allclose(half, 0.5 * (values + full), atol=1e-12)


@pytest.mark.parametrize("name", ["moving_average", "gaussian", "butterworth", "savgol", "spline"])
def test_every_smoothing_filter_actually_smooths(name):
    """Each one must reduce the jaggedness it was asked to remove."""
    values = jittery()
    assert roughness(apply_filter(name, values, CONTEXT)) < 0.25 * roughness(values)


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_every_filter_survives_a_very_short_curve(name):
    """Two-frame clips exist; no filter may crash or produce NaN on one."""
    for length in (0, 1, 2, 3):
        values = np.linspace(0.0, 1.0, length)
        result = apply_filter(name, values, FilterContext(fps=30.0, n_frames=length))
        assert result.shape == values.shape
        assert np.isfinite(result).all()


@pytest.mark.parametrize("name", EVERY_FILTER)
def test_non_finite_samples_are_interpolated_before_filtering(name):
    """One NaN must not poison a global filter's entire output."""
    values = jittery()
    holed = values.copy()
    holed[100:104] = np.nan
    result = apply_filter(name, holed, CONTEXT)
    assert np.isfinite(result).all()


def test_an_all_nan_curve_becomes_zero_rather_than_nan():
    """There is no motion to preserve, so the honest answer is none."""
    assert np.array_equal(
        apply_filter("gaussian", np.full(20, np.nan), CONTEXT), np.zeros(20)
    )


# -- individual filters ------------------------------------------------------


def test_moving_average_is_a_box_average_in_the_interior():
    """The interior samples are the plain mean of their window."""
    values = jittery(60, seed=3)
    result = apply_filter("moving_average", values, CONTEXT, {"window": 5})
    for index in (10, 30, 50):
        assert result[index] == pytest.approx(values[index - 2 : index + 3].mean())


def test_moving_average_window_is_forced_odd():
    """An even window has no centre, so it is rounded up to the next odd one."""
    values = jittery(60, seed=3)
    assert np.allclose(
        apply_filter("moving_average", values, CONTEXT, {"window": 4}),
        apply_filter("moving_average", values, CONTEXT, {"window": 5}),
    )


def test_moving_average_of_window_one_changes_nothing():
    """A one-sample window is a copy, not a filter."""
    values = jittery(40)
    assert np.allclose(apply_filter("moving_average", values, CONTEXT, {"window": 1}), values)


@pytest.mark.parametrize("name", ["moving_average", "gaussian", "savgol"])
def test_symmetric_filters_preserve_a_ramp_including_its_ends(name):
    """Odd-reflected padding means a pan does not fake a deceleration at the end."""
    ramp = 0.8 * np.arange(120, dtype=np.float64)
    assert np.allclose(apply_filter(name, ramp, CONTEXT), ramp, atol=1e-6)


def test_gaussian_sigma_controls_how_much_is_removed():
    """A wider kernel gives a smoother curve."""
    values = jittery()
    narrow = roughness(apply_filter("gaussian", values, CONTEXT, {"sigma": 2.0}))
    wide = roughness(apply_filter("gaussian", values, CONTEXT, {"sigma": 12.0}))
    assert wide < narrow


def test_gaussian_cutoff_hz_overrides_sigma():
    """Given a frequency, the width is derived from it and the frame rate."""
    values = jittery()
    by_hz = apply_filter("gaussian", values, CONTEXT, {"cutoff_hz": 1.0, "sigma": 999.0})
    expected_sigma = math.sqrt(math.log(2.0)) * CONTEXT.fps / (2.0 * math.pi)
    by_sigma = apply_filter("gaussian", values, CONTEXT, {"sigma": expected_sigma})
    assert np.allclose(by_hz, by_sigma)


def test_butterworth_cuts_by_frequency():
    """A lower cutoff removes more; that is what makes the control meaningful."""
    values = jittery()
    loose = roughness(apply_filter("butterworth", values, CONTEXT, {"cutoff_hz": 4.0}))
    tight = roughness(apply_filter("butterworth", values, CONTEXT, {"cutoff_hz": 0.5}))
    assert tight < loose


def test_butterworth_above_nyquist_is_refused_with_the_frame_rate_in_the_message():
    """"Remove nothing" is not a filter setting, and the user is told why."""
    with pytest.raises(SmoothingError, match="Nyquist"):
        apply_filter("butterworth", jittery(), FilterContext(fps=24.0), {"cutoff_hz": 20.0})


def test_savgol_preserves_a_sharp_deliberate_move():
    """A whip pan survives Savitzky-Golay far better than a box average."""
    ramp_up = np.concatenate([np.zeros(60), np.linspace(0.0, 60.0, 60), np.full(60, 60.0)])
    savgol = apply_filter("savgol", ramp_up, CONTEXT, {"window": 21, "polyorder": 2})
    boxed = apply_filter("moving_average", ramp_up, CONTEXT, {"window": 21})
    assert np.abs(savgol - ramp_up).max() < np.abs(boxed - ramp_up).max()


def test_spline_lam_controls_the_amount_of_smoothing():
    """Bigger penalty, smoother curve."""
    values = jittery()
    light = roughness(apply_filter("spline", values, CONTEXT, {"lam": 1.0}))
    heavy = roughness(apply_filter("spline", values, CONTEXT, {"lam": 10000.0}))
    assert heavy < light


def test_spline_cutoff_hz_derives_the_penalty():
    """A frequency is a friendlier knob than a raw roughness penalty."""
    values = jittery()
    by_hz = apply_filter("spline", values, CONTEXT, {"cutoff_hz": 1.0})
    omega = 2.0 * math.pi * 1.0 / CONTEXT.fps
    lam = 1.0 / (2.0 - 2.0 * math.cos(omega)) ** 2
    assert np.allclose(by_hz, apply_filter("spline", values, CONTEXT, {"lam": lam}))


def test_spline_above_nyquist_is_refused():
    """Same rule as any other frequency-domain control."""
    with pytest.raises(SmoothingError, match="Nyquist"):
        apply_filter("spline", jittery(), FilterContext(fps=24.0), {"cutoff_hz": 30.0})


def test_tripod_locks_the_curve_to_the_reference_frame():
    """Every sample becomes the reference frame's value — a locked-off camera."""
    values = jittery()
    for reference in (0, 55, 239):
        context = FilterContext(fps=30.0, reference_frame=reference, n_frames=values.size)
        locked = apply_filter("tripod", values, context)
        assert np.allclose(locked, values[reference])


def test_tripod_clamps_an_out_of_range_reference_frame():
    """A reference beyond the curve is clamped rather than raising IndexError."""
    values = jittery(20)
    context = FilterContext(fps=30.0, reference_frame=999, n_frames=20)
    assert np.allclose(apply_filter("tripod", values, context), values[-1])


def test_tripod_takes_no_parameters_beyond_strength():
    """It has nothing to tune; partial locking is what strength is for."""
    assert [p.name for p in get_filter("tripod").all_params] == ["strength"]


# -- context and schema objects ----------------------------------------------


def test_filter_context_rejects_a_non_positive_frame_rate():
    """Frequencies are meaningless without a real frame rate."""
    with pytest.raises(SmoothingError, match="fps"):
        FilterContext(fps=0.0)


def test_filter_context_reports_nyquist():
    """The limit the frequency controls are checked against."""
    assert FilterContext(fps=50.0).nyquist_hz == 25.0


def test_filter_param_schema_round_trips():
    """The parameter schema is JSON-safe and complete."""
    param = FilterParam("window", "int", 15, minimum=1, unit="frames", description="w")
    assert param.to_dict() == {
        "name": "window",
        "kind": "int",
        "default": 15,
        "minimum": 1,
        "maximum": None,
        "unit": "frames",
        "description": "w",
    }
