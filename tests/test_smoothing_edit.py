"""Phase 5 curve editor: sub-range smoothing and keyframe pinning.

The two operations a filter cannot express — "smooth only this part" and
"whatever else you do, hit this frame exactly" — plus their round trip through
the locked v1 project schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import SmoothingError
from stable.core.project import SmoothingSpec
from stable.smoothing.channels import (
    PIN_DIRECTIVE,
    RANGE_DIRECTIVE,
    ChannelSmoothing,
    SmoothingSettings,
)
from stable.smoothing.edit import blend_subrange, pin_curve
from stable.smoothing.filters import FilterContext


@pytest.fixture
def noisy() -> np.ndarray:
    """A slow ramp with fast noise on top — something worth smoothing."""
    rng = np.random.default_rng(3)
    frames = np.arange(120, dtype=np.float64)
    return 0.2 * frames + rng.normal(scale=2.0, size=frames.shape)


@pytest.fixture
def flat_smoothed(noisy: np.ndarray) -> np.ndarray:
    """A deliberately crude "smoothed" curve: the mean, so differences are obvious."""
    return np.full_like(noisy, float(noisy.mean()))


# -- sub-range -----------------------------------------------------------------


def test_subrange_keeps_smoothing_inside_and_raw_outside(noisy, flat_smoothed):
    out = blend_subrange(noisy, flat_smoothed, 40, 79, blend=5)
    assert out[50:70] == pytest.approx(flat_smoothed[50:70], abs=1e-9)
    assert out[:30] == pytest.approx(noisy[:30], abs=1e-9)
    assert out[90:] == pytest.approx(noisy[90:], abs=1e-9)


def test_subrange_crossfade_is_monotone_and_leaves_no_step(noisy, flat_smoothed):
    out = blend_subrange(noisy, flat_smoothed, 40, 79, blend=8)
    weight = (out - noisy) / (flat_smoothed - noisy)
    ramp = weight[32:41]
    assert np.all(np.diff(ramp) >= -1e-9)
    assert ramp[0] == pytest.approx(0.0, abs=1e-9)
    assert ramp[-1] == pytest.approx(1.0, abs=1e-9)


def test_zero_blend_is_a_hard_edge(noisy, flat_smoothed):
    out = blend_subrange(noisy, flat_smoothed, 40, 79, blend=0)
    assert out[39] == pytest.approx(noisy[39])
    assert out[40] == pytest.approx(flat_smoothed[40])


def test_subrange_validates_its_arguments(noisy, flat_smoothed):
    with pytest.raises(SmoothingError, match="end < start"):
        blend_subrange(noisy, flat_smoothed, 50, 10)
    with pytest.raises(SmoothingError, match="same length"):
        blend_subrange(noisy, flat_smoothed[:10], 0, 5)
    with pytest.raises(SmoothingError, match="blend must be >= 0"):
        blend_subrange(noisy, flat_smoothed, 0, 5, blend=-1)


# -- pinning -------------------------------------------------------------------


def test_pinned_frames_land_exactly_on_the_raw_value(noisy, flat_smoothed):
    pins = [10, 60, 110]
    out = pin_curve(noisy, flat_smoothed, pins, radius=15)
    for frame in pins:
        assert out[frame] == pytest.approx(noisy[frame], abs=1e-9)


def test_a_pin_is_local_and_leaves_distant_frames_smoothed(noisy, flat_smoothed):
    out = pin_curve(noisy, flat_smoothed, [60], radius=10)
    assert out[60] == pytest.approx(noisy[60], abs=1e-9)
    assert out[20] == pytest.approx(flat_smoothed[20], abs=1e-9)
    assert out[100] == pytest.approx(flat_smoothed[100], abs=1e-9)


def test_pins_closer_than_two_radii_blend_into_each_other_and_stay_exact(noisy, flat_smoothed):
    out = pin_curve(noisy, flat_smoothed, [50, 58], radius=20)
    assert out[50] == pytest.approx(noisy[50], abs=1e-9)
    assert out[58] == pytest.approx(noisy[58], abs=1e-9)
    # Nothing between two pins may overshoot the residuals they interpolate.
    residuals = sorted([noisy[50] - flat_smoothed[50], noisy[58] - flat_smoothed[58]])
    between = out[50:59] - flat_smoothed[50:59]
    assert between.min() >= residuals[0] - 1e-9
    assert between.max() <= residuals[1] + 1e-9


def test_pin_correction_is_continuous(noisy, flat_smoothed):
    out = pin_curve(noisy, flat_smoothed, [30, 90], radius=12)
    correction = out - flat_smoothed
    # No discontinuity: the correction never jumps more than its own scale in a
    # single frame, which is what stops a pin putting a step in the render.
    assert np.abs(np.diff(correction)).max() < np.abs(correction).max()


def test_pins_outside_the_clip_and_duplicates_are_ignored(noisy, flat_smoothed):
    out = pin_curve(noisy, flat_smoothed, [60, 60, -3, 999], radius=10)
    assert out[60] == pytest.approx(noisy[60], abs=1e-9)
    assert np.isfinite(out).all()


def test_no_pins_is_a_passthrough(noisy, flat_smoothed):
    assert pin_curve(noisy, flat_smoothed, [], radius=10) == pytest.approx(flat_smoothed)


def test_pin_radius_must_be_positive(noisy, flat_smoothed):
    with pytest.raises(SmoothingError, match="radius must be positive"):
        pin_curve(noisy, flat_smoothed, [5], radius=0)


# -- integration with the channel configuration --------------------------------


def context(n_frames: int) -> FilterContext:
    """A 25 fps filter context of the given clip length."""
    return FilterContext(fps=25.0, reference_frame=0, channel="x", n_frames=n_frames)


def test_channel_applies_filter_then_range_then_pins(noisy):
    channel = ChannelSmoothing(
        channel="x",
        enabled=True,
        sub_range=(40, 79),
        range_blend=5,
        pins=[60],
        pin_radius=8,
    )
    channel.add("gaussian", {"sigma": 12.0})
    out = channel.apply(noisy, context(noisy.shape[0]))

    assert out[60] == pytest.approx(noisy[60], abs=1e-9)  # pinning wins, applied last
    assert out[:30] == pytest.approx(noisy[:30], abs=1e-9)  # outside the range: untouched
    # Inside the range but away from the pin, the Gaussian is really in charge.
    assert abs(out[45] - noisy[45]) > 1e-6


def test_pinning_survives_being_applied_after_a_strong_filter(noisy):
    channel = ChannelSmoothing(channel="x", enabled=True, pins=[15, 70, 110], pin_radius=10)
    channel.add("butterworth", {"cutoff_hz": 0.4})
    out = channel.apply(noisy, context(noisy.shape[0]))
    for frame in (15, 70, 110):
        assert out[frame] == pytest.approx(noisy[frame], abs=1e-9)


def test_disabled_channel_ignores_range_and_pins(noisy):
    channel = ChannelSmoothing(
        channel="x", enabled=False, sub_range=(10, 20), pins=[15], pin_radius=5
    )
    channel.add("tripod")
    assert channel.apply(noisy, context(noisy.shape[0])) == pytest.approx(noisy)


# -- project-file round trip ----------------------------------------------------


def test_range_and_pins_round_trip_through_the_flat_v1_spec_list():
    settings = SmoothingSettings.uniform("gaussian", {"sigma": 6.0}, channels=["x", "y"])
    settings.channel("x").sub_range = (12, 88)
    settings.channel("x").range_blend = 4
    settings.channel("x").pins = [20, 50]
    settings.channel("x").pin_radius = 9

    specs = settings.to_specs()
    assert all(isinstance(spec, SmoothingSpec) for spec in specs)
    assert {spec.filter for spec in specs} == {"gaussian", RANGE_DIRECTIVE, PIN_DIRECTIVE}

    # Straight through JSON, as the project file stores it.
    restored = SmoothingSettings.from_specs(
        [SmoothingSpec.from_dict(spec.to_dict()) for spec in specs]
    )
    assert restored.channel("x").sub_range == (12, 88)
    assert restored.channel("x").range_blend == 4
    assert restored.channel("x").pins == [20, 50]
    assert restored.channel("x").pin_radius == 9
    assert restored.channel("y").sub_range is None
    assert restored.channel("y").pins == []
    assert [s.filter for s in restored.channel("x").filters] == ["gaussian"]


def test_directives_never_reach_the_filter_registry():
    # A directive is not a filter; from_specs has to route it before lookup, or
    # loading any project with a pin would explode on an unknown filter name.
    settings = SmoothingSettings.from_specs(
        [
            SmoothingSpec(channel="r", filter="gaussian", params={"sigma": 4.0}),
            SmoothingSpec(channel="r", filter=PIN_DIRECTIVE, params={"frame": 7, "radius": 3}),
        ]
    )
    assert settings.channel("r").pins == [7]
    assert settings.channel("r").pin_radius == 3
    assert len(settings.channel("r").filters) == 1


def test_disabled_directives_are_kept_but_not_applied():
    settings = SmoothingSettings.from_specs(
        [
            SmoothingSpec(channel="x", filter="gaussian", params={"sigma": 4.0}),
            SmoothingSpec(
                channel="x", filter=RANGE_DIRECTIVE, params={"start": 1, "end": 9}, enabled=False
            ),
        ]
    )
    assert settings.channel("x").sub_range is None


def test_malformed_directives_are_rejected_with_a_useful_message():
    with pytest.raises(SmoothingError, match="needs a 'frame' parameter"):
        SmoothingSettings.from_specs([SmoothingSpec(channel="x", filter=PIN_DIRECTIVE, params={})])
    with pytest.raises(SmoothingError, match="invalid smoothing range"):
        SmoothingSettings.from_specs(
            [SmoothingSpec(channel="x", filter=RANGE_DIRECTIVE, params={"start": 9, "end": 2})]
        )


def test_settings_to_dict_exposes_the_editor_state():
    settings = SmoothingSettings.uniform("gaussian", {"sigma": 3.0}, channels=["y"])
    settings.channel("y").sub_range = (5, 25)
    settings.channel("y").pins = [10]
    payload = settings.to_dict()
    assert payload["y"]["range"] == [5, 25]
    assert payload["y"]["pins"] == [10]
    assert payload["x"]["range"] is None
