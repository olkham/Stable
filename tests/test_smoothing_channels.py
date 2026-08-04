"""Per-channel smoothing: enable, strength, chains, and the project round trip."""

from __future__ import annotations

import numpy as np
import pytest

from stable.core.errors import SmoothingError
from stable.core.geometry import CHANNELS
from stable.core.project import Project, SmoothingSpec
from stable.smoothing import ChannelSmoothing, SmoothingSettings, smooth_curve
from stable.smoothing.filters import FilterContext

CONTEXT = FilterContext(fps=30.0, reference_frame=0, n_frames=200)


def wobble(length: int = 200, seed: int = 0) -> np.ndarray:
    """Return a jittery curve to smooth."""
    rng = np.random.default_rng(seed)
    return np.linspace(0.0, 30.0, length) + rng.normal(scale=2.0, size=length)


# -- one channel -------------------------------------------------------------


def test_a_channel_with_no_filters_is_inactive_and_passes_through():
    """No smoothing means no stabilisation on that channel — the same statement."""
    channel = ChannelSmoothing(channel="x")
    values = wobble()
    assert channel.active is False
    assert channel.strength == 0.0
    assert np.array_equal(channel.apply(values, CONTEXT), values)


def test_a_disabled_channel_passes_through_even_with_filters():
    """The master switch wins over the chain."""
    channel = ChannelSmoothing(channel="x", enabled=False)
    channel.add("gaussian", {"sigma": 10.0})
    values = wobble()
    assert channel.active is False
    assert np.array_equal(channel.apply(values, CONTEXT), values)


def test_filters_chain_in_order():
    """A chain applies each filter to the previous one's output."""
    values = wobble()
    channel = ChannelSmoothing(channel="x")
    channel.add("gaussian", {"sigma": 4.0})
    channel.add("moving_average", {"window": 5})

    expected = smooth_curve(
        values,
        [
            SmoothingSpec(channel="x", filter="gaussian", params={"sigma": 4.0}),
            SmoothingSpec(channel="x", filter="moving_average", params={"window": 5}),
        ],
        CONTEXT,
    )
    assert np.allclose(channel.apply(values, CONTEXT), expected)


def test_a_disabled_filter_in_a_chain_is_skipped():
    """Individual filters can be muted without deleting them."""
    values = wobble()
    specs = [
        SmoothingSpec(channel="x", filter="gaussian", params={"sigma": 9.0}, enabled=False),
        SmoothingSpec(channel="x", filter="moving_average", params={"window": 3}),
    ]
    only_second = smooth_curve(values, specs[1:], CONTEXT)
    assert np.allclose(smooth_curve(values, specs, CONTEXT), only_second)


def test_strength_reads_and_writes_the_whole_chain():
    """One slider per channel, stored in the filters' own ``strength`` parameter."""
    channel = ChannelSmoothing(channel="r")
    channel.add("gaussian", {"sigma": 6.0})
    channel.add("moving_average")
    assert channel.strength == 1.0

    channel.strength = 0.4
    assert channel.strength == 0.4
    assert all(spec.params["strength"] == 0.4 for spec in channel.filters)


def test_strength_is_range_checked():
    """A strength outside [0, 1] is not a partial application of anything."""
    channel = ChannelSmoothing(channel="x")
    channel.add("gaussian")
    with pytest.raises(SmoothingError, match="strength"):
        channel.strength = 1.5


def test_adding_an_invalid_filter_fails_immediately():
    """A typo is caught when the chain is built, not halfway through a render."""
    channel = ChannelSmoothing(channel="x")
    with pytest.raises(SmoothingError, match="unknown smoothing filter"):
        channel.add("bandpass")
    with pytest.raises(SmoothingError, match="cuttoff"):
        channel.add("butterworth", {"cuttoff": 1.0})


def test_a_channel_rejects_specs_for_another_channel():
    """A chain that quietly smoothed the wrong curve would be very hard to spot."""
    with pytest.raises(SmoothingError, match="targets channel"):
        ChannelSmoothing(
            channel="x", filters=[SmoothingSpec(channel="y", filter="gaussian")]
        )


def test_an_unknown_channel_name_is_rejected():
    """Only the four solved channels exist."""
    with pytest.raises(SmoothingError, match="unknown channel"):
        ChannelSmoothing(channel="z")


def test_the_context_channel_is_set_from_the_configuration():
    """Filters are told which curve they are looking at, whatever the caller passed."""
    seen: list[str] = []
    channel = ChannelSmoothing(channel="s")
    channel.add("tripod")
    original = FilterContext(fps=30.0, channel="x", n_frames=10)

    from stable.smoothing import FILTERS

    real = FILTERS["tripod"].fn

    def spy(values, context, params):
        seen.append(context.channel)
        return real(values, context, params)

    object.__setattr__(FILTERS["tripod"], "fn", spy)
    try:
        channel.apply(np.zeros(10), original)
    finally:
        object.__setattr__(FILTERS["tripod"], "fn", real)
    assert seen == ["s"]


# -- all channels ------------------------------------------------------------


def test_settings_fill_in_every_channel_as_disabled():
    """Channels nobody configured are off, so they are never stabilised."""
    settings = SmoothingSettings()
    assert set(settings.channels) == set(CHANNELS)
    assert settings.active_channels == []


def test_uniform_applies_one_filter_to_the_channels_asked_for():
    """The common case: one filter, one setting, several channels."""
    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 1.0}, channels=["x", "y"])
    assert settings.active_channels == ["x", "y"]
    assert settings.channel("r").active is False
    assert settings.channel("x").filters[0].filter == "butterworth"


def test_tripod_locks_every_channel_by_default():
    """Tripod mode is a smoothing configuration, not a special render path."""
    settings = SmoothingSettings.tripod()
    assert settings.active_channels == list(CHANNELS)
    assert all(spec.filter == "tripod" for spec in settings.to_specs())


def test_tripod_can_lock_a_subset_at_partial_strength():
    """"Lock the rotation, let it drift a little" is expressible."""
    settings = SmoothingSettings.tripod(channels=["r"], strength=0.5)
    assert settings.active_channels == ["r"]
    assert settings.channel("r").strength == 0.5


def test_set_enabled_and_set_strength_reach_the_right_channels():
    """The two per-channel controls the smoothing panel exposes."""
    settings = SmoothingSettings.uniform("gaussian", {"sigma": 5.0})
    settings.set_enabled("s", False)
    settings.set_strength(0.25, channels=["x", "y"])

    assert settings.active_channels == ["x", "y", "r"]
    assert settings.channel("x").strength == 0.25
    assert settings.channel("r").strength == 1.0


def test_unknown_channels_are_rejected_everywhere():
    """The same error whichever door the bad name comes in through."""
    with pytest.raises(SmoothingError, match="unknown channel"):
        SmoothingSettings.uniform("gaussian", channels=["q"])
    with pytest.raises(SmoothingError, match="unknown channel"):
        SmoothingSettings().channel("q")
    with pytest.raises(SmoothingError, match="unknown channel"):
        SmoothingSettings(channels={"q": ChannelSmoothing(channel="x")})


# -- project file round trip -------------------------------------------------


def test_settings_round_trip_through_the_project_schema():
    """The grouped in-memory view and the flat on-disk list are the same thing."""
    settings = SmoothingSettings.uniform("butterworth", {"cutoff_hz": 0.8}, channels=["x", "y"])
    settings.channel("r").filters.append(
        SmoothingSpec(channel="r", filter="gaussian", params={"sigma": 12.0})
    )
    settings.channel("r").enabled = True
    settings.set_strength(0.6, channels=["x"])

    restored = SmoothingSettings.from_specs(settings.to_specs())
    assert restored.active_channels == settings.active_channels
    assert restored.channel("x").strength == 0.6
    assert restored.channel("r").filters[0].params["sigma"] == 12.0
    assert restored.to_specs() == settings.to_specs()


def test_a_disabled_channel_survives_the_round_trip_with_its_chain():
    """Turning a channel off must not throw the user's filter settings away."""
    settings = SmoothingSettings.uniform("savgol", {"window": 31}, channels=["s"])
    settings.set_enabled("s", False)

    specs = settings.to_specs()
    assert specs and all(spec.enabled is False for spec in specs)
    restored = SmoothingSettings.from_specs(specs)
    assert restored.channel("s").active is False
    assert restored.channel("s").filters[0].params["window"] == 31


def test_settings_load_from_a_real_project_file(tmp_path):
    """The smoothing block written by a project file is read back as settings."""
    project = Project()
    project.smoothing = SmoothingSettings.uniform(
        "butterworth", {"cutoff_hz": 1.5, "strength": 0.75}
    ).to_specs()
    reloaded = Project.load(project.save(tmp_path / "p.stable.json"))

    settings = SmoothingSettings.from_specs(reloaded.smoothing)
    assert settings.active_channels == list(CHANNELS)
    assert settings.channel("y").strength == 0.75
    assert settings.channel("y").filters[0].params["cutoff_hz"] == 1.5


def test_settings_summarise_themselves_for_a_ui():
    """The smoothing panel needs enable, strength and the chain per channel."""
    summary = SmoothingSettings.uniform("gaussian", {"sigma": 3.0}, channels=["x"]).to_dict()
    assert summary["x"]["active"] is True
    assert summary["x"]["strength"] == 1.0
    assert summary["x"]["filters"][0]["filter"] == "gaussian"
    assert summary["r"]["active"] is False
