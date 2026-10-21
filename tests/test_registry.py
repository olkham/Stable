"""Contract tests for tracker registration and entry-point discovery."""

from __future__ import annotations

from importlib import metadata

import pytest
from conftest import FakeEntryPoint

from stable.core.errors import TrackerError, TrackerNotFoundError
from stable.trackers.base import TrackerPlugin
from stable.trackers.dummy import DummyTracker
from stable.trackers.registry import ENTRY_POINT_GROUP, TrackerRegistry


class ThirdPartyTracker(TrackerPlugin):
    """Stand-in for a tracker shipped by a separate pip package."""

    name = "third_party"
    display_name = "Third Party"
    requires_gpu = False
    license = "Apache-2.0"

    def track(self, video, queries, frame_range, direction="bidir", progress=None):
        return None  # never executed in these tests


class GpuOnlyTracker(ThirdPartyTracker):
    """A tracker that refuses to run without a GPU."""

    name = "gpu_only"
    requires_gpu = True


class UnavailableTracker(ThirdPartyTracker):
    """A tracker whose optional dependency is missing."""

    name = "unavailable"

    @classmethod
    def is_available(cls) -> bool:
        return False


class AbstractTracker(TrackerPlugin):
    """A subclass that forgot to implement track()."""

    name = "abstract"


@pytest.fixture
def reg() -> TrackerRegistry:
    return TrackerRegistry()


def test_builtin_dummy_is_discovered(reg):
    assert "dummy" in reg.discover()
    assert reg.get("dummy") is DummyTracker


def test_entry_points_are_loaded(reg):
    names = reg.discover(entry_points=[FakeEntryPoint("third_party", ThirdPartyTracker)])
    assert names == ["dummy", "third_party"]
    assert reg.get("third_party") is ThirdPartyTracker


def test_the_package_advertises_its_own_entry_point():
    """`pip install -e .` must publish the stable.trackers group end to end."""
    assert metadata.version("stable-tracker")
    names = {ep.name for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}
    assert "dummy" in names
    fresh = TrackerRegistry()
    assert "dummy" in fresh.discover(strict=True)


def test_a_broken_plugin_never_takes_the_app_down(reg, caplog):
    broken = FakeEntryPoint("broken", None, error=ImportError("no such module"))
    names = reg.discover(entry_points=[broken, FakeEntryPoint("third_party", ThirdPartyTracker)])
    assert names == ["dummy", "third_party"]
    assert any("broken" in record.message for record in caplog.records)


def test_strict_discovery_surfaces_plugin_failures(reg):
    broken = FakeEntryPoint("broken", None, error=ImportError("boom"))
    with pytest.raises(TrackerError, match="broken"):
        reg.discover(entry_points=[broken], strict=True)


def test_registration_rules(reg):
    reg.register(ThirdPartyTracker)
    reg.register(ThirdPartyTracker)  # same class again is a no-op

    class Clashing(ThirdPartyTracker):
        """A different class claiming a taken name."""

    with pytest.raises(TrackerError, match="already registered"):
        reg.register(Clashing)
    reg.register(Clashing, replace=True)
    assert reg.get("third_party") is Clashing

    with pytest.raises(TrackerError):
        reg.register(AbstractTracker)
    with pytest.raises(TrackerError):
        reg.register(object)  # type: ignore[arg-type]

    class Nameless(ThirdPartyTracker):
        """A plugin that forgot to set a name."""

        name = ""

    with pytest.raises(TrackerError, match="non-empty"):
        reg.register(Nameless)


def test_unregister_and_containment(reg):
    reg.register(ThirdPartyTracker)
    assert "third_party" in reg and len(reg) == 1
    reg.unregister("third_party")
    reg.unregister("third_party")  # idempotent
    assert "third_party" not in reg


def test_unknown_tracker_names_are_reported_clearly(reg):
    # A name no first- or third-party plugin will ever claim: real tracker names
    # (cotracker3, tapir, ...) are registered by this package's own entry points.
    with pytest.raises(TrackerNotFoundError, match="dummy"):
        reg.get("no_such_tracker")


def test_create_instantiates_and_prepares(reg):
    plugin = reg.create("dummy")
    assert isinstance(plugin, DummyTracker)
    assert plugin.is_prepared
    assert plugin.device in ("cpu", "cuda", "mps")

    unprepared = reg.create("dummy", prepare=False)
    assert not unprepared.is_prepared
    unprepared.prepare("cpu")
    assert unprepared.is_prepared
    unprepared.release()
    assert not unprepared.is_prepared


def test_create_refuses_gpu_only_plugins_on_cpu(reg):
    reg.register(GpuOnlyTracker)
    with pytest.raises(TrackerError, match="requires a GPU"):
        reg.create("gpu_only", device="cpu")


def test_describe_is_ui_ready(reg):
    reg.discover(entry_points=[FakeEntryPoint("unavailable", UnavailableTracker)])
    rows = {row["name"]: row for row in reg.describe()}
    assert rows["dummy"]["display_name"] == "Dummy (testing)"
    assert rows["dummy"]["license"] == "MIT"
    assert rows["dummy"]["available"] is True
    assert rows["unavailable"]["available"] is False
    assert set(rows["dummy"]) == {
        "name",
        "display_name",
        "requires_gpu",
        "license",
        "tracks_boxes",
        "inputs",
        "available",
        "unavailable_reason",
    }


def test_an_available_tracker_has_nothing_to_explain(reg):
    rows = {row["name"]: row for row in reg.describe()}
    assert rows["dummy"]["unavailable_reason"] is None


def test_an_unavailable_tracker_says_what_would_fix_it(reg):
    """ "Unavailable" alone is indistinguishable from "broken"."""

    class NeedsSomething(ThirdPartyTracker):
        name = "needs_something"

        @classmethod
        def is_available(cls) -> bool:
            return False

        @classmethod
        def unavailable_reason(cls) -> str:
            return "needs_something needs widgets: pip install widgets"

    reg.discover(entry_points=[FakeEntryPoint("needs_something", NeedsSomething)])
    row = {r["name"]: r for r in reg.describe()}["needs_something"]
    assert row["available"] is False
    assert "pip install widgets" in row["unavailable_reason"]


def test_one_exploding_plugin_does_not_hide_the_others(reg):
    """A bad plugin must cost its own row, not the whole list."""

    class Exploding(ThirdPartyTracker):
        name = "exploding"

        @classmethod
        def is_available(cls) -> bool:
            raise RuntimeError("boom")

    reg.discover(entry_points=[FakeEntryPoint("exploding", Exploding)])
    rows = {row["name"]: row for row in reg.describe()}
    assert rows["dummy"]["available"] is True
    assert rows["exploding"]["available"] is False
    assert "boom" in rows["exploding"]["unavailable_reason"]


def test_lookup_triggers_discovery_lazily(reg):
    assert len(reg) == 0
    reg.get("dummy")
    assert len(reg) >= 1
    assert "TrackerRegistry" in repr(reg)


# -- per-point options reaching (only) the plugins that take them --------------
#
# `create(options=...)` is how a per-point setting reaches the tracker that
# honours it. Settings are stored per *point* and follow it across a tracker
# change, so the filter has to be exactly right in both directions: drop what
# the new plugin cannot take, and keep what it can — including arguments it
# only declares through a base class.


class ForwardingTracker(ThirdPartyTracker):
    """A plugin whose real arguments live on a base it forwards `**kwargs` to.

    The shape every deep tracker here has (and `klt` over `lk`): the subclass
    names a couple of its own arguments and passes the rest up.
    """

    name = "forwarding_base"

    def __init__(self, *, window: int = 7, mode: str = "fast") -> None:
        super().__init__()
        self.window = window
        self.mode = mode


class ForwardingSubclass(ForwardingTracker):
    name = "forwarding"

    def __init__(self, *, extra: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extra = extra


def test_kwargs_does_not_mean_a_plugin_accepts_anything(reg):
    """The regression: a stored setting reaching a plugin that cannot take it.

    `**kwargs` used to be read as "accepts everything", so an option stored
    while a point was on one tracker was passed verbatim to the next one and
    raised `__init__() got an unexpected keyword argument` from the *base* the
    subclass forwards to — which is a crash, not the documented "ignored".
    """
    reg.discover(entry_points=[FakeEntryPoint("forwarding", ForwardingSubclass)])
    plugin = reg.create("forwarding", options={"window": 21, "nonsense": 3}, prepare=False)
    assert isinstance(plugin, ForwardingSubclass)
    assert plugin.window == 21  # declared on the base, so it is honoured
    assert not hasattr(plugin, "nonsense")  # declared nowhere, so it is dropped


def test_every_installed_tracker_survives_another_trackers_settings():
    """Switching a point's tracker must never be able to break tracking.

    Per-point settings are stored under the plugin's own argument names and are
    deliberately *not* cleared when the tracker changes, so every plugin has to
    tolerate the union of everything any other plugin might have stored. This is
    the user-facing bug ("TAPIR/RAFT/CoTracker used to work") as a test.
    """
    from stable.server.tracker_options import SETTINGS
    from stable.trackers.registry import registry as process_registry

    stale = {setting.attribute: setting.minimum for setting in SETTINGS}
    for name in process_registry.discover():
        plugin = process_registry.create(name, options=stale, prepare=False)
        assert plugin.name == name


def test_a_plugin_only_declaring_kwargs_receives_nothing_rather_than_crashing(reg):
    """Free-form `**kwargs` cannot be told apart from forwarding, so it is strict.

    A plugin that means to take arbitrary keyword arguments has to declare them;
    passing an argument that might raise is the worse of the two readings.
    """

    class Swallower(ThirdPartyTracker):
        name = "swallower"

        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.seen = kwargs

    reg.discover(entry_points=[FakeEntryPoint("swallower", Swallower)])
    assert reg.create("swallower", options={"whatever": 1}, prepare=False).seen == {}
