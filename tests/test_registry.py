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
    assert set(rows["dummy"]) == {"name", "display_name", "requires_gpu", "license", "available"}


def test_lookup_triggers_discovery_lazily(reg):
    assert len(reg) == 0
    reg.get("dummy")
    assert len(reg) >= 1
    assert "TrackerRegistry" in repr(reg)
