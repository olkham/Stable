"""Contract tests for device selection (cuda when available, cpu always)."""

from __future__ import annotations

import pytest

from stable.core import device as device_mod
from stable.core.device import available_devices, device_info, select_device
from stable.core.errors import DeviceError


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    """Keep an ambient STABLE_DEVICE out of these tests."""
    monkeypatch.delenv("STABLE_DEVICE", raising=False)


def test_cpu_is_always_available():
    assert "cpu" in available_devices()
    assert available_devices()[-1] == "cpu"


def test_auto_selection_prefers_the_best_device():
    assert select_device() == available_devices()[0]
    assert select_device("auto") == available_devices()[0]
    assert select_device("") == available_devices()[0]


def test_explicit_cpu_is_honoured():
    assert select_device("cpu") == "cpu"
    assert select_device("CPU") == "cpu"


def test_unavailable_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "torch_module", lambda: None)
    assert select_device("cuda") == "cpu"
    assert select_device("mps") == "cpu"
    assert available_devices() == ["cpu"]


def test_strict_mode_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(device_mod, "torch_module", lambda: None)
    with pytest.raises(DeviceError, match="not available"):
        select_device("cuda", strict=True)


def test_unknown_device_strings_are_rejected():
    with pytest.raises(DeviceError, match="unknown device"):
        select_device("banana")
    with pytest.raises(DeviceError, match="malformed"):
        select_device("cuda:x")


def test_environment_variable_override(monkeypatch):
    monkeypatch.setenv("STABLE_DEVICE", "cpu")
    assert select_device() == "cpu"
    monkeypatch.setenv("STABLE_DEVICE", "auto")
    assert select_device() == available_devices()[0]
    # An explicit argument still wins over the environment.
    monkeypatch.setenv("STABLE_DEVICE", "banana")
    assert select_device("cpu") == "cpu"


def test_fake_cuda_environment(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"FakeGPU{index}"

    class FakeTorch:
        __version__ = "9.9.9"
        cuda = FakeCuda()
        backends = type("B", (), {"mps": None})()

    monkeypatch.setattr(device_mod, "torch_module", lambda: FakeTorch)
    assert available_devices() == ["cuda", "cpu"]
    assert select_device() == "cuda"
    assert select_device("cuda:1") == "cuda:1"
    assert select_device("cuda:5") == "cpu"
    with pytest.raises(DeviceError, match="only 2 CUDA"):
        select_device("cuda:5", strict=True)

    info = device_info()
    assert info["torch"] == "9.9.9"
    assert info["cuda"] is True
    assert info["cuda_devices"] == ["FakeGPU0", "FakeGPU1"]
    assert info["selected"] == "cuda"


def test_device_info_shape():
    info = device_info()
    assert set(info) == {"torch", "cuda", "cuda_devices", "mps", "selected"}
    assert info["selected"] in available_devices()


def test_broken_torch_import_degrades_quietly(monkeypatch):
    class Exploding:
        @property
        def cuda(self):
            raise RuntimeError("driver on fire")

    monkeypatch.setattr(device_mod, "torch_module", lambda: Exploding())
    assert available_devices() == ["cpu"]
    assert select_device("cuda") == "cpu"
