"""The installer's GPU step: which CUDA wheel, and did it actually work.

This is the part of ``install.bat`` / ``install.sh`` that could not be tested
while it was shell — which is exactly why it is Python. The expensive half (a
2 GB pip download) is never run here; what is tested is the decision that
precedes it and the verification that follows it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import install_gpu  # noqa: E402

SMI_OUTPUT = """\
Tue Aug  5 09:14:22 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 551.86       Driver Version: 551.86       CUDA Version: 12.4      |
|-------------------------------+----------------------+----------------------+
"""

# Drivers from the 6xx series word it differently, and reading only the older
# spelling reported "no NVIDIA driver detected" on a machine with a TITAN RTX
# in it. Copied verbatim from `nvidia-smi` on that machine.
SMI_OUTPUT_UMD = """\
Wed Aug  5 13:49:26 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.88                 KMD Version: 610.88        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA TITAN RTX             WDDM  |   00000000:01:00.0  On |                  N/A |
"""


def test_the_driver_cuda_version_is_read_from_nvidia_smi():
    assert install_gpu.driver_cuda_version(runner=lambda _args: SMI_OUTPUT) == (12, 4)


def test_the_newer_cuda_umd_version_spelling_is_understood():
    assert install_gpu.driver_cuda_version(runner=lambda _args: SMI_OUTPUT_UMD) == (13, 3)


def test_no_driver_is_not_an_error():
    """A machine with no NVIDIA card is a supported configuration, not a failure."""

    def explode(_args):
        raise FileNotFoundError("nvidia-smi")

    assert install_gpu.driver_cuda_version(runner=explode) is None
    assert install_gpu.driver_cuda_version(runner=lambda _a: "not an smi table") is None


def test_only_wheels_the_driver_can_run_are_offered():
    """CUDA is backward-compatible, so the driver's version is a ceiling."""
    assert install_gpu.candidate_tags((12, 4)) == ["cu124", "cu121", "cu118"]
    assert install_gpu.candidate_tags((12, 1)) == ["cu121", "cu118"]
    assert install_gpu.candidate_tags((11, 8)) == ["cu118"]


def test_a_new_driver_is_offered_the_new_wheels_first(fake_pip=None):
    """The reported machine: driver CUDA 13.3, so cu130 leads and the rest follow."""
    assert install_gpu.candidate_tags((13, 3))[0] == "cu130"
    assert install_gpu.candidate_tags((13, 3))[-1] == "cu118"


def test_a_driver_too_old_for_any_wheel_still_gets_one_to_try():
    """Better to try the oldest and fail loudly than to silently do nothing."""
    assert install_gpu.candidate_tags((10, 2)) == ["cu118"]


def test_an_unknown_driver_version_tries_everything():
    assert install_gpu.candidate_tags(None) == [tag for _v, tag in install_gpu.CUDA_INDEXES]


def test_index_urls_point_at_pytorch():
    assert install_gpu.index_url("cu124") == "https://download.pytorch.org/whl/cu124"


def test_torchvision_is_installed_alongside_torch():
    """RAFT (todo 28) lives in torchvision, and a mismatched pair is a classic break."""
    assert install_gpu.PACKAGES == ("torch", "torchvision")


# -- the install loop ---------------------------------------------------------


@pytest.fixture
def fake_pip(monkeypatch):
    """Record pip invocations and let a test choose which wheel 'works'."""
    calls: list[list[str]] = []
    state = {"working_tag": None}

    def fake_run(args, **_kwargs):
        calls.append(args)

        class Result:
            returncode = 0

        return Result()

    def fake_status(_python=None):
        installed = [c for c in calls if "install" in c]
        if not installed:
            return install_gpu.TorchStatus(installed=False)
        url = installed[-1][-1]
        tag = url.rsplit("/", 1)[-1]
        works = tag == state["working_tag"]
        return install_gpu.TorchStatus(
            installed=True,
            version=f"2.4.0+{tag}",
            cuda=works,
            devices=("NVIDIA TITAN RTX",) if works else (),
        )

    monkeypatch.setattr(install_gpu.subprocess, "run", fake_run)
    monkeypatch.setattr(install_gpu, "torch_status", fake_status)
    return calls, state


def test_the_first_working_wheel_wins_and_the_rest_are_not_tried(fake_pip):
    calls, state = fake_pip
    state["working_tag"] = "cu124"

    status = install_gpu.install_cuda_torch("py", ["cu126", "cu124", "cu121"])

    assert status.cuda is True
    assert status.devices == ("NVIDIA TITAN RTX",)
    installs = [c[-1] for c in calls if "install" in c]
    assert installs == [
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu124",
    ]


def test_a_wheel_that_installs_but_sees_no_gpu_is_a_failure(fake_pip):
    """The common real-world break: pip succeeds, cuda.is_available() is False."""
    calls, state = fake_pip
    state["working_tag"] = None  # no wheel ever works

    status = install_gpu.install_cuda_torch("py", ["cu124", "cu118"])

    assert status.cuda is False
    # Both were tried, and each dead build was removed before the next attempt.
    assert len([c for c in calls if "install" in c]) == 2
    assert len([c for c in calls if "uninstall" in c]) == 2


def test_an_existing_cpu_build_is_removed_first(fake_pip):
    """pip's ordering of 2.4.0+cpu against 2.4.0+cu124 is not worth relying on."""
    calls, state = fake_pip
    state["working_tag"] = "cu124"

    install_gpu.install_cuda_torch("py", ["cu124"], had_torch=True)

    assert "uninstall" in calls[0]
    assert calls[0][-2:] == ["torch", "torchvision"]


# -- main() ------------------------------------------------------------------


def test_a_machine_with_a_working_gpu_is_left_alone(monkeypatch, capsys):
    monkeypatch.setattr(
        install_gpu,
        "torch_status",
        lambda _p=None: install_gpu.TorchStatus(
            installed=True, version="2.4.0+cu124", cuda=True, devices=("NVIDIA TITAN RTX",)
        ),
    )
    assert install_gpu.main(["--yes"]) == 0
    out = capsys.readouterr().out
    assert "Nothing to do" in out
    assert "NVIDIA TITAN RTX" in out


def test_a_machine_with_no_gpu_exits_cleanly(monkeypatch, capsys):
    """An optional accelerator must never fail an install."""
    monkeypatch.setattr(
        install_gpu, "torch_status", lambda _p=None: install_gpu.TorchStatus(installed=False)
    )
    monkeypatch.setattr(install_gpu, "driver_cuda_version", lambda: None)

    assert install_gpu.main(["--yes"]) == 0
    assert "No NVIDIA driver detected" in capsys.readouterr().out


def test_skip_does_nothing(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(install_gpu, "torch_status", lambda _p=None: called.append(1))
    assert install_gpu.main(["--skip"]) == 0
    assert called == []


def test_the_environment_variable_can_turn_it_off(monkeypatch, capsys):
    monkeypatch.setenv("STABLE_INSTALL_GPU", "0")
    called = []
    monkeypatch.setattr(install_gpu, "torch_status", lambda _p=None: called.append(1))
    assert install_gpu.main([]) == 0
    assert called == []


def test_dry_run_prints_the_commands_without_installing(monkeypatch, capsys):
    monkeypatch.setattr(
        install_gpu, "torch_status", lambda _p=None: install_gpu.TorchStatus(installed=False)
    )
    monkeypatch.setattr(install_gpu, "driver_cuda_version", lambda: (12, 4))
    ran = []
    monkeypatch.setattr(install_gpu.subprocess, "run", lambda *a, **k: ran.append(a))

    assert install_gpu.main(["--yes", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "download.pytorch.org/whl/cu124" in out
    assert ran == []
