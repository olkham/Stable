"""Install the CUDA build of PyTorch into this environment, if a GPU is here.

``install.bat`` / ``install.sh`` call this after the main install. It is a
separate Python script rather than more shell because the interesting part is a
decision — *which* CUDA wheel index matches this driver — and a decision worth
getting right is worth testing, which a batch file is not.

Why this is not just ``pip install stable-tracker[torch]``
----------------------------------------------------------
PyPI's default ``torch`` wheel is **CPU-only on Windows**. Installing the
``torch`` extra therefore produces a working install in which the GPU is never
touched, which is exactly the state todo 6 complains about ("0% of the GPU").
The CUDA builds live on PyTorch's own index, one URL per CUDA version, and the
right one depends on the driver — so the extra deliberately does not pick, and
this script does.

What it decides
---------------
``nvidia-smi`` reports the highest CUDA version the *driver* supports. CUDA is
backward-compatible — a newer driver runs older wheels — so that number is a
ceiling, not a requirement, and the candidate list is "the best match, then
progressively older" rather than a single guess.

The check that matters is not "did pip succeed" but
``torch.cuda.is_available()``: a mismatched wheel installs perfectly happily and
then reports no GPU, which is the single most common way this goes wrong (see
docs/windows-gpu-setup.md's troubleshooting). So each candidate is installed and
then *verified*, and a wheel that installs but cannot see the card is treated as
a failure and the next one is tried.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

__all__ = [
    "CUDA_INDEXES",
    "driver_cuda_version",
    "candidate_tags",
    "index_url",
    "torch_status",
    "main",
]

CUDA_INDEXES: tuple[tuple[tuple[int, int], str], ...] = (
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
)
"""Known PyTorch wheel indexes, newest first, with the CUDA version each needs.

Optimistic at the top on purpose. A tag PyTorch has not published yet simply
404s, pip fails, and :func:`install_cuda_torch` moves to the next one — so
listing a newer index costs one failed request, while omitting it costs the
user the faster build for as long as this table is stale. The old entries are
the safety net and have been published for years.
"""

INDEX_TEMPLATE = "https://download.pytorch.org/whl/{tag}"

# torchvision comes along because it carries the RAFT architecture and weights
# the `raft` tracker needs, and because a torchvision built against a different
# CUDA than torch is a classic and confusing breakage.
PACKAGES = ("torch", "torchvision")


@dataclass(frozen=True)
class TorchStatus:
    """What PyTorch currently is in this environment.

    Args:
        installed: Whether ``import torch`` works at all.
        version: The reported version string, e.g. ``2.4.0+cu124``.
        cuda: Whether ``torch.cuda.is_available()`` is true.
        devices: Names of the visible CUDA devices.
    """

    installed: bool
    version: str | None = None
    cuda: bool = False
    devices: tuple[str, ...] = ()


def driver_cuda_version(runner: object = None) -> tuple[int, int] | None:
    """Return the highest CUDA version this machine's driver supports.

    Args:
        runner: Injection seam for tests; a callable taking the ``nvidia-smi``
            argument list and returning its stdout, or raising.

    Returns:
        ``(major, minor)`` parsed from ``nvidia-smi``'s header, or ``None`` when
        there is no NVIDIA driver (or ``nvidia-smi`` is not on PATH, which on a
        machine with a working driver does not happen).
    """
    if runner is None:
        if shutil.which("nvidia-smi") is None:
            return None

        def runner(args: list[str]) -> str:  # type: ignore[misc]
            return subprocess.run(
                args, capture_output=True, text=True, timeout=30, check=True
            ).stdout

    try:
        output = runner(["nvidia-smi"])  # type: ignore[operator]
    except Exception:
        return None
    # Two spellings in the wild. Older drivers print "CUDA Version: 12.4";
    # drivers from the 6xx series print "CUDA UMD Version: 13.3" instead, and
    # matching only the first is why this returned "no NVIDIA driver detected"
    # on a machine with a TITAN RTX sitting in it.
    match = re.search(r"CUDA(?:\s+[A-Z]+)?\s+Version:\s*(\d+)\.(\d+)", output or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def candidate_tags(cuda: tuple[int, int] | None) -> list[str]:
    """Return the wheel tags to try, best match first.

    Args:
        cuda: The driver's supported CUDA version, or ``None`` if unknown.

    Returns:
        Tags from :data:`CUDA_INDEXES` the driver can run, newest first. An
        unknown driver version gets the whole list rather than nothing: trying
        is cheap next to leaving a GPU unused, and each attempt is verified.
    """
    if cuda is None:
        return [tag for _needed, tag in CUDA_INDEXES]
    return [tag for needed, tag in CUDA_INDEXES if needed <= cuda] or [CUDA_INDEXES[-1][1]]


def index_url(tag: str) -> str:
    """Return the PyTorch wheel index URL for a CUDA tag such as ``cu124``."""
    return INDEX_TEMPLATE.format(tag=tag)


def torch_status(python: str | None = None) -> TorchStatus:
    """Report what PyTorch is in the target environment.

    Run as a subprocess rather than by importing: this script may be executing
    under a different interpreter than the one being installed into, and
    importing torch into this process would pin the DLLs it is about to replace.

    Args:
        python: Interpreter to inspect; defaults to the current one.

    Returns:
        The current :class:`TorchStatus`.
    """
    probe = (
        "import json,torch;"
        "print(json.dumps({'v':torch.__version__,'c':torch.cuda.is_available(),"
        "'d':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
    )
    try:
        out = subprocess.run(
            [python or sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception:
        return TorchStatus(installed=False)
    if out.returncode != 0:
        return TorchStatus(installed=False)
    try:
        payload = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return TorchStatus(installed=False)
    return TorchStatus(
        installed=True,
        version=str(payload.get("v")),
        cuda=bool(payload.get("c")),
        devices=tuple(str(d) for d in payload.get("d") or ()),
    )


def _run(args: list[str], *, dry_run: bool) -> int:
    """Echo and run a command, returning its exit code (0 when dry-running)."""
    print("  $ " + " ".join(args), flush=True)
    if dry_run:
        return 0
    return subprocess.run(args).returncode


def _uninstall_cpu_torch(python: str, *, dry_run: bool) -> None:
    """Remove an existing CPU build so the CUDA wheel is not skipped as satisfied.

    pip compares ``2.4.0+cpu`` and ``2.4.0+cu124`` by their local version
    segments, which is not an ordering anyone should rely on for a 2 GB
    download. Uninstalling first makes the outcome the same either way.
    """
    _run([python, "-m", "pip", "uninstall", "-y", *PACKAGES], dry_run=dry_run)


def install_cuda_torch(
    python: str, tags: list[str], *, dry_run: bool = False, had_torch: bool = False
) -> TorchStatus:
    """Install and verify the first CUDA wheel set that can actually see the GPU.

    Args:
        python: Interpreter to install into.
        tags: CUDA tags to try, in order.
        dry_run: Print the commands without running them.
        had_torch: Whether a torch build is already installed and must be
            removed first.

    Returns:
        The :class:`TorchStatus` after the attempt — check ``.cuda``.
    """
    if had_torch:
        _uninstall_cpu_torch(python, dry_run=dry_run)

    for tag in tags:
        url = index_url(tag)
        print(f"\nTrying the {tag} build ...", flush=True)
        code = _run(
            [python, "-m", "pip", "install", *PACKAGES, "--index-url", url],
            dry_run=dry_run,
        )
        if dry_run:
            continue
        if code != 0:
            print(f"  {tag}: pip could not install this build; trying an older one.")
            continue
        status = torch_status(python)
        if status.cuda:
            return status
        print(
            f"  {tag}: installed, but torch.cuda.is_available() is False "
            "— wrong CUDA for this driver; trying an older build."
        )
        _uninstall_cpu_torch(python, dry_run=dry_run)

    return TorchStatus(installed=False) if not dry_run else TorchStatus(installed=True)


def _ask(question: str, *, default: bool) -> bool:
    """Ask a yes/no question, defaulting when the answer is empty or unreadable."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer.startswith("y")


def main(argv: list[str] | None = None) -> int:
    """Detect the GPU, install the matching CUDA PyTorch, and report the outcome.

    Returns:
        ``0`` whether or not a GPU was found — this is an optional accelerator,
        and an installer must not fail because a machine has no NVIDIA card.
        A non-zero code is reserved for "you asked for GPU support explicitly
        and it could not be delivered".
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter to install into")
    parser.add_argument("--yes", action="store_true", help="do not ask; install if a GPU is found")
    parser.add_argument("--skip", action="store_true", help="do nothing (for scripted installs)")
    parser.add_argument("--index-url", help="use this wheel index instead of choosing one")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running")
    args = parser.parse_args(argv)

    if args.skip or os.environ.get("STABLE_INSTALL_GPU") == "0":
        print("Skipping GPU setup (asked not to).")
        return 0

    print("\n=== GPU acceleration ===")
    current = torch_status(args.python)
    if current.cuda:
        print(f"PyTorch {current.version} already sees your GPU:")
        for device in current.devices:
            print(f"  - {device}")
        print("Nothing to do.")
        return 0

    cuda = driver_cuda_version()
    if cuda is None:
        print(
            "No NVIDIA driver detected (nvidia-smi did not report a CUDA version).\n"
            "Stable runs fine without one: lk, template, correlation, kalman,\n"
            "particle, meanshift and klt are all CPU trackers. Only the deep\n"
            "trackers (CoTracker3, TAPIR, RAFT) use a GPU."
        )
        return 0

    print(f"NVIDIA driver supports up to CUDA {cuda[0]}.{cuda[1]}.")
    if current.installed:
        print(f"PyTorch {current.version} is installed but cannot see the GPU; replacing it.")

    forced = args.index_url
    tags = ["(explicit index)"] if forced else candidate_tags(cuda)
    if not forced:
        print("Will try, in order: " + ", ".join(tags))
    print("This downloads roughly 2-3 GB.")

    interactive = sys.stdin is not None and sys.stdin.isatty()
    approved = (
        True
        if (args.yes or os.environ.get("STABLE_INSTALL_GPU") == "1" or not interactive)
        else _ask("Install the CUDA build of PyTorch now?", default=True)
    )
    if not approved:
        print("Skipped. You can run this later:  python scripts/install_gpu.py")
        return 0

    if forced:
        code = _run(
            [args.python, "-m", "pip", "install", *PACKAGES, "--index-url", forced],
            dry_run=args.dry_run,
        )
        status = TorchStatus(installed=True) if args.dry_run else torch_status(args.python)
        if code != 0:
            print("\nThat index did not install; see the pip output above.")
            return 1
    else:
        status = install_cuda_torch(
            args.python, tags, dry_run=args.dry_run, had_torch=current.installed
        )

    if args.dry_run:
        print("\n(dry run: nothing was installed)")
        return 0

    if status.cuda:
        print(f"\nPyTorch {status.version} is installed and can see:")
        for device in status.devices:
            print(f"  - {device}")
        print("\nCheck any time with:  stable devices")
        return 0

    print(
        "\nCould not get a CUDA build working on this driver.\n"
        "Stable still works on the CPU. See docs/windows-gpu-setup.md — the\n"
        "usual fix is a driver update, or pointing this script at a specific\n"
        "index:  python scripts/install_gpu.py --index-url "
        f"{index_url(CUDA_INDEXES[-1][1])}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
