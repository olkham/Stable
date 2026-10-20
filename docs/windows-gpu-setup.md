# GPU setup on Windows

Stable always works without a GPU — `lk` and `template` are pure OpenCV, and
the deep trackers (CoTracker3, TAPIR) fall back to CPU automatically (slow,
but correct; see PLAN.md's "degrade, don't die"). This page is for turning on
CUDA acceleration for those two, natively on Windows (no WSL, no Docker —
Stable is a local single-user tool, and running it natively is what lets
`stable serve` open your browser for you).

If you only want the classic trackers, skip straight to the main
[README](../README.md#install) — none of this is required.

## 1. Confirm your GPU and driver

You need an NVIDIA GPU with a reasonably current driver (Turing/RTX 20-series
or newer is comfortable; older cards may work but are not tested here).

Open PowerShell and run:

```powershell
nvidia-smi
```

If that prints a table with your GPU's name and a driver version, you are set.
If it errors with "command not found" or similar, install/update the driver
from <https://www.nvidia.com/download/index.aspx> (the "Game Ready" or
"Studio" driver both work — you do not need the separate CUDA Toolkit
installer; PyTorch's wheels bring their own CUDA runtime).

## 2. Install Python and Stable

Install Python 3.11 or 3.12 from <https://python.org> (the Windows Store
version works too). Then, in PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install --upgrade pip
pip install "stable-tracker[video,server]"
```

At this point `stable trackers` lists `lk`/`template`/`dummy` as available and
`cotracker3`/`tapir` as installed-but-unavailable (no PyTorch yet):

```powershell
stable trackers
```

## 3. Install the CUDA build of PyTorch

**`install.bat` / `install.sh` now do this for you.** They run
`scripts/install_gpu.py`, which reads your driver's CUDA version from
`nvidia-smi`, installs the newest matching wheel from PyTorch's own index, and
— the part that matters — *verifies* `torch.cuda.is_available()` afterwards,
falling back to an older CUDA build if the first one installs but cannot see
the card. On a machine with no NVIDIA GPU it says so and exits without
installing anything.

You can run it on its own at any time:

```powershell
python scripts/install_gpu.py            # asks first
python scripts/install_gpu.py --yes      # no prompt (for scripts)
python scripts/install_gpu.py --dry-run  # show what it would do
```

Set `STABLE_INSTALL_GPU=0` to make the installers skip it entirely, or `=1` to
accept without prompting.

### Doing it by hand

`pip install stable-tracker[torch]` deliberately does **not** pick a CUDA
build: PyPI's default `torch` wheel is CPU-only on Windows, and the right build
depends on your driver. Pick the index matching your driver's CUDA version from
<https://pytorch.org/get-started/locally/>, or start here:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

A newer driver can run an older CUDA wheel (CUDA is backward-compatible), so if
one fails to install or `torch.cuda.is_available()` comes back `False`, try an
older tag — `cu126`, `cu124`, `cu121`, `cu118` — from the same index before
assuming something else is wrong. Install `torchvision` alongside `torch` and
from the *same* index: it carries the RAFT tracker's architecture and weights,
and a torchvision built against a different CUDA than torch fails confusingly.

Verify:

```powershell
stable devices
```

```text
selected: cuda
torch:    2.4.0+cu124
cuda:     True
  - NVIDIA GeForce RTX 4080
```

If `cuda` is `False` here, PyTorch installed but cannot see your GPU — almost
always a mismatched CUDA wheel (step 3) or a driver that needs updating (step
1); it is not a Stable problem, `torch.cuda.is_available()` is exactly what
Stable calls too (`stable.core.device.select_device`).

## 4. Install the deep trackers' model code

`raft` needs nothing further — its architecture and weights ship inside
torchvision, which step 3 installed, so it becomes available as soon as CUDA
PyTorch is in place. Nor do `ocean` and `siamdw`: their network code ships with
Stable itself, and their (pinned, MIT) checkpoints download on first use.

CoTracker3 and TAPIR are different: their model code lives only on GitHub and
is not Stable's to vendor. The checkpoints download separately (next step);
this installs the network *architectures*:

```powershell
pip install "tapnet @ git+https://github.com/google-deepmind/tapnet.git"
pip install "cotracker @ git+https://github.com/facebookresearch/co-tracker.git"
```

`stable trackers` says which of these is missing and what to run — an
unavailable tracker reports the reason rather than just greying out.

This needs `git` on PATH (<https://git-scm.com/download/win>) and, for some
transitive dependencies, the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
if you see a `DLL load failed` error importing `torch` or `cv2`.

`stable trackers` should now show both as available:

```text
cotracker3  CoTracker3 (Meta)        (CC-BY-NC-4.0)
dummy       Dummy (testing)          (MIT)
lk          Lucas-Kanade (optical flow)  (MIT)
tapir       TAPIR / BootsTAP (DeepMind)  (Apache-2.0)
template    Template Matching (NCC)  (MIT)
```

## 5. Download the checkpoints

Weights are never bundled and download lazily on first use — either let
`stable serve`'s model manager (the "Models" button) do it with a couple of
clicks, or from the command line:

```powershell
stable models download bootstapir                       # Apache-2.0, no opt-in needed
stable models download cotracker3_offline --accept-noncommercial
```

Both checkpoints now ship **pinned** (a verified sha256 is checked into
`stable/trackers/tapir.py` / `cotracker3.py` — see
[CONTRIBUTING.md](../CONTRIBUTING.md#pinning-a-checkpoint) for how and when
they were verified), so a plain download just works and is checked against
that digest; `--allow-unpinned` / `STABLE_ALLOW_UNPINNED_WEIGHTS=1` is only
needed if you point a `ModelSpec` at a *different* file (a mirror, a newer
release) that has no pinned digest yet.

CoTracker3's weights are additionally CC-BY-NC (non-commercial); the
`--accept-noncommercial` flag above (or `STABLE_ACCEPT_NONCOMMERCIAL=1`, or
the matching checkbox in the model manager) is that licence's explicit
opt-in. This is a deliberate fail-closed design, not a bug to route around —
see `stable/trackers/weights.py`'s module docstring.

## 6. Run it

```powershell
stable serve
```

opens your browser at the local UI. Windows Defender / Firewall may prompt to
allow Python to accept connections on the local network the first time — this
is `stable serve` binding `127.0.0.1`; you can allow it for private networks
only, or deny the public-network prompt, since nothing here needs to be
reachable off your machine.

## Troubleshooting

- **`stable devices` shows `cuda: False` after installing torch.** Re-check
  steps 1 and 3 — this is a PyTorch/driver mismatch, not something Stable
  controls. `pip uninstall torch` and reinstall from the exact index URL for
  your driver's CUDA version fixes it in almost every case.
- **`CUDA out of memory` while tracking.** The deep trackers process the clip
  in windows (`default_chunk_size` frames at a time, downscaled to
  `default_model_resolution`) specifically so this should not happen on an
  8 GB card at default settings; if it still does, pass a smaller
  `chunk_size` when constructing the tracker (see CONTRIBUTING.md), or it
  will fall back to CPU automatically and finish (just slowly) rather than
  fail the job.
- **Long path errors installing dependencies.** Enable Windows' long path
  support once, as Administrator:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`,
  then restart the terminal.
- **Prefer a container instead?** See the CUDA `Dockerfile` at the repo root —
  it targets headless Linux GPU boxes; on Windows itself, running natively as
  above is simpler and is what lets `stable serve` reach your browser.
