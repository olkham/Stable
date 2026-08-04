# Contributing to Stable

The whole point of this project is that **the next SOTA tracker should be a
package you install, not a patch to this repo**. This document is mostly about
how to do exactly that. Everything else — bug fixes, solver work, UI — follows
the usual "open an issue, send a PR" path and the conventions at the bottom.

```bash
git clone https://github.com/stable-tracker/stable && cd stable
pip install -e ".[dev,video,server]"   # add [torch] for the deep trackers
python -m pytest                        # must be green before and after your change
python -m ruff check . && python -m mypy

cd web && npm install && npm run build && cd ..   # frontend, only if you touch web/
stable serve                                        # try the UI: builds if needed, opens a browser
```

`python -m pytest -m integration` on its own runs just the headless CLI
pipeline against the bundled sample clip (`stable/examples`,
`tests/test_cli_sample.py`); `-m "not integration"` runs everything else.
CI (`.github/workflows/ci.yml`) runs both, plus lint, the frontend build, and
a packaging smoke test (build a wheel, install it into a clean venv, run
`stable serve`) on Ubuntu and Windows.

---

## Adding a tracker

### The contract, in one screen

A tracker is a class with four attributes and one real method:

```python
from stable.trackers import TrackerPlugin
from stable.core.types import TrackResult

class MyTracker(TrackerPlugin):
    name = "my_tracker"                 # machine name; project files store this
    display_name = "My Tracker (Lab X)" # what the UI's picker shows
    requires_gpu = False                # True only if it *cannot* run on a CPU
    license = "Apache-2.0"              # licence of the plugin AND its weights

    def prepare(self, device: str) -> None: ...      # optional: load weights
    def release(self) -> None: ...                   # optional: free GPU memory
    @classmethod
    def is_available(cls) -> bool: ...               # optional: deps installed?

    def track(self, video, queries, frame_range, direction="bidir",
              progress=noop_progress) -> TrackResult:
        ...
```

Five rules `track()` must obey. They are not style preferences — the solver, the
merge layer and the UI all assume them, and the shared test suite checks them:

1. **Return the whole clip timeline.** `points` is `[N, T, 2]` with
   `T = video.frame_count`, indexed by *absolute* frame number. Frames outside
   `frame_range` are `NaN` / confidence `0` / `visible False`. There are no
   range offsets anywhere in this codebase; do not invent one.
2. **Preserve query order.** `result.point(i)` belongs to `queries[i]`.
3. **Normalise confidence to `[0, 1]` yourself.** See *Calibrating confidence*
   below — this is the part people get wrong.
4. **One run, no merging.** `track()` tracks from the given queries in the given
   direction. Multi-anchor merging is `stable.trackers.merge`, applied
   tracker-agnostically to every plugin. Never merge inside a plugin.
5. **Report progress.** Call `progress(fraction, message)` often enough for a
   progress bar, and never assume it does something.

Raise `stable.core.errors.TrackerError` for anything you refuse to do.

### Publishing it

Stable discovers trackers through the `stable.trackers` entry-point group. In
your own package's `pyproject.toml`:

```toml
[project.entry-points."stable.trackers"]
my_tracker = "my_package.tracker:MyTracker"
```

`pip install my-package` is the entire integration step: the tracker appears in
`stable trackers`, in the UI picker, and project files can name it. A plugin
that fails to import is logged and skipped, never fatal — so a broken
third-party tracker cannot take the app down.

### Wiring it into a project file

Nothing to do. `points[].tracker` stores your `name`; everything downstream sees
only `TrackResult`.

---

## Adding a deep-learning tracker

Neural trackers share far more plumbing than model code, so subclass
`DeepTrackerPlugin` (`stable/trackers/deep.py`) instead of `TrackerPlugin`. It
already implements windowed inference, resolution scaling, re-anchoring across
window boundaries, overlap stitching, GPU→CPU fallback, lazy checksummed weight
download and timeline assembly. You supply three things.

`stable/trackers/tapir.py` is the reference example — read it alongside this.

### 1. A `ModelSpec` — where the checkpoint comes from

```python
from stable.trackers.weights import ModelSpec, register_model

MY_WEIGHTS = register_model(ModelSpec(
    key="my_model_v2",                       # cache dir name, stable forever
    url="https://.../my_model_v2.pt",
    sha256="a1b2...64 hex chars",            # see "Pinning a checkpoint"
    license="Apache-2.0",
    non_commercial=False,                    # True => explicit opt-in to download
    notes="From the v2 release; trained on ...",
))
```

Checkpoints land in `~/.stable/models/<key>/` (override with `STABLE_MODEL_DIR`).
**Never commit weights**, and never add a model as a hard dependency.

### 2. `_build_backend(device, weights) -> TrackerBackend`

Load the network and wrap it in an object with three methods:

```python
class MyBackend:
    device: str
    def infer(self, frames, queries) -> BackendPrediction: ...
    def to_device(self, device: str) -> None: ...
    def release(self) -> None: ...
```

`infer` receives `frames` `[L, H, W, 3]` uint8 RGB **in run order** (a backward
run gets the clip reversed, so you never handle direction) and `queries`
`[N, 3]` rows of `(local_frame, x, y)` in those frames' pixel space. It returns a
`BackendPrediction` with `points` `[L, N, 2]`, `visibility` `[L, N]` in `[0, 1]`,
and an optional second `confidence` head. Return *raw* model probabilities —
calibration is the plugin's job, not the backend's.

Keep model-specific quirks (channel order, `(t, y, x)` vs `(t, x, y)`, `[-1, 1]`
normalisation, batch dimensions) inside the backend. `TapirBackend` shows all
four.

Also override `_implementation_available()` to report whether the package
holding the model code imports, so the UI can grey the tracker out instead of
failing mid-job.

### 3. `normalize_confidence(prediction) -> (confidence, visible)`

See the next section.

### Window size and input resolution

```python
default_chunk_size = 32            # frames per inference window
default_overlap = 8                # crossfaded frames between windows
default_model_resolution = (256, 256)   # (H, W) fed to the model; None = native
```

These are what make a 30 s 1080p clip fit in 8 GB: never more than one window
exists at a time, and it is downscaled first. Pick `chunk_size` so
`chunk_size x H x W` activations fit with room to spare, and an `overlap` long
enough that the model has real temporal context after re-anchoring (8 frames is
a reasonable floor). Users can override both per-instance.

---

## Calibrating confidence

Every plugin's confidence must mean the same thing, because the solver weights
tracks by it and the UI colours the timeline by it:

> **confidence ≈ the probability that this position is accurate to within a few
> pixels.** `0` = no information (occluded, lost, never tracked). `1` = as good
> as a user-placed anchor.

Recipe, using `stable.trackers.confidence`:

```python
raw = combine_scores(prediction.visibility, prediction.confidence)  # independent -> multiply
confidence = normalize_scores(raw, low=0.15, high=0.90)             # dead zones at both ends
visible = visibility_from_confidence(confidence, 0.5)
```

* **Multiply independent signals.** "Is it visible?" and "is this where it is?"
  must *both* hold, so `0.9 x 0.8 = 0.72`.
* **Cut off the bottom.** Below `low` the model is guessing; map that to exactly
  `0` so the solver drops the sample instead of half-trusting it.
* **Saturate the top.** Above `high` the differences are noise.
* Expose `low`/`high` as constructor arguments (`confidence_range=...`) so users
  can retune against their own footage.

How the shipped plugins do it:

| Plugin | Raw signal | Calibration |
|---|---|---|
| `lk` | `1 - fb_error / threshold` (forward-backward check) | linear ramp to 0 at 2 px |
| `template` | NCC peak, clamped at 0 | identity |
| `cotracker3` | `visibility x confidence` heads | `(0.10, 0.95)` |
| `tapir` | `(1 - sigmoid(occlusion)) x (1 - sigmoid(expected_dist))` | `(0.15, 0.90)` |

Sanity check your choice: on clean, unambiguous footage your plugin should score
`> 0.9`, and through a full occlusion it should approach `0`. The shared suite
asserts both.

---

## Pinning a checkpoint

Downloads are verified against `ModelSpec.sha256`, and an unpinned spec **refuses
to download** rather than silently trusting the network. To pin one:

```bash
STABLE_ALLOW_UNPINNED_WEIGHTS=1 python -c "
from stable.trackers.weights import ensure_weights
from stable.trackers.tapir import BOOTSTAPIR
print(ensure_weights(BOOTSTAPIR))"
# the digest is logged at WARNING; verify it against the publisher, then put it
# in the ModelSpec and send a PR
```

Users can also pin without editing code, via
`STABLE_MODEL_SHA256_<KEY>` (key upper-cased, non-alphanumerics as `_`).

The checkpoints shipped in this repo are currently **unpinned**: the digests
must be taken from a real download and checked against the publisher before
being written down, and we will not commit a value we have not verified. If you
download one, please send its digest.

## Licensing model weights

Stable's code is MIT. Weights are not, and the difference matters to users
shipping commercial work:

* `license` on the plugin class is the **weights'** licence, not ours. The UI
  shows it.
* Set `non_commercial=True` on the `ModelSpec` for anything CC-BY-NC or similar.
  Such weights are **never bundled** and are only downloaded after an explicit
  opt-in (`accept_noncommercial=True` or `STABLE_ACCEPT_NONCOMMERCIAL=1`).
  CoTracker3 is the current example; TAPIR (Apache-2.0) is not.
* If you cannot state the licence of a checkpoint, do not add it.

---

## Testing a new tracker

`tests/test_tracker_suite.py` is the shared contract suite — every shipped
plugin runs the *same* tests. Add your name to its `TRACKERS` list and teach
`make_plugin` how to build it; if it passes, it is a drop-in tracker.

For a deep tracker, do not require a GPU or a checkpoint in tests. Inject a fake
backend instead:

```python
MyTracker(backend=MarkerBackend(), chunk_size=6, overlap=2, model_resolution=None)
```

`MarkerBackend` (in `tests/conftest.py`) finds the synthetic clip's marker in the
pixels it is handed, so everything the *plugin* owns — window planning, frame
loading, coordinate scaling, re-anchoring, stitching, confidence normalisation,
timeline assembly — is genuinely exercised, and only the network is stubbed.
Anything that truly needs hardware should be `pytest.mark.skipif`-ed on device
availability, never left to fail on a laptop.

---

## Project conventions

* **Python 3.11+**, `ruff check` and `mypy` clean, 100-column lines.
* **Docstrings on every public contract** — modules, classes, functions,
  properties. `tests/test_public_api.py` enforces this.
* **Google-style docstrings**; say *why*, not just *what*. If a constant is a
  judgement call (a threshold, a window size), the docstring says what would
  change it.
* **Never mutate a `TrackResult`/`PointTrack`.** Raw tracker output is immutable
  by design; edits are the corrections layer. `replace()` returns a new track.
* **Absolute frame indices everywhere.** No range-relative arrays.
* **Optional dependencies stay optional.** Import `torch`, `cv2`, `av` lazily
  inside the function or method that needs them, and report a missing one
  through `is_available()` and a clear `TrackerError`.
* **Tests state the behaviour, not the implementation.** A test name should read
  as a claim about the product.
* If a change alters a contract in `PLAN.md`, update `PLAN.md` in the same PR
  and say why.
