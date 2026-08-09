# Stable — Open-Source Point Tracking & 2D Stabilization

**Goal:** A free alternative to Resolve's point tracker / Studio's IntelliTrack. Track user-defined points with swappable SOTA deep-learning trackers, solve X/Y/Rotation/Scale motion, let the user edit tracks and smooth curves, and export tracking data or a stabilized video. 2D similarity only — no planar/3D.

**This file is the execution handover.** Each phase below is self-contained: give a phase (plus the Architecture + Data Model sections) to the named agent and it can build it. Execute phases in order; each has acceptance criteria.

---

## Stack (decided — do not relitigate)

- **Backend:** Python 3.11+, FastAPI + WebSockets (progress streaming), single local process the user runs (`stable serve`). PyTorch for DL trackers, CUDA when available, CPU fallback always.
- **Video I/O:** PyAV (FFmpeg) decode; on-disk frame/proxy cache; FFmpeg encode with audio passthrough.
- **Frontend:** React + TypeScript + Vite. Canvas overlay on video frames served as JPEG proxies over HTTP; no WebRTC.
- **Classic CV:** OpenCV (pyramidal Lucas-Kanade, NCC template matching) — the no-GPU baseline.
- **Packaging:** pip package + Dockerfile (CUDA); models downloaded on demand to `~/.stable/models`.
- **License:** MIT for our code. Note: CoTracker weights are non-commercial (CC-BY-NC); TAPIR is Apache-2.0. Plugins declare their license; NC models are opt-in downloads, never bundled.

## Architecture

```
stable/
  core/        # data model, project file, geometry types
  video/       # decode, frame cache, proxy generation, encode
  trackers/    # plugin registry + implementations (lk, template, cotracker3, tapir)
  solver/      # tracks -> per-frame similarity (X,Y,R,S), RANSAC outlier rejection
  smoothing/   # filters over X,Y,R,S curves
  render/      # inverse-warp stabilization, border policy, export
  server/      # FastAPI app, job queue, websocket progress
  cli/         # headless: track / solve / stabilize / export
web/           # React frontend
```

### Tracker plugin contract (the swappability core)

```python
@dataclass
class TrackResult:
    points: np.ndarray      # [N, T, 2] float32, NaN where invalid
    confidence: np.ndarray  # [N, T] float32 in [0,1], normalized per-plugin
    visible: np.ndarray     # [N, T] bool

class TrackerPlugin(ABC):
    name: str; display_name: str; requires_gpu: bool; license: str
    def prepare(self, device: str) -> None: ...          # load weights
    def track(self, video: VideoSource, queries: list[Query],  # Query = (frame, x, y)
              frame_range: tuple[int, int], direction: Literal["fwd","bwd","bidir"],
              progress: Callback) -> TrackResult: ...
```

Registry via entry points (`stable.trackers` group) so third-party trackers install as pip packages. Everything downstream (solver, UI, editing) sees only `TrackResult` — this is what makes swapping SOTA models trivial.

### Multi-anchor tracking (key robustness feature)

Unlike trackers that take one template/initial position, a point's `queries` list holds **several user-placed anchors** across the clip — positions known to be true. *Anchor* is the one word for this, everywhere: the type is `core.types.Query` because schema v1 stores the list under `points[].queries`, but nothing the operator reads calls it anything else (todo 6, seventh round — the UI used to say "keyframe" as well, for the same object, and nobody could tell whether that was a second concept). Tracking runs bidirectionally from every anchor; between adjacent anchors the overlapping forward/backward runs are blended by confidence with distance-to-anchor falloff, under the hard constraint that the merged track passes through every anchor exactly. Where overlapping runs disagree, merged confidence drops — surfacing drift in the UI so the user adds another anchor there. Merge logic is tracker-agnostic in `trackers/merge.py`; plugins only ever do single-run tracking.

### Motion solver

Per frame, from all enabled point tracks vs. a reference frame: confidence-weighted similarity fit (Umeyama) with RANSAC when ≥3 points. 1 point → X,Y only; 2+ points → X,Y,R,S. Output: four curves `X(t), Y(t), R(t), S(t)` + per-frame residual (solve quality). Stabilization warp = `smoothed(t) ∘ raw(t)⁻¹` applied as a single affine per frame.

### Manual edits are non-destructive

Raw tracker output is immutable. User edits are a correction layer: keyframed offsets blended over a falloff range, hard replacements, interpolated gap fills, or "retrack from frame N with corrected position as new query". Undo = remove correction. This is a hard requirement — design everything in Phase 0 around it.

## Data model (project file, JSON, versioned)

```jsonc
{ "version": 1,
  "video": { "path": "...", "fps": 0, "frames": 0, "w": 0, "h": 0 },
  "points": [ { "id": "p1", "queries": [{"frame": 0, "x": 0, "y": 0}],
                "tracker": "cotracker3", "enabled_for_solve": true,
                "raw": "tracks/p1.npz",                  // points/conf/visible
                "corrections": [ /* keyframed edits, typed */ ] } ],
  "solve": { "reference_frame": 0, "channels": {"x":true,"y":true,"r":true,"s":true} },
  "smoothing": [ { "channel": "x", "filter": "butterworth", "params": {"cutoff_hz": 1.0} } ],
  "render": { "border": "auto_zoom", "codec": "h264", "crf": 18 } }
```

Exports: CSV and JSON of raw/corrected tracks and solved+smoothed X,Y,R,S curves.

---

## Phases

| # | Phase | Agent | Why this agent |
|---|-------|-------|----------------|
| 0 | Foundations & contracts | **Opus** | Interfaces here lock in everything; wrong choices are expensive |
| 1 | Video I/O + classic trackers + CLI | **Sonnet** | Well-specified plumbing against fixed contracts |
| 2 | SOTA tracker plugins (CoTracker3, TAPIR) | **Opus** | Model integration, chunked inference, confidence calibration are subtle |
| 3 | Solver + smoothing + stabilized render | **Opus** | The math core; robustness = the product's whole value |
| 4 | Web UI core | **Sonnet** | Large but conventional React/FastAPI build against a fixed API |
| 5 | Manual track editing + curve editor | **Opus** | Edit semantics and interaction model are the hard design problem |
| 6 | Packaging, tests, docs, model manager | **Sonnet** | Mechanical, well-specified |

### Phase 0 — Foundations (Opus) — **done**
Repo scaffold per Architecture above; `core` types (Query, TrackResult, Correction, SimilarityTransform); tracker plugin registry with entry-point discovery; project file schema v1 with load/save/migrate; device selection util (`cuda`→`cpu`). No UI, no real trackers — one `dummy` tracker for tests.
**Accept:** `pip install -e .` works; unit tests for schema round-trip and registry discovery pass; docstrings on every public contract.

**Contract decisions locked in Phase 0** (later phases must not relitigate these):
- `TrackResult` axis `T` is the **whole clip** (`T = video.frame_count`) indexed by absolute frame; frames outside a run's range are `NaN` / conf 0 / not visible — removes every range/offset off-by-one between plugins, solver and UI.
- Added `PointTrack` (one point's `[T,2]`/`[T]`/`[T]` slice) alongside `TrackResult`: per-point `.npz` storage, corrections and merging are all per-point, so the plural type alone was the wrong granularity.
- `TrackResult`/`PointTrack` copy their inputs and mark the arrays read-only — makes "raw tracker output is immutable" enforced, not just documented.
- `SimilarityTransform` rotation is **radians**, rotation/scale about the origin (`about_pivot()` folds a pivot into the translation); one unit everywhere, degrees are display-only.
- `TrackerPlugin` gained non-abstract `release()`, `is_available()`, `is_prepared` (additive: lets the UI grey out unusable trackers and free GPU memory without a contract change).
- `VideoSource` is a Protocol in `core/video.py` (plus a dependency-free `SyntheticVideoSource`) so trackers never import the decode stack; Phase 1's PyAV class implements it.
- `trackers/merge.py` API landed early with `plan_anchor_runs` + validation + report + the unambiguous single-run/anchor-passthrough merge; only the overlapping-run blend raises `NotImplementedError` for Phase 2 (**implemented in Phase 2**). Planning emits 2 runs per inter-anchor span, so cost stays ~2x clip length regardless of anchor count.
- `apply_corrections(track, corrections)` is defined and pure but raises for enabled corrections until Phase 5 (**implemented in Phase 5**); the five correction types are typed and serialised now so project files are stable.

### Phase 1 — Video + classic trackers + CLI (Sonnet) — **done**
PyAV `VideoSource` (random access, frame cache, half/quarter-res proxy generation); FFmpeg export util with audio passthrough; `lk` (pyramidal LK with forward-backward error → confidence) and `template` (NCC, adaptive template update) plugins; CLI: `stable track <video> --point x,y@frame --tracker lk --out tracks.json`.
**Accept:** track a point through a real test clip end-to-end from CLI; forward-backward error drops confidence on occlusion; proxy scrub of a 1080p clip is I/O-bound, not decode-bound.

### Phase 2 — SOTA trackers (Opus) — **done**
CoTracker3 and TAPIR (BootsTAP weights) as plugins; windowed/chunked inference for long clips with overlap stitching; GPU with automatic CPU fallback; per-plugin confidence normalized to [0,1] so the solver treats all trackers uniformly; lazy model download with checksum. Finish `trackers/merge.py` multi-anchor merging — the API, run planner, report and anchor passthrough exist from Phase 0; what remains is the confidence-weighted blend with anchor falloff and the disagreement → lowered-confidence rule for frames covered by two runs (currently `NotImplementedError`) — applied uniformly to all plugins including Phase 1's. Document how to add the next SOTA model in `CONTRIBUTING.md` (this doc is a deliverable — it's the project's stated purpose).
**Accept:** both plugins pass the same integration suite as `lk`; a 30 s 1080p clip tracks without OOM on an 8 GB GPU; CPU fallback completes (slow is fine); a deliberately drifting track is fixed by adding one mid-clip anchor — merged path passes through all anchors, confidence dips where runs disagreed.

**Contract additions in Phase 2** (all additive; no Phase 0 contract was changed):
- `DeepTrackerPlugin` (`trackers/deep.py`) implements the whole chunked pipeline once — window planning, model-resolution scaling, re-anchoring, stitching, GPU→CPU fallback, timeline assembly — so a model plugin supplies only a `ModelSpec`, `_build_backend()` and `normalize_confidence()`.
- `TrackerBackend` is a NumPy-in/NumPy-out seam around the network: torch never enters the pipeline code, and injecting a fake backend makes the whole path testable with no GPU, no torch and no weights.
- Chunking (`trackers/chunking.py`) works in **run-local index space** (0 = the anchor), so forward and backward runs share one implementation and the module stays dependency-free.
- Confidence is defined project-wide as *P(this position is accurate to within a few pixels)*; `trackers/confidence.py` holds the shared helpers and each plugin declares a `(low, high)` calibration. Documented in CONTRIBUTING.md.
- Model weights live in `trackers/weights.py` (not a new top-level package, to keep the Architecture tree as specified); `WeightsError` added to `core/errors.py`.
- **Deviation:** the two shipped `ModelSpec`s ship with `sha256=None` (unpinned) — a digest that has not been verified against a real download would be a lie, so downloads instead *fail closed* and require an explicit `STABLE_ALLOW_UNPINNED_WEIGHTS=1`; pinning instructions are in CONTRIBUTING.md.
- Two Phase 0 tests asserting `merge_anchor_runs` raises `NotImplementedError` on overlap were replaced by tests of the blend it now performs, and `test_registry` stopped using `"cotracker3"` as its example of an unknown tracker name.

### Phase 3 — Solver + smoothing + render (Opus) — **done**
Weighted Umeyama similarity solve + RANSAC (≥3 pts) with confidence gating; degrade gracefully (1 pt → XY; low-confidence frames → interpolate + flag); smoothing library over each channel: moving average, Gaussian, Butterworth (zero-phase filtfilt), Savitzky–Golay, spline fit, and **tripod mode** (lock to reference frame); per-channel enable + strength; stabilization renderer: per-frame inverse similarity warp (OpenCV, GPU path optional), border policies = crop / auto-zoom (minimal zoom covering all frames) / transparent pad; CLI `stable stabilize`.
**Accept:** synthetic test — a clip warped by known X,Y,R,S recovers the ground-truth motion within tolerance; stabilized real handheld clip shows no border gaps in auto-zoom; audio survives export.

**Contract additions in Phase 3** (all additive; no Phase 0 or 2 contract was changed):
- Solved curves are reported **about a pivot** (frame centre by default): `x`/`y` are the pivot's displacement, `r`/`s` unchanged. `SimilarityTransform` stays origin-based and `MotionSolve.transform_at()` folds the pivot back in via the existing `about_pivot()`. Rationale: origin-based `tx`/`ty` make a centre rotation read as hundreds of pixels of translation, correlated with `r`, so independently smoothing the four curves fights itself and the curve editor shows numbers no operator can interpret.
- The `r` curve is **unwrapped** across the clip, so a shot crossing ±π is a continuous curve a filter can work on and a gap across the seam does not interpolate the long way round.
- `SolveStatus` (`SOLVED`/`TRANSLATION_ONLY`/`INTERPOLATED`/`HELD`/`IDENTITY`) + `MotionSolve.flagged` are the solve's equivalent of the merge report's confidence dip — the per-frame "go fix this" signal Phase 4's UI turns into a heat strip. Only an unusable *reference frame* raises; every other shortfall degrades and is recorded.
- `solve.channels` splits by kind: `r`/`s` off **constrains the fit** (a genuine rigid/scale-locked solve, not a full solve with the channel discarded); `x`/`y` off zeroes that pivot-relative curve. Documented in `solver/similarity.py`.
- RANSAC requires consensus **beyond its own minimal sample** (≥3 of ≥3 points); a two-point "consensus" is no evidence, since a minimal sample always fits itself. Failing that, the plain weighted fit over every point is returned and the residual flags the frame.
- Smoothing `strength` is a universal parameter of every filter (`params["strength"]`), so per-channel enable + strength round-trip through the locked v1 schema with no invented keys. `SmoothingSettings`/`ChannelSmoothing` are only a grouped in-memory *view* of the flat `SmoothingSpec` list.
- **Deviation:** the DSP (Butterworth design, `sosfiltfilt`, Savitzky–Golay, Reinsch smoothing spline, banded Cholesky) is implemented in pure NumPy in `smoothing/dsp.py` rather than taking a SciPy dependency — numpy is the project's only hard dep and a stabiliser that won't start without a 40 MB scientific stack is a worse product. Tests assert exact agreement with SciPy wherever it happens to be installed.
- Border policies read as: `crop` = the common area at its own (smaller, even) size, no scaling; `auto_zoom` = that area scaled back to delivery size; `transparent` = full frame, RGBA. The crop/zoom is computed exactly (closed-form largest centred rectangle inside every warped quad), so "minimal zoom, no gaps" is a guarantee rather than a search.
- **Deviation:** `transparent` into a video file **fails closed** with an explanatory error instead of silently flattening the alpha — no codec this project encodes to carries one. An extension-less/directory output writes a numbered PNG sequence instead.
- `stable solve` ships alongside `stable stabilize` (both were already promised by `cli/__init__`), giving the CSV/JSON curve exports the Data-model section requires. `stable filters` prints the filter catalogue.
- `SolverError`, `SmoothingError` and `RenderError` added to `core/errors.py`.
- `coverage_scale` snaps to exactly 1.0 within 1e-9 so a shot that needed no crop does not pick up a 1.0000000000000004 zoom and resample every frame for nothing.
- `render.rescale_transform(warp, scale)` re-expresses a full-resolution warp for a proxy, so Phase 4's stabilised preview warps half-res frames without its own copy of the geometry.
- **Phase 4 integration note (actioned in Phase 5):** `server/{solve,smoothing,render}_stub.py` predate this phase. The real replacements are `solver.solve_motion` + `SolveConfig`, `smoothing.smooth_motion` + `SmoothingSettings.from_specs`, and `render.StabilizationPlan`/`warp_frame`. One behavioural difference to watch: the stub's `curve_transform` reads curves as origin-based (`x` → `tx`), while real curves are pivot-relative — use `MotionSolve.transform_at()` / `SmoothedMotion.stabilization_transform()` rather than rebuilding transforms from raw channel values. **Done: the stubs are deleted and `server/motion.py` holds the glue; see the Phase 5 section.**

### Phase 4 — Web UI core (Sonnet) — **done**
FastAPI: project CRUD, frame/proxy serving, job queue (track/solve/render as background jobs), WebSocket progress. React: video viewer with canvas overlay, transport controls + timeline, click-to-place points with multiple anchors per point (marked on the timeline, and each one able to track forwards or backwards from itself), tracker picker, run tracking with live progress, track paths overlaid with per-frame confidence coloring, timeline heat-strip per point showing confidence dips (the "go fix this" signal), solve panel, smoothing panel with raw-vs-smoothed curve preview, stabilized preview toggle (server-warped proxy frames), export dialog (data / video).
**Accept:** full happy path in the browser: load video → place 3 points → track (CoTracker3) → solve → smooth → preview stabilized → export MP4 + CSV.

**Built in `stable/server/` + `web/`; Phase 3 was concurrent, so the tracking half of the happy path (upload → proxy → multi-anchor track via `lk`/`template`/`dummy`/any installed deep tracker → confidence-coloured overlay → heat-strip) is wired against the real Phase 0-2 stack and verified end-to-end in a real browser (Playwright-driven Chrome against the built frontend served by `stable/server/app.py`). CoTracker3 isn't installed in this environment (needs `torch` + Meta's `cotracker` package), so the browser verification used `lk`; the tracker picker already greys out unavailable plugins via `TrackerPlugin.is_available()`, so CoTracker3 slots in with no UI change once installed.**

**Solve/smoothing/render did not exist yet (Phase 3 concurrent), so those three panels were first wired against thin, clearly-marked stand-ins (`stable/server/{solve,smoothing,render}_stub.py`) behind the same job-queue interface. ~~Those stubs are gone~~ — they were replaced by the real Phase 3 modules at the start of Phase 5; see "Phase 3 wire-up" below.**

**Deviation:** no CLI change — `stable serve` (opening a browser, one-command launch) is explicitly Phase 6's deliverable; Phase 4 added `python -m stable.server` (in `stable/server/__main__.py`, not `stable/cli/`) as the dev/test entry point instead, so as not to touch `stable/cli/` while Phase 6 hasn't defined the real command yet.

### Phase 5 — Manual editing + curve editor (Opus) — **done**
On-frame drag of a tracked point → correction keyframe (offset-blend with falloff, or hard replace); gap fill by spline interpolation; "retrack from here" using corrected position; delete range; full undo/redo (corrections layer makes this cheap). Curve editor for X,Y,R,S: pan/zoom, per-channel visibility, apply smoothing to a sub-range, keyframe pinning (smoothed curve must pass through pinned frames), before/after ghosting.
**Accept:** deliberately break a track (occlusion clip), repair it in ≤5 interactions, re-solve, no visible pop in stabilized output; undo restores exact prior state.

#### Phase 3 wire-up (done first, as part of Phase 5)
`stable/server/{solve,smoothing,render}_stub.py` are **deleted**. `stable/server/motion.py` now holds the (non-mathematical) server glue over `stable.solver` / `stable.smoothing` / `stable.render`: the on-disk curve cache, the JSON payloads and a memoised `StabilizationPlan`.
- The pivot gotcha Phase 3 recorded is handled: nothing rebuilds a transform from raw channel values. The preview and export go through `MotionSolve.transform_at()` / `SmoothedMotion.stabilization_transform()` / `StabilizationPlan.output_transform()`, with `render.rescale_transform()` re-expressing the warp for the proxy. `tests/server/test_wireup.py` measures this: tripod-stabilised frames must land back on the reference frame, which the old origin-based reading misses by ~2–5 px.
- Real border policies reach the UI: `PUT /api/projects/{id}/render` (the previously unused `RenderSettingsUpdate` schema) drives both the preview and the export, and `transparent` fails closed on video export with an explanatory 400.
- `GET /api/filters` serves `smoothing.describe_filters()`, and the smoothing panel builds every control from it. The stub-era parameter names the frontend had hardcoded (`spline {smoothing}`, `tripod {reference_frame}`) were wrong for the real catalogue and are gone; a bad filter list is now rejected with a 400 and the previous one restored rather than left unsmoothable.
- The solve payload keeps `x,y,r,s` at the top level (no frontend change needed) and gains Phase 3's per-frame quality record — `status`, `n_inliers`, `flagged`, `pivot`, `summary`. `residual` is `null` where nothing was fitted rather than a fake `0`.

**Contract additions in Phase 5** (all additive; no Phase 0/2/3 contract was changed):
- `apply_corrections()` is implemented (Phase 0 defined it and left it raising). Added a keyword-only `resolve_retrack` seam so splicing a cached re-tracking run needs no filesystem knowledge inside `stable.core` — the server supplies `tracking.retrack_resolver(project)`. Gap-fill's `spline` is a Catmull-Rom through the surrounding valid samples: a linear fill puts a visible corner at each end of a repair, which is the thing being fixed.
- `stable/smoothing/edit.py` — `blend_subrange()` and `pin_curve()`, the two curve-editor operations a filter cannot express. Pinning is exact by construction (the residual at each pin is interpolated, not iterated toward) and *local*: pins further apart than `2 * radius` each decay to zero, so pinning frame 100 cannot drag frame 400.
- `ChannelSmoothing` gained `sub_range` / `range_blend` / `pins` / `pin_radius`, applied after the filter chain (pinning last, so "passes through this frame" is a guarantee). They round-trip through the **locked v1 schema** as reserved `@`-prefixed `filter` names (`@range`, `@pin`) in the same flat `SmoothingSpec` list — no key was invented, and `@` cannot collide with a registered filter.
- `filters._fill_gaps` became public as `fill_gaps`, so sub-range blending and pinning compare against exactly the curve the filters saw.
- `stable/server/history.py` — undo/redo as a stack of *corrections-layer snapshots*, persisted to `history.json`. Snapshots rather than per-operation inverses: "undo restores the exact prior state" is then true by construction, a multi-point edit undoes as one step, and the whole undoable state is a few hundred bytes of JSON. Tracking/solving/smoothing are deliberately **not** on the stack — they are recomputable, and putting them there would turn "undo my nudge" into "throw away twenty minutes of GPU time".
- `stable/server/routes/edits.py` — corrections CRUD, `POST .../retrack`, and `POST .../edits/{undo,redo}`. The request body for a correction *is* its project-file form (`CorrectionIn` is `extra="allow"` with only `kind` required), so `correction_from_dict` stays the single definition of what an edit is instead of a Pydantic mirror that can drift.
- `GET .../points/{id}/track` gained `corrections`, `gaps` and (when enabled corrections exist) `raw_x`/`raw_y` — the before/after ghosting source and the one-click gap list.
- `tracking.run_point_retrack()` caches its run at `tracks/<point>.retrack-<correction>.npz` and references it from the correction, so the multi-anchor baseline is never overwritten and reopening a project needs no GPU. A `bidir` retrack is two single-direction runs (the plugin contract is single-run only).
- **Deviation:** the acceptance test's break is *imposed* on a known-good track rather than tracked off a real occlusion clip. Handing an occlusion clip to `lk` and assuming it fails usefully makes the assertion depend on how badly the tracker happens to do; the day it improves, the test silently stops testing the repair path. "No visible pop" is measured in the output (`stabilization_transform(t) . truth(t)` applied to the frame centre, worst second difference), not in the warp — the warp is *supposed* to be jittery, since it carries the inverse of the clip's judder. Result: 14.1 px before the repair, 0.061 px after, exactly matching the never-broken baseline.
- **Deviation:** the headline repair in the acceptance test is *one* interaction (`delete_range`), not the two-or-more the criterion allows: with other points still covering those frames the solver's graceful degradation carries them, and the solve becomes numerically identical to the unbroken one. Gap-fill is tested as the separate case where the point itself is needed (50 px error → 3 px).
- Frontend: `EditPanel`, `CurveEditor` and `web/src/smoothing.ts` (which splits/rejoins the flat spec list so the smoothing panel and the curve editor cannot clobber each other's settings). The curve editor draws one lane per channel — pixels, radians and a unitless scale on a shared axis would make every readout a lie.
- Verified end-to-end in a real browser (Playwright-driven Chromium against the built frontend): upload → 3 points → track (`lk`) → solve → smooth → stabilized preview → curve editor (zoom, sub-range, pin — asserting the smoothed curve passes exactly through the pin) → on-frame drag → undo/redo via Ctrl+Z → delete range → gap fill → re-solve → export MP4 + CSV, with no console errors.

### Phase 6 — Packaging & polish (Sonnet) — **done**
Model download manager UI; `stable serve` one-command launch that opens the browser; pip package + CUDA Dockerfile; Windows GPU setup docs; CI (lint, unit, headless integration on a bundled 5 s test clip); README with GIF demo; sample project.
**Accept:** fresh Windows machine: `pip install stable-tracker`, `stable serve`, complete the Phase 4 happy path.

**Model manager:** `stable/server/routes/models.py` (`GET/POST/DELETE /api/models*`) is a thin server layer over `stable.trackers.weights` — it does not change that module's `describe()`/`ModelSpec` contracts (locked by Phase 2 tests), it composes them with which installed tracker plugin(s) need each checkpoint and the process-wide opt-in env flags (`weights.env_flags()`, additive). Downloads reuse the existing `JobManager`/WebSocket progress machinery under a reserved pseudo-project id (`jobs.MODELS_JOB_SCOPE = "_models"`) rather than a second progress mechanism, streamed at a new `/ws/models` (alongside the existing `/ws/projects/{id}`, both now sharing one `_stream_jobs` implementation in `routes/ws.py`). `web/src/components/ModelManager.tsx` is the browser UI (opened from the project picker and from inside a project); `stable models [list|download|remove]` is the CLI equivalent for headless use. Both surface the Phase 2 fail-closed rule directly — unpinned/non-commercial downloads need an explicit checkbox or flag, never a default-on workaround.

**`stable serve`:** `stable/cli/serve.py`. Looks for a built frontend via `stable.server.app.web_dist_dir()` (see packaging below); if none is found and a `web/` source checkout is sitting next to the installed package, runs `npm install`/`npm run build` itself before starting uvicorn and opening the browser once the port answers. `stable/server/__main__.py` (Phase 4's thin dev runner) is unchanged and still works for development.

**Packaging:** `stable/server/app.py`'s frontend resolution (`web_dist_dir()`) now checks a packaged location, `stable/server/webdist/`, before the source checkout's `web/dist/` — populated by `scripts/build_frontend.py` and shipped as wheel `package-data` (`pyproject.toml`), so `pip install stable-tracker` needs no Node.js at all. The repo commits a built snapshot at `stable/server/webdist/` (regenerate with `python scripts/build_frontend.py` after a frontend change; `tests/server/test_web_dist.py` guards against it going missing) so `pip install .` from this checkout works standalone — verified by hand: `python -m build --wheel`, install into a throwaway venv with no source tree on `PYTHONPATH`, `stable serve --no-browser` answers `/api/health`, `/api/models` and `/` with the real built UI. `stable/examples/` (a bundled ~5 s synthetic clip + its three trackable feature points, generated by `scripts/generate_sample_clip.py`) is the one asset shared by `stable sample`, the CI headless-integration test (`tests/test_cli_sample.py`, marked `pytest.mark.integration`) and the packaging smoke test — never real footage, never model weights, 20 KB on disk.

**Docker / Windows GPU:** `Dockerfile` (CUDA 12.4 base, Python 3.11 via deadsnakes, PyTorch's matching CUDA wheel, both deep trackers' model code; no Node.js needed since the frontend ships pre-built) targets headless Linux GPU boxes. `docs/windows-gpu-setup.md` covers the native-Windows path instead (driver check, CUDA PyTorch wheel selection, model-code install, checkpoint download opt-ins, `stable devices`/`stable serve`, troubleshooting) — running natively rather than in a container is what lets `stable serve` reach the user's browser.

**CI:** `.github/workflows/ci.yml` — `lint` (ruff + mypy), `unit` (`pytest -m "not integration"`, matrixed on Python 3.11/3.12), `integration` (`pytest -m integration`, the bundled-clip pipeline, plus a `stable sample` smoke run), `frontend` (`tsc --noEmit` + `vite build`), and `package` (build a wheel, install into a clean venv on **both Ubuntu and Windows runners**, run `stable serve --no-browser` and curl `/api/health`, `/api/models` and `/`) — the last job is Phase 6's accept criterion executed directly, not just implied by the others passing.

**Deviations:**
- **Bugfix, not a contract change:** capturing the README's demo GIF against a real running server exposed a real Phase 4 bug in `web/src/components/Viewer.tsx` — the effect that redraws the canvas did not list its own "an image just finished loading" signal (`forceRedraw`'s counter) as a dependency, so a frame image could load successfully and never actually get painted until some *unrelated* prop happened to change. This directly threatens "complete the Phase 4 happy path" (the literal Phase 6 accept criterion) under exactly the conditions of a first-run/scripted session, so it is fixed here (one line: the counter is now a real dependency of that `useEffect`) rather than left in place or routed around for the screenshot. No component contract, prop, or API changed.
- **GIF demo achieved for real, not stubbed:** `docs/demo.gif` is captured by driving the actual server + a real Chromium (Playwright) through upload → track → solve → curve editor → stabilized preview → model manager, screenshotted and assembled with Pillow. The demo clip used for the screenshots is a brighter, larger, throwaway clip (not the shipped `stable/examples` asset, which stays tiny and is what tests/CI actually exercise) generated the same way as the Phase 3 `WarpedClipSource` fixture, purely so the screenshots are legible; nothing under `stable/` or `tests/` depends on it and it is not committed.
- **No live GitHub Actions run:** this sandbox has no GitHub Actions runner and no Docker daemon (`docker` is not installed), so `.github/workflows/ci.yml` and the `Dockerfile` are written to standard patterns and validated by running their equivalent commands locally (ruff/mypy, both pytest marker sets, `npm run build`, and — for the `package` job specifically — the exact wheel-build-then-clean-venv-install-then-`stable serve` sequence, by hand, successfully) rather than by an actual CI/Docker execution.
- This repository has no `.git` history in this environment (predates Phase 6; not introduced here). The CI workflow, badges and Docker/GitHub URLs assume the stated `github.com/stable-tracker/stable` remote once one exists.

## Handover notes

- Run phases sequentially; 1 can start once 0's contracts merge. 4 can run in parallel with 3 if the API surface for solve/render is stubbed.
- Each phase = one subagent run. Prompt: "Read PLAN.md. Implement Phase N. Respect the Stack/Architecture/Data-model sections as fixed decisions. Meet the acceptance criteria; write tests proving them."
- If a phase agent wants to change a contract, it must update PLAN.md in the same PR and say why.
- Keep test media tiny (bundle one 5 s 720p clip); never commit model weights.
