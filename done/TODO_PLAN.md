# Todo execution plan

Source: `todos` (28 items). Grouped into workstreams with **strict file ownership** so agents run in parallel without colliding. Each agent: stay inside your OWNS list, never edit another workstream's files, keep the full suite green (`pytest`, `ruff`, `mypy`, `npm run build`).

Baseline: Phases 0–6 complete, suite green, `pip install -e .[all]` working, `stable serve --port 8080` verified.

**Note on port 8000:** `iphlpsvc` holds it on this machine. Not a Stable bug, but W4 owns making `serve` fall back to a free port with a clear message.

---

## W1 — Anchor correctness & retracking (Opus) — todos 11, 12, 13, 15, 20

The highest-value cluster: todo 13 is a stated bug and the others are the same idea seen from different angles — **a user-placed keyframe is ground truth and must never be silently overridden**.

- **13 [BUG]** Anchors placed at the start are ignored by tracking. Fix so every user anchor is honoured. When a run reaches a frame holding an anchor it disagrees with (beyond a threshold), it must **stop** rather than drift past it, and report where and by how much.
- **11** Every manual correction is automatically promoted to a ground-truth anchor, so later retracks treat it as a fixed constraint instead of overwriting it.
- **12** Track forwards to the next keyframe / backwards to the previous keyframe (bounded runs, not whole-clip).
- **20** Retrack for a user-specified number of frames rather than the whole sequence.
- **15** With multiple keyframes, track all forwards and backwards; every resulting path must intersect every keyframe. Any that doesn't is flagged low-confidence and drawn **red on the timeline**.

Existing `trackers/merge.py` already blends bidirectional runs per anchor with exact anchor passthrough — 13 is likely the server not passing all anchors into the run, so start there before changing merge maths.

**OWNS:** `stable/trackers/merge.py`, `stable/core/corrections.py`, `stable/server/tracking.py`, `stable/server/routes/edits.py`, `stable/server/routes/points.py`, `stable/server/schemas.py`, `stable/cli/track.py`, `web/src/components/PointsPanel.tsx`, `web/src/components/Timeline.tsx`

### W1 — done

**The bug (13), reproduced first.** The hint was right: the merge was never at fault. `merge.py` has guaranteed exact anchor passthrough since Phase 0, and the *full-track* path does hand it every anchor — but `run_point_retrack` passed only the one corrected position, and `apply_corrections` then spliced the resulting run over **every frame in that direction**, anchors included. A run started at frame 5 flattened the operator's keyframe at frame 15 by ~12 px. That is `test_a_retrack_honours_every_anchor_it_passes`, written failing before anything was changed.

**One idea, applied in both directions.** `tracking.effective_anchors(point)` is now the single answer to "what has the operator told us about this point" — its pins plus every manual correction that states a position. Both the full track and the retrack plan and guard against that list.

- `merge.enforce_anchor_agreement()` (new, pure, tracker-agnostic) walks a run outward from its own anchor and **stops it at the first keyframe it lands more than `tolerance_px` from** (default 6 px), truncating from that frame onward and returning an `AnchorViolation` — frame, both positions, distance, origin, direction. Keyframes it *agrees* with are walked straight through, so many-keyframe runs stay honoured end to end. The run's own anchor is never a violation.
- Nothing in `plan_anchor_runs` / `merge_anchor_runs` changed. Additive only: `AnchorViolation`, `DEFAULT_ANCHOR_TOLERANCE_PX`, `enforce_anchor_agreement`.

**Todo 11 — promotion without breaking undo.** `corrections.ground_truth_anchors()` promotes `replace` and `retrack` (the kinds that carry an absolute position) to anchors. A drag arrives as `offset`, which is *relative* — replaying it over a re-tracked path would nudge an already-correct position a second time — so `POST /corrections` appends a companion `replace` at the same frame carrying the resulting absolute position. The drag still blends its neighbours; the frame the operator pointed at is pinned exactly, before and after a re-track.
Deliberate consequence: promotion lives **entirely in the corrections layer**, never in `point.queries`. `server/history.py` snapshots corrections only, so undo un-promotes for free and in the same step — which a `queries` mutation could not have done without touching a file W1 does not own.

**Todos 12 and 20 — bounded runs.** `tracking.bounded_frame_range()` caps a run at the neighbouring keyframe (`stop_at_keyframe`) or at N frames (`max_frames`), tighter bound winning, never leaving the clip and always containing its own start frame. Because a `RetrackCorrection` only ever splices the frames its cached run actually covered, bounding the run bounds the *edit* — no range bookkeeping was needed in `apply_corrections`. `RetrackRequest.x`/`y` became optional: omitted, the run starts from wherever the point already is, which is what "carry on from here" means and what the new panel buttons send.

**Todo 15 — flagged, not falsified.** Violations are persisted (raw track: in the existing `.report.json`; retracks: a `.violations.json` sidecar beside the run — *not* `report_path_for`, which strips the whole suffix chain and would have collided with the raw track's own report) and re-aggregated by `load_anchor_violations()` across the raw run plus every *enabled* retrack correction, so toggling a fix off stops it flagging. `GET .../points/{id}/track` gained `ground_truth`, `anchor_violations`, `flagged_anchors`; both job results gained the same. The keyframe's own confidence is **not** lowered — exact anchor passthrough at confidence 1.0 is a Phase 0 contract — the signal is the flag, and the timeline draws those pins red with a full-height tick.

**Frontend.** `Timeline.tsx` paints unreached keyframes red and now pins promoted corrections too. `PointsPanel.tsx` gained the bounded-run controls (◀ track / track ▶, "to next keyframe" | "for N frames" | "to end of clip"), marks promoted keyframes `✎`, and lists each miss as "⚑ keyframe f15 missed by 12.3 px". The new fields are typed locally in those two components rather than in `web/src/types.ts` / `api/client.ts`, which W1 does not own.

**Tests (43 new).** `tests/server/test_anchor_integrity.py` (the bug, the stop-and-report, agreeing runs passing through, promotion, promotion surviving a re-track, undo un-promoting, both bounds, bidirectional bounds, position-free start, and the multi-keyframe flagged/clean pair); `tests/test_merge_guard.py` (the guard in isolation, both directions, first-not-worst, immutability of the input); `tests/test_anchor_promotion.py` (which kinds promote, disabled edits retiring, corrections outranking pins, every bound case).

**Not done, deliberately:** `cli/track.py` is untouched — it is single-run by design and `--frame-range` already bounds it; nothing in these five todos needed it. `run_point_tracking`'s `direction` argument is still ignored, but its docstring no longer claims otherwise: a full track *is* every anchor both ways, and dropping half the planned runs would leave the clip partly uncovered.

**Suite:** `pytest -m "not integration"`, `ruff check`, `ruff format --check` and `mypy` clean across W1's files; `npm run build` green. (Residual failures seen during the run belonged to W2's and W5's in-flight files, not W1's.)

## W2 — Stabilised output (Opus) — todo 17 — **done**

- **17 [BUG]** Generating the stabilised output video doesn't work. Reproduce first, fix, and add a regression test.
- Target the stabilisation centres on: average of all keypoints, or a specific user-selected keypoint.
- Crop size around that target.
- Border fill for the exposed edges: solid colour, blurred frame, stretched/mirrored edge, or a combination — user selectable.

Existing `render/warp.py` has crop / auto-zoom / transparent policies; extend rather than replace, and keep `SimilarityTransform` semantics (curves are **pivot-relative**, use `MotionSolve.transform_at()` / `SmoothedMotion.stabilization_transform()`, never `from_channels`).

**OWNS:** `stable/render/**`, `stable/server/motion.py`, `stable/server/routes/export.py`, `stable/server/routes/solve.py`, `stable/cli/stabilize.py`, `web/src/components/ExportDialog.tsx`, `web/src/components/SolvePanel.tsx`

### What the bug actually was

Not a crash, which is why nothing in the suite caught it: **the export succeeded and produced a re-encode of the source.**

Found by reading the user's own workspace (`~/.stable/projects/`) rather than by guessing — the synthetic reproductions all passed. Project `5d9bbb9e0689` (1920x1080, 24 fps, one tracked point) held: a clean solve saying the camera moved **151 px in x and 89 px in y**, two Butterworth filters with `"enabled": false`, and a 7.8 MB `renders/stabilized.mp4`. Decoding it against the source gave a mean absolute difference of ~1.5/255 — codec noise. Nothing had been stabilised. The job said `done`, reported `zoom: 1.0`, and gave no hint anything was wrong.

The cause is structural. The warp is `smoothed(t) . raw(t)^-1`, so with **no enabled smoothing filter and no stabilisation target the warp is exactly the identity on every frame**. `stable stabilize` never hits it because `cli/motion.py` applies `DEFAULT_SMOOTHING = "butterworth:cutoff_hz=1.0"` when the user does not choose one; the server has no such default and silently rendered the identity. That is precisely why `stable sample` works from the CLI and the browser path does not.

(The user had then set the border policy to `transparent`, which greys the export button out permanently — a second dead end, now explained inline in the dialog and made unnecessary by the border fill.)

### The fix

- `StabilizationPlan.max_displacement` / `.is_no_op` measure how far the plan actually moves the picture (corners, stabilisation warps only — the crop/zoom fit is framing, not stabilisation). `IDENTITY_TOLERANCE_PX = 0.01`.
- `POST /export/video` now **plans at submit time** (cheap — no pixels) and refuses a no-op with a 400 naming both ways out (`routes/export.NO_OP_DETAIL`), instead of spending minutes encoding a copy. The `transparent` check still runs first, so its existing message is unchanged.
- `stable stabilize` prints the same warning to stderr rather than refusing — a CLI user who passes explicit flags is entitled to whatever they asked for.

### The options (todo 17's three bullets)

New `stable/render/options.py` (`StabilizationTarget`, `CropSpec`, `RenderOptions`, `average_positions`, `hold_gaps`) and `stable/render/fill.py` (`BorderFill`, `composite_fill`).

- **Target** — `frame` (default, Phase 3 behaviour unchanged), `centroid` (average of every solve-enabled point's *corrected* track) or `point` (one point), pinned to its position on an anchor frame (default: the solve's reference frame, so the framing does not move). **A target stabilises on its own**, with no smoothing configured at all — which is the direct answer to the bug.
- **Crop** — `CropSpec(scale, centre)`. `scale=None` is the exact minimal gapless crop Phase 3 computed; an explicit scale overrides it in either direction, and `centre="target"` crops *around* the target. `plan_output()` gained `centre`/`crop` and now reports `crop_scale` (requested) beside `coverage` (measured) plus `gaps`, so "will the fill be visible" is answered rather than implied.
- **Fill** — base plate `solid` / `stretch` (`BORDER_REPLICATE`) / `mirror` (`BORDER_REFLECT_101`) / `blur`, with `blur`, `tint` and `feather` stacking on **any** plate — that is how "a combination of these" is expressed without a fifth mode. The plain solid case is still a single `warpAffine`; anything else costs two more warps and a blend.

### Pivot safety (the contract that must not break)

The target lock is `T(q(anchor) - q(t)) . W(t)` where `W(t)` is `SmoothedMotion.stabilization_transform(t)` — a **pure translation composed on top** of the sanctioned warp. Nothing is rebuilt from channel values anywhere. `test_the_target_lock_composes_onto_the_sanctioned_warp` asserts rotation and scale are bit-identical to the base warp, which a `from_channels` reconstruction could not satisfy. `tests/server/test_wireup.py` still passes untouched.

### Plumbing

- Options persist in `render_options.json`, a **sidecar** beside `solve.json`/`smoothing.json` (`motion.RENDER_OPTIONS_CACHE`). The locked v1 project schema is not touched — a test asserts `project["render"]` still has exactly `border`/`codec`/`crf`.
- `GET`/`PUT /api/projects/{id}/render/options` in `routes/solve.py` (partial updates; validation delegated to `render/options.py`; drops the memoised plan like the border policy does). Pydantic models are local to the module because `schemas.py` is W1's.
- The **stabilized preview goes through the same `preview_plan()`**, with `blur`/`feather` rescaled for the proxy, so what you see is still what you get.
- `GET /api/projects/{id}/export/renders` lists finished renders — previously the only route to one was a download link held in component state, so closing the dialog lost it.
- CLI: `--target frame|centroid|point:INDEX`, `--target-frame`, `--crop`, `--crop-centre`, `--fill`, `--fill-color`, `--fill-blur`, `--fill-tint`, `--feather`.
- `ExportDialog.tsx` exposes all of it. It calls the two new endpoints with plain `fetch` because `web/src/api/client.ts` is W4's file.

### Tests

`tests/test_render_options.py` (32, library level), `tests/server/test_stabilized_output.py` (14, HTTP level, opens by reconstructing the reported project), plus 8 new CLI cases in `tests/test_cli_stabilize.py`.

`test_export_video_end_to_end` in `test_solve_and_export.py` was exporting a no-op and now configures smoothing first — it was passing for the wrong reason, and its old form *is* the bug.

Verified against the real project: as the user had it → `is_no_op=True`, refused with an explanation; with a target → the point that travelled 157x89 px travels **0.0 x 0.0 px** in the output, crop computed at 83.5%, no gaps.

### Notes for other agents

- `stable/server/webdist/` (the packaged frontend snapshot `stable serve` prefers) was regenerated with `python scripts/build_frontend.py`, since a stale bundle means `stable serve` ships a UI without this fix. It is a build artefact built from whatever `web/src` holds, so it picked up W1's in-flight `PointsPanel.tsx`/`Timeline.tsx` too — **re-run that script after your own frontend changes.**
- Suite at hand-off: `1698 passed, 6 failed`; all 6 are W5's tracker plugins not yet registered as entry points (`TrackerNotFoundError: no tracker named 'raft'` etc.). Excluding `tests/test_tracker_suite.py` and `tests/test_deep_trackers.py`: **1534 passed**. `ruff`/`mypy` are clean across every W2 file; the remaining findings are in W5's new modules and `tests/server/test_models.py`.

## W3 — Viewer & interaction UX (Sonnet) — todos 1, 2, 3, 7, 8, 9, 10, 21

- **2** Fill the canvas with the frame (currently doesn't use available space).
- **3** Zoomable canvas (and pan) — essential for precise point placement.
- **8** Click-to-move editing: select an active point, then clicking on a frame moves it there. Much faster than dragging; keep drag working too.
- **9** Keyboard controls for next/previous frame (and ideally play/pause, jump to keyframe).
- **7** Toggle tracks on/off to see the frame behind.
- **10** Show/hide individual overlay elements (dotted lines etc.) — the display clutters fast.
- **1** Upload progress bar.
- **21** Smoothing charts: click already jumps to a frame; make click-and-drag scrub through frames in real time, like the main timeline.

**OWNS:** `web/src/components/Viewer.tsx`, `web/src/components/CurveEditor.tsx`, `web/src/components/EditPanel.tsx`, `web/src/components/ProjectPicker.tsx`, `web/src/components/SmoothingPanel.tsx`, `web/src/App.tsx`, `web/src/main.tsx`, `web/src/colors.ts`, `web/src/smoothing.ts`, `web/src/state/ProjectContext.tsx`

## W4 — Performance & live feedback (Opus) — todos 4, 5, 6, 16

- **6** Barely uses 25% CPU and 0% GPU. Use all available resources: parallel decode/proxy generation, batched GPU inference, worker threads. Measure before and after and put the numbers in the PR.
- **5** Proxy frames flicker black when navigating — prefetch/cache adjacent frames so scrubbing is smooth.
- **4** Live updates of track processing (stream partial results, don't wait for completion).
- **16** Let the user step through frames while the tracker is still running, to watch it perform.
- Also: make `stable serve` fall back to a free port when the requested one is unavailable, and print the port it actually bound. Print the URL only *after* a successful bind.

**OWNS:** `stable/video/**`, `stable/server/jobs.py`, `stable/server/routes/ws.py`, `stable/server/app.py`, `stable/server/storage.py`, `stable/cli/serve.py`, `stable/trackers/deep.py`, `stable/trackers/chunking.py`, `web/src/components/JobProgress.tsx`, `web/src/hooks/**`, `web/src/api/client.ts`

### W4 — done

**First, the suite was not green — it was crashing.** `pytest tests/server` did not fail, it killed the interpreter with an access violation (`0xC0000005`). Every server test file passed *alone*, so the failure only appeared in a full run and had been committed. `Workspace.get_video_source` hands **one** `PyAVVideoSource` to background jobs and request threads alike, so `Workspace.close()` — run by `reset_workspace()` between test modules, and by the app lifespan on shutdown — freed the FFmpeg container while a proxy job was still decoding through it. Reproduced in 40 lines (close a source while a thread reads it) before anything was changed.

`PyAVVideoSource.close()` now takes the same lock the decode path holds, so an in-flight decode finishes and every later one raises `VideoError` instead of touching freed memory; abandoned decode generators are closed explicitly rather than left for the GC to finalise beside their replacement. `delete_project` was the same bug from the other side — it evicted the cached source without closing it, so on Windows the open handle silently defeated the `rmtree(ignore_errors=True)` that followed.

**Todo 6 — measured, and the first number was wrong.** `scripts/benchmark_video.py` runs each optimisation against its own disabled setting on one clip (240 frames, 1920x1080, 32 cores):

| | before | after | |
|---|---|---|---|
| proxy generation (1 → 8 encode threads) | 180 f/s | 306 f/s | **1.71x** |
| multi-run decode, single-threaded | 272 f/s | 309 f/s | **1.14x** |
| **4 concurrent runs (shared decoder → pool)** | **34.6 f/s** | **485 f/s** | **14.03x** |
| scattered scrub (the UI's pattern) | 11.5 f/s | 11.5 f/s | 1.00x |

The first run of this reported **24.61x** for multi-run decode. That was an artefact: the "before" case had been given a 16 MB cache, where `PyAVVideoSource`'s real default is 64 frames. Corrected to the actual old default it is 1.14x. The honest headline is the **concurrent** row — pooling exists so parallel jobs stop dragging one decoder's cursor across the clip, and that is where the 14x is. The scrub row is there to show read-ahead costs the UI nothing.

**Todos 4 and 16 — live results, not a live percentage.** `jobs.py` already published the decoder's frame cursor; what was missing was *what the tracker found*. `tracking.LivePartialTrack` accumulates each planned run as it lands and publishes the path so far — full clip length, `null` where not yet covered, the same shape a finished `TrackData` has, so the viewer draws a partial path with the code that draws a finished one. Deliberately **not** a partial merge: blending is `merge_anchor_runs`' job and running it early on a partial set could only disagree with the real answer. Publishing is duck-typed (`getattr(progress, "merge", None)`), so the CLI and every pre-existing test pass a plain callback and publish nothing.

Frontend: `useLiveTracking` indexes in-flight runs by point; the viewer draws the partial path dashed in the point's colour with a ring on the frame being read; the timeline shades the span covered and marks the cursor; `JobProgress` says "frame 812 · run 2/4" instead of only a percentage.

**Todo 5 — two causes, both fixed.** `useFrameImages` keeps the previous frame on screen until the next has actually decoded (the picture only ever changes to another picture, never to black) *and* prefetches neighbours biased in the direction of travel. The old code painted immediately and drew `#111` on a miss, which is the flash.

**`stable serve`** binds the socket itself — requested port, then the next ten, then whatever the OS gives — and prints the URL only once that succeeded, naming the port actually held. Port 8000 is held by `iphlpsvc` on this box, so the old code printed a correct-looking URL and then a traceback. Verified live: falls back to 8001 and answers there.

**Not done:** todo 1 (upload progress) was already done by W3 via `XMLHttpRequest` in `ProjectPicker.tsx`; moving it into `api/client.ts` now that W4 owns that file would be churn for no behaviour change, so it stays.

**Tests:** `tests/server/test_live_tracking.py` (7), `tests/test_video_source.py` +3 (the close race — before the fix these did not fail, they killed the interpreter), `tests/test_cli_serve.py` +6.

## W5 — Tracker algorithms (Sonnet) — todos 22–28

Each as a `TrackerPlugin` registered via the `stable.trackers` entry-point group (see `CONTRIBUTING.md`). Confidence must be normalised to [0,1] on the shared scale in `trackers/confidence.py`.

- **22** CoTracker3 and TAPIR already exist but report `[unavailable]` because PyTorch isn't installed and checkpoints are unpinned. Make them genuinely usable: verify against real weights, pin the SHA256s you actually observe, document the install.
- **23** Correlation tracker. **24** Kalman filter tracker. **25** Particle filter tracker. **26** Mean-shift tracker. **27** KLT tracker (note: `lk.py` is already pyramidal Lucas–Kanade — either extend it with feature (re)selection or justify a separate plugin rather than duplicating it). **28** A deep-learning tracker using a pre-trained model, beyond CoTracker3/TAPIR.

24 and 25 are estimators rather than appearance trackers: implement each as a measurement source (NCC/correlation) combined with the filter, and say so in the docstring.

**OWNS:** new `stable/trackers/*.py` plugin modules, `pyproject.toml` (entry points + extras), `CONTRIBUTING.md`, `docs/**`. Do **not** edit `merge.py`, `base.py`, `registry.py`, `confidence.py` contracts.

## W6 — Templates, export, sessions (Sonnet) — todos 14, 18, 19 — *after W1 and W3*

- **14** Template matching: let the user set the template size and see the template patch. With multiple keyframes, show all templates as a grid of thumbnails in the side panel.
- **18** Export tracked data in various formats (CSV, JSON). **Verify before building** — `stable solve` already writes CSV and JSON and the export dialog offers both; this may only need extra formats or is already done.
- **19** Save/load tracking sessions. **Verify before building** — projects already persist to disk via `server/storage.py`; likely already satisfied, in which case say so rather than duplicating it.

**OWNS:** `stable/trackers/template.py`, `web/src/components/PointsPanel.tsx` (after W1), a new template panel component, `stable/server/routes/projects.py`

### W6 — done

**Todo 14 — the size, and the patch it produces.** Per-point tracker settings live in `tracker_options.json`, a sidecar beside `solve.json` / `smoothing.json` / `render_options.json` — the same choice W2 made for the render options, and for the same reason: the locked v1 project schema stays untouched (a test asserts the setting never appears in `project.stable.json`). `TrackerRegistry.create` gained an `options=` argument that is **filtered against the plugin's own constructor**, so settings follow a point across a tracker change instead of erroring — switching a template point to LK and tracking it is a test, not a hope.

`GET/PUT .../points/{id}/patches` and `GET .../points/{id}/patches/image` serve the settings and the patch itself, cropped on demand from the source clip (a patch is a pure function of clip/frame/position/radius; caching it would only mean invalidating it every time an anchor moves). `PatchPanel.tsx` puts a size slider above a grid of one thumbnail per keyframe, drawn pixelated so the texture being judged is not smoothed away. The slider previews sizes through a `radius` query parameter before anything is saved, so dragging shows the consequence and only "Apply" commits it.

The grid's "this patch does not fit" marker uses **the tracker's own edge rule**, so a keyframe the panel shows as ✕ is exactly one the tracker would refuse — the preview cannot flatter the run.

**Corrected after use: the panel was invisible.** The first version rendered only when `point.tracker === "template"`, which is a literal reading of todo 14 and wrong in practice — projects use the default LK tracker, so the panel never appeared and the todo looked unimplemented. The need is not template-specific: "what does the tracker see here, and does it still look like the keyframe" is the same question whatever is doing the looking.

So patches are now shown for **every** point, sized from whatever window its own plugin declares — `tracker_window()` reads `template_radius` / `patch_radius` / `window_radius` / `win_size` (halved, since LK and KLT state a full side length) from the plugin's constructor, and reports which one it used. A tracker declaring none of them has no local window at all (the deep trackers see whole frames) and the panel says the size is a preview rather than inventing a number. Only the size *control* stays template-specific, because only that plugin takes its window from a stored setting; elsewhere the real number is shown instead of a slider that would silently change nothing.

Two additions the report asked for directly: the panel shows the patch at the **current frame** — where the tracker actually put the point, not just where it was pinned (`kind: "tracked"` vs `"keyframe"`, with a keyframe winning on frames that have both) — and `AllPatchesStrip` shows every point's patch on this frame in one row, so "which of my points has drifted" is a glance rather than a click each. Patch URLs carry the edit revision, since a correction changes the pixels without changing the URL.

Verified against a real project: two LK points, radius 10 read from `win_size`, `adjustable: false`, nine keyframe thumbnails and a live tracked patch that visibly follows the subject.

**Todo 18 — verified, then extended.** CSV and JSON already existed for both tracks and curves, via `stable solve` and the export dialog, so todo 18's two named formats were done. What was missing was the *"use in other applications"* half: no compositor reads either. Added After Effects keyframe data (`format=ae`, `stable/core/trackdata.py`) — tab-separated, CRLF, pastes onto a layer's Position, coordinates unflipped because AE also measures from the top-left with Y down, and gaps **skipped** rather than written as zero so a missing frame does not import as the point jumping to the corner.

Deliberately not Nuke: a `Tracker4` node is an intricate knob serialisation that would have to be checked against Nuke itself to be trusted, and shipping a format that cannot be verified here is worse than not shipping it.

**Todo 19 — already satisfied; verified rather than rebuilt.** Projects persist continuously through `server/storage.py`, so "pause and resume without losing progress" is a claim about the existing design. `tests/server/test_session_persistence.py` walks a full session (points, anchors, a track, a manual correction, solve settings, template settings), throws the entire server away, and reopens from disk in a new one: the corrected path, its anchors, the correction on top of it and the sidecar settings all come back, and the proxy frames are still servable. Nothing was built for this todo. The second test covers the other half — a deleted project really leaves disk, which on Windows is only true because `delete_project` now closes the video source first.

**Tests:** `tests/server/test_template_options.py` (11), `tests/test_trackdata.py` (8), `tests/server/test_session_persistence.py` (2), `tests/server/test_solve_and_export.py` +3.

---

## Waves

1. **W1, W2, W5** in parallel (file sets disjoint).
2. **W3, W4** in parallel, once wave 1 lands.
3. **W6** last.
