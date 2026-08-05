# Implementation plan — second feedback round

Source: `new_todos` (17 items). Same execution model as `TODO_PLAN.md`: workstreams
with **strict file ownership** so agents run in parallel without colliding. Each
agent stays inside its OWNS list, never edits another workstream's files, and
keeps the suite green (`pytest -m "not integration"`, `ruff check`,
`ruff format --check`, `mypy stable`, `npm run build`, then
`python scripts/build_frontend.py`).

Baseline at time of writing: suite green apart from two **pre-existing**
failures in `tests/test_phase3_acceptance.py` (rotation off by 0.0002 against a
0.0002 tolerance) — confirmed on a clean tree, not caused by recent work, and
owned by nobody below. Whoever touches the solver first should decide whether
the tolerance or the solve is wrong.

Every "**Found:**" line below was checked against the code, not assumed.

---

## Already delivered — do not re-implement

- **13 — shortcut key to add an anchor.** Done. `A` pins the selected point
  where it already is on the current frame; `⇧A` (or `A` with nothing to pin)
  arms click-to-place; pressing again disarms. `web/src/state/ProjectContext.tsx`
  (`addAnchorHere` / `addAnchorAt`).

- **19 — "when I click add anchor, it now automatically adds it at the current
  track position".** This describes the behaviour that shipped with 13, so it
  reads as a report rather than a request. **Open question**, listed again at the
  bottom: if it is unwanted, the fix is small and belongs to W3.

---

## W1 — Memory and responsiveness (Opus) — todos 21, 23, 17

The two crashes-in-practice items. Both are measurable, so **measure before and
after and put the numbers in the PR** (`scripts/benchmark_video.py` is the
precedent).

### 21 [BUG] mean-shift exhausts memory

**Found:** not a mean-shift bug as such — every classic tracker keeps an
**unbounded per-run frame cache** of converted frames, keyed by frame index:
`gray_cache` in `lk.py:192`, `template.py:138`, `correlation.py:163`,
`kalman.py:235`, `particle.py:203`, and `hsv_cache` in `meanshift.py:186`.
Nothing evicts. At 1920×1080 that is 2.07 MB/frame for the grayscale ones and
**6.2 MB/frame for mean-shift's HSV** — ~6 GB over a 1000-frame clip, which is
why mean-shift is the one that dies first and why the failed allocation
(24,883,200 bytes = 1920×1080×12) is itself unremarkable.

The decoder underneath already has a bounded cache
(`PyAVVideoSource(cache_size=64)`, `stable/video/source.py:111`); the trackers'
private ones are the leak.

- Give the conversion cache a bound (an LRU sized in **bytes**, not frames, so
  the limit means the same thing on SD and 4K). A run is sequential per
  direction, so a small window is all any of them re-reads — verify that claim
  against the access pattern rather than assuming it.
- One helper, used by all six, so this cannot regress in five places
  independently. `stable/trackers/framecache.py` (new) is the obvious home, and
  it is additive: no plugin contract changes.
- Test: a run over a synthetic clip long enough to blow the old ceiling, with
  the cache's own high-water mark asserted below the bound. A memory test that
  measures the process is flaky; measure the cache.

### 23 [BUG] scrubbing bogs down and the UI stops updating

**Found:** the user's hypothesis is right, and the mechanism is worse than
"patches are slow". `GET .../points/{id}/patches/image` crops from
**`workspace.get_video_source()` — the full-resolution source clip**
(`routes/points.py`, `tracker_options.template_patch`), so every thumbnail is a
full-res random-access seek. `AllPatchesStrip` requests **one image per point
per frame**, plus `PatchPanel` re-fetches `/patches` JSON on every frame change.
Scrubbing an 8-point project therefore queues ~9 full-res seeks per frame behind
`PyAVVideoSource`'s single decode lock, on FastAPI's bounded sync threadpool.
Proxy frames are cheap disk reads (`ProxyStore`) but they queue behind the same
exhausted threadpool, which is exactly the reported symptom: frames stick, then
everything stops.

Four fixes, in order of effect:

- **Serve patches from the proxy tier** when one is ready, scaling the radius by
  the proxy factor, and say in the response which tier the pixels came from. A
  72 px thumbnail does not need 1:1 source pixels. Keep the source path for the
  "Apply"-time preview where texture judgement matters.
- **Cache** rendered patches keyed by `(point, frame, radius, edit revision)` —
  the revision is already in the URL, so this is a small LRU plus an
  `ETag`/`Cache-Control`, letting the browser skip the round trip on the way
  back through frames it has already seen.
- **Debounce** the panel's fetches while the frame is moving fast, and cap
  concurrent patch requests client-side.
- **Do not let a patch request block a frame request**: make the patch endpoints
  `async def` with the decode in a worker, or give patches their own bounded
  executor. Whichever, the invariant to state and test is *proxy frame latency
  must not depend on how many patch requests are in flight*.

- Test: an HTTP-level test that fires N patch requests and asserts frame
  latency stays flat; a client-side test is not worth building for this.

### 17 window box on canvas smaller than the panel's patch

**Found:** overlay maths look right — the canvas draws the frame into the rect
derived from `project.video.w/h` (full-res) and boxes in the same source-pixel
space, so proxy resolution should cancel out. The likely explanation is that the
side panel magnifies: a 21 px patch is drawn at `THUMB_PX = 72`, i.e. 3.4×,
against a 1:1 box on the canvas.

**Measure it before changing anything**: place a point on a clip whose
dimensions are known, set a 101 px window, and check the drawn box spans 101
source pixels. If it does, the fix is legibility, not geometry — state the
patch's source-pixel size on the thumbnail and/or draw the canvas box at the
same magnification the panel uses. If it does not, the bug is in `computeTransform`
and this becomes a correctness fix with a regression test.

**OWNS:** `stable/trackers/framecache.py` (new), `stable/trackers/{lk,template,correlation,kalman,particle,meanshift}.py`,
`stable/server/tracker_options.py`, `stable/server/routes/points.py`,
`stable/video/proxy.py`, `web/src/components/PatchPanel.tsx`,
`web/src/hooks/useFrameImages.ts`, `scripts/benchmark_video.py`

---

## W2 — Export: preview screen, target mapping, render library (Opus) — todos 5, 7

The largest item, and the one with the most existing machinery to build on.

### 5 export preview screen with interactive targets

**Found:** the pieces exist. `render/options.py` has `StabilizationTarget`
(`frame` / `centroid` / `point`, plus `anchor_frame`), and
`render/stabilize.py:_target_transforms` composes the target lock as a **pure
translation on top of** `SmoothedMotion.stabilization_transform(t)` — a contract
`tests/server/test_stabilized_output.py::test_the_target_lock_composes_onto_the_sanctioned_warp`
pins, and which must survive this work. `plan_output()` already reports
`crop_scale`, `coverage` and `gaps`, and there is a stabilized-preview endpoint
going through the same `preview_plan()`.

What is missing is (a) seeing the consequence before rendering and (b) targets
the operator places rather than picks.

- **Target placement.** An interactive canvas showing one frame with the output
  rectangle overlaid, drag-and-drop target positions, and thirds/centre guides
  to snap to. A target is `(point_id → (x, y))` in **output** coordinates.
- **N-point mapping.** Two or more pairs generalise the current translation into
  a full similarity: reuse `solver/similarity.solve_similarity(source, target,
  weights, constraints=SolveConstraints(...))` — it is already the weighted
  Umeyama fit, already degrades to translation on a single pair, and already
  takes confidences as weights. Compose its result onto the sanctioned warp the
  same way `_target_transforms` composes the translation today. **Never rebuild
  a transform from channel values** (`from_channels` is banned here for the
  reason W2 documented previously).
  - State explicitly what happens when the pairs disagree (three points cannot
    generally be mapped by a similarity): the fit is least-squares, so report the
    residual per pair in the preview rather than silently averaging.
- **Preview.** Frame + output rect + per-pair residual + `coverage`/`gaps` from
  the existing plan, updating live as targets move. Planning is cheap (no
  pixels), so this can re-plan on drag.
- New screen, not a wider dialog: `web/src/components/ExportScreen.tsx`, with
  `ExportDialog.tsx` reduced to launching it.

### 7 unique, timestamped, manageable renders

**Found:** `ExportVideoRequest.filename` defaults to `"stabilized.mp4"`
(`schemas.py:133`) and the job writes straight to `renders/<filename>`, so every
export **overwrites the last one**. `GET .../export/renders` (list) and
`GET .../export/download/{filename}` already exist; there is no delete.

- Default the name to a timestamp (`stabilized-20260805-142530.mp4`), keep an
  explicit `filename` honoured, and never silently overwrite: if the name is
  taken, suffix it.
- `DELETE .../export/renders/{filename}` with the same path-escape guard
  `download_export` already uses.
- The render library goes in the new export screen: name, size, date, download,
  delete.

**OWNS:** `stable/render/**`, `stable/server/motion.py`,
`stable/server/routes/export.py`, `stable/server/routes/solve.py`,
`stable/cli/stabilize.py`, `web/src/components/ExportDialog.tsx`,
`web/src/components/ExportScreen.tsx` (new), `web/src/components/SolvePanel.tsx`

---

## W5 — New trackers (Sonnet) — todos 27, 30

Additive: new plugin modules only. Do **not** edit `merge.py`, `base.py`,
`registry.py`, `confidence.py` contracts, or the existing plugins (W1 is
changing their caches).

### 27 SAM3-based tracker, for occlusion and reappearance

<https://huggingface.co/docs/transformers/en/model_doc/sam3_tracker>

- Build on `DeepTrackerPlugin` (`trackers/deep.py`): it already gives chunked
  inference, GPU→CPU fallback, the backend seam and the model-resolution
  handling, exactly as `cotracker3`/`tapir`/`raft` use it.
- Weights go through `trackers/weights.py`: **pin the SHA256 you actually
  observe**, and set `license` to what the checkpoint really carries — the model
  manager fails closed on unpinned or non-commercial weights and that is
  deliberate.
- Occlusion is the point of this tracker: map its own presence/visibility output
  onto `confidence.py`'s shared [0, 1] scale, and make `visible=False` mean
  occluded-but-tracked rather than lost. A test with a subject that leaves frame
  and returns is the acceptance criterion — nothing else here distinguishes it
  from the trackers already installed.
- Verify against real weights before claiming it works. Document the install.

### 30 feature-based tracker (SIFT / ORB)

- **SIFT** (mainline OpenCV since 4.4, patent expired) and **ORB** (free) — one
  plugin parameterised by detector, or two thin registrations, whichever reads
  better next to `klt`/`lk`.
- **Not SURF**: it is `opencv-contrib-python-nonfree` only and patent-encumbered.
  Say so in the module docstring rather than shipping a plugin most installs
  cannot import.
- The per-point window settings apply here: declare the descriptor patch size
  under one of the names in `server/tracker_options.SETTINGS` (or add a row) so
  the size slider and the canvas boxes work for it with no UI change.

**OWNS:** new `stable/trackers/sam3.py`, `stable/trackers/features.py`,
`pyproject.toml` (entry points + extras), `CONTRIBUTING.md`, `docs/**`,
new test files

---

## W3 — Track-oriented UI and interaction (Sonnet) — todos 9, 25, 11, 19 — *after W1*

### 9 [BUG] a deleted point's track comes back — and the panel rework

**Found: two independent causes, both real.**

1. **Ids are recycled.** `Project.new_point_id()` returns the *lowest free*
   `pN` (`core/project.py:513`), so deleting `p1` and placing a new point gives
   you `p1` again.
2. **Nothing that is keyed by that id gets cleaned.** `delete_point` unlinks
   `tracks/p1.npz` and its `.report.json` **only** — the retrack sidecars
   (`tracks/p1.<correction_id>.npz`), `p1.violations.json` and the point's entry
   in `tracker_options.json` all survive and are inherited by the new point. On
   the client, `tracks["p1"]` stays in `ProjectContext`'s cache (nothing calls
   `invalidateTrack` on delete), so the **old path is drawn immediately** —
   which is what the report describes.

Fix all three: sweep every per-point artefact on delete, invalidate the client
cache, and make ids monotonic per project so recycling cannot mask a missed
sweep. Monotonic ids are the belt to the sweep's braces — the first point is
still `p1`, so the change is invisible except after a delete. Test: delete,
re-add, assert the new point has no track, no stored window setting and no
violations, and that nothing on disk still carries the old id.

**The rework the report asks for.** Points become **track cards**: one card per
track carrying its own visibility toggles (path / marker / anchors / ghost /
window), replacing the global checkbox row in `Viewer.tsx`. The global toggles
stay as a master row ("all tracks off"), because "see the frame behind
everything" is still a one-click need (todo 7 of the first round).
`web/src/components/TracksPanel.tsx` (new) replaces `PointsPanel.tsx`;
per-track visibility lives in `ProjectContext` so `Viewer` and the cards agree.

### 25 no edit mode

Anchors and moves should work in playback mode. `editing` currently gates
click-to-move and drag (`Viewer.tsx`), and `EditPanel` owns the toggle.

- Drop the mode: dragging a marker always edits; clicking empty canvas with a
  point selected always moves it; `A` always pins. What `EditPanel` keeps is the
  *kind* of edit (offset vs replace, falloff), which is a genuine choice.
- The risk is an accidental edit while scrubbing, so: only the **selected**
  track is editable, a drag must exceed a few pixels to count (there is already
  a 0.5 px no-op guard in `commitMove` — raise it and make it a real hysteresis),
  and every edit is already undoable. Say so in the hint line.

### 11 the Solve tab is unclear

Explain it where the user is: a one-paragraph "what this does" at the top of the
solve panel (tracks → one camera motion → X/Y/R/S curves → smoothing → warp),
the reference frame's role stated in the control that sets it, and what
enabling/disabling a point does to the fit. Prose, not new machinery.

### 19 anchor-on-click behaviour

Pending the open question below. If the answer is "I wanted click-to-place",
the change is: the `+ anchor` button always arms placing, `A` keeps pinning in
place. One conditional in `TracksPanel` + a line in the hint.

**OWNS:** `web/src/components/TracksPanel.tsx` (new, replaces
`PointsPanel.tsx`), `web/src/components/Viewer.tsx`,
`web/src/components/EditPanel.tsx`, `web/src/components/SolvePanel.tsx` *(if W2
has landed; otherwise defer 11's panel copy)*, `web/src/state/ProjectContext.tsx`,
`stable/core/project.py`, `stable/server/routes/points.py` *(delete sweep only —
after W1)*

---

## W4 — Clip range, playback and live tracking (Opus) — todos 1, 3, 15 — *after W3*

### 1 In/Out markers

**Found:** `run_point_tracking` hardcodes `frame_range = (0, video.frame_count)`
(`server/tracking.py`), and export renders the whole clip. The bounded-run
machinery from the first round (`bounded_frame_range`, `stop_at_keyframe`,
`max_frames`) is per-run, not per-project, so this is genuinely new state.

- Store it in a **sidecar**, `clip_range.json` (`stable/server/clip_range.py`,
  new), the same choice `render_options.json` and `tracker_options.json` made —
  the locked v1 project schema stays untouched. `{"in": int, "out": int}`,
  clamped to the clip, `out` exclusive to match every other range in this
  codebase.
- Consumed by: `run_point_tracking` (the run's `frame_range`), the export job
  (render only that span), and the timeline (shaded outside the range).
- UI: draggable I/O handles on `Timeline.tsx`, `I`/`O` keys to set from the
  playhead, and a "whole clip" reset. Show the range in the export screen so it
  is never a surprise.
- Test: a track with an in/out set covers exactly that span and leaves the rest
  `NaN`; an export produces exactly `out - in` frames.

### 3 playback fps

**Found:** the playback loop is a `setInterval` at `1000 / project.video.fps`
(`Timeline.tsx:69-88`), i.e. always the source rate, and it does not compensate
for a frame that arrives late.

- A rate control (¼×, ½×, 1×, 2×, or an fps box) next to the transport.
- While there, make the loop time-based rather than interval-based: at high
  rates `setInterval` silently falls behind and playback runs slower than it
  claims. `requestAnimationFrame` with an accumulator, dropping frames rather
  than drifting.

### 15 watch the tracker work

**Found:** the data is already published — `LivePartialTrack` streams the path
so far and the decoder's cursor, `useLiveTracking` indexes it, `Viewer` draws
the partial path dashed and `JobProgress` reports "frame 812 · run 2/4". What is
missing is the *viewer following along*.

- A "follow the tracker" toggle that scrubs the current frame to the live cursor
  while a run is in flight, and releases as soon as the user scrubs manually
  (never fight the pointer). With W1's patch caching landed, that scrubbing is
  affordable; before it, this todo would make 23 worse — hence the ordering.

**OWNS:** `stable/server/clip_range.py` (new), `stable/server/routes/projects.py`,
`stable/server/tracking.py`, `stable/server/jobs.py`,
`web/src/components/Timeline.tsx`, `web/src/components/JobProgress.tsx`,
`web/src/hooks/useLiveTracking.ts`

---

## Waves

1. **W1, W2, W5** in parallel — disjoint file sets, all three unblock work below.
2. **W3** — needs W1's `routes/points.py` and `PatchPanel` changes to have landed.
3. **W4** — needs W3's `ProjectContext`/`Viewer`/`Timeline` shape, and W1's
   caching before "follow the tracker" is anything but a stress test.

## Open questions

1. **Todo 19.** Is "it now automatically adds it at the current track position" a
   report of the new `A`/`+ anchor` behaviour, or a complaint? If you want the
   button to always ask for a click, say so and W3 will change it — `A` would
   keep pinning in place either way.
   Answer: It is a report, not a complaint. The current behaviour is fine. We'll re-evaluate if we get complaints from users.
2. **Todo 5, multi-target.** With three or more pairs a similarity generally
   *cannot* satisfy all of them. Preferred behaviour: least-squares fit with
   per-pair residuals shown (recommended, and what the solver already does), or
   refuse more than two pairs?
   Answer: Least-squares fit, show residuals.
3. **Todo 1.** Should In/Out bound *tracking* as well as export, or only export?
   The plan assumes both, since "it's not always the full length we want to
   track" reads that way — but it means a track outside the range is discarded
   rather than kept and hidden.
   Answer: Bound tracking as well as export.

---

# Execution report

All five workstreams executed in one pass (no parallel agents). Suite at
hand-off: **2052 passed, 2 failed** — both the pre-existing
`tests/test_phase3_acceptance.py` rotation-tolerance failures noted in the
baseline above, confirmed unchanged on a clean tree. `ruff check`,
`ruff format --check` and `mypy stable` clean across every file touched;
`npm run build` green and `stable/server/webdist/` regenerated.

## W1 — memory and responsiveness

**21 — mean-shift's memory.** Not mean-shift: all six classic trackers kept an
unbounded `dict` of colour-converted frames for the length of a run. New
`stable/trackers/frames.py` points the *existing* byte-budgeted LRU
(`stable/video/cache.py`) at those conversions — one eviction policy for the
application rather than a second one to drift. Measured on a 400-frame 1080p
mean-shift run: **2488 MB → 199 MB** peak. `tests/test_tracker_frame_cache.py`
pins the bound *and* that eviction never changes a track (every plugin run
twice, once with room for one frame).

**23 — scrubbing.** Patch thumbnails were cropped from the **full-resolution
source**, one per point per frame, behind the decoder's single lock — so proxy
frames queued behind an exhausted threadpool. Patches now come from the proxy
tier by default (`source=original` where texture is being judged), are cached
server-side keyed by a mtime-derived stale key with `ETag`/`Cache-Control`, and
the panel debounces while the frame is moving. Proxy frames gained cache headers
too. `tests/server/test_patch_serving.py` asserts frame latency does not depend
on patch traffic.

**17 — the window box.** Measured, not changed: the overlay and the frame share
one transform, so proxy resolution cancels and the geometry was right. The panel
magnifies (a 21px patch drawn at 72px), which is what "smaller on the canvas"
was. It now states its own size and magnification.

## W2 — export

**5 — the export screen.** New `web/src/components/ExportScreen.tsx` (a tab in
the export dialog) drawing the output rectangle *on the frame*, with draggable
targets and thirds guides, backed by a new `GET .../export/preview` that plans
rather than renders. Multi-point mapping generalises the single-target
translation into `solve_similarity` composed on the sanctioned warp — two pairs
really do remove rotation and scale (`tests/test_target_mapping.py`), three or
more are least-squares with the per-pair miss reported, as decided.

Two bugs the live run found and the tests now cover: a frame where every pair
has zero confidence made the weighted fit raise (`weights sum to zero`) and
killed the whole render; and target destinations live *before* the crop/zoom
fit, so the preview now sends that frame's warp for the screen to place
crosshairs with, and reports `target`/`landed` in one space.

**7 — renders.** Timestamped, never overwriting (`-2`, `-3` on collision), with
`DELETE .../export/renders/{name}` and a library in the screen.

## W5 — trackers

**30 — SIFT/ORB** (`stable/trackers/features.py`): matched against the *anchor*
frame rather than the previous one, so a failed frame is a gap rather than
inherited drift. Detection is whole-frame with positional filtering (a cropped
window changes SIFT's scale space). Not SURF, and the docstring says why.
`orb` is in the shared suite; `sift` is not, because that suite's clip is a
gradient plus one square where SIFT finds a single keypoint — its contract is
asserted in `tests/test_feature_trackers.py` against textured footage instead,
including the occlusion-and-recovery case.

**27 — SAM 3** (`stable/trackers/sam3.py`): built on the *video* model
(`Sam3TrackerVideoModel` + `propagate_in_video_iterator`), which is the
memory-based propagation the occlusion requirement actually needs. The point
rides the mask's motion rather than snapping to its centroid, and
`object_score_logits` becomes the occlusion signal.

**Verification level, stated plainly:** the class and method names, signatures
and output fields were checked against `transformers` 5.14.1 (installed), and
`Sam3Backend`'s own logic is tested against a faithful stub
(`tests/test_sam3_tracker.py`). It has **not** been run on real weights:
`facebook/sam3` is a gated repo, and accepting Meta's licence is the operator's
decision, not an agent's. The plan's "verify against real weights" step is
therefore **not done** and is the first thing to do with it.

## W3 — track-oriented UI

**9** had three causes and all three are fixed: `delete_point` now sweeps every
file named for the point (retrack runs, violations sidecar, stored settings),
`new_point_id` counts past the highest id in use, and the client drops its
cached track. Note the honest limit, tested as such: emptying a project does
restart at `p1`, which is safe *because* the sweep leaves nothing to inherit.

`PointsPanel` became `TracksPanel`: one card per track, each with its own
overlay toggles overriding the global defaults (`overlaysFor` in the context).

**25** — no edit mode. The selected track is always draggable and always
click-to-move; only the selected one, so a scrub cannot become a stray edit.
**11** — the Solve panel now opens with what a solve *is*, and each control says
what it does.

## W4 — range, playback, live

**1** — `stable/server/clip_range.py` (`clip_range.json` sidecar) bounds both
tracking and export, as decided. `I`/`O` set the markers, the timeline dims
what is outside them. **Known interaction:** the solve is measured *from* its
reference frame, so a reference frame outside the range leaves it with nothing
to solve; the solve already refuses with that message and the timeline warns,
rather than a range silently moving the reference frame.

**3** — playback rate (¼x–4x) on a `requestAnimationFrame` accumulator, so a
rate that cannot be sustained drops frames instead of quietly running slow.
**15** — "follow tracker" keeps the playhead on the live cursor and releases on
the first manual scrub.

## Live verification

Against a running server, on the bundled clip: range `[20, 70)` → tracking
covered exactly frames 20–69 (50 of 120) and the export wrote 50 frames;
`sift`/`orb`/`sam3` registered and an ORB track completed; the preview reported
128x96 output, 0.971 coverage, no gaps; a two-point mapping reported
`uses_rotation_and_scale: true` with **0.000 px** residual on both; two exports
produced two timestamped files and one was deleted; deleting a track swept its
files and its replacement started empty.

## Not done

- SAM 3 against real weights (gated licence — see above).
- `transformers` (and a `tokenizers` pin) were installed into the working venv
  to verify the SAM 3 API. It is declared as the `sam3` extra in
  `pyproject.toml`; uninstall it if you would rather not carry it.
