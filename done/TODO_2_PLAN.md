# Implementation plan — third feedback round

Source: `todo_2` (5 items). Same conventions as `TODO_PLAN.md` / `NEW_TODO_PLAN.md`:
each work item states what was **found in the code**, not what was assumed, and
the suite stays green throughout (`pytest -m "not integration"`, `ruff check`,
`ruff format --check`, `mypy stable`, `npm run build`, then
`python scripts/build_frontend.py`).

Baseline: the two pre-existing `tests/test_phase3_acceptance.py` rotation-tolerance
failures noted in the previous round are still there and are still nobody's here.

This round is small enough to run as one pass — the five items touch four
disjoint areas (export UI, audio muxing, timeline transport, tracker progress,
canvas drawing), so there is nothing to serialise.

---

## T1 — One export screen; retire the Video tab

> "The new Preview UI in the video export is good. However we have to go back to
> the existing Video tab to do the export."

**Found:** `ExportDialog.tsx` has three tabs — `data`, `video`, `preview`. The
`preview` tab is `ExportScreen.tsx` (the frame, the output rectangle, draggable
targets, the render library). The `video` tab holds *everything else*: filename,
codec, CRF, the `frame`/`centroid`/`point` target modes with their anchor frame,
crop size + centre, border policy, border fill (mode/colour/blur/tint/feather),
the **Export** button, `JobProgress`, the result summary, and a second copy of
the render list. So the operator composes on one tab and exports on another, and
the render library exists twice.

The two tabs are not redundant in an interesting way — the preview screen simply
never grew the controls. Everything the video tab does has an equivalent home in
the screen's side column.

**Work:**

- Move every control out of `ExportDialog`'s video tab into `ExportScreen`'s side
  column, grouped: **Output** (stats + crop) → **Stabilise on** (target) →
  **Border** (policy + fill) → **Encode** (filename, codec, CRF) → **Export**
  (button, progress, result) → **Renders** (the library that is already there).
- Delete the `video` tab. `ExportDialog` becomes a two-tab shell: **Export** (the
  screen) and **Data** (the CSV/JSON/AE links, unchanged).
- The target control has to reconcile two vocabularies. `TARGET_MODES` is
  `frame | centroid | point | map`; the screen only ever wrote `map` (and `frame`
  when the last pair was removed), the video tab only ever wrote the other three.
  One **Stabilise on** select now offers all four, and the section below it
  changes with the mode: placed crosshairs for `map`, a point picker for `point`,
  an anchor-frame field for `point`/`centroid`, help text for `frame`. Pressing
  **+ target** switches the mode to `map`, because that is what placing a target
  means. Removing the last pair goes back to `frame` — `StabilizationTarget`
  refuses a `map` with no pairs, so this is required, not cosmetic.
- Show the **In/Out range** in the screen (T2 is the reason it matters), read
  from the context's `clipRange`, so what will be written is stated where the
  export is started.
- The `transparent` border still refuses to export; the explanation moves with
  the button.
- Widen the side column (320 → 380 px) and give the export sections their own
  styles. The modal is already widened by `.modal:has(.export-screen)`.

**Owns:** `web/src/components/ExportDialog.tsx`,
`web/src/components/ExportScreen.tsx`, `web/src/styles.css`.

---

## T2 — [BUG] the export ignores In/Out and pads with the last frame

> "I set In/Out markers and the exported video was still the full length … the
> remaining frames were just the last frame repeated until the end."

**Found: the video stream is already correct; the *audio* is not.**

`export_video_route` computes `_render_range()` from `clip_range.json` and hands
it to `render_stabilized(frame_range=…)`, which yields exactly `out - in` frames
and encodes them with `pts = 0…N-1`. That half works, and
`tests/server/test_clip_range.py::test_the_export_writes_exactly_the_ranged_frames`
proves it — by counting **video** packets, which is why the bug survived it.

The audio is copied by `stable/video/encode.py::_copy_audio`, which demuxes
**the entire source stream** with its original timestamps; `_transcode_audio`
does the same. So a 120-frame clip exported as `[20, 70)` gets a 50-frame video
track and a 120-frame-long audio track in one container. The container's
duration is the longer of the two, and a player with nothing left to decode
holds the last decoded frame on screen — exactly the report. The audio is also
offset by the in point, so it would be out of sync as well as too long.

**Work:**

- `export_video` gains `audio_range: tuple[float, float] | None` — the seconds of
  source audio to carry over. `None` keeps today's behaviour byte for byte.
- `_copy_audio` seeks to the start, drops packets outside the window and rebases
  their `pts`/`dts` to zero. `_transcode_audio` filters decoded frames the same
  way before resampling (it already drops timestamps, so the encoder relays the
  timeline itself).
- `render_stabilized` derives the window from `frame_range` and the clip's fps,
  and passes `None` when the range is the whole clip — so the untrimmed path is
  not merely equivalent but *identical*, which is the regression that matters.
- Trimming is packet-granular for passthrough (an audio packet is ~20-40 ms). It
  is stated in the docstring rather than papered over; sample-exact trimming
  means re-encoding, which is a worse trade than a frame of slack.

**Tests:**

- `tests/test_video_encode.py`: an export with `audio_range` produces an audio
  stream whose duration matches the range, not the source; without it the audio
  is untouched.
- `tests/server/test_clip_range.py`: the ranged export's **audio** duration
  matches its **video** duration (the assertion that would have caught this).

**Owns:** `stable/video/encode.py`, `stable/render/stabilize.py`,
`tests/test_video_encode.py`, `tests/server/test_clip_range.py`.

---

## T3 — Jump to the In/Out markers

**Found:** `Timeline.tsx` sets the markers (`[ In`, `Out ]`, and the `I`/`O`
keys in `ProjectContext`) but nothing navigates to them; with the range off
screen at the far end of a long clip, checking it means scrubbing for it.

**Work:**

- Two buttons beside the existing marker buttons: jump the playhead to `in`, and
  to `out - 1` (the last frame *inside* the range — `out` is exclusive and
  points at a frame that will not be exported, which is not where the operator
  means to land).
- `Shift+I` / `Shift+O` do the same from the keyboard, which is the convention
  every NLE uses. The unshifted keys keep setting the markers, so the existing
  muscle memory is untouched; today `Shift+I` sets the in point, which nothing
  relies on.
- Both are disabled when no range is set — with the whole clip in force there is
  no marker to go to.
- The shortcut line under the viewer gains `⇧I/⇧O go to in/out`.

**Owns:** `web/src/components/Timeline.tsx`,
`web/src/state/ProjectContext.tsx`, `web/src/components/Viewer.tsx` (hint text).

---

## T4 — [BUG] tracking progress sits at 0%

> "shows 'running 0% frame xxxx' — the frame number is increasing but the
> percentage is not."

**Found:** the classic trackers report progress **once per query point**, at the
end of the run:

    progress((qi + 1) / len(queries), f"point {qi + 1}/{len(queries)}")

— `lk.py:230`, `template.py:183`, `correlation.py:210`, `kalman.py:280`,
`particle.py:250`, `meanshift.py:244`, `features.py:293`. Server-side tracking
gives every run exactly **one** query (`plugin.track(video, [spec.anchor], …)`),
so that line fires once, when the run is already over. In between, the fraction
never moves and the message is empty — which is precisely "running 0%", the
status standing in for a message that was never sent. The frame number does move
because it comes from somewhere else entirely: the decoder telemetry cursor in
`jobs.py::_FrameCursor`.

The deep trackers are fine (`deep.py:545` reports per inference chunk), which is
why this only shows up on the classic ones.

**Work:**

- New `stable/trackers/progress.py`: `planned_steps(queries, frame_range,
  direction)` — how many frames a `track()` call will step through — and
  `RunProgress`, whose `steps(range)` wraps an index range and reports one
  frame's worth of progress per iteration.
- All seven classic trackers already share the same shape: a `_run_direction`
  helper taking `indices: range`, and `indices` is *only ever iterated*. So the
  reporter wraps the range at the call site and **no `_run_direction` signature
  changes** — seven files, a few lines each, no per-tracker arithmetic to get
  wrong seven different ways.
- The per-query line goes; `progress(1.0, "done")` stays.
- Fractions are reported with an **empty message**, so
  `tracking.py::run_point_tracking`'s `sub_progress` keeps supplying `run 2/4`
  as it does today — the tracker knows where it is in *its* run, not where its
  run sits in the job, and it should not pretend otherwise.
- `JobProgress.tsx` stops repeating the run counter it already shows via
  `job.message`, so the line reads `run 2/4 (37%) · frame 812` instead of naming
  the run twice.

**Tests:** `tests/test_tracker_suite.py` (the shared plugin suite) gains a
contract test: a run over a multi-frame range reports **more than one** fraction,
strictly increasing, ending at 1.0. That is the assertion whose absence let this
ship.

**Owns:** `stable/trackers/progress.py` (new),
`stable/trackers/{lk,template,correlation,kalman,particle,meanshift,features}.py`,
`web/src/components/JobProgress.tsx`, `tests/test_tracker_suite.py`.

---

## T5 — Colour the paths by track, or by confidence

**Found:** `Viewer.tsx` strokes each path segment with
`confidenceColor((c[f-1] + c[f]) / 2)` and fills the current-frame marker with
`confidenceColor(c[frame])`. Both are per-*frame* colours, so every track on the
canvas is drawn from the same red→amber→green ramp and they are impossible to
tell apart. The per-track colour (`pointColor(point.id)`) already exists and is
used for anchor pins, the live partial path, timeline pins and panel swatches —
it is only the path and marker that ignore it.

**Work:**

- `pathColor: "confidence" | "track"` in `ProjectContext` (session state, like
  `overlays` — the other overlay preferences do not persist either), with a
  select in the viewer toolbar beside the overlay checkboxes.
- `track` mode strokes the whole path in one pass in the point's own colour, and
  fills the marker to match. `confidence` mode is exactly today's per-segment
  drawing. The marker keeps its white (or amber, while dragging) outline either
  way, so "which point is selected" is unaffected by the choice.
- The timeline heat strip stays confidence-coloured: it is *for* the selected
  track's confidence, and it is the readout the toggle lets the canvas give up.
- The toggle's tooltip says what is being traded, since "green" means two
  different things in the two modes.

**Owns:** `web/src/state/ProjectContext.tsx`, `web/src/components/Viewer.tsx`.

---

## Order

T2 and T4 are backend fixes with tests; T1, T3 and T5 are frontend. T1 touches
`ExportScreen`/`ExportDialog`, T3 touches `Timeline`/`ProjectContext`, T5 touches
`Viewer`/`ProjectContext` — the only shared file is `ProjectContext`, and the two
additions to it (`pathColor`, the shifted key branch) do not overlap. One pass,
then rebuild the bundle once at the end.

## Verification

Beyond the suite: run the server on the bundled clip and check the two bugs
against real output — an export with In/Out set must produce a file whose
**duration** (not only its frame count) matches the range, and a classic-tracker
run must show a percentage that climbs.

---

# Execution report

All five items delivered in one pass. Suite at hand-off: **2068 passed, 7
failed** — every failure is `ModuleNotFoundError: torch` in
`tests/test_sam3_tracker.py` and one case in `tests/test_deep_trackers.py`,
confirmed unchanged by stashing this round's work and re-running. PyTorch is not
installed in this environment; nothing here touches it. `ruff check` and
`mypy stable` clean, `npm run build` green, `stable/server/webdist/` regenerated.

The two `tests/test_phase3_acceptance.py` rotation failures the baseline
mentioned now pass.

## T2 — the export length (the real bug)

The video stream was **already correct**: `_render_range` and
`render_stabilized(frame_range=…)` wrote exactly `out - in` frames, and the
existing test proved it — by counting video packets. The audio was the problem.
`_copy_audio` demuxed the *whole* source stream with its original timestamps, so
a 120-frame clip exported as `[20, 70)` got 50 frames of video and 120 frames'
worth of audio in one container. A container's duration is the longer of its
streams, so the file *played* for the length of the source and every player held
the last decoded frame for the remainder. That is "the remaining frames were
just the last frame repeated" exactly — and it means the audio was out of sync
by the in point as well as too long.

`export_video` now takes `audio_range` (seconds of source audio to carry) and
both writers honour it: passthrough seeks, drops packets outside the window and
rebases the rest onto zero; transcode filters decoded frames before resampling.
`render_stabilized` passes `None` for a whole-clip render, so that path is
*identical* rather than merely equivalent.

The trim is packet-granular for passthrough — an audio packet is 20-40 ms and
cutting inside one means re-encoding it — so the copied track can start up to
one packet late and end up to one packet early. Stated in the module docstring,
not papered over.

**Measured live**, on a 180-frame 24 fps clip with an 8-second audio track,
exported as `[40, 100)`: 60 frames of video (2.500 s), audio **2.496 s**,
container duration **2.514 s**. Before the fix the same export's audio measured
**2.005 s against an expected 0.417 s** on the suite's clip — the regression
test was run against the reverted code to confirm it catches this, rather than
assumed to.

## T4 — the stuck percentage

Seven classic trackers reported progress once per *query*, at the end of the
run. Server-side tracking gives each run exactly one query, so that was one
report, fired when the run was over — "running 0%" for the whole of a long
track, with the frame counter beside it climbing because it comes from the
decoder telemetry cursor instead.

New `stable/trackers/progress.py` counts in frames: `planned_steps` says how
many a `track()` call will take, `RunProgress.steps` wraps an index range and
reports one frame's worth per iteration. All seven trackers already shared the
same shape — a `_run_direction` taking `indices: range`, only ever iterated — so
the reporter wraps the range at the call site and **no helper signature
changed**. Fractions carry no message, so the tracking job's own "run 2/4"
survives; `JobProgress.tsx` stopped appending that counter a second time.

`tests/test_tracker_suite.py` gained the contract test whose absence let this
ship: on **one** query, progress must be heard from more than once before the
final 1.0. Against the reverted `lk` it fails with `assert 1 > 1`. Live through
the HTTP route, a run now reports 0.000 → 0.944 across nine distinct fractions
with the message `run 1/1`.

## T1 — one export screen

The Video tab is gone. Every control it held — filename, codec, CRF, target
mode, anchor frame, crop size and centre, border policy, fill mode/colour/blur/
tint/feather, the Export button, progress and result — now lives in the export
screen's side column, beside the preview that shows what they do. The render
library existed on both tabs; now it exists once. `ExportDialog` is a two-tab
shell: **Video** (the screen) and **Data** (the CSV/JSON/AE links, untouched).

The one design decision worth naming: the screen only ever wrote the `map`
target mode and the old tab only ever wrote `frame`/`centroid`/`point`. There is
now one "Stabilise on" select offering all four, with the section under it
changing to match — crosshairs for `map`, a point picker for `point`, an anchor
frame for `point`/`centroid`. "+ target" switches the mode to `map`, because
placing a target *is* that choice; removing the last pair goes back to `frame`,
which is required rather than cosmetic since `StabilizationTarget` refuses a
`map` with no pairs.

The screen also states what the export will write — the In/Out range, or "the
whole clip" — which is where T2's bug should have been visible before spending
minutes on a render.

## T3 — jumping to the markers

Two buttons (`⇤`, `⇥`) beside the ones that set the markers, and `⇧I`/`⇧O` from
the keyboard, which is the convention NLEs use. `⇧O` lands on `out - 1`: `out`
is exclusive and names a frame that will be neither tracked nor exported, which
is not where "go to the out point" means to land. Both are disabled with no
range set, since there is then no marker to go to.

## T5 — path colour

`pathColor` in the context, a select in the viewer toolbar. `by track` strokes
the whole trail in the point's own colour (one path, not one per segment) and
fills the current-frame marker to match — the marker follows the same choice,
because telling tracks apart is little use if the trails differ and the dots on
this frame do not. `by confidence` is exactly the previous drawing. The
timeline's heat strip stays confidence-coloured either way, so the signal is
never entirely given up.

## Not done

- Nothing from `todo_2` was left out.
- `scripts/build_frontend.py` could not run end to end here: its `npm ci` step
  fails with `EPERM` unlinking a locked `esbuild.exe`. `npm run build` in `web/`
  succeeded and the copy half of that script was performed directly, so
  `stable/server/webdist/` is current — but the script itself is unverified on
  this machine.
