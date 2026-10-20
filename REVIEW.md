# Stable — consolidated review

One document for every review and feedback round so far, plus the full-project
review of **2026-10-04**. It replaces the per-round plan files —
`done/TODO_PLAN.md`, `done/NEW_TODO_PLAN.md`, `done/TODO_2_PLAN.md`,
`done/TODO_3_PLAN.md`, `TODOS_4_PLAN.md`, `TODOS_5_PLAN.md`, `TODO_6_PLAN.md`,
`TODO_8_PLAN.md` and `TODO_9_PLAN.md`. Their full text is in git history; the
last commit that has all of them is `25f4cf5` (e.g.
`git show 25f4cf5:TODO_9_PLAN.md`). The raw request notes (`todos_4`, `todo_9`,
`done/todos`, …) are left where they are: they are the user's input, not a
review.

`PLAN.md` is **not** folded in. It is the architecture and contract handover
that the code, the tests and `MANIFEST.in` cite, not a review.

The conventions every round has followed still hold: state what was **found in
the code**, not what was assumed; keep the suite green (`pytest -m "not
integration"`, `ruff check`, `ruff format --check` on touched files,
`mypy stable`, `npm run build`, then `python scripts/build_frontend.py` so the
packaged `stable/server/webdist` matches the source).

## Contents

1. [Where things stand](#1-where-things-stand)
2. [Git state and the `fable_rework` branch](#2-git-state-and-the-fable_rework-branch)
3. [The feedback rounds so far](#3-the-feedback-rounds-so-far)
4. [Review findings, 2026-10-04](#4-review-findings-2026-10-04)
5. [Prioritised fix list](#5-prioritised-fix-list)
6. [Executed in this pass](#6-executed-in-this-pass)
7. [Carried forward](#7-carried-forward)
8. [Ocean and SiamDW from TracKit, 2026-10-20](#8-ocean-and-siamdw-from-trackit-2026-10-20)

---

## 1. Where things stand

Feature-complete through PLAN.md's Phase 6, plus ten feedback rounds and a
storage-path refactor on top. The architecture has held up well: one result
type for every tracker, a non-destructive correction layer, a frame WebSocket
that keeps pixels and overlay geometry in one message, and reference-counted
bitmaps on the client. The problems found in this review are almost all
**cost that grows with clip length** in paths that run once per frame, and a
handful of **input and launch regressions**.

Measured on this machine (Windows 11, Python 3.11.9, Node 24), before any
change and after this pass (§6):

| Check | Before | After |
|---|---|---|
| `pytest -m "not integration"` | **2,332 passed, 3 failed, 1 error** | 2,350 passed, 2 failed (the two solver-tolerance tests below) |
| `pytest -m integration` | not run | 8 passed |
| `ruff check stable tests` | clean | clean; every touched file also `ruff format`-clean |
| `mypy stable` | clean (92 files) | clean (92 files) |
| `npm run build` | clean (297.8 kB JS, 94.4 kB gzip) | clean (300.1 kB JS, 95.4 kB gzip) |
| packaged `stable/server/webdist` | **stale** — commit `7129d05` changed `ModelManager.tsx` without rebuilding it | rebuilt from this source |

The four failures before:

* `test_phase3_acceptance.py` ×2 — the solver rotation tolerance
  (`2.079e-4` against `2e-4`). Recorded since round 7; numerical, not a
  regression. Still open (§7).
* `test_cli_serve.py::test_serve_subcommand_is_registered` — **new**. Commit
  `7129d05` changed the `stable serve` default host to `0.0.0.0` and the test
  to expect `localhost`. See U2.
* `test_frame_stream.py` setup **error** — `PermissionError: [WinError 5]` from
  `os.replace` in `Project.save`. A real Windows race, not a flaky test. See R1.

---

## 2. Git state and the `fable_rework` branch

At the start of this review: `main` = `origin/main`, clean tree, no stashes.
One other branch, `fable_rework` (local and remote, identical), one commit
(`3d34e9d`, 2026-08-06) diverging from `main` at `73da44b`. `main` had five
commits since.

**What `fable_rework` is.** An implementation of the *original* wording of
`todo_6`: "remove the concept of anchor points … just have keyframes", plus an
auto-update that re-tracks a point whenever a keyframe is added, moved or
deleted, "a magic, intuitive experience". It renamed the API to `/keyframes`,
added auto-track (default on), an atomic `PATCH` move, keyframe provenance
(`source: pin | correction`) and `JobManager.cancel_queued_matching`.

**Why it was not merged as code.** Three days later `todo_6` was rewritten on
`main` (`03c6e20`) to the opposite decision — "remove the concept of keyframes
and just have anchor points" — with the auto-update and "magic" paragraphs
deleted. `main` implemented that version (round 7 below) and built rounds 9
and 10 and the storage refactor on it. Taking the branch's content would
reverse the user's own later decision and conflict with everything since.

**What was done.** `fable_rework` is merged into `review/perf-ux-pass` with the
`ours` strategy (`25f4cf5`): its commit is in history and the branch now shows
as merged, and `main`'s tree is unchanged. What the branch had that `main`
lacks, and why each stays out:

| `fable_rework` feature | Decision |
|---|---|
| "keyframe" as the one word | Declined — the rewritten `todo_6` chose "anchor" |
| Auto-track on every anchor edit (default on) | Declined — dropped from the rewritten `todo_6`; `main` shows "⟳ anchors changed — re-track" instead, because a full track is minutes of GPU and must not start behind the operator's back |
| Atomic `PATCH …/keyframes/{frame}` move | Not needed — `main` moves an anchor within its frame with one `POST` (replace); moving one *across* frames is not a feature |
| `JobManager.cancel_queued_matching` (supersede queued runs) | Only useful with auto-track; could come back with job cancellation (U4) |
| Deleting a promoted anchor unlinks the retrack's cached `.npz` | Conflicts with `main`'s undo: a deleted anchor (and its run) can be restored, so the file must stay |

Not pushed anywhere. The branch can be deleted once `review/perf-ux-pass` is
merged.

---

## 3. The feedback rounds so far

| Round | Request | Theme | Status |
|---|---|---|---|
| 1 | `done/todos` (28 items) | Anchor correctness, stabilised output, viewer UX, performance and live feedback, seven new trackers, templates/export/sessions | Done, except #22's checkpoint pinning |
| 2 | `done/new_todos` | Memory and scrubbing, export preview with target mapping, render library, SAM 3 and SIFT/ORB, track-oriented panel, In/Out, playback rate, watching a run | Done; SAM 3 never run on real (gated) weights |
| 3 | `done/todo_2` | One export screen, export honours In/Out, jump to markers, tracking %, path colour | Done |
| 4 | `done/todo_3` | Export as a tab, live curves, smoothed path on canvas, crop unsetting targets, cursor/tool palette, collapsible cards, solve selection, delete from image, export queue | Done |
| 5 | `todos_4` | Side-by-side output, frame WebSocket, overlay/picture sync | Done; the "· showing N" frame label is no longer in the UI (U9) |
| 6 | `todos_5` | Detached-bitmap crash, play range, live crop preview, targets on the output canvas, Edit tab removed, patch cache warmer | Done |
| 7 | `todo_6` (rewritten) | One word, *anchor*; every anchor deletable; track forwards/backwards from an anchor; "anchors changed" | Done (`fable_rework`'s keyframe version superseded, §2) |
| 8 | `todo_7` | Search radius, cancel tracking, per-track eye, hidden = not editable, one bidirectional button, refresh all, delete from canvas, full-res tracking, size interpolation | **Partial** — never had a plan file; audited in §3.8 |
| 9 | `todo_8` | Add-anchor stays armed, ghost removed, replace by default, disclosure arrow, track eye, track colour, labels, separators | Done; its one hole (U6) closed in this pass |
| 10 | `todo_9` | Per-track smoothing, composite tracks, play range on In/Out, clean output view, ~30 fps default, render queue | Done |
| — | commit `7129d05` | Data root beside the install, legacy `~/.stable` migration, video-path repair | Done; introduced U2 and the stale `webdist`, both fixed in this pass |

### 3.1 Round 1 — `done/todos`

* **Anchors are ground truth.** `tracking.effective_anchors()` (pins plus every
  correction that states a position) feeds both full tracks and retracks;
  `merge.enforce_anchor_agreement()` stops a run at the first anchor it misses
  by more than 6 px and reports it, drawn red on the timeline. Corrections are
  promoted to anchors inside the correction layer, so undo un-promotes for
  free. Bounded runs (`to next anchor`, `N frames`).
* **The "broken" export** was a successful re-encode: no smoothing and no
  target make the warp the identity. Plans are now checked at submit time and a
  no-op is refused with an explanation. Targets, crop scale and border fill
  (solid / stretch / mirror / blur) added.
* **Performance:** the shared decoder was replaced by a pool (4 concurrent
  runs: 34.6 → 485 frames/s), proxy generation went multi-threaded (1.71×), a
  close-while-decoding crash fixed, live partial paths and the frame cursor
  streamed from running jobs, `stable serve` falls back to a free port.
* **Trackers:** correlation, Kalman, particle, mean-shift, KLT, RAFT,
  plus CoTracker3/TAPIR made reachable.
* **Open:** #22 asked for the deep checkpoints to be pinned to observed
  SHA-256s. Both still ship unpinned (§7).

### 3.2 Round 2 — `done/new_todos`

* Classic trackers kept an unbounded dict of converted frames: bounded LRU,
  2,488 MB → 199 MB peak on a 400-frame 1080p mean-shift run.
* Scrubbing stalled because patch thumbnails were cut from full-res source per
  point per frame; patches now default to the proxy, are cached with ETags, and
  the panel debounces.
* Export preview screen with draggable targets; two targets solve rotation and
  scale, three or more are least-squares with the per-target miss shown.
  Timestamped, deletable renders.
* SIFT/ORB tracker matched against the anchor frame; SAM 3 video tracker
  (**not run on real weights** — the repo is gated; §7).
* Track-oriented panel, no edit mode, In/Out markers bounding tracking and
  export, playback rate on a time-based loop, "follow tracker".

### 3.3 Round 3 — `done/todo_2`

One export screen (the Video tab retired); the export honours In/Out (it used
to pad with the last frame); jump-to-marker buttons and `⇧I`/`⇧O`; tracking
percentage fixed; paths coloured by confidence or by track.

### 3.4 Round 4 — `done/todo_3`

Export became a sidebar tab drawing on the real viewer; the curve editor and
smoothing panel share one copy of the smoothing state; the smoothed path is
drawn on the canvas; adjusting the crop no longer unsets targets; the cursor
and a tool palette say what a click does; collapsible cards; solve selection
moved to the Solve tab; delete an anchor from its image; exports queue.

### 3.5 Round 5 — `todos_4`

* The overlay is drawn for the frame **on the canvas** (`viewFrame`), not the
  playhead — the "no-negotiation" sync requirement.
* `stable/server/frames.py` is the one producer of a frame and its geometry;
  `/ws/projects/{id}/frames` coalesces requests (only the newest *want*
  survives), serves wants before prefetch, and puts pixels and geometry in one
  binary message. HTTP stays as the fallback.
* Side-by-side output canvas; `/export/preview` asked once per settings change
  rather than once per frame.
* **Drift since:** W1 promised a frame label reading `frame 259 / 500 ·
  showing 244` while a frame is in flight. The current label shows only the
  playhead (U9).

### 3.6 Round 6 — `todos_5`

Reference-counted `Decoded` bitmaps ended the "image source is detached" blank
app; drawing is guarded as a second lock. Play range; crop slider commits while
dragging (one request in flight, one queued); targets drawn and draggable on the
output canvas via `output_warp`; the Edit tab removed (drag mode in the toolbar,
per-track edits on the card, anchors draggable); a single-worker patch warmer
fills the cache when an anchor is made (and caught a Windows `os.replace`
failure on the way — the same class of bug as R1).

### 3.7 Round 7 — `todo_6`

"Anchor" everywhere (`stop_at_keyframe` kept as a deprecated alias); every
anchor deletable, taking the corrections it implied with it, through undo; a
never-tracked point gets an empty baseline so "place an anchor, track forwards"
works; `anchors_changed` flags a track computed from a different anchor set;
the second anchor list in the patch panel removed.

### 3.8 Round 8 — `todo_7` (audited now; no plan was ever written)

| Item | Status in the code |
|---|---|
| Replace "pyramid levels" with a search radius | **Partial.** LK still exposes `max_level` as its control; the panel adds a derived "Search reach ±N px" beside it (`tracker_options.py`, `PatchPanel.tsx`) |
| Cancel tracking that is running | **Not done.** `JobManager.cancel` refuses anything past `queued` (U4) |
| One show/hide toggle per track | Done in round 9 (`TrackVisibility.tsx`) |
| No edits to a hidden track | Hit tests skipped hidden tracks, but click-to-move and the add-anchor tool still edited the *selected* one (U6) — **closed in this pass** (§6.3) |
| Track forwards and backwards with one button | **Partial.** "Track all N" runs every anchor both ways; there is no single ◀▶ from one anchor, though the API already accepts `direction: "bidir"` |
| Refresh all tracks between anchors | **Not done** as a multi-track action; per track only |
| Delete anchors from the canvas | Done in round 7 |
| Proxy or full resolution for tracking? | Full resolution — jobs track on `workspace.get_video_source()`, never the proxy |
| Interpolate the window size between anchors | **Not done** |

### 3.9 Round 9 — `todo_8`

The add-anchor tool stays armed; the raw-path ghost removed; "Drag makes"
defaults to hard replace; disclosure arrow on export target cards; a per-track
eye as a layer above the per-element overrides; a persisted per-track colour
(`PointSpec.color`); "Colour Paths by"; toolbar separators.

### 3.10 Round 10 — `todo_9`

A `smooth` correction per track (gaps stay gaps, confidence untouched); a
composite track as the per-frame mean of others (`stable/core/derive.py`, not
time-warped — corrected after real footage); play range switches on when a
range is first set; a clean output view; the playback rate opens near 30 fps;
a stored render queue with an ordered runner and measured estimates.

---

## 4. Review findings, 2026-10-04

IDs are stable so later rounds can refer to them. **Priority**: high = do now,
medium = next, low = when convenient. **Effort**: S < ½ day, M ≈ 1–2 days, L > 2
days. Line numbers are as of `25f4cf5`.

### 4.1 Performance

**P1 · Export tab: every streamed frame recomputes the whole clip — high, M**
`routes/ws.py` `_geometry_payload` runs for each frame the export tab draws. It
re-reads and re-parses `project.stable.json`, `solve.json` and the render
options; `frames.load_smoothed` **re-runs the smoothing chain over the whole
clip** (it caches curves but rebuilds the object from the solve);
`motion.target_residuals` → `resolve_targets` reloads every target track's
`.npz` and re-applies its corrections; and a target on the *smoothed* path calls
`smoothed_paths`, a Python loop over every frame of every track. All of it to
report one frame's residuals. Measured with a synthetic 4-track project and two
targets (one raw, one smoothed), per frame:

| Clip length | Per-frame geometry | Ceiling |
|---|---|---|
| 1,000 frames | 26 ms | ~38 frames/s |
| 3,000 frames | 73 ms | ~14 frames/s |
| 10,000 frames | 235 ms | ~4 frames/s |

That is before the output warp is even produced, and 10,000 frames is 83 s of
the 120 fps footage this tool is used on.
*Fix:* memoise a per-project "geometry context" (project, plan, options,
smoothed motion, **resolved target paths**, revision) keyed on file stamps, so a
frame costs a few `stat`s and O(targets). Fold the project file's stamp into the
plan key while there (R3).

**P2 · Playback redraws every path, segment by segment, every frame — high, M**
`Viewer.tsx` main draw effect (l. 480–911) repaints the frame *and* every
track's full path on each new frame, hover state change and mouse move during a
drag. In the default "colour by confidence" mode each segment is its own
`beginPath`/`stroke` with a freshly formatted colour string
(l. 609–622): N tracks × T frames strokes per redraw — 50,000 for five tracks
on a 10,000-frame clip — which dominates the frame budget during playback and
makes dragging a point lag. The canvas backing store is also reallocated on
every draw (l. 487–490), which clears and re-creates it.
*Fix:* the trails do not change from one frame to the next, so draw them once
into an off-screen layer and copy that per frame, redrawing it only when the
view, the tracks or the overlay settings change; build each track's paths once
(confidence quantised into 32 shades, one path per shade); resize the canvas
only when its size changes. (Stroking cached paths on every frame without the
layer was tried first and measured: see §6.4.)

**P3 · The timeline repaints the whole clip on every playhead move — high, S**
`Timeline.tsx` (l. 128–249) redraws the heat strip with one `fillRect` and one
`confidenceColor()` string **per frame of the clip** whenever `currentFrame`
changes, i.e. every frame of playback, and reallocates the canvas each time.
On a long clip most of those rects land on the same pixel column, and only the
last one drawn shows — so a short confidence dip can be invisible on exactly the
strip whose job is to show it. The canvas is not DPR-aware (blurry on HiDPI)
and does not resize with the window until the next redraw.
*Fix:* draw the static strip (heat, corrections, anchors, range dimming) once
into an off-screen layer, per pixel column, coloured by the **worst** frame in
the column; per frame, blit it and draw the playhead. DPR-aware, resized by a
`ResizeObserver`.

**P4 · Live tracking resends the whole partial path several times a second — medium, M**
`tracking.LivePartialTrack.publish` puts two clip-length lists on the job, and
`jobs._FrameCursor` updates the same job every 0.1 s, so `_stream_jobs`
(`routes/ws.py` l. 80–102) re-sends the complete event — both arrays — on every
150 ms poll for the rest of the job. On the client every message is a
`setJobs`, which changes the project context and re-renders every panel.
*Fix:* send the partial path only when it changes (a `track_rev` counter) and
let the client keep the last one; coalesce socket messages into one state update
per animation frame.

**P5 · One context for everything — medium, L**
`ProjectContext` publishes one value containing the playhead, the job table,
live tracking, tracks, selection and settings. Every consumer — both Tracks
panels, the Export, Solve and Smoothing panels, the curve editor — re-renders on
every frame of playback and every job message.
*Fix:* move the high-frequency state (playhead/playback, jobs/live tracking) into
their own contexts or a selector store (`useSyncExternalStore`), so a panel only
re-renders for what it reads.

**P6 · Uploading a clip holds all of it in memory and blocks the server — medium, S**
`routes/projects.py` `create_project` is an `async` handler that does
`await file.read()` (the whole clip in RAM — these are multi-GB files) and then
writes it and probes it with PyAV **on the event loop**, freezing every
WebSocket and request until it finishes.
*Fix:* stream the spooled upload to `video/source.<ext>` with
`shutil.copyfileobj` in a worker thread.

**P7 · `GET /projects/{id}` recomputes every track — low, S**
`serialize._load_track_stats` loads each point's `.npz`, re-applies its
corrections (including per-track smoothing) and parses its merge report on every
project refresh, which the client asks for after most actions.
*Fix:* cache the stats per point keyed on the project file's and the track's
stamps.

**P8 · Smaller per-frame costs — low, S**
Each Tracks-panel strip redraws its own per-column strip and reallocates its
canvas on every playhead move (the main timeline's fix, P3, applies as is);
`_stream_jobs` deep-compares every job's event dict on every poll. (The stream
also re-loaded the project for every output frame — fixed with P1.)

### 4.2 UI/UX

**U1 · Keyboard shortcuts stop working after clicking any button or checkbox — high, S**
The global handlers (`ProjectContext.tsx` l. 816 and 843, `Viewer.tsx` l. 1235,
`PatchPanel.tsx` l. 106) ignore **every** key while focus is on a `BUTTON` or
any `INPUT`. Clicking leaves focus on the control in Chromium, so after
pressing Play, a tool, a tab, a track's ◀/▶ or toggling "Overlays", the
arrows, `I`/`O`, `A`, `V/N/D/H`, `↑/↓` and `Delete` all do nothing until the
operator clicks the canvas. Only Space/Enter (which activate a button) and the
keys a slider or text field uses should be deferred.
*Fix:* one shared `keyBelongsToControl(event)` that knows which keys each kind
of control consumes, used by every handler.

**U2 · `stable serve` on Windows prints a dead URL, never opens the browser, and is open to the LAN — high, S**
`7129d05` changed the default `--host` to `0.0.0.0` (`cli/serve.py` l. 85,
`server/__main__.py`). The announced and auto-opened URL is built from the bind
address, and on Windows `http://0.0.0.0:8000/` cannot be connected to
(`WinError 10049`, verified): the printed link fails, and `_wait_until_serving`
polls it for 20 s and gives up, so the browser never opens. It also exposes an
unauthenticated API that can upload, delete and run GPU jobs to the whole
network by default, and `allow_origins=["*"]` (`app.py` l. 101–106) lets any
web page the user visits drive it cross-origin — although the UI itself is
always same-origin (static mount, or Vite's proxy).
*Fix:* loopback by default (`localhost`, which the test already expects) with
`STABLE_HOST` / `--host 0.0.0.0` to opt into the network; announce and open a
loopback URL whatever was bound, and say how to reach it from elsewhere; limit
CORS to the Vite dev server's origins.

**U3 · `Ctrl` `+` / `−` / `0` are captured by the canvas zoom — high, S**
`Viewer.tsx` l. 1232–1252 handles `+`, `-`, `0` without checking modifiers and
calls `preventDefault`, so the browser's own page zoom shortcuts zoom the
canvas instead. Bundled with U1.

**U4 · A running track cannot be cancelled — medium, M**
`todo_7`. Only queued jobs can be dropped (`jobs.py` l. 343–373). A mistaken
run on a deep tracker is minutes of GPU with no way out but restarting the
server.
*Fix:* cooperative cancellation — a cancelled flag on the job that the progress
callback checks and raises on; trackers already report progress per frame or
chunk, and nothing is written until a run completes.

**U5 · Mouse-only interaction — low, M**
The canvas, timeline, track strips and curve editor use mouse events only, so
pens and touch screens cannot place, drag or scrub. *Fix:* pointer events with
pointer capture.

**U6 · Click-to-move edits a hidden track — high, S**
`todo_7`/`todo_8` asked that a hidden track cannot be edited. Hit tests skip
hidden tracks, but `Viewer.tsx` `handleClick` (l. 1269–1298) still applies a
correction to the **selected** track at the clicked position when that track is
hidden, and the add-anchor tool anchors it — an edit to something not on the
screen, on a stray click. The hint line meanwhile invites the click.
*Fix:* refuse both for a hidden selected track and say why in the hint.
Bundled with U1.

**U7 · Errors that go nowhere — medium, S**
Several buttons await a request without handling its failure, so a refusal is an
unhandled rejection and nothing on screen changes: "Track all N", the tracker
picker and "Delete track" (`TrackTimelinePanel.tsx` l. 409–413, 461–466,
489–494, and the classic panel's equivalents), project delete
(`ProjectPicker.tsx` l. 96–100). *Fix:* route them to the panel's existing
error line.

**U8 · Two Tracks panels — medium, M**
`flags.ts` keeps the classic card and the timeline redesign side by side
"while the redesign is being judged"; every feature since has been written
twice or shared through helper components. The timeline variant is the default.
*Decide, then delete the other variant and the flag.*

**U9 · The frame label hides a lagging picture — low, S**
Round 5 promised `frame 259 / 500 · showing 244` while a frame is in flight; the
label shows only the playhead (`Viewer.tsx` l. 1978). *Fix:* show `viewFrame`
when it differs.

**U10 · Accessibility basics — low, S**
Tool buttons are glyphs with a `title` only and no `aria-label`; the tool group
is `role="radiogroup"` around `aria-pressed` buttons; project rows are clickable
`<li>`s that cannot be focused; there is no `:focus-visible` style; the
timeline is not a slider to assistive technology. (Cosmetic, same area: there
is no favicon, so every page load logs a 404.)

**U11 · `todo_7` leftovers — medium, M–L**
A pixel search radius for LK instead of pyramid levels; one ◀▶ button from an
anchor (`direction: "bidir"` is already supported); "re-track every track";
a window size interpolated between anchors for subjects that change size.

### 4.3 Reliability and security

**R1 · Saving a project fails on Windows while anything is reading it — high, S**
`Project.save` writes a temp file and `os.replace`s it over
`project.stable.json` (`core/project.py` l. 482–486). On Windows a rename onto
a file that another handle has open fails with `WinError 5`, and the server
reads that file constantly — every HTTP handler, every job, and the frame
stream per frame in the export tab. The baseline run hit it in
`test_frame_stream.py`. In use it surfaces as an edit failing with "Access is
denied", or a **multi-minute tracking job failing at its final save** after the
track has been written. Reads race the other way: an `open` during the replace
can fail, which `load_project` turns into a 404.
*Fix:* retry both briefly (bounded backoff, Windows only) — the standard remedy
for this sharing violation.

**R2 · Side-car JSON is written in place — medium, S**
`history.json`, `render_queue.json`, `clip_range.json`, tracker options, merge
reports and the motion caches use `write_text` (truncate, then write), so a
concurrent reader can see half a file and fail to parse it. *Fix:* one
`atomic_write_text` helper (temp file + the R1 retrying replace) for all of
them.

**R3 · The preview plan can outlive a change to the tracks it holds — medium, S**
The plan cache key (`frames._plan_key`) covers the caches, options, markers,
border and clip size, but not the project file, while the plan's targets are
resolved from the *current* tracks. The interactive edit routes (corrections,
undo/redo, adding or deleting an anchor) clear the plan cache themselves, but
the other writers do not: a finished track or retrack job, a `PATCH`ed point
(taking it out of the solve changes an "all tracks" target), a derived or
deleted point. After one of those the preview keeps the old fit until something
else changes — and an export, which builds a fresh plan, does not match it.
*Fix:* include the project file's stamp in the key, so every writer counts
(done with P1). *Corrected during execution:* the first draft of this finding
said every edit was affected; the edit routes' explicit cache clears were
found when the regression test passed without the fix, and the test now uses
an uncovered writer.

**R4 · Exposure** — see U2: all interfaces by default, open CORS, no
authentication. WebSocket origins are not checked either (low).

### 4.4 Code health and process

* **C1** Stale `stable/server/webdist` (see §1) — rebuild after any frontend
  change; consider a CI check that the committed build matches `web/src`.
* **C2** The two solver-tolerance failures have been "the number to come back
  to" for four rounds. Either loosen the tolerance with a reason or find the
  drift.
* **C3** Both deep checkpoints are still unpinned (round 1 #22).
* **C4** Plan files were scattered across the root and `done/` — consolidated
  here.
* **C5** `fable_rework` — merged as superseded (§2); delete once this branch
  lands.

---

## 5. Prioritised fix list

Ranked by user impact against effort and risk. The first five are executed in
this pass (§6); the rest are proposed.

| # | ID | Fix | Area | Effort | Status |
|---|---|---|---|---|---|
| 1 | R1 | Retry Windows sharing violations when saving/loading a project | Reliability | S | Executed |
| 2 | U2, R4 | `stable serve`: loopback default, a URL that works on Windows, CORS limited to the dev server | UX / security | S | Executed |
| 3 | U1, U3, U6 | Input handling: shortcuts after clicking a control, browser zoom keys, no click-edits on hidden tracks | UX | S | Executed |
| 4 | P2, P3 | Playback rendering: trails drawn once into a layer, cached timeline strip | Performance | M | Executed |
| 5 | P1, R3 | Export tab: memoised per-frame geometry context | Performance | M | Executed |
| 6 | P4 | Live-tracking events: send the partial path only when it changes; batch on the client | Performance | M | Proposed |
| 7 | P6 | Stream uploads to disk off the event loop | Performance | S | Proposed |
| 8 | U4 | Cancel a running track | UX | M | Proposed |
| 9 | R2 | Atomic writes for every side-car | Reliability | S | Proposed |
| 10 | U7 | Surface the errors that are currently dropped | UX | S | Proposed |
| 11 | P5 | Split the project context | Performance | L | Proposed |
| 12 | U8 | Pick one Tracks panel, delete the flag | UX / health | M | Proposed |
| 13 | U11 | `todo_7` leftovers | UX | M–L | Proposed |
| 14 | P7, P8 | Smaller per-request and per-frame costs | Performance | S | Proposed |
| 15 | U5, U9, U10 | Pointer events, the "showing N" label, accessibility basics | UX | S–M | Proposed |
| 16 | C2, C3 | Solver tolerance; pin the checkpoints | Health | S | Proposed |

---

## 6. Executed in this pass

Branch `review/perf-ux-pass`, on top of `main` and the `fable_rework` merge
(`25f4cf5`). One commit per fix, each with its tests (R1 has a small
follow-up), then one rebuilding the packaged frontend:

| Commit | Fix |
|---|---|
| `8dc36b8`, `51fe202` | R1 — retry Windows sharing violations on the project file (and not a directory, which can never clear) |
| `ab03d8e` | U2, R4 — `stable serve` on loopback, a reachable URL, CORS for the dev server only |
| `675ea19` | U1, U3, U6 — input handling |
| `cb03521` | P2, P3 — playback rendering |
| `19bf7d5` | P1, R3 — export-tab geometry context |
| `c2dd703` | Packaged `stable/server/webdist` rebuilt (it was already stale, C1) |

The frontend fixes were verified in a real browser: the app served against a
throwaway workspace (a 3,000-frame, six-track, solved and smoothed synthetic
project), driven by Playwright in system Chrome on this machine's GPU, with the
same script run against a build of `main` and a build of this branch.

### 6.1 R1 — Windows sharing violations on the project file

`Project.save` retries `os.replace`, and `Project.load` retries the read, on
`PermissionError` with a bounded backoff (about 0.6 s in all), on Windows only;
elsewhere the error is never transient and is raised at once, as before. A
directory where the file should be is also reported as `PermissionError` on
Windows; it can never clear, so it is not retried (found while checking the
suite's timings: it cost 0.6 s per failure).

*Verified:* a test that saves while a real reader holds the file open passes,
and fails with `WinError 5` when the retry is switched off. Five more tests pin
the retry count, the give-up, no retry off Windows, the read path, and no
retry for a directory. The `test_frame_stream.py` setup error from the baseline
is gone.

### 6.2 U2 + R4 — `stable serve`

Default host `localhost` again, with `STABLE_HOST` to change the default and
`--host 0.0.0.0` to opt into the network (the Dockerfile already passes it).
The announced, polled and opened URL is always one this machine can reach
(`browse_url`): a wildcard bind is shown as `http://localhost:<port>/`, plus a
line naming how to reach it from elsewhere and that Stable has no login.
`python -m stable.server` defaults to loopback too. CORS allows only the Vite
dev server's origins; the UI itself is same-origin, so nothing it does changed.
README documents the network option.

*Verified:* the baseline's `test_serve_subcommand_is_registered` failure now
passes; new tests cover `STABLE_HOST`, the URL for wildcard and IPv6 binds,
that a wildcard bind polls and opens a loopback URL and prints the warning, and
that a foreign origin gets no CORS headers (and a failed preflight for `DELETE`)
while the dev server's origin does.

### 6.3 U1 + U3 + U6 — input handling

`web/src/keyboard.ts` decides which keys the focused control really uses —
every key for text fields, text areas, selects and editable content; the
movement keys for sliders and radios; Space/Enter for buttons, checkboxes and
other activatable controls — and every global handler (frame navigation and
tools, undo/redo, canvas zoom, the patch panel's Delete) defers only those.
Canvas zoom ignores `Ctrl`/`⌘`/`Alt` chords, so the browser's page zoom works
again. Click-to-move and the add-anchor tool refuse a hidden selected track
with a hint, and the cursor and hint line stop inviting the click.

*Verified:* the key-ownership rules against 18 cases in a scratch harness, and
in the browser:

| Check | `main` | This branch |
|---|---|---|
| Click Play, click Pause, press → three times | frame does not move | frame +3 |
| Hide the selected track, click the frame | `POST …/p1/corrections` sent | nothing sent; hint explains |

### 6.4 P2 + P3 — playback rendering

* **Trails as a layer.** Every track's trails (the path, the smoothed path, a
  running job's partial path) are drawn into an off-screen layer that is
  redrawn only when the view, the tracks, their smoothing, the selection, the
  colouring, the overlay switches or a live partial path change; a frame of
  playback copies the layer and draws only what is about that frame (markers,
  rings, anchors, windows, the export overlay). The paths themselves are built
  once per loaded track as `Path2D`s in source pixels (`web/src/trackPaths.ts`),
  confidence quantised to 32 shades — indistinguishable on the ramp.
* **Timeline.** The strip is drawn off-screen once per change of what it
  shows, per pixel column, each column taking its *worst* frame (so short dips
  stay visible on long clips — the strip's whole job); DPR-aware; resized by a
  `ResizeObserver`; per frame it is one copy and the playhead.
* Canvases are reallocated only when their size changes (`web/src/canvas.ts`).

*Measured* over 6 s of playback, two runs each (the clip is 30 fps):

| Playback | Video frames painted | Main thread busy | JS time |
|---|---|---|---|
| 1×, `main` | 177–179 | 36–37% | 1.8–1.9 s |
| 1×, this branch | 176 | 12–13% | 0.4–0.5 s |
| 4×, `main` | 279–328 | 56–66% | 2.8–3.3 s |
| 4×, this branch | 450–451 (the display's cap) | 25% | 0.9 s |

*Found on the way, and why the trails are a layer.* The first version stroked
the cached per-shade paths on every frame instead. In isolation that halved the
cost of a redraw (9.1 ms against 18.8 ms with a GPU sync, six 3,000-frame
tracks), and it cut the main thread to 11–17% — but on this machine's
integrated GPU a few long multi-contour strokes were slower to *rasterise* than
thousands of one-segment ones, and at 4× it painted fewer frames than `main`
(244–250 against 279–328). Measuring the Viewer and Timeline halves separately
isolated it to the Viewer. Drawing pixels that do not change once, rather than
cheaply every frame, fixed both sides.

### 6.5 P1 + R3 — export-tab geometry

`frames.geometry_context()` builds the project, plan, options, smoothed
motion, every target's resolved path and the revision once, and hands back the
same object until a file they came from changes (keyed on `stat` stamps, so an
unchanged call reads nothing). `frame_geometry` and `target_residuals` take the
resolved paths; the frame stream uses the context for geometry and for output
frames (no project re-parse per output frame). The plan key now includes the
project file's stamp (R3), and `recompute_smoothed` clears the contexts along
with the plan cache.

*Measured* with the same synthetic 4-track, two-target project as §4.1:

| Clip length | Per-frame geometry before | After (warm) | First frame after a change |
|---|---|---|---|
| 3,000 frames | 73 ms | 0.1 ms | 0.4 s (as before: the plan build) |
| 10,000 frames | 235 ms | 0.1 ms | 1.2 s (as before) |

*Verified:* the context is reused while nothing changes and rebuilt after an
options change; cached geometry equals geometry computed from scratch for a raw
and a smoothed target; and taking a held point out of the solve moves the
preview (residual stays ~0) — a test that fails when the project file is left
out of the key.

---

## 7. Carried forward

Everything in §5 not marked executed, plus these older items still open:

* Round 1 #22 — pin the CoTracker3 / TAPIR checkpoints to verified SHA-256s.
* Round 2 — run SAM 3 against real `facebook/sam3` weights (needs the licence
  accepted by the operator).
* Rounds 7–10 — the two `test_phase3_acceptance.py` rotation-tolerance
  failures.
* Round 8 (`todo_7`) — see §3.8 and U4, U11.

---

## 8. Ocean and SiamDW from TracKit, 2026-10-20

Request: implement the two trackers in `github.com/researchmm/TracKit` — Ocean
(ECCV 2020) and SiamDW (CVPR 2019) — alongside the others. Both are now plugins
(`ocean`, `siamdw`), available wherever PyTorch and OpenCV are, with pinned MIT
checkpoints and no extra package to install.

**What was built.**

* `stable/trackers/siamese_nets.py` — both networks ported to plain PyTorch
  from the authors' code (MIT notice kept), stateless and batched, importable
  without torch. Verified against the authors' SOTS module definitions on
  identical inputs: maximum deviation ~1e-5 at search sizes 255/271/287.
* `stable/trackers/siamese.py` — the shared point-tracking pipeline: a template
  fixed at the anchor, sub-pixel crops, `search_radius` honoured exactly (the
  search grows past the trained 255 px and candidates beyond it are masked),
  batching, holding through a loss, and the CPU fallback (which has to rebuild
  the templates on the new device).
* `stable/trackers/ocean.py`, `stable/trackers/siamdw.py` — each network's
  decoding (TracKit's anchor-free box penalties / SiamFC's three-scale bicubic
  peak) and its checkpoint.
* `stable/trackers/deep.py` — the device/weights/fallback half of
  `DeepTrackerPlugin` became its base class, `ModelTrackerPlugin`, so a
  frame-at-a-time model reuses it without inheriting windowing. Additive:
  `DeepTrackerPlugin`'s constructor and behaviour are unchanged, and every
  existing deep-tracker test passed untouched before anything was built on it.

**What measuring changed.** Each of these was found by running the real
checkpoints, not by reading the papers:

| Finding | Response |
|---|---|
| Handed a patch, Ocean regresses the box of the *object* it thinks the patch belongs to — 3.7 px median off the patch | The point rides on the network's reading of the anchor frame itself (0.9 px); both plugins do this |
| Ocean's per-axis scale random-walked to its clamps (90th-pct error 19 px on handheld footage) | Isotropic scale, a gate on readings whose box changes shape, `size_lr` 0.3 (90th pct under 10 px) |
| Ocean's classifier scored 0.86 with the target covered | Confidence is classifier x feature cosine at the found position (AUC 0.98 against >3 px errors) |
| SiamDW's raw response depends on the template (peaks 7-15) | Confidence is the peak over the anchor frame's own peak |
| Holding on every weak frame lost Ocean's points in a fast pan | `lost_below`, separate from the confidence floor (defaults to it) |
| The SiamDW release has 715,493 denormal weights: 3.5 s a CPU frame | Denormals flushed around CPU calls: 46 ms |

**Measured at the shipped defaults** (eight 40-60 frame clips of known motion,
eight points each, plus six occluder sweeps; `lk`/`template` for contrast):

| Tracker | Median / 90th pct error | Occluder sweeps recovered | Max confidence while covered |
|---|---|---|---|
| `siamdw` | 0.65 / 1.8 px | 6 of 6 | 0.00 |
| `ocean` | 0.74 / 4.9 px | 5 of 6 | 0.02 |
| `lk` | 0.19 / 1.4 px | 0 of 6 | 1.00 |
| `template` | 0.70 / 2.6 px | 1 of 6 | 0.92 |

On the bundled sample clip through `stable track` and through a two-anchor
server job, both tracked to 0.31-0.66 px median. Ocean is accurate on
object-like features and weaker on smooth natural-image patches (90th pct
12 px there, SiamDW 2.5 px); both module docstrings say so.

**Checks.** `pytest -m "not integration"`: the new modules (contract suite for
both plugins, `test_siamese_trackers.py`, `test_siamese_nets.py`) all pass,
also with PyTorch hidden as on CI (the networks module then skips); the only
failures are the two §7 rotation-tolerance tests. `ruff check`, `ruff format
--check` on touched files and `mypy stable` are clean. No frontend change: the
picker, the pattern/search sliders and the model manager are all driven by
what the plugins declare.

**Open.**

* `Ocean.pth` is hosted by the authors' SOTS repository, which has no licence
  file; it is labelled MIT because TracKit's MIT README designates SOTS as its
  model source. `SiamDW.pth` is verified identical to the MIT SiamDW release.
* Both checkpoints are Google Drive downloads (`drive.usercontent.google.com`).
  TracKit's own Drive links have already died once; a pinned digest makes a
  changed or quota-blocked file fail closed, but a mirror would be sturdier.
* Ocean's object-aware (deformable-convolution) refinement and its online
  classifier, and OceanPlus, are not ported. The only Ocean checkpoint still
  published is the offline model, which has neither branch; both need TracKit's
  compiled CUDA extensions; and OceanPlus is a segmentation tracker.
