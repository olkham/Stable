# Implementation plan — tenth feedback round

Source: `todo_9` (two features, two improvements, two more features — six items).
Same conventions as the earlier rounds (`done/TODO_PLAN.md` … `TODO_8_PLAN.md`):
every item states what was **found in the code**, not what was assumed, and the
suite stays green throughout (`pytest -m "not integration"`, `ruff check`,
`ruff format --check`, `mypy stable`, `npm run build`, then
`python scripts/build_frontend.py`).

Baseline, measured rather than assumed: the full non-integration suite was run
before any edit and reported **2 failures**, both in
`tests/test_phase3_acceptance.py`
(`test_the_solve_recovers_the_ground_truth_motion` and
`test_the_recovered_curves_match_the_known_channels`) — the same numerical
solver tolerance the seventh, eighth and ninth rounds recorded (a solved
rotation of `2.133e-2` against `2.154e-2 ± 2e-4`). Nothing in this round goes
near the solver, so that is the number to come back to, not a target to improve.

After: **2295 collected, the same 2 failed**, unchanged, and 43 new tests added
across `tests/test_corrections_smooth.py` (12), `tests/test_derive.py` (12),
`tests/server/test_render_queue.py` (13), `tests/server/test_edits.py` (+6) and
`tests/server/test_points.py` (+6). `ruff check` passes over `stable` and
`tests`, `ruff format --check` passes on every file touched, `mypy stable` is
clean over 90 files, `npm run build` (`tsc --noEmit` + Vite) is clean, and
`python scripts/build_frontend.py` has refreshed the packaged
`stable/server/webdist` so `stable serve` serves this build.

Scope: **these six items only**. Nothing else in `todo_9`, nothing adjacent that
looks tempting on the way past.

---

## The items, and what each one actually is in the code

| # | `todo_9` line | Verdict |
|---|---|---|
| 1 | smooth individual tracks; the smoothing menu doesn't say what it applies to | Two things: a new `smooth` correction kind, and the panel saying what it is |
| 2 | derive a track from other tracks (the mean of their paths) | New pure averaging module + one route + one shared panel |
| 3 | setting In/Out should auto-enable "play range" | One transition, in the context that owns both |
| 4 | toggle to hide every overlay on the **output preview** canvas | New view preference; the left canvas already has one, the right does not |
| 5 | default the playback rate to ≈30 fps for the clip's frame rate | Derive the initial rate from `video.fps` |
| 6 | queue renders instead of starting them; process on demand | New queue sidecar + ordered runner + estimator + panel |

---

## 1. Smoothing an individual track

### What is there now

`web/src/components/SmoothingPanel.tsx` draws one row per `Channel`
(`"x" | "y" | "r" | "s"`, `web/src/types.ts`). Those four are the **camera**
curves the solver recovers from *every solve-enabled track at once*
(`stable/solver/motion.py`, reached through `stable/server/motion.py`
`solve_tracks`), not one track's path. Nothing on the panel says so, which is
exactly the report: "it's not clear which track path this applies to". It
applies to none of them, and to all of them.

There is no way to smooth one track's path at all. The edit vocabulary a track
does have is the correction layer (`stable/core/corrections.py`): five typed,
non-destructive, individually toggleable edits applied in order by
`apply_corrections`.

### What changes

**(a) A sixth correction kind, `smooth`.** `SmoothCorrection` carries a filter
name, its parameters, an inclusive `[start, end]` frame range and a `blend`
width. Applying it runs `stable.smoothing.filters.apply_filter` over the track's
`x` and `y` separately and cross-fades the result into the range with
`stable.smoothing.edit.blend_subrange` — the same two functions the channel
smoothing uses, so a track path and a camera curve are smoothed by one
implementation rather than two.

Three details that are decisions, not accidents:

* **Gaps stay gaps.** `apply_filter` interpolates non-finite samples away before
  filtering (it has to — one `NaN` poisons a global filter). The filtered values
  are therefore written back **only where the track already had a position**, so
  smoothing never invents a sample in a gap. Filling a gap is `gap_fill`, and it
  should stay a separate, visible edit.
* **Confidence is untouched.** Smoothing changes where the tracker says the
  feature is, not how sure it was.
* **`start`/`end` are required.** `GapFillCorrection` and
  `DeleteRangeCorrection` state theirs the same way, `_label` in
  `routes/edits.py` already renders `kind point [start-end]` from them, and
  "the whole clip" is honestly expressible as `0 … T-1`. An optional range would
  have added a second spelling of the same thing.

`apply_corrections` gains an optional `fps`, because a Butterworth cutoff is in
Hz and the clip's frame rate is the only thing that turns that into frames.
`load_display_track` passes `project.video.fps`; every existing caller keeps
working, and the default is documented rather than silent.

The import of `stable.smoothing.filters` from `stable.core.corrections` is
**deferred into the function**. `stable/smoothing/__init__.py` imports
`channels`, which imports `stable.core.project`, which imports this module — a
module-level import here would be a genuine import cycle, not a style
preference. The comment in the code says that.

Nothing else on the server needs a route: `POST .../points/{id}/corrections`
takes a correction's own project-file form, and `correction_from_dict` is the
one validator (`stable/server/schemas.py` `CorrectionIn` is deliberately open).
`GROUND_TRUTH_KINDS` and `ANCHORING_KINDS` are *not* extended — a smooth states
no absolute position, so it is not an anchor.

**(b) The panel says what it smooths.** `SmoothingPanel` gets a header note
naming the four channels as the camera motion solved from the N solve-enabled
tracks, and pointing at the per-track control for the other thing. The per-track
control is a new block on `TrackEdits.tsx` (already the home of every per-point
edit, already carrying the `Frames [start]–[end]` row with `in`/`out` buttons):
a filter picker built from `GET /api/filters` — the same catalogue-driven
controls the smoothing panel uses, so adding a filter to the registry makes it
appear in both — and a "Smooth path" button.

### Files

* `stable/core/corrections.py` — `SmoothCorrection`, `_apply_smooth`, the
  registry entry, `fps` on `apply_corrections`.
* `stable/server/tracking.py` — `load_display_track` passes the clip's fps.
* `web/src/types.ts` — `"smooth"` on `CorrectionKind`, the new fields.
* `web/src/components/TrackEdits.tsx` — the control and its `describe()` case.
* `web/src/components/SmoothingPanel.tsx` — the note.
* `tests/test_corrections_smooth.py` (new), plus a server round-trip in
  `tests/server/test_edits.py`.

---

## 2. A track derived from other tracks

### What is there now

Nothing. `PointSpec` (`stable/core/project.py`) is anchors + tracker + a `raw`
`.npz` path + corrections; a track is produced by `run_point_tracking`, which is
the only writer of `tracks/{id}.npz`.

### What changes

> **Corrected after testing on real footage.** The first implementation averaged
> at matching *normalised progress* — `p3[0.25] = mean(p1[0.25], p2[0.25])` — as
> the `todo_9` text proposes, to sidestep contributors covering different frames.
> That is wrong, and visibly so: it time-warps any source whose covered span is
> not the longest. A track over frames 0–180 stretched across 0–200 replays its
> motion at nine tenths speed, so the derived path *leads it early and lags it
> late*, wandering off the line between the two features it should sit between —
> exactly what was reported back. Two features in one shot share a clock;
> averaging them has to respect it, and a frame is the clock. The section below
> describes what is built now.

A pure module `stable/core/derive.py`, with no knowledge of projects, files or
HTTP:

* `mean_track(tracks, mode)` — the per-frame mean. On frame `f` the answer is
  the centroid of wherever the sources are on frame `f`; for two tracks, the
  midpoint of the straight line joining them. No source is ever read at a frame
  other than its own.
* `covered_span(track)` — the inclusive first/last frames a track has a position
  on, which is how "where is this path" is asked of an array that always spans
  the whole clip.

A differing length is therefore not a resampling but a question about **which
frames the answer covers** (`SPAN_MODES`):

* `overlap` (the default) — only the frames *every* source has a position on, so
  every sample really is the mean of all of them. A path that is sometimes the
  midpoint of two features and sometimes just one of them is not one thing.
* `union` — every frame *any* source covers, averaging whichever are present,
  with a step wherever a source starts or stops contributing.

The route `POST /api/projects/{id}/points/derive` takes
`{"point_ids": [...], "mode": "overlap" | "union"}`, writes the clip-length
result to `tracks/{new_id}.npz` and appends a `PointSpec` anchored at the
derived path's first frame. Sources that share no frames at all are refused
under `overlap` with a message naming the mode that *can* combine them, rather
than a bare failure.

One thing still decided here rather than by the request: **the new point's
tracker**. Meaningless for a path nobody tracked, but `PointSpec` requires one
and a later re-track has to do something; it takes the first source's.

### A composite is not a tracked track

Second correction from use: the card for a derived track offered a tracker
dropdown, pattern/search sizes, an anchor list and Track/Re-track buttons —
every one of them a control for something that did not happen to it. Nothing
chose an algorithm, no window was matched, no run produced the path, and the
single anchor it carries is there because `PointSpec` needs one.

So `derived_from` is now **stored on the point** (`PointSpec.derived_from`, an
optional key like `color`, absent meaning "an ordinary tracked point" — no
migration). It began as a key on the create response alone, which was enough to
say "made from p1 + p2" once and lost it on the next reload; the panel needs it
every time it draws the card.

With it, both Tracks panels drop the tracker picker (showing `composite`), the
`PatchPanel`, the anchor row and every run button for one of these, and list the
tracks it was made from instead. Its **edits stay** — an averaged path can be
nudged, gap-filled or smoothed like any other. Three consequences worth stating:

* `run_point_tracking` clears `derived_from`. The UI hides the button that gets
  there, but the route is open, and a point still claiming to be an average
  while holding a freshly tracked path would be a lie the panel would draw.
* A deleted source is **kept** in the list, drawn as `p2 (deleted)`. "Made from
  p1 and p2" is what happened, and the path baked from them did not change when
  one went away.
* `AllPatchesStrip` skips composites, for the reason it already skips whole-frame
  trackers: the pixels under the midpoint of two features are a picture of
  neither.

No undo entry: `stable/server/history.py` snapshots the *corrections* layer, so
adding a point is not something it can restore, and `POST /points` (creating a
point by clicking) is not recorded either. Deleting the derived track is the
`×` that deletes any other, and `delete_point` already sweeps every file named
for it.

The UI is one shared component, `web/src/components/DeriveTrack.tsx`, used by
**both** Tracks panels — `TracksPanel.tsx` and `TrackTimelinePanel.tsx` are two
implementations behind a flag (`web/src/flags.ts`), and a feature that existed
in only one would make the flag impossible to judge. That is the same reason
`TrackVisibility.tsx` is shared.

### Files

* `stable/core/derive.py` (new), `stable/server/routes/points.py` (the route),
  `stable/server/schemas.py` (`DeriveTrackRequest`).
* `web/src/api/client.ts`, `web/src/components/DeriveTrack.tsx` (new), and one
  line in each Tracks panel.
* `tests/test_derive.py` and `tests/server/test_derive_points.py` (both new).
  `test_the_mean_never_leads_or_lags_a_source` and
  `test_the_derived_track_is_not_time_warped_by_the_shorter_source` are the
  reported bug, pinned at both levels: two features moving *identically* a fixed
  distance apart, one tracked over half as many frames. Their midpoint is that
  same motion on every shared frame, so any resampling of either source changes
  the derived path's speed and the test fails.

---

## 3. Setting In/Out enables "play range"

`playRangeOnly` is state on `ProjectContext` (`web/src/state/ProjectContext.tsx`),
defaulting to `false`, and `Timeline.tsx` reads it. `setRangeMarker` writes the
markers and nothing else, so the operator sets In and Out and then has to find a
checkbox to see the effect.

`setRangeMarker` turns it on **on the transition from "no range" to "a range"**
— when the response's `whole_clip` goes false having been true. Only that
transition: switching it on again every time a marker moves would override a
deliberate un-tick while a range is in force, which is worse than not helping at
all. `clearRange` turns it off, because there is then no range to play and the
checkbox is disabled anyway.

One file: `web/src/state/ProjectContext.tsx`.

---

## 4. A clean output preview

`Viewer.tsx` draws two canvases. The left one already has the master `Overlays`
switch in the toolbar (`overlaysEnabled`, `ProjectContext`), which is exactly
"let me see the frame". The right one — the side-by-side **output** canvas,
drawn by the effect at `Viewer.tsx:929` — has no such switch: it always draws the
thirds guides, the snap guide and every target crosshair, plus a DOM label and a
hint over it. There is no way to see the delivered frame on its own.

`cleanOutput` joins `showGuides` / `sideBySide` / `snap` on `ExportContext` —
view preferences, kept out of the render options the server stores because they
change nothing about the render. While it is on the output effect draws the
picture and stops, the label and the hint are not rendered, and
`grabOutputTarget` refuses: dragging a crosshair you cannot see is not a
feature. The checkbox sits with the other two in `ExportPanel`.

Files: `web/src/state/ExportContext.tsx`, `web/src/components/Viewer.tsx`,
`web/src/components/ExportPanel.tsx`.

---

## 5. A default playback rate near 30 fps

`Timeline.tsx` offers `PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4]` and
`ProjectContext` starts `playbackRate` at `1`, so a 120 fps clip plays at 120
fps — four times faster than the shot was, which is not a review speed.

The list and the choice move to `web/src/playback.ts`:
`defaultPlaybackRate(fps)` returns the offered rate whose `fps × rate` is
closest to 30 (ties going to the slower one, and `1` for a clip whose frame rate
is unknown). 120 → 0.25, 60 → 0.5, 48 → 0.5, 30/25/24 → 1. The context applies it
when the clip's frame rate becomes known and when it changes — which is when the
project changes, and is exactly when re-deciding is right. The operator's own
choice is never overwritten afterwards, because nothing else changes those
dependencies.

The timeline's fps readout already prints `fps × rate`; its tooltip gains the
reason.

Files: `web/src/playback.ts` (new), `web/src/state/ProjectContext.tsx`,
`web/src/components/Timeline.tsx`.

---

## 6. A render queue that does not start on its own

### What is there now

`POST /api/projects/{id}/export/video` (`stable/server/routes/export.py`) plans
the render, refuses a no-op, and **submits the job immediately** with
`exclusive=True`. The panel's button already says "Add to export queue", and the
"Queue" section under it is `Object.values(jobs).filter(kind === "export")` —
i.e. a list of jobs that are already running or already done. There is nothing
to reorder, nothing to estimate, and the only removal is `cancelJob`, which the
manager refuses the moment a job has started (`JobManager.cancel`).

Ordering is not guaranteed either: `_enter_exclusive` waits on a
`threading.Condition` and `notify_all` wakes every waiter, so several exports
submitted together do not necessarily encode in the order they were submitted.
Reordering a queue the server will not honour would be a lie.

### What changes

A new module `stable/server/render_queue.py` owning three things:

**The queue itself** — `render_queue.json`, a sidecar in the project directory
beside `clip_range.json` / `render_options.json`, so the locked v1 project schema
stays untouched. One entry per queued render, holding *everything the render
depends on*, captured at the moment it was queued: filename, codec, CRF, border
policy, the whole `RenderOptions` dict, the smoothing filter list, and the
resolved frame range. That capture is the reason a queue is worth having — line
up "this crop" then "that crop" and each renders what it was queued with, which
is what the current route already promises for a single submit and what a
sidecar has to promise across a restart. The plan is *rebuilt* from the snapshot
when the entry runs, since a `StabilizationPlan` is an in-memory object over the
whole clip and cannot be stored.

An entry is validated when it is queued, not when it runs: the plan is built and
`is_no_op` checked there and then, so a render that would do nothing is refused
while the operator is still looking at the settings that caused it.

**An ordered runner** — one daemon thread per project, submitting the *next*
pending entry only once the previous one has reached a terminal state. Not
"submit them all and let `exclusive` sort it out": that neither preserves the
operator's order (see above) nor leaves the worker pool alone — `JobManager`
runs 2–4 workers, and six exports waiting for an exclusive slot would occupy
every one of them while a tracking job sat behind them. `exclusive=True` stays
on each job as the guard against a manual export submitted alongside.

**An estimate** — `render_stats.json`, a short rolling list of
seconds-per-megapixel-frame measured from renders that have actually finished on
this machine. An entry's estimate is its frame count × its output size × that
figure. Before anything has been measured a documented seed constant is used and
the answer is flagged `measured: false`, which the panel shows as "rough": an
estimate that says where it came from is useful, and an unlabelled guess is not.

Routes, all under the existing `/projects/{id}/export` prefix:

| Route | Does |
|---|---|
| `POST /queue` | capture the current settings as an entry (does **not** render) |
| `GET /queue` | the entries, each with its estimate |
| `PUT /queue/order` | reorder the pending entries |
| `DELETE /queue/{entry_id}` | drop one |
| `POST /queue/run` | start the runner |

`POST /export/video` is left exactly as it is — an immediate render is still the
right answer for a script or a one-off, and it is what the existing tests cover.
The panel stops using it.

The panel's Queue section becomes the real queue: filename, frames, output size,
codec, estimate and status per row, `▲`/`▼` to reorder and `×` to drop while an
entry is pending, a total estimate, and a **Process render queue (N)** button.
Rows that are running show the live job progress, matched by the `job_id` the
entry records — so it stays one list rather than two.

### Files

* `stable/server/render_queue.py` (new), `stable/server/routes/export.py`,
  `stable/server/schemas.py`.
* `web/src/types.ts`, `web/src/api/client.ts`,
  `web/src/state/ExportContext.tsx`, `web/src/components/ExportPanel.tsx`,
  `web/src/styles.css`.
* `tests/server/test_render_queue.py` (new).

---

## Order of work

1. Item 3 and item 5 (contained, no server change) — they make the rest easier
   to exercise by hand.
2. Item 4.
3. Item 1 (core → server → web).
4. Item 2 (core → server → web).
5. Item 6 (the largest).
6. Full suite, `ruff`, `mypy`, `npm run build`, `scripts/build_frontend.py`.

---

## What was built, where it differed from the plan

Everything above landed. Four things are worth recording because they were
decided while writing rather than before:

**One filter catalogue, one fetch** (`web/src/hooks/useFilterCatalogue.ts`). Both
smoothing controls build themselves from `GET /api/filters`, and the per-track
one renders once per open track card — so the request is memoised at module
scope instead of fired per card. The Smoothing panel's own copy of that fetch
was replaced by the hook, which is why the two can no longer disagree about what
filters exist.

**A `clear finished` button on the queue** rather than a bulk route. Finished
entries stop being work but stay visible so a run's outcome can be read; the
button dismisses them with one `DELETE` per entry through the endpoint that
already exists, so "what may be removed" is decided in exactly one place.

**`unique_render_name` grew a `taken` argument.** A queued render has claimed a
filename but not written it, so the on-disk existence check cannot see it —
three unnamed entries queued in the same second would all have been handed the
same timestamped name. `tests/server/test_render_queue.py::test_queued_renders_never_share_a_filename`
is that.

**The queue captures the smoothing filter list too**, not only the render
options. "Queue up many different runs" is at least as often two smoothings as
two crops, and without it two entries that differ only in cutoff would render
identically. It is a `dataclasses.replace` of the project at plan time — a copy,
so one entry's settings cannot leak into anything else reading that project.
