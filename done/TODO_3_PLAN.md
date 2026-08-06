# Implementation plan — fourth feedback round

Source: `todo_3` (9 items). Same conventions as `done/TODO_PLAN.md`,
`done/NEW_TODO_PLAN.md` and `done/TODO_2_PLAN.md`: every work item states what
was **found in the code**, not what was assumed, and the suite stays green
throughout (`pytest -m "not integration"`, `ruff check`, `ruff format --check`,
`mypy stable`, `npm run build`, then `python scripts/build_frontend.py`).

Baseline: the two pre-existing `tests/test_phase3_acceptance.py`
rotation-tolerance failures noted in the previous two rounds are still there and
are still nobody's here.

---

## Order of work

The nine items are not independent, and three of them rewrite the same file
(`Viewer.tsx`), so they are sequenced rather than fanned out:

| # | Item | Owns | Depends on |
|---|------|------|-----------|
| T2 | Curves update live with smoothing | `ProjectContext`, `SmoothingPanel`, `CurveEditor` | — |
| T3 | Smoothed track path on the canvas | `routes/solve.py`, `ProjectContext`, `Viewer` | T2 |
| T1 | Export becomes a sidebar tab | `App`, new `ExportContext`/`ExportPanel`, `Viewer` | — |
| T7 | [BUG] the crop unsets the placed targets | `ExportContext`, `routes/solve.py`, `routes/export.py` | T1 |
| T4 | Cursors + an interaction-mode toolbar | `ProjectContext`, `Viewer`, `TracksPanel` | T1, T3 |
| T5 | Track cards collapse by hand | `TracksPanel` | — |
| T6 | Solve checkbox leaves the Tracks panel | `TracksPanel`, `SolvePanel` | — |
| T9 | Delete a keyframe from its image | `routes/points.py`, `PatchPanel` | — |
| T8 | Exports queue | `jobs.py`, `routes/export.py`, `ExportPanel` | T1 |

T2 first because T3 needs the shared smoothing state it creates. T1 before T4
because both rewrite the viewer's draw and interaction paths, and doing them in
one pass would make neither reviewable.

---

## T2 — the curves view updates as the smoothing changes

> "When we update smoothing settings, the curves view doesn't get updates until
> the user closes and reopens the curves view."

**Found:** `SmoothingPanel.tsx` and `CurveEditor.tsx` each hold their *own* copy
of the same server state. Both call `api.getSmoothing(project.id)` in a `load`
of their own, both keep `raw` / `smoothed` / `model` in local `useState`, and
both re-fetch only when `editRevision` changes — which
`ProjectContext.settle()` bumps for *track* edits (corrections, undo, retrack),
never for smoothing. So `SmoothingPanel.commit()` PUTs the new filter list,
adopts the curves the server computes, and the curve editor two hundred pixels
away is still showing the previous answer until it remounts. The remount is
what "closes and reopens" does.

**Work:** one copy of the state, in the context, and both views read it.

- `ProjectContext` gains `smoothing: SmoothingResponse | null`,
  `smoothingError`, `reloadSmoothing()`, `commitSmoothing(filters)` and
  `smoothingRevision` (bumped whenever the smoothed curves change — T3 keys off
  it). It loads once per project, reloads on `editRevision` and whenever a
  `solve` job finishes, since a new solve is new raw curves.
- `SmoothingPanel` and `CurveEditor` delete their `load`/`commit`/`raw`/
  `smoothed`/`model` state and use the context's. Writing through one function
  is what makes "the panel changed it" and "the editor changed it" indis­
  tinguishable to both views.
- The curve editor's before/after ghost (`previous.current`) stays local, but is
  now driven by an effect watching the shared `smoothed` — it is a property of
  *that view's* history, not of the project.

**Owns:** `web/src/state/ProjectContext.tsx`,
`web/src/components/SmoothingPanel.tsx`, `web/src/components/CurveEditor.tsx`.

---

## T3 — the canvas shows what the smoothing did to the track

> "When smoothing is applied to a track, the track line in the canvas should be
> updated to reflect the smoothed path."

**Found:** the viewer draws exactly one trail per point — `track.x/track.y`,
the corrected *source-space* path — plus a faint raw ghost where corrections
exist. Nothing on the canvas reflects the smoothing at all. The smoothing does
not produce a per-point path anywhere in the codebase: it produces four camera
curves, and `SmoothedMotion.stabilization_transforms()` turns them into the warp
`smoothed(t) . raw(t)^-1`. The path an operator wants to see is that warp
applied to the point: **where this feature ends up once the shot is
stabilised**, which is precisely "how the smoothing affects this track".

The alternative — pushing the point through the full
`StabilizationPlan.output_transform` — would fold in the export's stabilisation
target and crop fit, so the trail would move when the crop moved. That is the
export preview's job (T1), not this one.

**Work:**

- New `GET /api/projects/{id}/smoothing/tracks`: for every point with a track,
  the corrected path pushed through `stabilization_transforms()`, `null` where
  the track has no position. Computed from the two cached files, so it costs a
  `[T]` matrix multiply per point and no pixels. 409 before there is a solve, in
  the same shape `/export/preview` already uses.
- `ProjectContext` caches the answer (`smoothedTracks`) and re-reads it whenever
  `smoothingRevision` changes — the reason T2 comes first.
- `Overlays` gains a sixth element, `smoothed`, so it takes part in the existing
  global-default / per-track-override machinery rather than becoming a seventh
  special case. Default on: with no smoothing enabled the warp is the identity
  and the two trails coincide, which is a true statement about the project.
- The viewer draws it in the point's own colour, solid, over the source trail,
  with a hollow ring at the current frame — "the point is here now, and would be
  *there* after stabilisation".

**Owns:** `stable/server/routes/solve.py`, `web/src/api/client.ts`,
`web/src/types.ts`, `web/src/state/ProjectContext.tsx`,
`web/src/components/Viewer.tsx`.

---

## T1 — the export is a tab, not a modal

> "Incorporate the export UI into the main UI … another tab on the side panel …
> The preview will be able to use the same canvas and the timeline as the main
> UI."

**Found:** `ExportDialog` is a full-screen modal with two tabs, and the Export
tab is `ExportScreen` — 946 lines holding *its own* canvas, *its own*
`ResizeObserver`, *its own* view transform, *its own* frame-stepping buttons and
*its own* image loader, none of which the viewer's equivalents can be reached
from. So the operator composes the shot on a second copy of the picture that
cannot zoom, cannot pan, has no timeline, no playback and no track overlays, and
the moment the modal opens the real one is hidden behind it.

Everything the screen draws is derived from two server calls
(`/render/options`, `/export/preview`) plus the frame image. Nothing about it
needs a canvas of its own.

**Work:**

- New `web/src/state/ExportContext.tsx`, nested inside `ProjectProvider`: the
  options, the preview, the render library, and every mutation
  (`patchOptions`, `putPairs`, `addTarget`, `removeTarget`). It also carries
  `active` — true while the Export tab is the visible one — because that is what
  tells the viewer to draw the export overlay, and a boolean in a shared context
  beats prop-drilling through the workspace.
- New `web/src/components/ExportPanel.tsx`: the whole side column of the old
  screen (stats, target, crop, border, encode, queue, renders) plus the Data
  links that were the modal's second tab. It is a sidebar panel like the other
  four, so it inherits their layout and scrolling.
- `Viewer` draws the export overlay when the tab is active: the discarded area
  dimmed, the output rectangle, the thirds guides, and the draggable target
  crosshairs — in the viewer's own view transform, so the export preview now
  zooms, pans and scrubs with the timeline like everything else.
  While the export tab is active, a mousedown on a crosshair takes precedence
  over point editing; with the stabilized preview on, the overlay hides itself
  for the same reason the point overlay does (source-space coordinates over a
  warped picture would lie).
- `App` gets a fifth tab and loses the header's Export button and the modal.
  `ExportDialog.tsx` and `ExportScreen.tsx` are deleted.

**Owns:** `web/src/App.tsx`, `web/src/state/ExportContext.tsx` (new),
`web/src/components/ExportPanel.tsx` (new), `web/src/components/Viewer.tsx`,
`web/src/styles.css`; deletes `ExportDialog.tsx`, `ExportScreen.tsx`.

---

## T7 — [BUG] adjusting the crop unsets the placed targets

> "if the user adjusts the Crop size, the UI flickers and the target points
> become unset (even though they are still within the target area). The warning
> is also displayed 'this render would not move the picture at all…'"

**Found: the server is not the culprit, and the client has three defects that
together produce exactly this.** A crop-only `PUT /render/options` provably
keeps a `map` target — the route merges per block and the pairs survive
(pinned by a new test below). What the screen does with that answer is the
problem:

1. **The crop slider is a controlled input over a network round trip.**
   `value={crop.scale}` comes from `opts`, which is only replaced when the PUT
   returns, so every `onChange` while dragging renders the *old* value back into
   the thumb. That is the flicker, and it also fires one PUT **and one
   `/export/preview`** per pixel of drag.
2. **`/export/preview` re-plans the whole clip on every one of those.**
   `export_preview` calls `motion_glue.build_plan` directly, ignoring
   `solve_routes.preview_plan`, which exists precisely to memoise it on the
   caches' mtimes. So a two-second drag is dozens of concurrent O(T) plans; some
   of them lose, and a failed preview is `setPreview(null)`.
3. **The placed targets are read out of the preview.** `pairs`, `paired` and the
   crosshairs all come from `preview?.target.pairs ?? []` — so the moment a
   preview request fails or has not come back yet, the panel believes there are
   no targets: both rows flip from "remove" back to "+ target" and the
   crosshairs vanish. That is the "become unset", and it is why they are "still
   within the target area" — nothing ever moved them.

There is also a genuine read-modify-write race on the server: two overlapping
`PUT /render/options` each load the sidecar, merge their own block and write the
whole file back, so under a burst the later write can carry the earlier one's
stale copy of a block it did not touch.

**Work:**

- The panel reads the target from `options.target` — the authoritative body of
  the PUT/GET it just made — and never from the preview. The preview keeps
  geometry (`rect`, `warp`, `coverage`, residuals) only. A preview that fails
  now shows an error over the frame and leaves every control exactly as it was.
- The crop scale becomes a local draft committed on release (`onChange` moves
  the thumb, `onPointerUp`/`onKeyUp` sends one PUT). No lag, no storm.
- `patchOptions` carries a monotonic request id; a response that is not the
  newest is dropped, so out-of-order answers cannot install stale options.
- `export_preview` goes through `preview_plan`, i.e. the memoised plan.
- `PUT /render/options` takes a per-project lock around read-merge-write.
- Tests: a crop-only update keeps the pairs and the preview still reports them;
  concurrent crop and target updates both survive; the preview is served from
  the memoised plan.

**Owns:** `web/src/state/ExportContext.tsx`,
`web/src/components/ExportPanel.tsx`, `stable/server/routes/solve.py`,
`stable/server/routes/export.py`, `tests/server/test_export_screen.py`.

---

## T4 — the cursor says what a click will do, and a toolbar says what mode it is

> "It's currently always a hand … pointer when hovering over a track point,
> crosshair or plus when the user can add a new point. Perhaps we can also add a
> toolbar … move, add, delete points, or pan."

**Found:** the viewer has three cursor classes (`--placing` crosshair,
`--editing` grab, `--dragging` grabbing) chosen from `placingMode` and "is a
point selected", and *nothing* from what is under the pointer — there is no
hover state at all. So with a track selected the whole canvas is a grab hand,
including empty picture where a click will move the point rather than grab
anything. `placingMode` is already a three-valued mode (`none`, `new-point`,
`add-anchor`); it is a toolbar without a toolbar.

**Work:**

- `PlacingMode` becomes `Tool`: `move | add-point | add-anchor | delete | pan`,
  in the context, with `placingMode` retired and its three call sites
  (`TracksPanel`'s two buttons, the `A` shortcut) moved over.
- A toolbar in the viewer's existing control row, one button per tool with its
  keyboard shortcut in the title (`V`, `N`, `A`, `D`, `H`).
- The viewer tracks what is under the pointer (`hoverPointId`) and picks the
  cursor from tool + hover: `grab`/`grabbing` over the draggable (selected)
  point, `pointer` over another point that a click would select, `crosshair`
  where a click places or moves something, `not-allowed` where the delete tool
  has nothing to delete. Middle-drag still pans in every tool.
- With the `move` tool, clicking a *different* point's marker now selects it
  instead of moving the selected one to that spot — which is what the `pointer`
  cursor has just promised.
- The delete tool removes the anchor pin under the cursor if there is one, and
  otherwise the whole track (confirmed first — deleting a track is not undoable
  the way a correction is). The hint bar states both.

**Owns:** `web/src/state/ProjectContext.tsx`, `web/src/components/Viewer.tsx`,
`web/src/components/TracksPanel.tsx`, `web/src/styles.css`.

---

## T5 — track cards collapse and expand by hand

> "They do it automatically currently, but only when the user clicks on a
> different track."

**Found:** nothing in `TracksPanel` collapses anything. What looks like
collapsing is that three blocks — `PatchPanel`, the bounded-run row and the
keyframe patch grid — are gated on `isSelected`, so selecting another track
hides them here and shows them there. The operator has no say in it, and cannot
have two cards open at once to compare them.

**Work:** an explicit `▸`/`▾` toggle per card, tri-state like the overlay
overrides already in this file: a card the operator has never touched follows
the old default (expanded when selected), and one they *have* touched keeps what
they chose, whatever the selection does. Collapsed shows the identity row and
the summary only; expanded shows everything it shows today.

**Owns:** `web/src/components/TracksPanel.tsx`, `web/src/styles.css`.

---

## T6 — choosing what to solve belongs to the Solve tab

> "Remove the Solve checkbox from the Tracks panel."

**Found:** the checkbox is `points-list__enabled` in `TracksPanel`, writing
`enabled_for_solve` through `PATCH /points/{id}`. `SolvePanel` reads the same
flag (`trackedCount`) and its empty-state text points the operator *back* to the
Tracks tab — "leave its solve box ticked on the Tracks tab" — which is the
confusion the todo describes, written down in the app.

**Work:** the checkbox moves into `SolvePanel` as a list of every point with its
swatch, frame count and confidence, plus **all** / **none** buttons — so
choosing what goes into the solve is one screen with the button that runs it.
Untracked points are listed and disabled, because "why is p4 not in my solve" is
a question the list should answer rather than dodge by hiding it. The
now-incorrect Tracks-tab instruction goes with it.

**Owns:** `web/src/components/TracksPanel.tsx`,
`web/src/components/SolvePanel.tsx`, `web/src/styles.css`.

---

## T9 — delete a keyframe from its image

> "When the user clicks on a keyframe image in the track card they should be
> able to hit delete … There should also be an icon to delete the keyframe."

**Found:** `PatchPanel`'s keyframe thumbnails are click-to-seek and nothing
else. `TracksPanel` already deletes a keyframe from the little `f12` chips
(`handleDeleteAnchor` → `api.deleteAnchor` → `afterEdit`), so the endpoint and
the refresh path exist; the images just do not offer it.

One thing does not carry over. The grid is built from `effective_anchors`, which
is *pins plus the keyframes manual corrections were promoted to*, and
`DELETE /anchors/{frame}` only removes pins — the panel currently cannot tell
the two apart, because the patch entries say `kind: "keyframe"` for both.

**Work:**

- The patch entry gains `pinned`, true when the keyframe is one of the point's
  own `queries`. A promoted keyframe shows its `×` disabled, saying that it
  comes from a correction and pointing at the Edit panel, instead of offering a
  delete that would 400.
- Clicking a keyframe image selects it (and seeks, as now); `Delete`/`Backspace`
  removes the selected one; each keyframe thumbnail also carries a `×`.
- The last remaining pin cannot be deleted — the server refuses, because a point
  with no anchor has nothing to track from — so the `×` is disabled with that as
  its reason rather than letting the click fail.

**Owns:** `stable/server/routes/points.py`,
`web/src/components/PatchPanel.tsx`, `web/src/types.ts`,
`web/src/styles.css`, `tests/server/test_point_patches.py`.

---

## T8 — exports queue

> "Exporting videos should be a queable process, allowing users to queue
> multiple export jobs and have them processed in the background."

**Found: half of it is already there and the other half is actively wrong.**
`JobManager` is a FIFO queue with a worker pool and an `exclusive=True` flag
"for work that saturates a shared resource (an FFmpeg encode)" — and
`export_video_route` does not pass it, so N submitted exports all run at once
and fight over the cores.

Worse, a queued export does not carry its own settings. `_run` re-loads the
project, the smoothing cache **and** the render options *when the worker gets to
it*, and re-plans from those. So queueing "this crop" and then "that crop" gives
two renders of whichever crop was last saved — the queue would silently be a
lie. The plan is already built at submit time (for the no-op check) and thrown
away.

There is also no way to cancel: `JobStatus` has a `"cancelled"` member that
nothing ever sets.

**Work:**

- Submit-time capture: the plan built for validation, the clip range and the
  encode settings are closed over and used by the job. What you queued is what
  gets rendered, and the O(T) plan is built once instead of twice.
- `exclusive=True`, so exports run one at a time and the rest wait — which is
  what makes the list a queue rather than a stampede.
- `JobManager.cancel(job_id)` marks a *queued* job cancelled (the worker skips
  it) and refuses a running one, with `POST /jobs/{id}/cancel` in front of it.
- The export panel grows a queue: every export job newest first, with its status,
  progress, resulting filename or error, and a cancel button while it is still
  queued. The Export button says "Add to queue" and no longer waits.
- Tests: two queued exports render their own settings; the second waits for the
  first; a queued export can be cancelled and a running one cannot.

**Owns:** `stable/server/jobs.py`, `stable/server/routes/export.py`,
`stable/server/routes/projects.py`, `web/src/api/client.ts`,
`web/src/components/ExportPanel.tsx`, `tests/server/test_export_screen.py`.

---

## Checks

Between items: `pytest -m "not integration" -q`, `ruff check .`,
`ruff format --check .`, `mypy stable`, `cd web && npm run build`.
At the end: `python scripts/build_frontend.py` so the packaged snapshot
`tests/server/test_web_dist.py` guards is not left stale.
