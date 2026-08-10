# Implementation plan — ninth feedback round

Source: `todo_8` (one bug, seven improvements). Same conventions as the earlier
rounds (`done/TODO_PLAN.md` … `TODO_6_PLAN.md`): every item states what was
**found in the code**, not what was assumed, and the suite stays green
throughout (`pytest -m "not integration"`, `ruff check`, `ruff format --check`,
`mypy stable`, `npm run build`, then `python scripts/build_frontend.py`).

Baseline, measured rather than assumed: the full non-integration suite was run
before any edit and reported **2 failures**, both in
`tests/test_phase3_acceptance.py`
(`test_the_solve_recovers_the_ground_truth_motion` and
`test_the_recovered_curves_match_the_known_channels`) — the same numerical
solver tolerance the seventh round recorded (a solved rotation of `2.133e-2`
against `2.154e-2 ± 2e-4`). Nothing in this round touches the solver, so that
is the number to come back to, not a target to improve.

After: **2215 passed, 2 failed of 2217** — the same two, unchanged.
`ruff check`, `ruff format --check` on every file touched, `mypy stable`
(88 files) and `npm run build` (`tsc --noEmit` + Vite) are all clean, and
`python scripts/build_frontend.py` has refreshed the packaged
`stable/server/webdist` so `stable serve` serves this build rather than the
previous one.

Scope: **these eight items only**. Nothing else in `todo_8`, nothing adjacent
that looks tempting on the way past.

---

## The items, and what each one actually is in the code

| # | `todo_8` line | Verdict |
|---|---|---|
| 1 | add-anchor tool disarms itself after one placement | Real bug, one line |
| 2 | remove the "Ghost" lines | Delete an overlay channel end to end |
| 3 | default "Drag makes" to hard replace | One default, plus the prose that quotes it |
| 4 | expand/contract arrow on the export target card | Disclosure affordance for an existing expansion |
| 5 | one click to hide a whole track | New per-track visibility layer, above the per-element overrides |
| 6 | let the user change a track's colour | New persisted `color` on the point — server + client |
| 7 | "Paths [dropdown]" → "Colour Paths by [dropdown]" | Label and option copy |
| 8 | vertical separators in the toolbar | One element, one CSS rule |

---

### 1. The add-anchor tool disarms itself (bug)

`web/src/components/Viewer.tsx` `handleClick`:

```ts
if (tool === "add-anchor" && selectedPointId) {
  setTool("move");                                   // ← the bug
  await addAnchorAt(selectedPointId, viewFrame, x, y);
```

Placing an anchor switches the tool back to Move, so anchoring the same track on
four frames is four trips to the toolbar. Reported exactly as such.

**Fix:** delete the `setTool("move")`. The tool stays armed until it is put away
deliberately, and there are already three ways to do that — `Esc`, `V`, and `A`
(which toggles `add-anchor` off, `ProjectContext.tsx`), plus the panel buttons
that call `setTool("move")` when the armed tool is clicked again. The hint line
under the canvas gains "— stays armed; Esc when done" so the stickiness is
stated rather than discovered.

**Deliberately not changed:** `add-point` keeps disarming itself. Its two entry
points (both Tracks panels' header button) say "Click video to place…" —
singular, and a tool that keeps making tracks out of stray clicks on the picture
is a different and worse bug. The reported item is about anchors.

### 2. Remove the "Ghost" lines

The ghost is a whole overlay channel: `Overlays.ghost` (`ProjectContext.tsx`,
plus `ALL_OVERLAYS` / `NO_OVERLAYS`), the `OVERLAY_LABELS` row that gives it its
checkbox above the canvas and its per-track pill in *both* Tracks panels, and
~25 lines in the `Viewer` draw effect that stroke `track.raw_x` / `raw_y`
dashed. `TracksPanel.tsx` and `TrackTimelinePanel.tsx` render it only through
`OVERLAY_LABELS`, so they need no edit of their own.

**Fix:** delete the channel — type field, both constants, the label row, the
draw block, and the `showGhost` binding. Prose that describes the app as
ghosting the raw path (the `Viewer` module docstring) goes with it.

**Deliberately not changed:** the server keeps sending `raw_x` / `raw_y` and
`types.ts` keeps declaring them. Removing a served field is an API change this
item did not ask for, and the curve editor's *own* before/after ghost (previous
smoothing, `CurveEditor.tsx`) is a different feature that was not reported.

### 3. Default "Drag makes" to hard replace

`ProjectContext.tsx`: `useState<DragMode>("offset")`. One word to change, plus
the two places that describe the default in prose (the `DragMode` docstring's
"the everyday …" framing stays true; the viewer hint already reads the live
value, so it needs nothing).

### 4. An expand/contract arrow on the export target card

`ExportPanel.tsx`: a target card already expands — clicking anywhere on the
`<li>` toggles `selected`, which reveals source, position, snap spots and the
track checkboxes. Nothing on the card says so.

**Fix:** a `▸`/`▾` disclosure button as the first child of `.target-list__head`,
with `aria-expanded`, toggling exactly the same `selected` state (not a second
one — the selection also highlights the crosshair on both canvases, and two
notions of "open" would drift). Styled from the existing
`.points-list__disclose`, which is what the Tracks card's caret already uses.

### 5. One click to hide a whole track

Today hiding a track means clicking six per-element pills (`Paths`, `Smoothed`,
`Points`, `Anchors`, `Ghost`, `Window`) on its card, and after item 2 that is
still five. The per-element state is deliberately tri-state — override, or
follow the default — so "set them all to false" would destroy whatever
per-element choices the operator had made and could not be undone by one click.

**Fix:** a *separate* visibility layer, above the overrides rather than made of
them. `ProjectContext` gains `hiddenTracks: Record<string, boolean>` with
`isTrackHidden(pointId)` and `setTrackHidden(pointId, hidden)`, and
`overlaysFor` resolves in one order:

```
master switch off  →  nothing
track hidden       →  nothing
otherwise          →  defaults, with this track's overrides on top
```

so unhiding restores exactly what was there before. An eye button goes on the
track row head in **both** Tracks panels (`TracksPanel`'s `points-list__row`,
`TrackTimelinePanel`'s `track-row__head`) — the row itself, not behind the
card's disclosure or the `⋯`, because "one click" is the request. Hidden state
is drawn as a struck-through eye in CSS rather than a second glyph, so it
cannot land on a font that lacks it. The eye (and the colour control below)
live in one new `components/TrackVisibility.tsx` rather than being written
twice: two panels behind a flag can only be judged against each other if they
mean the same thing by the same control.

While a track is hidden its per-element pills are `disabled` — nothing they can
say changes what is drawn — which is the treatment the master `Overlays` switch
already gives the checkboxes above the canvas.

Two hit-tests follow the eye: `hitTest` and `deleteUnderPointer` in `Viewer.tsx`
skip hidden tracks. Selecting — or deleting — a track that is not on the screen
is not a feature, and the delete tool reaching an invisible anchor is the
hazardous version of it.

**Deliberately not changed:** the main timeline strip and the panels' own
anchor/status rows still show hidden tracks. The item enumerates the canvas
overlay elements, and a track you cannot find in the panel is a worse problem
than the one being fixed.

### 6. Let the user change a track's colour

`pointColor(pointId)` (`colors.ts`) hashes the id into an eight-entry palette
and is called from seven components. Two tracks can therefore collide on one
colour with no recourse, which is the complaint.

Colour is a property of the track, so it is **persisted with the project**
rather than kept in the tab:

* `PointSpec.color: str | None` (`stable/core/project.py`) — `None` means "use
  the palette", so every existing project file loads unchanged. Validated in
  `__post_init__` as `#rrggbb` and normalised to lower case, the same shape of
  guard `SmoothingSpec`/`RenderSettings` already use for their enums.
* `PointUpdate.color` (`schemas.py`) and `update_point` (`routes/points.py`) —
  `""` clears it back to the palette, since a JSON `null` is already "leave it
  alone" for every other field on that body.
* `point_summary` (`serialize.py`) carries `color`, so the UI needs no second
  request.
* Client: `types.ts` gains `color`, `api.updatePoint` accepts it, and
  `ProjectContext` exposes `colorFor(pointId)` — the point's colour, else
  `pointColor(pointId)`. Every drawing call site switches to it (`Viewer`,
  both Tracks panels, `Timeline`, `PatchPanel`, `SolvePanel`, `ExportPanel`);
  all seven already consume the project context.
* The control is a native `<input type="color">` styled as the existing swatch,
  in the row head of both Tracks panels — the swatch is already the thing that
  means "this track's colour", so it becomes the thing you click to change it,
  with a `↺` reset (shown only when the point has a colour) that sends `""`.
  The picker reports every colour the pointer crosses inside it, so the swatch
  follows the hand and the project is told 400 ms after it stops — one request
  and one project save per decision, not per pixel. Committing on blur instead
  would not work: the OS dialogue keeps focus.

Tests: a server test that the colour round-trips through `PATCH` and is cleared
by `""`, and a project-file test that it survives save/load and that a bad value
is refused. `PLAN.md`'s data-model block gains the field (the contract rule in
`CONTRIBUTING.md`).

### 7. "Paths [dropdown]" → "Colour Paths by [dropdown]"

`Viewer.tsx` toolbar: the label is the bare word `Paths` and the options read
"by confidence" / "by track", so the control reads as a noun with two adverbs.
Becomes the label `Colour Paths by` with options `confidence` and `track`, as
requested. The `title` keeps the explanation of what each choice is for.

### 8. Vertical separators in the toolbar

The toolbar is four groups — tools, zoom, drag mode, visualisation options —
separated by nothing but a 14 px gap. Add a `.viewer__sep` (1 px, stretched,
`--border`, `aria-hidden`) between the groups. Presentation only.

---

## Order of work

1. Items 1, 3, 7, 8 — self-contained edits to `Viewer.tsx` / `ProjectContext.tsx`
   and `styles.css`.
2. Item 2 — the ghost removal, which changes `Overlays` and so must land before
   anything else touches overlay resolution.
3. Item 5 — the visibility layer on top of the trimmed `Overlays`.
4. Item 4 — `ExportPanel` disclosure.
5. Item 6 — server first (schema, route, serialiser, tests, `PLAN.md`), client
   second (`types.ts`, `client.ts`, `colorFor`, the seven call sites, the
   swatch control).
6. `npm run build`, `pytest -m "not integration"`, `ruff check`,
   `ruff format --check`, `mypy stable`, then `python scripts/build_frontend.py`
   so the bundled `stable/server/webdist` matches the source.

## What could go wrong, and the answer

* **`colorFor` in a canvas effect's dependency list.** It is a `useCallback` over
  `project.points`, so it changes identity exactly when a colour changes — which
  is when a redraw is wanted. No effect gains a per-render dependency.
* **A hidden track that cannot be found again.** The eye is on the row head, on
  the same row as the id, and hidden state is visible there; nothing else about
  the track disappears from the panel.
* **Old project files.** `color` is absent → `None` → the palette, which is
  exactly today's behaviour. `test_unknown_keys_are_ignored` already covers the
  reverse direction (a new file opened by older code ignores the field).
