# Implementation plan — sixth feedback round

Source: `todos_5` (7 items). Same conventions as `done/TODO_PLAN.md`,
`done/NEW_TODO_PLAN.md`, `done/TODO_2_PLAN.md`, `done/TODO_3_PLAN.md` and
`TODOS_4_PLAN.md`: every work item states what was **found in the code**, not
what was assumed, and the suite stays green throughout (`pytest -m "not
integration"`, `ruff check`, `ruff format --check`, `mypy stable`,
`npm run build`, then `python scripts/build_frontend.py`).

Baseline, measured rather than assumed: 7 failures on this machine, all of them
"PyTorch is not installed" —
`tests/test_deep_trackers.py::test_pad_to_multiple_only_pads_when_it_must` and
six in `tests/test_sam3_tracker.py`. Nothing in this round touches the deep
trackers, so that is the number to come back to.

---

## What the seven items actually are

Three of them are about the export screen the last round built, two are about
patches, one is a crash, and one is a small transport control. They are not
independent: the crash is *in* the frame stream the export screen draws from, so
it goes first — everything else in the export view is worked on over the top of
it.

| # | Item (`todos_5` line) | Owns | Depends on |
|---|---|---|---|
| W1 | The detached-bitmap crash (12–22) | `api/frameStream.ts`, `Viewer` | — |
| W2 | Play only between In/Out (1) | `ProjectContext`, `Timeline` | — |
| W3 | Crop preview follows the slider (24) | `ExportPanel` | — |
| W4 | Targets on the output canvas (10) | `server/frames.py`, `frameStream.ts`, `ExportContext`, `Viewer`, `styles.css` | W1 |
| W5 | The Edit tab goes away (7–8) | `App`, `Viewer`, `TracksPanel`, new `TrackEdits`; deletes `EditPanel` | — |
| W6 | Patches: cache them, and don't show them where they mean nothing (3, 5) | `routes/points.py`, `PatchPanel` | — |
| W7 | Tests | `tests/server/*` | W4, W6 |

---

## W1 — the crash: a bitmap freed while a frame is still assembling

> `InvalidStateError: Failed to execute 'drawImage' … The image source is
> detached` at `Viewer.tsx:314`, on a reasonably quick scrub in the export view.

**Found.** `Viewer.tsx:314` is `ctx.drawImage(picture.image, …)` where `picture`
is `frames.view.images[mainVariant]` — a `Decoded` whose `image` is an
`ImageBitmap`. "Detached" means something called `.close()` on it. Two places
do, and one of them is wrong:

* `FrameStream.release()` (`api/frameStream.ts:303`) refuses to close a bitmap
  that is in **the published view**, deferring it instead. That is the only
  ownership it knows about.
* `show()` (`:479`) pulls cached pictures straight into `assembly.images` — a
  frame that is *part-way through arriving* holds `Decoded` references that the
  cache also holds, and `release()` has never heard of it.

So `store()`'s LRU eviction (`:342`) and `clear()` (`:325`) will close a bitmap
an in-flight assembly is holding. When that assembly completes it publishes a
view containing a closed bitmap, the viewer draws it, and `drawImage` throws
from inside a `useEffect` — which React treats as a render error and unmounts
the tree. That is the blank interface, exactly.

Why the export view and why "reasonably quickly": `CACHE_LIMIT.output` is 30
against a `PREFETCH_AHEAD.output` of 3, `ASSEMBLY_LIMIT` is 4, and every
`patchOptions` call fires `invalidateOutput()` → `clear("output")`. A scrub with
the export tab up is the one situation where several assemblies are alive at
once *and* the cache underneath them is being churned.

`flushReleases()` at publish time is the same bug from the other end: it frees
bitmaps the *previous* view was drawn from, at a moment when React has not yet
re-rendered. Nothing redraws in that window today, so it has not bitten — but it
is one `sizeTick` away from doing so.

**Work.** Reference-count `Decoded`, which is what both guards were
approximating:

* `decode()` hands back a `Decoded` with one reference (the cache's).
* `retain()` / `release()`; the underlying `ImageBitmap.close()` or
  `URL.revokeObjectURL` happens exactly when the count reaches zero.
* Every owner retains: the variant cache, each `Assembly` that holds it, the
  published `FrameView`. Each drops its reference when it is replaced, evicted,
  cleared or superseded.
* `deferredRelease` / `flushReleases` / the "is it on screen" scan are deleted.
  A bitmap on the canvas is a bitmap the published view holds a reference to,
  which is the same statement without the special case.

Belt and braces, because a blank app is a bad failure mode for *any* cause: the
viewer's two `drawImage` calls are guarded, and a picture that will not draw is
reported as an error over a black frame rather than taken down the React tree
with it.

**Owns:** `web/src/api/frameStream.ts`, `web/src/components/Viewer.tsx`.

---

## W2 — play only between the In/Out markers

> "Add checkbox on the playback timeline to play only between in/out markers."

**Found.** `Timeline.tsx:102` is the playback loop, and it knows nothing about
the range: it advances from wherever the playhead is until
`next >= project.video.frames`, then stops and rewinds to 0. `clipRange` is
already on the context and already drawn (`:212` dims outside it), and tracking
and export already honour it — playback is the last thing that does not.

**Work.** `playRangeOnly` beside `playbackRate` in `ProjectContext`, a checkbox
in the transport next to the In/Out group, disabled with a title saying why when
no range is set. The loop clamps to `[in, out)`: a playhead outside the range
jumps to `in` when play starts, and reaching `out` stops and returns to `in` —
the same "stop and rewind" the whole-clip case already does, applied to the
range rather than to the clip. Not a loop: that would be a different feature
from the one asked for, and Play is still Play.

**Owns:** `web/src/state/ProjectContext.tsx`, `web/src/components/Timeline.tsx`.

---

## W3 — the crop preview follows the slider

> "the preview doesn't get updated until the user releases the mouse button."

**Found.** `ExportPanel.tsx:405-418`. The slider's value is local `cropDraft`
state and the only calls to `commitCrop` are on `onPointerUp` / `onKeyUp`. That
was deliberate (the comment at `:112` explains it): a controlled input over a
round trip renders the stored value back into the thumb on every move, which
rubber-bands, and one PUT per pixel of drag is a plan rebuild per pixel. The
consequence is the one being complained about — `patchOptions` is what re-plans,
re-fetches the preview and calls `invalidateOutput()`, so *both* canvases sit on
the old crop until the button comes up.

**Work.** Keep the draft as the control's value — that half was right and the
rubber-band must not come back — and commit *during* the drag, throttled: at
most one `patchOptions` in flight, at most one queued behind it, ~120 ms
trailing, and the final value always sent on release.

Not a client-side approximation of the rectangle. The crop scale changes
`plan.geometry`, which moves the output rectangle on the left canvas **and** the
picture on the right; approximating one and re-planning the other would put the
two canvases into disagreement, which is the thing the last round exists to
prevent. The server side is a memoised O(T) fit plus a small JSON write, and at
~8 Hz on a local single-user tool that is affordable — W7 measures it rather
than asserting it.

**Owns:** `web/src/components/ExportPanel.tsx`.

---

## W4 — the targets, on the output canvas too

> "show the target points on the right output canvas too … in sync with the left
> canvas … also be editable in the right (output) view canvas … The 'show
> thirds' option should also apply to the right (output) canvas too."

**Found.** The output canvas (`Viewer.tsx:703`) draws `outputPicture`
letterboxed and nothing else — no overlay, no handlers, and
`styles.css:252` sets `cursor: default` on it. It cannot draw the crosshairs
today even if asked to: `frame_geometry` (`server/frames.py:328`) sends `rect`
(the output rectangle's corners **in source pixels**) and `warp` (the
stabilisation warp, which is the space a mapped target's destination is stated
in) — but not the transform that maps source pixels into **output** pixels,
which is what the right canvas is in. That transform exists,
`plan.output_transform(idx)`, and `rect` is literally its inverse applied to the
output corners; it is simply not on the wire.

**Work.**

*Server.* `frame_geometry` gains `output_warp`: the four numbers of
`plan.output_transform(idx)`, which is a `SimilarityTransform` like `warp`
already is. Sent rather than inferred from three corners of `rect` — the
inference is exact for a similarity but is a second implementation of a fact the
plan already holds, and this is the module whose whole reason for existing is
that there is one of each.

*Client.* `FrameGeometry` and `ExportPreview` carry `output_warp` (the preview
too, so the socket-down fallback is not a differently-shaped geometry). The
output canvas gets:

* its own fit transform (output pixels → canvas pixels; still not linked to the
  left canvas's zoom/pan, which remain different coordinate spaces);
* thirds when `showGuides` is on — here they are thirds of the whole output
  frame, because on this canvas the output rectangle *is* the frame;
* one crosshair per pair at
  `applyWarp(output_warp, unapplyWarp(warp, pair.x, pair.y))` — the same two
  helpers the left canvas already uses, composed;
* a drag: canvas → output px → `unapplyWarp(output_warp)` → source px →
  `applyWarp(warp)` → the stored destination. Same `nudgeTarget` while the
  button is down, same single `putPairs` on release, same
  `exportState.dragging` highlight — which is what makes the crosshair move on
  **both** canvases at once. "In sync" is not a synchronisation step here: both
  canvases draw from one `frames.view`, so they are the same frame by
  construction, and both read `exportState.pairs`, so they are the same targets
  by construction.

Everything else about the right canvas is unchanged: no point markers, no tools,
no zoom. It is the file, plus the one thing being composed on it.

**Owns:** `stable/server/frames.py`, `web/src/api/frameStream.ts`,
`web/src/state/ExportContext.tsx`, `web/src/components/Viewer.tsx`,
`web/src/styles.css`.

---

## W5 — the Edit tab goes away

> "The Edit tab doesn't really serve a purpose right now … Hide the edit tab and
> move the functionality to the canvas … allowing patches to be moved and
> deleted directly on the canvas and track cards in the sidepanel. And moving
> the setting to blend/hard move points when editing to the toolbar."

**Found.** `App.tsx:16` lists five tabs. `EditPanel.tsx` holds four things, and
the todo is right about three of them:

* **undo/redo** (`:144`) — already in the workspace header (`App.tsx:62`), so
  it is a duplicate;
* **`dragMode` + `falloff`** (`:165`) — the only genuinely global control on the
  panel, and the one the todo asks to move;
* **gap fill / delete range / retrack-from-here** (`:188`, `:276`) — per point;
  `TracksPanel` already has a *better* retrack (`◀ track` / `track ▶` with a
  bound, `:412`), so that one is a duplicate too;
* **the corrections list** (`:296`) — per point, and has no other home.

Meanwhile the canvas already does the moving and deleting the todo credits it
with: drag and click-to-move (`Viewer.tsx:1030`, `:873`), delete a track, delete
an anchor pin (`deleteUnderPointer`, `:924`). What it cannot do is **move** a
pin — anchors are drawn (`:535`) and hit-tested for deletion, but not draggable.

**Work.**

1. `dragMode` and `falloff` move into the viewer toolbar, beside the tools, as
   the todo asks. The hint line under the canvas already states what a move will
   do, so the two now sit within a glance of each other.
2. Anchor pins become draggable in the `move` tool: mousedown within
   `GRAB_RADIUS` of a pin on `viewFrame` starts a drag, release stores the new
   position through `addAnchorAt`. The server already treats an anchor added at
   an existing frame as a replace and says so
   (`routes/points.py:189`, "a pin drag in the UI is expressed as
   delete-then-add"), so this is one request. Pins take precedence over track
   markers under the cursor: a pin is the smaller, more deliberate thing, and it
   sits on top of the marker it pins. The cursor says which is which.
3. Deleting a pin stays where it is (the `delete` tool on the canvas; the ×
   on each keyframe chip in the track card, `TracksPanel.tsx:312`; the × on each
   keyframe thumbnail, `PatchPanel.tsx:194`) — three routes to it already, all
   working, none of them the Edit tab.
4. The per-point half of `EditPanel` — corrections list, gap fill, delete range
   — is extracted into `TrackEdits({ pointId })` and rendered inside the track
   card's disclosure, so a point's edits live with the point. This is the part
   that would otherwise be *lost* rather than *moved*, which is the difference
   between hiding a tab and deleting a feature.
5. `App.tsx` drops the `edit` tab and `EditPanel.tsx` is deleted.

**Owns:** `web/src/App.tsx`, `web/src/components/Viewer.tsx`,
`web/src/components/TracksPanel.tsx`, `web/src/components/TrackEdits.tsx`
(new), `web/src/styles.css`; deletes `web/src/components/EditPanel.tsx`.

---

## W6 — patches: cache them, and stop showing them where they mean nothing

### Item 3 — "check and verify"

> "When a patch/keyframe is made it should be cached on the server … This may
> have already been resolved with performance improvements, check and verify."

**Found: mostly yes, and here is the part that is not.** The previous round did
put a real cache in (`routes/points.py:556-622`):

* `_PATCH_CACHE`, a 64 MB LRU keyed on
  `(project, point, frame, size, source, stale_key)`, where `stale_key` is the
  project file's and the track file's mtimes — so an edit invalidates exactly
  what it should and nothing else;
* an `ETag` off that key plus `Cache-Control: private, max-age=30`, so a scrub
  back over covered frames is a 304;
* `source="proxy"` as the **default**, reading the half-res JPEG rather than
  seeking the clip;
* and `AllPatchesStrip` (`PatchPanel.tsx:373`) stopped fetching altogether — it
  crops the frame already decoded on the canvas.

What is left is exactly what the todo describes. `PatchPanel`'s keyframe grid
asks for `source="original"` (`:226`, deliberately — that panel is where texture
is judged, and proxy scaling is what takes texture away). `original` is a
random-access seek into the full-resolution clip, and the cache is only ever
filled *by the first slow request*. So opening a track card with eight
keyframes is eight clip seeks serialised behind the decoder's lock, every time
the server has been restarted — which is the "slow to load" being reported, and
it is a cold-cache problem, not a missing-cache one.

**Work.** Fill the cache when the keyframe is *made*, which is what the todo
asks for literally:

* a single-worker warmer with a de-duplicating in-flight set, so it can never
  pile threads onto the decoder lock, and a queue bound so a backlog of guesses
  cannot outlive the interest in them (or delay process shutdown, since the
  executor is joined at exit);
* `POST .../points` and `POST .../points/{id}/anchors` schedule a warm of the
  new keyframe's patch at the point's effective radius, in both sources;
* `GET .../patches` schedules a warm of every keyframe it lists — which covers
  the anchors that pre-date this change and every restart, and lands while the
  panel is still laying out its `<img>` tags;
* failures are swallowed. A warm cache is an optimisation, and an optimisation
  that can fail a request is a bug.

**Found while building it, and worth recording.** The first version resolved
each patch's position *inside* the worker (`load_project`, then
`_patch_position`). On Windows that broke the request that triggered it: the
project file is saved by `os.replace` onto itself, and a background reader
holding it open makes that fail with `WinError 5` — `test_point_patches.py`
caught it immediately. So the caller resolves everything (position, radius, and
the cache key itself) on the request thread, after its own save, and the worker
does nothing but read pixels, encode and store. A warmer that can break the save
it was triggered by is worse than no warmer.

### Item 5 — no patch images for trackers that do not use patches

> "For algos that don't use patches we don't need to show the patch/keyframe
> image in the side panel."

**Found.** The server already knows: `tracker_window`
(`tracker_options.py:558`) returns `(DEFAULT_PREVIEW_RADIUS, None)` when the
tracker has no local window, and `/patches` passes that through as
`radius_source: null` with `window.adjustable: false`. `PatchPanel` reads it
only for a *tooltip* (`patchSizeExplanation`, `:478`: "…has no local search
window — it sees whole frames — so this is a preview size only") and draws the
grid of thumbnails anyway — pictures of a region nothing matches.

**Work.** With `radius_source === null`:

* no "This frame" thumbnail and no thumbnail grid; the keyframes are still
  listed, as plain frame chips with their delete ×, because W5 needs deleting a
  keyframe to keep working from here;
* the explanation is shown as text rather than hidden in a `title`;
* `AllPatchesStrip` skips those points, and renders nothing at all when no point
  has a window;
* the warmer above skips them too — there is no sense rendering images nothing
  will draw.

**Owns:** `stable/server/routes/points.py`,
`web/src/components/PatchPanel.tsx`.

---

## W7 — tests

* `tests/server/test_frame_stream.py`: a want's geometry carries `output_warp`,
  and applying it to `rect`'s four corners gives `(0,0)`, `(W,0)`, `(W,H)`,
  `(0,H)` — the new field pinned against the field the screen already trusts,
  so the two cannot drift.
* `tests/server/test_export_screen.py`: the REST preview carries `output_warp`
  too, since it is the fallback and must not be a different geometry.
* `tests/server/test_patch_serving.py`: a patch is in the cache after the anchor
  that created it, without anything having asked for its image — the warmer
  does what it says. Plus a measurement of the crop-scale round trip
  (`PUT /render/options` + `GET /export/preview`) so W3's ~8 Hz has a number
  behind it rather than an assertion.
* `tests/server/test_tracker_windows.py`: the deep-tracker case already pinned
  `radius_source: null` on `/patches`; extended with the same fact on the point
  *summary* (`window.attribute`), which is the field the all-points strip
  filters on — two views of one fact, decided in one place.

Both warming tests were checked against their own absence: with the warmer
disabled they fail with "read the proxy" / "read the clip", which is the
assertion doing its job rather than passing for a different reason.

There is no frontend test runner in `web/` (`npm run build` is `tsc --noEmit &&
vite build`), so the type-check is the frontend gate, as in previous rounds.

---

## Checks

Between items: `pytest -m "not integration" -q`, `ruff check .`,
`ruff format --check .`, `mypy stable`, `cd web && npm run build`.
At the end: `python scripts/build_frontend.py`, so the packaged snapshot
`tests/server/test_web_dist.py` guards is not left stale.
