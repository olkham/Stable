# Implementation plan — fifth feedback round

Source: `todos_4` (2 items). Same conventions as `done/TODO_PLAN.md`,
`done/NEW_TODO_PLAN.md`, `done/TODO_2_PLAN.md` and `done/TODO_3_PLAN.md`: every
work item states what was **found in the code**, not what was assumed, and the
suite stays green throughout (`pytest -m "not integration"`, `ruff check`,
`ruff format --check`, `mypy stable`, `npm run build`, then
`python scripts/build_frontend.py`).

Baseline: the two pre-existing `tests/test_phase3_acceptance.py`
rotation-tolerance failures noted in the previous rounds are still there and are
still nobody's here.

---

## What the two items actually are

They look like one feature and one optimisation. They are not: the side-by-side
view **is** the thing that makes the current transport untenable, and the
transport change **is** what makes the operator's no-negotiation requirement
(picture in sync with track data) true rather than approximately true.

### Found — the frame path today

* `Viewer.tsx` gets its picture from `hooks/useFrameImages.ts`, which loads
  `/api/projects/{id}/frames/{n}` (or `.../stabilized`) with an `Image()` per
  frame, caches decoded images by URL, and prefetches ±12/4 neighbours.
* `/frames/{n}` is a `FileResponse` off the proxy store — a disk read with an
  ETag, so a browser re-scrub costs a 304. Cheap.
* `/frames/{n}/stabilized` (`routes/solve.py:486`) is **not** cheap and **not**
  cacheable: JPEG decode → `warp_frame` → JPEG encode, per request, and the
  pixels change whenever the render options change so no validator can be
  offered. Today it is only reached by the header's "Stabilized preview"
  toggle. Item 1 puts it on screen *permanently* while the Export tab is open,
  at playback rate.
* `/export/preview` (`routes/export.py:386`) is re-asked **once per frame**:
  `ExportContext.reloadPreview` is a `useCallback` closing over `currentFrame`,
  and the effect that calls it lists that callback in its deps. So scrubbing
  with the Export tab up fires one planning request per frame — which is
  precisely the log the todo pastes (`preview?frame=247`, `244`, `259`, out of
  order). The plan itself is memoised (`preview_plan`), so each one is cheap;
  the round trip per frame is not, and the out-of-order arrivals are handled by
  a sequence guard that simply *drops* the losers.

### Found — the sync defect, which no transport fixes by itself

`useFrameImages` deliberately keeps the previous image on screen until the next
one decodes (its docstring: "the picture only ever changes to another picture").
Everything drawn *over* that picture — every track path marker, the export's
output rectangle, its thirds guides, its target crosshairs — is indexed by
`currentFrame`, the playhead. During a scrub those are different frames. The
canvas therefore draws frame 244's pixels under frame 259's markers, and the
markers visibly float off their features.

That is the todo's "absolute MUST, no-negotiation" already being violated, on
the transport we have. Moving to a socket does not fix it; drawing everything
for **the frame whose pixels are actually on the canvas** does. So that is W1,
and it lands before any transport work.

### Answering the transport question honestly

> Could we consider using a WebSocket … rather than making multiple GET
> requests? This would reduce the overhead of establishing multiple connections…

The connection overhead is not the win — the browser keep-alives a single
HTTP/1.1 connection to a local server, so there are no repeated handshakes.
Three other things are the win, and they are real:

1. **Cancellation.** Over HTTP the server cannot know that the request for
   frame 244 is worthless because the user is now on 259; it warps all three.
   Over one socket the server keeps only the newest want per variant and drops
   the superseded ones. A 60-frame scrub becomes ~5 warps instead of 60. On the
   stabilized variant that is the difference between usable and not.
2. **Head-of-line.** The browser caps ~6 connections per origin
   (`MAX_IN_FLIGHT = 6` in the hook exists because of it). The frame the user is
   *looking at* can queue behind prefetches. One socket has one queue and the
   server controls its order — wants first, prefetch only when idle.
3. **Atomicity, which is the MUST.** Pixels and the per-frame geometry that is
   drawn over them arrive in **one message**, so there is no window in which
   they can disagree. Two HTTP responses can always land in either order.

What we do **not** get for free is the browser's HTTP cache, so the client keeps
its own decoded cache keyed by `variant:frame:rev` — it already keeps one keyed
by URL, so this is a change of key, not of concept. And the REST routes stay:
they are the fallback when the socket is not open, and they are what the server
tests assert against.

---

## Order of work

| # | Item | Owns | Depends on |
|---|------|------|-----------|
| W1 | Draw for the frame that is on the canvas | `Viewer` | — |
| W2 | One server-side producer of "a frame + its geometry" | `server/frames.py` (new), `routes/solve.py`, `routes/export.py` | — |
| W3 | The frame WebSocket | `routes/ws.py`, `app.py` | W2 |
| W4 | The client stream, with an HTTP fallback | `api/frameStream.ts` (new), `state/FrameStreamContext.tsx` (new), `Viewer`, `App` | W1, W3 |
| W5 | The side-by-side output canvas | `Viewer`, `ExportPanel`, `ExportContext`, `styles.css` | W4 |
| W6 | Stop the per-frame `/export/preview` | `ExportContext`, `ExportPanel`, `Viewer` | W4 |
| W7 | Tests | `tests/server/test_frame_stream.py` (new), `test_export_screen.py` | W3, W6 |

W1 first because it is small, is the requirement, and is independently correct.
W2 before W3 so the socket and the two REST routes cannot drift about what a
frame or its geometry *is*. W5 after W4 because a second canvas fed by
per-frame HTTP warps is the thing we are trying not to ship.

---

## W1 — the overlay is drawn for the frame that is on the canvas

**Found:** `Viewer.tsx` draws `frames.image` (possibly the previous frame — the
hook says so, via `isCurrent: false`) and indexes every overlay by
`currentFrame`. The two disagree for as long as a frame is in flight.

**Work:** the frame source reports *which frame* its image is, and the viewer
takes a single `viewFrame` from that. Every draw and every canvas interaction —
markers, live-tracking cursor, smoothed ring, window boxes, hit tests,
`commitMove`, `deleteUnderPointer`, `addAnchorAt` — reads `viewFrame`, never
`currentFrame`.

Interactions move with the picture deliberately: an edit lands on the frame the
operator can see, not on a playhead position whose picture has not arrived. When
nothing is in flight the two are equal, which is the normal case.

The frame label says so when they differ (`frame 259 / 500 · showing 244`),
because a viewer that silently shows a different frame than the playhead claims
is worse than one that admits it.

**Owns:** `web/src/components/Viewer.tsx`.

---

## W2 — one producer of "a frame and its geometry"

**Found:** the knowledge is duplicated and about to be triplicated.
`get_stabilized_frame` (`routes/solve.py`) holds proxy-scale rescaling, the
`_scaled_fill` correction, the RGBA→RGB fallback and the JPEG encode.
`export_preview` (`routes/export.py`) holds the per-frame geometry — the output
rectangle mapped back to source pixels, the pre-fit stabilisation warp, the
target residuals. The socket needs both, and a second copy of either is a
future bug where the preview and the picture disagree.

**Work:** new `stable/server/frames.py` with three functions and no HTTP in it:

* `source_frame_jpeg(workspace, project_id, index, tier)` → `(bytes, rev)`.
  Reads the proxy JPEG **off disk as bytes**; it is already a JPEG, so nothing
  decodes or re-encodes on this path.
* `output_frame_jpeg(workspace, project_id, index, tier)` → `(bytes, rev)`.
  Exactly what `get_stabilized_frame` does today, moved wholesale.
* `frame_geometry(project, plan, options, index)` → the per-frame dict:
  `rect`, `warp`, `output`, `targets`. Exactly what `export_preview` computes
  today, moved wholesale.
* `plan_revision(...)` → a short string identifying the inputs a rendered output
  frame depends on (the same tuple `preview_plan` memoises on). It is what lets
  the client throw away stale output frames instead of showing them.

`get_stabilized_frame` and `export_preview` are rewritten to call these, so
their responses are unchanged by construction and the existing tests pin that.

**Owns:** `stable/server/frames.py` (new), `stable/server/routes/solve.py`,
`stable/server/routes/export.py`.

---

## W3 — `/ws/projects/{id}/frames`

**Found:** `routes/ws.py` already owns the project socket and its polling loop,
and `app.py` mounts that router without the `/api` prefix. A second socket
belongs beside the first.

**Protocol.** Client → server, JSON text:

```jsonc
{"type": "want",     "seq": 41, "frame": 247, "variants": ["source", "output"], "tier": "half"}
{"type": "prefetch", "frames": [248, 249, 250], "variants": ["source"], "tier": "half"}
```

Server → client, **binary**, so the JPEG is not base64'd into a text frame:

```
[uint32 big-endian header length][header, UTF-8 JSON][JPEG bytes]
```

with the header carrying `frame`, `variant`, `seq`, `rev`, and — for the frame
the client actually wants — `geometry`, the W2 per-frame block. Failures come
back as JSON text (`{"type": "error", "frame": …, "variant": …, "detail": …}`)
rather than silence, so "no solve yet" reads as a state and not as a hang.

**Scheduling.** One reader task and one worker task per connection. The reader
keeps **only the newest `want` per variant** — an older one is dropped
unanswered, which is the whole point — and a bounded (32) prefetch queue. The
worker serves wants first, prefetch only when there is no want outstanding, and
does the blocking JPEG/warp work in `asyncio.to_thread` one at a time, so a
scrub cannot pile up threads.

**Owns:** `stable/server/routes/ws.py`.

---

## W4 — the client stream

**Work:** `web/src/api/frameStream.ts` — a small non-React class per project:

* opens the socket, reconnects with the same 1 s retry `useJobsSocket` uses;
* `request(frame, variants)` sends one `want`, `prefetch(frames, variants)`
  sends one `prefetch`;
* decodes payloads with `createImageBitmap` (off the main thread; `.close()` on
  eviction) and caches by `variant:frame:rev`, capped and insertion-order
  evicted exactly as the current hook is;
* **drops the whole cache for a variant when its `rev` changes**, which is how
  an options change stops showing yesterday's warp;
* **falls back to HTTP** whenever the socket is not `OPEN` — same `Image()` load
  against the same REST URLs as today. A broken socket degrades to the current
  behaviour rather than to a black canvas.

`web/src/state/FrameStreamContext.tsx` wraps it for React, between
`ProjectProvider` and `ExportProvider` in `App.tsx`. It publishes one atomic
value:

```ts
{ frame, images: { source?, output? }, geometry }   // all of one frame, or the previous one
```

A frame is only published when **every requested variant for it** has arrived,
which is what keeps the left canvas, the right canvas and the overlay on the
same frame. It also drives the prefetch (same ±12/4 direction-of-travel logic,
moved), exposes `setWantOutput` for the export panel and `invalidateOutput()`
for `ExportContext` to call after an options change.

`hooks/useFrameImages.ts` is deleted; its two hard-won lessons (never blank the
canvas, prefetch in the direction of travel) move into the new module's
docstring and code.

**Owns:** `web/src/api/frameStream.ts` (new),
`web/src/state/FrameStreamContext.tsx` (new), `web/src/App.tsx`,
`web/src/components/Viewer.tsx`; deletes `web/src/hooks/useFrameImages.ts`.

---

## W5 — the side-by-side output canvas

> "a second canvas right of the main canvas … show what the exported video will
> look like after applying the current settings. Keep the main canvas on the
> left with the frame overlay exactly as we have now."

**Work:** the viewer's canvas area becomes a flex row. The left canvas is
untouched — same overlay, same zoom, same pan, same tools. The right canvas
appears only while the Export tab is open, and draws the `output` variant for
`viewFrame`, letterboxed to the **output** aspect ratio (which is not the
source's when the crop is not 1:1), with no overlay on it at all: it is the
file, not a workspace.

It is deliberately **not** zoom/pan-linked to the left canvas. The two are in
different coordinate spaces once the render crops and zooms, so a shared
transform would be a lie; the right one is always "fit", which is what
"what will the exported video look like" means.

A `Side-by-side output` checkbox in `ExportPanel` (next to the thirds-guides
one, state in `ExportContext` beside `showGuides`) turns it off for operators
who want the full width for composing. Default on — it is the feature.

States it must show rather than hide: no solve/smoothing yet (the `output`
variant 409s — say so), and the transparent border policy (JPEG has no alpha,
so the preview shows the black the render composites against; the panel already
warns that no codec here carries alpha).

**Owns:** `web/src/components/Viewer.tsx`,
`web/src/components/ExportPanel.tsx`, `web/src/state/ExportContext.tsx`,
`web/src/styles.css`.

---

## W6 — `/export/preview` stops being a per-frame call

**Found:** see above — `reloadPreview` closes over `currentFrame`, so the
planning call is made once per frame of every scrub, and its per-frame answers
(`warp`, `rect`, `targets`) race each other.

**Work:** the preview keeps only what is true for the whole clip — output size,
coverage, `crop_scale`, `zoom`, `gaps`, `max_displacement`, `is_no_op`,
`frames`, and the stored target. It is re-fetched when the *options*, the
smoothing or the edits change, and not when the playhead moves.

Everything per-frame comes from the stream's `geometry`, i.e. from the same
message as the pixels:

* `Viewer` draws `rect`, the thirds and the crosshairs from it, and drags
  targets through its `warp`;
* `ExportPanel` reads the residuals and `addTarget`'s warp from it.

`ExportContext` exposes `frameGeometry` (stream first, last preview as the
fallback for the moment before the first frame message) so neither consumer has
to know where it came from. The REST route is unchanged — it is still the
fallback, and still what the tests assert.

Net effect on the pasted log: one `/export/preview` per settings change instead
of one per frame, and zero `/frames/…` GETs while the socket is up.

**Owns:** `web/src/state/ExportContext.tsx`,
`web/src/components/ExportPanel.tsx`, `web/src/components/Viewer.tsx`.

---

## W7 — tests

New `tests/server/test_frame_stream.py`, over the existing solved-project
fixtures:

* a `want` for the `source` variant answers with a binary message whose header
  names that frame and whose body is byte-identical to `GET /frames/{n}`;
* a `want` for `output` is byte-identical to `GET /frames/{n}/stabilized` —
  the two paths cannot drift because W2 gave them one implementation;
* the header's `geometry` equals `GET /export/preview?frame=n`'s `rect`,
  `warp` and `targets` — the *sync* guarantee, asserted rather than asserted-to;
* a burst of wants is answered for the newest frame (superseded ones are
  dropped, not queued);
* an unsolved project gets an `error` message for `output`, not a dropped
  socket, while `source` still works;
* `rev` changes when the render options change, and is stable when they do not.

Extended `tests/server/test_export_screen.py`: the preview's whole-clip fields
are unaffected by `frame=`, which is what lets the client stop asking per frame.

---

## Checks

Between items: `pytest -m "not integration" -q`, `ruff check .`,
`ruff format --check .`, `mypy stable`, `cd web && npm run build`.
At the end: `python scripts/build_frontend.py`, so the packaged snapshot
`tests/server/test_web_dist.py` guards is not left stale.
