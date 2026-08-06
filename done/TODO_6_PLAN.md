# TODO 6 — One concept: keyframes. Auto-updating tracks.

## The problem (todo_6)

The user sees **two concepts that are really one**: "anchor points" (pinned
`PointSpec.queries`, placed with `A` or the add-anchor tool) and "keyframes"
(the ground-truth frames promoted from manual corrections, shown with a `✎`
prefix). Both are "the operator says the point is truly HERE at frame F".
Nobody understands why there are two names, two chip styles, two delete paths.

The second gap: **nothing updates by itself.** Add/move/delete a pin and the
path on screen is stale until the user finds and presses Track. todo_6 asks
for a "magic" feel: push a few keyframes around, the path follows.

## What already exists (do NOT rebuild)

The engine already does everything todo_6 asks for mechanically:

* **Bidirectional tracking per keyframe** — `plan_anchor_runs`
  (stable/trackers/merge.py) plans fwd+bwd runs from every anchor; between
  two adjacent anchors exactly two runs overlap.
* **Reconciliation** — `merge_anchor_runs` blends overlapping fwd/bwd runs
  by confidence with distance-to-anchor falloff; **disagreement between the
  two directions lowers merged confidence** — that dip is already the
  "flag it to the user" signal. Exact passthrough at every anchor.
* **Discrepancy flags** — `enforce_anchor_agreement` stops a run at a
  keyframe it disagrees with and reports `AnchorViolation` (frame, px);
  the server persists `anchor_violations` + `flagged_anchors` in the track
  sidecar and the timeline draws them red.
* **Unification primitive** — `effective_anchors(point)`
  (stable/server/tracking.py) = pins ∪ promoted corrections, correction
  wins on frame collision. This IS the unified keyframe list already.
* **Live streaming** — `LivePartialTrack` publishes merged positions
  per-run over the jobs WebSocket; the viewer already draws partial paths.
* **Job queue** — `JobManager` with `cancel()` for queued jobs.

So this is a **unification + automation + presentation** job, not an engine
job. The corrections layer (raw-is-immutable, undo = drop a correction)
stays exactly as it is — it is a design invariant.

## Target UX (definition of done)

1. There is exactly one word in the product: **keyframe**. One diamond marker
   on the canvas, one tick style on the timeline, one chip in the point card,
   one delete gesture. "Anchor" does not appear anywhere a user can see.
2. Set a keyframe at any time: `A` pins the tracked position on this frame,
   click/drag on the frame states a position (as today — no mode). Drag an
   existing keyframe diamond to move it. Delete from chip ✕, patch panel, or
   Alt-click. Deleting the last keyframe of a point is refused (a point must
   have one truth position).
3. **Auto-track (default ON, toggleable):** any keyframe add / move / delete
   automatically re-tracks the point, debounced, superseding any queued run
   for the same point. The path visibly re-flows via the existing live
   partial stream. No Track button hunt. (The Track button stays for
   full manual control and for auto-track = off.)
4. Where fwd/bwd runs disagreed, or a run was stopped at a keyframe it
   disagreed with, the timeline shows it (existing red ticks + confidence
   heat) and the point card says what to do ("runs disagree here — drag the
   point or add a keyframe"). ↑/↓ jumps through the unified keyframe list
   (it already walks `ground_truth`; keep, rename).

## Workstreams

### W1 — Server: unified keyframe API + auto-track (Opus)

`stable/server/routes/points.py`, `edits.py`, `schemas.py`, `tracking.py`,
`jobs.py`.

1. **Unified keyframe routes** (replace the `/anchors` pair — this app ships
   frontend+server together; no external API consumers; delete, don't alias):
   * `GET  /points/{id}/keyframes` — the `effective_anchors` list, each with
     `{frame, x, y, source: "pin" | "correction", correction_id?}`.
   * `POST /points/{id}/keyframes` `{frame, x, y}` — states a truth position.
     Replaces any pin on that frame; if a ground-truth correction sits on that
     frame, update/replace it instead of shadowing it (one frame, one truth).
   * `DELETE /points/{id}/keyframes/{frame}` — removes whatever backs the
     keyframe on that frame: the pin, and/or deletes the promoting
     correction (and its retrack sidecar, reusing the existing correction
     delete path). Refuse to delete the last remaining keyframe.
   * `PATCH /points/{id}/keyframes/{frame}` `{frame?, x?, y?}` — move in one
     atomic mutation (the UI's drag), not delete-then-add over two requests.
2. **Auto-track:** each mutating keyframe route takes `"track": "auto" | "none"`
   (default `"auto"`). On `"auto"` it submits the existing full multi-anchor
   track job for the point **after first cancelling any queued track job for
   the same point** (supersede — add a small helper on `JobManager` or tag
   jobs with `point_id` in their label and use the existing `cancel()`).
   Response gains `{"job_id": ...}` when a job was submitted. Full-point
   retrack is the correct grain: merge must re-run globally anyway and cost
   is capped at ~2× clip regardless of keyframe count.
3. **Track payload:** `GET .../track` already returns `ground_truth`,
   `anchor_violations`, `flagged_anchors`. Rename payload keys to
   `keyframes`, `keyframe_violations`, `flagged_keyframes` (frontend is
   updated in W2; grep for every consumer incl. tests). Keep the project-file
   schema (`queries`, corrections) **unchanged** — no migration, v1 files
   keep loading.
4. Update server tests: `tests/server/test_points.py`,
   `test_anchor_integrity.py`, `test_patch_serving.py`,
   `test_session_persistence.py`, `test_point_deletion.py`, plus new tests:
   keyframe PATCH move, delete-removes-backing-correction, auto-track
   supersedes queued job, `track:"none"` submits nothing.

### W2 — Frontend: one concept, magic feel (Opus)

`web/src/state/ProjectContext.tsx`, `api/client.ts`, `types.ts`,
`components/{Viewer,Timeline,TracksPanel,PatchPanel,TrackEdits}.tsx`.

1. **Terminology sweep:** "Anchor" → "Keyframe" in every label, tooltip,
   overlay name (`OVERLAY_LABELS`: "Anchors" → "Keyframes"), CSS class,
   function name (`addAnchorHere` → `addKeyframeHere`, `anchorHitTest` →
   `keyframeHitTest`, `AnchorDrag` → `KeyframeDrag`), and the API client
   (`addAnchor`/`deleteAnchor` → keyframe calls on the new W1 routes,
   plus the new `moveKeyframe` PATCH).
2. **One chip, one marker:** the point card renders the unified
   `GET /keyframes` list — no `✎` prefix, no pinned-vs-promoted split;
   every chip is deletable (server decides what that means). Patch panel
   `kind: "keyframe"` likewise. Diamond markers on canvas are draggable →
   PATCH move. Remove the `add-anchor` tool mode if the modeless gesture
   covers it (keep `A` / `⇧A` shortcuts, retitled).
3. **Auto-track:** a project-level toggle (default ON, persisted in UI
   state). Keyframe mutations pass `track:"auto"`; debounce rapid drags
   (fire PATCH on drag end, not per mousemove). Subscribe to the returned
   job over the existing jobs WebSocket so the live partial path re-flows.
   While an auto job is superseded/re-queued, don't flash errors.
4. **Discrepancy UX:** keep red flagged ticks; when `keyframe_violations`
   is non-empty, the point card shows one actionable line ("tracking
   disagrees with your keyframe at frame N by M px — adjust or add a
   keyframe nearby") that clicks through to the frame.
5. ↑/↓ navigation uses the unified keyframe list (rename the
   `ground_truth` plumbing to `keyframes`).

### W3 — Legacy cleanup, docs, green build (Sonnet)

1. **Dead code sweep:** no orphaned `add-anchor` tool code, no unused
   `AnchorCreate` schema, no `/anchors` route remnants, no stale CSS
   classes, no unused exports in `schemas.py` / `client.ts` / `types.ts`.
   `grep -i anchor` across `web/src` and `stable/server` should return only
   (a) internal engine terms in `stable/trackers/merge.py` et al (fine —
   `plan_anchor_runs` is engine vocabulary, not UX) and (b) the solver's
   `anchor_frame` render-target concept, which is unrelated to this todo.
2. **Docs:** README (UI walkthrough paragraph, design-invariants
   "Multi-anchor by default" bullet reworded to keyframes-as-UX /
   anchors-as-engine), PLAN.md note, `web/README.md`.
3. **Everything green:** `python -m pytest`, `python -m ruff check .`,
   `python -m mypy`, `cd web && npm run build`. Regenerate
   `stable/server/webdist` via `python scripts/build_frontend.py` if that is
   the established flow (check first) so `stable serve` serves the new UI.

## Sequencing

W1 → W2 → W3, strictly. W1 and W2 each leave the repo compiling and tests
green for their half; W3 is the janitor and auditor.

## Non-goals

* No engine/merge math changes. No project-file schema bump.
* No renaming of `stable.trackers.merge` internals (`plan_anchor_runs` etc.)
  — that is engine vocabulary with heavy test coverage, invisible to users.
* CLI flags (`--point x,y@frame`) unchanged.
* todo_7 ("refresh all tracks") is out of scope, but W1's auto-track
  endpoint shape should not preclude it.
