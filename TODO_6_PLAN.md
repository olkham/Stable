# Implementation plan — seventh feedback round

Source: `todo_6` (one item, three sentences of consequence). Same conventions as
`done/TODO_PLAN.md`, `done/NEW_TODO_PLAN.md`, `done/TODO_2_PLAN.md`,
`done/TODO_3_PLAN.md`, `TODOS_4_PLAN.md` and `TODOS_5_PLAN.md`: every work item
states what was **found in the code**, not what was assumed, and the suite stays
green throughout (`pytest -m "not integration"`, `ruff check`,
`ruff format --check`, `mypy stable`, `npm run build`, then
`python scripts/build_frontend.py`).

Baseline, measured rather than assumed: **2 failures** on this machine, both in
`tests/test_phase3_acceptance.py` (`test_the_solve_recovers_the_ground_truth_motion`
and `test_the_recovered_curves_match_the_known_channels`), and both numerical —
the solved rotation is `2.079e-4` against a `2e-4` tolerance. Nothing in this
round touches the solver, so that is the number to come back to. (The seven
"PyTorch is not installed" failures the last round recorded are gone: torch is
installed in this checkout's `.venv` now.)

---

## What the item actually is

> "We have the concept of keyframes and anchor points, but the user doesn't
> understand what they are or how to use them. They are essentially the same
> thing."

They are not *essentially* the same thing — they are **literally** the same
thing, and have been since Phase 0. There is exactly one type,
`stable.core.types.Query` ("at frame `f` this point is truly at `(x, y)`"), one
list of them per point (`PointSpec.queries`), and one function that answers
"what has the operator stated about this point" (`tracking.effective_anchors`).

What the operator sees is that one thing under **two names, four glyphs and two
lists**:

| Where | Says |
|---|---|
| `TracksPanel` chips | "Anchor keyframe" / "Keyframe promoted from a manual correction", `✎f12` |
| `TracksPanel` "+ anchor" button | "Pin this point at frame N" |
| `Viewer` tool palette | "Add keyframe: pin the selected track where you click" |
| `Viewer` hint line | "Select a track first — a keyframe belongs to one" |
| `Viewer` shortcut help | "↑/↓ keyframe … A anchor here" |
| `PatchPanel` | "Keyframes — click one, then Delete, or use its ×" (a **second** list of the same frames, immediately under the first) |
| Run limit | "to next keyframe" |
| HTTP | `stop_at_keyframe`, `kind: "keyframe"`, `ground_truth`, `anchor_frames` |

So the first half of the fix is a vocabulary purge, and it is not cosmetic: two
words for one concept is the entire reported bug.

The second half is that behind the two words sit **three real gaps**, all of
which are the parts of the sentence "set an anchor at any point in time, then
track forwards or backwards from it, and delete it and have the tracking
update":

1. **You cannot track forwards or backwards from an anchor on a fresh point.**
   The directional run controls in `TracksPanel` are behind
   `point.has_track` (`TracksPanel.tsx:413`), and they have to be: the server's
   bounded run lands as a `RetrackCorrection`, and `load_display_track`
   (`tracking.py:690`) returns `None` when `points[].raw` is unset, so a run on
   an untracked point computes a track that nothing can display. A brand-new
   anchor's only option is "Track", which is every anchor, both directions,
   the whole working range.
2. **Half of the anchors cannot be deleted.** `effective_anchors` is pins *plus*
   the positions manual corrections were promoted to (`corrections.py:348`).
   The UI lists both, and `DELETE .../anchors/{frame}` (`points.py:222`) only
   filters `queries` — so a promoted anchor's `×` is disabled with "delete the
   correction on the Edit tab". Worse, `RetrackCorrection` is in
   `GROUND_TRUTH_KINDS`: **every directional run creates a second anchor at its
   own start frame**, so tracking from an anchor silently makes an undeletable
   twin of it.
3. **Deleting an anchor changes nothing.** The merged raw track already passes
   exactly through the deleted anchor's position and keeps doing so until a
   full re-track is run, and nothing anywhere says the track is now out of date
   with its anchors.

---

## Work items

| # | Item | Owns | Depends on |
|---|---|---|---|
| W1 | One word: *anchor* | every user-facing string, `schemas.py`, `tracking.py`, docstrings | — |
| W2 | One list: every anchor is deletable, and deleting one removes what it anchored | `corrections.py`, `routes/points.py`, `history.py` | — |
| W3 | Track forwards / backwards **from an anchor**, on any point | `tracking.py`, `serialize.py`, `TracksPanel` | W1 |
| W4 | An anchor change the tracking cannot absorb says so | `tracking.py`, `serialize.py`, `TracksPanel` | W3 |
| W5 | The second anchor list goes away | `PatchPanel`, `styles.css` | W1 |
| W6 | Tests and docs | `tests/`, `README.md` | W1–W5 |

Deliberately **not** in scope, and why:

* **Renaming `queries` in the project file.** It is schema v1 and locked; a
  migration to rename a key nothing user-facing reads would be churn with a
  compatibility risk and no UX gain. `Query`/`queries` stay as the storage name,
  and `stable.core.types.Query`'s docstring says the UI word for it is "anchor".
* **Auto-re-tracking after an anchor is deleted.** A full track is minutes of
  GPU on a real clip. Deleting an anchor must not start one behind the
  operator's back; W4 makes the staleness visible and one click away instead.

---

## W1 — one word: *anchor*

**Found.** 311 occurrences of "keyframe" across 52 files. They fall into three
groups, and only the first is wrong:

* **The UI's second name for an anchor** — every row of the table above. Goes.
* **After Effects keyframe data** (`stable/core/trackdata.py`, the export
  button, `AE_FORMAT`). That is After Effects' own term for its own format and
  is correct usage. Stays, untouched.
* **The curve editor's pinned keyframes** (`smoothing.ts`, `CurveEditor.tsx`) —
  a *curve* keyframe on the X/Y/R/S channels, which is a different object from a
  point anchor and is not what the report is about. Stays.

**Change.**

* Frontend strings, labels, tooltips, hints and the shortcut help line in
  `Viewer`, `TracksPanel`, `PatchPanel`, `Timeline`, `ProjectContext`,
  `types.ts`, `client.ts`. The tool palette's `◆` becomes "Add anchor", the run
  limit becomes "runs stop at the next anchor", `↑`/`↓` are documented as
  "next/previous anchor".
* HTTP: `stop_at_keyframe` → `stop_at_anchor` on `RetrackRequest`, with the old
  name accepted as a deprecated alias (`AliasChoices`) so a stale browser tab or
  a script does not start failing; `PointPatch.kind` `"keyframe"` → `"anchor"`;
  and the track payload's `ground_truth` → `anchors`, which was a *fourth* name
  for the same list. That last one also moves onto the shared `TrackData` type —
  three components had re-declared it locally, which is how one list came to
  look like two.
* Python: `bounded_frame_range(stop_at_keyframe=…)` → `stop_at_anchor`, and the
  docstrings that describe the concept in `types.py`, `corrections.py`,
  `merge.py`, `project.py`, `tracking.py`, `routes/points.py`, `routes/edits.py`.
* `README.md`'s viewer paragraph (which now says what an anchor *is*, since
  "the user doesn't understand what they are" is half the report) and `PLAN.md`'s
  multi-anchor and Phase-4 sections.

**Why it is worth touching the API name.** The UI took its second word *from*
the code. Leaving `stop_at_keyframe` on the wire while the panel above it says
"to next anchor" is exactly how the two vocabularies grew apart the first time.

---

## W2 — one list: every anchor is deletable

**Found.** `points.py:233` — `_mutate` rebuilds `spec.queries` and nothing else,
so an anchor that came from a correction survives its own deletion. `PatchPanel`
and `TracksPanel` both disable the `×` for those and explain the workaround.

Found while doing it: `PatchPanel` gates that `×` on `patch.pinned`, and **the
server never sends a `pinned` field at all** (`points.py`'s `entry()` emits
`frame`, `x`, `y`, `kind`, `available`). So it read `undefined`, and *every*
anchor in that grid was undeletable with the message "this comes from a manual
correction". The field goes rather than gets implemented: with every anchor
deletable there is nothing left for it to say.

**Change.**

* New `stable.core.corrections.corrections_anchored_at(corrections, frame)`:
  the enabled corrections that put an anchor on `frame` — `replace` and
  `retrack` (which state a position, i.e. `GROUND_TRUTH_KINDS`) plus an
  `offset` keyframed there, because a drag arrives as `offset` + companion
  `replace` and half a drag is not a state the operator asked for.
* `DELETE .../anchors/{frame}` removes the pin **and** those corrections, and
  refuses only when the frame has no anchor at all or when it is the point's
  last one. So deleting a directional run's anchor deletes the run it produced —
  which is "the tracking should update accordingly", exactly, for every span
  that lives in the corrections layer.
* The deletion goes through `EditHistory`, and `Snapshot` grows a second key per
  point so an undo restores `queries` as well as `corrections`. Old history
  files (a bare list per point) still load — they are read as corrections-only.
  Points absent from a snapshot keep their anchors rather than being emptied: a
  point created after the snapshot was taken must not undo into an anchorless
  one, which `Project.validate` would not catch and the tracker could not run.
* Frontend: one chip per anchor, all identical, all with a working `×`. How an
  anchor came to exist survives only in its tooltip.

---

## W3 — track forwards or backwards from an anchor

**Found.** `run_point_retrack` already does the right thing for a *tracked*
point: bounded, guarded, resolving its start position from the anchors first
(`_resolve_start_position`, `tracking.py:615`). Two things stop it being the
"track from this anchor" gesture the report describes:

* with `points[].raw` unset the correction it appends has no baseline to splice
  onto, so `GET .../track` 404s and the run is invisible;
* `_load_track_stats` (`serialize.py:29`) reads the **raw** npz, so a card whose
  whole track came from corrections reports "0 frames · 0% confidence".

**Change.**

* `run_point_retrack` seeds an empty clip-length `PointTrack` as the point's raw
  track when it has none, in the same read-modify-write that appends the
  correction. An all-`NaN` baseline plus a run that covered frames 100–160 is
  precisely "I have tracked frames 100 to 160 from my anchor at 100".
* `_load_track_stats` reports the **displayed** track (raw + corrections), which
  is what the canvas draws, what the solver consumes and what the export writes.
  `worst_frames` still comes from the merge report, since that is a diagnostic
  of the run and not of the corrected result.
* `TracksPanel`: each anchor chip gets `◀` and `▶` — track backwards/forwards
  from *that anchor* — next to its `×`, with the existing bound selector
  ("to next anchor" by default, "for N frames", "to end of clip") moved up
  beside the anchor row it now governs. The from-the-playhead pair stays for
  "carry on from where I am", and neither is gated on `has_track` any more.

---

## W4 — an anchor change the tracking cannot absorb says so

**Found.** Nothing compares a point's anchors against the anchors its stored
track was actually computed from. `_save_report` writes `anchor_frames`, which
is close but records only frames — a pin *moved* on the same frame leaves the
frame list identical while the merged track still runs through the old position.

**Change.**

* `_save_report` also records the anchors, with positions — the point's *own*,
  not the effective set the run planned through. The effective set includes the
  promoted ones, so recording it would make every point with a correction claim
  to need a re-track the instant it finished one.
* `point_summary` gains `anchors_changed`: true when the point has a track and
  its own anchors differ from the recorded ones. A report written before this
  round records nothing to compare, and is read as "cannot tell" rather than as
  a "yes" — a sidecar that cannot answer must not nag.
* `TracksPanel` says "anchors changed — re-track" on the card, next to the
  button that does it. Which also closes the everyday version of this: place an
  anchor, watch nothing happen, wonder what an anchor is for.

---

## W5 — the second anchor list goes away

**Found.** `PatchPanel` draws its own list of the same frames directly under the
card's anchor chips: a thumbnail grid labelled "Keyframes" when the tracker has
a window, and a bare chip list — with its own `×` — when it does not
(`PatchPanel.tsx:352`). For a deep tracker the panel is therefore *only* a
duplicate of the row above it.

**Change.** The thumbnails stay (they show pixels, which the chips cannot) and
are labelled "Anchors". The window-less branch drops its chip list entirely and
keeps the explanation of why there is no patch to show. One list of anchors per
card, and it is the one with the run buttons on it.

---

## W6 — tests and docs

New, in `tests/server/test_anchors.py` (16 tests):

* a fresh anchor tracks forwards on a point that has never been tracked, and
  only the frames the run covered come back;
* a run from an anchor stops at the next one, under the new name **and** under
  the deprecated `stop_at_keyframe`;
* the point then reports the frames it actually covered, not the empty baseline;
* deleting an anchor removes the run that started from it, and those frames go
  back to having no position;
* a drag-promoted anchor can be deleted at all, and takes the drag with it —
  asserted by the track returning to its *raw* value at that frame;
* undo brings a deleted anchor back, and brings its run back with it;
* the last anchor cannot be deleted; deleting one that is not there says so;
* `anchors_changed` is false after a track, true after adding, moving or
  deleting an anchor, false again after re-tracking, false for a correction, and
  false for a track run with corrections already in place.

Updated: `tests/test_anchor_promotion.py`, `tests/server/test_anchor_integrity.py`,
`tests/server/test_point_patches.py` and `tests/server/test_patch_serving.py` for
the renamed argument, the renamed patch `kind` and the renamed payload key.

Docs: `README.md`'s viewer paragraph, and `PLAN.md` where it describes anchors
as keyframes.

---

## Result

`pytest -m "not integration"`: the 2 baseline failures above, nothing else.
`ruff check` clean, `mypy stable` clean (88 files), `npm run build` clean, and
the touched files are `ruff format` clean (the repo has 35 other files that are
not, all of them pre-existing and untouched here).
