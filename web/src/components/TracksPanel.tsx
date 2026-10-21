import { useState } from "react";
import * as api from "../api/client";
import type { TracksPanel as PanelVariant } from "../flags";
import { useProjectContext } from "../state/ProjectContext";
import type { Query, TrackerInfo } from "../types";
import { CompositeSources, DeriveTrackPanel } from "./DeriveTrack";
import { JobProgress } from "./JobProgress";
import { NewTrack } from "./NewTrack";
import { PanelSwitch } from "./PanelSwitch";
import { AllPatchesStrip, PatchPanel } from "./PatchPanel";
import { TrackEdits } from "./TrackEdits";
import { TrackEye, TrackSwatch } from "./TrackVisibility";
import { OVERLAY_LABELS } from "./Viewer";

/** Colour of an anchor no tracking run could reach — the "go and look" mark. */
const FLAGGED_ANCHOR_COLOR = "#ff3b30";

/**
 * Why the tracker a point is *already set to* cannot run here, if it cannot.
 *
 * A point can carry a tracker that has since become unavailable — a project
 * made on a machine with PyTorch, opened on one without. The dropdown greys
 * that option out, which is silent about the point that is using it, so the
 * row says so too.
 */
function unavailableReason(trackers: TrackerInfo[], name: string): string | null {
  const match = trackers.find((t) => t.name === name);
  if (!match || match.available) return null;
  return match.unavailable_reason ?? `${name} is not available in this environment`;
}

/** How far a run started from an anchor should reach (todos 12 and 20). */
type RunLimit = "anchor" | "clip" | "frames";

/**
 * What an anchor is, in one sentence, on the row that lists them.
 *
 * The report this round answers is "the user doesn't understand what they are
 * or how to use them", and the answer is not only to stop calling them two
 * things — it is to say what they do, where they are.
 */
const ANCHOR_HELP =
  "An anchor says where this point really is on one frame. Tracking runs out from " +
  "the anchors, so ◀ / ▶ track from one, and deleting one removes what was tracked from it.";

/** How far a run will go, for the tooltips on the buttons that start one. */
function runLabel(limit: RunLimit, frames: number): string {
  if (limit === "anchor") return "as far as the next anchor";
  if (limit === "frames") return `${frames} frames`;
  return "to the end of the clip";
}

/**
 * One card per track: what it is, what it sees, what it drew, and what it shows.
 *
 * This was a list of *points*, which is how the data model thinks but not how
 * the work does. The unit an operator handles is a track — its tracker, its
 * anchors, its patches, its runs, and (todo 9) which of its overlays are on the
 * canvas. Putting the visibility toggles on the card is the whole point of the
 * change: with several tracks up, "hide the dotted trails" almost always means
 * *that* track's trails, and the global checkboxes above the canvas could only
 * ever mean all of them.
 *
 * **Anchors are the whole model** (todo 6, seventh round). An anchor is one
 * statement of where the feature is on one frame; every anchor is listed here
 * once, tracks forwards or backwards from itself, and can be deleted — taking
 * whatever it anchored with it. An anchor the tracker could not reconcile with
 * what it saw is listed in red here and on the timeline rather than being
 * quietly averaged away, and a run started from one is bounded — to the next
 * anchor, or to a number of frames — instead of always ploughing to the end of
 * the clip.
 */
export function TracksPanel({
  variant,
  onVariantChange,
}: {
  variant: PanelVariant;
  onVariantChange: (next: PanelVariant) => void;
}) {
  const {
    project,
    trackers,
    refresh,
    currentFrame,
    selectedPointId,
    setSelectedPointId,
    setCurrentFrame,
    tool,
    setTool,
    newTrack,
    startNewTrack,
    cancelNewTrack,
    addAnchorHere,
    deleteAnchorAt,
    runFrom,
    latestJobFor,
    tracks,
    invalidateTrack,
    overlays,
    overlaysFor,
    setTrackOverlay,
    trackOverrides,
    isTrackHidden,
  } = useProjectContext();

  const [runLimit, setRunLimit] = useState<RunLimit>("anchor");
  const [runFrames, setRunFrames] = useState(10);
  const [runError, setRunError] = useState<string | null>(null);
  // Which cards the operator has explicitly opened or closed (todo 5, fourth
  // round). Tri-state, like the overlay overrides below: a card nobody has
  // touched follows the old behaviour — open when its track is selected — and
  // one that *has* been touched keeps what it was told, whatever the selection
  // does next. That is the whole complaint: collapsing happened to you, and
  // two cards could never be open at once to compare them.
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  async function handleTrack(pointId: string) {
    await api.trackPoint(project.id, pointId, {});
    invalidateTrack(pointId);
  }

  async function handleTrackerChange(pointId: string, tracker: string) {
    await api.updatePoint(project.id, pointId, { tracker });
    await refresh();
  }

  async function handleDelete(pointId: string) {
    await api.deletePoint(project.id, pointId);
    // Drop the cached track with it. Without this the path stayed in state and
    // was drawn again the moment a new point took the same id — the client half
    // of "when I delete a point and add a new one back it restores the previous
    // track". The server sweeps its files; this sweeps ours.
    invalidateTrack(pointId);
    if (selectedPointId === pointId) setSelectedPointId(null);
    await refresh();
  }

  // Deleting an anchor and starting a run both go through the context, because
  // the timeline panel behind the flag does the same two things and a rule this
  // load-bearing must not have two implementations that can drift apart.
  async function handleDeleteAnchor(pointId: string, frame: number) {
    setRunError(await deleteAnchorAt(pointId, frame));
  }

  /**
   * Track one direction from `frame`, bounded by the panel's limit.
   *
   * No x/y is sent: the run starts from wherever the point already is on that
   * frame — the anchor there when there is one, which is what makes the chips'
   * ◀ / ▶ mean "track from *this* anchor", and the tracked position otherwise,
   * which is what "carry on from the playhead" means.
   */
  async function handleRun(pointId: string, frame: number, direction: "fwd" | "bwd") {
    setRunError(
      await runFrom(pointId, frame, direction, {
        ...(runLimit === "anchor" ? { stop_at_anchor: true } : {}),
        ...(runLimit === "frames" ? { max_frames: runFrames } : {}),
      }),
    );
  }

  return (
    <div className="panel points-panel">
      <div className="panel__header">
        <h3>Tracks</h3>
        <PanelSwitch variant={variant} onChange={onVariantChange} />
        <button
          className={newTrack ? "button--active" : ""}
          onClick={() => (newTrack ? cancelNewTrack() : startNewTrack())}
          title="Choose a tracker and its settings, then mark what it should follow on the video (N for a point, B for a box)"
        >
          + New track
        </button>
      </div>

      <NewTrack />

      {project.points.length === 0 && (
        <p className="panel__empty">
          No tracks yet. Click "New track", choose a tracker, then mark what it should follow on the
          video.
        </p>
      )}

      {/* Every point's patch on this frame, so "which one has drifted" is one
          glance rather than one click per point (todo 14). */}
      <AllPatchesStrip />

      <ul className="points-list">
        {project.points.map((point) => {
          const trackJob = latestJobFor("track", point.id);
          const retrackJob = latestJobFor("retrack", point.id);
          const isSelected = point.id === selectedPointId;
          const track = tracks[point.id];
          // The point's own anchors, plus the ones its corrections were
          // promoted to. Its own come from the project (refreshed the moment
          // one is placed); promotions only exist on the track, which is
          // fetched and cached separately — so reading the list off the track
          // alone left a just-placed anchor missing here until the next run
          // replaced that cache, even though the canvas drew it at once.
          const promoted = (track?.anchors ?? []).filter(
            (a) => !point.queries.some((q) => q.frame === a.frame),
          );
          const anchors: Query[] = [...point.queries, ...promoted].sort(
            (a, b) => a.frame - b.frame,
          );
          const flagged = new Set(track?.flagged_anchors ?? []);
          const violations = track?.anchor_violations ?? [];
          const isOpen = expanded[point.id] ?? isSelected;
          // A track averaged from other tracks (todo 2, tenth round, second
          // pass). No algorithm produced it, no window was matched, nothing was
          // anchored and nothing can be re-run — so the controls for all four
          // are dropped and what it is made of is shown instead.
          const composite = point.derived_from.length > 0;
          return (
            <li
              key={point.id}
              className={`points-list__item ${isSelected ? "points-list__item--selected" : ""} ${
                isOpen ? "" : "points-list__item--collapsed"
              }`}
              onClick={() => setSelectedPointId(point.id)}
            >
              <div className="points-list__row">
                <button
                  className="points-list__disclose"
                  aria-expanded={isOpen}
                  title={isOpen ? "Collapse this track" : "Expand this track"}
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpanded((prev) => ({ ...prev, [point.id]: !isOpen }));
                  }}
                >
                  {isOpen ? "▾" : "▸"}
                </button>
                {/* On the row rather than inside the card: hiding a track and
                    recolouring it are both one click, from wherever the track
                    is, open or shut (todos 6 and 7, ninth round). */}
                <TrackEye pointId={point.id} />
                <TrackSwatch pointId={point.id} className="points-list__swatch" />
                <span className="points-list__id">{point.id}</span>
                {!isOpen && point.has_track && (
                  <span className="points-list__confidence">
                    {point.tracked_frames}f · {(point.mean_confidence * 100).toFixed(0)}%
                  </span>
                )}
                {/* No algorithm produced a composite, so there is none to
                    choose — "composite" is the whole of what there is to say. */}
                {composite ? (
                  <span
                    className="points-list__composite"
                    style={isOpen ? undefined : { display: "none" }}
                    title="This path is the frame-by-frame average of other tracks, so no tracking algorithm applies to it"
                  >
                    composite
                  </span>
                ) : (
                  <>
                    <select
                      value={point.tracker}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => handleTrackerChange(point.id, e.target.value)}
                      style={isOpen ? undefined : { display: "none" }}
                    >
                      {/* A box track is initialised from a box, which only some
                          trackers take; the server refuses any other (400). */}
                      {trackers
                        .filter((t) => !point.box || t.tracks_boxes || t.name === point.tracker)
                        .map((t) => (
                        <option
                          key={t.name}
                          value={t.name}
                          disabled={!t.available}
                          // A disabled option with no explanation is
                          // indistinguishable from a broken one; the server says
                          // what to install, so pass it straight through.
                          title={t.unavailable_reason ?? undefined}
                        >
                          {t.display_name}
                          {!t.available ? " (unavailable)" : ""}
                        </option>
                      ))}
                    </select>
                    {point.box && (
                      <span
                        className="points-list__composite"
                        style={isOpen ? undefined : { display: "none" }}
                        title="Drawn as a bounding box: the position is the box centre, with a per-frame size"
                      >
                        box
                      </span>
                    )}
                    {unavailableReason(trackers, point.tracker) && (
                      <span
                        className="points-list__warning"
                        title={unavailableReason(trackers, point.tracker) ?? undefined}
                      >
                        ⚠ not installed
                      </span>
                    )}
                  </>
                )}
                {/* The "solve" checkbox used to be here (todo 6, fourth
                    round). Choosing which tracks go into the solve now happens
                    on the Solve tab, next to the button that runs it, so the
                    two halves of one decision are on one screen. */}
                <button
                  className="points-list__delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(point.id);
                  }}
                  title="Delete point"
                >
                  ×
                </button>
              </div>

              {/* The anchors, and everything an anchor can do: go there, track
                  forwards or backwards from it, or be deleted — which takes the
                  run it started with it. One row, one word, one × per anchor
                  whether the operator placed it or a drag promoted it (todo 6,
                  seventh round). */}
              {/* What a composite is made of, in place of the anchors it does
                  not have. Its single anchor exists because every point needs
                  one, not because anything was anchored. */}
              {isOpen && composite && (
                <div onClick={(e) => e.stopPropagation()}>
                  <CompositeSources point={point} />
                </div>
              )}

              {isOpen && !composite && (
              <div className="points-list__anchors" onClick={(e) => e.stopPropagation()}>
                <span className="points-list__label" title={ANCHOR_HELP}>
                  Anchors
                </span>
                {anchors.map((q) => {
                  const missed = flagged.has(q.frame);
                  const own = point.queries.some((p) => p.frame === q.frame);
                  const last = anchors.length <= 1;
                  return (
                    <span
                      key={q.frame}
                      className={`anchor-chip ${q.frame === currentFrame ? "anchor-chip--here" : ""}`}
                      style={missed ? { borderColor: FLAGGED_ANCHOR_COLOR } : undefined}
                    >
                      <button
                        className="link-button"
                        style={missed ? { color: FLAGGED_ANCHOR_COLOR } : undefined}
                        title={
                          missed
                            ? "No tracking run reached this anchor — needs manual review. Click to go there."
                            : own
                              ? "You put this anchor here. Click to go to it."
                              : "Anchor from an edit you made on this frame. Click to go to it."
                        }
                        onClick={() => setCurrentFrame(q.frame)}
                      >
                        f{q.frame}
                      </button>
                      <button
                        className="anchor-chip__run"
                        title={`Track backwards from this anchor (${runLabel(runLimit, runFrames)})`}
                        onClick={() => handleRun(point.id, q.frame, "bwd")}
                      >
                        ◀
                      </button>
                      <button
                        className="anchor-chip__run"
                        title={`Track forwards from this anchor (${runLabel(runLimit, runFrames)})`}
                        onClick={() => handleRun(point.id, q.frame, "fwd")}
                      >
                        ▶
                      </button>
                      <button
                        className="anchor-chip__delete"
                        disabled={last}
                        title={
                          last
                            ? "A track needs at least one anchor to track from — this is its last one"
                            : "Delete this anchor, and whatever was tracked from it"
                        }
                        onClick={() => handleDeleteAnchor(point.id, q.frame)}
                      >
                        ×
                      </button>
                    </span>
                  );
                })}
                <button
                  className={`anchor-add ${
                    tool === "add-anchor" && isSelected ? "button--active" : ""
                  }`}
                  title={`Anchor this track at frame ${currentFrame} where it already is (shortcut: A), or click the frame to place it somewhere else (⇧A)`}
                  onClick={async () => {
                    setSelectedPointId(point.id);
                    if (tool === "add-anchor" && isSelected) {
                      setTool("move");
                      return;
                    }
                    // Same rule as the shortcut: anchor where the point already
                    // is, and only ask for a click when there is nothing there
                    // to anchor.
                    if (!(await addAnchorHere(point.id))) setTool("add-anchor");
                  }}
                >
                  + anchor at f{currentFrame}
                </button>
                {/* How far every ◀ / ▶ on this panel reaches. Beside the
                    buttons it governs rather than in a settings panel: it is
                    read once per run and changed almost as often. */}
                <select
                  value={runLimit}
                  onChange={(e) => setRunLimit(e.target.value as RunLimit)}
                  title="How far a run from an anchor should go"
                >
                  <option value="anchor">runs stop at the next anchor</option>
                  <option value="frames">runs cover N frames</option>
                  <option value="clip">runs go to the end of the clip</option>
                </select>
                {runLimit === "frames" && (
                  <input
                    type="number"
                    min={1}
                    max={Math.max(project.video.frames, 1)}
                    value={runFrames}
                    style={{ width: "4rem" }}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isNaN(v)) setRunFrames(Math.max(1, Math.round(v)));
                    }}
                  />
                )}
              </div>
              )}

              {/* What this track draws on the canvas (todo 9). The row above
                  the canvas sets the defaults; these override them for this
                  track alone, which is what makes a busy shot readable. A
                  toggle showing the default is not "off", it is "same as
                  everyone" — hence the tri-state title. */}
              {isOpen && (
              <div className="track-overlays" onClick={(e) => e.stopPropagation()}>
                {OVERLAY_LABELS.map(([key, label, help]) => {
                  const override = trackOverrides[point.id]?.[key];
                  const active = overlaysFor(point.id)[key];
                  // Nothing this row says can change what is drawn while the
                  // eye is off, so it says nothing rather than lying (the same
                  // treatment the master Overlays switch gets above the canvas).
                  const hidden = isTrackHidden(point.id);
                  return (
                    <button
                      key={key}
                      disabled={hidden}
                      className={`track-overlay ${active ? "track-overlay--on" : ""} ${
                        override === undefined ? "" : "track-overlay--set"
                      }`}
                      title={`${help}
${
                        hidden
                          ? "This track is hidden — click the eye on its row to show it"
                          : override === undefined
                            ? `Following the default (${overlays[key] ? "shown" : "hidden"}) — click to override`
                            : "Overridden for this track — click again to follow the default"
                      }`}
                      onClick={() =>
                        setTrackOverlay(point.id, key, override === undefined ? !overlays[key] : null)
                      }
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
              )}

              {/* The patches this point is tracked on (todo 14). Shown for
                  every tracker, not just template matching: "what does the
                  tracker see here" is the same question whatever is doing the
                  looking. Behind the card's own disclosure since todo 5, so two
                  tracks' patches can be compared side by side. */}
              {/* Pattern and search sizes are a tracker's settings, and a
                  composite has no tracker to settle them for. */}
              {isOpen && !composite && <PatchPanel pointId={point.id} />}

              {isOpen && composite && (
                <div className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  <span className="points-list__confidence">
                    {point.tracked_frames} frames · {(point.mean_confidence * 100).toFixed(0)}%
                    confidence
                  </span>
                </div>
              )}

              {isOpen && !composite && (
              <div className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                <button
                  className={point.anchors_changed ? "button--active" : ""}
                  title={
                    point.anchors_changed
                      ? "Track every anchor again, forwards and backwards, and merge the runs"
                      : "Track every anchor, forwards and backwards, and merge the runs"
                  }
                  onClick={() => handleTrack(point.id)}
                >
                  {point.has_track ? "Re-track" : "Track"}
                </button>
                {/* An anchor is an input to tracking, so changing one leaves
                    this track answering the old question. Saying so is the
                    whole of "delete an anchor and the tracking updates" for the
                    merged baseline — the spans a run from an anchor produced go
                    away on their own, but the merge cannot be un-merged without
                    running it again, and that is minutes of GPU nobody asked
                    for on a click. */}
                {point.anchors_changed && (
                  <span
                    className="points-list__warning"
                    title="This track was computed from a different set of anchors. Re-track to plan through the ones you have now."
                  >
                    ⟳ anchors changed
                  </span>
                )}
                {point.has_track && (
                  <span className="points-list__confidence">
                    {point.tracked_frames} frames · {(point.mean_confidence * 100).toFixed(0)}%
                    confidence
                  </span>
                )}
                {point.worst_frames.length > 0 && (
                  <span
                    className="points-list__warning"
                    title="Frames where the multi-anchor merge disagreed most — consider adding an anchor here"
                    onClick={() => setCurrentFrame(point.worst_frames[0])}
                  >
                    ⚠ check frame {point.worst_frames[0]}
                  </span>
                )}
                <JobProgress job={trackJob} />
              </div>
              )}

              {/* The same run, from the playhead rather than from an anchor:
                  "carry on from where I am", for the frames between anchors.
                  No longer behind `has_track` — a run from an anchor is how a
                  track gets started, not only how one gets repaired. */}
              {isOpen && !composite && (
                <div className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  <button
                    title={`Track backwards from frame ${currentFrame} (${runLabel(runLimit, runFrames)})`}
                    onClick={() => handleRun(point.id, currentFrame, "bwd")}
                  >
                    ◀ track
                  </button>
                  <button
                    title={`Track forwards from frame ${currentFrame} (${runLabel(runLimit, runFrames)})`}
                    onClick={() => handleRun(point.id, currentFrame, "fwd")}
                  >
                    track ▶
                  </button>
                  <span>from f{currentFrame}</span>
                  <JobProgress job={retrackJob} />
                </div>
              )}

              {/* This track's gaps, range repairs and corrections (todo 4,
                  sixth round) — what the Edit tab held that was not already
                  somewhere better. All of it is per point, so it belongs on the
                  point's card next to its anchors and its patches. */}
              {isOpen && <TrackEdits pointId={point.id} />}

              {isOpen && violations.length > 0 && (
                <ul className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  {violations.map((v) => (
                    <li
                      key={`${v.frame}-${v.origin}-${v.direction}`}
                      style={{ color: FLAGGED_ANCHOR_COLOR, cursor: "pointer", listStyle: "none" }}
                      title={`The run from frame ${v.origin} stopped here rather than drift past an anchor it disagreed with`}
                      onClick={() => setCurrentFrame(v.frame)}
                    >
                      ⚑ anchor f{v.frame} missed by {v.distance.toFixed(1)} px
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>

      {/* A track made from the ones above (todo 2, tenth round). Below the
          list because it is about the list, and shared with the other Tracks
          panel so the flag stays a judgement about layout. */}
      <DeriveTrackPanel />

      {runError && <p className="panel__empty">{runError}</p>}
    </div>
  );
}
