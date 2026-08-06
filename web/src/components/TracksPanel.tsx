import { useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import type { Keyframe, TrackerInfo } from "../types";
import { JobProgress } from "./JobProgress";
import { AllPatchesStrip, PatchPanel } from "./PatchPanel";
import { TrackEdits } from "./TrackEdits";
import { OVERLAY_LABELS } from "./Viewer";

/** How far a keyframe may be from where a run landed before it is flagged. */
const FLAGGED_KEYFRAME_COLOR = "#ff3b30";

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

/** How far a bounded run should reach (todos 12 and 20). */
type RunLimit = "clip" | "keyframe" | "frames";

/** The bounded-run body `POST .../points/{id}/retrack` accepts (W1). */
interface BoundedRunBody {
  frame: number;
  direction: "fwd" | "bwd";
  stop_at_keyframe?: boolean;
  max_frames?: number;
}

/**
 * Submit a bounded run.
 *
 * The endpoint takes an *optional* position (omitted here: the run starts from
 * wherever the point already is) plus the two bounds. `web/src/api/client.ts`
 * still describes the older, position-required body and belongs to another
 * workstream, so the shape is asserted here rather than by widening that file.
 */
function submitBoundedRun(projectId: string, pointId: string, body: BoundedRunBody) {
  type ClientBody = Parameters<typeof api.retrackPoint>[2];
  return api.retrackPoint(projectId, pointId, body as unknown as ClientBody);
}

/**
 * One card per track: what it is, what it sees, what it drew, and what it shows.
 *
 * This was a list of *points*, which is how the data model thinks but not how
 * the work does. The unit an operator handles is a track — its tracker, its
 * keyframes, its patches, its runs, and (todo 9) which of its overlays are on
 * the canvas. Putting the visibility toggles on the card is the whole point of
 * the change: with several tracks up, "hide the dotted trails" almost always
 * means *that* track's trails, and the global checkboxes above the canvas could
 * only ever mean all of them.
 *
 * Keyframes are the operator's truth. One a tracking run reached and disagreed
 * with is listed in red here and on the timeline rather than being quietly
 * averaged away, and a run started from one is bounded — to the next keyframe,
 * or to a number of frames — instead of always ploughing to the end of the clip.
 */
export function TracksPanel() {
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
    addKeyframeHere,
    deleteKeyframe,
    latestJobFor,
    tracks,
    invalidateTrack,
    overlays,
    overlaysFor,
    setTrackOverlay,
    trackOverrides,
  } = useProjectContext();

  const [runLimit, setRunLimit] = useState<RunLimit>("keyframe");
  const [runFrames, setRunFrames] = useState(10);
  const [runError, setRunError] = useState<string | null>(null);
  // Why the last ✕ did nothing, if it did nothing — the server refuses to leave
  // a point with no stated position at all, which is a rule worth saying out
  // loud rather than a click that silently fails.
  const [keyframeError, setKeyframeError] = useState<string | null>(null);
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

  async function handleDeleteKeyframe(pointId: string, frame: number) {
    setKeyframeError(await deleteKeyframe(pointId, frame));
  }

  /**
   * Retrack from the current frame in one direction, bounded by the panel's
   * limit. No x/y is sent: the run starts from wherever the point already is,
   * which is what "carry on tracking from here" means.
   */
  async function handleBoundedRun(pointId: string, direction: "fwd" | "bwd") {
    setRunError(null);
    const body: BoundedRunBody = { frame: currentFrame, direction };
    if (runLimit === "keyframe") body.stop_at_keyframe = true;
    if (runLimit === "frames") body.max_frames = runFrames;
    try {
      await submitBoundedRun(project.id, pointId, body);
      invalidateTrack(pointId);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="panel points-panel">
      <div className="panel__header">
        <h3>Tracks</h3>
        <button
          className={tool === "add-point" ? "button--active" : ""}
          onClick={() => setTool(tool === "add-point" ? "move" : "add-point")}
        >
          {tool === "add-point" ? "Click video to place…" : "+ Add track"}
        </button>
      </div>

      {project.points.length === 0 && (
        <p className="panel__empty">No tracks yet. Click "Add track" then click on the video.</p>
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
          // Every position the operator has stated for this point, as one list:
          // the server reconciles pins and the positions manual corrections
          // state, and which storage backs a given one is none of this panel's
          // business. Before the first track there is nothing cached, and the
          // point's own queries are the same list in the other storage.
          const keyframes: Keyframe[] =
            track?.keyframes ??
            point.queries.map((q) => ({ ...q, source: "pin" as const, correction_id: null }));
          const flagged = new Set(track?.flagged_keyframes ?? []);
          const violations = track?.keyframe_violations ?? [];
          const isOpen = expanded[point.id] ?? isSelected;
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
                <span className="points-list__swatch" style={{ background: pointColor(point.id) }} />
                <span className="points-list__id">{point.id}</span>
                {!isOpen && point.has_track && (
                  <span className="points-list__confidence">
                    {point.tracked_frames}f · {(point.mean_confidence * 100).toFixed(0)}%
                  </span>
                )}
                <select
                  value={point.tracker}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => handleTrackerChange(point.id, e.target.value)}
                  style={isOpen ? undefined : { display: "none" }}
                >
                  {trackers.map((t) => (
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
                {unavailableReason(trackers, point.tracker) && (
                  <span
                    className="points-list__warning"
                    title={unavailableReason(trackers, point.tracker) ?? undefined}
                  >
                    ⚠ not installed
                  </span>
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

              {isOpen && (
              <div className="points-list__keyframes" onClick={(e) => e.stopPropagation()}>
                {keyframes.map((k) => {
                  const missed = flagged.has(k.frame);
                  return (
                    <span
                      key={k.frame}
                      className="keyframe-chip"
                      style={missed ? { borderColor: FLAGGED_KEYFRAME_COLOR } : undefined}
                      title={
                        missed
                          ? `Tracking disagrees with your keyframe at frame ${k.frame} — click to go there`
                          : `Keyframe at frame ${k.frame} — click to go there, × to remove it`
                      }
                      onClick={() => setCurrentFrame(k.frame)}
                    >
                      <span style={missed ? { color: FLAGGED_KEYFRAME_COLOR } : undefined}>
                        f{k.frame}
                      </span>
                      <button
                        className="keyframe-chip__delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteKeyframe(point.id, k.frame);
                        }}
                      >
                        ×
                      </button>
                    </span>
                  );
                })}
                <button
                  className={`keyframe-add ${
                    tool === "add-keyframe" && isSelected ? "button--active" : ""
                  }`}
                  title={`Set a keyframe at frame ${currentFrame} where this point already is (shortcut: A), or click the frame to place it somewhere else (⇧A)`}
                  onClick={async () => {
                    setSelectedPointId(point.id);
                    if (tool === "add-keyframe" && isSelected) {
                      setTool("move");
                      return;
                    }
                    // Same rule as the shortcut: state where the point already
                    // is, and only ask for a click when there is nothing there
                    // to state.
                    if (!(await addKeyframeHere(point.id))) setTool("add-keyframe");
                  }}
                >
                  + keyframe
                </button>
              </div>
              )}
              {isOpen && keyframeError && (
                <div className="points-list__error" onClick={(e) => e.stopPropagation()}>
                  {keyframeError}
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
                  return (
                    <button
                      key={key}
                      className={`track-overlay ${active ? "track-overlay--on" : ""} ${
                        override === undefined ? "" : "track-overlay--set"
                      }`}
                      title={`${help}
${
                        override === undefined
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
              {isOpen && <PatchPanel pointId={point.id} />}

              {isOpen && (
              <div className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                <button onClick={() => handleTrack(point.id)}>
                  {point.has_track ? "Re-track" : "Track"}
                </button>
                {point.has_track && (
                  <span className="points-list__confidence">
                    {point.tracked_frames} frames · {(point.mean_confidence * 100).toFixed(0)}%
                    confidence
                  </span>
                )}
                {point.worst_frames.length > 0 && (
                  <span
                    className="points-list__warning"
                    title="Frames where the tracking runs disagreed most — consider setting a keyframe here"
                    onClick={() => setCurrentFrame(point.worst_frames[0])}
                  >
                    ⚠ check frame {point.worst_frames[0]}
                  </span>
                )}
                <JobProgress job={trackJob} />
              </div>
              )}

              {point.has_track && isOpen && (
                <div className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  <button
                    title={`Track backwards from frame ${currentFrame}`}
                    onClick={() => handleBoundedRun(point.id, "bwd")}
                  >
                    ◀ track
                  </button>
                  <button
                    title={`Track forwards from frame ${currentFrame}`}
                    onClick={() => handleBoundedRun(point.id, "fwd")}
                  >
                    track ▶
                  </button>
                  <span>from f{currentFrame}</span>
                  <select
                    value={runLimit}
                    onChange={(e) => setRunLimit(e.target.value as RunLimit)}
                    title="How far the run should go"
                  >
                    <option value="keyframe">to next keyframe</option>
                    <option value="frames">for N frames</option>
                    <option value="clip">to end of clip</option>
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
                  <JobProgress job={retrackJob} />
                </div>
              )}

              {/* This track's gaps, range repairs and corrections (todo 4,
                  sixth round) — what the Edit tab held that was not already
                  somewhere better. All of it is per point, so it belongs on the
                  point's card next to its keyframes and its patches. */}
              {isOpen && <TrackEdits pointId={point.id} />}

              {/* Where the tracker and the operator disagree (W1's
                  `keyframe_violations`). Said as the actionable thing it is —
                  the run stopped rather than drift through a keyframe it could
                  not reach — and clicking goes to the frame to fix it. */}
              {isOpen && violations.length > 0 && (
                <ul className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  {violations.map((v) => (
                    <li
                      key={`${v.frame}-${v.origin}-${v.direction}`}
                      style={{ color: FLAGGED_KEYFRAME_COLOR, cursor: "pointer", listStyle: "none" }}
                      title={`Tracking ${
                        v.direction === "fwd" ? "forwards" : "backwards"
                      } from frame ${v.origin} stopped here rather than drift past a keyframe it could not reach. Click to go to frame ${
                        v.frame
                      }.`}
                      onClick={() => setCurrentFrame(v.frame)}
                    >
                      ⚑ tracking disagrees with your keyframe at frame {v.frame} by{" "}
                      {v.distance.toFixed(1)} px — adjust the point or set a keyframe nearby
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>

      {runError && <p className="panel__empty">{runError}</p>}
    </div>
  );
}
