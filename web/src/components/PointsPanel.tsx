import { useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import type { Query, TrackData, TrackerInfo } from "../types";
import { JobProgress } from "./JobProgress";
import { AllPatchesStrip, PatchPanel } from "./PatchPanel";

/** How far a keyframe may be from where a run landed before it is flagged. */
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

/** One keyframe a tracking run reached and disagreed with. */
interface AnchorViolation {
  frame: number;
  anchor: [number, number];
  tracked: [number, number];
  distance: number;
  origin: number;
  direction: "fwd" | "bwd";
}

/**
 * The anchor-integrity fields `GET .../points/{id}/track` adds (W1), kept local
 * to the components that read them rather than on the shared `TrackData`.
 *
 * `ground_truth` is every position the operator has stated — the point's pins
 * *plus* the keyframes their manual corrections were promoted to — so a drag
 * appears here as the anchor it has become.
 */
interface AnchorIntegrity {
  ground_truth?: Query[];
  anchor_violations?: AnchorViolation[];
  flagged_anchors?: number[];
}

type PointTrack = TrackData & AnchorIntegrity;

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
 * Point list: tracker picker, anchor pins, "track" trigger with live progress,
 * bounded forward/backward retracking, and the per-point confidence /
 * disagreement summary that tells the user which point needs another anchor.
 *
 * Keyframes are ground truth. A pin the tracker could not reconcile with what it
 * saw is listed in red here and on the timeline rather than being quietly
 * averaged away, and a run started from one is bounded — to the next keyframe,
 * or to a number of frames — instead of always ploughing to the end of the clip.
 */
export function PointsPanel() {
  const {
    project,
    trackers,
    refresh,
    currentFrame,
    selectedPointId,
    setSelectedPointId,
    setCurrentFrame,
    placingMode,
    setPlacingMode,
    addAnchorHere,
    afterEdit,
    latestJobFor,
    tracks,
    invalidateTrack,
  } = useProjectContext();

  const [runLimit, setRunLimit] = useState<RunLimit>("keyframe");
  const [runFrames, setRunFrames] = useState(10);
  const [runError, setRunError] = useState<string | null>(null);

  async function handleTrack(pointId: string) {
    await api.trackPoint(project.id, pointId, {});
    invalidateTrack(pointId);
  }

  async function handleTrackerChange(pointId: string, tracker: string) {
    await api.updatePoint(project.id, pointId, { tracker });
    await refresh();
  }

  async function handleEnabledChange(pointId: string, enabled: boolean) {
    await api.updatePoint(project.id, pointId, { enabled_for_solve: enabled });
    await refresh();
  }

  async function handleDelete(pointId: string) {
    await api.deletePoint(project.id, pointId);
    if (selectedPointId === pointId) setSelectedPointId(null);
    await refresh();
  }

  async function handleDeleteAnchor(pointId: string, frame: number) {
    try {
      await api.deleteAnchor(project.id, pointId, frame);
      // The track carries the keyframe list, so it has to be re-read too — a
      // project refresh alone would leave the deleted pin listed as if it had
      // been promoted from a correction.
      await afterEdit(pointId);
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
    }
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
        <h3>Points</h3>
        <button
          className={placingMode === "new-point" ? "button--active" : ""}
          onClick={() => setPlacingMode(placingMode === "new-point" ? "none" : "new-point")}
        >
          {placingMode === "new-point" ? "Click video to place…" : "+ Add point"}
        </button>
      </div>

      {project.points.length === 0 && (
        <p className="panel__empty">No points yet. Click "Add point" then click on the video.</p>
      )}

      {/* Every point's patch on this frame, so "which one has drifted" is one
          glance rather than one click per point (todo 14). */}
      <AllPatchesStrip />

      <ul className="points-list">
        {project.points.map((point) => {
          const trackJob = latestJobFor("track", point.id);
          const retrackJob = latestJobFor("retrack", point.id);
          const isSelected = point.id === selectedPointId;
          const track = tracks[point.id] as PointTrack | undefined;
          // The point's own pins, plus the keyframes its corrections were
          // promoted to. Pins come from the project (refreshed the moment one
          // is placed); promotions only exist on the track, which is fetched
          // and cached separately — so reading the list off `ground_truth`
          // alone left a just-placed anchor missing here until the next track
          // run replaced that cache, even though the canvas drew it at once.
          const promoted = (track?.ground_truth ?? []).filter(
            (g) => !point.queries.some((q) => q.frame === g.frame),
          );
          const anchors: Query[] = [...point.queries, ...promoted].sort(
            (a, b) => a.frame - b.frame,
          );
          const flagged = new Set(track?.flagged_anchors ?? []);
          const violations = track?.anchor_violations ?? [];
          return (
            <li
              key={point.id}
              className={`points-list__item ${isSelected ? "points-list__item--selected" : ""}`}
              onClick={() => setSelectedPointId(point.id)}
            >
              <div className="points-list__row">
                <span className="points-list__swatch" style={{ background: pointColor(point.id) }} />
                <span className="points-list__id">{point.id}</span>
                <select
                  value={point.tracker}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => handleTrackerChange(point.id, e.target.value)}
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
                <label className="points-list__enabled" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={point.enabled_for_solve}
                    onChange={(e) => handleEnabledChange(point.id, e.target.checked)}
                  />
                  solve
                </label>
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

              <div className="points-list__anchors" onClick={(e) => e.stopPropagation()}>
                {anchors.map((q) => {
                  const missed = flagged.has(q.frame);
                  const pinned = point.queries.some((p) => p.frame === q.frame);
                  return (
                    <span
                      key={q.frame}
                      className="anchor-chip"
                      style={missed ? { borderColor: FLAGGED_ANCHOR_COLOR } : undefined}
                      title={
                        missed
                          ? "No tracking run reached this keyframe — needs manual review"
                          : pinned
                            ? "Anchor keyframe"
                            : "Keyframe promoted from a manual correction"
                      }
                      onClick={() => setCurrentFrame(q.frame)}
                    >
                      <span style={missed ? { color: FLAGGED_ANCHOR_COLOR } : undefined}>
                        {pinned ? "" : "✎"}f{q.frame}
                      </span>
                      {pinned && point.queries.length > 1 && (
                        <button
                          className="anchor-chip__delete"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteAnchor(point.id, q.frame);
                          }}
                        >
                          ×
                        </button>
                      )}
                    </span>
                  );
                })}
                <button
                  className={`anchor-add ${
                    placingMode === "add-anchor" && isSelected ? "button--active" : ""
                  }`}
                  title={`Pin this point at frame ${currentFrame} where it already is (shortcut: A), or click the frame to place it somewhere else (⇧A)`}
                  onClick={async () => {
                    setSelectedPointId(point.id);
                    if (placingMode === "add-anchor" && isSelected) {
                      setPlacingMode("none");
                      return;
                    }
                    // Same rule as the shortcut: pin where the point already
                    // is, and only ask for a click when there is nothing there
                    // to pin.
                    if (!(await addAnchorHere(point.id))) setPlacingMode("add-anchor");
                  }}
                >
                  + anchor
                </button>
              </div>

              {/* The patches this point is tracked on (todo 14). Shown for
                  every tracker, not just template matching: "what does the
                  tracker see here" is the same question whatever is doing the
                  looking. Only for the selected point, so the list stays
                  scannable when a shot has a dozen points. */}
              {isSelected && <PatchPanel pointId={point.id} />}

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
                    title="Frames where the multi-anchor merge disagreed most — consider adding an anchor here"
                    onClick={() => setCurrentFrame(point.worst_frames[0])}
                  >
                    ⚠ check frame {point.worst_frames[0]}
                  </span>
                )}
                <JobProgress job={trackJob} />
              </div>

              {point.has_track && isSelected && (
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

              {violations.length > 0 && (
                <ul className="points-list__stats" onClick={(e) => e.stopPropagation()}>
                  {violations.map((v) => (
                    <li
                      key={`${v.frame}-${v.origin}-${v.direction}`}
                      style={{ color: FLAGGED_ANCHOR_COLOR, cursor: "pointer", listStyle: "none" }}
                      title={`The run from frame ${v.origin} stopped here rather than drift past a keyframe it disagreed with`}
                      onClick={() => setCurrentFrame(v.frame)}
                    >
                      ⚑ keyframe f{v.frame} missed by {v.distance.toFixed(1)} px
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
