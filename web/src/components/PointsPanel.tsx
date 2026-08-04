import { useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import type { Query, TrackData } from "../types";
import { JobProgress } from "./JobProgress";

/** How far a keyframe may be from where a run landed before it is flagged. */
const FLAGGED_ANCHOR_COLOR = "#ff3b30";

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
      await refresh();
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

      <ul className="points-list">
        {project.points.map((point) => {
          const trackJob = latestJobFor("track", point.id);
          const retrackJob = latestJobFor("retrack", point.id);
          const isSelected = point.id === selectedPointId;
          const track = tracks[point.id] as PointTrack | undefined;
          // Fall back to the point's own pins until its track has loaded, so
          // the anchor list is never empty just because a fetch is in flight.
          const anchors: Query[] = track?.ground_truth ?? point.queries;
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
                    <option key={t.name} value={t.name} disabled={!t.available}>
                      {t.display_name}
                      {!t.available ? " (unavailable)" : ""}
                    </option>
                  ))}
                </select>
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
                  onClick={() => {
                    setSelectedPointId(point.id);
                    setPlacingMode(
                      placingMode === "add-anchor" && isSelected ? "none" : "add-anchor",
                    );
                  }}
                >
                  + anchor
                </button>
              </div>

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
