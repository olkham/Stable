import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import { JobProgress } from "./JobProgress";

/**
 * Point list: tracker picker, anchor pins, "track" trigger with live progress,
 * and the per-point confidence/disagreement summary that tells the user which
 * point needs another anchor.
 */
export function PointsPanel() {
  const {
    project,
    trackers,
    refresh,
    selectedPointId,
    setSelectedPointId,
    setCurrentFrame,
    placingMode,
    setPlacingMode,
    latestJobFor,
    invalidateTrack,
  } = useProjectContext();

  async function handleTrack(pointId: string, tracker: string) {
    await api.trackPoint(project.id, pointId, { tracker });
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
          const isSelected = point.id === selectedPointId;
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
                {point.queries.map((q) => (
                  <span key={q.frame} className="anchor-chip" onClick={() => setCurrentFrame(q.frame)}>
                    f{q.frame}
                    {point.queries.length > 1 && (
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
                ))}
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
                <button onClick={() => handleTrack(point.id, point.tracker)}>
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
            </li>
          );
        })}
      </ul>
    </div>
  );
}
