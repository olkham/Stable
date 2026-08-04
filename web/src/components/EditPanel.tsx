import { useEffect, useState } from "react";
import * as api from "../api/client";
import { useProjectContext } from "../state/ProjectContext";
import type { Correction } from "../types";
import { JobProgress } from "./JobProgress";

/** One-line human description of what an edit does, for the correction list. */
function describe(correction: Correction): string {
  switch (correction.kind) {
    case "offset": {
      const dx = (correction.dx ?? 0).toFixed(1);
      const dy = (correction.dy ?? 0).toFixed(1);
      const falloff = correction.falloff_before ?? 0;
      return `nudge (${dx}, ${dy}) @ ${correction.frame}${falloff ? ` ±${falloff}f` : ""}`;
    }
    case "replace":
      return `set to (${(correction.x ?? 0).toFixed(1)}, ${(correction.y ?? 0).toFixed(1)}) @ ${correction.frame}`;
    case "gap_fill":
      return `${correction.method} fill ${correction.start}–${correction.end}`;
    case "delete_range":
      return `delete ${correction.start}–${correction.end}`;
    case "retrack":
      return `retrack ${correction.direction} from ${correction.frame}`;
    default:
      return correction.kind;
  }
}

/**
 * Manual track editing: the correction list for the selected point, the repair
 * actions (gap fill, delete range, retrack from here), and undo/redo.
 *
 * Nothing here rewrites a track — every button appends, toggles or removes one
 * entry in the point's corrections list, which is what makes undo a single
 * cheap operation rather than a history of arrays.
 */
export function EditPanel() {
  const {
    project,
    currentFrame,
    setCurrentFrame,
    selectedPointId,
    setSelectedPointId,
    tracks,
    editing,
    setEditing,
    dragMode,
    setDragMode,
    falloff,
    setFalloff,
    history,
    undo,
    redo,
    applyCorrection,
    afterEdit,
    editError,
    setEditError,
    latestJobFor,
  } = useProjectContext();

  const point = project.points.find((p) => p.id === selectedPointId) ?? null;
  const track = selectedPointId ? tracks[selectedPointId] : undefined;
  const corrections = track?.corrections ?? [];
  const gaps = track?.gaps ?? [];
  const retrackJob = latestJobFor("retrack", selectedPointId ?? undefined);

  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(0);
  const [method, setMethod] = useState<"spline" | "linear">("spline");
  const [direction, setDirection] = useState<"fwd" | "bwd" | "bidir">("fwd");

  // Default the range fields to a sensible window around the playhead the first
  // time a point is selected, so the buttons are usable without typing.
  useEffect(() => {
    setRangeStart(Math.max(0, currentFrame - 3));
    setRangeEnd(Math.min(project.video.frames - 1, currentFrame + 3));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPointId]);

  async function toggle(correction: Correction) {
    if (!selectedPointId) return;
    setEditError(null);
    try {
      await api.setCorrectionEnabled(
        project.id,
        selectedPointId,
        correction.id,
        !correction.enabled,
      );
      await afterEdit(selectedPointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  async function remove(correction: Correction) {
    if (!selectedPointId) return;
    setEditError(null);
    try {
      await api.deleteCorrection(project.id, selectedPointId, correction.id);
      await afterEdit(selectedPointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  async function revert() {
    if (!selectedPointId) return;
    setEditError(null);
    try {
      await api.clearCorrections(project.id, selectedPointId);
      await afterEdit(selectedPointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  async function retrack() {
    if (!selectedPointId || !track) return;
    const x = track.x[currentFrame];
    const y = track.y[currentFrame];
    if (x == null || y == null) {
      setEditError(
        "the point has no position at this frame — drag it onto its target first, then retrack",
      );
      return;
    }
    setEditError(null);
    try {
      await api.retrackPoint(project.id, selectedPointId, {
        frame: currentFrame,
        x,
        y,
        direction,
        tracker: point?.tracker,
      });
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="panel edit-panel">
      <div className="panel__header">
        <h3>Edit</h3>
        <div className="edit-panel__history">
          <button onClick={() => void undo()} disabled={!history.can_undo} title={history.undo_label ?? "nothing to undo"}>
            ↶ Undo
          </button>
          <button onClick={() => void redo()} disabled={!history.can_redo} title={history.redo_label ?? "nothing to redo"}>
            ↷ Redo
          </button>
        </div>
      </div>

      {editError && <div className="error-text">{editError}</div>}

      <label className="channel-check edit-panel__toggle">
        <input type="checkbox" checked={editing} onChange={(e) => setEditing(e.target.checked)} />
        Click or drag points on the frame to correct them
      </label>
      {editing && (
        <p className="modal__hint">
          Select a point, then click its new location on the frame to move it there — or drag it
          directly. Both apply the same correction below.
        </p>
      )}

      <div className="field-row">
        <label>Drag makes</label>
        <select value={dragMode} onChange={(e) => setDragMode(e.target.value as "offset" | "replace")}>
          <option value="offset">offset (blended)</option>
          <option value="replace">hard replace</option>
        </select>
        {dragMode === "offset" && (
          <label className="param-input">
            falloff
            <input
              type="number"
              min={0}
              value={falloff}
              onChange={(e) => setFalloff(Math.max(0, Number(e.target.value)))}
            />
          </label>
        )}
      </div>

      {!point && <p className="panel__empty">Select a point to edit it.</p>}

      {point && (
        <>
          <div className="edit-panel__section">
            <h4>Repair a range</h4>
            <div className="field-row">
              <label>Frames</label>
              <input
                type="number"
                min={0}
                max={Math.max(project.video.frames - 1, 0)}
                value={rangeStart}
                onChange={(e) => setRangeStart(Number(e.target.value))}
              />
              <span>–</span>
              <input
                type="number"
                min={0}
                max={Math.max(project.video.frames - 1, 0)}
                value={rangeEnd}
                onChange={(e) => setRangeEnd(Number(e.target.value))}
              />
              <button onClick={() => setRangeStart(currentFrame)}>in</button>
              <button onClick={() => setRangeEnd(currentFrame)}>out</button>
            </div>
            <div className="field-row">
              <select value={method} onChange={(e) => setMethod(e.target.value as "spline" | "linear")}>
                <option value="spline">spline</option>
                <option value="linear">linear</option>
              </select>
              <button
                onClick={() =>
                  void applyCorrection(point.id, {
                    kind: "gap_fill",
                    start: rangeStart,
                    end: rangeEnd,
                    method,
                    confidence: 0.5,
                  })
                }
              >
                Fill gap
              </button>
              <button
                onClick={() =>
                  void applyCorrection(point.id, {
                    kind: "delete_range",
                    start: rangeStart,
                    end: rangeEnd,
                  })
                }
              >
                Delete range
              </button>
            </div>
            <p className="modal__hint">
              Deleting a bad span is often the whole repair: the solver carries those frames on the
              remaining points. Fill it when this point itself is needed.
            </p>
          </div>

          {gaps.length > 0 && (
            <div className="edit-panel__section">
              <h4>Gaps ({gaps.length})</h4>
              <div className="edit-panel__gaps">
                {gaps.map(([start, end]) => (
                  <span key={`${start}-${end}`} className="anchor-chip">
                    <button className="link-button" onClick={() => setCurrentFrame(start)}>
                      {start}–{end}
                    </button>
                    <button
                      className="anchor-chip__delete"
                      title="Fill this gap"
                      onClick={() =>
                        void applyCorrection(point.id, {
                          kind: "gap_fill",
                          start,
                          end,
                          method: "spline",
                          confidence: 0.5,
                        })
                      }
                    >
                      ⤳
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="edit-panel__section">
            <h4>Retrack from frame {currentFrame}</h4>
            <div className="field-row">
              <select
                value={direction}
                onChange={(e) => setDirection(e.target.value as "fwd" | "bwd" | "bidir")}
              >
                <option value="fwd">forward</option>
                <option value="bwd">backward</option>
                <option value="bidir">both ways</option>
              </select>
              <button onClick={() => void retrack()}>Retrack</button>
            </div>
            <JobProgress job={retrackJob} />
            <p className="modal__hint">
              Runs {point.tracker} again from this point's current position and splices the result
              in as a correction — the merged baseline underneath is left alone.
            </p>
          </div>

          <div className="edit-panel__section">
            <h4>
              Corrections ({corrections.length})
              {corrections.length > 0 && (
                <button className="link-button edit-panel__revert" onClick={() => void revert()}>
                  revert all
                </button>
              )}
            </h4>
            {corrections.length === 0 && (
              <p className="panel__empty">No edits — this point is raw tracker output.</p>
            )}
            <ul className="corrections-list">
              {corrections.map((correction) => (
                <li
                  key={correction.id}
                  className={`corrections-list__item ${correction.enabled ? "" : "corrections-list__item--off"}`}
                >
                  <input
                    type="checkbox"
                    checked={correction.enabled}
                    onChange={() => void toggle(correction)}
                    title="Toggle this edit off to compare"
                  />
                  <button
                    className="link-button corrections-list__label"
                    onClick={() =>
                      setCurrentFrame(correction.frame ?? correction.start ?? currentFrame)
                    }
                  >
                    {describe(correction)}
                  </button>
                  <button
                    className="points-list__delete"
                    onClick={() => void remove(correction)}
                    title="Remove this edit"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}

      {project.points.length > 0 && !point && (
        <div className="edit-panel__section">
          {project.points.map((p) => (
            <button key={p.id} className="link-button" onClick={() => setSelectedPointId(p.id)}>
              {p.id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
