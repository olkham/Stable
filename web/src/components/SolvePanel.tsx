import { useEffect, useState } from "react";
import * as api from "../api/client";
import { CHANNEL_LABEL } from "../colors";
import type { Channel, SolveCurves } from "../types";
import { CHANNELS } from "../types";
import { useProjectContext } from "../state/ProjectContext";
import { JobProgress } from "./JobProgress";

/**
 * Reference frame + channel selection, the "Solve" job trigger, and the solve's
 * quality summary.
 *
 * The job runs stable.solver: a confidence-weighted Umeyama fit with RANSAC
 * that degrades frame by frame rather than failing, recording how each frame
 * was arrived at. That record is what the flagged count below reports — the
 * "go fix this" signal the Edit panel and curve editor are there to act on.
 *
 * Note that turning `r` or `s` off *constrains the fit* (a genuinely rigid or
 * scale-locked solve), while turning `x` or `y` off zeroes that pivot-relative
 * translation curve — see stable/solver/similarity.py.
 */
export function SolvePanel() {
  const {
    project,
    refresh,
    currentFrame,
    setCurrentFrame,
    latestJobFor,
    editRevision,
    colorFor,
  } = useProjectContext();
  const [curves, setCurves] = useState<SolveCurves | null>(null);
  const job = latestJobFor("solve");
  const trackedCount = project.points.filter((p) => p.enabled_for_solve && p.has_track).length;

  useEffect(() => {
    api
      .getSolve(project.id)
      .then((res) => setCurves(res.curves))
      .catch(() => setCurves(null));
  }, [project.id, editRevision]);

  useEffect(() => {
    if (job?.status === "done") {
      api.getSolve(project.id).then((res) => setCurves(res.curves));
    }
  }, [job?.status, job?.updated_at, project.id]);

  async function toggleChannel(channel: Channel) {
    await api.updateSolveSettings(project.id, {
      channels: { ...project.solve.channels, [channel]: !project.solve.channels[channel] },
    });
    await refresh();
  }

  async function setReferenceFrame(frame: number) {
    await api.updateSolveSettings(project.id, { reference_frame: frame });
    await refresh();
  }

  async function handleSolve() {
    await api.runSolve(project.id);
  }

  /**
   * Include or exclude one track (todo 6, fourth round).
   *
   * This checkbox used to live on every card in the Tracks panel, where it was
   * one decision split across two screens: you chose the tracks over there and
   * pressed Solve over here, and the empty state on this tab literally told you
   * to go back and tick a box. It is the same `enabled_for_solve` flag and the
   * same endpoint — only now it sits next to the button it governs.
   */
  async function setEnabled(pointId: string, enabled: boolean) {
    await api.updatePoint(project.id, pointId, { enabled_for_solve: enabled });
    await refresh();
  }

  async function setAllEnabled(enabled: boolean) {
    for (const point of project.points) {
      if (point.enabled_for_solve === enabled) continue;
      await api.updatePoint(project.id, point.id, { enabled_for_solve: enabled });
    }
    await refresh();
  }

  // `residual` is null on frames nothing was fitted for, so those are skipped
  // rather than counted as a perfect zero.
  const measured = (curves?.residual ?? []).filter((v): v is number => v !== null);
  const avgResidual = measured.length
    ? measured.reduce((a, b) => a + b, 0) / measured.length
    : null;
  const firstFlagged = curves?.flagged.findIndex(Boolean) ?? -1;

  return (
    <div className="panel solve-panel">
      <div className="panel__header">
        <h3>Solve</h3>
      </div>

      {/* Todo 11: "it's not clear what the Solve tab does". It is the step
          between tracking and stabilising, and nothing on screen said so. */}
      <p className="panel__explainer">
        <strong>Turns your tracks into one camera move.</strong> Several points
        moving separately become four curves for the whole shot — how far it
        panned (X, Y), how much it rotated (R) and zoomed (S) relative to the
        reference frame. Smoothing then decides which parts of that move to
        keep, and the export warps each frame by the difference. Without a
        solve there is nothing to smooth and nothing to stabilise.
      </p>

      <h4 className="solve-panel__heading">Tracks in the solve</h4>
      {project.points.length === 0 ? (
        <p className="field-help">No tracks yet — add one on the Tracks tab.</p>
      ) : (
        <>
          <ul className="solve-tracks">
            {project.points.map((point) => (
              <li key={point.id} className={point.has_track ? "" : "solve-tracks__untracked"}>
                <label
                  title={
                    point.has_track
                      ? `${point.tracked_frames} frames · ${(point.mean_confidence * 100).toFixed(0)}% confidence`
                      : "Not tracked yet — track it on the Tracks tab and it becomes available here"
                  }
                >
                  <input
                    type="checkbox"
                    checked={point.enabled_for_solve}
                    disabled={!point.has_track}
                    onChange={(e) => void setEnabled(point.id, e.target.checked)}
                  />
                  <span
                    className="points-list__swatch"
                    style={{ background: colorFor(point.id) }}
                  />
                  <span className="solve-tracks__id">{point.id}</span>
                  <span className="solve-tracks__stats">
                    {point.has_track
                      ? `${point.tracked_frames}f · ${(point.mean_confidence * 100).toFixed(0)}%`
                      : "not tracked"}
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <div className="field-row">
            <button onClick={() => void setAllEnabled(true)}>All</button>
            <button onClick={() => void setAllEnabled(false)}>None</button>
          </div>
        </>
      )}
      <p className="field-help">
        Which tracks the camera move is fitted from. Untracked ones are listed but cannot be
        chosen — "why is that point not in my solve?" is a question this list should answer rather
        than dodge by hiding it.
      </p>

      <div className="field-row">
        <label>Reference frame</label>
        <input
          type="number"
          min={0}
          max={Math.max(project.video.frames - 1, 0)}
          value={project.solve.reference_frame}
          onChange={(e) => setReferenceFrame(Number(e.target.value))}
        />
        <button onClick={() => setReferenceFrame(currentFrame)}>Use current ({currentFrame})</button>
      </div>
      <p className="field-help">
        The frame the move is measured <em>from</em>: it is the one that does not
        move, so the shot is held where it sits here. Pick the framing you want
        to keep.
      </p>

      <div className="field-row">
        <label>Channels</label>
        <div className="channel-checks">
          {CHANNELS.map((c) => (
            <label key={c} className="channel-check">
              <input
                type="checkbox"
                checked={project.solve.channels[c]}
                onChange={() => toggleChannel(c)}
              />
              {CHANNEL_LABEL[c]}
            </label>
          ))}
        </div>
      </div>
      <p className="field-help">
        Which parts of the move to solve for. Turning <strong>R</strong> or{" "}
        <strong>S</strong> off <em>constrains the fit</em> — use it when you know
        the camera did not rotate or zoom, and want the tracks read as pure
        translation. Turning <strong>X</strong> or <strong>Y</strong> off zeroes
        that curve instead, holding the shot on that axis.
      </p>

      <button
        className="solve-panel__run"
        onClick={handleSolve}
        disabled={trackedCount === 0}
        title={
          trackedCount === 0
            ? "Track at least one solve-enabled track first — the solve is a fit over tracked points"
            : `Fit the camera move over ${trackedCount} tracked point(s)`
        }
      >
        Solve ({trackedCount} track{trackedCount === 1 ? "" : "s"})
      </button>
      {trackedCount === 0 && (
        <p className="field-help">
          Nothing to solve yet: track at least one point on the Tracks tab, then tick it in the
          list above.
        </p>
      )}
      <JobProgress job={job} />

      {curves && (
        <div className="solve-panel__summary">
          <div>{curves.x.length} frames solved</div>
          {avgResidual !== null && <div>avg residual: {avgResidual.toFixed(2)} px</div>}
          <div>
            {curves.summary.flagged === 0 ? (
              <span className="solve-panel__clean">every frame solved cleanly</span>
            ) : (
              <button
                className="link-button points-list__warning"
                onClick={() => firstFlagged >= 0 && setCurrentFrame(firstFlagged)}
                title="Frames that were degraded, interpolated, held, or fitted with a high residual"
              >
                ⚠ {curves.summary.flagged} frame{curves.summary.flagged === 1 ? "" : "s"} to check
                {firstFlagged >= 0 ? ` (from ${firstFlagged})` : ""}
              </button>
            )}
          </div>
          <MiniCurve
            label={CHANNEL_LABEL.x}
            values={curves.x}
            onSeek={setCurrentFrame}
            cursor={currentFrame}
          />
          <MiniCurve
            label={CHANNEL_LABEL.y}
            values={curves.y}
            onSeek={setCurrentFrame}
            cursor={currentFrame}
          />
          <p className="modal__hint">
            X and Y are how far the frame centre moved, in pixels. Open the Curves panel to edit
            them.
          </p>
        </div>
      )}
    </div>
  );
}

/** A tiny inline sparkline for a single raw curve — full editor lives in SmoothingPanel. */
function MiniCurve({
  label,
  values,
  cursor,
  onSeek,
}: {
  label: string;
  values: number[];
  cursor: number;
  onSeek: (frame: number) => void;
}) {
  const width = 220;
  const height = 36;
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const cursorX = (cursor / (values.length - 1)) * width;
  return (
    <div className="mini-curve">
      <span className="mini-curve__label">{label}</span>
      <svg
        width={width}
        height={height}
        onClick={(e) => {
          const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
          const frac = (e.clientX - rect.left) / rect.width;
          onSeek(Math.round(frac * (values.length - 1)));
        }}
      >
        <polyline points={points} fill="none" stroke="#5b9bd5" strokeWidth={1.5} />
        <line x1={cursorX} x2={cursorX} y1={0} y2={height} stroke="#fff" strokeWidth={1} />
      </svg>
    </div>
  );
}
