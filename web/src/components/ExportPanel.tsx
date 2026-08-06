import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { applyWarp, useExportContext } from "../state/ExportContext";
import { useFrameStream } from "../state/FrameStreamContext";
import { useProjectContext } from "../state/ProjectContext";
import type { JobEvent } from "../types";

const CODECS = ["h264", "h265", "vp9"];

const BORDER_HELP: Record<string, string> = {
  crop: "Output the largest area every frame covers. Nothing is scaled, so the picture stays pixel-exact and the file is smaller than the source.",
  auto_zoom:
    "That same area scaled back to delivery size — the minimal zoom that leaves no gaps. Same resolution in, same resolution out.",
  transparent:
    "No crop at all; uncovered area becomes alpha. Renders a PNG sequence, since no video codec here carries an alpha channel.",
};

const TARGET_HELP: Record<string, string> = {
  frame:
    "Undo only what the smoothing removed. With no smoothing filter enabled that is the identity — the export would just re-encode the source, so it is refused rather than run.",
  centroid:
    "Hold the average position of every solve-enabled tracked point still. Stabilises on its own, with or without smoothing.",
  point: "Hold one chosen point still, pinned to where it sat on the anchor frame.",
  map: "Hold chosen points at places you pick on the frame. Drag a crosshair on the video to compose the shot; add a second to correct rotation and scale as well as position.",
};

const TARGET_LABEL: Record<string, string> = {
  frame: "smoothing only",
  centroid: "average of all points",
  point: "a single point",
  map: "points placed on the frame",
};

const FILL_HELP: Record<string, string> = {
  solid: "A flat colour in the exposed edge.",
  stretch:
    "The outermost row and column smeared outwards. Natural on soft edges, streaky on busy ones.",
  mirror:
    "The frame reflected across its own edge. Real texture out there, at the cost of a visible symmetry.",
  blur: "The mirrored edge, blurred. No symmetry to spot and no hard seam — the usual choice.",
};

/**
 * How long the crop slider may run ahead of the server before it tells it.
 *
 * Long enough that a drag is a handful of round trips rather than one per pixel;
 * short enough that the two canvases look like they are following the thumb.
 */
const CROP_PREVIEW_MS = 120;

/**
 * Send a value to the server *while* a control is still being dragged.
 *
 * One request in flight at a time and one value waiting behind it — never a
 * queue. The waiting value is always the newest, so a drag that outruns the
 * server drops the steps in between rather than replaying them, and the server
 * is never asked to plan a crop the operator has already moved past. Keeping it
 * to one at a time also keeps the writes ordered, which matters here: each one
 * stores the crop and the last one to land is the one that stays stored.
 *
 * `flush` is the end of the gesture. It cancels anything waiting, lets whatever
 * is in flight finish so it cannot land *after* the final value, and then sends
 * that value — so the stored crop is always the one the thumb was left on.
 */
function useLiveCommit(send: (value: number | null) => Promise<void>, delay: number) {
  const sendRef = useRef(send);
  sendRef.current = send;
  // A one-element box rather than the value itself: `null` is a legitimate crop
  // scale ("automatic"), so "nothing is waiting" needs to be a different thing.
  const pending = useRef<[number | null] | null>(null);
  const inFlight = useRef<Promise<void> | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dispatch = useCallback(async (value: number | null) => {
    const attempt = (async () => {
      try {
        await sendRef.current(value);
      } catch {
        // `patchOptions` reports its own failures; a dropped preview step is
        // not a failure of the gesture.
      }
    })();
    inFlight.current = attempt;
    await attempt;
    if (inFlight.current === attempt) inFlight.current = null;
  }, []);

  const pump = useCallback(async () => {
    timer.current = null;
    if (inFlight.current || !pending.current) return;
    const [value] = pending.current;
    pending.current = null;
    await dispatch(value);
    if (pending.current && !timer.current) {
      timer.current = setTimeout(() => void pump(), delay);
    }
  }, [delay, dispatch]);

  const push = useCallback(
    (value: number | null) => {
      pending.current = [value];
      if (timer.current || inFlight.current) return;
      timer.current = setTimeout(() => void pump(), delay);
    },
    [delay, pump],
  );

  const flush = useCallback(
    async (value: number | null) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = null;
      pending.current = null;
      await inFlight.current;
      await dispatch(value);
    },
    [dispatch],
  );

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  return useMemo(() => ({ push, flush }), [push, flush]);
}

function humanBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} kB`;
  return `${bytes} B`;
}

function toHex(rgb: number[]): string {
  return `#${rgb
    .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0"))
    .join("")}`;
}

function fromHex(hex: string): number[] {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(value.slice(i, i + 2), 16) || 0);
}

/**
 * Everything about the export, in the side panel with the other four tabs.
 *
 * This is the old export modal's side column (todo 1, fourth round). The modal
 * had a canvas of its own that could not zoom, could not pan and had no
 * timeline, and it hid the real viewer while it was open; the picture is now
 * the viewer's, drawn by `Viewer.tsx` from the same `ExportContext` this panel
 * edits, so composing a shot and looking at it are the same screen. The Data
 * links that were the modal's second tab are the section at the bottom.
 *
 * A stabilisation target is *placed*, not merely picked: drag a crosshair onto
 * a thirds line and the shot is composed there. Two or more targets are a
 * different thing again — a similarity cannot generally satisfy two positions
 * by translation alone, so the fit uses rotation and scale, which is exactly
 * how mapping two tracked points onto two chosen places compensates for a
 * rotating, zooming camera. The panel says which of the two is in force, and
 * reports the per-pair miss, because with three or more the fit is
 * least-squares and pretending otherwise would be a lie about the result.
 */
export function ExportPanel() {
  const { project, refresh, clipRange, jobs, cancelJob, tracks, loadTrack } = useProjectContext();
  const {
    opts,
    pairs,
    preview,
    frameGeometry,
    error,
    busy,
    setActive,
    showGuides,
    setShowGuides,
    sideBySide,
    setSideBySide,
    patchOptions,
    putPairs,
    reloadPreview,
    renders,
    deleteRender,
  } = useExportContext();
  const { setWantOutput, setWantGeometry, streaming } = useFrameStream();

  const [borders, setBorders] = useState<string[]>([]);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Empty means "name it for me": the server stamps the time, so exports stop
  // overwriting each other (todo 7, first round).
  const [filename, setFilename] = useState("");
  const [codec, setCodec] = useState(project.render.codec || "h264");
  const [crf, setCrf] = useState(project.render.crf ?? 18);
  // The crop slider's live value. It is *not* driven by the stored options
  // (todo 7, fourth round): a controlled input over a network round trip
  // renders the old value back into the thumb on every move, which is the
  // flicker, and sends one plan request per pixel of drag. The thumb moves at
  // screen rate and one request goes out when the gesture ends.
  //
  // Three states, not two: `undefined` is "no draft, show what is stored",
  // while `null` is the draft "automatic" — collapsing them would make picking
  // automatic fall back to the stored manual scale until the PUT returned, and
  // the select would visibly flip back to "choose a size" and then forward
  // again, which is the flicker this exists to remove.
  const [cropDraft, setCropDraft] = useState<number | null | undefined>(undefined);
  // …and the crop is stored *during* the drag as well as at the end of it (todo
  // 7, sixth round): "the preview doesn't get updated until the user releases
  // the mouse button". The crop scale changes `plan.geometry`, which moves the
  // output rectangle on the left canvas and the picture on the right, so there
  // is no honest way to preview it client-side without also re-planning — a
  // guessed rectangle over a server-rendered output is two canvases disagreeing,
  // which is the thing the last round exists to prevent. So the server is asked,
  // just not once per pixel.
  const previewCrop = useLiveCommit(
    useCallback((scale: number | null) => patchOptions({ crop: { scale } }), [patchOptions]),
    CROP_PREVIEW_MS,
  );

  // The viewer draws the export overlay only while this panel is the one on
  // screen, so mounting *is* the signal.
  useEffect(() => {
    setActive(true);
    return () => setActive(false);
  }, [setActive]);

  // Warping a frame to show the output costs real work per frame, so it is only
  // asked for while this panel is up and the side-by-side view is on. Closing
  // the tab stops it rather than leaving it running behind another panel.
  useEffect(() => {
    setWantOutput(sideBySide);
    return () => setWantOutput(false);
  }, [sideBySide, setWantOutput]);

  // The output rectangle, the thirds and the crosshairs are this tab's overlay,
  // so this tab is what asks the stream to send the geometry they are drawn
  // from. Nothing else on screen reads it.
  useEffect(() => {
    setWantGeometry(true);
    return () => setWantGeometry(false);
  }, [setWantGeometry]);

  useEffect(() => {
    api
      .getBorderPolicies()
      .then(setBorders)
      .catch(() => setBorders(["crop", "auto_zoom", "transparent"]));
  }, []);

  const tracked = project.points.filter((p) => p.has_track);
  // The target list needs each point's position on this frame to place a new
  // crosshair on it, so make sure the paths are in the cache.
  useEffect(() => {
    for (const point of tracked) if (!(point.id in tracks)) void loadTrack(point.id);
  }, [tracked, tracks, loadTrack]);

  const border = project.render.border;
  const ranged = clipRange && !clipRange.whole_clip ? clipRange : null;
  const hasSolve = project.points.some((p) => p.has_track);

  const target = opts?.options.target;
  const crop = opts?.options.crop;
  const fill = opts?.options.fill;
  const mode = target?.mode ?? "frame";
  const paired = new Set(pairs.map((p) => p.point_id));
  // Residuals are per frame, so they come off the frame on the canvas rather
  // than off the whole-clip preview (todo 2, fifth round).
  const residuals = new Map(frameGeometry?.targets.map((row) => [row.point_id, row.residual]) ?? []);
  const cropScale = cropDraft !== undefined ? cropDraft : (crop?.scale ?? null);

  /** Store the crop scale and hand the control back to the stored value. */
  async function commitCrop(scale: number | null) {
    await previewCrop.flush(scale);
    setCropDraft(undefined);
  }

  // Every export ever submitted for this project, newest first (todo 8).
  const queue = useMemo(
    () =>
      Object.values(jobs)
        .filter((job: JobEvent) => job.kind === "export")
        .sort((a, b) => b.updated_at - a.updated_at),
    [jobs],
  );

  async function handleBorderChange(next: string) {
    setSubmitError(null);
    try {
      await api.updateRenderSettings(project.id, { border: next });
      await refresh();
      await reloadPreview();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExport() {
    setSubmitError(null);
    try {
      await api.updateRenderSettings(project.id, { codec, crf });
      await api.exportVideo(project.id, { filename, codec, crf });
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  }

  /**
   * Add a target for a point, starting where that point is right now.
   *
   * A destination is stated in the space the stabilisation warp lands in, so
   * the point's source position has to be pushed through that warp — which is
   * exactly why the preview sends it. "Hold it here" is the common case; the
   * operator drags the crosshair on the video if they want it somewhere else.
   */
  function addTarget(pointId: string) {
    if (!frameGeometry) return;
    const track = tracks[pointId];
    const px = track?.x[frameGeometry.frame];
    const py = track?.y[frameGeometry.frame];
    const source =
      px != null && py != null
        ? { x: px, y: py }
        : { x: frameGeometry.source.width / 2, y: frameGeometry.source.height / 2 };
    const at = applyWarp(frameGeometry.warp, source.x, source.y);
    void putPairs([...pairs, { point_id: pointId, x: at.x, y: at.y }]);
  }

  function removeTarget(pointId: string) {
    void putPairs(pairs.filter((pair) => pair.point_id !== pointId));
  }

  return (
    <div className="panel export-panel">
      <div className="panel__header">
        <h3>Export</h3>
        {busy && <span className="export-panel__saving">saving…</span>}
      </div>

      {preview ? (
        <div className="export-screen__stats">
          <div>
            <strong>
              {preview.output.width}×{preview.output.height}
            </strong>{" "}
            output · {(preview.coverage * 100).toFixed(1)}% of frame kept
          </div>
          <div>
            {preview.gaps
              ? "⚠ the border fill will be visible at this crop"
              : "no gaps — every output pixel comes from the frame"}
          </div>
          <div>moves the picture up to {preview.max_displacement.toFixed(1)} px</div>
          {/* What will actually be written. The In/Out markers bound the render
              (todo 1, first round), and a clip length that is not the one you
              expected is a slow thing to discover from the finished file. */}
          <div>
            {ranged
              ? `writes frames ${ranged.in}–${ranged.out - 1} (${ranged.frames} of ${ranged.clip_frames})`
              : `writes the whole clip (${preview.frames} frames)`}
          </div>
          {preview.is_no_op && <div className="export-screen__warning">{preview.no_op_detail}</div>}
        </div>
      ) : (
        <p className="panel__empty">
          {error ?? "Solve and smooth this project to preview the export."}
        </p>
      )}

      <label className="channel-check" title="Thirds of the output rectangle, drawn on the video">
        <input
          type="checkbox"
          checked={showGuides}
          onChange={(e) => setShowGuides(e.target.checked)}
        />
        Thirds guides on the video
      </label>
      {/* The output beside the frame (todo 1, fifth round): the left canvas keeps
          the picture and the overlay, the right one shows what would be written
          for this frame with these settings. Both come off one message from the
          server, so they can never be a frame apart. */}
      <label
        className="channel-check"
        title="Show the stabilised output for this frame beside the video. Off gives the composing canvas the full width."
      >
        <input
          type="checkbox"
          checked={sideBySide}
          onChange={(e) => setSideBySide(e.target.checked)}
        />
        Side-by-side output
      </label>
      {!streaming && (
        <p className="export-screen__hint">
          The frame socket is not connected, so frames and the preview are coming over plain
          requests — slower to scrub, and the output beside the video will lag the settings.
        </p>
      )}

      <h4>Stabilise on</h4>
      <div className="field-row">
        <select
          value={mode}
          onChange={(e) =>
            void patchOptions({
              target: {
                mode: e.target.value,
                point_id:
                  e.target.value === "point" ? (target?.point_id ?? tracked[0]?.id ?? null) : null,
              },
            })
          }
          disabled={!opts}
        >
          {(opts?.target_modes ?? ["frame"]).map((name) => (
            <option key={name} value={name}>
              {TARGET_LABEL[name] ?? name}
            </option>
          ))}
        </select>
        {mode === "point" && (
          <select
            value={target?.point_id ?? ""}
            onChange={(e) => void patchOptions({ target: { mode: "point", point_id: e.target.value } })}
          >
            {(opts?.points ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} ({p.frames} frames)
              </option>
            ))}
          </select>
        )}
      </div>
      {(mode === "point" || mode === "centroid") && (
        <div className="field-row">
          <label>Anchor frame</label>
          <input
            type="number"
            min={0}
            max={Math.max(project.video.frames - 1, 0)}
            value={target?.anchor_frame ?? project.solve.reference_frame}
            onChange={(e) => void patchOptions({ target: { anchor_frame: Number(e.target.value) } })}
          />
          <span className="modal__hint">the framing that gets held</span>
        </div>
      )}
      <p className="export-screen__hint">{TARGET_HELP[mode] ?? ""}</p>

      {/* Placed targets. The list is shown for every mode so "+ target" is
          reachable — pressing it *is* the choice of `map`, the same way dragging
          a crosshair is — but the residuals only mean anything once the fit is
          actually using them. */}
      {tracked.length === 0 ? (
        <p className="panel__empty">Track a point to use it as a stabilisation target.</p>
      ) : (
        <ul className="export-screen__targets">
          {tracked.map((point) => {
            const residual = residuals.get(point.id);
            return (
              <li key={point.id}>
                <span className="points-list__swatch" style={{ background: pointColor(point.id) }} />
                <span className="export-screen__target-id">{point.id}</span>
                {paired.has(point.id) ? (
                  <>
                    <span className="export-screen__residual">
                      {residual == null ? "no position on this frame" : `${residual.toFixed(2)} px off`}
                    </span>
                    <button onClick={() => removeTarget(point.id)}>remove</button>
                  </>
                ) : (
                  <button disabled={!frameGeometry} onClick={() => addTarget(point.id)}>
                    + target
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {mode === "map" && preview && (
        <p className="export-screen__hint">
          {preview.uses_rotation_and_scale
            ? "Two or more targets: the render rotates and scales the picture to put every point on its mark. With three or more it is a least-squares fit, and the miss per point is shown above."
            : "One target: the render shifts the picture to hold this point still. Add a second to correct rotation and scale too."}{" "}
          Drag a crosshair on the video to move it.
        </p>
      )}

      <h4>Crop</h4>
      <div className="field-row">
        <label>Size</label>
        <select
          value={cropScale == null ? "auto" : "manual"}
          onChange={(e) => {
            const next = e.target.value === "auto" ? null : (crop?.scale ?? 0.9);
            setCropDraft(next);
            void commitCrop(next);
          }}
          disabled={!opts}
        >
          <option value="auto">automatic (minimal, no gaps)</option>
          <option value="manual">choose a size</option>
        </select>
        {cropScale != null && (
          <>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.01}
              value={cropScale}
              // The thumb is local state so it never rubber-bands back to a
              // stored value mid-drag; the server is told at
              // CROP_PREVIEW_MS at most, so both canvases follow the thumb.
              onChange={(e) => {
                const next = Number(e.target.value);
                setCropDraft(next);
                previewCrop.push(next);
              }}
              // …and told exactly once more when the gesture ends, so what is
              // stored is what the thumb was left on.
              onPointerUp={() => void commitCrop(cropScale)}
              onKeyUp={() => void commitCrop(cropScale)}
            />
            <span>{Math.round(cropScale * 100)}%</span>
          </>
        )}
      </div>
      <div className="field-row">
        <label>Centred on</label>
        <select
          value={crop?.centre ?? "frame"}
          onChange={(e) => void patchOptions({ crop: { centre: e.target.value } })}
          disabled={!opts || mode === "frame"}
        >
          {(opts?.crop_centres ?? ["frame"]).map((c) => (
            <option key={c} value={c}>
              {c === "frame" ? "the frame centre" : "the stabilisation target"}
            </option>
          ))}
        </select>
      </div>
      <p className="export-screen__hint">
        Automatic keeps exactly as much as every frame covers, so there is nothing to fill. Keeping
        more than that exposes the edges — which is what the border fill is for. Changing it never
        touches the targets you have placed.
      </p>

      <h4>Border</h4>
      <div className="field-row">
        <label>Policy</label>
        <select value={border} onChange={(e) => void handleBorderChange(e.target.value)}>
          {borders.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      <p className="export-screen__hint">{BORDER_HELP[border] ?? ""}</p>
      <div className="field-row">
        <label>Fill edge</label>
        <select
          value={fill?.mode ?? "solid"}
          onChange={(e) => void patchOptions({ fill: { mode: e.target.value } })}
          disabled={!opts || border === "transparent"}
        >
          {(opts?.fill_modes ?? ["solid"]).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <input
          type="color"
          value={toHex(fill?.color ?? [0, 0, 0])}
          onChange={(e) => void patchOptions({ fill: { color: fromHex(e.target.value) } })}
          title="Fill colour, and the colour the tint blends toward"
        />
      </div>
      <div className="field-row">
        <label>Blur / tint / feather</label>
        <input
          type="number"
          min={0}
          max={200}
          step={1}
          value={fill?.blur ?? 0}
          onChange={(e) => void patchOptions({ fill: { blur: Number(e.target.value) } })}
          title="Gaussian blur applied to the fill, in pixels"
        />
        <input
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={fill?.tint ?? 0}
          onChange={(e) => void patchOptions({ fill: { tint: Number(e.target.value) } })}
          title="How far to blend the fill toward the colour above"
        />
        <input
          type="number"
          min={0}
          max={100}
          step={1}
          value={fill?.feather ?? 0}
          onChange={(e) => void patchOptions({ fill: { feather: Number(e.target.value) } })}
          title="Soften the join between the picture and the fill, in pixels"
        />
      </div>
      <p className="export-screen__hint">
        {FILL_HELP[fill?.mode ?? "solid"] ?? ""} Blur, tint and feather stack on top of any mode, so
        a mirrored edge faded toward a colour is one setting away.
      </p>

      <h4>Encode</h4>
      <div className="field-row">
        <label>Filename</label>
        <input
          value={filename}
          placeholder="stabilized-<timestamp>.mp4"
          onChange={(e) => setFilename(e.target.value)}
        />
      </div>
      <div className="field-row">
        <label>Codec</label>
        <select value={codec} onChange={(e) => setCodec(e.target.value)}>
          {CODECS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <label>CRF</label>
        <input
          type="number"
          min={0}
          max={51}
          value={crf}
          onChange={(e) => setCrf(Number(e.target.value))}
        />
      </div>

      <button
        className="export-screen__go"
        onClick={handleExport}
        disabled={border === "transparent"}
      >
        Add to export queue
      </button>
      {border === "transparent" && (
        <p className="export-screen__hint">
          No codec here carries an alpha channel. Switch to crop or auto_zoom — with a border fill
          the exposed edges are handled without needing alpha at all.
        </p>
      )}
      {submitError && <p className="export-screen__warning">{submitError}</p>}

      {/* The queue (todo 8). Each job carries the settings it was submitted
          with, so several can be lined up with different crops and each renders
          the one it was queued with; they run one at a time, because two FFmpeg
          encodes fight over the same cores to nobody's benefit. */}
      <h4>Queue</h4>
      {queue.length === 0 ? (
        <p className="export-screen__hint">Nothing queued. Exports run in the background.</p>
      ) : (
        <ul className="export-queue">
          {queue.map((job) => (
            <li key={job.id} className={`export-queue__job export-queue__job--${job.status}`}>
              <span className="export-queue__status">{job.status}</span>
              <span className="export-queue__name">
                {job.result?.filename ?? job.message ?? "queued"}
              </span>
              {job.status === "running" && (
                <span className="export-queue__bar">
                  <span style={{ width: `${Math.round(job.fraction * 100)}%` }} />
                </span>
              )}
              {job.status === "queued" && (
                <button onClick={() => void cancelJob(job.id)} title="Remove from the queue">
                  cancel
                </button>
              )}
              {job.error && <span className="export-screen__warning">{job.error}</span>}
            </li>
          ))}
        </ul>
      )}

      <h4>Renders</h4>
      {renders.length === 0 ? (
        <p className="export-screen__hint">No exports yet.</p>
      ) : (
        <ul className="export-screen__renders">
          {renders.map((file) => (
            <li key={file.filename}>
              <a href={file.url} download={file.filename}>
                {file.filename}
              </a>
              <span>{humanBytes(file.bytes)}</span>
              <button onClick={() => void deleteRender(file.filename)} title="Delete this render">
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* The old modal's second tab: the numbers behind the picture. */}
      <h4>Data</h4>
      <p className="export-screen__hint">Point tracks (raw + corrections applied):</p>
      <div className="export-links">
        <a href={api.tracksExportUrl(project.id, "csv")} download="tracks.csv">
          tracks.csv
        </a>
        <a href={api.tracksExportUrl(project.id, "json")} download="tracks.json">
          tracks.json
        </a>
        <a
          href={api.tracksExportUrl(project.id, "ae")}
          download="tracks-ae.txt"
          title="After Effects keyframe data — paste onto a layer's Position property"
        >
          After Effects
        </a>
      </div>
      <p className="export-screen__hint">Solved + smoothed X/Y/R/S curves:</p>
      <div className="export-links">
        <a href={api.curvesExportUrl(project.id, "csv")} download="curves.csv">
          curves.csv
        </a>
        <a href={api.curvesExportUrl(project.id, "json")} download="curves.json">
          curves.json
        </a>
      </div>
      {!hasSolve && <p className="panel__empty">Curves require a solve to have run first.</p>}
    </div>
  );
}
