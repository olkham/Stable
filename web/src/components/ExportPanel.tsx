import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { applyWarp, useExportContext } from "../state/ExportContext";
import { useFrameStream } from "../state/FrameStreamContext";
import { useProjectContext } from "../state/ProjectContext";

const CODECS = ["h264", "h265", "vp9"];

const BORDER_LABEL: Record<string, string> = {
  crop: "keep the picture, lose frame",
  auto_zoom: "keep the frame, zoom in",
  transparent: "keep everything, alpha the gaps",
};

const BORDER_HELP: Record<string, string> = {
  crop: "Output the largest area every frame covers. Nothing is scaled, so the picture stays pixel-exact and the delivered frame is smaller than the source.",
  auto_zoom:
    "Scale that same area back up to fill the frame — the minimal zoom that leaves no gaps, and never more than the shot needed.",
  transparent:
    "No crop at all; uncovered area becomes alpha. Renders a PNG sequence, since no video codec here carries an alpha channel.",
};

/** What the two policies cost once the targets have fixed the scale. */
const BORDER_WITH_MARKS: Record<string, string> = {
  crop: " Two or more targets already fix how much of the shot is in view, so this keeps them exactly where you put them and shows whatever gaps that leaves.",
  auto_zoom:
    " Two or more targets already fix how much of the shot is in view, so closing the gaps means zooming past them: they ride outwards from their centre, and the list above shows where they actually land.",
};

const SOURCE_HELP: Record<string, string> = {
  raw: "The tracked positions themselves — the solid trail on the video. This nails the feature: it does not move in the delivered frame, and every wobble the tracker saw comes out of the picture.",
  smoothed:
    "The smoothed estimate for the same tracks — the dashed trail on the video. This nails *that* curve, so the fit is driven by a de-noised estimate rather than by every sample, and the feature keeps whatever the smoothing treated as camera motion.",
};

const SOURCE_LABEL: Record<string, string> = {
  raw: "tracked positions (solid trail)",
  smoothed: "smoothed estimate (dashed trail)",
};

/** Delivery sizes worth one click. Anything else goes in the boxes. */
const DELIVERY_SIZES: [string, number, number][] = [
  ["1920x1080", 1920, 1080],
  ["1280x720", 1280, 720],
  ["3840x2160", 3840, 2160],
  ["1080x1080", 1080, 1080],
  ["1080x1920", 1080, 1920],
];

/** The named positions the mark can be dropped on, as output fractions. */
const SNAP_SPOTS: [string, number, number][] = [
  ["centre", 0.5, 0.5],
  ["top left third", 1 / 3, 1 / 3],
  ["top right third", 2 / 3, 1 / 3],
  ["bottom left third", 1 / 3, 2 / 3],
  ["bottom right third", 2 / 3, 2 / 3],
];

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

/** Keep a fraction of the output frame inside the output frame. */
function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

/**
 * One coordinate of the target's position, in output pixels.
 *
 * Typed freely and committed once, on blur or Enter. Storing per keystroke
 * would re-plan the whole clip for every digit — moving the mark changes the
 * crop, because the crop is grown around it — and "192" would go through 1 and
 * 19 on the way, each one a plan and each one visibly re-framing the preview.
 */
function PixelField({
  value,
  max,
  title,
  onCommit,
}: {
  value: number;
  max: number;
  title: string;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const parsed = Number(draft);
    setDraft(null);
    if (Number.isFinite(parsed)) onCommit(Math.min(Math.max(parsed, 0), max));
  };
  return (
    <input
      type="number"
      min={0}
      max={max}
      step={1}
      title={title}
      value={draft ?? Math.round(value)}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") setDraft(null);
      }}
    />
  );
}

function humanBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} kB`;
  return `${bytes} B`;
}

/**
 * A render estimate, in the units the number deserves.
 *
 * Rounded hard on purpose: this is a projection from an average of past
 * renders, and "7m" claims exactly as much as it can support where "6m 52s"
 * would claim a precision the estimate does not have.
 */
function humanDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
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
  const {
    project,
    refresh,
    clipRange,
    currentFrame,
    jobs,
    tracks,
    loadTrack,
    colorFor,
  } = useProjectContext();
  const {
    opts,
    marks,
    preview,
    frameGeometry,
    error,
    busy,
    setActive,
    showGuides,
    setShowGuides,
    sideBySide,
    setSideBySide,
    cleanOutput,
    setCleanOutput,
    snap,
    setSnap,
    selected,
    setSelected,
    commitMark,
    addTarget,
    patchTarget,
    deleteTarget,
    patchOptions,
    reloadPreview,
    renders,
    deleteRender,
    queue,
    enqueueRender,
    moveQueued,
    removeQueued,
    clearFinishedQueued,
    runQueue,
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

  const crop = opts?.options.crop;
  const fill = opts?.options.fill;
  const stored = opts?.options.export_range;
  const resolution = opts?.options.resolution;
  const points = opts?.options.target.points ?? [];
  const cropScale = cropDraft !== undefined ? cropDraft : (crop?.scale ?? null);
  // A mark's position is a fraction of the output, so it is shown in the pixels
  // of *this* output. Change the crop and the number moves; the shot does not,
  // which is the point of storing the fraction.
  const outSize = frameGeometry?.output ?? preview?.output ?? null;
  // What the crop would deliver at if nothing scaled it, for the placeholders.
  const cropSize = opts?.source_size && preview
    ? ([
        Math.round(opts.source_size[0] * preview.crop_scale),
        Math.round(opts.source_size[1] * preview.crop_scale),
      ] as [number, number])
    : null;
  const clipFrames = opts?.clip_frames ?? project.video.frames;
  const markers: [number, number] = opts?.clip_range ?? [0, clipFrames];
  const exportWindow: [number, number] = preview?.export_window ??
    opts?.export_window ?? [0, clipFrames];
  const customRange = stored != null && (stored.start != null || stored.end != null);

  /** Store the crop scale and hand the control back to the stored value. */
  async function commitCrop(scale: number | null) {
    await previewCrop.flush(scale);
    setCropDraft(undefined);
  }

  // The render queue, in the order it will render (todo 6, tenth round). The
  // server's list, not a view of the job table: an entry exists before any
  // worker has it, which is what makes it something to look at and move.
  const entries = queue?.entries ?? [];
  const pending = queue?.pending ?? 0;
  const finished = entries.filter((entry) =>
    ["done", "error", "cancelled"].includes(entry.status),
  ).length;

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

  /**
   * Capture the settings above as a queued render — and *only* that.
   *
   * It used to submit the render there and then, which is what todo 6 of this
   * round is about: an export is minutes of work, and "queue up many different
   * runs, then process them" needs the queueing and the starting to be two
   * decisions. The button below the list is the other one.
   */
  async function handleExport() {
    setSubmitError(null);
    try {
      await api.updateRenderSettings(project.id, { codec, crf });
      setSubmitError(await enqueueRender({ filename, codec, crf }));
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  }

  /**
   * Add a target point for one track, where that track is right now.
   *
   * A mark is a fraction of the *output* frame, so the track's position goes
   * through `output_warp` — the whole transform that produced the picture — and
   * is divided by the output size. "Leave it where it is" is the common opening
   * move; the operator drags the crosshair if they want it somewhere else.
   */
  function addMarkFor(pointIds: string[], source: string) {
    const first = pointIds[0];
    const track = first ? tracks[first] : undefined;
    const px = frameGeometry && track ? track.x[frameGeometry.frame] : null;
    const py = frameGeometry && track ? track.y[frameGeometry.frame] : null;
    let place = { u: 0.5, v: 0.5 };
    if (frameGeometry && px != null && py != null) {
      const at = applyWarp(frameGeometry.output_warp, px, py);
      place = {
        u: clamp01(at.x / frameGeometry.output.width),
        v: clamp01(at.y / frameGeometry.output.height),
      };
    }
    void addTarget({ place, point_ids: pointIds, source });
  }

  /** Add or drop one track from a mark's set; the mark holds their mean. */
  function toggleTrack(mark: { id: string; pointIds: string[] }, pointId: string) {
    const next = new Set(mark.pointIds);
    if (next.has(pointId)) next.delete(pointId);
    else next.add(pointId);
    void patchTarget(mark.id, { point_ids: [...next] });
  }

  /** Store one side of the delivery size; empty means "follow the crop". */
  function setResolution(side: "width" | "height", raw: string) {
    const value = raw.trim() === "" ? null : Math.max(2, Math.round(Number(raw)));
    if (value !== null && !Number.isFinite(value)) return;
    void patchOptions({ resolution: { [side]: value } });
  }

  /**
   * Store the export range, or clear it back to following the In/Out markers.
   *
   * Kept non-empty here rather than letting the server refuse it: typing a new
   * out point one digit at a time passes through values below the in point, and
   * a red error flashing under the operator's hands while they are still typing
   * is not information — it is noise.
   */
  function setRange(patch: { start?: number | null; end?: number | null }) {
    const next = { ...patch };
    if (typeof next.start === "number") {
      next.start = Math.min(Math.max(0, next.start), Math.max(exportWindow[1] - 1, 0));
    }
    if (typeof next.end === "number") {
      next.end = Math.min(Math.max(next.end, exportWindow[0] + 1), clipFrames);
    }
    void patchOptions({ export_range: next });
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
            {exportWindow[1] - exportWindow[0] < preview.frames
              ? `writes frames ${exportWindow[0]}–${exportWindow[1] - 1} (${exportWindow[1] - exportWindow[0]} of ${preview.frames})`
              : `writes the whole clip (${preview.frames} frames)`}
            {customRange ? " — the export's own range" : ranged ? " — the In/Out markers" : ""}
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
      {/* The left canvas has had a master overlay switch since the tools
          landed; the output beside it never did, so the one canvas showing
          what will actually be delivered could not be looked at on its own
          (todo 4, tenth round). */}
      {sideBySide && (
        <label
          className="channel-check"
          title="Draw the output frame with nothing over it — no thirds, no crosshairs, no label — so you can judge the delivered picture. Crosshairs cannot be dragged on that canvas while this is on; the canvas on the left still has them."
        >
          <input
            type="checkbox"
            checked={cleanOutput}
            onChange={(e) => setCleanOutput(e.target.checked)}
          />
          Clean output (hide its overlays)
        </label>
      )}
      {!streaming && (
        <p className="export-screen__hint">
          The frame socket is not connected, so frames and the preview are coming over plain
          requests — slower to scrub, and the output beside the video will lag the settings.
        </p>
      )}

      {/* The targets. One row per mark: where it goes in the delivered frame,
          which tracks it holds, and whether it holds their raw or smoothed
          positions. The fit satisfies as many of them as a similarity can —
          one or two exactly, three or more in the least-squares sense — so the
          miss is shown per mark rather than averaged into a single number. */}
      <div className="panel-section panel-section--counted">
        <h4>Targets</h4>
        <span className="export-screen__count">
          {points.length === 0
            ? "none"
            : `${points.length} · ${
                preview?.uses_rotation_and_scale ? "position, rotation & scale" : "position only"
              }`}
        </span>
      </div>

      {tracked.length === 0 ? (
        <p className="panel__empty">Track a point before mapping it to a target.</p>
      ) : (
        <>
          <ul className="target-list">
            {marks.map((mark) => {
              const isSelected = selected === mark.id;
              const held = mark.pointIds.length
                ? mark.pointIds.join(" + ")
                : "every solve-enabled track";
              return (
                <li
                  key={mark.id}
                  className={`target-list__item${isSelected ? " is-selected" : ""}`}
                  onClick={() => setSelected(isSelected ? null : mark.id)}
                >
                  <div className="target-list__head">
                    {/* The card has always opened on a click; nothing said so
                        (todo 5, ninth round). Same state as the click on the
                        row, not a second one — selecting a target also picks
                        out its crosshair on both canvases, and two notions of
                        "open" would drift apart. */}
                    <button
                      className="points-list__disclose"
                      aria-expanded={isSelected}
                      title={
                        isSelected
                          ? "Hide this target's settings"
                          : "Source, position, snap spots and which tracks this target holds"
                      }
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelected(isSelected ? null : mark.id);
                      }}
                    >
                      {isSelected ? "▾" : "▸"}
                    </button>
                    <span
                      className="points-list__swatch"
                      style={{ background: mark.pointId ? colorFor(mark.pointId) : "#e8e8e8" }}
                    />
                    <span className="target-list__id">{mark.id}</span>
                    <span className="target-list__where">
                      {outSize
                        ? `${Math.round(mark.u * outSize.width)}, ${Math.round(mark.v * outSize.height)}`
                        : `${(mark.u * 100).toFixed(0)}%, ${(mark.v * 100).toFixed(0)}%`}
                    </span>
                    <span className="export-screen__residual">
                      {mark.residual == null ? "—" : `${mark.residual.toFixed(2)} px off`}
                    </span>
                    <button
                      className="target-list__remove"
                      title="Delete this target"
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteTarget(mark.id);
                      }}
                    >
                      ×
                    </button>
                  </div>
                  <div className="target-list__holds" title={`Holds ${held}`}>
                    holds {mark.source === "smoothed" ? "smoothed " : ""}
                    {held}
                  </div>
                  {isSelected && (
                    <div className="target-list__editor" onClick={(e) => e.stopPropagation()}>
                      <div className="field-row">
                        <label>Source</label>
                        <select
                          value={mark.source}
                          onChange={(e) => void patchTarget(mark.id, { source: e.target.value })}
                        >
                          {(opts?.target_sources ?? ["raw"]).map((name) => (
                            <option key={name} value={name}>
                              {SOURCE_LABEL[name] ?? name}
                            </option>
                          ))}
                        </select>
                      </div>
                      <p className="export-screen__hint">{SOURCE_HELP[mark.source] ?? ""}</p>
                      <div className="field-row">
                        <label>Position</label>
                        {outSize ? (
                          <>
                            <PixelField
                              value={mark.u * outSize.width}
                              max={outSize.width}
                              title="Horizontal position in the exported frame, in pixels"
                              onCommit={(x) =>
                                void commitMark(mark.id, clamp01(x / outSize.width), mark.v)
                              }
                            />
                            <PixelField
                              value={mark.v * outSize.height}
                              max={outSize.height}
                              title="Vertical position in the exported frame, in pixels"
                              onCommit={(y) =>
                                void commitMark(mark.id, mark.u, clamp01(y / outSize.height))
                              }
                            />
                            <span className="modal__hint">
                              of {outSize.width}×{outSize.height}
                            </span>
                          </>
                        ) : (
                          <span className="modal__hint">solve and smooth to see the output size</span>
                        )}
                      </div>
                      <div className="export-screen__spots">
                        {SNAP_SPOTS.map(([label, u, v]) => (
                          <button
                            key={label}
                            className={
                              Math.abs(mark.u - u) < 1e-6 && Math.abs(mark.v - v) < 1e-6
                                ? "is-active"
                                : undefined
                            }
                            onClick={() => void commitMark(mark.id, u, v)}
                            title={`Put this target on the ${label}`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <div className="target-list__tracks">
                        <span className="modal__hint">
                          Tracks (several are held at their mean; none means every solve-enabled
                          track)
                        </span>
                        {tracked.map((point) => (
                          <label key={point.id} className="channel-check">
                            <input
                              type="checkbox"
                              checked={mark.pointIds.includes(point.id)}
                              onChange={() => toggleTrack(mark, point.id)}
                            />
                            <span
                              className="points-list__swatch"
                              style={{ background: colorFor(point.id) }}
                            />
                            {point.id}
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          <div className="export-screen__spots">
            {tracked.map((point) => (
              <button
                key={point.id}
                onClick={() => addMarkFor([point.id], "raw")}
                title={`Add a target holding ${point.id} where it is on this frame`}
              >
                + {point.id}
              </button>
            ))}
            <button
              onClick={() => addMarkFor([], "raw")}
              title="Add a target holding the mean of every solve-enabled track"
            >
              + all tracks
            </button>
          </div>
          <p className="export-screen__hint">
            {points.length === 0
              ? "With no targets the render only undoes what the Smoothing panel removed. Add one to hold a track somewhere in the delivered frame."
              : points.length === 1
                ? "One target: the render shifts the picture to hold that path there. Add a second and it will correct rotation and scale too."
                : "Two or more targets fix rotation and scale as well as position, which states the framing completely — so the crop cannot zoom, and any gaps are reported rather than zoomed away. With three or more the fit is least-squares and the miss per target is shown above."}{" "}
            Drag a crosshair on either canvas to move one.
          </p>
        </>
      )}

      <label
        className="channel-check"
        title="While dragging a crosshair, snap it to the centre, the thirds, and the horizontal or vertical line through any other point on screen"
      >
        <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
        Snap targets to the centre, thirds and other points
      </label>

      <div className="panel-section">
        <h4>Crop</h4>
        <span className="panel-section__note">How much of the shot survives, and what happens at the edges</span>
      </div>
      <div className="field-row">
        <label>When it overruns</label>
        <select value={border} onChange={(e) => void handleBorderChange(e.target.value)}>
          {borders.map((name) => (
            <option key={name} value={name}>
              {BORDER_LABEL[name] ?? name}
            </option>
          ))}
        </select>
      </div>
      <p className="export-screen__hint">
        {BORDER_HELP[border] ?? ""}
        {border !== "transparent" && points.length >= 2
          ? BORDER_WITH_MARKS[border] ?? ""
          : ""}
      </p>
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
      <label
        className="channel-check"
        title="Measure the automatic crop over the exported frames only"
      >
        <input
          type="checkbox"
          checked={crop?.limit_to_range ?? false}
          onChange={(e) => void patchOptions({ crop: { limit_to_range: e.target.checked } })}
          disabled={!opts}
        />
        Fit the crop to the export range
      </label>
      <p className="export-screen__hint">
        Automatic keeps exactly as much as every frame covers, so there is nothing to fill. Keeping
        more than that exposes the edges — which is what the border fill is for. Changing it never
        moves the target you have placed: the crop grows around it.
      </p>
      <p className="export-screen__hint">
        {crop?.limit_to_range
          ? `The crop is the largest one with no gaps between the export markers${
              preview ? ` — measured over ${preview.crop_measured_frames} frames` : ""
            }. Frames outside them are not being written, so they get no vote.`
          : "The crop is measured over the whole clip, so one violent second anywhere in it limits the whole render. Tick the box above to measure only what you are actually delivering."}
      </p>

      {/* The export's own range (todo 1, eighth round). Separate from the In/Out markers on
          purpose: tracking wants a margin either side of the shot and delivery
          does not, so making the two the same control means retyping one every
          time the other changes. */}
      {/* Delivery size, which is a different decision from field of view: crop
          hard into a shot and still hand over 1920x1080. Setting both sides
          also sets the crop window's shape, so a square delivery crops a square
          out of a 16:9 source rather than squeezing one into it. */}
      <div className="panel-section">
        <h4>Output resolution</h4>
        <span className="panel-section__note">The pixel size the crop is delivered at</span>
      </div>
      <div className="field-row">
        <label>Deliver at</label>
        <input
          type="number"
          min={2}
          step={2}
          placeholder={cropSize ? String(cropSize[0]) : "auto"}
          value={resolution?.width ?? ""}
          onChange={(e) => setResolution("width", e.target.value)}
          title="Output width in pixels; empty follows the crop"
        />
        <span className="modal__hint">x</span>
        <input
          type="number"
          min={2}
          step={2}
          placeholder={cropSize ? String(cropSize[1]) : "auto"}
          value={resolution?.height ?? ""}
          onChange={(e) => setResolution("height", e.target.value)}
          title="Output height in pixels; empty follows the crop"
        />
        <button
          onClick={() => void patchOptions({ resolution: { width: null, height: null } })}
          disabled={!resolution || (resolution.width == null && resolution.height == null)}
          title="Deliver the crop at whatever size it came out"
        >
          auto
        </button>
      </div>
      <div className="export-screen__spots">
        {DELIVERY_SIZES.map(([label, width, height]) => (
          <button
            key={label}
            className={
              resolution?.width === width && resolution?.height === height ? "is-active" : undefined
            }
            onClick={() => void patchOptions({ resolution: { width, height } })}
            title={`Deliver at ${width}x${height}`}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="export-screen__hint">
        {resolution?.width == null && resolution?.height == null
          ? `Delivering the crop at its own size${outSize ? ` — ${outSize.width}x${outSize.height}` : ""}.`
          : `Scaling the crop to ${outSize ? `${outSize.width}x${outSize.height}` : "the size above"}. Leave a side empty and it follows the crop; set both and the crop takes that shape.`}
      </p>

      <div className="panel-section">
        <h4>Export range</h4>
        <span className="panel-section__note">Which frames get written</span>
      </div>
      {/* Marked from the playhead, the way an In/Out pair is actually set:
          scrub to the frame you mean and press the button. Typing the number is
          still there for when you know it, but nobody knows it by looking. */}
      <div className="export-screen__markers">
        <button
          onClick={() => setRange({ start: currentFrame })}
          title={`Start the export at the current frame (${currentFrame})`}
        >
          ⇥ Set in
        </button>
        <button
          onClick={() => setRange({ end: currentFrame + 1 })}
          title={`End the export after the current frame (${currentFrame})`}
        >
          Set out ⇤
        </button>
        <button
          onClick={() => setRange({ start: null, end: null })}
          disabled={!customRange}
          title="Go back to writing exactly what the In/Out markers bound"
        >
          follow In/Out
        </button>
      </div>
      <div className="field-row">
        <label>Frames</label>
        <input
          type="number"
          min={0}
          max={Math.max(clipFrames - 1, 0)}
          value={exportWindow[0]}
          onChange={(e) => setRange({ start: Number(e.target.value) })}
          title="First frame written, inclusive"
        />
        <span className="modal__hint">to</span>
        <input
          type="number"
          min={1}
          max={clipFrames}
          value={exportWindow[1]}
          onChange={(e) => setRange({ end: Number(e.target.value) })}
          title="One past the last frame written"
        />
        <span className="modal__hint">of {clipFrames}</span>
      </div>
      <p className="export-screen__hint">
        {customRange
          ? `Writing ${exportWindow[1] - exportWindow[0]} frames — the export's own range. The In/Out markers (${markers[0]}–${markers[1] - 1}) still bound tracking.`
          : `Following the In/Out markers, so this render writes frames ${exportWindow[0]}–${exportWindow[1] - 1}. Setting either end above makes it the export's own range and leaves the markers alone.`}
      </p>

      <div className="panel-section">
        <h4>Edge fill</h4>
        <span className="panel-section__note">What goes in the area no frame covers</span>
      </div>
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

      <div className="panel-section">
        <h4>Encode</h4>
        <span className="panel-section__note">The file itself</span>
      </div>
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
        Add to render queue
      </button>
      {border === "transparent" && (
        <p className="export-screen__hint">
          No codec here carries an alpha channel. Switch to crop or auto_zoom — with a border fill
          the exposed edges are handled without needing alpha at all.
        </p>
      )}
      {submitError && <p className="export-screen__warning">{submitError}</p>}

      {/* The render queue (todo 6, tenth round). Queueing no longer starts
          anything: each entry stores the whole configuration it was queued with
          — options, border, smoothing, codec, range — so several runs can be
          lined up with different settings, looked at, reordered and dropped
          before any of them costs a minute of encoding. */}
      <div className="panel-section panel-section--counted">
        <h4>Render queue</h4>
        <span className="export-screen__count">
          {entries.length === 0
            ? "empty"
            : `${pending} waiting${
                pending > 0 ? ` · ${humanDuration(queue?.estimate_seconds ?? 0)}` : ""
              }${pending > 0 && !(queue?.estimate_measured ?? false) ? " (rough)" : ""}`}
        </span>
      </div>
      {entries.length === 0 ? (
        <p className="export-screen__hint">
          Nothing queued. "Add to render queue" captures the settings above as an entry — it does
          not start rendering, so you can queue several different runs and process them together.
        </p>
      ) : (
        <ul className="export-queue">
          {entries.map((entry, index) => {
            // The live progress of the job rendering it, matched by the id the
            // entry recorded — so one list shows both what is waiting and how
            // far the running one has got, rather than two lists to reconcile.
            const job = entry.job_id ? jobs[entry.job_id] : undefined;
            const waiting = entry.status === "pending";
            return (
              <li
                key={entry.id}
                className={`export-queue__job export-queue__job--${entry.status}`}
              >
                <span className="export-queue__status">{entry.status}</span>
                <span className="export-queue__name" title={`${entry.filename} · ${entry.codec} CRF ${entry.crf} · frames ${entry.frame_range[0]}–${entry.frame_range[1] - 1}`}>
                  {entry.filename}
                </span>
                <span className="export-queue__meta">
                  {entry.width}×{entry.height} · {entry.frames}f · {entry.codec}
                </span>
                <span
                  className="export-queue__meta"
                  title={
                    entry.estimate_measured
                      ? "Estimated from renders already finished on this machine"
                      : "A rough estimate — it sharpens once a render has finished here"
                  }
                >
                  ~{humanDuration(entry.estimate_seconds)}
                  {entry.estimate_measured ? "" : "?"}
                </span>
                {/* Reordering and dropping are for entries nothing has started
                    yet. One already handed to a worker has a job and soon a
                    file, and offering to shuffle it would be a promise the
                    server does not keep. */}
                <button
                  disabled={!waiting || index === 0}
                  title="Render this one earlier"
                  onClick={() => void moveQueued(entry.id, -1)}
                >
                  ▲
                </button>
                <button
                  disabled={!waiting || index === entries.length - 1}
                  title="Render this one later"
                  onClick={() => void moveQueued(entry.id, 1)}
                >
                  ▼
                </button>
                <button
                  className="target-list__remove"
                  title={
                    waiting
                      ? "Remove this from the queue"
                      : "Dismiss this entry. A render already running carries on — an encode cannot be safely abandoned part-way through its file."
                  }
                  onClick={() => void removeQueued(entry.id)}
                >
                  ×
                </button>
                {job && job.status === "running" && (
                  <span className="export-queue__bar">
                    <span style={{ width: `${Math.round(job.fraction * 100)}%` }} />
                  </span>
                )}
                {entry.error && <span className="export-screen__warning">{entry.error}</span>}
              </li>
            );
          })}
        </ul>
      )}
      {finished > 0 && (
        <button
          onClick={() => void clearFinishedQueued()}
          title="Dismiss the entries that have stopped being work. The files they produced stay under Renders below."
        >
          Clear {finished} finished
        </button>
      )}
      {entries.length > 0 && (
        <button
          className="export-screen__go"
          disabled={pending === 0}
          title={
            pending === 0
              ? "Nothing is waiting to be rendered"
              : `Render the ${pending} waiting ${pending === 1 ? "entry" : "entries"} in order, one at a time`
          }
          onClick={() => void runQueue()}
        >
          {queue?.running
            ? `Processing… (${pending} left)`
            : `Process render queue (${pending})`}
        </button>
      )}

      <div className="panel-section">
        <h4>Renders</h4>
        <span className="panel-section__note">Finished files</span>
      </div>
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
      <div className="panel-section">
        <h4>Data</h4>
        <span className="panel-section__note">Tracks and curves as numbers</span>
      </div>
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
