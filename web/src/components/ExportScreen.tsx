import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import { JobProgress } from "./JobProgress";

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
  map: "Hold chosen points at places you pick on the frame. Drag a crosshair to compose the shot; add a second to correct rotation and scale as well as position.",
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

/** Mirrors stable/render/options.py. The server validates; this only edits. */
interface RenderOptions {
  target: { mode: string; point_id: string | null; anchor_frame: number | null; pairs: TargetPair[] };
  crop: { scale: number | null; centre: string };
  fill: { mode: string; color: number[]; blur: number; tint: number; feather: number };
}

interface OptionsResponse {
  options: RenderOptions;
  target_modes: string[];
  crop_centres: string[];
  fill_modes: string[];
  points: { id: string; enabled_for_solve: boolean; frames: number }[];
}

/**
 * What `GET .../export/preview` answers with — planning only, no pixels, which
 * is what makes it cheap enough to re-ask while a target is being dragged.
 */
interface ExportPreview {
  frame: number;
  frames: number;
  source: { width: number; height: number };
  output: { width: number; height: number };
  /** The output rectangle's four corners, in **source** pixels. */
  rect: [number, number][];
  /** The stabilisation warp for this frame, target positions are stated after. */
  warp: Affine;
  coverage: number;
  crop_scale: number;
  zoom: number;
  gaps: boolean;
  border: string;
  max_displacement: number;
  is_no_op: boolean;
  no_op_detail: string | null;
  target: { mode: string; point_id: string | null; pairs: TargetPair[] };
  uses_rotation_and_scale: boolean;
  targets: {
    point_id: string;
    target: [number, number];
    landed: [number, number] | null;
    residual: number | null;
  }[];
}

interface TargetPair {
  point_id: string;
  x: number;
  y: number;
}

interface RenderedFile {
  filename: string;
  bytes: number;
  modified_at: number;
  url: string;
}

async function readJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // not JSON; the status text will have to do
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
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
 * The similarity the render applies to a frame, as the server sends it.
 *
 * Target destinations are stated in the space this lands in — *before* the
 * crop/zoom fit, which is only known once every frame's transform exists. So
 * placing a crosshair on the frame means inverting this, and dragging one means
 * applying it. Doing that with the output rectangle instead would fold the fit
 * in and put every target slightly off whenever the render crops.
 */
interface Affine {
  tx: number;
  ty: number;
  rotation: number;
  scale: number;
}

/** Where a source-pixel position lands after the stabilisation warp. */
function applyWarp(warp: Affine, x: number, y: number): { x: number; y: number } {
  const cos = Math.cos(warp.rotation) * warp.scale;
  const sin = Math.sin(warp.rotation) * warp.scale;
  return { x: cos * x - sin * y + warp.tx, y: sin * x + cos * y + warp.ty };
}

/** The inverse: which source pixel a warped position came from. */
function unapplyWarp(warp: Affine, x: number, y: number): { x: number; y: number } {
  const inverseScale = warp.scale === 0 ? 0 : 1 / warp.scale;
  const cos = Math.cos(-warp.rotation) * inverseScale;
  const sin = Math.sin(-warp.rotation) * inverseScale;
  const dx = x - warp.tx;
  const dy = y - warp.ty;
  return { x: cos * dx - sin * dy, y: sin * dx + cos * dy };
}

/**
 * The export screen: see what the settings mean, place the targets, and export.
 *
 * Two things the old dialog could not do. First, it described the output in
 * numbers — a crop scale and a coverage percentage — with no way to see what
 * survives into the frame; here the output rectangle is drawn *on the frame*,
 * re-planned as the settings change, so "how much am I losing" is a look rather
 * than an inference.
 *
 * Second, a stabilisation target could only be *picked* ("hold point p2 still"),
 * which pins it wherever it happened to sit. Here targets are *placed*: drag one
 * onto a thirds line and the shot is composed there. Two or more targets are a
 * different thing again — a similarity cannot generally satisfy two positions by
 * translation alone, so the fit uses rotation and scale, which is exactly how
 * mapping two tracked points onto two chosen places compensates for a rotating,
 * zooming camera. The screen says which of the two is in force, and reports the
 * per-pair miss, because with three or more the fit is least-squares and
 * pretending otherwise would be a lie about the result.
 *
 * Every export control lives here (todo 1, second round). They used to sit on a
 * separate "Video" tab, so composing the shot and exporting it were two screens
 * and the render library existed twice — "we have to go back to the Video tab
 * to do the export". The preview and the settings that drive it are now the
 * same screen, and there is one Export button, on it.
 */
export function ExportScreen() {
  const {
    project,
    refresh,
    currentFrame,
    setCurrentFrame,
    clipRange,
    tracks,
    loadTrack,
    latestJobFor,
  } = useProjectContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [preview, setPreview] = useState<ExportPreview | null>(null);
  const [opts, setOpts] = useState<OptionsResponse | null>(null);
  const [borders, setBorders] = useState<string[]>([]);
  const [renders, setRenders] = useState<RenderedFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showGuides, setShowGuides] = useState(true);
  const [sizeTick, setSizeTick] = useState(0);
  const image = useRef<HTMLImageElement | null>(null);
  const [imageTick, setImageTick] = useState(0);

  // Empty means "name it for me": the server stamps the time, so exports stop
  // overwriting each other (todo 7, first round).
  const [filename, setFilename] = useState("");
  const [codec, setCodec] = useState(project.render.codec || "h264");
  const [crf, setCrf] = useState(project.render.crf ?? 18);

  const job = latestJobFor("export");
  const tracked = project.points.filter((p) => p.has_track);
  const border = project.render.border;
  const ranged = clipRange && !clipRange.whole_clip ? clipRange : null;

  useEffect(() => {
    for (const point of tracked) if (!(point.id in tracks)) loadTrack(point.id);
  }, [tracked, tracks, loadTrack]);

  const loadPreview = useCallback(async () => {
    try {
      setPreview(
        await readJson<ExportPreview>(
          `/api/projects/${project.id}/export/preview?frame=${currentFrame}`,
        ),
      );
      setError(null);
    } catch (err) {
      setPreview(null);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [project.id, currentFrame]);

  const loadOptions = useCallback(() => {
    readJson<OptionsResponse>(`/api/projects/${project.id}/render/options`)
      .then(setOpts)
      .catch(() => setOpts(null));
  }, [project.id]);

  const loadRenders = useCallback(() => {
    readJson<RenderedFile[]>(`/api/projects/${project.id}/export/renders`)
      .then(setRenders)
      .catch(() => setRenders([]));
  }, [project.id]);

  useEffect(() => {
    void loadPreview();
  }, [loadPreview]);

  useEffect(() => {
    loadOptions();
    loadRenders();
  }, [loadOptions, loadRenders]);

  useEffect(() => {
    api
      .getBorderPolicies()
      .then(setBorders)
      .catch(() => setBorders(["crop", "auto_zoom", "transparent"]));
  }, []);

  // A finished render is a new file on disk, so re-read the list rather than
  // relying on a download link this component happens to be holding in state.
  useEffect(() => {
    if (job?.status === "done") {
      loadRenders();
      void loadPreview();
    }
  }, [job?.status, job?.updated_at, loadRenders, loadPreview]);

  // The frame underneath, from the proxy tier the timeline already uses.
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      image.current = img;
      setImageTick((n) => n + 1);
    };
    img.src = api.frameUrl(project.id, currentFrame);
  }, [project.id, currentFrame]);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const observer = new ResizeObserver(() => setSizeTick((n) => n + 1));
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  const view = useCallback(() => {
    const wrap = wrapRef.current;
    const width = preview?.source.width ?? project.video.w ?? 1;
    const height = preview?.source.height ?? project.video.h ?? 1;
    const dispW = wrap?.clientWidth ?? 1;
    const dispH = wrap?.clientHeight ?? 1;
    const scale = Math.min(dispW / width, dispH / height);
    return {
      scale,
      offsetX: (dispW - width * scale) / 2,
      offsetY: (dispH - height * scale) / 2,
      width,
      height,
    };
  }, [preview, project.video.w, project.video.h]);

  // -- draw ---------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const dispW = wrap.clientWidth || 1;
    const dispH = wrap.clientHeight || 1;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(dispW * dpr);
    canvas.height = Math.floor(dispH * dpr);
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, dispW, dispH);

    const v = view();
    const sx = (x: number) => v.offsetX + x * v.scale;
    const sy = (y: number) => v.offsetY + y * v.scale;

    if (image.current) {
      ctx.drawImage(image.current, v.offsetX, v.offsetY, v.width * v.scale, v.height * v.scale);
    }
    if (!preview) return;

    // Everything outside the output rectangle is what the export throws away.
    const rect = preview.rect;
    ctx.save();
    ctx.beginPath();
    ctx.rect(v.offsetX, v.offsetY, v.width * v.scale, v.height * v.scale);
    ctx.moveTo(sx(rect[0][0]), sy(rect[0][1]));
    for (const [x, y] of rect.slice(1)) ctx.lineTo(sx(x), sy(y));
    ctx.closePath();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fill("evenodd");
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = preview.gaps ? "#ffd166" : "#4da3ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(rect[0][0]), sy(rect[0][1]));
    for (const [x, y] of rect.slice(1)) ctx.lineTo(sx(x), sy(y));
    ctx.closePath();
    ctx.stroke();

    if (showGuides) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      // Thirds of the *output* rectangle, interpolated along its own edges —
      // exact, because a similarity keeps those edges parallel.
      const [tl, tr, br, bl] = rect;
      const along = (a: number[], b: number[], f: number) => [
        a[0] + f * (b[0] - a[0]),
        a[1] + f * (b[1] - a[1]),
      ];
      for (const f of [1 / 3, 2 / 3]) {
        const top = along(tl, tr, f);
        const bottom = along(bl, br, f);
        ctx.beginPath();
        ctx.moveTo(sx(top[0]), sy(top[1]));
        ctx.lineTo(sx(bottom[0]), sy(bottom[1]));
        ctx.stroke();
        const left = along(tl, bl, f);
        const right = along(tr, br, f);
        ctx.beginPath();
        ctx.moveTo(sx(left[0]), sy(left[1]));
        ctx.lineTo(sx(right[0]), sy(right[1]));
        ctx.stroke();
      }
    }
    ctx.restore();

    // Where each tracked point is on this frame.
    for (const point of tracked) {
      const track = tracks[point.id];
      const px = track?.x[currentFrame];
      const py = track?.y[currentFrame];
      if (px == null || py == null) continue;
      ctx.beginPath();
      ctx.arc(sx(px), sy(py), 4, 0, Math.PI * 2);
      ctx.fillStyle = pointColor(point.id);
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
    }

    // The targets themselves, drawn on the frame they would hold still.
    for (const pair of preview.target.pairs ?? []) {
      const at = unapplyWarp(preview.warp, pair.x, pair.y);
      const cx = sx(at.x);
      const cy = sy(at.y);
      ctx.save();
      ctx.strokeStyle = pointColor(pair.point_id);
      ctx.lineWidth = dragging === pair.point_id ? 3 : 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 14, cy);
      ctx.lineTo(cx + 14, cy);
      ctx.moveTo(cx, cy - 14);
      ctx.lineTo(cx, cy + 14);
      ctx.stroke();
      ctx.restore();
    }
  }, [preview, tracks, tracked, currentFrame, sizeTick, imageTick, dragging, showGuides, view]);

  // -- settings -----------------------------------------------------------------

  /**
   * Update the stored render options and re-plan.
   *
   * Everything on this screen goes through here, so a change to the crop and a
   * change to a target take the same route and the preview is never showing one
   * of them without the other.
   */
  const patchOptions = useCallback(
    async (patch: Record<string, unknown>) => {
      setBusy(true);
      try {
        setOpts(
          await readJson<OptionsResponse>(`/api/projects/${project.id}/render/options`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
          }),
        );
        await loadPreview();
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [project.id, loadPreview],
  );

  /** Store a new set of placed targets; an empty set is no target at all. */
  const putPairs = useCallback(
    (pairs: TargetPair[]) =>
      patchOptions({
        // `map` with no pairs is refused by the server (there would be nothing
        // to map), so removing the last one means going back to smoothing only.
        target: pairs.length ? { mode: "map", pairs } : { mode: "frame", pairs: [] },
      }),
    [patchOptions],
  );

  async function handleBorderChange(next: string) {
    setError(null);
    try {
      await api.updateRenderSettings(project.id, { border: next });
      await refresh();
      await loadPreview();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExport() {
    setError(null);
    try {
      await api.updateRenderSettings(project.id, { codec, crf });
      await api.exportVideo(project.id, { filename, codec, crf });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // -- interaction --------------------------------------------------------------

  function toSource(event: { clientX: number; clientY: number }) {
    const canvas = canvasRef.current!;
    const bounds = canvas.getBoundingClientRect();
    const v = view();
    return {
      x: (event.clientX - bounds.left - v.offsetX) / v.scale,
      y: (event.clientY - bounds.top - v.offsetY) / v.scale,
    };
  }

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!preview) return;
    const at = toSource(event);
    const v = view();
    const pairs = preview.target.pairs ?? [];
    const hit = pairs.find((pair) => {
      const p = unapplyWarp(preview.warp, pair.x, pair.y);
      return Math.hypot(p.x - at.x, p.y - at.y) * v.scale <= 14;
    });
    if (!hit) return;
    event.preventDefault();
    setDragging(hit.point_id);

    const move = (ev: MouseEvent) => {
      const moved = toSource(ev);
      const out = applyWarp(preview.warp, moved.x, moved.y);
      setPreview((prev) =>
        prev
          ? {
              ...prev,
              target: {
                ...prev.target,
                pairs: (prev.target.pairs ?? []).map((pair) =>
                  pair.point_id === hit.point_id ? { ...pair, x: out.x, y: out.y } : pair,
                ),
              },
            }
          : prev,
      );
    };
    const up = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      setDragging(null);
      const moved = toSource(ev);
      const out = applyWarp(preview.warp, moved.x, moved.y);
      void putPairs(
        pairs.map((pair) =>
          pair.point_id === hit.point_id ? { ...pair, x: out.x, y: out.y } : pair,
        ),
      );
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  /** Add a target for a point, starting where that point is right now. */
  function addTarget(pointId: string) {
    if (!preview) return;
    const track = tracks[pointId];
    const px = track?.x[currentFrame];
    const py = track?.y[currentFrame];
    // Start where the point already is: "hold it here", which is the common
    // case, and the operator drags from there if they want it elsewhere.
    const start =
      px != null && py != null
        ? applyWarp(preview.warp, px, py)
        : applyWarp(preview.warp, preview.source.width / 2, preview.source.height / 2);
    void putPairs([...(preview.target.pairs ?? []), { point_id: pointId, x: start.x, y: start.y }]);
  }

  function removeTarget(pointId: string) {
    if (!preview) return;
    void putPairs((preview.target.pairs ?? []).filter((pair) => pair.point_id !== pointId));
  }

  async function deleteRender(filename: string) {
    try {
      await readJson(`/api/projects/${project.id}/export/renders/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      loadRenders();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const target = opts?.options.target;
  const crop = opts?.options.crop;
  const fill = opts?.options.fill;
  const mode = target?.mode ?? "frame";
  const pairs = preview?.target.pairs ?? [];
  const paired = new Set(pairs.map((p) => p.point_id));
  const residuals = new Map(preview?.targets.map((row) => [row.point_id, row.residual]) ?? []);

  return (
    <div className="export-screen">
      <div ref={wrapRef} className="export-screen__canvas">
        <canvas ref={canvasRef} onMouseDown={handleMouseDown} />
        {!preview && (
          <div className="export-screen__placeholder">
            {error ?? "Solve and smooth this project to preview the export."}
          </div>
        )}
      </div>

      <div className="export-screen__side">
        {preview && (
          <>
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
              {/* What will actually be written. The In/Out markers bound the
                  render (todo 1, first round), and a clip length that is not
                  the one you expected is a slow thing to discover from the
                  finished file. */}
              <div>
                {ranged
                  ? `writes frames ${ranged.in}–${ranged.out - 1} (${ranged.frames} of ${ranged.clip_frames})`
                  : `writes the whole clip (${preview.frames} frames)`}
              </div>
              {preview.is_no_op && (
                <div className="export-screen__warning">{preview.no_op_detail}</div>
              )}
            </div>

            <div className="export-screen__frame-nav">
              <button onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}>◀</button>
              <span>
                frame {currentFrame} / {Math.max(preview.frames - 1, 0)}
              </span>
              <button
                onClick={() => setCurrentFrame(Math.min(preview.frames - 1, currentFrame + 1))}
              >
                ▶
              </button>
              <label className="channel-check">
                <input
                  type="checkbox"
                  checked={showGuides}
                  onChange={(e) => setShowGuides(e.target.checked)}
                />
                Thirds
              </label>
            </div>
          </>
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
                    e.target.value === "point"
                      ? (target?.point_id ?? tracked[0]?.id ?? null)
                      : null,
                },
              })
            }
            disabled={!opts || busy}
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
              onChange={(e) =>
                void patchOptions({ target: { mode: "point", point_id: e.target.value } })
              }
              disabled={busy}
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
              onChange={(e) =>
                void patchOptions({ target: { anchor_frame: Number(e.target.value) } })
              }
            />
            <span className="modal__hint">the framing that gets held</span>
          </div>
        )}
        <p className="export-screen__hint">{TARGET_HELP[mode] ?? ""}</p>

        {/* Placed targets. The list is shown for every mode so "+ target" is
            reachable — pressing it *is* the choice of `map`, the same way
            dragging a crosshair is — but the residuals only mean anything once
            the fit is actually using them. */}
        {tracked.length === 0 ? (
          <p className="panel__empty">Track a point to use it as a stabilisation target.</p>
        ) : (
          <ul className="export-screen__targets">
            {tracked.map((point) => {
              const residual = residuals.get(point.id);
              return (
                <li key={point.id}>
                  <span
                    className="points-list__swatch"
                    style={{ background: pointColor(point.id) }}
                  />
                  <span className="export-screen__target-id">{point.id}</span>
                  {paired.has(point.id) ? (
                    <>
                      <span className="export-screen__residual">
                        {residual == null
                          ? "no position on this frame"
                          : `${residual.toFixed(2)} px off`}
                      </span>
                      <button disabled={busy} onClick={() => removeTarget(point.id)}>
                        remove
                      </button>
                    </>
                  ) : (
                    <button disabled={busy || !preview} onClick={() => addTarget(point.id)}>
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
              : "One target: the render shifts the picture to hold this point still. Add a second to correct rotation and scale too."}
          </p>
        )}

        <h4>Crop</h4>
        <div className="field-row">
          <label>Size</label>
          <select
            value={crop?.scale == null ? "auto" : "manual"}
            onChange={(e) =>
              void patchOptions({
                crop: { scale: e.target.value === "auto" ? null : (crop?.scale ?? 0.9) },
              })
            }
            disabled={!opts || busy}
          >
            <option value="auto">automatic (minimal, no gaps)</option>
            <option value="manual">choose a size</option>
          </select>
          {crop?.scale != null && (
            <>
              <input
                type="range"
                min={0.2}
                max={1}
                step={0.01}
                value={crop.scale}
                onChange={(e) => void patchOptions({ crop: { scale: Number(e.target.value) } })}
              />
              <span>{Math.round(crop.scale * 100)}%</span>
            </>
          )}
        </div>
        <div className="field-row">
          <label>Centred on</label>
          <select
            value={crop?.centre ?? "frame"}
            onChange={(e) => void patchOptions({ crop: { centre: e.target.value } })}
            disabled={!opts || busy || mode === "frame"}
          >
            {(opts?.crop_centres ?? ["frame"]).map((c) => (
              <option key={c} value={c}>
                {c === "frame" ? "the frame centre" : "the stabilisation target"}
              </option>
            ))}
          </select>
        </div>
        <p className="export-screen__hint">
          Automatic keeps exactly as much as every frame covers, so there is nothing to fill.
          Keeping more than that exposes the edges — which is what the border fill is for.
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
            disabled={!opts || busy || border === "transparent"}
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
          {FILL_HELP[fill?.mode ?? "solid"] ?? ""} Blur, tint and feather stack on top of any mode,
          so a mirrored edge faded toward a colour is one setting away.
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
          Export stabilized video
        </button>
        {border === "transparent" && (
          <p className="export-screen__hint">
            No codec here carries an alpha channel. Switch to crop or auto_zoom — with a border
            fill the exposed edges are handled without needing alpha at all.
          </p>
        )}
        <JobProgress job={job} />
        {job?.status === "done" && job.result && (
          <p className="export-screen__hint">
            {job.result.width}×{job.result.height}
            {typeof job.result.frames === "number" ? ` · ${job.result.frames} frames` : ""}
            {typeof job.result.zoom === "number" ? ` · ${job.result.zoom.toFixed(3)}× zoom` : ""}
            {job.result.target ? ` · held ${job.result.target}` : ""}
            {job.result.gaps ? ` · edges filled (${job.result.fill})` : ""}
            {/* Audio only gets a mention when something happened to it — a
                plain passthrough is the expected case and says nothing. */}
            {job.result.audio_note ? ` · ${job.result.audio_note}` : ""}
          </p>
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
                <button onClick={() => deleteRender(file.filename)} title="Delete this render">
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        {error && <p className="export-screen__warning">{error}</p>}
      </div>
    </div>
  );
}
