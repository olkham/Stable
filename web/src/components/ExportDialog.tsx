import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
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
  target: { mode: string; point_id: string | null; anchor_frame: number | null };
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

interface RenderedFile {
  filename: string;
  bytes: number;
  modified_at: number;
}

// These endpoints live in stable/server/routes/{solve,export}.py. They are
// fetched directly rather than through api/client.ts so the whole stabilisation
// -options surface stays in one file.
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

function putJson<T>(path: string, body: unknown): Promise<T> {
  return readJson<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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

function humanBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} kB`;
  return `${bytes} B`;
}

/** Export dialog: data (tracks/curves, CSV or JSON) and stabilized video. */
export function ExportDialog({ onClose }: { onClose: () => void }) {
  const { project, refresh, latestJobFor } = useProjectContext();
  const [tab, setTab] = useState<"data" | "video">("data");
  const [filename, setFilename] = useState("stabilized.mp4");
  const [codec, setCodec] = useState(project.render.codec || "h264");
  const [crf, setCrf] = useState(project.render.crf ?? 18);
  const [borders, setBorders] = useState<string[]>([]);
  const [opts, setOpts] = useState<OptionsResponse | null>(null);
  const [renders, setRenders] = useState<RenderedFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  const job = latestJobFor("export");
  const hasSolve = project.points.some((p) => p.has_track);
  const border = project.render.border;

  const loadRenders = useCallback(() => {
    readJson<RenderedFile[]>(`/api/projects/${project.id}/export/renders`)
      .then(setRenders)
      .catch(() => setRenders([]));
  }, [project.id]);

  useEffect(() => {
    api
      .getBorderPolicies()
      .then(setBorders)
      .catch(() => setBorders(["crop", "auto_zoom", "transparent"]));
  }, []);

  useEffect(() => {
    readJson<OptionsResponse>(`/api/projects/${project.id}/render/options`)
      .then(setOpts)
      .catch(() => setOpts(null));
    loadRenders();
  }, [project.id, loadRenders]);

  // A finished render is a new file on disk, so re-read the list rather than
  // relying on a download link this component happens to be holding in state.
  useEffect(() => {
    if (job?.status === "done") loadRenders();
  }, [job?.status, job?.updated_at, loadRenders]);

  async function patchOptions(patch: Record<string, unknown>) {
    setError(null);
    try {
      setOpts(await putJson<OptionsResponse>(`/api/projects/${project.id}/render/options`, patch));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleBorderChange(next: string) {
    setError(null);
    try {
      await api.updateRenderSettings(project.id, { border: next });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExportVideo() {
    setError(null);
    try {
      await api.updateRenderSettings(project.id, { codec, crf });
      await api.exportVideo(project.id, { filename, codec, crf });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const target = opts?.options.target;
  const crop = opts?.options.crop;
  const fill = opts?.options.fill;
  const trackedPoints = opts?.points ?? [];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2>Export</h2>
          <button className="link-button" onClick={onClose}>
            close
          </button>
        </div>
        <div className="modal__tabs">
          <button className={tab === "data" ? "button--active" : ""} onClick={() => setTab("data")}>
            Data
          </button>
          <button
            className={tab === "video" ? "button--active" : ""}
            onClick={() => setTab("video")}
          >
            Video
          </button>
        </div>

        {tab === "data" && (
          <div className="modal__body">
            <p>Point tracks (raw + corrections applied):</p>
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
            <p>Solved + smoothed X/Y/R/S curves:</p>
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
        )}

        {tab === "video" && (
          <div className="modal__body">
            <div className="field-row">
              <label>Filename</label>
              <input value={filename} onChange={(e) => setFilename(e.target.value)} />
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
              <label>Quality (CRF)</label>
              <input
                type="number"
                min={0}
                max={51}
                value={crf}
                onChange={(e) => setCrf(Number(e.target.value))}
              />
            </div>

            <h4>Stabilise on</h4>
            <div className="field-row">
              <label>Target</label>
              <select
                value={target?.mode ?? "frame"}
                onChange={(e) =>
                  void patchOptions({
                    target: {
                      mode: e.target.value,
                      point_id:
                        e.target.value === "point"
                          ? (target?.point_id ?? trackedPoints[0]?.id ?? null)
                          : null,
                    },
                  })
                }
                disabled={!opts}
              >
                {(opts?.target_modes ?? ["frame"]).map((mode) => (
                  <option key={mode} value={mode}>
                    {mode === "frame"
                      ? "smoothing only"
                      : mode === "centroid"
                        ? "average of all points"
                        : "a single point"}
                  </option>
                ))}
              </select>
              {target?.mode === "point" && (
                <select
                  value={target.point_id ?? ""}
                  onChange={(e) =>
                    void patchOptions({ target: { mode: "point", point_id: e.target.value } })
                  }
                >
                  {trackedPoints.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.id} ({p.frames} frames)
                    </option>
                  ))}
                </select>
              )}
            </div>
            {target && target.mode !== "frame" && (
              <div className="field-row">
                <label>Anchor frame</label>
                <input
                  type="number"
                  min={0}
                  max={Math.max(project.video.frames - 1, 0)}
                  value={target.anchor_frame ?? project.solve.reference_frame}
                  onChange={(e) =>
                    void patchOptions({
                      target: { ...target, anchor_frame: Number(e.target.value) },
                    })
                  }
                />
                <span className="modal__hint">the framing that gets held</span>
              </div>
            )}
            <p className="modal__hint">{TARGET_HELP[target?.mode ?? "frame"] ?? ""}</p>
            {trackedPoints.length === 0 && (
              <p className="panel__empty">Track a point to use it as a stabilisation target.</p>
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
                disabled={!opts}
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
                  <span>{Math.round(crop.scale * 100)}% of frame</span>
                </>
              )}
            </div>
            <div className="field-row">
              <label>Centred on</label>
              <select
                value={crop?.centre ?? "frame"}
                onChange={(e) => void patchOptions({ crop: { centre: e.target.value } })}
                disabled={!opts || target?.mode === "frame"}
              >
                {(opts?.crop_centres ?? ["frame"]).map((c) => (
                  <option key={c} value={c}>
                    {c === "frame" ? "the frame centre" : "the stabilisation target"}
                  </option>
                ))}
              </select>
            </div>
            <p className="modal__hint">
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
            <p className="modal__hint">{BORDER_HELP[border] ?? ""}</p>
            <div className="field-row">
              <label>Fill exposed edge</label>
              <select
                value={fill?.mode ?? "solid"}
                onChange={(e) => void patchOptions({ fill: { mode: e.target.value } })}
                disabled={!opts || border === "transparent"}
              >
                {(opts?.fill_modes ?? ["solid"]).map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
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
            <p className="modal__hint">
              {FILL_HELP[fill?.mode ?? "solid"] ?? ""} Blur, tint and feather stack on top of any
              mode, so a mirrored edge faded toward a colour is one setting away.
            </p>

            <button onClick={handleExportVideo} disabled={border === "transparent"}>
              Export stabilized video
            </button>
            {border === "transparent" && (
              <p className="modal__hint">
                No codec here carries an alpha channel. Switch to crop or auto_zoom — with a border
                fill the exposed edges are handled without needing alpha at all.
              </p>
            )}
            {error && <div className="error-text">{error}</div>}
            <JobProgress job={job} />
            {job?.status === "done" && job.result && (
              <p className="modal__hint">
                {job.result.width}×{job.result.height}
                {typeof job.result.zoom === "number" ? ` · ${job.result.zoom.toFixed(3)}× zoom` : ""}
                {job.result.target ? ` · held ${job.result.target}` : ""}
                {job.result.gaps ? ` · edges filled (${job.result.fill})` : ""}
              </p>
            )}

            {renders.length > 0 && (
              <>
                <h4>Rendered</h4>
                <div className="export-links">
                  {renders.map((file) => (
                    <a
                      key={file.filename}
                      href={api.downloadUrl(project.id, file.filename)}
                      download={file.filename}
                    >
                      {file.filename} ({humanBytes(file.bytes)})
                    </a>
                  ))}
                </div>
              </>
            )}

            <p className="modal__hint">
              The stabilized render applies the solved + smoothed curves as one similarity per
              frame, with the source clip's audio passed through untouched. Every option here also
              drives the stabilized preview, so what you see is what you get.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
