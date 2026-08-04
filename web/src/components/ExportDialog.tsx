import { useEffect, useState } from "react";
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

/** Export dialog: data (tracks/curves, CSV or JSON) and stabilized video. */
export function ExportDialog({ onClose }: { onClose: () => void }) {
  const { project, refresh, latestJobFor } = useProjectContext();
  const [tab, setTab] = useState<"data" | "video">("data");
  const [filename, setFilename] = useState("stabilized.mp4");
  const [codec, setCodec] = useState(project.render.codec || "h264");
  const [crf, setCrf] = useState(project.render.crf ?? 18);
  const [borders, setBorders] = useState<string[]>([]);
  const [lastExported, setLastExported] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const job = latestJobFor("export");
  const hasSolve = project.points.some((p) => p.has_track);
  const border = project.render.border;

  useEffect(() => {
    api
      .getBorderPolicies()
      .then(setBorders)
      .catch(() => setBorders(["crop", "auto_zoom", "transparent"]));
  }, []);

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
      setLastExported(filename.endsWith(".mp4") ? filename : `${filename}.mp4`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

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
          <button className={tab === "video" ? "button--active" : ""} onClick={() => setTab("video")}>
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
            </div>
            <div className="field-row">
              <label>Quality (CRF)</label>
              <input
                type="number"
                min={0}
                max={51}
                value={crf}
                onChange={(e) => setCrf(Number(e.target.value))}
              />
            </div>
            <div className="field-row">
              <label>Border</label>
              <select value={border} onChange={(e) => void handleBorderChange(e.target.value)}>
                {borders.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <p className="modal__hint">{BORDER_HELP[border] ?? ""}</p>
            <button onClick={handleExportVideo} disabled={border === "transparent"}>
              Export stabilized video
            </button>
            {error && <div className="error-text">{error}</div>}
            <JobProgress job={job} />
            {job?.status === "done" && job.result && (
              <p className="modal__hint">
                {job.result.width}×{job.result.height}
                {typeof job.result.zoom === "number" ? ` · ${job.result.zoom.toFixed(3)}× zoom` : ""}
              </p>
            )}
            {job?.status === "done" && lastExported && (
              <div className="export-links">
                <a href={api.downloadUrl(project.id, lastExported)} download={lastExported}>
                  Download {lastExported}
                </a>
              </div>
            )}
            <p className="modal__hint">
              The stabilized render applies the solved + smoothed curves as one similarity per
              frame, with the source clip's audio passed through untouched. The border policy also
              drives the stabilized preview, so what you see is what you get.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
