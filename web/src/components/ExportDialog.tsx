import { useState } from "react";
import * as api from "../api/client";
import { useProjectContext } from "../state/ProjectContext";
import { ExportScreen } from "./ExportScreen";

/**
 * Export: the stabilized video, and the data behind it.
 *
 * This used to have three tabs — Data, Video, and a preview screen — with the
 * settings and the Export button on "Video" and the frame, the output rectangle
 * and the draggable targets on the preview. So composing a shot meant looking
 * at one tab and exporting it meant going back to another, and the list of
 * finished renders existed on both. Everything the Video tab did now lives on
 * the screen that shows what it does (todo 1, second round), and this component
 * is the shell around it: the export screen, and the data links, which are a
 * genuinely different job.
 */
export function ExportDialog({ onClose }: { onClose: () => void }) {
  const { project } = useProjectContext();
  const [tab, setTab] = useState<"export" | "data">("export");
  const hasSolve = project.points.some((p) => p.has_track);

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
          <button
            className={tab === "export" ? "button--active" : ""}
            onClick={() => setTab("export")}
          >
            Video
          </button>
          <button className={tab === "data" ? "button--active" : ""} onClick={() => setTab("data")}>
            Data
          </button>
        </div>

        {tab === "export" && <ExportScreen />}

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
      </div>
    </div>
  );
}
