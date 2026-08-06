import { useState } from "react";
import { CurveEditor } from "./components/CurveEditor";
import { ExportPanel } from "./components/ExportPanel";
import { ModelManager } from "./components/ModelManager";
import { TracksPanel } from "./components/TracksPanel";
import { ProjectPicker } from "./components/ProjectPicker";
import { SmoothingPanel } from "./components/SmoothingPanel";
import { SolvePanel } from "./components/SolvePanel";
import { Timeline } from "./components/Timeline";
import { Viewer } from "./components/Viewer";
import { ExportProvider } from "./state/ExportContext";
import { FrameStreamProvider } from "./state/FrameStreamContext";
import { ProjectProvider, useProjectContext } from "./state/ProjectContext";

/**
 * The sidebar's tabs.
 *
 * There was an "Edit" one (todo 4, sixth round). It is gone, not hidden: three
 * of the four things on it had grown a better home elsewhere — undo/redo is in
 * the header below, "retrack from here" is the bounded run on the track card,
 * and moving or deleting a point is something the canvas does directly — and
 * the fourth, a track's corrections and range repairs, is per point and now
 * lives on the point's own card. What was global about it, what a drag on the
 * frame *means*, is in the viewer's toolbar beside the tools that use it. If a
 * future editing feature needs a panel of its own, it can have one; until then
 * a tab that mostly duplicates two others is a tab that costs a click to learn
 * it had nothing new.
 */
type Tab = "points" | "solve" | "smoothing" | "export";

const TABS: { id: Tab; label: string }[] = [
  { id: "points", label: "Tracks" },
  { id: "solve", label: "Solve" },
  { id: "smoothing", label: "Smoothing" },
  { id: "export", label: "Export" },
];

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(null);

  if (!projectId) {
    return <ProjectPicker onOpen={setProjectId} />;
  }

  return (
    <ProjectProvider projectId={projectId} onClose={() => setProjectId(null)}>
      {/* The frame stream sits between the two: it follows the project's
          playhead, and the export reads each frame's geometry off the same
          message the pixels arrived in rather than asking for it separately
          (todo 2, fifth round). */}
      <FrameStreamProvider>
        <ExportProvider>
          <Workspace onClose={() => setProjectId(null)} />
        </ExportProvider>
      </FrameStreamProvider>
    </ProjectProvider>
  );
}

function Workspace({ onClose }: { onClose: () => void }) {
  const { project, stabilizedPreview, setStabilizedPreview, history, undo, redo } =
    useProjectContext();
  const [tab, setTab] = useState<Tab>("points");
  const [curvesOpen, setCurvesOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);

  return (
    <div className="workspace">
      <header className="workspace__header">
        <button className="link-button" onClick={onClose}>
          ← projects
        </button>
        <h1>{project.name}</h1>
        <div className="workspace__undo">
          <button
            onClick={() => void undo()}
            disabled={!history.can_undo}
            title={history.undo_label ? `Undo ${history.undo_label} (Ctrl+Z)` : "Nothing to undo"}
          >
            ↶
          </button>
          <button
            onClick={() => void redo()}
            disabled={!history.can_redo}
            title={
              history.redo_label ? `Redo ${history.redo_label} (Ctrl+Shift+Z)` : "Nothing to redo"
            }
          >
            ↷
          </button>
        </div>
        <label className="workspace__stabilized-toggle">
          <input
            type="checkbox"
            checked={stabilizedPreview}
            onChange={(e) => setStabilizedPreview(e.target.checked)}
          />
          Stabilized preview
        </label>
        <button
          className={curvesOpen ? "button--active" : ""}
          onClick={() => setCurvesOpen(!curvesOpen)}
        >
          Curves
        </button>
        <button onClick={() => setModelsOpen(true)}>Models</button>
      </header>

      <div className="workspace__body">
        <div className="workspace__main">
          <Viewer />
          <Timeline />
          {curvesOpen && <CurveEditor />}
        </div>
        <aside className="workspace__sidebar">
          <nav className="workspace__tabs">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                className={tab === id ? "button--active" : ""}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          {tab === "points" && <TracksPanel />}
          {tab === "solve" && <SolvePanel />}
          {tab === "smoothing" && <SmoothingPanel />}
          {/* The export is a tab, not a modal (todo 1, fourth round): its
              preview is the viewer to the left, so it must not cover it. */}
          {tab === "export" && <ExportPanel />}
        </aside>
      </div>

      {modelsOpen && <ModelManager onClose={() => setModelsOpen(false)} />}
    </div>
  );
}
