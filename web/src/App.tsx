import { useState } from "react";
import { CurveEditor } from "./components/CurveEditor";
import { EditPanel } from "./components/EditPanel";
import { ExportDialog } from "./components/ExportDialog";
import { ModelManager } from "./components/ModelManager";
import { PointsPanel } from "./components/PointsPanel";
import { ProjectPicker } from "./components/ProjectPicker";
import { SmoothingPanel } from "./components/SmoothingPanel";
import { SolvePanel } from "./components/SolvePanel";
import { Timeline } from "./components/Timeline";
import { Viewer } from "./components/Viewer";
import { ProjectProvider, useProjectContext } from "./state/ProjectContext";

type Tab = "points" | "edit" | "solve" | "smoothing";

const TABS: { id: Tab; label: string }[] = [
  { id: "points", label: "Points" },
  { id: "edit", label: "Edit" },
  { id: "solve", label: "Solve" },
  { id: "smoothing", label: "Smoothing" },
];

export default function App() {
  const [projectId, setProjectId] = useState<string | null>(null);

  if (!projectId) {
    return <ProjectPicker onOpen={setProjectId} />;
  }

  return (
    <ProjectProvider projectId={projectId} onClose={() => setProjectId(null)}>
      <Workspace onClose={() => setProjectId(null)} />
    </ProjectProvider>
  );
}

function Workspace({ onClose }: { onClose: () => void }) {
  const { project, stabilizedPreview, setStabilizedPreview, history, undo, redo } =
    useProjectContext();
  const [tab, setTab] = useState<Tab>("points");
  const [curvesOpen, setCurvesOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
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
        <button className="workspace__export" onClick={() => setExportOpen(true)}>
          Export
        </button>
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
          {tab === "points" && <PointsPanel />}
          {tab === "edit" && <EditPanel />}
          {tab === "solve" && <SolvePanel />}
          {tab === "smoothing" && <SmoothingPanel />}
        </aside>
      </div>

      {exportOpen && <ExportDialog onClose={() => setExportOpen(false)} />}
      {modelsOpen && <ModelManager onClose={() => setModelsOpen(false)} />}
    </div>
  );
}
