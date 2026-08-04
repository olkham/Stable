import { useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import type { ProjectMeta } from "../types";
import { ModelManager } from "./ModelManager";

/** Landing screen: open an existing project or upload a video to start a new one. */
export function ProjectPicker({ onOpen }: { onOpen: (projectId: string) => void }) {
  const [projects, setProjects] = useState<ProjectMeta[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [modelsOpen, setModelsOpen] = useState(false);

  const refresh = () => api.listProjects().then(setProjects).catch(() => undefined);

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload() {
    const file = fileInput.current?.files?.[0];
    if (!file) {
      setError("choose a video file first");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(file, name || file.name.replace(/\.[^.]+$/, ""));
      onOpen(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(projectId: string, evt: React.MouseEvent) {
    evt.stopPropagation();
    if (!confirm("Delete this project? This cannot be undone.")) return;
    await api.deleteProject(projectId);
    refresh();
  }

  return (
    <div className="picker">
      <h1>Stable</h1>
      <p className="picker__subtitle">Point tracking &amp; 2D stabilization</p>
      <button className="link-button" onClick={() => setModelsOpen(true)}>
        Manage model downloads →
      </button>

      <section className="picker__panel">
        <h2>New project</h2>
        <input ref={fileInput} type="file" accept="video/*" disabled={busy} />
        <input
          type="text"
          placeholder="Project name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={busy}
        />
        <button onClick={handleUpload} disabled={busy}>
          {busy ? "Uploading…" : "Upload & create"}
        </button>
        {error && <div className="error-text">{error}</div>}
      </section>

      {projects.length > 0 && (
        <section className="picker__panel">
          <h2>Existing projects</h2>
          <ul className="picker__list">
            {projects.map((p) => (
              <li key={p.id} className="picker__item" onClick={() => onOpen(p.id)}>
                <span className="picker__item-name">{p.name}</span>
                <span className="picker__item-file">{p.original_filename}</span>
                <button className="picker__delete" onClick={(e) => handleDelete(p.id, e)}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {modelsOpen && <ModelManager onClose={() => setModelsOpen(false)} />}
    </div>
  );
}
