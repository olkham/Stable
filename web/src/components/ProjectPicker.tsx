import { useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import type { ProjectDetail, ProjectMeta } from "../types";
import { ModelManager } from "./ModelManager";

/**
 * Upload a video with real progress (todo 1).
 *
 * `api.createProject` (`web/src/api/client.ts`, owned by another workstream)
 * uses `fetch`, which has no cross-browser upload-progress event, so this
 * hits the same `POST /api/projects` multipart endpoint directly through
 * `XMLHttpRequest` instead of widening that file's contract for one caller.
 * Mirrors `request()`'s error handling: a JSON `detail` field wins, otherwise
 * the status text does.
 */
function uploadProjectWithProgress(
  file: File,
  name: string,
  onProgress: (fraction: number) => void,
): Promise<ProjectDetail> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (name) form.append("name", name);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/projects");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as ProjectDetail);
      } else {
        const detail =
          body && typeof body === "object" && "detail" in body && typeof (body as { detail: unknown }).detail === "string"
            ? (body as { detail: string }).detail
            : xhr.statusText || `upload failed (${xhr.status})`;
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error("upload failed — network error"));
    xhr.send(form);
  });
}

/** Landing screen: open an existing project or upload a video to start a new one. */
export function ProjectPicker({ onOpen }: { onOpen: (projectId: string) => void }) {
  const [projects, setProjects] = useState<ProjectMeta[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
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
    setProgress(0);
    setError(null);
    try {
      const project = await uploadProjectWithProgress(
        file,
        name || file.name.replace(/\.[^.]+$/, ""),
        setProgress,
      );
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
        {busy && (
          <div className="job-progress" style={{ width: "100%" }}>
            <div className="job-progress__bar" style={{ width: "100%" }}>
              <div
                className="job-progress__fill"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
            <span className="job-progress__label">{Math.round(progress * 100)}%</span>
          </div>
        )}
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
