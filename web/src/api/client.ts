// Thin REST client for the FastAPI server (stable/server). Every function
// mirrors one route in stable/server/routes/*.py — see those modules'
// docstrings for what each endpoint actually does server-side.
import type {
  Correction,
  CorrectionsResponse,
  DeviceInfo,
  FilterInfo,
  HistoryStatus,
  JobEvent,
  ModelInfo,
  ModelsResponse,
  PointSummary,
  ProjectDetail,
  ProjectMeta,
  RenderSettings,
  SmoothingFilter,
  SmoothingResponse,
  PointPatches,
  SolveResponse,
  TrackData,
  TrackerInfo,
} from "../types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function putJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function patchJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// -- meta ---------------------------------------------------------------

export const getTrackers = (): Promise<TrackerInfo[]> => request("/api/trackers");
export const getDevices = (): Promise<DeviceInfo> => request("/api/devices");
/** The smoothing catalogue with each filter's parameter schema — the UI builds
 *  its controls from this rather than hardcoding parameter names. */
export const getFilters = (): Promise<FilterInfo[]> => request("/api/filters");
export const getBorderPolicies = (): Promise<string[]> => request("/api/border-policies");

// -- model checkpoint manager ---------------------------------------------

export const listModels = (): Promise<ModelsResponse> => request("/api/models");

export const listModelJobs = (): Promise<JobEvent[]> => request("/api/models/jobs");

export const downloadModel = (
  key: string,
  opts: { acceptNoncommercial?: boolean; allowUnpinned?: boolean } = {},
): Promise<{ job_id: string }> =>
  postJson(`/api/models/${encodeURIComponent(key)}/download`, {
    accept_noncommercial: opts.acceptNoncommercial ?? false,
    allow_unpinned: opts.allowUnpinned ?? false,
  });

export const deleteModel = (key: string): Promise<ModelInfo & { removed: boolean }> =>
  request(`/api/models/${encodeURIComponent(key)}`, { method: "DELETE" });

// -- projects -------------------------------------------------------------

export const listProjects = (): Promise<ProjectMeta[]> => request("/api/projects");

export function createProject(file: File, name: string): Promise<ProjectDetail> {
  const form = new FormData();
  form.append("file", file);
  if (name) form.append("name", name);
  return request("/api/projects", { method: "POST", body: form });
}

export const getProject = (projectId: string): Promise<ProjectDetail> =>
  request(`/api/projects/${projectId}`);

export const renameProject = (projectId: string, name: string): Promise<ProjectDetail> =>
  patchJson(`/api/projects/${projectId}`, { name });

export const deleteProject = (projectId: string): Promise<{ status: string }> =>
  request(`/api/projects/${projectId}`, { method: "DELETE" });

export const regenerateProxy = (projectId: string): Promise<{ job_id: string }> =>
  postJson(`/api/projects/${projectId}/proxy`);

export function frameUrl(projectId: string, frame: number, tier = "half"): string {
  return `/api/projects/${projectId}/frames/${frame}?tier=${tier}`;
}

export function stabilizedFrameUrl(projectId: string, frame: number, tier = "half"): string {
  return `/api/projects/${projectId}/frames/${frame}/stabilized?tier=${tier}`;
}

export const listJobs = (projectId: string): Promise<JobEvent[]> =>
  request(`/api/projects/${projectId}/jobs`);

export const getJob = (projectId: string, jobId: string): Promise<JobEvent> =>
  request(`/api/projects/${projectId}/jobs/${jobId}`);

// -- points -----------------------------------------------------------------

export const listPoints = (projectId: string): Promise<PointSummary[]> =>
  request(`/api/projects/${projectId}/points`);

export const createPoint = (
  projectId: string,
  frame: number,
  x: number,
  y: number,
  tracker?: string,
): Promise<PointSummary> =>
  postJson(`/api/projects/${projectId}/points`, { frame, x, y, tracker });

export const updatePoint = (
  projectId: string,
  pointId: string,
  patch: { tracker?: string; enabled_for_solve?: boolean },
): Promise<PointSummary> => patchJson(`/api/projects/${projectId}/points/${pointId}`, patch);

export const deletePoint = (projectId: string, pointId: string): Promise<{ status: string }> =>
  request(`/api/projects/${projectId}/points/${pointId}`, { method: "DELETE" });

export const addAnchor = (
  projectId: string,
  pointId: string,
  frame: number,
  x: number,
  y: number,
): Promise<PointSummary> =>
  postJson(`/api/projects/${projectId}/points/${pointId}/anchors`, { frame, x, y });

export const deleteAnchor = (
  projectId: string,
  pointId: string,
  frame: number,
): Promise<PointSummary> =>
  request(`/api/projects/${projectId}/points/${pointId}/anchors/${frame}`, { method: "DELETE" });

export const trackPoint = (
  projectId: string,
  pointId: string,
  opts: { tracker?: string; direction?: "fwd" | "bwd" | "bidir" } = {},
): Promise<{ job_id: string }> =>
  postJson(`/api/projects/${projectId}/points/${pointId}/track`, opts);

// -- tracked patches (todo 14) -----------------------------------------------

/** `frame` adds the position the tracker has *now* to the returned list. */
export const getPointPatches = (
  projectId: string,
  pointId: string,
  frame?: number,
): Promise<PointPatches> =>
  request(
    `/api/projects/${projectId}/points/${pointId}/patches` +
      (frame === undefined ? "" : `?frame=${frame}`),
  );

export const updatePointPatches = (
  projectId: string,
  pointId: string,
  body: { template_radius?: number | null; search_radius?: number | null },
): Promise<PointPatches> =>
  putJson(`/api/projects/${projectId}/points/${pointId}/patches`, body);

/**
 * URL of the patch at one frame — a keyframe's, or wherever the tracker put it.
 *
 * `radius` previews a size that has not been saved yet, so dragging the slider
 * updates the grid live. `rev` busts the browser cache: the pixels change when
 * an edit moves the track, and that alone would not change the URL.
 */
export function pointPatchUrl(
  projectId: string,
  pointId: string,
  frame: number,
  radius?: number,
  rev?: number,
): string {
  const size = radius === undefined ? "" : `&radius=${radius}`;
  const revision = rev === undefined ? "" : `&rev=${rev}`;
  return `/api/projects/${projectId}/points/${pointId}/patches/image?frame=${frame}${size}${revision}`;
}

export const getTrack = (projectId: string, pointId: string): Promise<TrackData> =>
  request(`/api/projects/${projectId}/points/${pointId}/track`);

// -- manual editing: the corrections layer -------------------------------------

export const listCorrections = (
  projectId: string,
  pointId: string,
): Promise<CorrectionsResponse> =>
  request(`/api/projects/${projectId}/points/${pointId}/corrections`);

/**
 * Append one typed edit. The body is the correction's own project-file form
 * (`kind` plus that type's fields) — see stable/core/corrections.py.
 */
export const addCorrection = (
  projectId: string,
  pointId: string,
  correction: Partial<Correction> & { kind: string },
): Promise<CorrectionsResponse> =>
  postJson(`/api/projects/${projectId}/points/${pointId}/corrections`, correction);

export const setCorrectionEnabled = (
  projectId: string,
  pointId: string,
  correctionId: string,
  enabled: boolean,
): Promise<CorrectionsResponse> =>
  patchJson(
    `/api/projects/${projectId}/points/${pointId}/corrections/${correctionId}`,
    { enabled },
  );

export const deleteCorrection = (
  projectId: string,
  pointId: string,
  correctionId: string,
): Promise<CorrectionsResponse> =>
  request(`/api/projects/${projectId}/points/${pointId}/corrections/${correctionId}`, {
    method: "DELETE",
  });

export const clearCorrections = (
  projectId: string,
  pointId: string,
): Promise<CorrectionsResponse> =>
  request(`/api/projects/${projectId}/points/${pointId}/corrections`, { method: "DELETE" });

export const retrackPoint = (
  projectId: string,
  pointId: string,
  opts: { frame: number; x: number; y: number; direction?: "fwd" | "bwd" | "bidir"; tracker?: string },
): Promise<{ job_id: string }> =>
  postJson(`/api/projects/${projectId}/points/${pointId}/retrack`, opts);

// -- undo / redo ----------------------------------------------------------------

export const getHistory = (projectId: string): Promise<HistoryStatus> =>
  request(`/api/projects/${projectId}/edits/history`);

interface StepResult {
  applied: string | null;
  history: HistoryStatus;
  points: PointSummary[];
}

export const undoEdit = (projectId: string): Promise<StepResult> =>
  postJson(`/api/projects/${projectId}/edits/undo`);

export const redoEdit = (projectId: string): Promise<StepResult> =>
  postJson(`/api/projects/${projectId}/edits/redo`);

// -- solve / smoothing --------------------------------------------------------

export const updateSolveSettings = (
  projectId: string,
  patch: { reference_frame?: number; channels?: Partial<Record<string, boolean>> },
) => putJson(`/api/projects/${projectId}/solve/settings`, patch);

export const getSolve = (projectId: string): Promise<SolveResponse> =>
  request(`/api/projects/${projectId}/solve`);

export const runSolve = (projectId: string): Promise<{ job_id: string }> =>
  postJson(`/api/projects/${projectId}/solve`);

export const getSmoothing = (projectId: string): Promise<SmoothingResponse> =>
  request(`/api/projects/${projectId}/smoothing`);

/**
 * Replace the whole ordered filter list. It may carry curve-editor directives
 * (`@range`, `@pin`) alongside real filters — that is how a sub-range and the
 * pinned keyframes round-trip through the locked v1 project schema.
 */
export const updateSmoothing = (
  projectId: string,
  filters: SmoothingFilter[],
): Promise<SmoothingResponse> => putJson(`/api/projects/${projectId}/smoothing`, { filters });

export const updateRenderSettings = (
  projectId: string,
  patch: { border?: string; codec?: string; crf?: number },
): Promise<RenderSettings> => putJson(`/api/projects/${projectId}/render`, patch);

// -- export -------------------------------------------------------------------

/** `ae` is After Effects keyframe data (stable/core/trackdata.py), not a data dump. */
export function tracksExportUrl(projectId: string, format: "csv" | "json" | "ae"): string {
  return `/api/projects/${projectId}/export/tracks?format=${format}`;
}

export function curvesExportUrl(projectId: string, format: "csv" | "json"): string {
  return `/api/projects/${projectId}/export/curves?format=${format}`;
}

export const exportVideo = (
  projectId: string,
  opts: { filename: string; codec: string; crf: number },
): Promise<{ job_id: string }> => postJson(`/api/projects/${projectId}/export/video`, opts);

export function downloadUrl(projectId: string, filename: string): string {
  return `/api/projects/${projectId}/export/download/${encodeURIComponent(filename)}`;
}
