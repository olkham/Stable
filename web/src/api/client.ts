// Thin REST client for the FastAPI server (stable/server). Every function
// mirrors one route in stable/server/routes/*.py — see those modules'
// docstrings for what each endpoint actually does server-side.
import type {
  ClipRangeInfo,
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
  RenderQueueState,
  RenderSettings,
  SmoothedTracksResponse,
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

/**
 * Drop a job that has not started yet.
 *
 * Only a queued one: work already running cannot be interrupted safely — an
 * encode half-way through a file is worse than one that finishes — so the
 * server answers 409 for it rather than pretending (todo 8, fourth round).
 */
export const cancelJob = (projectId: string, jobId: string): Promise<JobEvent> =>
  postJson(`/api/projects/${projectId}/jobs/${jobId}/cancel`);

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

/**
 * Change a point's tracker, its solve flag or its colour.
 *
 * An omitted field is left alone. `color: ""` is how the stored colour is
 * cleared back to the palette — `null` already means "leave it alone" here.
 */
export const updatePoint = (
  projectId: string,
  pointId: string,
  patch: { tracker?: string; enabled_for_solve?: boolean; color?: string },
): Promise<PointSummary> => patchJson(`/api/projects/${projectId}/points/${pointId}`, patch);

/** Which frames a derived path covers when its sources do not agree. */
export type DeriveSpan = "overlap" | "union";

/**
 * Create a new track from the per-frame mean of existing ones (todo 2, tenth round).
 *
 * On each frame the new path sits at the midpoint of the line joining the
 * sources on *that* frame. `mode` is what a length mismatch means: `overlap`
 * keeps only the frames every source covers, `union` keeps every frame any of
 * them covers.
 *
 * The response is an ordinary point summary with two extra keys saying what it
 * was made from — provenance the project file does not carry, so it is worth
 * showing while it is in hand.
 */
export const derivePoint = (
  projectId: string,
  pointIds: string[],
  mode: DeriveSpan = "overlap",
): Promise<PointSummary & { derived_from: string[]; derived_frames: [number, number] }> =>
  postJson(`/api/projects/${projectId}/points/derive`, { point_ids: pointIds, mode });

export const deletePoint = (projectId: string, pointId: string): Promise<{ status: string }> =>
  request(`/api/projects/${projectId}/points/${pointId}`, { method: "DELETE" });

/** Anchor a point at `(x, y)` on `frame`. An existing anchor there is replaced. */
export const addAnchor = (
  projectId: string,
  pointId: string,
  frame: number,
  x: number,
  y: number,
): Promise<PointSummary> =>
  postJson(`/api/projects/${projectId}/points/${pointId}/anchors`, { frame, x, y });

/**
 * Remove the anchor on `frame`, and whatever asserted it.
 *
 * An anchor can come from the point itself or from a manual correction that
 * states a position there (a drag, a directional run); the server removes both,
 * so a directional run's span goes away with the anchor it started from. 400
 * when there is no anchor on that frame, or when it is the point's last one.
 */
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

/**
 * Set one point's tracker sizes, keyed by the plugin's *own* argument names
 * (`template_radius`, `win_size`, `search_radius`, …) in the plugin's own
 * units — the shape `PointPatches.settings` describes. `null` clears an
 * override; a key the point's tracker does not accept is a 400 rather than a
 * silent no-op.
 */
export const updatePointPatches = (
  projectId: string,
  pointId: string,
  body: Record<string, number | null>,
): Promise<PointPatches> =>
  putJson(`/api/projects/${projectId}/points/${pointId}/patches`, body);

/**
 * URL of the patch at one frame — an anchor's, or wherever the tracker put it.
 *
 * `radius` previews a size that has not been saved yet, so dragging the slider
 * updates the grid live.
 *
 * `rev` busts the browser cache, because the pixels change when an edit moves
 * the track and that alone would not change the URL. Pass the *narrowest* thing
 * that identifies those pixels — the patch's own position, not a project-wide
 * edit counter. A counter re-fetches every thumbnail in the grid on every edit
 * anywhere, and at `source=original` each one is a full-resolution seek deep
 * into the clip; one correction cost twenty of them, which is most of why
 * editing a point felt slow (todo 2, fifth round).
 */
export function pointPatchUrl(
  projectId: string,
  pointId: string,
  frame: number,
  radius?: number,
  rev?: number | string,
  source?: "proxy" | "original",
): string {
  const size = radius === undefined ? "" : `&radius=${radius}`;
  const revision = rev === undefined ? "" : `&rev=${encodeURIComponent(String(rev))}`;
  // The server defaults to the proxy tier, which is what a 48px strip thumbnail
  // wants: a JPEG read instead of a full-resolution seek that would queue behind
  // the decoder while scrubbing. `original` is for judging texture.
  const tier = source === undefined ? "" : `&source=${source}`;
  return `/api/projects/${projectId}/points/${pointId}/patches/image?frame=${frame}${size}${revision}${tier}`;
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

/** How far a directional run may reach (stable/server/tracking.py). */
export interface RunBounds {
  /** Stop at the next anchor forwards / the previous one backwards. */
  stop_at_anchor?: boolean;
  /** Cover at most this many frames beyond the start frame. */
  max_frames?: number;
}

/**
 * Track one direction from one frame — "track forwards/backwards from here".
 *
 * `x`/`y` are optional: omitted, the run starts from wherever the point already
 * is on that frame, which is the anchor there when it has one. That is what the
 * anchor chips' ◀ / ▶ send, and it is why they work on a point that has never
 * been tracked at all.
 *
 * Unbounded it ploughs to the end of the clip, so callers pass a bound; the
 * panel's default is `stop_at_anchor`.
 */
export const retrackPoint = (
  projectId: string,
  pointId: string,
  opts: RunBounds & {
    frame: number;
    x?: number;
    y?: number;
    direction?: "fwd" | "bwd" | "bidir";
    tracker?: string;
  },
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

/**
 * Every tracked point's path after the smoothing's stabilisation warp — where
 * each feature ends up once the shot is stabilised (todo 3, fourth round).
 *
 * 409 before there is a solve and a smoothing to warp with, which callers read
 * as "nothing to draw yet" rather than as a failure.
 */
export const getSmoothedTracks = (projectId: string): Promise<SmoothedTracksResponse> =>
  request(`/api/projects/${projectId}/smoothing/tracks`);

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

/**
 * Render a stabilized video **now**. Omit `filename` (or send an empty one) and
 * the server names it for the time, so two exports are two files rather than
 * one overwriting the other.
 *
 * The panel uses `enqueueRender` instead (todo 6, tenth round): an export is
 * minutes of work, and lining several up with different settings before any of
 * them starts is the point of a queue. This is the one-off.
 */
export const exportVideo = (
  projectId: string,
  opts: { filename?: string | null; codec: string; crf: number },
): Promise<{ job_id: string }> =>
  postJson(`/api/projects/${projectId}/export/video`, {
    ...opts,
    filename: opts.filename ? opts.filename : null,
  });

// -- the render queue (todo 6, tenth round) ------------------------------------
//
// Five verbs over one stored list: add, read, reorder, drop, run. Each answers
// with the whole queue, so the panel never has to reconcile a local copy with
// what the server did — the same reasoning the target routes follow.

export const getRenderQueue = (projectId: string): Promise<RenderQueueState> =>
  request(`/api/projects/${projectId}/export/queue`);

/**
 * Capture the current export settings as a queued render, without starting it.
 *
 * Refuses exactly what an immediate render refuses — an unsolved project, a
 * transparent border, a render that would not move the picture — so a bad entry
 * is rejected while the settings that caused it are still on screen, rather
 * than minutes later at the front of the queue.
 */
export const enqueueRender = (
  projectId: string,
  opts: { filename?: string | null; codec: string; crf: number },
): Promise<RenderQueueState> =>
  postJson(`/api/projects/${projectId}/export/queue`, {
    ...opts,
    filename: opts.filename ? opts.filename : null,
  });

/** Put the pending entries in this order. Entries already running do not move. */
export const reorderRenderQueue = (
  projectId: string,
  ids: string[],
): Promise<RenderQueueState> =>
  putJson(`/api/projects/${projectId}/export/queue/order`, { ids });

export const removeQueuedRender = (
  projectId: string,
  entryId: string,
): Promise<RenderQueueState> =>
  request(`/api/projects/${projectId}/export/queue/${encodeURIComponent(entryId)}`, {
    method: "DELETE",
  });

/** Start working through the pending entries, one render at a time, in order. */
export const runRenderQueue = (
  projectId: string,
): Promise<RenderQueueState & { started: boolean }> =>
  postJson(`/api/projects/${projectId}/export/queue/run`);

export const deleteRender = (projectId: string, filename: string): Promise<{ status: string }> =>
  request(`/api/projects/${projectId}/export/renders/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });

export function downloadUrl(projectId: string, filename: string): string {
  return `/api/projects/${projectId}/export/download/${encodeURIComponent(filename)}`;
}

// -- the working range: In/Out markers (todo 1) --------------------------------

/** What tracking and export are limited to (`stable/server/clip_range.py`). */
export const getClipRange = (projectId: string): Promise<ClipRangeInfo> =>
  request(`/api/projects/${projectId}/range`);

/**
 * Move one marker, or clear the range.
 *
 * An omitted field leaves that marker alone (dragging one handle); sending both
 * as `null` goes back to the whole clip.
 */
export const updateClipRange = (
  projectId: string,
  body: { in?: number | null; out?: number | null },
): Promise<ClipRangeInfo> => putJson(`/api/projects/${projectId}/range`, body);
