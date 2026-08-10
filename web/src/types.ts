// Shapes mirroring stable/server's JSON responses. Kept close to the Python
// side deliberately (see stable/server/serialize.py, schemas.py, jobs.py) so
// there is one obvious place to look when a field's meaning is unclear.

export type Channel = "x" | "y" | "r" | "s";
export const CHANNELS: Channel[] = ["x", "y", "r", "s"];

/**
 * One anchor: "on frame `frame`, this point is truly at `(x, y)`".
 *
 * `Query` is the name the project schema stores it under (`points[].queries`,
 * locked at v1) and the only place that word appears. Everywhere the operator
 * can see, this is an **anchor** — one word for one concept, which is the whole
 * of todo 6, seventh round.
 */
export interface Query {
  frame: number;
  x: number;
  y: number;
}

export interface VideoRef {
  path: string;
  fps: number;
  frames: number;
  w: number;
  h: number;
}

export interface ProxyTierInfo {
  ready: boolean;
  width?: number;
  height?: number;
  count?: number;
  fps?: number;
}

/**
 * The regions a point's tracker matches and searches, in source pixels.
 *
 * Carried on every point summary so the canvas can draw the pattern and search
 * boxes for all of them without a request each. `search_radius` is null for the
 * algorithms that have no search area (phase correlation measures a shift
 * without scanning), and `attribute` names the setting the number came from.
 */
export interface PointWindow {
  radius: number;
  attribute: string | null;
  search_radius: number | null;
  search_attribute: string | null;
}

export interface PointSummary {
  id: string;
  /** This point's own anchors. See `Query` for why the field is not `anchors`. */
  queries: Query[];
  tracker: string;
  enabled_for_solve: boolean;
  /**
   * The colour this track is drawn in, or null for the palette's answer.
   *
   * Stored on the point rather than in the tab: the palette has eight colours
   * for an unbounded number of tracks, so two of them can collide, and a
   * choice made to break that tie is worth keeping across a reload. Read it
   * through `colorFor` in the project context, never directly.
   */
  color: string | null;
  window: PointWindow;
  has_track: boolean;
  tracked_frames: number;
  mean_confidence: number;
  worst_frames: number[];
  /**
   * Whether the anchors have changed since this point was tracked.
   *
   * Anchors are the *input* to tracking, so adding, moving or removing one
   * leaves the stored track answering a question nobody is asking any more. The
   * card says so and offers the re-track rather than starting minutes of GPU
   * work on its own.
   */
  anchors_changed: boolean;
}

/**
 * The In/Out markers: the frames tracking and export are limited to (todo 1).
 *
 * `out` is exclusive, like every other range in this codebase. `whole_clip`
 * distinguishes "no limit set" from "the limits happen to be the ends of the
 * clip" — the timeline draws them differently.
 */
export interface ClipRangeInfo {
  in: number;
  out: number;
  frames: number;
  clip_frames: number;
  whole_clip: boolean;
}

export interface SolveSettings {
  reference_frame: number;
  channels: Record<Channel, boolean>;
}

export interface SmoothingFilter {
  channel: Channel;
  filter: string;
  params: Record<string, number>;
  enabled: boolean;
}

export interface RenderSettings {
  border: string;
  codec: string;
  crf: number;
}

export interface ProjectDetail {
  id: string;
  name: string;
  created_at: number;
  original_filename: string;
  video: VideoRef;
  points: PointSummary[];
  solve: SolveSettings;
  smoothing: SmoothingFilter[];
  render: RenderSettings;
  proxy: Record<string, ProxyTierInfo>;
  proxy_job_id?: string;
}

export interface ProjectMeta {
  id: string;
  name: string;
  created_at: number;
  original_filename: string;
}

/**
 * One anchor a tracking run reached and disagreed with.
 *
 * The run stopped there rather than drifting past the operator's own word about
 * where the feature is, so this is the "go and look at this" signal: the
 * timeline paints the anchor red and the card lists the distance.
 */
export interface AnchorViolation {
  frame: number;
  anchor: [number, number];
  tracked: [number, number];
  distance: number;
  origin: number;
  direction: "fwd" | "bwd";
}

export interface TrackData {
  point_id: string;
  n_frames: number;
  x: (number | null)[];
  y: (number | null)[];
  confidence: number[];
  visible: boolean[];
  disagreement: (number | null)[];
  contributions: number[];
  anchor_frames: number[];
  /**
   * Every position the operator has stated for this point: its own anchors plus
   * the ones its manual corrections were promoted to.
   *
   * On the shared type rather than re-declared by each component that reads it —
   * three of them had their own copy of this field under the server's older
   * name (`ground_truth`), which is how one list came to look like two.
   */
  anchors: Query[];
  /** Anchors a run reached and disagreed with, with the frame and the distance. */
  anchor_violations: AnchorViolation[];
  /** Just the frames of `anchor_violations`, for the timeline's red marks. */
  flagged_anchors: number[];
  /** The typed, non-destructive edit layer applied to this point, in order. */
  corrections: Correction[];
  /** Inclusive [start, end] runs of frames with no position, for one-click filling. */
  gaps: [number, number][];
  /** The untouched merged track, present only when enabled corrections exist. */
  raw_x: (number | null)[] | null;
  raw_y: (number | null)[] | null;
}

// -- the corrections layer (stable/core/corrections.py) -----------------------

export type CorrectionKind = "offset" | "replace" | "gap_fill" | "delete_range" | "retrack";

/**
 * One manual edit. The wire form *is* the project-file form, so these fields
 * mirror the dataclasses in stable/core/corrections.py exactly — see that
 * module for what each type does to a track.
 */
export interface Correction {
  id: string;
  kind: CorrectionKind;
  enabled: boolean;
  frame?: number;
  dx?: number;
  dy?: number;
  falloff_before?: number;
  falloff_after?: number;
  shape?: "linear" | "smooth";
  x?: number;
  y?: number;
  start?: number;
  end?: number;
  method?: "linear" | "spline";
  confidence?: number;
  direction?: "fwd" | "bwd" | "bidir";
  tracker?: string | null;
  source?: string | null;
}

/** What the undo/redo buttons render from (stable/server/history.py). */
export interface HistoryStatus {
  can_undo: boolean;
  can_redo: boolean;
  undo_label: string | null;
  redo_label: string | null;
  undo_depth: number;
  redo_depth: number;
}

export interface CorrectionsResponse {
  point: PointSummary;
  corrections: Correction[];
  history: HistoryStatus;
}

// -- the smoothing filter catalogue (stable/smoothing/filters.py) --------------

export interface FilterParamInfo {
  name: string;
  kind: "int" | "float";
  default: number;
  minimum: number | null;
  maximum: number | null;
  unit: string;
  description: string;
}

export interface FilterInfo {
  name: string;
  display_name: string;
  description: string;
  params: FilterParamInfo[];
}

/** One patch in the grid: where it is, and whether it fits inside the frame. */
export interface PointPatch {
  frame: number;
  x: number;
  y: number;
  /** `anchor` is a position you set; `tracked` is where the tracker put it. */
  kind: "anchor" | "tracked";
  /** False when the patch would fall off the edge of the frame at this size. */
  available: boolean;
}

/**
 * One per-point tracker setting the operator may change
 * (stable/server/tracker_options.py's `SETTINGS`).
 *
 * `attribute` is the plugin's own constructor argument and the key a PUT sends,
 * so `value`/`minimum`/`maximum` are all in *that argument's* units: a
 * half-size for the radii, a full side for LK's `win_size` (`unit: "side"`), a
 * plain count for the pyramid depth (`unit: "count"`). `radius` is the same
 * value expressed in source pixels from the point, which is what the canvas
 * draws and what the patch preview asks for; it is 0 when the setting is not a
 * length.
 */
export interface TrackerSetting {
  attribute: string;
  role: "window" | "search";
  label: string;
  value: number;
  default: number;
  minimum: number;
  maximum: number;
  step: number;
  unit: "radius" | "side" | "count";
  radius: number;
  help: string;
  /** Whether this point has overridden the plugin's own default. */
  stored: boolean;
}

/** `GET .../points/{id}/patches` — everything the patch panel draws (todo 14). */
export interface PointPatches {
  point_id: string;
  tracker: string;
  /** Half-size of the region this point's tracker actually matches. */
  radius: number;
  /** Side of the square patch in pixels, i.e. `2 * radius + 1`. */
  patch_size: number;
  /**
   * Which constructor argument `radius` came from (`template_radius`,
   * `win_size`, …), or null when the tracker has no local window at all and
   * the size is only a preview.
   */
  radius_source: string | null;
  /** Whether this tracker's pattern size is the operator's to set. */
  adjustable: boolean;
  /** One control per row, in order — the panel knows no tracker names. */
  settings: TrackerSetting[];
  /** The pattern region, in source pixels. */
  window: { radius: number; size_px: number; attribute: string | null; adjustable: boolean };
  /**
   * The search region, in source pixels. `radius` is null where the algorithm
   * has no search area at all, in which case `note` says why.
   */
  search: {
    radius: number | null;
    size_px: number | null;
    attribute: string | null;
    adjustable: boolean;
    note: string;
  };
  template_radius: number;
  search_radius: number;
  defaults: { template_radius: number; search_radius: number };
  template_radius_range: [number, number];
  search_radius_range: [number, number];
  /** Whether anything has been overridden, i.e. whether "reset" does anything. */
  custom: boolean;
  /** One entry per anchor. */
  patches: PointPatch[];
  /** The frame being viewed, when the point has a position there. */
  current: PointPatch | null;
}

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

/**
 * What a running job has produced *so far* (stable/server/jobs.py's
 * `JobProgress.publish`). Every field is optional because it depends on what
 * kind of job is running and how far it has got:
 *
 * - `frame`/`frame_min`/`frame_max`/`frames_read` come from the decoder itself
 *   (stable/video/telemetry.py) and so appear for any job that reads video.
 * - `track` is the path found so far, published per completed run by
 *   `LivePartialTrack` in stable/server/tracking.py. Its `x`/`y` are the full
 *   clip length with `null` for frames not yet covered — the same shape as a
 *   finished `TrackData`, so the viewer draws both with one code path.
 */
export interface JobPartial {
  frame?: number;
  frame_min?: number;
  frame_max?: number;
  frames_read?: number;
  runs_done?: number;
  runs_total?: number;
  track?: {
    x: (number | null)[];
    y: (number | null)[];
    n_frames: number;
  };
}

export interface JobEvent {
  id: string;
  project_id: string;
  kind: string;
  target: string | null;
  status: JobStatus;
  fraction: number;
  message: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: Record<string, any> | null;
  partial: JobPartial | null;
  error: string | null;
  updated_at: number;
}

export interface TrackerInfo {
  name: string;
  display_name: string;
  requires_gpu: boolean;
  license: string;
  available: boolean;
  /**
   * Null when the tracker can run; otherwise what to install to make it,
   * straight from the plugin (`TrackerPlugin.unavailable_reason`). Shown as the
   * tooltip on the greyed-out option, because "unavailable" alone is
   * indistinguishable from "broken".
   */
  unavailable_reason: string | null;
}

export interface DeviceInfo {
  torch: string | null;
  cuda: boolean;
  cuda_devices: string[];
  mps: boolean;
  selected: string;
}

// -- model checkpoint manager (stable/trackers/weights.py, server/routes/models.py) --

/**
 * One downloadable checkpoint and its cache/licence state. `pinned` false
 * means no verified sha256 is known yet (Phase 2's documented deviation: the
 * two shipped checkpoints ship unpinned and downloads fail closed on that);
 * `non_commercial` true means the licence needs an explicit opt-in too. Both
 * gates are per-download opt-ins here, not silent workarounds.
 */
export interface ModelInfo {
  key: string;
  filename: string;
  url: string;
  license: string;
  non_commercial: boolean;
  pinned: boolean;
  path: string;
  downloaded: boolean;
  notes: string;
  /** Names of installed tracker plugins that need this checkpoint. */
  trackers: string[];
}

export interface ModelsResponse {
  models: ModelInfo[];
  /** Whether the process-wide opt-in env vars are set (see stable/trackers/weights.py). */
  env: { allow_unpinned: boolean; accept_noncommercial: boolean };
}

export type SolveStatus =
  | "solved"
  | "translation_only"
  | "interpolated"
  | "held"
  | "identity";

/**
 * The solver's output: four pivot-relative curves plus its per-frame quality
 * record. `x`/`y` are how far the frame *centre* moved, not an origin-based
 * translation — see stable/solver/motion.py.
 */
export interface SolveCurves {
  x: number[];
  y: number[];
  r: number[];
  s: number[];
  /** Weighted RMS fit error in pixels; null on frames nothing was fitted for. */
  residual: (number | null)[];
  n_points: number[];
  n_inliers: number[];
  status: SolveStatus[];
  /** Frames worth the user's attention — the timeline's "go fix this" strip. */
  flagged: boolean[];
  reference_frame: number;
  pivot: [number, number];
  summary: {
    frames: number;
    flagged: number;
    mean_residual_px: number;
    max_residual_px: number;
    status_counts: Record<SolveStatus, number>;
  };
}

export interface SolveResponse {
  settings: SolveSettings;
  curves: SolveCurves | null;
}

/** Per-channel smoothing state, including the curve editor's range and pins. */
export interface ChannelSmoothingInfo {
  enabled: boolean;
  active: boolean;
  strength: number;
  filters: SmoothingFilter[];
  range: [number, number] | null;
  range_blend: number;
  pins: number[];
  pin_radius: number;
}

export interface SmoothedCurves {
  x: number[];
  y: number[];
  r: number[];
  s: number[];
  settings: Record<Channel, ChannelSmoothingInfo>;
  fps: number;
}

export interface SmoothingResponse {
  filters: SmoothingFilter[];
  curves: { raw: SolveCurves | null; smoothed: SmoothedCurves | null };
}

/**
 * One point's path after the smoothing's stabilisation warp (todo 3).
 *
 * Where the feature ends up once the shot is stabilised — the residual motion
 * the filters left behind. `null` on frames the point has no position for, the
 * same convention `TrackData` uses.
 */
export interface SmoothedTrack {
  point_id: string;
  x: (number | null)[];
  y: (number | null)[];
}

export interface SmoothedTracksResponse {
  n_frames: number;
  points: SmoothedTrack[];
}
