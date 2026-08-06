import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { FrameGeometry } from "../api/frameStream";
import { useFrameStream } from "./FrameStreamContext";
import { useProjectContext } from "./ProjectContext";

/** Mirrors stable/render/options.py. The server validates; this only edits. */
export interface RenderOptions {
  target: {
    mode: string;
    point_id: string | null;
    anchor_frame: number | null;
    pairs: TargetPair[];
  };
  crop: { scale: number | null; centre: string };
  fill: { mode: string; color: number[]; blur: number; tint: number; feather: number };
}

export interface OptionsResponse {
  options: RenderOptions;
  target_modes: string[];
  crop_centres: string[];
  fill_modes: string[];
  points: { id: string; enabled_for_solve: boolean; frames: number }[];
}

export interface TargetPair {
  point_id: string;
  /** Destination x in the space the stabilisation warp lands in. */
  x: number;
  y: number;
}

/**
 * The similarity the render applies to a frame, as the server sends it.
 *
 * Target destinations are stated in the space this lands in — *before* the
 * crop/zoom fit, which is only known once every frame's transform exists. So
 * placing a crosshair on the frame means inverting this, and dragging one means
 * applying it. Doing that with the output rectangle instead would fold the fit
 * in and put every target slightly off whenever the render crops.
 */
export interface Affine {
  tx: number;
  ty: number;
  rotation: number;
  scale: number;
}

/**
 * What `GET .../export/preview` answers with — planning only, no pixels, which
 * is what makes it cheap enough to re-ask while a target is being dragged.
 *
 * Only the whole-clip half of it is read here now (todo 2, fifth round): output
 * size, coverage, zoom, `is_no_op` and so on do not depend on `frame`, so this
 * is asked once per settings change rather than once per frame of every scrub.
 * The per-frame half — `rect`, `warp`, `targets` — comes off the frame stream,
 * in the same message as the pixels it is drawn over, which is the only way to
 * be sure the two are the same frame. `frameGeometry` below is that, with this
 * as the fallback for the moment before the first frame arrives (and for as
 * long as the socket is down).
 */
export interface ExportPreview {
  frame: number;
  frames: number;
  source: { width: number; height: number };
  output: { width: number; height: number };
  /** The output rectangle's four corners, in **source** pixels. */
  rect: [number, number][];
  /** The stabilisation warp for this frame, target positions are stated after. */
  warp: Affine;
  /** Source pixels → output pixels, i.e. the same warp with the fit composed in. */
  output_warp: Affine;
  coverage: number;
  crop_scale: number;
  zoom: number;
  gaps: boolean;
  border: string;
  max_displacement: number;
  is_no_op: boolean;
  no_op_detail: string | null;
  target: { mode: string; point_id: string | null; pairs: TargetPair[] };
  uses_rotation_and_scale: boolean;
  targets: {
    point_id: string;
    target: [number, number];
    landed: [number, number] | null;
    residual: number | null;
  }[];
}

export interface RenderedFile {
  filename: string;
  bytes: number;
  modified_at: number;
  url: string;
}

/** Where a source-pixel position lands after the stabilisation warp. */
export function applyWarp(warp: Affine, x: number, y: number): { x: number; y: number } {
  const cos = Math.cos(warp.rotation) * warp.scale;
  const sin = Math.sin(warp.rotation) * warp.scale;
  return { x: cos * x - sin * y + warp.tx, y: sin * x + cos * y + warp.ty };
}

/** The inverse: which source pixel a warped position came from. */
export function unapplyWarp(warp: Affine, x: number, y: number): { x: number; y: number } {
  const inverseScale = warp.scale === 0 ? 0 : 1 / warp.scale;
  const cos = Math.cos(-warp.rotation) * inverseScale;
  const sin = Math.sin(-warp.rotation) * inverseScale;
  const dx = x - warp.tx;
  const dy = y - warp.ty;
  return { x: cos * dx - sin * dy, y: sin * dx + cos * dy };
}

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

interface ExportContextValue {
  /** The stored options and the valid choices, or null before the first load. */
  opts: OptionsResponse | null;
  /**
   * The placed stabilisation targets.
   *
   * Read from the *options*, never from the preview (todo 7, fourth round).
   * They used to come off `preview.target.pairs`, so any preview that failed or
   * had not come back yet made the panel believe there were no targets: the
   * crosshairs vanished and both rows flipped back to "+ target" while the
   * server still held them. The preview is geometry; the options are truth.
   */
  pairs: TargetPair[];
  /** The planned geometry for the whole clip — output size, coverage, zoom. */
  preview: ExportPreview | null;
  /**
   * The geometry of the frame **currently on the canvas**: its output
   * rectangle, its stabilisation warp and where each target lands.
   *
   * Off the frame stream, so it cannot be a frame ahead of the picture. Falls
   * back to the preview's own per-frame block when the socket is down, which is
   * the pre-stream behaviour.
   */
  frameGeometry: FrameGeometry | null;
  error: string | null;
  /** True while a change is being saved; a hint, never a reason to disable. */
  busy: boolean;
  /** Whether the Export tab is showing, i.e. whether the viewer draws its overlay. */
  active: boolean;
  setActive: (on: boolean) => void;
  showGuides: boolean;
  setShowGuides: (on: boolean) => void;
  /**
   * Whether the viewer shows the export's output beside the frame (todo 1,
   * fifth round). Kept here rather than in the frame stream so switching tabs
   * away and back does not forget the operator's choice; the panel is what
   * turns the stream's `output` variant on and off from it.
   */
  sideBySide: boolean;
  setSideBySide: (on: boolean) => void;
  /** Which target is being dragged on the canvas, for the viewer to highlight. */
  dragging: string | null;
  setDragging: (pointId: string | null) => void;
  /** Move one target while the pointer is down, without a round trip per pixel. */
  nudgeTarget: (pointId: string, x: number, y: number) => void;

  patchOptions: (patch: Record<string, unknown>) => Promise<void>;
  putPairs: (pairs: TargetPair[]) => Promise<void>;
  reloadPreview: () => Promise<void>;
  reloadOptions: () => void;

  renders: RenderedFile[];
  reloadRenders: () => void;
  deleteRender: (filename: string) => Promise<void>;
}

const ExportContext = createContext<ExportContextValue | null>(null);

export function useExportContext(): ExportContextValue {
  const ctx = useContext(ExportContext);
  if (!ctx) throw new Error("useExportContext must be used inside <ExportProvider>");
  return ctx;
}

/**
 * The export's own state, shared by the panel that edits it and the viewer that
 * draws it (todo 1, fourth round).
 *
 * The export used to be a modal with a canvas of its own — its own resize
 * observer, its own view transform, its own frame stepping — so composing a
 * shot meant working on a second copy of the picture that could not zoom, could
 * not pan, had no timeline and hid the real one behind it. Everything it drew
 * comes from two planning calls and the frame image, none of which needs a
 * canvas; so the state lives here, the controls live in a sidebar panel, and
 * the drawing happens on the viewer everything else already uses.
 */
export function ExportProvider({ children }: { children: ReactNode }) {
  const { project, currentFrame, smoothingRevision, editRevision, jobs } = useProjectContext();
  const { geometry: streamedGeometry, streaming, invalidateOutput } = useFrameStream();
  const [opts, setOpts] = useState<OptionsResponse | null>(null);
  const [preview, setPreview] = useState<ExportPreview | null>(null);
  const [renders, setRenders] = useState<RenderedFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState(false);
  const [showGuides, setShowGuides] = useState(true);
  const [sideBySide, setSideBySide] = useState(true);
  const [dragging, setDragging] = useState<string | null>(null);
  // Where a target has been dragged to but not yet stored, so the crosshair can
  // follow the pointer at screen rate while the server hears about it once.
  const [nudged, setNudged] = useState<Record<string, TargetPair>>({});

  // Out-of-order guard. Every option change re-plans, and a plan is O(T); under
  // a slider drag the answers do not come back in the order they were asked
  // for, and installing a stale one would silently undo the newest edit.
  const optionsSeq = useRef(0);
  const previewSeq = useRef(0);
  // Which frame to plan for, read rather than depended on: the whole-clip
  // answer is the same for every frame, so keying the request on the playhead
  // is what produced one `/export/preview` per frame of every scrub (todo 2,
  // fifth round). The frame only still matters on the fallback path below.
  const frameRef = useRef(currentFrame);
  frameRef.current = currentFrame;

  const reloadPreview = useCallback(async () => {
    const seq = ++previewSeq.current;
    try {
      const next = await readJson<ExportPreview>(
        `/api/projects/${project.id}/export/preview?frame=${frameRef.current}`,
      );
      if (seq !== previewSeq.current) return;
      setPreview(next);
      setError(null);
    } catch (err) {
      if (seq !== previewSeq.current) return;
      // The geometry is unknown; the settings are not. Losing the preview must
      // never look like losing the targets (todo 7).
      setPreview(null);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [project.id]);

  const reloadOptions = useCallback(() => {
    const seq = ++optionsSeq.current;
    readJson<OptionsResponse>(`/api/projects/${project.id}/render/options`)
      .then((next) => {
        if (seq === optionsSeq.current) setOpts(next);
      })
      .catch(() => undefined);
  }, [project.id]);

  const reloadRenders = useCallback(() => {
    readJson<RenderedFile[]>(`/api/projects/${project.id}/export/renders`)
      .then(setRenders)
      .catch(() => setRenders([]));
  }, [project.id]);

  useEffect(() => {
    reloadOptions();
    reloadRenders();
  }, [reloadOptions, reloadRenders]);

  // Only plan while the export is on screen: it is the one call in this app
  // that costs an O(T) fit, and nothing else shows its answer. Notably *not*
  // keyed on the playhead — see `reloadPreview`.
  useEffect(() => {
    if (active) void reloadPreview();
  }, [active, reloadPreview, smoothingRevision, editRevision]);

  // The fallback. With no socket there is no per-frame geometry riding with the
  // pixels, so the overlay has to come from somewhere: this is what the screen
  // did before the stream existed, kept for exactly as long as the socket is
  // down. Note this is also the *only* remaining per-frame preview request.
  useEffect(() => {
    if (active && !streaming) void reloadPreview();
  }, [active, streaming, currentFrame, reloadPreview]);

  // A finished export is a new file on disk, so re-read the list rather than
  // trusting a download link this component happens to be holding in state.
  const exportJobs = useMemo(
    () =>
      Object.values(jobs)
        .filter((job) => job.kind === "export")
        .sort((a, b) => b.updated_at - a.updated_at),
    [jobs],
  );
  const lastDone = exportJobs.find((job) => job.status === "done")?.updated_at;
  useEffect(() => {
    if (lastDone !== undefined) reloadRenders();
  }, [lastDone, reloadRenders]);

  /**
   * Update the stored render options and re-plan.
   *
   * Everything on the export screen goes through here, so a change to the crop
   * and a change to a target take the same route and the preview is never
   * showing one of them without the other.
   */
  const patchOptions = useCallback(
    async (patch: Record<string, unknown>) => {
      const seq = ++optionsSeq.current;
      setBusy(true);
      try {
        const next = await readJson<OptionsResponse>(
          `/api/projects/${project.id}/render/options`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
          },
        );
        if (seq !== optionsSeq.current) return;
        setOpts(next);
        setNudged({});
        setError(null);
        // New settings are new pixels: every rendered output frame in hand was
        // produced from the old ones. The server stamps them so a stale one can
        // never be *drawn* as current, but dropping them here is what stops the
        // side-by-side canvas sitting on the previous crop until the next frame
        // arrives.
        invalidateOutput();
        await reloadPreview();
      } catch (err) {
        if (seq === optionsSeq.current) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (seq === optionsSeq.current) setBusy(false);
      }
    },
    [project.id, reloadPreview, invalidateOutput],
  );

  const putPairs = useCallback(
    (pairs: TargetPair[]) =>
      patchOptions({
        // `map` with no pairs is refused by the server (there would be nothing
        // to map), so removing the last one means going back to smoothing only.
        target: pairs.length ? { mode: "map", pairs } : { mode: "frame", pairs: [] },
      }),
    [patchOptions],
  );

  const nudgeTarget = useCallback((pointId: string, x: number, y: number) => {
    setNudged((prev) => ({ ...prev, [pointId]: { point_id: pointId, x, y } }));
  }, []);

  const deleteRender = useCallback(
    async (filename: string) => {
      try {
        await readJson(
          `/api/projects/${project.id}/export/renders/${encodeURIComponent(filename)}`,
          { method: "DELETE" },
        );
        reloadRenders();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [project.id, reloadRenders],
  );

  const stored = opts?.options.target;
  const pairs = useMemo(() => {
    const base = stored?.mode === "map" ? (stored.pairs ?? []) : [];
    return base.map((pair) => nudged[pair.point_id] ?? pair);
  }, [stored, nudged]);

  /**
   * The geometry of the frame on the canvas.
   *
   * The stream's, whenever there is one — it came in the same message as the
   * pixels, so it describes the picture being looked at and nothing else. The
   * preview's own per-frame block is the fallback, and only trustworthy because
   * on that path the preview *is* re-fetched per frame.
   */
  const frameGeometry = useMemo<FrameGeometry | null>(() => {
    if (streamedGeometry) return streamedGeometry;
    if (!preview) return null;
    return {
      frame: preview.frame,
      frames: preview.frames,
      source: preview.source,
      output: preview.output,
      rect: preview.rect,
      warp: preview.warp,
      output_warp: preview.output_warp,
      uses_rotation_and_scale: preview.uses_rotation_and_scale,
      targets: preview.targets,
    };
  }, [streamedGeometry, preview]);

  const value = useMemo<ExportContextValue>(
    () => ({
      opts,
      pairs,
      preview,
      frameGeometry,
      error,
      busy,
      active,
      setActive,
      showGuides,
      setShowGuides,
      sideBySide,
      setSideBySide,
      dragging,
      setDragging,
      nudgeTarget,
      patchOptions,
      putPairs,
      reloadPreview,
      reloadOptions,
      renders,
      reloadRenders,
      deleteRender,
    }),
    [
      opts,
      pairs,
      preview,
      frameGeometry,
      error,
      busy,
      active,
      showGuides,
      sideBySide,
      dragging,
      nudgeTarget,
      patchOptions,
      putPairs,
      reloadPreview,
      reloadOptions,
      renders,
      reloadRenders,
      deleteRender,
    ],
  );

  return <ExportContext.Provider value={value}>{children}</ExportContext.Provider>;
}
