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

/**
 * Where a mark sits in the delivered frame, as a fraction of it.
 *
 * A fraction rather than a pixel, exactly as `stable/render/options.py` stores
 * it: the output's size is not known until the crop is planned, and a mark on
 * the thirds line has to stay on the thirds line when the operator crops
 * harder. Multiply by `geometry.output` to draw it; divide to read one back.
 */
export interface Placement {
  u: number;
  v: number;
}

/** One mark on the output. Mirrors `stable/render/options.TargetPoint`. */
export interface TargetPoint {
  id: string;
  place: Placement;
  /** The tracks held here; several means their mean, empty means all of them. */
  point_ids: string[];
  /** "raw" (the solid trail) or "smoothed" (the dashed one). */
  source: string;
  weight: number;
}

/** Mirrors stable/render/options.py. The server validates; this only edits. */
export interface RenderOptions {
  target: { points: TargetPoint[] };
  crop: { scale: number | null; limit_to_range: boolean };
  fill: { mode: string; color: number[]; blur: number; tint: number; feather: number };
  export_range: { start: number | null; end: number | null };
  resolution: { width: number | null; height: number | null };
}

export interface OptionsResponse {
  options: RenderOptions;
  target_sources: string[];
  fill_modes: string[];
  points: { id: string; enabled_for_solve: boolean; frames: number }[];
  source_size: [number, number];
  clip_frames: number;
  /** The In/Out markers, i.e. what the export range falls back to. */
  clip_range: [number, number];
  /** The frames a render would actually write with these options. */
  export_window: [number, number];
}

/**
 * One crosshair on screen: the mark, and where it is drawn on this frame.
 *
 * `u`/`v` are what is stored; `atU`/`atV` are where the fit actually puts the
 * held path this frame. They agree exactly for one or two marks, and can differ
 * for three or more, where a similarity cannot satisfy them all.
 */
export interface TargetMark {
  id: string;
  /** The track this mark holds, when it holds exactly one; else null. */
  pointId: string | null;
  pointIds: string[];
  source: string;
  u: number;
  v: number;
  atU: number;
  atV: number;
  /** How far the held path misses this mark by, in output pixels. */
  residual: number | null;
}

/**
 * The similarity the render applies to a frame, as the server sends it.
 *
 * Two of these come with every frame. `warp` is the stabilisation alone, before
 * the crop/zoom fit; `output_warp` is the whole thing, source pixels to output
 * pixels. Marks are stated as a fraction of the *output* frame (todo 1, eighth round), so
 * they go through `output_warp`: a mark is at `u × output.width` on the output
 * canvas, and at `unapplyWarp(output_warp, …)` of that on the source canvas.
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
  /** How many frames the automatic crop was measured over. */
  crop_measured_frames: number;
  zoom: number;
  gaps: boolean;
  border: string;
  max_displacement: number;
  is_no_op: boolean;
  no_op_detail: string | null;
  target: RenderOptions["target"];
  target_label: string;
  /** The frames this render would write, and the In/Out markers behind them. */
  export_window: [number, number];
  clip_range: [number, number];
  uses_rotation_and_scale: boolean;
  targets: {
    point_id: string | null;
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
   * The crosshairs on screen, one per target point.
   *
   * Read from the *options*, never from the preview (todo 7, fourth round).
   * They used to come off the preview, so any preview that failed or had not
   * come back yet made the panel believe there were no targets: the crosshairs
   * vanished and every row flipped back to "+ target" while the server still
   * held them. The preview is geometry; the options are truth.
   */
  marks: TargetMark[];
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
  /**
   * Whether a dragged mark snaps to the centre, the thirds and the other
   * points on screen. A composing aid, so it lives with the other view
   * preferences rather than in the render options the server stores.
   */
  snap: boolean;
  setSnap: (on: boolean) => void;
  /** Which mark is being dragged, by target id, for the viewer to highlight. */
  dragging: string | null;
  setDragging: (id: string | null) => void;
  /** Which mark the panel and the canvas are both pointing at. */
  selected: string | null;
  setSelected: (id: string | null) => void;
  /** Move one mark while the pointer is down, without a round trip per pixel. */
  nudgeMark: (id: string, u: number, v: number) => void;
  /** Store where a mark ended up. */
  commitMark: (id: string, u: number, v: number) => Promise<void>;
  /** Add a mark, and select it. Returns its new id. */
  addTarget: (patch: Partial<Omit<TargetPoint, "id">>) => Promise<string | null>;
  /** Change one mark: where it is, which tracks it holds, raw or smoothed. */
  patchTarget: (id: string, patch: Record<string, unknown>) => Promise<void>;
  /** Remove one mark. */
  deleteTarget: (id: string) => Promise<void>;

  patchOptions: (patch: Record<string, unknown>) => Promise<void>;
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
  const [snap, setSnap] = useState(true);
  const [dragging, setDragging] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // Where a mark has been dragged to but not yet stored, so the crosshair can
  // follow the pointer at screen rate while the server hears about it once.
  const [nudged, setNudged] = useState<Record<string, Placement>>({});

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

  const nudgeMark = useCallback((id: string, u: number, v: number) => {
    setNudged((prev) => ({ ...prev, [id]: { u, v } }));
  }, []);

  /**
   * Change one target point.
   *
   * A single-mark route rather than a whole-list PUT: two marks are edited
   * independently, and sending the list back would make moving one of them
   * carry a possibly stale copy of the other. The server mints ids and holds
   * the lock, so this stays a patch of exactly what changed.
   */
  const patchTarget = useCallback(
    async (id: string, patch: Record<string, unknown>) => {
      const seq = ++optionsSeq.current;
      setBusy(true);
      try {
        const next = await readJson<OptionsResponse>(
          `/api/projects/${project.id}/render/targets/${encodeURIComponent(id)}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
          },
        );
        if (seq !== optionsSeq.current) return;
        setOpts(next);
        setNudged({});
        setError(null);
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

  const commitMark = useCallback(
    (id: string, u: number, v: number) => patchTarget(id, { place: { u, v } }),
    [patchTarget],
  );

  const addTarget = useCallback(
    async (patch: Partial<Omit<TargetPoint, "id">>) => {
      const seq = ++optionsSeq.current;
      setBusy(true);
      try {
        const next = await readJson<OptionsResponse>(
          `/api/projects/${project.id}/render/targets`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
          },
        );
        if (seq !== optionsSeq.current) return null;
        setOpts(next);
        setError(null);
        invalidateOutput();
        await reloadPreview();
        // The server minted the id, so the newest mark is the one it just made.
        const added = next.options.target.points.at(-1)?.id ?? null;
        setSelected(added);
        return added;
      } catch (err) {
        if (seq === optionsSeq.current) setError(err instanceof Error ? err.message : String(err));
        return null;
      } finally {
        if (seq === optionsSeq.current) setBusy(false);
      }
    },
    [project.id, reloadPreview, invalidateOutput],
  );

  const deleteTarget = useCallback(
    async (id: string) => {
      const seq = ++optionsSeq.current;
      setBusy(true);
      try {
        const next = await readJson<OptionsResponse>(
          `/api/projects/${project.id}/render/targets/${encodeURIComponent(id)}`,
          { method: "DELETE" },
        );
        if (seq !== optionsSeq.current) return;
        setOpts(next);
        setSelected((current) => (current === id ? null : current));
        setError(null);
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

  /**
   * The crosshairs, each with the position it is drawn at on *this* frame.
   *
   * `atU`/`atV` differ from the stored mark only when the fit cannot satisfy
   * every mark at once — three or more of them, or a delivery whose shape the
   * marks cannot be squared with. The difference comes off the server's own
   * per-frame report of where each held path lands, the same numbers the
   * residual column shows, so the crosshair and the "N px off" beside it can
   * never disagree. One or two marks read zero and the crosshair sits exactly
   * where it was put.
   *
   * Rows come back in the order the marks were sent, so they line up by index:
   * a mark over several tracks has no single point id to key on, and inventing
   * one would be a second way for the two lists to fall out of step.
   */
  const marks = useMemo<TargetMark[]>(() => {
    const points = opts?.options.target.points ?? [];
    const output = frameGeometry?.output;
    const rows = frameGeometry?.targets ?? [];
    return points.map((point, index) => {
      const place = nudged[point.id] ?? point.place;
      const landed = rows[index]?.landed;
      const wanted = rows[index]?.target;
      const drift =
        landed && wanted && output?.width && output?.height
          ? {
              u: (landed[0] - wanted[0]) / output.width,
              v: (landed[1] - wanted[1]) / output.height,
            }
          : { u: 0, v: 0 };
      return {
        id: point.id,
        // A mark over several tracks (or over all of them) belongs to no one
        // point, so it gets no point colour — claiming one would say the
        // crosshair was that track's.
        pointId: point.point_ids.length === 1 ? point.point_ids[0] : null,
        source: point.source,
        pointIds: point.point_ids,
        u: place.u,
        v: place.v,
        atU: place.u + drift.u,
        atV: place.v + drift.v,
        residual: rows[index]?.residual ?? null,
      };
    });
  }, [opts, nudged, frameGeometry]);

  const value = useMemo<ExportContextValue>(
    () => ({
      opts,
      marks,
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
      snap,
      setSnap,
      dragging,
      setDragging,
      selected,
      setSelected,
      nudgeMark,
      commitMark,
      addTarget,
      patchTarget,
      deleteTarget,
      patchOptions,
      reloadPreview,
      reloadOptions,
      renders,
      reloadRenders,
      deleteRender,
    }),
    [
      opts,
      marks,
      preview,
      frameGeometry,
      error,
      busy,
      active,
      showGuides,
      sideBySide,
      snap,
      dragging,
      selected,
      nudgeMark,
      commitMark,
      addTarget,
      patchTarget,
      deleteTarget,
      patchOptions,
      reloadPreview,
      reloadOptions,
      renders,
      reloadRenders,
      deleteRender,
    ],
  );

  return <ExportContext.Provider value={value}>{children}</ExportContext.Provider>;
}
