import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import { confidenceColor } from "../colors";
import { applyWarp, unapplyWarp, useExportContext } from "../state/ExportContext";
import type { FrameGeometry } from "../api/frameStream";
import { useFrameStream } from "../state/FrameStreamContext";
import { useProjectContext } from "../state/ProjectContext";
import type { PointSummary, TrackData } from "../types";
import type { Overlays, Tool } from "../state/ProjectContext";

/** The overlay elements, their labels, and what each one is. */
export const OVERLAY_LABELS: [keyof Overlays, string, string][] = [
  ["paths", "Paths", "The trail a point traces through the clip"],
  [
    "smoothed",
    "Smoothed",
    "The same trail after the smoothing's stabilisation warp — where this feature ends up once the shot is stabilised. With no smoothing enabled the warp is the identity, so it sits exactly on the raw trail.",
  ],
  [
    "markers",
    "Points",
    "Where each point is on this frame — independent of the paths, so the trails can be hidden without losing the points",
  ],
  [
    "anchors",
    "Anchors",
    "The frames where you have stated the point's position — what every run tracks from",
  ],
  [
    "windows",
    "Window",
    "The pattern window (solid) and how far the tracker searches for it each frame (dashed)",
  ],
];

/** How close (in canvas CSS pixels) a mousedown has to be to grab a point. */
const GRAB_RADIUS = 12;

/** Sentinel `hoverPointId` for "the pointer is on an anchor". */
const ANCHOR_HOVER = " anchor-marker";

/** Sentinel `hoverPointId` for "the pointer is on an export target crosshair". */
const TARGET_HOVER = " export-target";

/** The tool palette: id, label, shortcut, and what a click does in it. */
const TOOLS: { id: Tool; label: string; key: string; help: string }[] = [
  ["move", "✥", "V", "Move: click a track to select it, drag or click to correct the selected one"],
  ["add-point", "✚", "N", "Add track: the next click on the frame starts a new one"],
  [
    "add-anchor",
    "◆",
    "A",
    "Add anchor: state where the selected track is on this frame, by clicking it. Runs track forwards and backwards from there",
  ],
  [
    "delete",
    "⌫",
    "D",
    "Delete: click an anchor to remove it, or a track's marker to remove the whole track",
  ],
  ["pan", "✋", "H", "Pan: drag the picture. Middle-drag does this in any tool"],
].map(([id, label, key, help]) => ({ id: id as Tool, label, key, help }));

/** Zoom is expressed as a multiple of "fit" (1 = the whole frame just fits). */
const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const WHEEL_ZOOM_FACTOR = 1.15;
const BUTTON_ZOOM_FACTOR = 1.25;

interface Drag {
  pointId: string;
  /** Where the point sits now, in source-video pixels. */
  fromX: number;
  fromY: number;
  /** Where the pointer has dragged it to, in source-video pixels. */
  toX: number;
  toY: number;
}

/**
 * An anchor being dragged on the frame (todo 4, sixth round).
 *
 * Moving one used to mean deleting it and placing a new one, which is two
 * gestures and a lost position if the second one is misjudged. The server has
 * always treated an anchor added at a frame that already has one as a replace —
 * `POST .../anchors` says so — so the whole move is one request on release, and
 * the anchor follows the pointer locally until then.
 */
interface AnchorDrag {
  pointId: string;
  frame: number;
  /** Where the pointer has dragged it to, in source-video pixels. */
  toX: number;
  toY: number;
}

interface Pan {
  x: number;
  y: number;
}

/**
 * The output canvas's own linear map: **output** pixels to canvas CSS pixels.
 *
 * Deliberately not the left canvas's `ViewTransform`. The two canvases are in
 * different coordinate spaces the moment the render crops or zooms, so sharing
 * one would be a lie about where things are; this one is always "fit", which is
 * what "what will the exported video look like" means. It exists as a value
 * because the same numbers place the crosshairs, hit-test them and convert a
 * drag back — three call sites that must never re-derive it separately.
 */
interface OutputFit {
  left: number;
  top: number;
  /** Canvas pixels per output pixel, x and y kept apart only for rounding. */
  sx: number;
  sy: number;
}

function outputToCanvas(fit: OutputFit, x: number, y: number) {
  return { x: fit.left + x * fit.sx, y: fit.top + y * fit.sy };
}

function canvasToOutput(fit: OutputFit, x: number, y: number) {
  return { x: (x - fit.left) / fit.sx, y: (y - fit.top) / fit.sy };
}

/**
 * Where a mark sits **on the output**, in output pixels.
 *
 * A mark is stored as a fraction of the output frame (todo 1, eighth round), so this is a
 * multiplication — and that is the whole reason the storage changed: the
 * previous form lived in the space *before* the crop/zoom fit, so every change
 * of crop moved the composition without anybody asking it to.
 */
function markOnOutput(geometry: FrameGeometry, u: number, v: number) {
  return { x: u * geometry.output.width, y: v * geometry.output.height };
}

/** Where that mark sits on the **source** frame, for the left canvas. */
function markOnSource(geometry: FrameGeometry, u: number, v: number) {
  const at = markOnOutput(geometry, u, v);
  return unapplyWarp(geometry.output_warp, at.x, at.y);
}

/** A source-pixel position, as a mark. Clamped: a mark is inside the output. */
function sourceToMark(geometry: FrameGeometry, x: number, y: number) {
  const at = applyWarp(geometry.output_warp, x, y);
  return {
    u: clamp01(at.x / geometry.output.width),
    v: clamp01(at.y / geometry.output.height),
  };
}

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

/** How close (canvas CSS pixels) a click has to be to grab a target crosshair. */
const TARGET_GRAB_RADIUS = 14;

/** How close (canvas CSS pixels) a dragged mark has to be to land on a line. */
const SNAP_RADIUS = 9;

/** The composition lines every frame has, as fractions of the output. */
const GUIDE_FRACTIONS = [1 / 3, 0.5, 2 / 3];

/**
 * Lines a dragged mark can land on, in output fractions, per axis.
 *
 * Two kinds, and the second is the one worth having: the thirds and the centre
 * are the same on every frame, but "level with that other feature" is a thing
 * only this frame knows — so the tracked points on screen contribute their own
 * horizontal and vertical, and lining a subject up with a horizon or a doorway
 * becomes a drag rather than an arithmetic problem.
 */
interface SnapLines {
  u: number[];
  v: number[];
}

/** Which line a mark landed on, so the canvases can show why it stopped there. */
interface SnapGuide {
  u: number | null;
  v: number | null;
}

/**
 * Pull a dragged mark onto a nearby line, one axis at a time.
 *
 * Independently per axis on purpose: dragging along the top thirds line should
 * stay on it while the horizontal position follows the pointer freely, which a
 * combined 2-D distance test cannot express.
 */
function snapMark(
  u: number,
  v: number,
  lines: SnapLines,
  tolerance: { u: number; v: number },
): { u: number; v: number; guide: SnapGuide } {
  const nearest = (value: number, candidates: number[], within: number) => {
    let best: number | null = null;
    let bestGap = within;
    for (const candidate of candidates) {
      const gap = Math.abs(candidate - value);
      if (gap <= bestGap) {
        best = candidate;
        bestGap = gap;
      }
    }
    return best;
  };
  const snappedU = nearest(u, lines.u, tolerance.u);
  const snappedV = nearest(v, lines.v, tolerance.v);
  return {
    u: snappedU ?? u,
    v: snappedV ?? v,
    guide: { u: snappedU, v: snappedV },
  };
}

/**
 * The one linear map between source-video pixels and canvas CSS pixels, used
 * by *every* draw call and *every* interaction handler below. Zoom and pan
 * interact (todo 3 + todo 8): a click has to land on the same source pixel the
 * frame was drawn at, at any zoom/pan level, or an edit silently lands in the
 * wrong place. Routing both directions through one pair of pure functions
 * (never re-deriving the arithmetic at each call site) is what keeps that true.
 */
interface ViewTransform {
  scale: number;
  originX: number;
  originY: number;
}

/** The scale that fits the whole source frame inside the display box, at zoom 1. */
function fitScale(dispW: number, dispH: number, videoW: number, videoH: number): number {
  if (dispW <= 0 || dispH <= 0 || videoW <= 0 || videoH <= 0) return 1;
  return Math.min(dispW / videoW, dispH / videoH);
}

function computeTransform(
  dispW: number,
  dispH: number,
  videoW: number,
  videoH: number,
  zoom: number,
  pan: Pan,
): ViewTransform {
  const scale = fitScale(dispW, dispH, videoW, videoH) * zoom;
  return { scale, originX: dispW / 2 + pan.x, originY: dispH / 2 + pan.y };
}

function sourceToCanvas(sx: number, sy: number, t: ViewTransform, videoW: number, videoH: number) {
  return { x: (sx - videoW / 2) * t.scale + t.originX, y: (sy - videoH / 2) * t.scale + t.originY };
}

function canvasToSource(cx: number, cy: number, t: ViewTransform, videoW: number, videoH: number) {
  return { x: (cx - t.originX) / t.scale + videoW / 2, y: (cy - t.originY) / t.scale + videoH / 2 };
}

/**
 * Where a point sits on one frame, in source pixels: what the tracker found
 * there, or the anchor the operator placed there if it has not been tracked
 * yet. Null when the point has neither, which is a frame with nothing to draw.
 */
function markerPosition(
  point: PointSummary,
  track: TrackData | undefined,
  frame: number,
): { x: number; y: number } | null {
  const x = track?.x[frame];
  const y = track?.y[frame];
  if (x != null && y != null) return { x, y };
  const anchor = point.queries.find((q) => q.frame === frame);
  return anchor ? { x: anchor.x, y: anchor.y } : null;
}

/**
 * Draw a decoded picture, and say whether it actually landed on the canvas.
 *
 * `drawImage` can refuse its source — an `ImageBitmap` whose pixels have been
 * freed throws `InvalidStateError: the image source is detached` — and a throw
 * from inside a `useEffect` is a render error, which React answers by unmounting
 * the tree. One bad frame becomes a blank application, which is what todo 6 of
 * this round reported. Reference counting in `api/frameStream.ts` is what stops
 * a live picture being freed in the first place; this is the second lock on the
 * same door. "There is no picture" is a state both canvases already draw, and it
 * is a far better answer to an unusable frame than taking the app down with it.
 */
function drawPicture(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  x: number,
  y: number,
  width: number,
  height: number,
): boolean {
  try {
    ctx.drawImage(image, x, y, width, height);
    return true;
  } catch {
    return false;
  }
}

/** Keep panned content covering the display box rather than drifting off it. */
function clampPan(pan: Pan, zoom: number, dispW: number, dispH: number, videoW: number, videoH: number): Pan {
  const scale = fitScale(dispW, dispH, videoW, videoH) * zoom;
  const halfW = (videoW * scale) / 2;
  const halfH = (videoH * scale) / 2;
  const clampAxis = (value: number, half: number, disp: number) => {
    if (half <= disp / 2) return 0; // frame doesn't fill this axis: no pan needed
    const limit = half - disp / 2;
    return Math.max(-limit, Math.min(limit, value));
  };
  return { x: clampAxis(pan.x, halfW, dispW), y: clampAxis(pan.y, halfH, dispH) };
}

/**
 * The canvas viewer: draws the current proxy frame plus an overlay of every
 * point's track path (confidence-coloured), its anchors, and handles
 * click-to-place for new points / new anchors — and, in edit mode, moving a
 * tracked point on the frame to correct it, either by dragging it or by
 * selecting it and clicking its new location.
 *
 * The canvas always fills the available space (letterboxed to the video's own
 * aspect ratio, never the browser's) and supports zoom (wheel, centred on the
 * cursor) and pan (middle-mouse drag) so a point can be placed precisely even
 * on a small or distant subject. A move never rewrites the track. It appends
 * one correction: a hard replace at this frame, or an offset blended over the
 * falloff window.
 */
export function Viewer() {
  const {
    project,
    currentFrame,
    setCurrentFrame,
    selectedPointId,
    setSelectedPointId,
    tool,
    setTool,
    addAnchorAt,
    afterEdit,
    invalidateTrack,
    stabilizedPreview,
    overlays,
    setOverlays,
    overlaysEnabled,
    setOverlaysEnabled,
    overlaysFor,
    isTrackHidden,
    colorFor,
    pathColor,
    setPathColor,
    followTracker,
    setFollowTracker,
    tracks,
    loadTrack,
    smoothedTracks,
    refresh,
    dragMode,
    setDragMode,
    falloff,
    setFalloff,
    applyCorrection,
    liveTracking,
  } = useProjectContext();
  // The export's overlay is drawn here rather than on a canvas of its own
  // (todo 1, fourth round), so the output rectangle and the stabilisation
  // targets zoom, pan and scrub with everything else.
  const exportState = useExportContext();
  // Frames, and the export geometry that arrives with them (todo 2, fifth
  // round). `viewFrame` — not `currentFrame` — is what this component draws and
  // edits at: see the comment on `viewFrame` below.
  const frames = useFrameStream();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  // The output canvas's fit, written by the effect that draws it and read by the
  // handlers that hit-test it — one set of numbers, so a crosshair is grabbed
  // exactly where it was drawn (todo 4, sixth round). Null when there is nothing
  // to draw on it yet.
  const outputFitRef = useRef<OutputFit | null>(null);
  const [outputHover, setOutputHover] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  // Bumped whenever the display box resizes (window resize, sidebar/curves
  // panel toggling), so the canvas is re-sized and re-fit to the new space
  // rather than staying at whatever size it happened to mount at (todo 2).
  const [sizeTick, setSizeTick] = useState(0);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [anchorDrag, setAnchorDrag] = useState<AnchorDrag | null>(null);
  // What the pointer is over: a point id, `TARGET_HOVER` for an export
  // crosshair, or null for bare picture. Drives the cursor (todo 4).
  const [hoverPointId, setHoverPointId] = useState<string | null>(null);
  // The line a mark being dragged has landed on, drawn on both canvases so the
  // crosshair stopping short of the pointer reads as "it caught on this",
  // rather than as the drag having gone wrong.
  const [snapGuide, setSnapGuide] = useState<SnapGuide | null>(null);
  const [panning, setPanning] = useState(false);
  // Why the delete tool did nothing, when it refuses — a click that silently
  // achieves nothing is indistinguishable from a broken one.
  const [deleteHint, setDeleteHint] = useState<string | null>(null);
  // A mousedown that grabbed a point (dragged or not) is followed by a native
  // "click" on the same element; without this the click handler would run
  // right after a drag release and re-apply the same move a second time.
  const suppressClickRef = useRef(false);

  // -- zoom & pan (todo 3) ------------------------------------------------------
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Pan>({ x: 0, y: 0 });
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [project.id]);

  // Overlay visibility (todos 7, 10) now lives in the context, because it is
  // per *track* as well as global (todo 9): the row of checkboxes below sets
  // the defaults, and each track card can override them for itself. `paths` is
  // the trail through time and `markers` is where the point is now — they were
  // one flag, which meant hiding a hundred dotted trails also hid the things
  // being tracked.

  const proxy = project.proxy["half"];
  const proxyReady = proxy?.ready && proxy.width && proxy.height;
  const videoW = project.video.w || 1;
  const videoH = project.video.h || 1;

  // Make sure every tracked point's data is loaded so the overlay can draw it.
  useEffect(() => {
    for (const point of project.points) {
      if (point.has_track && !(point.id in tracks)) loadTrack(point.id);
    }
  }, [project.points, tracks, loadTrack]);

  /**
   * The frame this canvas is actually showing, which is what everything below
   * is drawn and edited at (todo 2, fifth round).
   *
   * It is not always the playhead. Frames are never blanked while the next one
   * loads — a picture only ever changes to another picture — so during a scrub
   * `currentFrame` can be ahead of the pixels on screen. Drawing the markers,
   * the output rectangle and the target crosshairs at `currentFrame` over frame
   * 244's picture is exactly the "frames out of sync with the track data" this
   * round exists to remove, so they are drawn here instead, and a click lands on
   * the frame the operator can see rather than on one whose picture has not
   * arrived. With nothing in flight the two are equal, which is the normal case.
   */
  const viewFrame = frames.viewFrame;
  const mainVariant = stabilizedPreview ? "output" : "source";
  const picture = frames.view?.images[mainVariant] ?? null;
  // This canvas reports only what went wrong with *its* picture: with the
  // side-by-side view up, an output frame that cannot be produced is the right
  // canvas's problem to state, not this one's.
  const mainError = frames.view?.errors[mainVariant] ?? null;
  const error = mainError && stabilizedPreview ? "stabilized preview not ready yet" : mainError;
  // The output beside the frame, while the export tab is up (todo 1, fifth
  // round). Off the same view as the picture on the left, so the two canvases
  // are never a frame apart.
  const showOutput = exportState.active && exportState.sideBySide && !stabilizedPreview;
  const outputPicture = frames.view?.images.output ?? null;
  const outputError = frames.view?.errors.output ?? null;

  // The stabilized preview warps (and may crop) the frame, so overlay
  // coordinates — which are in *source* space — no longer line up with what is
  // drawn underneath. Points are hidden rather than drawn in the wrong place.
  const overlayValid = !stabilizedPreview;

  // Re-fit the canvas whenever the space it has to fill changes.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => setSizeTick((n) => n + 1));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !proxyReady) return;
    const dispW = wrap.clientWidth || 1;
    const dispH = wrap.clientHeight || 1;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(dispW * dpr));
    canvas.height = Math.max(1, Math.floor(dispH * dpr));
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, dispW, dispH);

    const t = computeTransform(dispW, dispH, videoW, videoH, zoom, pan);
    const toCanvasX = (sx: number) => (sx - videoW / 2) * t.scale + t.originX;
    const toCanvasY = (sy: number) => (sy - videoH / 2) * t.scale + t.originY;
    const frameLeft = toCanvasX(0);
    const frameTop = toCanvasY(0);
    const frameW = videoW * t.scale;
    const frameH = videoH * t.scale;

    if (!picture || !drawPicture(ctx, picture.image, frameLeft, frameTop, frameW, frameH)) {
      ctx.fillStyle = "#111";
      ctx.fillRect(frameLeft, frameTop, frameW, frameH);
    }
    // Nothing in source coordinates may be drawn over the stabilized preview:
    // it warps (and may crop) the frame, so a marker placed at the source
    // position would point at the wrong pixel. The master switch is the other
    // reason to draw no track overlay — the export's own overlay below is
    // governed by neither, since it is not per track.
    const overlayPoints = overlayValid && overlaysEnabled ? project.points : [];
    for (const point of overlayPoints) {
      const isSelected = point.id === selectedPointId;
      const track = tracks[point.id];
      const color = colorFor(point.id);
      const live = liveTracking[point.id];
      // What *this* track draws: the global defaults with its own overrides on
      // top (stable/../ProjectContext.tsx `overlaysFor`).
      const {
        paths: showPaths,
        smoothed: showSmoothed,
        markers: showMarkers,
        anchors: showAnchors,
        windows: showWindows,
      } = overlaysFor(point.id);

      // A run in flight (todos 4, 16): the path found so far is drawn over the
      // stale one in the point's own colour, and the frame the tracker is
      // reading right now gets a ring. Scrubbing back over covered frames shows
      // the tracker's answer for them before the job has finished.
      if (live?.x && live.y && showPaths) {
        ctx.save();
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeStyle = color;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        let started = false;
        for (let f = 0; f < live.x.length; f++) {
          const lx = live.x[f];
          const ly = live.y[f];
          if (lx == null || ly == null) {
            started = false;
            continue;
          }
          const px = toCanvasX(lx);
          const py = toCanvasY(ly);
          if (started) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          started = true;
        }
        ctx.stroke();
        ctx.restore();

        const hx = live.x[viewFrame];
        const hy = live.y[viewFrame];
        if (showMarkers && hx != null && hy != null) {
          ctx.save();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(toCanvasX(hx), toCanvasY(hy), 9, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
      }

      if (track) {
        // The before/after "ghost" of the raw path used to be drawn here (todo
        // 2, ninth round). It was a faint dotted trail a pixel or two from the
        // corrected one, on top of the trail, the smoothed trail and every
        // other track's — one more line to read for a fact the correction list
        // states in words. The server still sends `raw_x` / `raw_y`; nothing on
        // the canvas draws them.

        if (showPaths) {
          // Path: connect consecutive valid frames, coloured one of two ways
          // (todo 5, second round). `confidence` shades each short segment by
          // that frame's confidence — green = trust it, red = check it — which
          // is the better answer to "is this track any good" and the worse one
          // to "which of these four trails is p3", since every track is drawn
          // from the same ramp. `track` uses the point's own colour instead,
          // and can stroke the whole trail as one path.
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          if (pathColor === "track") {
            ctx.save();
            ctx.strokeStyle = color;
            ctx.beginPath();
            let started = false;
            for (let f = 0; f < track.n_frames; f++) {
              const x = track.x[f];
              const y = track.y[f];
              if (x == null || y == null) {
                started = false;
                continue;
              }
              const px = toCanvasX(x);
              const py = toCanvasY(y);
              if (started) ctx.lineTo(px, py);
              else ctx.moveTo(px, py);
              started = true;
            }
            ctx.stroke();
            ctx.restore();
          } else {
            for (let f = 1; f < track.n_frames; f++) {
              const x0 = track.x[f - 1];
              const y0 = track.y[f - 1];
              const x1 = track.x[f];
              const y1 = track.y[f];
              if (x0 == null || y0 == null || x1 == null || y1 == null) continue;
              ctx.strokeStyle = confidenceColor(
                (track.confidence[f - 1] + track.confidence[f]) / 2,
              );
              ctx.beginPath();
              ctx.moveTo(toCanvasX(x0), toCanvasY(y0));
              ctx.lineTo(toCanvasX(x1), toCanvasY(y1));
              ctx.stroke();
            }
          }
        }

        // What the smoothing did to this track (todo 3, fourth round): the same
        // path pushed through the stabilisation warp, i.e. where this feature
        // ends up once the shot is stabilised. Drawn over the source trail in
        // the point's own colour so the two can be read against each other —
        // the gap between them *is* the correction the export will apply. With
        // no smoothing enabled the warp is the identity and the two coincide,
        // which is a true statement about the project rather than a missing
        // feature.
        const stabilized = smoothedTracks[point.id];
        if (showSmoothed && stabilized) {
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = isSelected ? 2 : 1.25;
          ctx.setLineDash([7, 3]);
          ctx.beginPath();
          let started = false;
          for (let f = 0; f < stabilized.x.length; f++) {
            const sx = stabilized.x[f];
            const sy = stabilized.y[f];
            if (sx == null || sy == null) {
              started = false;
              continue;
            }
            const px = toCanvasX(sx);
            const py = toCanvasY(sy);
            if (started) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            started = true;
          }
          ctx.stroke();
          // Where this point sits on *this* frame after stabilisation, as a
          // hollow ring: "it is here now, and would be there in the export".
          const hx = stabilized.x[viewFrame];
          const hy = stabilized.y[viewFrame];
          if (hx != null && hy != null) {
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.arc(toCanvasX(hx), toCanvasY(hy), isSelected ? 7 : 5, 0, Math.PI * 2);
            ctx.stroke();
          }
          ctx.restore();
        }

        if (showMarkers) {
          // Current-frame marker, following the pointer while it is being dragged.
          const cx = track.x[viewFrame];
          const cy = track.y[viewFrame];
          if (cx != null && cy != null) {
            const dragging = drag !== null && drag.pointId === point.id;
            const mx = dragging ? drag.toX : cx;
            const my = dragging ? drag.toY : cy;
            if (dragging) {
              ctx.save();
              ctx.strokeStyle = "#ffffff";
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(toCanvasX(cx), toCanvasY(cy));
              ctx.lineTo(toCanvasX(mx), toCanvasY(my));
              ctx.stroke();
              ctx.restore();
            }
            ctx.beginPath();
            ctx.arc(toCanvasX(mx), toCanvasY(my), isSelected ? 6 : 4, 0, Math.PI * 2);
            // The marker follows the same choice as the path: telling tracks
            // apart is not much use if their trails differ but the dots on this
            // frame — the things actually being pointed at — do not.
            ctx.fillStyle =
              pathColor === "track" ? color : confidenceColor(track.confidence[viewFrame]);
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = dragging ? "#ffd166" : "#fff";
            ctx.stroke();
          }
        }
      }

      // Anchors (diamonds), always drawn from the point spec itself so they
      // show up even before the point has been tracked.
      if (showAnchors) {
        for (const q of point.queries) {
          // One being dragged follows the pointer, with a dashed leader back
          // to where it was, so a move can be judged against the position it is
          // replacing rather than only against the picture (todo 4, sixth round).
          const moving =
            anchorDrag && anchorDrag.pointId === point.id && anchorDrag.frame === q.frame
              ? anchorDrag
              : null;
          const px = toCanvasX(moving ? moving.toX : q.x);
          const py = toCanvasY(moving ? moving.toY : q.y);
          if (moving) {
            ctx.save();
            ctx.strokeStyle = "#ffffff";
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(toCanvasX(q.x), toCanvasY(q.y));
            ctx.lineTo(px, py);
            ctx.stroke();
            ctx.restore();
          }
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          const size = q.frame === viewFrame ? 7 : 5;
          ctx.fillStyle = color;
          ctx.strokeStyle = moving ? "#ffd166" : "#fff";
          ctx.lineWidth = 1;
          ctx.fillRect(-size / 2, -size / 2, size, size);
          ctx.strokeRect(-size / 2, -size / 2, size, size);
          ctx.restore();
        }
      }

      // The pattern the tracker matches and how far it looks for it, as two
      // boxes on the frame — the only way to answer "is my window big enough
      // for this feature, and does it move further than that between frames?"
      // by looking rather than by guessing. Selected point only: at eight
      // points this would be a mesh, and the size being judged is the one
      // whose sliders are open in the panel below.
      if (showWindows && isSelected) {
        const at = markerPosition(point, track, viewFrame);
        if (at) {
          const box = (radius: number, dashed: boolean, stroke: string) => {
            const side = (radius * 2 + 1) * t.scale;
            ctx.save();
            ctx.setLineDash(dashed ? [5, 4] : []);
            ctx.lineWidth = 1;
            ctx.strokeStyle = stroke;
            ctx.strokeRect(
              toCanvasX(at.x - radius) - 0.5,
              toCanvasY(at.y - radius) - 0.5,
              side,
              side,
            );
            ctx.restore();
          };
          const search = point.window?.search_radius ?? null;
          if (search !== null) box(search, true, "rgba(255,255,255,0.55)");
          if (point.window?.radius) box(point.window.radius, false, color);
        }
      }
    }

    // -- the export overlay (todo 1, fourth round) ------------------------------
    //
    // What the export will keep, drawn on the picture it will keep it from:
    // everything outside the output rectangle dimmed, the rectangle itself, its
    // thirds, and the stabilisation targets as draggable crosshairs. All of it
    // in the viewer's own transform, so it zooms and pans with the frame.
    // The *frame's* geometry, not the preview's: it arrived in the same message
    // as the pixels above, so the rectangle drawn here and the picture under it
    // are the same frame by construction (todo 2, fifth round).
    const geometry = exportState.frameGeometry;
    if (!exportState.active || !overlayValid || !geometry) return;
    const rect = geometry.rect;
    const gaps = exportState.preview?.gaps ?? false;

    ctx.save();
    ctx.beginPath();
    ctx.rect(frameLeft, frameTop, frameW, frameH);
    ctx.moveTo(toCanvasX(rect[0][0]), toCanvasY(rect[0][1]));
    for (const [x, y] of rect.slice(1)) ctx.lineTo(toCanvasX(x), toCanvasY(y));
    ctx.closePath();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fill("evenodd");
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = gaps ? "#ffd166" : "#4da3ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(rect[0][0]), toCanvasY(rect[0][1]));
    for (const [x, y] of rect.slice(1)) ctx.lineTo(toCanvasX(x), toCanvasY(y));
    ctx.closePath();
    ctx.stroke();

    if (exportState.showGuides) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      // Thirds of the *output* rectangle, interpolated along its own edges —
      // exact, because a similarity keeps those edges parallel.
      const [tl, tr, br, bl] = rect;
      const along = (a: number[], b: number[], f: number) => [
        a[0] + f * (b[0] - a[0]),
        a[1] + f * (b[1] - a[1]),
      ];
      for (const f of [1 / 3, 2 / 3]) {
        const top = along(tl, tr, f);
        const bottom = along(bl, br, f);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(top[0]), toCanvasY(top[1]));
        ctx.lineTo(toCanvasX(bottom[0]), toCanvasY(bottom[1]));
        ctx.stroke();
        const left = along(tl, bl, f);
        const right = along(tr, br, f);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(left[0]), toCanvasY(left[1]));
        ctx.lineTo(toCanvasX(right[0]), toCanvasY(right[1]));
        ctx.stroke();
      }
    }
    ctx.restore();

    // The line a dragged mark has landed on. Stated in output fractions, so on
    // this canvas it is the edge of the output rectangle mapped back through
    // the same transform everything else here uses — it stays glued to the
    // picture when the render rotates, which a screen-aligned line would not.
    if (snapGuide) {
      ctx.save();
      ctx.strokeStyle = "rgba(120,255,190,0.85)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      const segment = (from: { x: number; y: number }, to: { x: number; y: number }) => {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(from.x), toCanvasY(from.y));
        ctx.lineTo(toCanvasX(to.x), toCanvasY(to.y));
        ctx.stroke();
      };
      if (snapGuide.u != null) {
        segment(markOnSource(geometry, snapGuide.u, 0), markOnSource(geometry, snapGuide.u, 1));
      }
      if (snapGuide.v != null) {
        segment(markOnSource(geometry, 0, snapGuide.v), markOnSource(geometry, 1, snapGuide.v));
      }
      ctx.restore();
    }

    // The marks, drawn where the held path is on *this* frame. A mark is a
    // fraction of the *output*, so getting it onto the source picture means
    // going backwards through the whole transform that produced the output.
    for (const mark of exportState.marks) {
      const at = markOnSource(geometry, mark.atU, mark.atV);
      const cx = toCanvasX(at.x);
      const cy = toCanvasY(at.y);
      ctx.save();
      ctx.strokeStyle = mark.pointId ? colorFor(mark.pointId) : "#e8e8e8";
      ctx.lineWidth =
        exportState.dragging === mark.id || exportState.selected === mark.id ? 3 : 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 14, cy);
      ctx.lineTo(cx + 14, cy);
      ctx.moveTo(cx, cy - 14);
      ctx.lineTo(cx, cy + 14);
      ctx.stroke();
      // The mark's id, so a row in the panel and a crosshair on the picture are
      // visibly the same object once there is more than one of them.
      if (exportState.marks.length > 1) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(mark.id, cx + 15, cy - 8);
      }
      ctx.restore();
    }
  }, [
    project.points,
    videoW,
    videoH,
    tracks,
    smoothedTracks,
    viewFrame,
    selectedPointId,
    proxyReady,
    picture,
    frames.view?.revision,
    liveTracking,
    drag,
    anchorDrag,
    overlayValid,
    sizeTick,
    zoom,
    pan,
    overlaysEnabled,
    overlaysFor,
    colorFor,
    pathColor,
    exportState.active,
    exportState.frameGeometry,
    exportState.preview?.gaps,
    exportState.marks,
    exportState.showGuides,
    exportState.dragging,
    snapGuide,
  ]);

  /**
   * The side-by-side output canvas (todo 1, fifth round): the file, plus the one
   * thing that gets composed on it.
   *
   * It is not zoom/pan-linked to the canvas on the left — the two are in
   * different coordinate spaces the moment the render crops or zooms, so a
   * shared view transform would be a lie about where things are. This one is
   * always "fit", which is what "what will the exported video look like" means,
   * letterboxed to the *output's* aspect ratio rather than the source's.
   *
   * What it draws over the picture is only what the operator is *placing* (todo
   * 4, sixth round): the thirds, when they are on for the left canvas, and the
   * stabilisation crosshairs — which are draggable here too, so a shot can be
   * composed against the framing that will actually be delivered rather than
   * against the source it is cropped out of. Not the tracks, not the tools: this
   * is still the file, and a marker on it would be a claim about pixels the
   * export is throwing away.
   */
  useEffect(() => {
    const canvas = outputCanvasRef.current;
    if (!canvas) return;
    const wrap = canvas.parentElement;
    if (!wrap) return;
    const dispW = wrap.clientWidth || 1;
    const dispH = wrap.clientHeight || 1;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(dispW * dpr));
    canvas.height = Math.max(1, Math.floor(dispH * dpr));
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, dispW, dispH);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, dispW, dispH);

    const geometry = exportState.frameGeometry;
    // The picture is the output at *proxy* resolution; the geometry states it at
    // delivery resolution. The box is fitted from whichever is in hand and the
    // mapping is expressed in output pixels either way, so an overlay can be
    // drawn on a frame that has not arrived (or cannot be produced) and a
    // rounding difference between the two never displaces a crosshair.
    const boxW = outputPicture?.width ?? geometry?.output.width ?? 0;
    const boxH = outputPicture?.height ?? geometry?.output.height ?? 0;
    if (boxW <= 0 || boxH <= 0) {
      outputFitRef.current = null;
      return;
    }
    const scale = Math.min(dispW / boxW, dispH / boxH);
    const width = boxW * scale;
    const height = boxH * scale;
    const left = (dispW - width) / 2;
    const top = (dispH - height) / 2;
    if (outputPicture) drawPicture(ctx, outputPicture.image, left, top, width, height);

    if (!geometry) {
      outputFitRef.current = null;
      return;
    }
    const fit: OutputFit = {
      left,
      top,
      sx: width / geometry.output.width,
      sy: height / geometry.output.height,
    };
    outputFitRef.current = fit;

    // Thirds, on this canvas as well as the left one (todo 4, sixth round). Here
    // they are the thirds of the whole picture, because on this canvas the
    // output rectangle *is* the picture — which is the point of showing them
    // here: on the left they are thirds of a rectangle drawn inside a bigger
    // frame, and judging a composition against them is easier without the rest
    // of the frame around it.
    if (exportState.showGuides) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (const f of [1 / 3, 2 / 3]) {
        ctx.beginPath();
        ctx.moveTo(left + width * f, top);
        ctx.lineTo(left + width * f, top + height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(left, top + height * f);
        ctx.lineTo(left + width, top + height * f);
        ctx.stroke();
      }
      ctx.restore();
    }

    // The snap line, which on this canvas is exactly a screen-aligned line —
    // the output rectangle *is* the picture here, so a third of the output is a
    // third of what is drawn.
    if (snapGuide) {
      ctx.save();
      ctx.strokeStyle = "rgba(120,255,190,0.85)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      if (snapGuide.u != null) {
        ctx.beginPath();
        ctx.moveTo(left + width * snapGuide.u, top);
        ctx.lineTo(left + width * snapGuide.u, top + height);
        ctx.stroke();
      }
      if (snapGuide.v != null) {
        ctx.beginPath();
        ctx.moveTo(left, top + height * snapGuide.v);
        ctx.lineTo(left + width, top + height * snapGuide.v);
        ctx.stroke();
      }
      ctx.restore();
    }

    for (const mark of exportState.marks) {
      const at = markOnOutput(geometry, mark.atU, mark.atV);
      const c = outputToCanvas(fit, at.x, at.y);
      ctx.save();
      ctx.strokeStyle = mark.pointId ? colorFor(mark.pointId) : "#e8e8e8";
      ctx.lineWidth =
        exportState.dragging === mark.id || exportState.selected === mark.id ? 3 : 2;
      ctx.beginPath();
      ctx.arc(c.x, c.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(c.x - 14, c.y);
      ctx.lineTo(c.x + 14, c.y);
      ctx.moveTo(c.x, c.y - 14);
      ctx.lineTo(c.x, c.y + 14);
      ctx.stroke();
      if (exportState.marks.length > 1) {
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(mark.id, c.x + 15, c.y - 8);
      }
      ctx.restore();
    }
  }, [
    outputPicture,
    showOutput,
    sizeTick,
    colorFor,
    frames.view?.revision,
    exportState.frameGeometry,
    exportState.marks,
    exportState.showGuides,
    exportState.dragging,
    snapGuide,
  ]);

  /** Convert a mouse event into source-video pixel coordinates, at the current zoom/pan. */
  const toSource = useCallback(
    (evt: { clientX: number; clientY: number }) => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const canvasX = evt.clientX - rect.left;
      const canvasY = evt.clientY - rect.top;
      const t = computeTransform(rect.width, rect.height, videoW, videoH, zoom, pan);
      const src = canvasToSource(canvasX, canvasY, t, videoW, videoH);
      return { x: src.x, y: src.y, canvasX, canvasY };
    },
    [videoW, videoH, zoom, pan],
  );

  /** Return the tracked point whose current-frame marker is under the pointer. */
  const hitTest = useCallback(
    (canvasX: number, canvasY: number): string | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const t = computeTransform(rect.width, rect.height, videoW, videoH, zoom, pan);
      let best: { id: string; distance: number } | null = null;
      for (const point of project.points) {
        // A hidden track is not on the screen, so nothing on the screen can be
        // it (todo 6, ninth round): selecting — or, with the delete tool,
        // deleting — something invisible is never what the click meant.
        if (isTrackHidden(point.id)) continue;
        const track = tracks[point.id];
        const x = track?.x[viewFrame];
        const y = track?.y[viewFrame];
        if (x == null || y == null) continue;
        const p = sourceToCanvas(x, y, t, videoW, videoH);
        const distance = Math.hypot(p.x - canvasX, p.y - canvasY);
        if (distance <= GRAB_RADIUS && (!best || distance < best.distance)) {
          best = { id: point.id, distance };
        }
      }
      return best?.id ?? null;
    },
    [project.points, tracks, viewFrame, videoW, videoH, zoom, pan, isTrackHidden],
  );

  /**
   * The anchor under the pointer on the frame being shown, if any.
   *
   * Any point's, not only the selected one's: an anchor is drawn on exactly one
   * frame and takes a deliberate click to hit, so grabbing one cannot happen by
   * accident the way grabbing a marker in a crowd of them can.
   */
  const anchorHitTest = useCallback(
    (canvasX: number, canvasY: number): { pointId: string; query: PointSummary["queries"][number] } | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const bounds = canvas.getBoundingClientRect();
      const t = computeTransform(bounds.width, bounds.height, videoW, videoH, zoom, pan);
      let best: { pointId: string; query: PointSummary["queries"][number]; distance: number } | null =
        null;
      for (const point of project.points) {
        if (!overlaysFor(point.id).anchors) continue;
        for (const query of point.queries) {
          if (query.frame !== viewFrame) continue;
          const at = sourceToCanvas(query.x, query.y, t, videoW, videoH);
          const distance = Math.hypot(at.x - canvasX, at.y - canvasY);
          if (distance <= GRAB_RADIUS && (!best || distance < best.distance)) {
            best = { pointId: point.id, query, distance };
          }
        }
      }
      return best ? { pointId: best.pointId, query: best.query } : null;
    },
    [project.points, overlaysFor, viewFrame, videoW, videoH, zoom, pan],
  );

  /**
   * Apply the correction a move to (toX, toY) implies for `pointId`, shared by
   * both drag-release and click-to-move (todo 8) so the two input paths can
   * never disagree about what a "move" means.
   */
  const commitMove = useCallback(
    (pointId: string, toX: number, toY: number, shiftKey: boolean) => {
      const track = tracks[pointId];
      const fromX = track?.x[viewFrame];
      const fromY = track?.y[viewFrame];
      if (fromX == null || fromY == null) return;
      const dx = toX - fromX;
      const dy = toY - fromY;
      if (Math.hypot(dx, dy) < 0.5) return; // no meaningful move
      // Shift forces a hard replace whatever the panel says: "I know exactly
      // where this is" should always be one key away.
      const hard = shiftKey || dragMode === "replace";
      void applyCorrection(
        pointId,
        hard
          ? { kind: "replace", frame: viewFrame, x: toX, y: toY }
          : {
              kind: "offset",
              frame: viewFrame,
              dx,
              dy,
              falloff_before: falloff,
              falloff_after: falloff,
              shape: "smooth",
            },
      );
    },
    [tracks, viewFrame, dragMode, falloff, applyCorrection],
  );

  function zoomAtPoint(cx: number, cy: number, factor: number) {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const dispW = wrap.clientWidth || 1;
    const dispH = wrap.clientHeight || 1;
    setZoom((prevZoom) => {
      const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prevZoom * factor));
      if (nextZoom === prevZoom) return prevZoom;
      setPan((prevPan) => {
        // Keep the source point currently under the cursor fixed under it.
        const prevT = computeTransform(dispW, dispH, videoW, videoH, prevZoom, prevPan);
        const src = canvasToSource(cx, cy, prevT, videoW, videoH);
        const nextScale = fitScale(dispW, dispH, videoW, videoH) * nextZoom;
        const newOriginX = cx - (src.x - videoW / 2) * nextScale;
        const newOriginY = cy - (src.y - videoH / 2) * nextScale;
        const next = { x: newOriginX - dispW / 2, y: newOriginY - dispH / 2 };
        return clampPan(next, nextZoom, dispW, dispH, videoW, videoH);
      });
      return nextZoom;
    });
  }

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  // Wheel zooms, centred on the cursor. Registered as a non-passive native
  // listener (same reason as CurveEditor's wheel handler): React attaches
  // wheel handlers passively, so `preventDefault` in an `onWheel` prop is
  // ignored, and zooming has to stop the page/canvas from scrolling.
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      zoomAtPoint(
        event.clientX - rect.left,
        event.clientY - rect.top,
        event.deltaY > 0 ? 1 / WHEEL_ZOOM_FACTOR : WHEEL_ZOOM_FACTOR,
      );
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoW, videoH]);

  // '+'/'-' zoom, '0' resets to fit — a keyboard alternative to the wheel,
  // ignored while typing in a field.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const wrap = wrapRef.current;
      if (!wrap) return;
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomAtPoint(wrap.clientWidth / 2, wrap.clientHeight / 2, BUTTON_ZOOM_FACTOR);
      } else if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        zoomAtPoint(wrap.clientWidth / 2, wrap.clientHeight / 2, 1 / BUTTON_ZOOM_FACTOR);
      } else if (event.key === "0") {
        event.preventDefault();
        resetView();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoW, videoH]);

  async function handleClick(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (!canvasRef.current || !proxyReady) return;
    const { x, y, canvasX, canvasY } = toSource(evt);

    if (tool === "add-point") {
      const point = await api.createPoint(project.id, viewFrame, x, y);
      setTool("move");
      setSelectedPointId(point.id);
      await refresh();
      return;
    }
    if (tool === "add-anchor" && selectedPointId) {
      // The tool stays armed (todo 1, ninth round). It used to put itself away
      // after one anchor, which made anchoring the same track on four frames
      // four trips to the toolbar — and anchoring across frames is the whole
      // gesture. Esc, V, or A again puts it away.
      //
      // Goes through the context (not api.addAnchor + refresh) so the anchor
      // list is re-read with the track it lives on — see `addAnchorAt`.
      await addAnchorAt(selectedPointId, viewFrame, x, y);
      return;
    }
    if (tool === "delete") {
      await deleteUnderPointer(canvasX, canvasY);
      return;
    }
    if (tool !== "move" || !overlayValid || exportState.active) return;

    // Clicking a *different* track selects it — which is what the pointer
    // cursor over it has just promised (todo 4). Only after that does a click
    // on bare picture mean "move the selected point here".
    const hovered = hitTest(canvasX, canvasY);
    if (hovered && hovered !== selectedPointId) {
      setSelectedPointId(hovered);
      return;
    }
    if (!selectedPointId) return;
    // Click-to-move (todo 8, first round): the selected point jumps to wherever
    // was clicked, exactly as if it had been dragged there. A move is one
    // undoable correction, so the cost of a stray click is one Ctrl+Z.
    commitMove(selectedPointId, x, y, evt.shiftKey);
  }

  /**
   * The delete tool: take away whatever the pointer is on.
   *
   * An anchor first — it is the finer thing, and it is undoable. Only when
   * there is no anchor under the cursor does this mean the whole track, and
   * that is confirmed: a track is minutes of tracking and, unlike an anchor,
   * deleting one is not on the undo stack.
   */
  async function deleteUnderPointer(canvasX: number, canvasY: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const t = computeTransform(bounds.width, bounds.height, videoW, videoH, zoom, pan);
    for (const point of project.points) {
      // Same rule as the marker hit test: a hidden track's anchors are not
      // drawn, so the delete tool cannot reach one by accident.
      if (isTrackHidden(point.id)) continue;
      for (const query of point.queries) {
        if (query.frame !== viewFrame) continue;
        const at = sourceToCanvas(query.x, query.y, t, videoW, videoH);
        if (Math.hypot(at.x - canvasX, at.y - canvasY) > GRAB_RADIUS) continue;
        try {
          await api.deleteAnchor(project.id, point.id, query.frame);
        } catch (err) {
          // The server owns the "a point needs one anchor" rule, because it is
          // the only side that can see the anchors a correction implies.
          setDeleteHint(err instanceof Error ? err.message : String(err));
          return;
        }
        await afterEdit(point.id);
        return;
      }
    }
    const pointId = hitTest(canvasX, canvasY);
    if (!pointId) return;
    if (!window.confirm(`Delete the track ${pointId}? This cannot be undone.`)) return;
    await api.deletePoint(project.id, pointId);
    invalidateTrack(pointId);
    if (selectedPointId === pointId) setSelectedPointId(null);
    await refresh();
  }

  /**
   * The lines a mark being dragged can land on, in output fractions.
   *
   * The thirds and the centre, plus every *other* mark and every tracked point
   * that has a position on this frame — the last of those is what makes "line
   * the subject up with that doorway" a drag. The dragged mark itself is
   * excluded, or it would snap to where it already is and never move.
   */
  const snapLinesFor = useCallback(
    (key: string, geometry: FrameGeometry): SnapLines => {
      const lines: SnapLines = { u: [...GUIDE_FRACTIONS], v: [...GUIDE_FRACTIONS] };
      for (const mark of exportState.marks) {
        if (mark.id === key) continue;
        lines.u.push(mark.atU);
        lines.v.push(mark.atV);
      }
      for (const point of project.points) {
        const track = tracks[point.id];
        const x = track?.x[geometry.frame];
        const y = track?.y[geometry.frame];
        if (x == null || y == null) continue;
        const at = applyWarp(geometry.output_warp, x, y);
        lines.u.push(at.x / geometry.output.width);
        lines.v.push(at.y / geometry.output.height);
      }
      return lines;
    },
    [exportState.marks, project.points, tracks],
  );

  /**
   * Apply the snap, unless it is switched off, and remember what it caught on.
   *
   * `perCanvasPixel` is how much of the output one canvas pixel spans, which is
   * what makes the pull feel the same at any zoom: the tolerance is stated in
   * screen pixels, because that is where the operator's hand is.
   */
  const applySnap = useCallback(
    (
      key: string,
      geometry: FrameGeometry,
      u: number,
      v: number,
      perCanvasPixel: number,
    ): { u: number; v: number } => {
      if (!exportState.snap) {
        setSnapGuide(null);
        return { u, v };
      }
      const reach = SNAP_RADIUS * perCanvasPixel;
      const snapped = snapMark(u, v, snapLinesFor(key, geometry), {
        u: reach / geometry.output.width,
        v: reach / geometry.output.height,
      });
      setSnapGuide(snapped.guide.u == null && snapped.guide.v == null ? null : snapped.guide);
      return { u: snapped.u, v: snapped.v };
    },
    [exportState.snap, snapLinesFor],
  );

  /**
   * Start dragging a stabilisation mark, if the pointer is on one.
   *
   * The crosshair follows the pointer locally and the new position is stored
   * once, on release: a mark move re-plans the whole clip server-side, so one
   * request per gesture is the difference between a drag and a queue of them.
   *
   * Returns:
   *   Whether a mark was grabbed, i.e. whether the mousedown is spoken for.
   */
  function grabTarget(evt: React.MouseEvent<HTMLCanvasElement>): boolean {
    // The geometry of the frame being *shown*, for the same reason the overlay
    // is drawn from it: grabbing a crosshair has to invert the transform that
    // put it where the pointer is, and that is this frame's, not the playhead's.
    const geometry = exportState.frameGeometry;
    if (!geometry) return false;
    const canvas = canvasRef.current;
    if (!canvas) return false;
    const bounds = canvas.getBoundingClientRect();
    const t = computeTransform(bounds.width, bounds.height, videoW, videoH, zoom, pan);
    const at = toSource(evt);
    const hit = exportState.marks.find((mark) => {
      const p = markOnSource(geometry, mark.atU, mark.atV);
      const c = sourceToCanvas(p.x, p.y, t, videoW, videoH);
      return Math.hypot(c.x - at.canvasX, c.y - at.canvasY) <= TARGET_GRAB_RADIUS;
    });
    if (!hit) return false;
    evt.preventDefault();
    suppressClickRef.current = true;
    exportState.setDragging(hit.id);
    // One canvas pixel is 1/t.scale source pixels, and a source pixel is
    // `output_warp.scale` output pixels.
    const perCanvasPixel = geometry.output_warp.scale / (t.scale || 1);
    // What is dragged is the crosshair, which for a following target sits along
    // the trail rather than on the stored mark. Taking that difference off the
    // pointer means the drag slides the whole trail — the crosshair stays under
    // the pointer, and the stored mark keeps meaning "where it is at the anchor
    // frame".
    const drift = { u: hit.atU - hit.u, v: hit.atV - hit.v };

    const destination = (ev: MouseEvent) => {
      const moved = toSource(ev);
      const raw = sourceToMark(geometry, moved.x, moved.y);
      const snapped = applySnap(hit.id, geometry, raw.u, raw.v, perCanvasPixel);
      return { u: clamp01(snapped.u - drift.u), v: clamp01(snapped.v - drift.v) };
    };
    const move = (ev: MouseEvent) => {
      const to = destination(ev);
      exportState.nudgeMark(hit.id, to.u, to.v);
    };
    const up = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      exportState.setDragging(null);
      const to = destination(ev);
      setSnapGuide(null);
      void exportState.commitMark(hit.id, to.u, to.v);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return true;
  }

  /** Convert a mouse event on the *output* canvas into output pixels. */
  const toOutput = useCallback((evt: { clientX: number; clientY: number }) => {
    const canvas = outputCanvasRef.current;
    const fit = outputFitRef.current;
    if (!canvas || !fit) return null;
    const bounds = canvas.getBoundingClientRect();
    const canvasX = evt.clientX - bounds.left;
    const canvasY = evt.clientY - bounds.top;
    const at = canvasToOutput(fit, canvasX, canvasY);
    return { fit, canvasX, canvasY, x: at.x, y: at.y };
  }, []);

  /** Whether a stabilisation crosshair is under this point on the output canvas. */
  const outputTargetAt = useCallback(
    (canvasX: number, canvasY: number, fit: OutputFit, geometry: FrameGeometry) =>
      exportState.marks.find((mark) => {
        const at = markOnOutput(geometry, mark.atU, mark.atV);
        const c = outputToCanvas(fit, at.x, at.y);
        return Math.hypot(c.x - canvasX, c.y - canvasY) <= TARGET_GRAB_RADIUS;
      }) ?? null,
    [exportState.marks],
  );

  /**
   * Drag a stabilisation target on the **output** canvas (todo 4, sixth round).
   *
   * The same gesture as `grabTarget` on the left, in the other coordinate space:
   * placing a crosshair here says "put this feature *there in the delivered
   * frame*", which is the thing the operator is actually composing and which the
   * left canvas can only express indirectly, inside a rectangle drawn on the
   * uncropped source. Both write through the same `nudgeMark` / `commitMark`, so
   * the crosshair moves on both canvases at once and the server hears about the
   * move exactly once, on release.
   *
   * The geometry is captured at grab time for the length of the gesture, as on
   * the left canvas: the transform under a drag must not change halfway through
   * it, or the crosshair slides out from under the pointer.
   */
  function grabOutputTarget(evt: React.MouseEvent<HTMLCanvasElement>): boolean {
    const geometry = exportState.frameGeometry;
    const at = toOutput(evt);
    if (!geometry || !at) return false;
    const hit = outputTargetAt(at.canvasX, at.canvasY, at.fit, geometry);
    if (!hit) return false;
    evt.preventDefault();
    exportState.setDragging(hit.id);
    // On this canvas the mapping to output pixels *is* the fit, so one canvas
    // pixel spans 1/sx of them.
    const perCanvasPixel = 1 / (at.fit.sx || 1);
    // As on the left canvas: the crosshair is what is grabbed, so its distance
    // from the stored mark comes off the pointer before the mark is stored.
    const drift = { u: hit.atU - hit.u, v: hit.atV - hit.v };

    const destination = (ev: MouseEvent) => {
      const moved = toOutput(ev);
      if (!moved) return null;
      const snapped = applySnap(
        hit.id,
        geometry,
        clamp01(moved.x / geometry.output.width),
        clamp01(moved.y / geometry.output.height),
        perCanvasPixel,
      );
      return { u: clamp01(snapped.u - drift.u), v: clamp01(snapped.v - drift.v) };
    };
    const move = (ev: MouseEvent) => {
      const to = destination(ev);
      if (to) exportState.nudgeMark(hit.id, to.u, to.v);
    };
    const up = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      exportState.setDragging(null);
      const to = destination(ev);
      setSnapGuide(null);
      if (!to) return;
      void exportState.commitMark(hit.id, to.u, to.v);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return true;
  }

  function handleOutputMouseMove(evt: React.MouseEvent<HTMLCanvasElement>) {
    const geometry = exportState.frameGeometry;
    const at = toOutput(evt);
    const on = Boolean(geometry && at && outputTargetAt(at.canvasX, at.canvasY, at.fit, geometry));
    setOutputHover((prev) => (prev === on ? prev : on));
  }

  /** Start a pan drag from this event. Middle-button, or the pan tool. */
  function startPanDrag(evt: React.MouseEvent<HTMLCanvasElement>) {
    evt.preventDefault();
    const wrap = wrapRef.current;
    if (!wrap) return;
    const dispW = wrap.clientWidth || 1;
    const dispH = wrap.clientHeight || 1;
    const startX = evt.clientX;
    const startY = evt.clientY;
    const startPan = pan;
    const startZoom = zoom;
    setPanning(true);
    const move = (ev: MouseEvent) => {
      const next = { x: startPan.x + (ev.clientX - startX), y: startPan.y + (ev.clientY - startY) };
      setPan(clampPan(next, startZoom, dispW, dispH, videoW, videoH));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      setPanning(false);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function handleMouseDown(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (!proxyReady) return;

    // Middle-mouse drag pans in every tool — it never collides with placing,
    // dragging or click-to-move, all of which are left-button.
    if (evt.button === 1) {
      startPanDrag(evt);
      return;
    }

    if (evt.button !== 0) return;
    if (tool === "pan") {
      suppressClickRef.current = true;
      startPanDrag(evt);
      return;
    }
    if (!overlayValid) return;

    // The export's crosshairs come first while its tab is up: they are the
    // thing being composed, and a target sits on top of the point it holds, so
    // whichever is grabbed has to be decided rather than left to draw order.
    if (exportState.active && grabTarget(evt)) return;

    if (tool !== "move") return;
    const { canvasX, canvasY } = toSource(evt);

    // An anchor comes before the marker underneath it (todo 4, sixth round).
    // It sits on top of the position it states, so which of the two a click
    // means has to be decided here rather than left to draw order — and an
    // anchor is the more deliberate thing to have aimed at.
    const pin = anchorHitTest(canvasX, canvasY);
    if (pin) {
      evt.preventDefault();
      suppressClickRef.current = true;
      setSelectedPointId(pin.pointId);
      setAnchorDrag({
        pointId: pin.pointId,
        frame: pin.query.frame,
        toX: pin.query.x,
        toY: pin.query.y,
      });
      const movePin = (ev: MouseEvent) => {
        const p = toSource(ev);
        setAnchorDrag((prev) => (prev ? { ...prev, toX: p.x, toY: p.y } : prev));
      };
      const dropPin = (ev: MouseEvent) => {
        window.removeEventListener("mousemove", movePin);
        window.removeEventListener("mouseup", dropPin);
        setAnchorDrag(null);
        const p = toSource(ev);
        // An anchor dropped where it already was is not an edit, and re-storing
        // it would re-run every tracking decision that depends on it for nothing.
        if (Math.hypot(p.x - pin.query.x, p.y - pin.query.y) < 0.5) return;
        void addAnchorAt(pin.pointId, pin.query.frame, p.x, p.y);
      };
      window.addEventListener("mousemove", movePin);
      window.addEventListener("mouseup", dropPin);
      return;
    }

    const pointId = hitTest(canvasX, canvasY);
    // Only the selected track is draggable: with several tracks on screen,
    // grabbing whichever marker happens to be under the cursor is how a scrub
    // turns into an edit nobody asked for (todo 25's one real risk).
    if (!pointId || pointId !== selectedPointId) return;
    const track = tracks[pointId];
    const fromX = track.x[viewFrame]!;
    const fromY = track.y[viewFrame]!;
    evt.preventDefault();
    suppressClickRef.current = true;
    setSelectedPointId(pointId);
    setDrag({ pointId, fromX, fromY, toX: fromX, toY: fromY });

    const move = (ev: MouseEvent) => {
      const p = toSource(ev);
      setDrag((prev) => (prev ? { ...prev, toX: p.x, toY: p.y } : prev));
    };
    const up = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      const p = toSource(ev);
      setDrag(null);
      commitMove(pointId, p.x, p.y, ev.shiftKey);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  /**
   * What the pointer is over, so the cursor can say what a click would do.
   *
   * There was no hover state at all before (todo 4): the cursor came from the
   * mode and from "is a track selected", so it was a grab hand over the whole
   * canvas including bare picture where a click would *move* the point rather
   * than grab anything. State only changes when the answer changes, so this is
   * a cheap hit test per mouse move rather than a re-render per pixel.
   */
  function handleMouseMove(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (!proxyReady) return;
    const { canvasX, canvasY } = toSource(evt);
    const geometry = exportState.frameGeometry;
    if (exportState.active && geometry && overlayValid) {
      const canvas = canvasRef.current;
      if (canvas) {
        const bounds = canvas.getBoundingClientRect();
        const t = computeTransform(bounds.width, bounds.height, videoW, videoH, zoom, pan);
        const onTarget = exportState.marks.some((mark) => {
          const p = markOnSource(geometry, mark.atU, mark.atV);
          const c = sourceToCanvas(p.x, p.y, t, videoW, videoH);
          return Math.hypot(c.x - canvasX, c.y - canvasY) <= TARGET_GRAB_RADIUS;
        });
        if (onTarget) {
          setHoverPointId((prev) => (prev === TARGET_HOVER ? prev : TARGET_HOVER));
          return;
        }
      }
    }
    // Same precedence as the mousedown: the anchor wins over the marker under
    // it, so the cursor promises what the click will actually do.
    if (overlayValid && tool === "move" && anchorHitTest(canvasX, canvasY)) {
      setHoverPointId((prev) => (prev === ANCHOR_HOVER ? prev : ANCHOR_HOVER));
      return;
    }
    const found = overlayValid ? hitTest(canvasX, canvasY) : null;
    setHoverPointId((prev) => (prev === found ? prev : found));
  }

  // The cursor is the answer to "what does a click here do?" (todo 4).
  const cursorClass = (() => {
    if (drag || anchorDrag || panning) return "viewer__canvas-wrap--dragging";
    if (tool === "pan") return "viewer__canvas-wrap--grab";
    if (tool === "add-point" || tool === "add-anchor") return "viewer__canvas-wrap--placing";
    if (tool === "delete") {
      return hoverPointId ? "viewer__canvas-wrap--delete" : "viewer__canvas-wrap--inert";
    }
    if (hoverPointId === TARGET_HOVER || hoverPointId === ANCHOR_HOVER) {
      return "viewer__canvas-wrap--grab";
    }
    if (hoverPointId) {
      // The selected track is the draggable one; any other is a click away from
      // *being* the selected one, which is a pointer, not a grab.
      return hoverPointId === selectedPointId
        ? "viewer__canvas-wrap--grab"
        : "viewer__canvas-wrap--select";
    }
    return selectedPointId && overlayValid && !exportState.active
      ? "viewer__canvas-wrap--placing"
      : "";
  })();

  // One line saying what the frame is currently for. It is the same question
  // the cursor answers, in words, and only one of these can be true at a time.
  const hint = (() => {
    if (!overlayValid) return "Turn the stabilized preview off to edit points on the frame";
    if (deleteHint) return deleteHint;
    if (tool === "add-point") return "Click on the video to place a new track";
    if (tool === "add-anchor") {
      return selectedPointId
        ? `Click on the video to anchor ${selectedPointId} at frame ${viewFrame} — the tool stays on, so scrub and click again for the next one (Esc when done)`
        : "Select a track first — an anchor belongs to one";
    }
    if (tool === "delete") {
      return "Click an anchor to remove it, or a track's marker to remove the whole track";
    }
    if (tool === "pan") return "Drag to pan. Wheel or +/− zooms, 0 fits.";
    if (exportState.active) {
      return exportState.marks.length
        ? `Drag the crosshair to say where the target sits in the delivered frame${
            exportState.snap ? " — it snaps to the centre, the thirds and other points" : ""
          }. The dimmed area is what the export throws away.`
        : "The dimmed area is what the export throws away. Pick a stabilisation target in the panel to compose the shot.";
    }
    if (anchorDrag) return `Moving ${anchorDrag.pointId}'s anchor at frame ${anchorDrag.frame}`;
    if (hoverPointId === ANCHOR_HOVER) {
      return "Drag this anchor to move it, or use the delete tool to remove it";
    }
    if (selectedPointId) {
      return `Click to move ${selectedPointId}, or drag it directly — ${
        dragMode === "replace" ? "hard replace at this frame" : `offset, ${falloff}f falloff`
      } (Shift to replace, Ctrl+Z to undo)`;
    }
    return null;
  })();

  // The refusal is about one click, so it must not outlive the next one.
  useEffect(() => {
    if (!deleteHint) return;
    const timer = setTimeout(() => setDeleteHint(null), 4000);
    return () => clearTimeout(timer);
  }, [deleteHint]);

  // Whether any run is currently reporting a cursor — the only time the
  // follow toggle means anything.
  const liveRunning = Object.values(liveTracking).some((live) => live.frame != null);
  const zoomPct = Math.round(zoom * 100);
  const atFit = zoom === MIN_ZOOM && pan.x === 0 && pan.y === 0;

  return (
    <div className="viewer">
      <div className="viewer__toolbar" style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {/* What a click on the frame does, said out loud (todo 4). The two
            placing modes used to be reachable only from buttons buried in the
            Tracks panel, and everything else was one unnamed default. */}
        <div className="viewer__tools" role="radiogroup" aria-label="Canvas tool">
          {TOOLS.map(({ id, label, key, help }) => (
            <button
              key={id}
              className={tool === id ? "button--active" : ""}
              aria-pressed={tool === id}
              title={`${help} (${key})`}
              onClick={() => setTool(id)}
            >
              {label}
            </button>
          ))}
        </div>
        {/* The toolbar is four unrelated things in a row — what a click does,
            how close you are looking, what a drag means, and what is drawn on
            top — and a shared gap said none of that (todo 9, ninth round). */}
        <span className="viewer__sep" aria-hidden="true" />
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <button
            onClick={() => {
              const wrap = wrapRef.current;
              if (wrap) zoomAtPoint(wrap.clientWidth / 2, wrap.clientHeight / 2, 1 / BUTTON_ZOOM_FACTOR);
            }}
            disabled={zoom <= MIN_ZOOM}
            title="Zoom out"
          >
            −
          </button>
          <span style={{ minWidth: 40, textAlign: "center", fontSize: "0.8rem", color: "var(--text-dim)" }}>
            {zoomPct}%
          </span>
          <button
            onClick={() => {
              const wrap = wrapRef.current;
              if (wrap) zoomAtPoint(wrap.clientWidth / 2, wrap.clientHeight / 2, BUTTON_ZOOM_FACTOR);
            }}
            disabled={zoom >= MAX_ZOOM}
            title="Zoom in"
          >
            +
          </button>
          <button onClick={resetView} disabled={atFit} title="Reset zoom/pan (0)">
            Fit
          </button>
        </div>
        <span className="viewer__sep" aria-hidden="true" />
        {/* What a drag on the frame means (todo 4, sixth round). It was on the
            Edit tab, three clicks away from the gesture it governs and on a
            panel you had to leave to see the picture; the drag happens here, so
            the choice does too. Both are non-destructive corrections — the
            difference is only which one gets appended. */}
        <div
          className="viewer__drag-mode"
          title="What dragging or clicking a point does. Offset nudges the track and blends the nudge out either side; hard replace sets this one frame exactly. Shift always forces replace."
        >
          <span>Drag makes</span>
          <select value={dragMode} onChange={(e) => setDragMode(e.target.value as typeof dragMode)}>
            <option value="offset">offset (blended)</option>
            <option value="replace">hard replace</option>
          </select>
          {dragMode === "offset" && (
            <input
              type="number"
              min={0}
              value={falloff}
              onChange={(e) => setFalloff(Math.max(0, Number(e.target.value)))}
              title="Frames the offset blends out over, either side of the drag"
            />
          )}
        </div>
        <span className="viewer__sep" aria-hidden="true" />
        <div className="channel-checks">
          <label className="channel-check" title="Hide every track's overlay to see the frame underneath">
            <input
              type="checkbox"
              checked={overlaysEnabled}
              onChange={(e) => setOverlaysEnabled(e.target.checked)}
            />
            Overlays
          </label>
          {liveRunning && (
            <label
              className="channel-check"
              title="Keep the playhead on the frame the tracker is reading, so you can watch it work. Scrubbing releases it."
            >
              <input
                type="checkbox"
                checked={followTracker}
                onChange={(e) => setFollowTracker(e.target.checked)}
              />
              Follow tracker
            </label>
          )}
          {/* Confidence colouring answers "is this track any good"; the point's
              own colour answers "which track is this". Both matter and one
              canvas can only show one, so it is a choice rather than a default
              (todo 5, second round). The timeline heat strip stays confidence-
              coloured either way, so the signal is never entirely lost. */}
          <label
            className="channel-check"
            title="Colour each track's path and marker by its confidence (green = trust it, red = check it), or by the track's own colour so several tracks can be told apart. The timeline strip always shows confidence."
          >
            {/* "Paths [by confidence]" read as a noun with an adverb bolted on
                — it never said what the control *did* (todo 8, ninth round). */}
            Colour Paths by
            <select
              value={pathColor}
              onChange={(e) => setPathColor(e.target.value as typeof pathColor)}
              disabled={!overlaysEnabled}
            >
              <option value="confidence">confidence</option>
              <option value="track">track</option>
            </select>
          </label>
          {/* These are the *defaults*: a track card can override any of them
              for itself, which is what "show/hide UI items for a given track"
              needs once a shot has more than one or two (todo 9). */}
          {OVERLAY_LABELS.map(([key, label, help]) => (
            <label
              key={key}
              className="channel-check"
              style={{ opacity: overlaysEnabled ? 1 : 0.4 }}
              title={help}
            >
              <input
                type="checkbox"
                checked={overlays[key]}
                disabled={!overlaysEnabled}
                onChange={(e) => setOverlays({ ...overlays, [key]: e.target.checked })}
              />
              {label}
            </label>
          ))}
        </div>
      </div>
      {/* The picture, and — while the export tab is up — what the export would
          write beside it (todo 1, fifth round). The left canvas is unchanged:
          same overlay, same zoom, same pan, same tools. */}
      <div className="viewer__canvases">
        <div ref={wrapRef} className={`viewer__canvas-wrap ${cursorClass}`}>
          {proxyReady ? (
            <canvas
              ref={canvasRef}
              onClick={handleClick}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverPointId(null)}
            />
          ) : (
            <div className="viewer__placeholder">Generating proxy frames…</div>
          )}
          {error && <div className="viewer__error">{error}</div>}
          {hint && <div className="viewer__hint">{hint}</div>}
        </div>
        {showOutput && (
          <div
            className={`viewer__canvas-wrap viewer__canvas-wrap--output ${
              exportState.dragging
                ? "viewer__canvas-wrap--dragging"
                : outputHover
                  ? "viewer__canvas-wrap--grab"
                  : ""
            }`}
          >
            <canvas
              ref={outputCanvasRef}
              onMouseDown={(e) => {
                if (e.button === 0) grabOutputTarget(e);
              }}
              onMouseMove={handleOutputMouseMove}
              onMouseLeave={() => setOutputHover(false)}
            />
            <div className="viewer__canvas-label">
              output
              {exportState.preview
                ? ` · ${exportState.preview.output.width}×${exportState.preview.output.height}`
                : ""}
            </div>
            {outputError && <div className="viewer__error">{outputError}</div>}
            {exportState.marks.length > 0 && (
              <div className="viewer__hint">
                Drag a crosshair here to place it in the delivered frame.
              </div>
            )}
          </div>
        )}
      </div>
      <div className="viewer__frame-label">
        frame {currentFrame} / {Math.max(project.video.frames - 1, 0)}
        {" · "}
        <button
          className="link-button"
          onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))}
        >
          ◀
        </button>
        <button
          className="link-button"
          onClick={() => setCurrentFrame(Math.min(project.video.frames - 1, currentFrame + 1))}
        >
          ▶
        </button>
        <span style={{ marginLeft: 8, color: "var(--text-dim)" }}>
          ←/→ frame · space play/pause · ↑/↓ next/previous anchor · V/N/A/D/H tools (Esc = move) ·
          A anchor here (⇧A click to place) · I/O set in-out (⇧I/⇧O go there) · wheel/+/- zoom ·
          middle-drag pan
        </span>
      </div>
    </div>
  );
}
