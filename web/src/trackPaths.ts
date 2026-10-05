// Track paths as cached `Path2D`s, in source-video pixels.
//
// The viewer redraws on every frame of playback, every hover change and every
// mouse move of a drag, and a path is the whole clip: tens of thousands of
// segments per track on a long shot. Building them in canvas pixels meant
// walking every frame of every track on every redraw, and the confidence
// colouring stroked each segment on its own with a freshly formatted colour —
// which was most of what a redraw cost (REVIEW.md P2).
//
// So a path is built once per loaded track, in *source* pixels, and the viewer
// strokes it through its view transform (`strokeInView`): zooming, panning and
// a new frame cost no path work at all. Confidence is shaded in a fixed number
// of levels, one `Path2D` per level, so a track is at most that many strokes
// however long it is. At 32 levels the steps are far finer than the eye can
// tell apart on a red→amber→green ramp.

import { confidenceColor } from "./colors";

/** How many confidence shades a path is drawn in. */
export const CONFIDENCE_LEVELS = 32;

/** The colour of each shade, formatted once rather than per segment. */
export const CONFIDENCE_LEVEL_COLORS: readonly string[] = Array.from(
  { length: CONFIDENCE_LEVELS },
  (_, level) => confidenceColor(level / (CONFIDENCE_LEVELS - 1)),
);

type Coordinates = readonly (number | null)[];

const polylines = new WeakMap<Coordinates, Path2D>();
const shadedPaths = new WeakMap<Coordinates, (Path2D | null)[]>();

/**
 * One continuous path through every frame that has a position, broken at gaps.
 *
 * Cached on the `xs` array: a reloaded track, a new smoothing or a new live
 * update arrives as a new array, so a stale path can never be returned.
 */
export function polylinePath(xs: Coordinates, ys: Coordinates): Path2D {
  const cached = polylines.get(xs);
  if (cached) return cached;
  const path = new Path2D();
  let started = false;
  const count = Math.min(xs.length, ys.length);
  for (let f = 0; f < count; f++) {
    const x = xs[f];
    const y = ys[f];
    if (x == null || y == null) {
      started = false;
      continue;
    }
    if (started) path.lineTo(x, y);
    else path.moveTo(x, y);
    started = true;
  }
  polylines.set(xs, path);
  return path;
}

/**
 * The same path split by confidence: entry `k` holds every segment shaded at
 * level `k` (`null` where no segment is), to be stroked in
 * `CONFIDENCE_LEVEL_COLORS[k]`.
 *
 * A segment joins frame `f - 1` to frame `f` and is shaded by the mean of their
 * confidences, exactly as the per-segment drawing it replaces was.
 */
export function confidencePaths(
  xs: Coordinates,
  ys: Coordinates,
  confidence: readonly number[],
): (Path2D | null)[] {
  const cached = shadedPaths.get(xs);
  if (cached) return cached;
  const paths: (Path2D | null)[] = new Array(CONFIDENCE_LEVELS).fill(null);
  const count = Math.min(xs.length, ys.length, confidence.length);
  for (let f = 1; f < count; f++) {
    const x0 = xs[f - 1];
    const y0 = ys[f - 1];
    const x1 = xs[f];
    const y1 = ys[f];
    if (x0 == null || y0 == null || x1 == null || y1 == null) continue;
    const mean = (confidence[f - 1] + confidence[f]) / 2;
    const level = Math.round(Math.max(0, Math.min(1, mean)) * (CONFIDENCE_LEVELS - 1));
    const path = paths[level] ?? (paths[level] = new Path2D());
    path.moveTo(x0, y0);
    path.lineTo(x1, y1);
  }
  shadedPaths.set(xs, paths);
  return paths;
}

/** Where source pixels land on a canvas: `canvas = source × scale + offset`. */
export interface SourceToCanvas {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/**
 * Stroke a source-pixel path at a fixed on-screen width and dash.
 *
 * The context's own transform is replaced for the stroke and restored after,
 * so line widths and dashes are stated in CSS pixels as everywhere else.
 * `strokeStyle` is whatever the caller set.
 */
export function strokeInView(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  view: SourceToCanvas,
  dpr: number,
  lineWidth: number,
  dash: readonly number[] = [],
): void {
  const scale = view.scale || 1;
  ctx.save();
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * view.offsetX, dpr * view.offsetY);
  ctx.lineWidth = lineWidth / scale;
  ctx.setLineDash(dash.map((length) => length / scale));
  ctx.stroke(path);
  ctx.restore();
}
