// Which frames of a solve were measured, and which the solver filled in.
//
// A frame with no usable track — none with a position and enough confidence —
// is interpolated (or held at the clip's ends) by stable/solver/motion.py. The
// camera curves read as smoothly there as anywhere, so a smoothed curve that is
// flat across those frames looks like a feature held still, while the
// stabilised trail on the canvas, which warps the track itself, shows the
// feature moving. Both are right; these helpers are how the curve views say
// which frames the curves actually know about.

import type { SolveStatus } from "./types";

/** Statuses whose camera motion was filled in rather than measured. */
const UNMEASURED: ReadonlySet<SolveStatus> = new Set<SolveStatus>([
  "interpolated",
  "held",
  "identity",
]);

/** Whether a frame's camera motion was filled in rather than measured. */
export function isUnmeasured(status: SolveStatus | undefined): boolean {
  return status !== undefined && UNMEASURED.has(status);
}

/**
 * Runs of frames whose camera motion the solve filled in, as inclusive
 * `[first, last]` pairs, in frame order.
 */
export function unmeasuredRuns(status: readonly SolveStatus[] | undefined): [number, number][] {
  const runs: [number, number][] = [];
  if (!status) return runs;
  let start = -1;
  for (let frame = 0; frame <= status.length; frame++) {
    const filled = frame < status.length && isUnmeasured(status[frame]);
    if (filled && start < 0) start = frame;
    if (!filled && start >= 0) {
      runs.push([start, frame - 1]);
      start = -1;
    }
  }
  return runs;
}
