import { useMemo } from "react";
import type { JobEvent, JobPartial } from "../types";

/**
 * What a tracking job is doing right now, in the form the viewer and timeline
 * want it (todos 4 and 16).
 *
 * A run over a long clip takes minutes. Rather than a percentage and a wait,
 * the server publishes two things as it goes — the frame its decoder is on, and
 * the path found so far, one completed run at a time (see `JobPartial`). This
 * turns those into per-point state, so the operator can scrub back over frames
 * the tracker has already covered and watch it work instead of reviewing it
 * afterwards.
 */
export interface LiveTracking {
  /** The point being tracked. */
  pointId: string;
  /** Which kind of run it is, for wording in the UI. */
  kind: "track" | "retrack";
  /** The frame the tracker's decoder most recently read, if known. */
  frame: number | null;
  /** The span of frames it has touched, for the timeline's progress shading. */
  span: { from: number; to: number } | null;
  /** Positions found so far; `null` at frames not yet covered. */
  x: (number | null)[] | null;
  y: (number | null)[] | null;
  /** Completed planned runs, e.g. 1 of 2 (forwards done, backwards to go). */
  runsDone: number;
  runsTotal: number;
}

function toLive(job: JobEvent): LiveTracking | null {
  if (!job.target) return null;
  const partial: JobPartial = job.partial ?? {};
  const span =
    partial.frame_min != null && partial.frame_max != null
      ? { from: partial.frame_min, to: partial.frame_max }
      : null;
  return {
    pointId: job.target,
    kind: job.kind === "retrack" ? "retrack" : "track",
    frame: partial.frame ?? null,
    span,
    x: partial.track?.x ?? null,
    y: partial.track?.y ?? null,
    runsDone: partial.runs_done ?? 0,
    runsTotal: partial.runs_total ?? 0,
  };
}

/**
 * Index every in-flight track/retrack job by the point it is tracking.
 *
 * Keyed by point rather than by job because that is how every consumer asks the
 * question ("is *this* point being tracked right now?"), and because a point
 * only ever has one run in flight — the panel disables its buttons while one is
 * going. If two ever overlapped, the most recently updated wins, which is the
 * one whose partial results are freshest.
 */
export function useLiveTracking(jobs: Record<string, JobEvent>): Record<string, LiveTracking> {
  return useMemo(() => {
    const out: Record<string, LiveTracking> = {};
    const seenAt: Record<string, number> = {};
    for (const job of Object.values(jobs)) {
      if (job.status !== "running") continue;
      if (job.kind !== "track" && job.kind !== "retrack") continue;
      const live = toLive(job);
      if (!live) continue;
      if (seenAt[live.pointId] != null && seenAt[live.pointId] > job.updated_at) continue;
      seenAt[live.pointId] = job.updated_at;
      out[live.pointId] = live;
    }
    return out;
  }, [jobs]);
}
