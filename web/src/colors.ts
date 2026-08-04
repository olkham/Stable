/** Shared colour vocabulary: confidence, point identity, and the four channels. */

import type { Channel } from "./types";

/** Channel labels and colours, shared by the smoothing panel and curve editor. */
export const CHANNEL_LABEL: Record<Channel, string> = {
  x: "X",
  y: "Y",
  r: "Rotation",
  s: "Scale",
};

export const CHANNEL_COLOR: Record<Channel, string> = {
  x: "#5b9bd5",
  y: "#ed7d31",
  r: "#70ad47",
  s: "#d64545",
};

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

const LOW: [number, number, number] = [214, 69, 69]; // red — low confidence
const MID: [number, number, number] = [214, 176, 58]; // amber — medium
const HIGH: [number, number, number] = [69, 176, 110]; // green — high confidence

/** Map a confidence in [0, 1] onto a red -> amber -> green colour, as `rgb(...)`. */
export function confidenceColor(confidence: number): string {
  const c = Math.max(0, Math.min(1, confidence));
  const [from, to, t] = c < 0.5 ? [LOW, MID, c / 0.5] : [MID, HIGH, (c - 0.5) / 0.5];
  const r = Math.round(lerp(from[0], to[0], t));
  const g = Math.round(lerp(from[1], to[1], t));
  const b = Math.round(lerp(from[2], to[2], t));
  return `rgb(${r}, ${g}, ${b})`;
}

/** A stable, readable colour per point id, for distinguishing points on the canvas. */
const PALETTE = [
  "#5b9bd5",
  "#ed7d31",
  "#a5a5a5",
  "#ffc000",
  "#70ad47",
  "#c00000",
  "#7030a0",
  "#00b0f0",
];

export function pointColor(pointId: string): string {
  let hash = 0;
  for (let i = 0; i < pointId.length; i++) hash = (hash * 31 + pointId.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(hash) % PALETTE.length];
}
