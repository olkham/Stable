// The project file stores smoothing as one flat, ordered list of
// {channel, filter, params, enabled} entries (schema v1, locked in Phase 0).
// Two of those "filter" names are reserved directives rather than filters:
// "@range" carries a sub-range, "@pin" carries one pinned keyframe — see
// stable/smoothing/channels.py.
//
// The smoothing panel edits filters and the curve editor edits ranges and pins,
// but both PUT the *whole* list. Splitting it into a per-channel model here and
// rebuilding it from that model is what stops either panel silently wiping the
// other's settings.

import type { Channel, SmoothingFilter } from "./types";
import { CHANNELS } from "./types";

export const RANGE_DIRECTIVE = "@range";
export const PIN_DIRECTIVE = "@pin";
export const DIRECTIVES: string[] = [RANGE_DIRECTIVE, PIN_DIRECTIVE];

export const DEFAULT_PIN_RADIUS = 20;
export const DEFAULT_RANGE_BLEND = 5;

export interface ChannelConfig {
  /** The channel's filter chain, in order. The UI keeps it to one entry. */
  filters: SmoothingFilter[];
  /** Inclusive [start, end] the chain applies to, or null for the whole clip. */
  range: [number, number] | null;
  rangeBlend: number;
  /** Frames the smoothed curve must pass through exactly. */
  pins: number[];
  pinRadius: number;
}

export type SmoothingModel = Record<Channel, ChannelConfig>;

function emptyChannel(): ChannelConfig {
  return {
    filters: [],
    range: null,
    rangeBlend: DEFAULT_RANGE_BLEND,
    pins: [],
    pinRadius: DEFAULT_PIN_RADIUS,
  };
}

export function emptyModel(): SmoothingModel {
  return CHANNELS.reduce((model, channel) => {
    model[channel] = emptyChannel();
    return model;
  }, {} as SmoothingModel);
}

/** Group a flat project-file spec list into the per-channel model. */
export function splitSpecs(specs: SmoothingFilter[]): SmoothingModel {
  const model = emptyModel();
  for (const spec of specs) {
    const channel = model[spec.channel];
    if (!channel) continue;
    if (spec.filter === RANGE_DIRECTIVE) {
      if (!spec.enabled) continue;
      const start = Number(spec.params.start ?? 0);
      const end = Number(spec.params.end ?? 0);
      channel.range = [start, end];
      channel.rangeBlend = Number(spec.params.blend ?? DEFAULT_RANGE_BLEND);
    } else if (spec.filter === PIN_DIRECTIVE) {
      if (!spec.enabled) continue;
      const frame = Number(spec.params.frame ?? 0);
      channel.pinRadius = Number(spec.params.radius ?? channel.pinRadius);
      if (!channel.pins.includes(frame)) channel.pins.push(frame);
    } else {
      channel.filters.push({ ...spec, params: { ...spec.params } });
    }
  }
  for (const channel of CHANNELS) model[channel].pins.sort((a, b) => a - b);
  return model;
}

/** Flatten the per-channel model back into the list the API expects. */
export function joinSpecs(model: SmoothingModel): SmoothingFilter[] {
  const specs: SmoothingFilter[] = [];
  for (const channel of CHANNELS) {
    const config = model[channel];
    for (const spec of config.filters) {
      specs.push({ ...spec, channel, params: { ...spec.params } });
    }
    if (config.range) {
      specs.push({
        channel,
        filter: RANGE_DIRECTIVE,
        params: {
          start: config.range[0],
          end: config.range[1],
          blend: config.rangeBlend,
        },
        enabled: true,
      });
    }
    for (const frame of config.pins) {
      specs.push({
        channel,
        filter: PIN_DIRECTIVE,
        params: { frame, radius: config.pinRadius },
        enabled: true,
      });
    }
  }
  return specs;
}

/** Whether a channel will actually be smoothed (and therefore stabilised). */
export function isActive(config: ChannelConfig): boolean {
  return config.filters.some((spec) => spec.enabled);
}
