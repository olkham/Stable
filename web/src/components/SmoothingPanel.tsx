import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "../api/client";
import { CHANNEL_COLOR, CHANNEL_LABEL } from "../colors";
import { isActive, joinSpecs, splitSpecs, type SmoothingModel } from "../smoothing";
import { useProjectContext } from "../state/ProjectContext";
import type { Channel, FilterInfo } from "../types";
import { CHANNELS } from "../types";

/**
 * Per-channel filter picker with a raw-vs-smoothed preview.
 *
 * The filter list and every parameter's type, default and bounds come from
 * `GET /api/filters` — the server's own catalogue (stable/smoothing/filters.py).
 * Nothing about a filter is hardcoded here, so adding one to the registry makes
 * it appear with the right controls, and a rename can never silently desync the
 * two halves of the app.
 *
 * The sub-range and pins a channel may also carry belong to the curve editor;
 * this panel preserves them untouched when it writes the filter list back.
 *
 * The filter list and the curves are the context's, not this panel's (todo 2,
 * fourth round). They used to be local state fetched here and again in the
 * curve editor, so a change made in one was invisible to the other until it
 * remounted — "the curves view doesn't get updates until you close and reopen
 * it". Both now read and write the same copy.
 */
export function SmoothingPanel() {
  const { currentFrame, setCurrentFrame, smoothing, smoothingError, commitSmoothing } =
    useProjectContext();

  const [catalogue, setCatalogue] = useState<FilterInfo[]>([]);

  useEffect(() => {
    api
      .getFilters()
      .then(setCatalogue)
      .catch(() => setCatalogue([]));
  }, []);

  const raw = smoothing?.curves.raw ?? null;
  const smoothed = smoothing?.curves.smoothed ?? null;
  const model = useMemo(
    () => (smoothing ? splitSpecs(smoothing.filters) : null),
    [smoothing],
  );
  const error = smoothingError;

  const defaultsFor = useCallback(
    (name: string): Record<string, number> => {
      const info = catalogue.find((entry) => entry.name === name);
      const params: Record<string, number> = {};
      for (const param of info?.params ?? []) params[param.name] = param.default;
      return params;
    },
    [catalogue],
  );

  function update(channel: Channel, mutate: (config: SmoothingModel[Channel]) => void) {
    if (!model) return;
    const next: SmoothingModel = { ...model, [channel]: { ...model[channel] } };
    next[channel].filters = next[channel].filters.map((spec) => ({
      ...spec,
      params: { ...spec.params },
    }));
    mutate(next[channel]);
    void commitSmoothing(joinSpecs(next));
  }

  if (!model) {
    return (
      <div className="panel smoothing-panel">
        <div className="panel__header">
          <h3>Smoothing</h3>
        </div>
        {error ? (
          <div className="error-text">{error}</div>
        ) : (
          <p className="panel__empty">Loading…</p>
        )}
      </div>
    );
  }

  return (
    <div className="panel smoothing-panel">
      <div className="panel__header">
        <h3>Smoothing</h3>
      </div>
      {error && <div className="error-text">{error}</div>}
      {!raw && <p className="panel__empty">Run solve first to preview smoothing.</p>}

      {CHANNELS.map((channel) => {
        const config = model[channel];
        const spec = config.filters[0];
        const info = catalogue.find((entry) => entry.name === spec?.filter);
        return (
          <div key={channel} className="smoothing-channel">
            <div className="smoothing-channel__row">
              <label className="channel-check">
                <input
                  type="checkbox"
                  checked={isActive(config)}
                  onChange={(e) =>
                    update(channel, (next) => {
                      if (!e.target.checked) {
                        next.filters = next.filters.map((f) => ({ ...f, enabled: false }));
                        return;
                      }
                      next.filters = next.filters.length
                        ? next.filters.map((f) => ({ ...f, enabled: true }))
                        : [
                            {
                              channel,
                              filter: "butterworth",
                              params: defaultsFor("butterworth"),
                              enabled: true,
                            },
                          ];
                    })
                  }
                />
                <span style={{ color: CHANNEL_COLOR[channel] }}>{CHANNEL_LABEL[channel]}</span>
              </label>
              <select
                value={spec?.filter ?? "butterworth"}
                onChange={(e) =>
                  update(channel, (next) => {
                    next.filters = [
                      {
                        channel,
                        filter: e.target.value,
                        params: defaultsFor(e.target.value),
                        enabled: true,
                      },
                    ];
                  })
                }
              >
                {catalogue.map((entry) => (
                  <option key={entry.name} value={entry.name} title={entry.description}>
                    {entry.display_name}
                  </option>
                ))}
              </select>
            </div>

            {spec && info && (
              <div className="smoothing-channel__params">
                {info.params.map((param) => (
                  <label key={param.name} className="param-input" title={param.description}>
                    {param.name}
                    {param.unit ? ` (${param.unit})` : ""}
                    <input
                      type="number"
                      step={param.kind === "int" ? 1 : 0.1}
                      min={param.minimum ?? undefined}
                      max={param.maximum ?? undefined}
                      value={spec.params[param.name] ?? param.default}
                      onChange={(e) =>
                        update(channel, (next) => {
                          next.filters[0] = {
                            ...next.filters[0],
                            params: {
                              ...next.filters[0].params,
                              [param.name]: Number(e.target.value),
                            },
                          };
                        })
                      }
                    />
                  </label>
                ))}
              </div>
            )}

            {(config.range || config.pins.length > 0) && (
              <p className="modal__hint">
                {config.range ? `range ${config.range[0]}–${config.range[1]}` : "whole clip"}
                {config.pins.length > 0 ? ` · ${config.pins.length} pinned` : ""} — set in the curve
                editor
              </p>
            )}

            {raw && (
              <CurvePreview
                raw={raw[channel]}
                smoothed={smoothed ? smoothed[channel] : undefined}
                color={CHANNEL_COLOR[channel]}
                cursor={currentFrame}
                onSeek={setCurrentFrame}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/** A small raw-vs-smoothed sparkline for one channel. */
function CurvePreview({
  raw,
  smoothed,
  color,
  cursor,
  onSeek,
}: {
  raw: number[];
  smoothed: number[] | undefined;
  color: string;
  cursor: number;
  onSeek: (frame: number) => void;
}) {
  const width = 480;
  const height = 56;
  if (raw.length < 2) return null;
  const all = (smoothed ? raw.concat(smoothed) : raw).filter((v) => Number.isFinite(v));
  if (all.length === 0) return null;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const toPoints = (values: number[]) =>
    values
      .map((v, i) => {
        const x = (i / (values.length - 1)) * width;
        const y = height - ((v - min) / span) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  const cursorX = (cursor / (raw.length - 1)) * width;

  // Click jumps to a frame; holding and dragging scrubs through frames in real
  // time, the same as the main timeline (todo 21). A plain click is just a
  // drag with zero movement, so one handler covers both.
  function seek(clientX: number, svg: SVGSVGElement) {
    const rect = svg.getBoundingClientRect();
    const frac = (clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(raw.length - 1, Math.round(frac * (raw.length - 1)))));
  }

  function handleMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    const svg = e.currentTarget;
    seek(e.clientX, svg);
    const move = (ev: MouseEvent) => seek(ev.clientX, svg);
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  return (
    <svg
      className="curve-chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      onMouseDown={handleMouseDown}
    >
      <polyline
        points={toPoints(raw)}
        fill="none"
        stroke="#888"
        strokeDasharray="3,2"
        strokeWidth={1}
      />
      {smoothed && (
        <polyline points={toPoints(smoothed)} fill="none" stroke={color} strokeWidth={1.75} />
      )}
      <line
        x1={cursorX}
        x2={cursorX}
        y1={0}
        y2={height}
        stroke="#fff"
        strokeWidth={1}
        opacity={0.6}
      />
    </svg>
  );
}
