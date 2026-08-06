import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CHANNEL_COLOR, CHANNEL_LABEL } from "../colors";
import { joinSpecs, splitSpecs, type ChannelConfig, type SmoothingModel } from "../smoothing";
import { useProjectContext } from "../state/ProjectContext";
import type { Channel, SmoothedCurves } from "../types";
import { CHANNELS } from "../types";

const LANE_HEIGHT = 84;
const LANE_GAP = 8;
const RULER_HEIGHT = 18;
const PAD_LEFT = 46;

interface View {
  /** First and last frame currently drawn (fractional while zooming). */
  start: number;
  end: number;
}

/**
 * The X/Y/R/S curve editor.
 *
 * One lane per channel, because the four channels are in different units —
 * pixels, radians and a unitless scale — and overlaying them on a shared axis
 * makes every readout a lie. Each lane auto-fits its own visible range, so a
 * half-degree rotation wobble is as readable as a 40-pixel pan.
 *
 * Interactions: wheel zooms the time axis about the cursor, dragging pans it,
 * shift-drag selects a frame range, clicking a lane moves the playhead. The
 * selected range and the pinned keyframes are per channel and are stored in the
 * project file as smoothing directives, so they survive a reload.
 *
 * The curves themselves come from the context (todo 2, fourth round), which is
 * also what the smoothing panel writes through — so a filter changed over there
 * redraws these lanes at once instead of on the next remount.
 */
export function CurveEditor() {
  const { project, currentFrame, setCurrentFrame, smoothing, smoothingError, commitSmoothing } =
    useProjectContext();

  const [visible, setVisible] = useState<Record<Channel, boolean>>({
    x: true,
    y: true,
    r: true,
    s: true,
  });
  const [view, setView] = useState<View>({ start: 0, end: Math.max(project.video.frames - 1, 1) });
  const [selection, setSelection] = useState<[number, number] | null>(null);

  const raw = smoothing?.curves.raw ?? null;
  const smoothed = smoothing?.curves.smoothed ?? null;
  const model = useMemo(() => (smoothing ? splitSpecs(smoothing.filters) : null), [smoothing]);
  const error = smoothingError;

  // The previous smoothed curves, kept so a change can be shown against what it
  // replaced ("before/after ghosting") rather than only against the raw solve.
  // This is a property of *this view's* history, not of the project, so it stays
  // local — an effect captures whatever the shared curves were a moment ago.
  const previous = useRef<SmoothedCurves | null>(null);
  const latest = useRef<SmoothedCurves | null>(null);
  useEffect(() => {
    if (smoothed !== latest.current) {
      previous.current = latest.current;
      latest.current = smoothed;
    }
  }, [smoothed]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const frames = raw ? raw.x.length : project.video.frames;

  useEffect(() => {
    setView((current) =>
      current.end > frames - 1 ? { start: 0, end: Math.max(frames - 1, 1) } : current,
    );
  }, [frames]);

  const lanes = useMemo(() => CHANNELS.filter((channel) => visible[channel]), [visible]);

  // -- drawing -----------------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !raw) return;
    const width = wrap.clientWidth || 640;
    const height = RULER_HEIGHT + lanes.length * (LANE_HEIGHT + LANE_GAP);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const plotWidth = Math.max(width - PAD_LEFT - 8, 10);
    const span = Math.max(view.end - view.start, 1e-6);
    const toX = (frame: number) => PAD_LEFT + ((frame - view.start) / span) * plotWidth;

    // Ruler with frame numbers at a round interval that fits the zoom level.
    ctx.fillStyle = "#1a1d21";
    ctx.fillRect(0, 0, width, RULER_HEIGHT);
    const targetTicks = Math.max(2, Math.floor(plotWidth / 70));
    const rawStep = span / targetTicks;
    const magnitude = Math.pow(10, Math.floor(Math.log10(Math.max(rawStep, 1))));
    const step = Math.max(1, Math.ceil(rawStep / magnitude) * magnitude);
    ctx.fillStyle = "#8a9099";
    ctx.font = "10px system-ui, sans-serif";
    for (let frame = Math.ceil(view.start / step) * step; frame <= view.end; frame += step) {
      const x = toX(frame);
      ctx.fillRect(x, RULER_HEIGHT - 4, 1, 4);
      ctx.fillText(String(frame), x + 2, 10);
    }

    lanes.forEach((channel, index) => {
      const top = RULER_HEIGHT + index * (LANE_HEIGHT + LANE_GAP);
      const bottom = top + LANE_HEIGHT;
      const config = model?.[channel];

      ctx.fillStyle = "#15181b";
      ctx.fillRect(PAD_LEFT, top, plotWidth, LANE_HEIGHT);

      // Fit this lane to what is actually on screen, over every series it draws.
      const from = Math.max(0, Math.floor(view.start));
      const to = Math.min(frames - 1, Math.ceil(view.end));
      const series: number[][] = [raw[channel].slice(from, to + 1)];
      if (smoothed) series.push(smoothed[channel].slice(from, to + 1));
      if (previous.current) series.push(previous.current[channel].slice(from, to + 1));
      let min = Infinity;
      let max = -Infinity;
      for (const values of series) {
        for (const value of values) {
          if (!Number.isFinite(value)) continue;
          if (value < min) min = value;
          if (value > max) max = value;
        }
      }
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        min = 0;
        max = 1;
      }
      const pad = (max - min) * 0.12 || 0.5;
      min -= pad;
      max += pad;
      const toY = (value: number) => bottom - ((value - min) / (max - min)) * LANE_HEIGHT;

      // The sub-range this channel's smoothing is restricted to.
      if (config?.range) {
        ctx.fillStyle = "rgba(91, 155, 213, 0.12)";
        const x0 = toX(config.range[0]);
        const x1 = toX(config.range[1]);
        ctx.fillRect(x0, top, Math.max(x1 - x0, 1), LANE_HEIGHT);
      }
      if (selection) {
        ctx.fillStyle = "rgba(255, 209, 102, 0.16)";
        const x0 = toX(Math.min(selection[0], selection[1]));
        const x1 = toX(Math.max(selection[0], selection[1]));
        ctx.fillRect(x0, top, Math.max(x1 - x0, 1), LANE_HEIGHT);
      }

      const stroke = (values: number[], style: string, lineWidth: number, dash: number[]) => {
        ctx.save();
        ctx.strokeStyle = style;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dash);
        ctx.beginPath();
        let started = false;
        for (let frame = from; frame <= to; frame++) {
          const value = values[frame];
          if (!Number.isFinite(value)) {
            started = false;
            continue;
          }
          const x = toX(frame);
          const y = toY(value);
          if (started) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          started = true;
        }
        ctx.stroke();
        ctx.restore();
      };

      // Ghosting: the raw solve (what the camera did) dashed, the smoothing this
      // panel replaced faint, and the current smoothing solid on top.
      stroke(raw[channel], "#6d747c", 1, [3, 3]);
      if (previous.current) stroke(previous.current[channel], "rgba(255,255,255,0.22)", 1.5, []);
      if (smoothed) stroke(smoothed[channel], CHANNEL_COLOR[channel], 1.8, []);

      // Flagged frames — the solve's "go fix this" strip, along the lane's floor.
      ctx.fillStyle = "rgba(214, 69, 69, 0.75)";
      for (let frame = from; frame <= to; frame++) {
        if (raw.flagged?.[frame]) ctx.fillRect(toX(frame), bottom - 3, 1.5, 3);
      }

      // Pins: the frames the smoothed curve is forced through.
      for (const frame of config?.pins ?? []) {
        if (frame < from || frame > to) continue;
        const x = toX(frame);
        const y = toY(raw[channel][frame]);
        ctx.beginPath();
        ctx.moveTo(x, y - 5);
        ctx.lineTo(x + 5, y);
        ctx.lineTo(x, y + 5);
        ctx.lineTo(x - 5, y);
        ctx.closePath();
        ctx.fillStyle = "#ffd166";
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      ctx.fillStyle = CHANNEL_COLOR[channel];
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(CHANNEL_LABEL[channel], 6, top + 12);
      ctx.fillStyle = "#8a9099";
      ctx.font = "9px system-ui, sans-serif";
      const value = smoothed ? smoothed[channel][currentFrame] : raw[channel][currentFrame];
      if (Number.isFinite(value)) ctx.fillText(value.toFixed(3), 6, top + 26);
    });

    if (currentFrame >= view.start && currentFrame <= view.end) {
      const x = toX(currentFrame);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }, [raw, smoothed, model, lanes, view, selection, currentFrame, frames]);

  // -- interaction --------------------------------------------------------------

  const frameAt = useCallback(
    (clientX: number): number => {
      const wrap = wrapRef.current;
      if (!wrap) return 0;
      const rect = wrap.getBoundingClientRect();
      const plotWidth = Math.max(rect.width - PAD_LEFT - 8, 10);
      const fraction = (clientX - rect.left - PAD_LEFT) / plotWidth;
      const frame = view.start + fraction * (view.end - view.start);
      return Math.max(0, Math.min(frames - 1, Math.round(frame)));
    },
    [view, frames],
  );

  const zoomAt = useCallback(
    (clientX: number, deltaY: number) => {
      const anchor = frameAt(clientX);
      const factor = deltaY > 0 ? 1.25 : 0.8;
      setView((current) => {
        const span = (current.end - current.start) * factor;
        const clamped = Math.max(4, Math.min(frames - 1, span));
        const ratio = (anchor - current.start) / Math.max(current.end - current.start, 1e-6);
        let start = anchor - ratio * clamped;
        let end = start + clamped;
        if (start < 0) {
          end -= start;
          start = 0;
        }
        if (end > frames - 1) {
          start -= end - (frames - 1);
          end = frames - 1;
        }
        return { start: Math.max(0, start), end };
      });
    },
    [frameAt, frames],
  );

  // React attaches wheel handlers passively, so `preventDefault` inside an
  // `onWheel` prop is ignored (and warns). Zooming has to stop the page
  // scrolling, so the listener is registered explicitly as non-passive.
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomAt(event.clientX, event.deltaY);
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  function handleMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    const startFrame = frameAt(event.clientX);
    if (event.shiftKey) {
      // Shift-drag marks the frame range a sub-range smoothing will apply to.
      setSelection([startFrame, startFrame]);
      const move = (ev: MouseEvent) => setSelection([startFrame, frameAt(ev.clientX)]);
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      return;
    }
    if (event.altKey || event.button === 1) {
      // Alt-drag pans the time axis.
      const origin = { x: event.clientX, view };
      const move = (ev: MouseEvent) => {
        const wrap = wrapRef.current;
        if (!wrap) return;
        const plotWidth = Math.max(wrap.clientWidth - PAD_LEFT - 8, 10);
        const perPixel = (origin.view.end - origin.view.start) / plotWidth;
        let start = origin.view.start - (ev.clientX - origin.x) * perPixel;
        const span = origin.view.end - origin.view.start;
        start = Math.max(0, Math.min(frames - 1 - span, start));
        setView({ start, end: start + span });
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      return;
    }
    // A plain click scrubs, and — matching the main timeline (todo 21) — held
    // and dragged it keeps scrubbing in real time rather than only seeking once.
    setCurrentFrame(startFrame);
    const move = (ev: MouseEvent) => setCurrentFrame(frameAt(ev.clientX));
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  // -- actions -------------------------------------------------------------------

  const activeChannels = lanes;

  function updateChannels(mutate: (config: ChannelConfig, channel: Channel) => void) {
    if (!model) return;
    const next: SmoothingModel = { ...model };
    for (const channel of activeChannels) {
      next[channel] = { ...next[channel], filters: [...next[channel].filters] };
      mutate(next[channel], channel);
    }
    void commitSmoothing(joinSpecs(next));
  }

  function smoothSelection() {
    if (!selection) return;
    const [a, b] = selection;
    updateChannels((config, channel) => {
      config.range = [Math.min(a, b), Math.max(a, b)];
      if (config.filters.length === 0) {
        // Restricting a channel that has no filter yet would do nothing, so the
        // range comes with a sane default the user can then tune.
        config.filters = [
          { channel, filter: "gaussian", params: { sigma: 8, cutoff_hz: 0 }, enabled: true },
        ];
      }
    });
  }

  function clearRange() {
    updateChannels((config) => {
      config.range = null;
    });
  }

  function togglePin() {
    updateChannels((config) => {
      config.pins = config.pins.includes(currentFrame)
        ? config.pins.filter((frame) => frame !== currentFrame)
        : [...config.pins, currentFrame].sort((p, q) => p - q);
    });
  }

  function clearPins() {
    updateChannels((config) => {
      config.pins = [];
    });
  }

  const pinnedHere =
    model !== null && activeChannels.every((channel) => model[channel].pins.includes(currentFrame));

  return (
    <div className="curve-editor">
      <div className="curve-editor__toolbar">
        <span className="curve-editor__title">Curves</span>
        {CHANNELS.map((channel) => (
          <label key={channel} className="channel-check">
            <input
              type="checkbox"
              checked={visible[channel]}
              onChange={(e) => setVisible({ ...visible, [channel]: e.target.checked })}
            />
            <span style={{ color: CHANNEL_COLOR[channel] }}>{CHANNEL_LABEL[channel]}</span>
          </label>
        ))}
        <span className="curve-editor__spacer" />
        <button onClick={() => setView({ start: 0, end: Math.max(frames - 1, 1) })}>Fit</button>
        <button onClick={smoothSelection} disabled={!selection || !model}>
          {selection
            ? `Smooth ${Math.min(selection[0], selection[1])}–${Math.max(selection[0], selection[1])}`
            : "Smooth range"}
        </button>
        <button onClick={clearRange} disabled={!model}>
          Whole clip
        </button>
        <button onClick={togglePin} disabled={!model}>
          {pinnedHere ? `Unpin ${currentFrame}` : `Pin ${currentFrame}`}
        </button>
        <button onClick={clearPins} disabled={!model}>
          Clear pins
        </button>
      </div>

      {error && <div className="error-text">{error}</div>}
      {!raw && <p className="panel__empty">Run solve to see the motion curves.</p>}

      <div ref={wrapRef} className="curve-editor__plot" onMouseDown={handleMouseDown}>
        <canvas ref={canvasRef} />
      </div>

      <p className="curve-editor__legend">
        <span className="curve-editor__key curve-editor__key--raw" /> raw solve
        <span className="curve-editor__key curve-editor__key--ghost" /> previous smoothing
        <span className="curve-editor__key curve-editor__key--now" /> current smoothing
        {" · "}wheel zooms, alt-drag pans, shift-drag selects a range, click/drag scrubs
      </p>
    </div>
  );
}
