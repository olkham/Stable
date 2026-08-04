import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { confidenceColor, pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";

/** How close (in canvas CSS pixels) a mousedown has to be to grab a point. */
const GRAB_RADIUS = 12;

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

interface Pan {
  x: number;
  y: number;
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
 * point's track path (confidence-coloured), its anchor pins, and handles
 * click-to-place for new points / new anchors — and, in edit mode, moving a
 * tracked point on the frame to correct it, either by dragging it or by
 * selecting it and clicking its new location.
 *
 * The canvas always fills the available space (letterboxed to the video's own
 * aspect ratio, never the browser's) and supports zoom (wheel, centred on the
 * cursor) and pan (middle-mouse drag) so a point can be placed precisely even
 * on a small or distant subject. A move never rewrites the track. It appends
 * one correction: an offset blended over the falloff window, or a hard
 * replace at this frame. The raw path stays underneath and is ghosted, so the
 * user sees exactly what their edit changed.
 */
export function Viewer() {
  const {
    project,
    currentFrame,
    setCurrentFrame,
    selectedPointId,
    setSelectedPointId,
    placingMode,
    setPlacingMode,
    stabilizedPreview,
    tracks,
    loadTrack,
    refresh,
    editing,
    dragMode,
    falloff,
    applyCorrection,
  } = useProjectContext();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const imageCache = useRef(new Map<string, HTMLImageElement>());
  // Bumped once a frame image finishes loading, to make the draw effect below
  // re-run and pick it up from imageCache -- it does nothing on its own, since
  // it is not otherwise a drawing input, but must be a real dependency of that
  // effect (not just referenced from inside it) or the newly-loaded image
  // never gets painted until some *other* prop happens to change the canvas.
  const [redrawTick, forceRedraw] = useState(0);
  // Bumped whenever the display box resizes (window resize, sidebar/curves
  // panel toggling), so the canvas is re-sized and re-fit to the new space
  // rather than staying at whatever size it happened to mount at (todo 2).
  const [sizeTick, setSizeTick] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
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

  // -- overlay visibility (todos 7, 10) -----------------------------------------
  const [tracksVisible, setTracksVisible] = useState(true);
  const [showPaths, setShowPaths] = useState(true);
  const [showAnchors, setShowAnchors] = useState(true);
  const [showGhost, setShowGhost] = useState(true);

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

  const frameSrc = useMemo(() => {
    if (!proxyReady) return null;
    return stabilizedPreview
      ? api.stabilizedFrameUrl(project.id, currentFrame)
      : api.frameUrl(project.id, currentFrame);
  }, [proxyReady, stabilizedPreview, project.id, currentFrame]);

  useEffect(() => {
    if (!frameSrc) return;
    setError(null);
    const cached = imageCache.current.get(frameSrc);
    if (cached && cached.complete) {
      forceRedraw((n) => n + 1);
      return;
    }
    const img = new Image();
    img.onload = () => {
      imageCache.current.set(frameSrc, img);
      // Cap the cache so a long scrub session does not grow unbounded.
      if (imageCache.current.size > 240) {
        const oldest = imageCache.current.keys().next().value;
        if (oldest) imageCache.current.delete(oldest);
      }
      forceRedraw((n) => n + 1);
    };
    img.onerror = () =>
      setError(stabilizedPreview ? "stabilized preview not ready yet" : "frame not available");
    img.src = frameSrc;
  }, [frameSrc, stabilizedPreview]);

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

    const img = frameSrc ? imageCache.current.get(frameSrc) : undefined;
    if (img) {
      ctx.drawImage(img, frameLeft, frameTop, frameW, frameH);
    } else {
      ctx.fillStyle = "#111";
      ctx.fillRect(frameLeft, frameTop, frameW, frameH);
    }
    if (!overlayValid || !tracksVisible) return;

    for (const point of project.points) {
      const isSelected = point.id === selectedPointId;
      const track = tracks[point.id];
      const color = pointColor(point.id);

      if (track) {
        // Before/after ghosting: where corrections exist the server also sends
        // the untouched raw path, drawn faintly behind the corrected one so the
        // edit is visible rather than merely believed.
        if (showGhost && track.raw_x && track.raw_y) {
          ctx.save();
          ctx.globalAlpha = 0.4;
          ctx.setLineDash([4, 3]);
          ctx.lineWidth = 1;
          ctx.strokeStyle = "#ffffff";
          ctx.beginPath();
          let started = false;
          for (let f = 0; f < track.n_frames; f++) {
            const rx = track.raw_x[f];
            const ry = track.raw_y[f];
            if (rx == null || ry == null) {
              started = false;
              continue;
            }
            const px = toCanvasX(rx);
            const py = toCanvasY(ry);
            if (started) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            started = true;
          }
          ctx.stroke();
          ctx.restore();
        }

        if (showPaths) {
          // Path: connect consecutive valid frames, colouring each short
          // segment by that frame's confidence — this is the "confidence
          // colouring" the track path needs (green = trust it, red = check it).
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          for (let f = 1; f < track.n_frames; f++) {
            const x0 = track.x[f - 1];
            const y0 = track.y[f - 1];
            const x1 = track.x[f];
            const y1 = track.y[f];
            if (x0 == null || y0 == null || x1 == null || y1 == null) continue;
            ctx.strokeStyle = confidenceColor((track.confidence[f - 1] + track.confidence[f]) / 2);
            ctx.beginPath();
            ctx.moveTo(toCanvasX(x0), toCanvasY(y0));
            ctx.lineTo(toCanvasX(x1), toCanvasY(y1));
            ctx.stroke();
          }

          // Current-frame marker, following the pointer while it is being dragged.
          const cx = track.x[currentFrame];
          const cy = track.y[currentFrame];
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
            ctx.fillStyle = confidenceColor(track.confidence[currentFrame]);
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = dragging ? "#ffd166" : "#fff";
            ctx.stroke();
          }
        }
      }

      // Anchor pins (diamonds), always drawn from the point spec itself so
      // they show up even before the point has been tracked.
      if (showAnchors) {
        for (const q of point.queries) {
          const px = toCanvasX(q.x);
          const py = toCanvasY(q.y);
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          const size = q.frame === currentFrame ? 7 : 5;
          ctx.fillStyle = color;
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1;
          ctx.fillRect(-size / 2, -size / 2, size, size);
          ctx.strokeRect(-size / 2, -size / 2, size, size);
          ctx.restore();
        }
      }
    }
  }, [
    project.points,
    videoW,
    videoH,
    tracks,
    currentFrame,
    selectedPointId,
    proxyReady,
    frameSrc,
    drag,
    overlayValid,
    redrawTick,
    sizeTick,
    zoom,
    pan,
    tracksVisible,
    showPaths,
    showAnchors,
    showGhost,
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
        const track = tracks[point.id];
        const x = track?.x[currentFrame];
        const y = track?.y[currentFrame];
        if (x == null || y == null) continue;
        const p = sourceToCanvas(x, y, t, videoW, videoH);
        const distance = Math.hypot(p.x - canvasX, p.y - canvasY);
        if (distance <= GRAB_RADIUS && (!best || distance < best.distance)) {
          best = { id: point.id, distance };
        }
      }
      return best?.id ?? null;
    },
    [project.points, tracks, currentFrame, videoW, videoH, zoom, pan],
  );

  /**
   * Apply the correction a move to (toX, toY) implies for `pointId`, shared by
   * both drag-release and click-to-move (todo 8) so the two input paths can
   * never disagree about what a "move" means.
   */
  const commitMove = useCallback(
    (pointId: string, toX: number, toY: number, shiftKey: boolean) => {
      const track = tracks[pointId];
      const fromX = track?.x[currentFrame];
      const fromY = track?.y[currentFrame];
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
          ? { kind: "replace", frame: currentFrame, x: toX, y: toY }
          : {
              kind: "offset",
              frame: currentFrame,
              dx,
              dy,
              falloff_before: falloff,
              falloff_after: falloff,
              shape: "smooth",
            },
      );
    },
    [tracks, currentFrame, dragMode, falloff, applyCorrection],
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
    const { x, y } = toSource(evt);

    if (placingMode === "new-point") {
      const point = await api.createPoint(project.id, currentFrame, x, y);
      setPlacingMode("none");
      setSelectedPointId(point.id);
      await refresh();
    } else if (placingMode === "add-anchor" && selectedPointId) {
      await api.addAnchor(project.id, selectedPointId, currentFrame, x, y);
      setPlacingMode("none");
      await refresh();
    } else if (placingMode === "none" && editing && overlayValid && selectedPointId) {
      // Click-to-move (todo 8): the selected point jumps to wherever was
      // clicked, exactly as if it had been dragged there.
      commitMove(selectedPointId, x, y, evt.shiftKey);
    }
  }

  function handleMouseDown(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (!proxyReady) return;

    if (evt.button === 1) {
      // Middle-mouse drag pans, regardless of mode — it never collides with
      // placing, dragging or click-to-move, all of which are left-button.
      evt.preventDefault();
      const wrap = wrapRef.current;
      if (!wrap) return;
      const dispW = wrap.clientWidth || 1;
      const dispH = wrap.clientHeight || 1;
      const startX = evt.clientX;
      const startY = evt.clientY;
      const startPan = pan;
      const startZoom = zoom;
      const move = (ev: MouseEvent) => {
        const next = { x: startPan.x + (ev.clientX - startX), y: startPan.y + (ev.clientY - startY) };
        setPan(clampPan(next, startZoom, dispW, dispH, videoW, videoH));
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
      return;
    }

    if (evt.button !== 0) return;
    if (!editing || placingMode !== "none" || !overlayValid) return;
    const { canvasX, canvasY } = toSource(evt);
    const pointId = hitTest(canvasX, canvasY);
    if (!pointId) return; // not on a marker -- the click handler may still move it
    const track = tracks[pointId];
    const fromX = track.x[currentFrame]!;
    const fromY = track.y[currentFrame]!;
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

  const cursorClass = drag
    ? "viewer__canvas-wrap--dragging"
    : placingMode !== "none"
      ? "viewer__canvas-wrap--placing"
      : editing && overlayValid
        ? "viewer__canvas-wrap--editing"
        : "";

  const zoomPct = Math.round(zoom * 100);
  const atFit = zoom === MIN_ZOOM && pan.x === 0 && pan.y === 0;

  return (
    <div className="viewer">
      <div className="viewer__toolbar" style={{ display: "flex", alignItems: "center", gap: 14 }}>
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
        <div className="channel-checks">
          <label className="channel-check" title="Hide every point overlay to see the frame underneath">
            <input
              type="checkbox"
              checked={tracksVisible}
              onChange={(e) => setTracksVisible(e.target.checked)}
            />
            Tracks
          </label>
          <label
            className="channel-check"
            style={{ opacity: tracksVisible ? 1 : 0.4 }}
            title="Track path lines and the current-frame marker"
          >
            <input
              type="checkbox"
              checked={showPaths}
              disabled={!tracksVisible}
              onChange={(e) => setShowPaths(e.target.checked)}
            />
            Paths
          </label>
          <label
            className="channel-check"
            style={{ opacity: tracksVisible ? 1 : 0.4 }}
            title="Anchor keyframe pins"
          >
            <input
              type="checkbox"
              checked={showAnchors}
              disabled={!tracksVisible}
              onChange={(e) => setShowAnchors(e.target.checked)}
            />
            Anchors
          </label>
          <label
            className="channel-check"
            style={{ opacity: tracksVisible ? 1 : 0.4 }}
            title="The dotted before/after ghost trail a correction leaves behind"
          >
            <input
              type="checkbox"
              checked={showGhost}
              disabled={!tracksVisible}
              onChange={(e) => setShowGhost(e.target.checked)}
            />
            Ghost trails
          </label>
        </div>
      </div>
      <div ref={wrapRef} className={`viewer__canvas-wrap ${cursorClass}`}>
        {proxyReady ? (
          <canvas ref={canvasRef} onClick={handleClick} onMouseDown={handleMouseDown} />
        ) : (
          <div className="viewer__placeholder">Generating proxy frames…</div>
        )}
        {error && <div className="viewer__error">{error}</div>}
        {placingMode !== "none" && (
          <div className="viewer__hint">
            {placingMode === "new-point"
              ? "Click on the video to place a new point"
              : "Click on the video to add an anchor here"}
          </div>
        )}
        {placingMode === "none" && editing && overlayValid && (
          <div className="viewer__hint">
            {selectedPointId
              ? `Click to move the selected point, or drag it directly — ${
                  dragMode === "replace" ? "hard replace at this frame" : `offset, ${falloff}f falloff`
                } (hold Shift to replace)`
              : "Select a point to move or drag it on the frame"}
          </div>
        )}
        {editing && !overlayValid && (
          <div className="viewer__hint">
            Turn the stabilized preview off to edit points on the frame
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
          ←/→ frame · space play/pause · ↑/↓ keyframe · wheel/+/- zoom · middle-drag pan
        </span>
      </div>
    </div>
  );
}
