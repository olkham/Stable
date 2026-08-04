import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { confidenceColor, pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";

/** How close (in canvas pixels) a mousedown has to be to grab a point. */
const GRAB_RADIUS = 12;

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
 * The canvas viewer: draws the current proxy frame plus an overlay of every
 * point's track path (confidence-coloured), its anchor pins, and handles
 * click-to-place for new points / new anchors — and, in edit mode, dragging a
 * tracked point on the frame to correct it.
 *
 * A drag never rewrites the track. It appends one correction: an offset blended
 * over the falloff window, or a hard replace at this frame. The raw path stays
 * underneath and is ghosted, so the user sees exactly what their edit changed.
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
  const imageCache = useRef(new Map<string, HTMLImageElement>());
  // Bumped once a frame image finishes loading, to make the draw effect below
  // re-run and pick it up from imageCache -- it does nothing on its own, since
  // it is not otherwise a drawing input, but must be a real dependency of that
  // effect (not just referenced from inside it) or the newly-loaded image
  // never gets painted until some *other* prop happens to change the canvas.
  const [redrawTick, forceRedraw] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);

  const proxy = project.proxy["half"];
  const proxyReady = proxy?.ready && proxy.width && proxy.height;
  const proxyScale = proxyReady ? proxy.width! / project.video.w : 1;

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !proxyReady) return;
    canvas.width = proxy.width!;
    canvas.height = proxy.height!;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const img = frameSrc ? imageCache.current.get(frameSrc) : undefined;
    if (img) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#111";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (!overlayValid) return;

    for (const point of project.points) {
      const isSelected = point.id === selectedPointId;
      const track = tracks[point.id];
      const color = pointColor(point.id);

      if (track) {
        // Before/after ghosting: where corrections exist the server also sends
        // the untouched raw path, drawn faintly behind the corrected one so the
        // edit is visible rather than merely believed.
        if (track.raw_x && track.raw_y) {
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
            const px = rx * proxyScale;
            const py = ry * proxyScale;
            if (started) ctx.lineTo(px, py);
            else ctx.moveTo(px, py);
            started = true;
          }
          ctx.stroke();
          ctx.restore();
        }

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
          ctx.moveTo(x0 * proxyScale, y0 * proxyScale);
          ctx.lineTo(x1 * proxyScale, y1 * proxyScale);
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
            ctx.moveTo(cx * proxyScale, cy * proxyScale);
            ctx.lineTo(mx * proxyScale, my * proxyScale);
            ctx.stroke();
            ctx.restore();
          }
          ctx.beginPath();
          ctx.arc(mx * proxyScale, my * proxyScale, isSelected ? 6 : 4, 0, Math.PI * 2);
          ctx.fillStyle = confidenceColor(track.confidence[currentFrame]);
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = dragging ? "#ffd166" : "#fff";
          ctx.stroke();
        }
      }

      // Anchor pins (diamonds), always drawn from the point spec itself so
      // they show up even before the point has been tracked.
      for (const q of point.queries) {
        const px = q.x * proxyScale;
        const py = q.y * proxyScale;
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
  }, [
    project.points,
    tracks,
    currentFrame,
    selectedPointId,
    proxyReady,
    proxy,
    proxyScale,
    frameSrc,
    drag,
    overlayValid,
    redrawTick,
  ]);

  /** Convert a mouse event into source-video pixel coordinates. */
  const toSource = useCallback(
    (evt: { clientX: number; clientY: number }) => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const canvasX = ((evt.clientX - rect.left) / rect.width) * canvas.width;
      const canvasY = ((evt.clientY - rect.top) / rect.height) * canvas.height;
      return { x: canvasX / proxyScale, y: canvasY / proxyScale, canvasX, canvasY };
    },
    [proxyScale],
  );

  /** Return the tracked point whose current-frame marker is under the pointer. */
  const hitTest = useCallback(
    (canvasX: number, canvasY: number): string | null => {
      let best: { id: string; distance: number } | null = null;
      for (const point of project.points) {
        const track = tracks[point.id];
        const x = track?.x[currentFrame];
        const y = track?.y[currentFrame];
        if (x == null || y == null) continue;
        const distance = Math.hypot(x * proxyScale - canvasX, y * proxyScale - canvasY);
        if (distance <= GRAB_RADIUS && (!best || distance < best.distance)) {
          best = { id: point.id, distance };
        }
      }
      return best?.id ?? null;
    },
    [project.points, tracks, currentFrame, proxyScale],
  );

  async function handleClick(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current || !proxyReady || placingMode === "none") return;
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
    }
  }

  function handleMouseDown(evt: React.MouseEvent<HTMLCanvasElement>) {
    if (!editing || !proxyReady || placingMode !== "none" || !overlayValid) return;
    const { canvasX, canvasY } = toSource(evt);
    const pointId = hitTest(canvasX, canvasY);
    if (!pointId) return;
    const track = tracks[pointId];
    const fromX = track.x[currentFrame]!;
    const fromY = track.y[currentFrame]!;
    evt.preventDefault();
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
      const dx = p.x - fromX;
      const dy = p.y - fromY;
      if (Math.hypot(dx, dy) < 0.5) return; // a click, not a drag
      // Shift forces a hard replace whatever the panel says: "I know exactly
      // where this is" should always be one key away.
      const hard = ev.shiftKey || dragMode === "replace";
      void applyCorrection(
        pointId,
        hard
          ? { kind: "replace", frame: currentFrame, x: p.x, y: p.y }
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

  return (
    <div className="viewer">
      <div className={`viewer__canvas-wrap ${cursorClass}`}>
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
            Drag a point to correct it —{" "}
            {dragMode === "replace"
              ? "hard replace at this frame"
              : `offset, ${falloff}f falloff (hold Shift to replace)`}
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
      </div>
    </div>
  );
}
