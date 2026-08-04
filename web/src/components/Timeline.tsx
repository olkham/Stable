import { useEffect, useRef } from "react";
import { confidenceColor, pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";

/**
 * Transport controls + scrub bar + per-point anchor pins + confidence
 * heat-strip for the selected point — the timeline's "go fix this" signal:
 * a red dip in the strip is where the merge disagreed or confidence dropped.
 *
 * Corrections on the selected point are marked along the top edge, so an edit
 * made on the frame is visible on the timeline without opening a panel.
 */
export function Timeline() {
  const { project, currentFrame, setCurrentFrame, isPlaying, setIsPlaying, selectedPointId, tracks } =
    useProjectContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const frames = Math.max(project.video.frames, 1);
  const selectedTrack = selectedPointId ? tracks[selectedPointId] : undefined;

  // Playback loop, paced by the source video's fps. Reads the current frame
  // from a ref (not the `currentFrame` closure) so the interval does not need
  // to be torn down and rebuilt on every single frame advance.
  const frameRef = useRef(currentFrame);
  frameRef.current = currentFrame;
  useEffect(() => {
    if (!isPlaying) return;
    const fps = project.video.fps || 24;
    const id = setInterval(() => {
      const next = frameRef.current + 1;
      if (next >= project.video.frames) {
        setIsPlaying(false);
        setCurrentFrame(0);
      } else {
        setCurrentFrame(next);
      }
    }, 1000 / fps);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, project.video.fps, project.video.frames]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = canvas.clientWidth || 800;
    const height = 28;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    // Heat-strip: one column per frame, coloured by confidence (grey if untracked).
    for (let f = 0; f < frames; f++) {
      const x0 = Math.floor((f / frames) * width);
      const x1 = Math.max(x0 + 1, Math.floor(((f + 1) / frames) * width));
      let color = "#333";
      if (selectedTrack) {
        const valid = selectedTrack.x[f] != null;
        if (valid) color = confidenceColor(selectedTrack.confidence[f]);
      }
      ctx.fillStyle = color;
      ctx.fillRect(x0, 0, x1 - x0, height);
    }

    // Correction markers for the selected point, along the top edge: a bar for
    // a range edit, a tick for a keyframe one.
    for (const correction of selectedTrack?.corrections ?? []) {
      if (!correction.enabled) continue;
      const from = correction.start ?? correction.frame ?? 0;
      const to = correction.end ?? correction.frame ?? 0;
      const x0 = (from / frames) * width;
      const x1 = ((to + 1) / frames) * width;
      ctx.fillStyle = correction.kind === "delete_range" ? "#d64545" : "#ffd166";
      ctx.fillRect(x0, 0, Math.max(x1 - x0, 2), 3);
    }

    // Anchor pins for every point (selected point's pins are brighter).
    for (const point of project.points) {
      const color = pointColor(point.id);
      for (const q of point.queries) {
        const x = (q.frame / frames) * width;
        ctx.fillStyle = color;
        ctx.globalAlpha = point.id === selectedPointId ? 1 : 0.55;
        ctx.beginPath();
        ctx.moveTo(x, height);
        ctx.lineTo(x - 4, height - 9);
        ctx.lineTo(x + 4, height - 9);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }

    // Playhead.
    const px = (currentFrame / frames) * width;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }, [project.points, selectedTrack, selectedPointId, currentFrame, frames]);

  function seekFromEvent(evt: React.MouseEvent<HTMLDivElement>) {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (evt.clientX - rect.left) / rect.width));
    setCurrentFrame(Math.min(frames - 1, Math.round(fraction * frames)));
  }

  return (
    <div className="timeline">
      <div className="timeline__transport">
        <button onClick={() => setIsPlaying(!isPlaying)}>{isPlaying ? "Pause" : "Play"}</button>
        <input
          type="number"
          className="timeline__frame-input"
          min={0}
          max={frames - 1}
          value={currentFrame}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (!Number.isNaN(v)) setCurrentFrame(Math.max(0, Math.min(frames - 1, v)));
          }}
        />
        <span className="timeline__fps">{project.video.fps.toFixed(2)} fps</span>
      </div>
      <div
        ref={wrapRef}
        className="timeline__strip"
        onMouseDown={(e) => {
          seekFromEvent(e);
          const move = (ev: MouseEvent) =>
            seekFromEvent(ev as unknown as React.MouseEvent<HTMLDivElement>);
          const up = () => {
            window.removeEventListener("mousemove", move);
            window.removeEventListener("mouseup", up);
          };
          window.addEventListener("mousemove", move);
          window.addEventListener("mouseup", up);
        }}
      >
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
