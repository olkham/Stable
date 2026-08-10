import { useEffect, useMemo, useRef } from "react";
import { confidenceColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";

/** Colour of an anchor no tracking run could actually reach (todos 13, 15). */
const FLAGGED_ANCHOR_COLOR = "#ff3b30";

/**
 * Transport controls + scrub bar + per-point anchor marks + confidence
 * heat-strip for the selected point — the timeline's "go fix this" signal:
 * a red dip in the strip is where the merge disagreed or confidence dropped.
 *
 * Corrections on the selected point are marked along the top edge, so an edit
 * made on the frame is visible on the timeline without opening a panel. Anchors
 * are drawn in the point's colour when a run actually passed through them and
 * in red when one reached them and disagreed — "this anchor was never hit, go
 * look at it".
 */
/** Playback speeds offered, as multiples of the clip's own frame rate. */
const PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4];

/**
 * Why the range and the solve can disagree.
 *
 * The solve is measured *from* its reference frame, so a reference frame
 * outside the In/Out range has no tracked points on it and the solve refuses
 * (with a message saying so). Surfacing it here is cheaper than discovering it
 * from a failed job.
 */
const RANGE_HELP =
  "Tracking and export are limited to these frames. Keep the solve's reference frame inside the range, or the solve will have no tracked points to measure from.";

export function Timeline() {
  const {
    project,
    currentFrame,
    setCurrentFrame,
    isPlaying,
    setIsPlaying,
    playbackRate,
    setPlaybackRate,
    playRangeOnly,
    setPlayRangeOnly,
    clipRange,
    setRangeMarker,
    clearRange,
    goToMarker,
    selectedPointId,
    tracks,
    liveTracking,
    colorFor,
  } = useProjectContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const frames = Math.max(project.video.frames, 1);
  const selectedTrack = selectedPointId ? tracks[selectedPointId] : undefined;
  // The range when one is actually in force, else null — "the markers happen to
  // sit at the ends of the clip" and "there are no markers" are different
  // states, and only the first has anything to jump to.
  const ranged = clipRange && !clipRange.whole_clip ? clipRange : null;

  // Anchors per point that no run reached, and the full anchor list (the
  // point's own plus the ones manual corrections were promoted to, so a drag
  // shows up on the timeline as the anchor it has become).
  const flagged = useMemo(() => {
    const out: Record<string, Set<number>> = {};
    for (const [pointId, data] of Object.entries(tracks)) {
      out[pointId] = new Set(data.flagged_anchors ?? []);
    }
    return out;
  }, [tracks]);

  const anchorFrames = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const point of project.points) {
      const anchors = tracks[point.id]?.anchors;
      out[point.id] = (anchors ?? point.queries).map((q) => q.frame);
    }
    return out;
  }, [project.points, tracks]);

  // Playback loop (todo 3). Two changes from the old `setInterval` at the
  // clip's own fps: the rate is the operator's to choose, and the pacing is
  // *time-based*. An interval that cannot keep up silently falls behind — at
  // 2x on a big clip it would claim 48 fps and deliver 20 — so the elapsed time
  // decides how many frames to advance, dropping frames rather than drifting.
  // The current frame is read from a ref so the loop is not rebuilt on every
  // advance.
  //
  // What it plays is the In/Out range when the operator asks for that (todo 1,
  // sixth round) and the whole clip otherwise. `out` is exclusive everywhere in
  // this app — the strip dims from it, tracking and the export stop before it —
  // so the last frame played is `out - 1`, and reaching it stops and returns to
  // `in`, which is exactly what reaching the end of the clip already does.
  const frameRef = useRef(currentFrame);
  frameRef.current = currentFrame;
  const playFrom = playRangeOnly && ranged ? ranged.in : 0;
  const playTo = playRangeOnly && ranged ? ranged.out : project.video.frames;
  useEffect(() => {
    if (!isPlaying) return;
    const fps = Math.max(1, (project.video.fps || 24) * playbackRate);
    // Starting outside the range is "play it from the top", not "play nothing".
    if (frameRef.current < playFrom || frameRef.current >= playTo) {
      frameRef.current = playFrom;
      setCurrentFrame(playFrom);
    }
    let raf = 0;
    let last = performance.now();
    let carry = 0;
    const step = (now: number) => {
      const advance = Math.floor((carry + ((now - last) * fps) / 1000) % 1e6);
      carry += ((now - last) * fps) / 1000 - advance;
      last = now;
      if (advance > 0) {
        const next = frameRef.current + advance;
        if (next >= playTo) {
          setIsPlaying(false);
          setCurrentFrame(playFrom);
          return;
        }
        setCurrentFrame(next);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, project.video.fps, playbackRate, playFrom, playTo]);

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
    // a range edit, a tick for a single-frame one.
    for (const correction of selectedTrack?.corrections ?? []) {
      if (!correction.enabled) continue;
      const from = correction.start ?? correction.frame ?? 0;
      const to = correction.end ?? correction.frame ?? 0;
      const x0 = (from / frames) * width;
      const x1 = ((to + 1) / frames) * width;
      ctx.fillStyle = correction.kind === "delete_range" ? "#d64545" : "#ffd166";
      ctx.fillRect(x0, 0, Math.max(x1 - x0, 2), 3);
    }

    // Anchors for every point (the selected point's are brighter). One a run
    // reached and disagreed with goes red and gets a full-height tick, so "this
    // anchor still needs a human" is visible without selecting the point.
    for (const point of project.points) {
      const color = colorFor(point.id);
      const missed = flagged[point.id];
      for (const frame of anchorFrames[point.id] ?? []) {
        const x = (frame / frames) * width;
        const unreached = missed?.has(frame) ?? false;
        ctx.fillStyle = unreached ? FLAGGED_ANCHOR_COLOR : color;
        ctx.globalAlpha = point.id === selectedPointId ? 1 : 0.55;
        ctx.beginPath();
        ctx.moveTo(x, height);
        ctx.lineTo(x - 4, height - 9);
        ctx.lineTo(x + 4, height - 9);
        ctx.closePath();
        ctx.fill();
        if (unreached) {
          ctx.fillRect(x - 1, 0, 2, height);
        }
        ctx.globalAlpha = 1;
      }
    }

    // A tracking run in flight (todos 4, 16): shade the span its decoder has
    // reached and mark the frame it is on, so "how far has it got, and is it
    // worth scrubbing back to look" is answerable without opening a panel.
    for (const live of Object.values(liveTracking)) {
      const color = colorFor(live.pointId);
      if (live.span) {
        const x0 = (live.span.from / frames) * width;
        const x1 = ((live.span.to + 1) / frames) * width;
        ctx.save();
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = color;
        ctx.fillRect(x0, height - 5, Math.max(x1 - x0, 1), 5);
        ctx.restore();
      }
      if (live.frame != null) {
        const x = (live.frame / frames) * width;
        ctx.fillStyle = color;
        ctx.fillRect(x - 1, 0, 2, height);
      }
    }

    // Everything outside the In/Out range is dimmed: it is not going to be
    // tracked and will not be exported, and that should be visible at a glance
    // rather than only in a panel.
    if (clipRange && !clipRange.whole_clip) {
      ctx.save();
      ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
      const inX = (clipRange.in / frames) * width;
      const outX = (clipRange.out / frames) * width;
      ctx.fillRect(0, 0, inX, height);
      ctx.fillRect(outX, 0, width - outX, height);
      ctx.strokeStyle = "#4da3ff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(inX, 0);
      ctx.lineTo(inX, height);
      ctx.moveTo(outX, 0);
      ctx.lineTo(outX, height);
      ctx.stroke();
      ctx.restore();
    }

    // Playhead.
    const px = (currentFrame / frames) * width;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }, [
    project.points,
    selectedTrack,
    selectedPointId,
    currentFrame,
    frames,
    flagged,
    anchorFrames,
    liveTracking,
    clipRange,
    colorFor,
  ]);

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
        <select
          className="timeline__rate"
          value={playbackRate}
          onChange={(e) => setPlaybackRate(Number(e.target.value))}
          title="Playback speed. The clip's own frame rate is 1x."
        >
          {PLAYBACK_RATES.map((rate) => (
            <option key={rate} value={rate}>
              {rate}x
            </option>
          ))}
        </select>
        <span className="timeline__fps">
          {(project.video.fps * playbackRate).toFixed(1)} fps
          {playbackRate !== 1 && ` (clip is ${project.video.fps.toFixed(2)})`}
        </span>
        {/* In/Out markers (todo 1): what tracking and export are limited to.
            Setting them was never the hard part — finding them again on a long
            clip was, so each one also has a "go there" button beside it. */}
        <span className="timeline__range">
          <button onClick={() => void setRangeMarker("in", currentFrame)} title="Set the in point to this frame (I)">
            [ In
          </button>
          <button
            className="timeline__goto"
            onClick={() => goToMarker("in")}
            disabled={!ranged}
            title="Go to the in point (⇧I)"
          >
            ⇤
          </button>
          <button
            className="timeline__goto"
            onClick={() => goToMarker("out")}
            disabled={!ranged}
            title="Go to the last frame inside the range (⇧O)"
          >
            ⇥
          </button>
          <button onClick={() => void setRangeMarker("out", currentFrame + 1)} title="Set the out point to this frame (O)">
            Out ]
          </button>
          <span className="timeline__range-label" title={RANGE_HELP}>
            {ranged ? `${ranged.in}–${ranged.out - 1} (${ranged.frames}f)` : "whole clip"}
          </span>
          {ranged && (
            <button onClick={() => void clearRange()} title="Track and export the whole clip again">
              reset
            </button>
          )}
          {/* Play the range rather than the clip (todo 1, sixth round). Offered
              whatever the range is, and disabled when there is none, so it is
              discoverable before the markers are set rather than appearing out
              of nowhere once they are. */}
          <label
            className="channel-check"
            title={
              ranged
                ? "Play only the frames between the In and Out markers. Play stops at the out point and returns to the in point."
                : "Set In and Out markers first — with no range there is nothing to limit playback to."
            }
          >
            <input
              type="checkbox"
              checked={playRangeOnly}
              disabled={!ranged}
              onChange={(e) => setPlayRangeOnly(e.target.checked)}
            />
            play range
          </label>
        </span>
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
