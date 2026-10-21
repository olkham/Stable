import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "../api/client";
import { confidenceColor } from "../colors";
import type { TracksPanel as PanelVariant } from "../flags";
import { useProjectContext } from "../state/ProjectContext";
import type { PointSummary, Query, TrackData, TrackerInfo } from "../types";
import { CompositeSources, DeriveTrackPanel } from "./DeriveTrack";
import { JobProgress } from "./JobProgress";
import { NewTrack } from "./NewTrack";
import { PanelSwitch } from "./PanelSwitch";
import { AllPatchesStrip, PatchPanel } from "./PatchPanel";
import { TrackEdits } from "./TrackEdits";
import { TrackEye, TrackSwatch } from "./TrackVisibility";
import { OVERLAY_LABELS } from "./Viewer";

/**
 * The Tracks panel, rebuilt around the anchor timeline (design B).
 *
 * The card this replaces was eight stacked blocks with three separate ways to
 * start a run — the ◀/▶ on an anchor chip, a "from the playhead" pair four
 * blocks lower, and Track — none of them near each other, all governed by a
 * "how far" selector kept somewhere else again. The chips also threw away the
 * one thing an anchor is *about*: an anchor is a position **in time**, and a
 * wrapping list of pills says nothing about where in the clip it sits.
 *
 * So a track is drawn as a strip of the clip. Coverage, confidence, the gaps,
 * the anchors, the one no run could reach, the playhead and any run in flight
 * are all read off it at a glance. Clicking picks a **subject** — an anchor, or
 * the playhead — and the single inspector row underneath acts on whatever that
 * is. One pair of direction buttons, in one place, that never moves; the thing
 * they apply to is what changes.
 *
 * Everything that is set once per track rather than used constantly — the
 * tracker, its window sizes, the overlay toggles, the correction list — is
 * behind the row's `⋯`, which is what gets the resting card down to two lines
 * and several tracks onto one screen.
 *
 * Behind a flag (`flags.ts`) alongside the original panel, so the two can be
 * judged against each other on real footage.
 */
export function TrackTimelinePanel({
  variant,
  onVariantChange,
}: {
  variant: PanelVariant;
  onVariantChange: (next: PanelVariant) => void;
}) {
  const { project, newTrack, startNewTrack, cancelNewTrack, selectedPointId, setSelectedPointId } =
    useProjectContext();

  // How far every run from this panel reaches. One setting, in the header,
  // because it governs every ◀/▶ below it — on the old card it lived inside one
  // track's anchor row while quietly governing another track's buttons too.
  const [runLimit, setRunLimit] = useState<RunLimit>("anchor");
  const [runFrames, setRunFrames] = useState(25);
  const [error, setError] = useState<string | null>(null);

  return (
    <div className="panel points-panel">
      <div className="panel__header">
        <h3>Tracks</h3>
        <button
          className={newTrack ? "button--active" : ""}
          onClick={() => (newTrack ? cancelNewTrack() : startNewTrack())}
          title="Choose a tracker and its settings, then mark what it should follow on the video (N for a point, B for a box)"
        >
          + New track
        </button>
      </div>

      <NewTrack />

      <div className="panel__subhead">
        <span className="lbl">Runs</span>
        <select
          value={runLimit}
          onChange={(e) => setRunLimit(e.target.value as RunLimit)}
          title="How far every ◀ / ▶ below reaches. Tracking from an anchor to the next one is the everyday case; the whole clip is rarely what you mean."
        >
          <option value="anchor">stop at the next anchor</option>
          <option value="frames">cover N frames</option>
          <option value="clip">go to the end</option>
        </select>
        {runLimit === "frames" && (
          <input
            type="number"
            min={1}
            max={Math.max(project.video.frames, 1)}
            value={runFrames}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (!Number.isNaN(v)) setRunFrames(Math.max(1, Math.round(v)));
            }}
          />
        )}
        <span style={{ flex: 1 }} />
        <PanelSwitch variant={variant} onChange={onVariantChange} />
      </div>

      {project.points.length === 0 && (
        <p className="panel__empty">
          No tracks yet. Click "New track", choose a tracker, then mark what it should follow on the
          video.
        </p>
      )}

      {/* Every point's patch on this frame — "which one has drifted", which the
          strips below cannot answer because they are about time, not pixels. */}
      <AllPatchesStrip />

      <ul className="track-list">
        {project.points.map((point) => (
          <TrackRow
            key={point.id}
            point={point}
            selected={point.id === selectedPointId}
            onSelect={() => setSelectedPointId(point.id)}
            bounds={boundsFor(runLimit, runFrames)}
            onError={setError}
          />
        ))}
      </ul>

      {/* A track made from the ones above (todo 2, tenth round) — the same
          component the classic panel uses, so both variants offer it. */}
      <DeriveTrackPanel />

      {error && (
        <p className="panel__empty error-text" onClick={() => setError(null)}>
          {error}
        </p>
      )}
    </div>
  );
}

/** How far a run started from this panel should reach (todos 12 and 20). */
type RunLimit = "anchor" | "clip" | "frames";

/**
 * A stable empty list, for the strip of a track with no anchors to draw.
 *
 * A fresh `[]` each render would change identity every time and redraw every
 * composite's strip on every render of anything — the same reason `anchors`
 * below is memoised.
 */
const EMPTY_ANCHORS: Query[] = [];

function boundsFor(limit: RunLimit, frames: number): api.RunBounds {
  if (limit === "anchor") return { stop_at_anchor: true };
  if (limit === "frames") return { max_frames: frames };
  return {};
}

/** How far a run will go, in words, for the buttons that start one. */
function reachLabel(bounds: api.RunBounds): string {
  if (bounds.stop_at_anchor) return "as far as the next anchor";
  if (bounds.max_frames) return `${bounds.max_frames} frames`;
  return "to the end of the clip";
}

/**
 * One track: what state it is in, where its anchors are, and what to do next.
 *
 * Two lines at rest. The strip is the whole status report — a track with full
 * green coverage and no red diamond needs no reading at all — and the inspector
 * only appears for the selected track, because run controls for a track you are
 * not looking at are just noise with buttons.
 */
function TrackRow({
  point,
  selected,
  onSelect,
  bounds,
  onError,
}: {
  point: PointSummary;
  selected: boolean;
  onSelect: () => void;
  bounds: api.RunBounds;
  onError: (message: string | null) => void;
}) {
  const {
    project,
    trackers,
    tracks,
    loadTrack,
    currentFrame,
    setCurrentFrame,
    liveTracking,
    latestJobFor,
    invalidateTrack,
    refresh,
    deleteAnchorAt,
    runFrom,
    addAnchorHere,
    setTool,
    overlays,
    overlaysFor,
    setTrackOverlay,
    trackOverrides,
    colorFor,
    isTrackHidden,
  } = useProjectContext();

  const track = tracks[point.id];
  // The anchor the inspector is acting on, or null for "the playhead". Kept per
  // row so switching between tracks does not lose where you were in each.
  const [pickedAnchor, setPickedAnchor] = useState<number | null>(null);
  const [setupOpen, setSetupOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (point.has_track && !(point.id in tracks)) void loadTrack(point.id);
  }, [point.has_track, point.id, tracks, loadTrack]);

  // The point's own anchors plus the ones its corrections were promoted to. Its
  // own come from the project, which refreshes the moment one is placed; the
  // promotions only exist on the track, which is cached separately.
  //
  // Memoised because the strip redraws whenever these change identity, and a
  // fresh array every render would mean a redraw of every track on every render
  // of anything.
  const anchors: Query[] = useMemo(() => {
    const promoted = (track?.anchors ?? []).filter(
      (a) => !point.queries.some((q) => q.frame === a.frame),
    );
    return [...point.queries, ...promoted].sort((a, b) => a.frame - b.frame);
  }, [point.queries, track?.anchors]);
  const flagged = useMemo(() => new Set(track?.flagged_anchors ?? []), [track?.flagged_anchors]);
  const gaps = useMemo(() => interiorGaps(track), [track]);
  const violations = track?.anchor_violations ?? [];
  const live = liveTracking[point.id];

  // What the inspector is acting on. An anchor picked on the strip, and it
  // stays picked while the playhead moves — the label says which frame, so
  // "track from the anchor I chose" survives a scrub. Standing *on* an anchor
  // makes that anchor the subject without a click, because on that frame there
  // is no meaningful difference between the two and offering to add a second
  // anchor where one already sits reads as a bug.
  //
  // A picked anchor that has since been deleted is no anchor at all; that falls
  // back to the playhead rather than acting on a frame that means nothing now.
  const anchorFrame =
    pickedAnchor !== null && anchors.some((a) => a.frame === pickedAnchor)
      ? pickedAnchor
      : (anchors.find((a) => a.frame === currentFrame)?.frame ?? null);
  const subjectFrame = anchorFrame ?? currentFrame;
  const status = statusOf(point, gaps.length, flagged.size);
  // A track averaged from other tracks. Not a tracker's output, so most of this
  // card is about something that did not happen to it.
  const composite = point.derived_from.length > 0;
  // Whether a run can start here at all: the tracker needs a position to track
  // from, which is an anchor on this frame or a tracked sample there.
  const runnable = anchorFrame !== null || (track?.x[currentFrame] ?? null) !== null;

  async function act(action: () => Promise<string | null>) {
    setBusy(true);
    onError(await action());
    setBusy(false);
  }

  const pick = useCallback(
    (frame: number, anchor: Query | null) => {
      onSelect();
      setPickedAnchor(anchor ? anchor.frame : null);
      setCurrentFrame(frame);
    },
    [onSelect, setCurrentFrame],
  );

  return (
    <li
      className={`track-row ${selected ? "track-row--selected" : ""}`}
      onClick={onSelect}
      style={{ borderLeftColor: selected ? colorFor(point.id) : "transparent" }}
    >
      <div className="track-row__head">
        {/* One click, on the resting row: hide everything this track draws, or
            give it a colour of its own (todos 6 and 7, ninth round). */}
        <TrackEye pointId={point.id} />
        <TrackSwatch pointId={point.id} className="track-row__swatch" />
        <span className="track-row__id">{point.id}</span>
        <span className={`track-pill track-pill--${status.tone}`} title={status.help}>
          {status.label}
        </span>
        <span style={{ flex: 1 }} />
        {point.has_track && (
          <span className="track-row__conf" title="Mean confidence over the frames this track covers">
            {(point.mean_confidence * 100).toFixed(0)}%
          </span>
        )}
        <button
          className={`track-row__more ${setupOpen ? "button--active" : ""}`}
          aria-expanded={setupOpen}
          title="Tracker, window sizes, overlays and edits — the things you set once"
          onClick={(e) => {
            e.stopPropagation();
            setSetupOpen(!setupOpen);
          }}
        >
          ⋯
        </button>
      </div>

      <TrackStrip
        point={point}
        color={colorFor(point.id)}
        track={track}
        // A composite carries one anchor because every point must, not because
        // anything was anchored — drawing a diamond for it would claim a
        // keyframe that nobody placed and nothing tracked from.
        anchors={composite ? EMPTY_ANCHORS : anchors}
        flagged={flagged}
        gaps={gaps}
        frames={Math.max(project.video.frames, 1)}
        currentFrame={currentFrame}
        pickedAnchor={composite ? null : anchorFrame}
        liveFrame={live?.frame ?? null}
        onPick={pick}
      />

      <div className="track-row__legend">
        <span>f0</span>
        <span>
          {point.has_track
            ? `${point.tracked_frames} of ${project.video.frames} f`
            : `${anchors.length} anchor${anchors.length === 1 ? "" : "s"}, not tracked`}
          {gaps.length > 0 && ` · ${gaps.length} gap${gaps.length === 1 ? "" : "s"}`}
        </span>
        <span>f{Math.max(project.video.frames - 1, 0)}</span>
      </div>

      {/* A composite has nothing to run and nothing to anchor — its path is the
          average of other tracks, not the output of a tracker. So the inspector
          is replaced by what it *is* made of (todo 2, tenth round, second
          pass), rather than offering controls that would each need a reason
          why they do nothing. Its edits stay available under ⋯: a composite
          path can still be nudged, filled or smoothed like any other. */}
      {selected && composite && (
        <div className="track-inspector" onClick={(e) => e.stopPropagation()}>
          <CompositeSources point={point} />
        </div>
      )}

      {selected && !composite && (
        <div className="track-inspector" onClick={(e) => e.stopPropagation()}>
          <div className="track-inspector__row">
            <span className="track-inspector__subject">
              {anchorFrame !== null ? (
                <>
                  <b style={{ color: flagged.has(anchorFrame) ? FLAGGED : colorFor(point.id) }}>
                    ◆
                  </b>{" "}
                  anchor f{anchorFrame}
                </>
              ) : (
                <>▏playhead f{currentFrame}</>
              )}
            </span>
            <button
              disabled={busy || !runnable}
              title={
                runnable
                  ? `Track backwards from f${subjectFrame}, ${reachLabel(bounds)}`
                  : `${point.id} has no position on f${currentFrame} to track from — put an anchor here first`
              }
              onClick={() => void act(() => runFrom(point.id, subjectFrame, "bwd", bounds))}
            >
              ◀ track
            </button>
            <button
              disabled={busy || !runnable}
              title={
                runnable
                  ? `Track forwards from f${subjectFrame}, ${reachLabel(bounds)}`
                  : `${point.id} has no position on f${currentFrame} to track from — put an anchor here first`
              }
              onClick={() => void act(() => runFrom(point.id, subjectFrame, "fwd", bounds))}
            >
              track ▶
            </button>
            {/* One slot, one meaning: make the anchor this subject is missing,
                or remove the one it is. */}
            {anchorFrame !== null ? (
              <button
                className="track-inspector__delete"
                disabled={busy || anchors.length <= 1}
                title={
                  anchors.length <= 1
                    ? "A track needs at least one anchor to track from — this is its last one"
                    : "Delete this anchor, and whatever was tracked from it"
                }
                onClick={() => void act(() => deleteAnchorAt(point.id, anchorFrame))}
              >
                ×
              </button>
            ) : (
              <button
                className="track-inspector__add"
                disabled={busy}
                title={`Anchor ${point.id} at frame ${currentFrame} where it already is, or click the frame to place it (⇧A)`}
                onClick={async () => {
                  if (!(await addAnchorHere(point.id))) setTool("add-anchor");
                }}
              >
                + anchor
              </button>
            )}
          </div>

          <div className="track-inspector__row">
            <button
              className={point.anchors_changed ? "button--active" : ""}
              disabled={busy}
              title={`Track from all ${anchors.length} anchors, forwards and backwards, and merge the runs`}
              onClick={async () => {
                await api.trackPoint(project.id, point.id, {});
                invalidateTrack(point.id);
              }}
            >
              {point.has_track ? "Re-track" : "Track"} all {anchors.length}
            </button>
            {point.anchors_changed && (
              <span
                className="track-pill track-pill--warn"
                title="This track was computed from a different set of anchors. Re-track to plan through the ones you have now."
              >
                ⟳ anchors changed
              </span>
            )}
            <JobProgress job={latestJobFor("track", point.id)} />
            <JobProgress job={latestJobFor("retrack", point.id)} />
          </div>

          {violations.length > 0 && (
            <ul className="track-inspector__flags">
              {violations.map((v) => (
                <li
                  key={`${v.frame}-${v.origin}-${v.direction}`}
                  title={`The run from f${v.origin} stopped here rather than drift past an anchor it disagreed with`}
                  onClick={() => pick(v.frame, anchors.find((a) => a.frame === v.frame) ?? null)}
                >
                  ⚑ anchor f{v.frame} missed by {v.distance.toFixed(1)} px
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Set once, not used constantly — so it is one click away rather than
          four blocks of permanent card. */}
      {setupOpen && (
        <div className="track-setup" onClick={(e) => e.stopPropagation()}>
          <div className="track-setup__row">
            <span className="lbl">Tracker</span>
            {/* No algorithm produced a composite, so there is none to choose.
                Saying "composite" is the whole of what there is to say. */}
            {composite ? (
              <span
                className="track-pill track-pill--idle"
                title="This path is the frame-by-frame average of other tracks, so no tracking algorithm applies to it"
              >
                composite
              </span>
            ) : (
              <>
                <select
                  value={point.tracker}
                  onChange={async (e) => {
                    await api.updatePoint(project.id, point.id, { tracker: e.target.value });
                    await refresh();
                  }}
                >
                  {trackers.map((t) => (
                    <option key={t.name} value={t.name} disabled={!t.available} title={t.unavailable_reason ?? undefined}>
                      {t.display_name}
                      {!t.available ? " (unavailable)" : ""}
                    </option>
                  ))}
                </select>
                {unavailableReason(trackers, point.tracker) && (
                  <span
                    className="points-list__warning"
                    title={unavailableReason(trackers, point.tracker) ?? undefined}
                  >
                    ⚠ not installed
                  </span>
                )}
              </>
            )}
            <span style={{ flex: 1 }} />
            <button
              className="track-setup__delete"
              title="Delete this track and everything tracked for it"
              onClick={async () => {
                if (!window.confirm(`Delete the track ${point.id}? This cannot be undone.`)) return;
                await api.deletePoint(project.id, point.id);
                invalidateTrack(point.id);
                await refresh();
              }}
            >
              Delete track
            </button>
          </div>

          <div className="track-setup__row">
            <span className="lbl">Overlays</span>
            {OVERLAY_LABELS.map(([key, label, help]) => {
              const override = trackOverrides[point.id]?.[key];
              const active = overlaysFor(point.id)[key];
              // Dead controls while the row's eye is off: nothing here can
              // change what is drawn until the track is shown again.
              const hidden = isTrackHidden(point.id);
              return (
                <button
                  key={key}
                  disabled={hidden}
                  className={`track-overlay ${active ? "track-overlay--on" : ""} ${
                    override === undefined ? "" : "track-overlay--set"
                  }`}
                  title={`${help}
${
                    hidden
                      ? "This track is hidden — click the eye on its row to show it"
                      : override === undefined
                        ? `Following the default (${overlays[key] ? "shown" : "hidden"}) — click to override`
                        : "Overridden for this track — click again to follow the default"
                  }`}
                  onClick={() =>
                    setTrackOverlay(point.id, key, override === undefined ? !overlays[key] : null)
                  }
                >
                  {label}
                </button>
              );
            })}
          </div>

          {/* The pattern and search sizes are a tracker's settings, and a
              composite has no tracker to settle them for. Its edits stay: an
              averaged path can be nudged, filled or smoothed like any other. */}
          {!composite && <PatchPanel pointId={point.id} />}
          <TrackEdits pointId={point.id} />
        </div>
      )}
    </li>
  );
}

/** Colour of an anchor no tracking run could reach — the "go and look" mark. */
const FLAGGED = "#ff3b30";

const STRIP_HEIGHT = 26;
/** How close (in CSS pixels) a click has to be to an anchor to mean that anchor. */
const ANCHOR_GRAB_PX = 8;

/**
 * One track, drawn on the clip's own time axis.
 *
 * Everything the old card said in four separate blocks — how much is tracked,
 * how good it is, where the holes are, where the anchors sit, which one no run
 * could reach, where the playhead is, how far a running job has got — is one
 * picture, because all of it is a function of the same axis.
 *
 * Drawn per *pixel column* rather than per frame: a long clip has thousands of
 * frames and this redraws on every playhead move, so the cost is bounded by the
 * width of the panel instead of the length of the footage. Gaps are then drawn
 * from their real ranges with a minimum width, so a three-frame hole in a
 * two-thousand-frame clip is still visible rather than rounded away.
 */
function TrackStrip({
  point,
  color,
  track,
  anchors,
  flagged,
  gaps,
  frames,
  currentFrame,
  pickedAnchor,
  liveFrame,
  onPick,
}: {
  point: PointSummary;
  /** The track's colour, resolved by the row (its own, or the palette's). */
  color: string;
  track: TrackData | undefined;
  anchors: Query[];
  flagged: Set<number>;
  gaps: [number, number][];
  frames: number;
  currentFrame: number;
  pickedAnchor: number | null;
  liveFrame: number | null;
  onPick: (frame: number, anchor: Query | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => setWidth(canvas.clientWidth));
    observer.observe(canvas);
    setWidth(canvas.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.floor(STRIP_HEIGHT * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const atFrame = (frame: number) => (frame / frames) * width;

    // Untracked ground.
    ctx.fillStyle = "#2a2f37";
    ctx.fillRect(0, 0, width, STRIP_HEIGHT);

    // Coverage, one column at a time, coloured by that frame's confidence.
    if (track) {
      for (let px = 0; px < Math.ceil(width); px++) {
        const frame = Math.min(frames - 1, Math.floor((px / width) * frames));
        if (track.x[frame] == null) continue;
        ctx.fillStyle = confidenceColor(track.confidence[frame] ?? 0);
        ctx.fillRect(px, 0, 1, STRIP_HEIGHT);
      }
    }

    // Holes inside the tracked span, drawn from their real ranges so a short
    // one does not vanish into a pixel column that rounded past it.
    for (const [start, end] of gaps) {
      const x0 = atFrame(start);
      const x1 = Math.max(atFrame(end + 1), x0 + 2);
      ctx.fillStyle = "#4a2626";
      ctx.fillRect(x0, 0, x1 - x0, STRIP_HEIGHT);
      ctx.fillStyle = "rgba(255, 123, 114, 0.55)";
      ctx.fillRect(x0, STRIP_HEIGHT - 3, x1 - x0, 3);
    }

    // A run in flight, in the point's own colour: how far it has got.
    if (liveFrame != null) {
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(atFrame(liveFrame) - 1, 0, 2, STRIP_HEIGHT);
      ctx.globalAlpha = 1;
    }

    // Anchors. The picked one is bigger and ringed; one no run could reach is
    // red, which is the same language the main timeline speaks.
    for (const anchor of anchors) {
      const x = atFrame(anchor.frame);
      const missed = flagged.has(anchor.frame);
      const picked = anchor.frame === pickedAnchor;
      const size = picked ? 9 : 7;
      ctx.save();
      ctx.translate(x, STRIP_HEIGHT - size / 2 - 1);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = missed ? FLAGGED : color;
      ctx.strokeStyle = picked ? "#ffffff" : "rgba(255, 255, 255, 0.75)";
      ctx.lineWidth = picked ? 1.5 : 1;
      ctx.fillRect(-size / 2, -size / 2, size, size);
      ctx.strokeRect(-size / 2, -size / 2, size, size);
      ctx.restore();
      if (missed) {
        ctx.fillStyle = FLAGGED;
        ctx.fillRect(x - 0.5, 0, 1, STRIP_HEIGHT);
      }
    }

    // Playhead last, over everything, as it is on the main timeline.
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(atFrame(currentFrame) - 1, 0, 2, STRIP_HEIGHT);
  }, [
    width,
    track,
    anchors,
    flagged,
    gaps,
    frames,
    currentFrame,
    pickedAnchor,
    liveFrame,
    point.id,
    color,
  ]);

  /** Which anchor a click means, or null for "the frame you clicked". */
  function anchorAt(canvasX: number): Query | null {
    let best: { anchor: Query; distance: number } | null = null;
    for (const anchor of anchors) {
      const distance = Math.abs((anchor.frame / frames) * width - canvasX);
      if (distance <= ANCHOR_GRAB_PX && (!best || distance < best.distance)) {
        best = { anchor, distance };
      }
    }
    return best?.anchor ?? null;
  }

  return (
    <canvas
      ref={canvasRef}
      className="track-strip"
      style={{ height: STRIP_HEIGHT }}
      title="Click an anchor to act on it, or anywhere else to go to that frame"
      onMouseDown={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const canvasX = event.clientX - bounds.left;
        const anchor = anchorAt(canvasX);
        const frame = Math.max(
          0,
          Math.min(frames - 1, Math.round((canvasX / bounds.width) * frames)),
        );
        onPick(anchor ? anchor.frame : frame, anchor);
      }}
    />
  );
}

/**
 * The runs of frames with no position that sit *inside* what was tracked.
 *
 * Not the server's `gaps`, which counts every invalid frame including the ones
 * before a track starts and after it ends: on a point tracked from one anchor
 * over forty frames that would paint the whole clip as damage. A hole is only a
 * hole if there is a tracked frame on both sides of it.
 */
function interiorGaps(track: TrackData | undefined): [number, number][] {
  if (!track) return [];
  const first = track.x.findIndex((v) => v != null);
  if (first < 0) return [];
  let last = track.x.length - 1;
  while (last > first && track.x[last] == null) last--;
  const runs: [number, number][] = [];
  let start: number | null = null;
  for (let frame = first; frame <= last; frame++) {
    if (track.x[frame] == null && start === null) start = frame;
    else if (track.x[frame] != null && start !== null) {
      runs.push([start, frame - 1]);
      start = null;
    }
  }
  return runs;
}

/**
 * The one thing worth saying about a track at a glance.
 *
 * Deliberately a single pill rather than every fact at once: the strip beside
 * it carries the detail, and a row that lists four states is a row nobody
 * reads. Ordered by what would make you open the track — an anchor no run could
 * reach beats anything else, and "ok" earns the quietest treatment there is.
 */
function statusOf(
  point: PointSummary,
  gaps: number,
  missed: number,
): { tone: "ok" | "warn" | "crit" | "idle"; label: string; help: string } {
  // A composite is not a tracked track and none of the states below apply to
  // it: no run reached or missed an anchor, and "re-track" is not a thing it
  // can be asked to do (todo 2, tenth round, second pass).
  if (point.derived_from.length > 0) {
    return {
      tone: "idle",
      label: "composite",
      help: `The frame-by-frame average of ${point.derived_from.join(" + ")} — not tracked itself`,
    };
  }
  if (!point.has_track) {
    return { tone: "idle", label: "not tracked", help: "Track from an anchor to start this off" };
  }
  if (missed > 0) {
    return {
      tone: "crit",
      label: `⚑ ${missed}`,
      help: `${missed} anchor${missed === 1 ? "" : "s"} no run could reach — these need a human`,
    };
  }
  if (point.anchors_changed) {
    return {
      tone: "warn",
      label: "re-track",
      help: "The anchors have changed since this track was computed",
    };
  }
  if (gaps > 0) {
    return {
      tone: "warn",
      label: `${gaps} gap${gaps === 1 ? "" : "s"}`,
      help: "Frames inside the tracked span with no position — fill or delete them under ⋯",
    };
  }
  return { tone: "ok", label: "ok", help: "Tracked throughout, every anchor reached" };
}

/**
 * Why the tracker a point is *already set to* cannot run here, if it cannot.
 *
 * A project made on a machine with PyTorch, opened on one without: the dropdown
 * greys that option out, which is silent about the point already using it.
 */
function unavailableReason(trackers: TrackerInfo[], name: string): string | null {
  const match = trackers.find((t) => t.name === name);
  if (!match || match.available) return null;
  return match.unavailable_reason ?? `${name} is not available in this environment`;
}
