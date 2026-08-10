/**
 * Making a new track out of the ones already there (todo 2, tenth round).
 *
 * "Create path3 from path1 and path2, where p3 = mean(p1, p2)." On each frame
 * the new path sits at the midpoint of the straight line joining the chosen
 * tracks on *that* frame. The server does the averaging
 * (`stable/core/derive.py`); this is the choosing.
 *
 * Shared by both Tracks panels rather than written into one, for the same reason
 * `TrackVisibility.tsx` is: the two are behind a flag (`flags.ts`) and are being
 * judged against each other, and a feature that existed in only one of them
 * would make that judgement about the wrong thing.
 */

import { useState } from "react";
import * as api from "../api/client";
import { useProjectContext } from "../state/ProjectContext";
import type { PointSummary } from "../types";

/** What each span mode means, on the control that picks it. */
const SPAN_HELP: Record<api.DeriveSpan, string> = {
  overlap:
    "Only the frames every chosen track has a position on, so every point of the new path really is the midpoint of all of them.",
  union:
    "Every frame any chosen track has a position on, averaging whichever are there. More coverage, at the cost of a step wherever a track starts or stops — the new path jumps toward the ones still tracking.",
};

/**
 * What a composite track is made of, in place of the controls it has none of.
 *
 * A composite has no tracker, no matched window and no run behind it, so the
 * tracker picker, the pattern/search sizes, the anchor list and the track
 * buttons are all questions about something that did not happen. The card drops
 * them and shows this instead — which is the one thing there *is* to say about
 * how the path was made.
 *
 * A source that has since been deleted is drawn as plain text: the id is a
 * record of what the path was averaged from, not a live reference, and the path
 * did not change when the source went away.
 */
export function CompositeSources({ point }: { point: PointSummary }) {
  const { project, colorFor, setSelectedPointId, setCurrentFrame, tracks } = useProjectContext();
  if (point.derived_from.length === 0) return null;

  return (
    <div className="composite-sources">
      <span className="lbl">Made from</span>
      {point.derived_from.map((sourceId) => {
        const source = project.points.find((p) => p.id === sourceId);
        if (!source) {
          return (
            <span key={sourceId} className="composite-sources__gone" title="This track has been deleted. The path averaged from it is unchanged.">
              {sourceId} (deleted)
            </span>
          );
        }
        return (
          <button
            key={sourceId}
            className="composite-sources__item"
            title={`Select ${sourceId}, one of the tracks this path is the average of`}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedPointId(sourceId);
              const first = tracks[sourceId]?.anchors?.[0]?.frame;
              if (first !== undefined) setCurrentFrame(first);
            }}
          >
            <span
              className="points-list__swatch"
              style={{ background: colorFor(sourceId) }}
            />
            {sourceId}
          </button>
        );
      })}
      <span className="modal__hint">
        averaged frame by frame — on each frame this sits at their midpoint
      </span>
    </div>
  );
}

export function DeriveTrackPanel() {
  const { project, refresh, setSelectedPointId, loadTrack } = useProjectContext();
  const [open, setOpen] = useState(false);
  const [chosen, setChosen] = useState<string[]>([]);
  const [span, setSpan] = useState<api.DeriveSpan>("overlap");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [made, setMade] = useState<string | null>(null);

  // Only a tracked point has a path to average. An untracked one is a position
  // on one frame, and the server says so rather than guessing — but offering it
  // here would be offering a click that cannot work.
  const tracked = project.points.filter((point) => point.has_track);

  function toggle(pointId: string) {
    setChosen((prev) =>
      prev.includes(pointId) ? prev.filter((id) => id !== pointId) : [...prev, pointId],
    );
  }

  async function create() {
    setBusy(true);
    setError(null);
    setMade(null);
    try {
      const point = await api.derivePoint(project.id, chosen, span);
      await refresh();
      await loadTrack(point.id);
      setSelectedPointId(point.id);
      // The span is worth saying: it is the one thing the mode changes, and
      // "overlap" quietly producing a much shorter track than the operator
      // expected is exactly the surprise worth heading off.
      setMade(
        `${point.id} — the mean of ${point.derived_from.join(" + ")}, over frames ` +
          `${point.derived_frames[0]}–${point.derived_frames[1]}`,
      );
      setChosen([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // Nothing to offer until there are two paths to average, and a control that
  // can never do anything is worse than one that is not there.
  if (tracked.length < 2) return null;

  return (
    <div className="derive-track">
      <button
        className="link-button track-edits__toggle"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        title="Make a new track from the average of two or more tracked paths"
      >
        {open ? "▾" : "▸"} New track from others
      </button>

      {open && (
        <>
          <div className="derive-track__pick">
            {tracked.map((point) => (
              <label key={point.id} className="channel-check">
                <input
                  type="checkbox"
                  checked={chosen.includes(point.id)}
                  onChange={() => toggle(point.id)}
                />
                {point.id}
                <span className="modal__hint"> {point.tracked_frames}f</span>
              </label>
            ))}
          </div>
          <div className="field-row">
            <label>Frames</label>
            <select
              value={span}
              onChange={(e) => setSpan(e.target.value as api.DeriveSpan)}
              title={SPAN_HELP[span]}
            >
              <option value="overlap">only frames every track covers</option>
              <option value="union">every frame any track covers</option>
            </select>
            <button
              disabled={busy || chosen.length < 2}
              title={
                chosen.length < 2
                  ? "Choose at least two tracks to average"
                  : `Create a track holding the mean of ${chosen.join(" + ")}`
              }
              onClick={() => void create()}
            >
              {busy ? "Averaging…" : "Create mean track"}
            </button>
          </div>
          <p className="modal__hint">
            On every frame the new track sits at the midpoint of the line joining the tracks you
            chose on that frame — the centre of them, for more than two. {SPAN_HELP[span]}
          </p>
          {made && <p className="modal__hint">Created {made}.</p>}
          {error && <p className="error-text">{error}</p>}
        </>
      )}
    </div>
  );
}
