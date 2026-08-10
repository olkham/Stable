/**
 * The two controls a track row carries about *itself*: is it drawn, and in what
 * colour (todos 6 and 7, ninth round).
 *
 * Shared rather than written twice because there are two Tracks panels behind a
 * flag (`flags.ts`) and both must mean exactly the same thing by an eye and a
 * swatch — a visibility control that behaved differently in the two variants
 * would make the flag impossible to judge.
 */

import { useEffect, useState } from "react";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";

/** How long the picker has to be still before the colour is stored. */
const COMMIT_MS = 400;

/**
 * Hide or show everything one track draws on the frame, in one click.
 *
 * The per-element pills below it stay untouched while it is off: this is a
 * layer above them (`overlaysFor`), so unhiding restores whatever the operator
 * had chosen rather than a row of defaults.
 */
export function TrackEye({ pointId }: { pointId: string }) {
  const { isTrackHidden, setTrackHidden } = useProjectContext();
  const hidden = isTrackHidden(pointId);
  return (
    <button
      className={`track-eye ${hidden ? "track-eye--off" : ""}`}
      aria-pressed={hidden}
      aria-label={`${hidden ? "Show" : "Hide"} ${pointId} on the frame`}
      title={
        hidden
          ? `${pointId} is hidden on the frame — click to show it again`
          : `Hide everything ${pointId} draws on the frame: its path, markers, anchors and boxes`
      }
      onClick={(e) => {
        e.stopPropagation();
        setTrackHidden(pointId, !hidden);
      }}
    >
      👁
    </button>
  );
}

/**
 * The track's colour, as the swatch that has always shown it.
 *
 * The palette gives eight colours to an unbounded number of tracks, so two of
 * them can collide with no recourse — hence a real choice. A native picker
 * reports every colour the pointer passes over inside it, and each commit is a
 * request *and* a project save, so the swatch follows the hand and the project
 * is told once the hand stops (`COMMIT_MS`). Not on blur alone: the OS dialogue
 * keeps focus, so "I chose a colour and nothing happened" would be the result.
 *
 * `↺` appears only once the point carries a colour of its own, since that is
 * the only time going back to the palette means anything.
 */
export function TrackSwatch({ pointId, className }: { pointId: string; className: string }) {
  const { project, colorFor, setPointColor } = useProjectContext();
  const stored = project.points.find((p) => p.id === pointId)?.color ?? null;
  const current = colorFor(pointId);
  // The picker is live under the operator's hand; the project is told once, on
  // release. Re-seeded whenever the stored answer changes, so an undo, another
  // panel, or a refresh is never argued with.
  const [draft, setDraft] = useState(current);
  useEffect(() => setDraft(current), [current]);
  useEffect(() => {
    if (draft.toLowerCase() === current.toLowerCase()) return;
    const timer = setTimeout(() => void setPointColor(pointId, draft), COMMIT_MS);
    return () => clearTimeout(timer);
  }, [draft, current, pointId, setPointColor]);

  return (
    <span className="track-swatch">
      <input
        type="color"
        className={className}
        value={draft}
        aria-label={`Colour of ${pointId}`}
        title={`The colour ${pointId} is drawn in, everywhere. Click to change it.`}
        onClick={(e) => e.stopPropagation()}
        onChange={(e) => setDraft(e.target.value)}
      />
      {stored && (
        <button
          className="track-swatch__reset"
          title={`Back to the automatic colour (${pointColor(pointId)})`}
          onClick={(e) => {
            e.stopPropagation();
            void setPointColor(pointId, null);
          }}
        >
          ↺
        </button>
      )}
    </span>
  );
}
