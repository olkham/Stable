import type { TracksPanel } from "../flags";

/**
 * Switch between the two Tracks panels (`flags.ts`).
 *
 * Rendered by *both* variants, in the same place in the header, so whichever
 * one you are looking at the way back is where you left it. It says what it
 * would switch *to*, not what is on: a control that names the current state
 * reads as a label, and this one is a door.
 */
export function PanelSwitch({
  variant,
  onChange,
}: {
  variant: TracksPanel;
  onChange: (next: TracksPanel) => void;
}) {
  const next: TracksPanel = variant === "timeline" ? "classic" : "timeline";
  return (
    <button
      className="panel-switch"
      title={
        next === "classic"
          ? "Switch back to the original track cards. Remembered until you switch again; ?panel=classic in the URL opens a tab straight into it."
          : "Switch to the timeline panel: each track's anchors drawn on the clip, with one set of run controls. ?panel=timeline opens a tab straight into it."
      }
      onClick={() => onChange(next)}
    >
      {next === "classic" ? "classic panel" : "timeline panel"} ⇄
    </button>
  );
}
