// Feature flags: things that are built, shipped, and still being judged.
//
// This exists for one job at a time. Right now that job is the Tracks panel,
// which has two implementations — the card it has always had, and the
// timeline-per-track redesign — so the two can be used against each other on
// real footage rather than compared in the abstract.
//
// Resolution order, highest first:
//
//   1. `?panel=timeline` / `?panel=classic` in the URL. Per tab and not
//      remembered, which is what makes a genuine side-by-side possible: open
//      the same project twice, one variant each, and switch windows.
//   2. Whatever was last chosen with the switch in the panel header,
//      remembered in `localStorage` so a reload does not undo the choice.
//   3. The default below.
//
// A flag that outlives its decision is technical debt with a nice name. When
// one variant wins, delete the other and delete this.

import { useCallback, useState } from "react";

/** Which Tracks panel to render. */
export type TracksPanel = "timeline" | "classic";

/**
 * What a fresh browser gets.
 *
 * `timeline` — the new one is on by default because it is the one under
 * judgement; a variant nobody sees collects no opinions. The switch back is in
 * the panel header, one click, and it sticks.
 */
export const DEFAULT_TRACKS_PANEL: TracksPanel = "timeline";

const STORAGE_KEY = "stable.flags.tracksPanel";
const QUERY_KEY = "panel";

function isVariant(value: unknown): value is TracksPanel {
  return value === "timeline" || value === "classic";
}

/** Read the flag from the URL, then storage, then the default. */
export function readTracksPanel(): TracksPanel {
  try {
    const fromUrl = new URLSearchParams(window.location.search).get(QUERY_KEY);
    if (isVariant(fromUrl)) return fromUrl;
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (isVariant(stored)) return stored;
  } catch {
    // Private-mode storage, a sandboxed frame: a flag is never worth a crash.
  }
  return DEFAULT_TRACKS_PANEL;
}

/**
 * The flag, and a setter that remembers the choice.
 *
 * The URL parameter seeds the initial value but does not pin it — switching in
 * a tab opened with `?panel=classic` still works, and still writes the choice
 * through, which is the behaviour that surprises least.
 */
export function useTracksPanelFlag(): [TracksPanel, (next: TracksPanel) => void] {
  const [variant, setVariant] = useState<TracksPanel>(readTracksPanel);
  const choose = useCallback((next: TracksPanel) => {
    setVariant(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Not remembering it is a smaller failure than refusing to switch.
    }
  }, []);
  return [variant, choose];
}
