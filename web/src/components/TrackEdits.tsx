import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "../api/client";
import { useFilterCatalogue } from "../hooks/useFilterCatalogue";
import { useProjectContext } from "../state/ProjectContext";
import type { Correction } from "../types";

/**
 * The filter a new per-track smoothing starts on.
 *
 * Mirrors `stable/core/corrections.py`'s `DEFAULT_SMOOTH_FILTER`, and for the
 * same reason: a track path is a few hundred noisy pixel samples, not a motion
 * curve being cut by frequency, so a width in frames is the parameter an
 * operator can actually reason about.
 */
const DEFAULT_SMOOTH_FILTER = "gaussian";

/** One-line human description of what an edit does, for the correction list. */
function describe(correction: Correction): string {
  switch (correction.kind) {
    case "offset": {
      const dx = (correction.dx ?? 0).toFixed(1);
      const dy = (correction.dy ?? 0).toFixed(1);
      const falloff = correction.falloff_before ?? 0;
      return `nudge (${dx}, ${dy}) @ ${correction.frame}${falloff ? ` ±${falloff}f` : ""}`;
    }
    case "replace":
      return `set to (${(correction.x ?? 0).toFixed(1)}, ${(correction.y ?? 0).toFixed(1)}) @ ${correction.frame}`;
    case "gap_fill":
      return `${correction.method} fill ${correction.start}–${correction.end}`;
    case "delete_range":
      return `delete ${correction.start}–${correction.end}`;
    case "retrack":
      return `retrack ${correction.direction} from ${correction.frame}`;
    case "smooth":
      return `${correction.filter ?? "smooth"} smooth ${correction.start}–${correction.end}`;
    default:
      return correction.kind;
  }
}

/**
 * One track's edits: its gaps, its range repairs, and its correction list.
 *
 * This is what is left of the Edit tab (todo 4, sixth round). That tab held four
 * things and three of them were duplicates by the time it was removed: undo/redo
 * is in the workspace header, "retrack from here" is the bounded run on this
 * same card — which can stop at the next anchor rather than always ploughing
 * to the end of the clip — and moving or deleting a point is something the
 * canvas has done directly since the tools landed. The genuinely global setting
 * it carried, what a drag *means*, moved to the viewer's toolbar where the drag
 * happens.
 *
 * What was not a duplicate is here, and it is here rather than on a tab of its
 * own because all of it is per point: a correction belongs to a track, and
 * reading a list of them next to the track's anchors and its patches is the
 * whole reason the card exists.
 *
 * Nothing here rewrites a track — every button appends, toggles or removes one
 * entry in the point's corrections list, which is what makes undo a single cheap
 * operation rather than a history of arrays.
 */
export function TrackEdits({ pointId }: { pointId: string }) {
  const {
    project,
    currentFrame,
    setCurrentFrame,
    tracks,
    applyCorrection,
    afterEdit,
    editError,
    setEditError,
  } = useProjectContext();

  const track = tracks[pointId];
  const corrections = track?.corrections ?? [];
  const gaps = track?.gaps ?? [];

  const [open, setOpen] = useState(false);
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(0);
  const [method, setMethod] = useState<"spline" | "linear">("spline");
  // The per-track smoothing (todo 1, tenth round): which filter, and its own
  // parameters. The catalogue is the server's, so nothing about a filter is
  // written down here — see `useFilterCatalogue`.
  const catalogue = useFilterCatalogue();
  const [filter, setFilter] = useState(DEFAULT_SMOOTH_FILTER);
  const [params, setParams] = useState<Record<string, number>>({});
  const info = useMemo(
    () => catalogue.find((entry) => entry.name === filter),
    [catalogue, filter],
  );

  const defaultsFor = useCallback(
    (name: string): Record<string, number> => {
      const entry = catalogue.find((row) => row.name === name);
      const values: Record<string, number> = {};
      for (const param of entry?.params ?? []) values[param.name] = param.default;
      return values;
    },
    [catalogue],
  );

  // Seed the parameter boxes from the chosen filter's own defaults, and re-seed
  // them when the catalogue arrives (it is a request, so the first render has
  // nothing to read). Never overwrites a value the operator has typed for the
  // filter currently chosen — the effect is keyed on the filter, not on them.
  useEffect(() => {
    setParams(defaultsFor(filter));
  }, [filter, defaultsFor]);

  // Default the range fields to a sensible window around the playhead the first
  // time the section is opened, so the buttons are usable without typing.
  useEffect(() => {
    if (!open) return;
    setRangeStart(Math.max(0, currentFrame - 3));
    setRangeEnd(Math.min(project.video.frames - 1, currentFrame + 3));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, pointId]);

  async function toggle(correction: Correction) {
    setEditError(null);
    try {
      await api.setCorrectionEnabled(project.id, pointId, correction.id, !correction.enabled);
      await afterEdit(pointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  async function remove(correction: Correction) {
    setEditError(null);
    try {
      await api.deleteCorrection(project.id, pointId, correction.id);
      await afterEdit(pointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  async function revert() {
    setEditError(null);
    try {
      await api.clearCorrections(project.id, pointId);
      await afterEdit(pointId);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }

  // Nothing to say about a track with no edits and no gaps, so the section is
  // not drawn at all rather than drawn empty — most cards are in that state.
  const worthShowing = corrections.length > 0 || gaps.length > 0 || open;

  return (
    <div className="track-edits" onClick={(e) => e.stopPropagation()}>
      <button
        className="link-button track-edits__toggle"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        title="Gaps, range repairs and every correction on this track"
      >
        {open ? "▾" : "▸"} Edits
        {corrections.length > 0 && ` (${corrections.length})`}
        {gaps.length > 0 && ` · ${gaps.length} gap${gaps.length === 1 ? "" : "s"}`}
      </button>

      {editError && open && <div className="error-text">{editError}</div>}

      {worthShowing && gaps.length > 0 && (
        <div className="track-edits__gaps">
          {gaps.map(([start, end]) => (
            <span key={`${start}-${end}`} className="anchor-chip">
              <button className="link-button" onClick={() => setCurrentFrame(start)}>
                {start}–{end}
              </button>
              <button
                className="anchor-chip__delete"
                title="Fill this gap"
                onClick={() =>
                  void applyCorrection(pointId, {
                    kind: "gap_fill",
                    start,
                    end,
                    method: "spline",
                    confidence: 0.5,
                  })
                }
              >
                ⤳
              </button>
            </span>
          ))}
        </div>
      )}

      {open && (
        <>
          <div className="field-row">
            <label>Frames</label>
            <input
              type="number"
              min={0}
              max={Math.max(project.video.frames - 1, 0)}
              value={rangeStart}
              onChange={(e) => setRangeStart(Number(e.target.value))}
            />
            <span>–</span>
            <input
              type="number"
              min={0}
              max={Math.max(project.video.frames - 1, 0)}
              value={rangeEnd}
              onChange={(e) => setRangeEnd(Number(e.target.value))}
            />
            <button onClick={() => setRangeStart(currentFrame)}>in</button>
            <button onClick={() => setRangeEnd(currentFrame)}>out</button>
          </div>
          <div className="field-row">
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as "spline" | "linear")}
            >
              <option value="spline">spline</option>
              <option value="linear">linear</option>
            </select>
            <button
              onClick={() =>
                void applyCorrection(pointId, {
                  kind: "gap_fill",
                  start: rangeStart,
                  end: rangeEnd,
                  method,
                  confidence: 0.5,
                })
              }
            >
              Fill gap
            </button>
            <button
              onClick={() =>
                void applyCorrection(pointId, {
                  kind: "delete_range",
                  start: rangeStart,
                  end: rangeEnd,
                })
              }
            >
              Delete range
            </button>
          </div>
          <p className="modal__hint">
            Deleting a bad span is often the whole repair: the solver carries those frames on the
            remaining points. Fill it when this point itself is needed.
          </p>

          {/* Smoothing *this* track's path (todo 1, tenth round). The Smoothing
              tab's X/Y/Rotation/Scale are the camera motion solved from every
              track at once and belong to none of them individually; this is the
              other thing, and it is on the track's own card because that is the
              only place the question "which path?" has an answer. Same filter
              catalogue, same controls, built from the server's registry. */}
          <span className="patch-panel__label">Smooth path</span>
          <div className="field-row">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              title="Which filter to run over this track's x and y between the frames above"
            >
              {catalogue.map((entry) => (
                <option key={entry.name} value={entry.name} title={entry.description}>
                  {entry.display_name}
                </option>
              ))}
            </select>
            <button
              disabled={!info}
              title={
                info
                  ? `Smooth this track's path over frames ${rangeStart}–${rangeEnd}. Non-destructive: it lands in the list below and can be toggled off.`
                  : "Waiting for the filter catalogue"
              }
              onClick={() =>
                void applyCorrection(pointId, {
                  kind: "smooth",
                  start: rangeStart,
                  end: rangeEnd,
                  filter,
                  params,
                })
              }
            >
              Smooth path
            </button>
            <button
              disabled={!info}
              title="Smooth the whole clip rather than the range above"
              onClick={() =>
                void applyCorrection(pointId, {
                  kind: "smooth",
                  start: 0,
                  end: Math.max(project.video.frames - 1, 0),
                  filter,
                  params,
                })
              }
            >
              whole clip
            </button>
          </div>
          {info && (
            <div className="smoothing-channel__params">
              {info.params.map((param) => (
                <label key={param.name} className="param-input" title={param.description}>
                  {param.name}
                  {param.unit ? ` (${param.unit})` : ""}
                  <input
                    type="number"
                    step={param.kind === "int" ? 1 : 0.1}
                    min={param.minimum ?? undefined}
                    max={param.maximum ?? undefined}
                    value={params[param.name] ?? param.default}
                    onChange={(e) =>
                      setParams((prev) => ({ ...prev, [param.name]: Number(e.target.value) }))
                    }
                  />
                </label>
              ))}
            </div>
          )}
          <p className="modal__hint">
            Smooths this track's own trajectory, and nothing else — the frames outside the range
            are untouched, gaps stay gaps, and the confidence is left alone. The edges of the
            range are cross-faded so the path does not step where the smoothing stops.
          </p>

          {corrections.length === 0 ? (
            <p className="panel__empty">No edits — this point is raw tracker output.</p>
          ) : (
            <>
              <span className="patch-panel__label">
                Corrections
                <button className="link-button edit-panel__revert" onClick={() => void revert()}>
                  revert all
                </button>
              </span>
              <ul className="corrections-list">
                {corrections.map((correction) => (
                  <li
                    key={correction.id}
                    className={`corrections-list__item ${correction.enabled ? "" : "corrections-list__item--off"}`}
                  >
                    <input
                      type="checkbox"
                      checked={correction.enabled}
                      onChange={() => void toggle(correction)}
                      title="Toggle this edit off to compare"
                    />
                    <button
                      className="link-button corrections-list__label"
                      onClick={() =>
                        setCurrentFrame(correction.frame ?? correction.start ?? currentFrame)
                      }
                    >
                      {describe(correction)}
                    </button>
                    <button
                      className="points-list__delete"
                      onClick={() => void remove(correction)}
                      title="Remove this edit"
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  );
}
