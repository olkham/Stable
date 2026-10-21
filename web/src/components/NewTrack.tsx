import { useProjectContext } from "../state/ProjectContext";
import type { TrackInput } from "../types";
import { SettingSlider } from "./SettingSlider";

const INPUT_LABEL: Record<TrackInput, string> = { point: "Point", box: "Box" };

/**
 * The card for a track that does not exist yet.
 *
 * Trackers are started differently — most from a point, with a pattern window
 * and a search area worth sizing first; some from a bounding box — so the
 * order is: choose the tracker, see its options and the marking tool it wants,
 * and only then mark the video. Picking the tracker *after* the click meant the
 * window could only be judged on a track already made with the wrong one.
 *
 * Rendered at the top of both Tracks panels while the draft is open. It knows
 * no tracker by name: what a tracker takes comes from its `inputs`, and its
 * settings from the server's description of them, the same rows the patch
 * panel draws for an existing track. The canvas shows the window under the
 * cursor from the same preview, so the sliders can be judged on the footage.
 */
export function NewTrack() {
  const {
    trackers,
    newTrack,
    updateNewTrack,
    cancelNewTrack,
    newTrackPreview,
    newTrackPreviewError,
  } = useProjectContext();
  if (!newTrack) return null;

  const info = trackers.find((t) => t.name === newTrack.tracker);
  const inputs = info?.inputs ?? [];
  const notes = [info?.license, info?.requires_gpu ? "GPU" : null].filter(Boolean);
  const settings = newTrackPreview?.tracker === newTrack.tracker ? newTrackPreview : null;

  /** Drop one option, so the plugin's own default applies again. */
  function resetOption(attribute: string) {
    const { [attribute]: _dropped, ...rest } = newTrack!.options;
    updateNewTrack({ options: rest });
  }

  return (
    <div className="new-track" onClick={(e) => e.stopPropagation()}>
      <label className="new-track__row">
        <span className="new-track__label">Tracker</span>
        <select
          value={newTrack.tracker}
          onChange={(e) => updateNewTrack({ tracker: e.target.value })}
        >
          {trackers.map((t) => (
            <option
              key={t.name}
              value={t.name}
              disabled={!t.available}
              title={t.unavailable_reason ?? undefined}
            >
              {t.display_name}
            </option>
          ))}
        </select>
      </label>
      {notes.length > 0 && <span className="new-track__note">{notes.join(" · ")}</span>}

      <div className="new-track__row">
        <span className="new-track__label">Mark with</span>
        {inputs.length > 1 ? (
          <div className="new-track__segmented" role="radiogroup" aria-label="Mark with">
            {inputs.map((input) => (
              <button
                key={input}
                className={newTrack.input === input ? "button--active" : ""}
                aria-pressed={newTrack.input === input}
                onClick={() => updateNewTrack({ input })}
              >
                {INPUT_LABEL[input]}
              </button>
            ))}
          </div>
        ) : (
          <span className="new-track__note">
            {newTrack.input === "box" ? "Takes a box only" : "Takes a point only"}
          </span>
        )}
      </div>

      {newTrack.input === "box" ? (
        <span className="new-track__note">
          The box you draw is the pattern: the tracker follows the box's size.
        </span>
      ) : (
        <div className="patch-panel__settings">
          {settings?.settings.map((setting) => {
            const chosen = newTrack.options[setting.attribute];
            return (
              <SettingSlider
                key={setting.attribute}
                setting={setting}
                value={chosen ?? setting.value}
                onChange={(next) =>
                  updateNewTrack({ options: { ...newTrack.options, [setting.attribute]: next } })
                }
                extra={
                  chosen !== undefined && (
                    <button
                      className="link-button"
                      title={`Back to the default (${setting.default})`}
                      onClick={(e) => {
                        e.preventDefault();
                        resetOption(setting.attribute);
                      }}
                    >
                      ↺
                    </button>
                  )
                }
              />
            );
          })}
          {settings && settings.settings.length === 0 && (
            <span className="new-track__note">This tracker has no options to set.</span>
          )}
          {settings && settings.search.radius === null && (
            <span className="new-track__note">{settings.search.note}</span>
          )}
          {newTrackPreviewError && <span className="error-text">{newTrackPreviewError}</span>}
        </div>
      )}

      <div className="new-track__prompt">
        <span>
          {newTrack.input === "box"
            ? "Drag a box around the object on the video"
            : "Click the feature to follow on the video"}
        </span>
        <button onClick={cancelNewTrack}>Cancel</button>
      </div>
    </div>
  );
}
