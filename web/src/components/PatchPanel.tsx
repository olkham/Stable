import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import type { PointPatches, TrackerSetting } from "../types";

/** Largest a thumbnail is drawn, whatever the patch's own pixel size. */
const THUMB_PX = 72;

/** How long the frame has to stop moving before the panel re-reads itself. */
const SETTLE_MS = 150;

/**
 * The pixels a point is actually being tracked on (todo 14).
 *
 * Todo 14 is phrased in terms of template matching — "see the template patch",
 * "a grid of thumbnails" — but the need is not specific to that tracker: for
 * *any* point, the useful question is "what does the tracker see here, and does
 * it still look like what it saw at the keyframe?". A template that is a crisp
 * corner at frame 0 and motion-blurred sky at frame 200 explains a failed track
 * instantly, and no other view in this app shows it.
 *
 * So the panel is shown for every point, and so is the size control: "the thing
 * I am tracking is bigger than 33 px" is a statement about the subject, not
 * about which plugin happens to be selected. The server says which settings the
 * point's own tracker accepts and in what units (`settings`, from
 * stable/server/tracker_options.py), and one control is drawn per row — the
 * panel names no tracker and knows no argument, so a plugin that declares a
 * window gets a working control here for free, and one that declares none gets
 * an explanation instead of a slider that would change nothing.
 *
 * Patches are rendered server-side from the source clip and drawn pixelated on
 * purpose: at this size, smoothing would hide exactly the texture being judged.
 */
export function PatchPanel({ pointId }: { pointId: string }) {
  const { project, currentFrame, setCurrentFrame, editRevision, refresh } = useProjectContext();
  const [info, setInfo] = useState<PointPatches | null>(null);
  // Pending edits to the settings, keyed by the plugin's own argument name.
  // Only what has been touched lives here, so a reload never fights the user.
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const next = await api.getPointPatches(project.id, pointId, currentFrame);
      setInfo(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
    // `editRevision` is a dependency on purpose: a correction or a re-track
    // moves the point, so every patch below is now of different pixels.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, pointId, currentFrame, editRevision]);

  // Debounced, because this fires on every frame change: dragging the playhead
  // across a thousand frames used to mean a thousand of these in flight, which
  // is most of how scrubbing came to a halt. Waiting for the scrub to settle
  // costs nothing visible — the thumbnails below are a considered look, not a
  // per-frame readout.
  useEffect(() => {
    const timer = setTimeout(() => void reload(), info === null ? 0 : SETTLE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload]);

  // A different point (or a tracker change) invalidates every pending edit:
  // the settings are per point and named per plugin.
  useEffect(() => setDraft({}), [pointId, info?.tracker]);

  if (error) return <div className="error-text">{error}</div>;
  if (!info) return <div className="patch-panel">Loading patches…</div>;

  const settings = info.settings;
  const windowSetting = settings.find((s) => s.role === "window");
  const value = (setting: TrackerSetting) => draft[setting.attribute] ?? setting.value;
  const dirty = settings.some((s) => value(s) !== s.value);
  // What the thumbnails should show: the pattern size being previewed, which is
  // the pending window value while the slider is being dragged.
  const previewRadius =
    (windowSetting ? radiusOf(windowSetting, value(windowSetting)) : 0) || info.radius;
  // Cache-buster: a patch URL is (point, frame, radius), but the *pixels* also
  // change when the track moves, and that does not alter the URL.
  const rev = editRevision;

  async function commit(body: Record<string, number | null>) {
    setBusy(true);
    try {
      const next = await api.updatePointPatches(project.id, pointId, body);
      setInfo(next);
      setDraft({});
      setError(null);
      // The canvas draws its pattern/search boxes from the point summary, so
      // the sizes have to be re-read there too or the boxes lag the panel.
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  /** Save every pending change in one request. */
  const apply = () =>
    commit(
      Object.fromEntries(
        settings.filter((s) => value(s) !== s.value).map((s) => [s.attribute, value(s)]),
      ),
    );

  /** Clear every override this point has, restoring the plugin's own defaults. */
  const reset = () => commit(Object.fromEntries(settings.map((s) => [s.attribute, null])));

  function thumb(patch: PointPatches["patches"][number], key: string) {
    const isCurrent = patch.frame === currentFrame;
    return (
      <figure
        key={key}
        className={`patch-thumb ${isCurrent ? "patch-thumb--current" : ""} patch-thumb--${patch.kind}`}
        onClick={() => setCurrentFrame(patch.frame)}
        title={
          patch.kind === "keyframe"
            ? `Keyframe ${patch.frame} — the position you set. Click to jump there.`
            : `Frame ${patch.frame} — where the tracker put it. Click to jump there.`
        }
      >
        {patch.available ? (
          <img
            src={api.pointPatchUrl(
              project.id,
              pointId,
              patch.frame,
              previewRadius,
              rev,
              // 1:1 source pixels here: this panel is where the operator judges
              // whether the patch has enough texture to track, which is exactly
              // what proxy scaling takes away. It is a handful of URLs for one
              // point, so the cost is bounded — unlike the strip below.
              "original",
            )}
            width={THUMB_PX}
            height={THUMB_PX}
            alt={`Patch at frame ${patch.frame}`}
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div
            className="patch-thumb__missing"
            style={{ width: THUMB_PX, height: THUMB_PX }}
            title="This patch would fall outside the frame at this size"
          >
            ✕
          </div>
        )}
        <figcaption>
          {patch.kind === "keyframe" ? "◆" : ""}f{patch.frame}
        </figcaption>
      </figure>
    );
  }

  return (
    <div className="patch-panel" onClick={(e) => e.stopPropagation()}>
      <div className="patch-panel__settings">
        {settings.map((setting) => (
          <label key={setting.attribute} className="patch-setting" title={setting.help}>
            <span className="patch-setting__label">{setting.label}</span>
            <input
              type="range"
              min={setting.minimum}
              max={setting.maximum}
              step={setting.step}
              value={value(setting)}
              disabled={busy}
              onChange={(e) =>
                setDraft((prev) => ({ ...prev, [setting.attribute]: Number(e.target.value) }))
              }
            />
            <span className="patch-setting__value">
              {describeSetting(setting, value(setting))}
              {value(setting) !== setting.value && (
                <span className="patch-panel__pending"> (preview)</span>
              )}
            </span>
          </label>
        ))}

        {settings.length === 0 && (
          <span className="patch-panel__size" title={patchSizeExplanation(info)}>
            {info.patch_size}px window · {info.tracker}
          </span>
        )}
        {/* The reach in pixels, whenever no slider above already states it —
            a pyramid depth is a count, and "3 levels" does not answer "how far
            can this point move between frames?". */}
        {!settings.some((s) => s.role === "search" && s.unit !== "count") && (
          <span className="patch-panel__note" title={info.search.note}>
            {info.search.radius === null
              ? "No search area"
              : `Search reach ±${info.search.radius}px`}
          </span>
        )}

        {settings.length > 0 && (
          <div className="patch-panel__controls">
            <button onClick={apply} disabled={busy || !dirty}>
              Apply
            </button>
            <button
              onClick={reset}
              disabled={busy || !info.custom}
              title={`Back to ${info.tracker}'s own defaults`}
            >
              Reset
            </button>
            {dirty && (
              <span className="patch-panel__pending">
                Not saved — the next run still uses the old size
              </span>
            )}
          </div>
        )}
      </div>

      {info.current && (
        <div className="patch-panel__now">
          <span className="patch-panel__label">This frame</span>
          {thumb(info.current, "current")}
          {/* Why this looks bigger than the box on the canvas: it is. A patch is
              a few dozen pixels drawn at THUMB_PX, and without saying so the two
              views of the same window look like two different sizes. */}
          <span className="patch-panel__note">
            {previewRadius * 2 + 1}px window, shown at{" "}
            {(THUMB_PX / (previewRadius * 2 + 1)).toFixed(1)}×
          </span>
        </div>
      )}

      {info.patches.length === 0 ? (
        <p className="patch-panel__empty">
          No keyframes yet — place one to see the patch it would track from.
        </p>
      ) : (
        <>
          <span className="patch-panel__label">Keyframes</span>
          <div className="patch-panel__grid">
            {info.patches.map((patch) => thumb(patch, `k${patch.frame}`))}
          </div>
        </>
      )}

      {info.patches.some((p) => !p.available) && (
        <p className="patch-panel__warning">
          ✕ marks a keyframe too close to the frame edge for this size — the tracker would refuse
          it too. Reduce the size or move the keyframe inward.
        </p>
      )}
    </div>
  );
}

/** Size of a thumbnail in the all-points strip — smaller, since there are many. */
const STRIP_PX = 48;

/**
 * Every tracked point's patch on the current frame, side by side.
 *
 * The per-point panel above answers "is *this* point still on target"; this
 * answers "which of my points has drifted", which is the question you actually
 * have on a shot with eight of them. One row, current frame, click to select.
 *
 * Each thumbnail is fetched at the point's own effective radius (the image
 * endpoint defaults to it), so no metadata round trip is needed — a point whose
 * patch cannot be built at this frame simply hides itself via `onError`, which
 * is the right outcome for a strip that is a glance rather than a report.
 */
export function AllPatchesStrip() {
  const { project, currentFrame, selectedPointId, setSelectedPointId, editRevision } =
    useProjectContext();
  const [broken, setBroken] = useState<Record<string, boolean>>({});

  // A frame change or an edit makes every previous failure stale.
  useEffect(() => setBroken({}), [currentFrame, editRevision]);

  const tracked = project.points.filter((point) => point.has_track || point.queries.length > 0);
  if (tracked.length === 0) return null;
  const visible = tracked.filter((point) => !broken[point.id]);

  return (
    <div className="patch-strip">
      <span className="patch-panel__label">
        Tracking on frame {currentFrame}
        {visible.length === 0 && " — no patch available here"}
      </span>
      <div className="patch-strip__row">
        {tracked.map((point) => (
          <figure
            key={point.id}
            className={`patch-thumb ${
              point.id === selectedPointId ? "patch-thumb--current" : ""
            }`}
            style={{ display: broken[point.id] ? "none" : undefined }}
            onClick={() => setSelectedPointId(point.id)}
            title={`${point.id} (${point.tracker}) — click to select`}
          >
            <img
              src={api.pointPatchUrl(project.id, point.id, currentFrame, undefined, editRevision)}
              width={STRIP_PX}
              height={STRIP_PX}
              alt={`${point.id} patch`}
              style={{ imageRendering: "pixelated" }}
              onError={() => setBroken((prev) => ({ ...prev, [point.id]: true }))}
            />
            <figcaption style={{ color: pointColor(point.id) }}>{point.id}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

function patchSizeExplanation(info: PointPatches): string {
  if (info.radius_source) {
    return `The region ${info.tracker} matches, from its own ${info.radius_source}.`;
  }
  return `${info.tracker} has no local search window — it sees whole frames — so this is a preview size only.`;
}

/** The radius in source pixels a setting's value corresponds to. */
function radiusOf(setting: TrackerSetting, value: number): number {
  if (setting.unit === "side") return Math.floor(value / 2);
  if (setting.unit === "radius") return value;
  return setting.radius; // not a length (pyramid depth); its own radius stands
}

/**
 * How a value reads to the operator: in pixels they can measure on the frame.
 *
 * A pattern is stated as the square it covers, because that is the box drawn on
 * the canvas; a search area as the distance the point may travel from where it
 * was, because that is the question being answered ("does it move more than
 * this between frames?"). A pyramid depth is neither, so it is stated as
 * itself.
 */
function describeSetting(setting: TrackerSetting, value: number): string {
  if (setting.unit === "count") return `${value} levels`;
  if (setting.role === "search") return `±${radiusOf(setting, value)}px`;
  const side = setting.unit === "side" ? value : value * 2 + 1;
  return `${side}px square`;
}
