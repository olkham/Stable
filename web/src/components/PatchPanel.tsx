import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import { pointColor } from "../colors";
import { useProjectContext } from "../state/ProjectContext";
import type { PointPatches } from "../types";

/** Largest a thumbnail is drawn, whatever the patch's own pixel size. */
const THUMB_PX = 72;

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
 * So the panel is shown for every point. What varies by tracker is only the
 * *size* control: the template tracker takes its window from a stored setting,
 * so there the slider changes what the next run will actually match; every
 * other plugin's window comes from its own constructor (LK's `win_size`,
 * mean-shift's `window_radius`, and so on), and the panel reports the real
 * number rather than offering a control that would silently change nothing.
 *
 * Patches are rendered server-side from the source clip and drawn pixelated on
 * purpose: at this size, smoothing would hide exactly the texture being judged.
 */
export function PatchPanel({ pointId }: { pointId: string }) {
  const { project, currentFrame, setCurrentFrame, editRevision } = useProjectContext();
  const [info, setInfo] = useState<PointPatches | null>(null);
  const [radius, setRadius] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const next = await api.getPointPatches(project.id, pointId, currentFrame);
      setInfo(next);
      setRadius((current) => (current === null || !busy ? next.radius : current));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
    // `editRevision` is a dependency on purpose: a correction or a re-track
    // moves the point, so every patch below is now of different pixels.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, pointId, currentFrame, editRevision]);

  useEffect(() => {
    void reload();
  }, [reload]);

  if (error) return <div className="error-text">{error}</div>;
  if (!info || radius === null) return <div className="patch-panel">Loading patches…</div>;

  const [minRadius, maxRadius] = info.template_radius_range;
  const dirty = radius !== info.radius;
  // Cache-buster: a patch URL is (point, frame, radius), but the *pixels* also
  // change when the track moves, and that does not alter the URL.
  const rev = editRevision;

  async function apply(value: number | null) {
    setBusy(true);
    try {
      const next = await api.updatePointPatches(project.id, pointId, { template_radius: value });
      setInfo(next);
      setRadius(next.radius);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

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
            src={api.pointPatchUrl(project.id, pointId, patch.frame, radius ?? undefined, rev)}
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
      <div className="patch-panel__controls">
        {info.adjustable ? (
          <>
            <label>
              Template size
              <input
                type="range"
                min={minRadius}
                max={maxRadius}
                value={radius}
                disabled={busy}
                onChange={(e) => setRadius(Number(e.target.value))}
              />
            </label>
            <span className="patch-panel__size">
              {radius * 2 + 1}px{dirty && <span className="patch-panel__pending"> (preview)</span>}
            </span>
            <button onClick={() => apply(radius)} disabled={busy || !dirty}>
              Apply
            </button>
            <button
              onClick={() => apply(null)}
              disabled={busy || !info.custom}
              title={`Back to the tracker's default (${info.defaults.template_radius * 2 + 1}px)`}
            >
              Reset
            </button>
          </>
        ) : (
          <span className="patch-panel__size" title={patchSizeExplanation(info)}>
            {info.patch_size}px window · {info.tracker}
          </span>
        )}
      </div>

      {info.current && (
        <div className="patch-panel__now">
          <span className="patch-panel__label">This frame</span>
          {thumb(info.current, "current")}
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
    return `The region ${info.tracker} matches, from its own ${info.radius_source}. Switch this point to the template tracker to set it.`;
  }
  return `${info.tracker} has no local search window — it sees whole frames — so this is a preview size only.`;
}
