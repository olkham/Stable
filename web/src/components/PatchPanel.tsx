import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import { useFrameStream } from "../state/FrameStreamContext";
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
 * it still look like what it saw at the anchor?". A template that is a crisp
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
  const { project, currentFrame, setCurrentFrame, editRevision, refresh, afterEdit } =
    useProjectContext();
  const [info, setInfo] = useState<PointPatches | null>(null);
  // The anchor image the operator has clicked, so `Delete` has something to
  // act on (todo 9, fourth round). Null once it is gone or the point changes.
  const [selectedAnchor, setSelectedAnchor] = useState<number | null>(null);
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
  useEffect(() => setSelectedAnchor(null), [pointId]);

  /**
   * Remove one anchor, from its own picture.
   *
   * Goes through `afterEdit`, not a bare refresh: the anchor list this panel
   * and the timeline draw comes with the *track*, which is cached separately
   * from the project, so refreshing only the project would leave a deleted
   * anchor on screen until the next tracking run replaced that cache.
   */
  const deleteAnchor = useCallback(
    async (frame: number) => {
      try {
        await api.deleteAnchor(project.id, pointId, frame);
        setSelectedAnchor(null);
        await afterEdit(pointId);
        await reload();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [project.id, pointId, afterEdit, reload],
  );

  // Delete/Backspace removes the selected anchor's image. Ignored while typing
  // in a field, the same guard the rest of the app's shortcuts use.
  useEffect(() => {
    if (selectedAnchor === null) return;
    function onKey(event: KeyboardEvent) {
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      event.preventDefault();
      void deleteAnchor(selectedAnchor as number);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedAnchor, deleteAnchor]);

  if (error) return <div className="error-text">{error}</div>;
  if (!info) return <div className="patch-panel">Loading patches…</div>;

  const settings = info.settings;
  /**
   * Whether this point's tracker matches a *region* at all (todo 5, sixth round).
   *
   * `radius_source` is the constructor argument the window came from, and the
   * server sends null for the trackers that have none — the deep ones, which see
   * whole frames. For those there is no patch: the radius below is a preview
   * size this panel invented, and a grid of thumbnails of it is a picture of
   * something the algorithm never looks at. The panel then says so and shows
   * nothing else — the anchors are listed on the card above, and listing them
   * again here was the second of the two lists this round exists to remove.
   */
  const hasWindow = info.radius_source !== null;
  // A point must keep one anchor to track from, so the last one's × is refused
  // here rather than left to fail on the server.
  const anchorCount = info.patches.filter((p) => p.kind === "anchor").length;
  const windowSetting = settings.find((s) => s.role === "window");
  const value = (setting: TrackerSetting) => draft[setting.attribute] ?? setting.value;
  const dirty = settings.some((s) => value(s) !== s.value);
  // What the thumbnails should show: the pattern size being previewed, which is
  // the pending window value while the slider is being dragged.
  const previewRadius =
    (windowSetting ? radiusOf(windowSetting, value(windowSetting)) : 0) || info.radius;

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

  /** Why an anchor's × is unavailable, or null when it can be deleted. */
  function undeletableReason(patch: PointPatches["patches"][number]): string | null {
    if (patch.kind !== "anchor") return "Only anchors can be deleted";
    if (anchorCount <= 1) {
      return "A track needs at least one anchor to track from — this is its last one";
    }
    return null;
  }

  function thumb(patch: PointPatches["patches"][number], key: string) {
    const isCurrent = patch.frame === currentFrame;
    const isAnchor = patch.kind === "anchor";
    const isSelected = isAnchor && patch.frame === selectedAnchor;
    const blocked = undeletableReason(patch);
    return (
      <figure
        key={key}
        className={`patch-thumb ${isCurrent ? "patch-thumb--current" : ""} ${
          isSelected ? "patch-thumb--selected" : ""
        } patch-thumb--${patch.kind}`}
        onClick={() => {
          setCurrentFrame(patch.frame);
          // Clicking an anchor's image also *selects* it, which is what gives
          // `Delete` something to remove (todo 9).
          if (isAnchor) setSelectedAnchor(patch.frame);
        }}
        title={
          isAnchor
            ? `Anchor at frame ${patch.frame} — the position you set. Click to jump there and select it; Delete removes it.`
            : `Frame ${patch.frame} — where the tracker put it. Click to jump there.`
        }
      >
        {isAnchor && (
          <button
            className="patch-thumb__delete"
            disabled={blocked !== null}
            title={blocked ?? `Delete the anchor at frame ${patch.frame}`}
            onClick={(e) => {
              e.stopPropagation();
              if (!blocked) void deleteAnchor(patch.frame);
            }}
          >
            ×
          </button>
        )}
        {patch.available ? (
          <img
            src={api.pointPatchUrl(
              project.id,
              pointId,
              patch.frame,
              previewRadius,
              // This patch's own centre, so its URL changes when — and only
              // when — the pixels it shows do. It used to be the project's edit
              // counter, which every correction bumped, so correcting one point
              // on one frame re-fetched the *whole* grid; at `original` that is
              // a full-resolution seek per thumbnail, twenty deep-clip seeks for
              // one drag, and it is what made editing feel laggy. An anchor does
              // not move when a correction is made elsewhere, so now it does not
              // reload either.
              `${patch.x.toFixed(1)}x${patch.y.toFixed(1)}`,
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
          {patch.kind === "anchor" ? "◆" : ""}f{patch.frame}
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

        {settings.length === 0 && hasWindow && (
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

      {hasWindow && info.current && (
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

      {/* One list of anchors per card, and it is the one on the card itself
          (todo 6, seventh round). What this panel adds is the *pixels* at each
          of them, so it draws them when there are pixels to draw and says why
          not when there are not — it no longer repeats the row above as chips. */}
      {!hasWindow ? (
        <p className="patch-panel__note">{patchSizeExplanation(info)}</p>
      ) : info.patches.length === 0 ? (
        <p className="patch-panel__empty">
          No anchors yet — place one to see the patch it would track from.
        </p>
      ) : (
        <>
          <span className="patch-panel__label">
            Anchors — click one, then Delete, or use its ×
          </span>
          <div className="patch-panel__grid">
            {info.patches.map((patch) => thumb(patch, `k${patch.frame}`))}
          </div>
        </>
      )}

      {hasWindow && info.patches.some((p) => !p.available) && (
        <p className="patch-panel__warning">
          ✕ marks an anchor too close to the frame edge for this size — the tracker would refuse
          it too. Reduce the size or move the anchor inward.
        </p>
      )}
    </div>
  );
}

/** Size of a thumbnail in the all-points strip — smaller, since there are many. */
const STRIP_PX = 48;

/**
 * Every tracked point's patch on the frame being shown, side by side.
 *
 * The per-point panel above answers "is *this* point still on target"; this
 * answers "which of my points has drifted", which is the question you actually
 * have on a shot with eight of them. One row, click to select.
 *
 * **Cropped from the frame already on the canvas, not fetched.** Each thumbnail
 * used to be an `<img>` whose `src` carried the frame number, so the strip fired
 * one server-side crop *per point per frame*: four points at 24 fps is ~100
 * requests a second, each a decode and a re-encode, and it was enough on its own
 * to starve the frame stream and stall playback (todo 2, fifth round). Every
 * pixel it needs is already here — the proxy frame is decoded on the canvas, the
 * point's position comes with its track, and the pattern radius is on the point
 * itself — so the crop is a `drawImage` and the strip costs nothing at all.
 *
 * It also draws `viewFrame`, so it shows the same frame as the canvas rather
 * than one the playhead has moved past.
 */
export function AllPatchesStrip() {
  const {
    project,
    selectedPointId,
    setSelectedPointId,
    tracks,
    loadTrack,
    stabilizedPreview,
    colorFor,
  } =
    useProjectContext();
  const { view, viewFrame } = useFrameStream();
  const picture = view?.images.source ?? null;

  // Only the points whose tracker actually matches a region (todo 5, sixth
  // round). `window.attribute` is null for the trackers that see whole frames,
  // and a thumbnail of a patch such a tracker never looks at answers "which one
  // has drifted" with a picture that has nothing to do with why it drifted.
  //
  // A composite is the same case and more so (todo 2, tenth round, second
  // pass): no tracker matched anything for it, and the pixels under the
  // midpoint of two features are usually a picture of neither.
  const tracked = project.points.filter(
    (point) =>
      (point.has_track || point.queries.length > 0) &&
      point.window?.attribute !== null &&
      point.derived_from.length === 0,
  );

  // The strip needs each point's path to know where to crop.
  useEffect(() => {
    for (const point of tracked) if (point.has_track && !(point.id in tracks)) void loadTrack(point.id);
  }, [tracked, tracks, loadTrack]);

  if (tracked.length === 0) return null;
  // The proxy is a scaled copy of the clip, and radii and track positions are in
  // source pixels, so everything below is converted once here.
  const proxy = project.proxy["half"];
  const scale = proxy?.width && project.video.w ? proxy.width / project.video.w : 1;

  return (
    <div className="patch-strip">
      <span className="patch-panel__label">
        Tracking on frame {viewFrame}
        {stabilizedPreview && " — turn the stabilized preview off to see the tracked patches"}
      </span>
      <div className="patch-strip__row">
        {tracked.map((point) => (
          <figure
            key={point.id}
            className={`patch-thumb ${
              point.id === selectedPointId ? "patch-thumb--current" : ""
            }`}
            onClick={() => setSelectedPointId(point.id)}
            title={`${point.id} (${point.tracker}) — click to select`}
          >
            <PatchThumb
              picture={picture}
              x={tracks[point.id]?.x[viewFrame] ?? null}
              y={tracks[point.id]?.y[viewFrame] ?? null}
              radius={point.window?.radius || DEFAULT_PREVIEW_RADIUS}
              scale={scale}
            />
            <figcaption style={{ color: colorFor(point.id) }}>{point.id}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

/** Fallback pattern radius, matching the server's for a tracker with no window. */
const DEFAULT_PREVIEW_RADIUS = 16;

/**
 * One point's patch, cut out of the decoded frame.
 *
 * Drawn unsmoothed for the same reason the fetched thumbnails were drawn
 * `pixelated`: at 48 px, smoothing hides exactly the texture being judged.
 */
function PatchThumb({
  picture,
  x,
  y,
  radius,
  scale,
}: {
  picture: { image: CanvasImageSource; width: number; height: number } | null;
  x: number | null;
  y: number | null;
  radius: number;
  scale: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, STRIP_PX, STRIP_PX);
    if (!picture || x == null || y == null) {
      // No position on this frame is a real answer — "the tracker has nothing
      // here" — so it is drawn as an empty cell rather than a missing one.
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(0, 0, STRIP_PX, STRIP_PX);
      return;
    }
    const half = Math.max(1, (radius + 0.5) * scale);
    ctx.imageSmoothingEnabled = false;
    try {
      ctx.drawImage(
        picture.image,
        x * scale - half,
        y * scale - half,
        half * 2,
        half * 2,
        0,
        0,
        STRIP_PX,
        STRIP_PX,
      );
    } catch {
      // The frame's pixels are gone (see `drawPicture` in Viewer.tsx): a thumbnail
      // that cannot be cropped is an empty cell, never an unmounted application.
      ctx.fillStyle = "rgba(255,255,255,0.05)";
      ctx.fillRect(0, 0, STRIP_PX, STRIP_PX);
    }
  }, [picture, x, y, radius, scale]);

  return <canvas ref={canvasRef} width={STRIP_PX} height={STRIP_PX} />;
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
