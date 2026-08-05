import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import { useProjectContext } from "../state/ProjectContext";
import type { TemplateOptions } from "../types";

/** Largest a thumbnail is drawn, whatever the patch's own pixel size. */
const THUMB_PX = 72;

/**
 * Template-matching controls for one point (todo 14).
 *
 * Two things the operator could not see before: what size patch the tracker is
 * matching, and what that patch actually contains at each of their keyframes.
 * The size is a slider, and every keyframe gets a thumbnail in a grid — which
 * is the useful view, because a template that works at frame 0 and is a patch
 * of motion-blurred sky at frame 200 explains a failed track instantly.
 *
 * Thumbnails are rendered by the server from the source clip
 * (`stable/server/tracker_options.py`), and the slider previews sizes through
 * the `radius` query parameter *before* they are saved — so dragging shows the
 * consequence, and only "Apply" writes it. Patches are drawn pixelated on
 * purpose: at this size smoothing would hide exactly the texture being judged.
 */
export function TemplatePanel({ pointId }: { pointId: string }) {
  const { project, currentFrame, setCurrentFrame } = useProjectContext();
  const [options, setOptions] = useState<TemplateOptions | null>(null);
  const [radius, setRadius] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const next = await api.getTemplateOptions(project.id, pointId);
      setOptions(next);
      setRadius(next.template_radius);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [project.id, pointId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  if (error) return <div className="error-text">{error}</div>;
  if (!options || radius === null) return <div className="template-panel">Loading template…</div>;

  const [minRadius, maxRadius] = options.template_radius_range;
  const dirty = radius !== options.template_radius;

  async function apply(value: number | null) {
    setBusy(true);
    try {
      const next = await api.updateTemplateOptions(project.id, pointId, {
        template_radius: value,
      });
      setOptions(next);
      setRadius(next.template_radius);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="template-panel" onClick={(e) => e.stopPropagation()}>
      <div className="template-panel__controls">
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
        <span className="template-panel__size">
          {radius * 2 + 1}px
          {dirty && <span className="template-panel__pending"> (preview)</span>}
        </span>
        <button onClick={() => apply(radius)} disabled={busy || !dirty}>
          Apply
        </button>
        <button
          onClick={() => apply(null)}
          disabled={busy || !options.custom}
          title={`Back to the tracker's default (${options.defaults.template_radius * 2 + 1}px)`}
        >
          Reset
        </button>
      </div>

      {options.anchors.length === 0 ? (
        <p className="template-panel__empty">
          This point has no keyframes yet — place one to see its template patch.
        </p>
      ) : (
        <div className="template-panel__grid">
          {options.anchors.map((anchor) => (
            <figure
              key={anchor.frame}
              className={`template-thumb ${
                anchor.frame === currentFrame ? "template-thumb--current" : ""
              }`}
              onClick={() => setCurrentFrame(anchor.frame)}
              title={`Keyframe ${anchor.frame} — click to jump there`}
            >
              {anchor.available ? (
                <img
                  src={api.templatePatchUrl(project.id, pointId, anchor.frame, radius)}
                  width={THUMB_PX}
                  height={THUMB_PX}
                  alt={`Template patch at frame ${anchor.frame}`}
                  style={{ imageRendering: "pixelated" }}
                />
              ) : (
                <div
                  className="template-thumb__missing"
                  style={{ width: THUMB_PX, height: THUMB_PX }}
                  title="This patch would fall outside the frame at this size"
                >
                  ✕
                </div>
              )}
              <figcaption>f{anchor.frame}</figcaption>
            </figure>
          ))}
        </div>
      )}
      {options.anchors.some((a) => !a.available) && (
        <p className="template-panel__warning">
          ✕ marks a keyframe too close to the frame edge for this size — the tracker would
          refuse it too. Reduce the size or move the keyframe inward.
        </p>
      )}
    </div>
  );
}
