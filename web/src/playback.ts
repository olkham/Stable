// Playback speed: the rates offered, and which one a clip should open at.
//
// Shared between the timeline (which draws the picker) and the project context
// (which holds the rate), so "what speeds exist" has one definition. It used to
// live in `Timeline.tsx` alone, which is why the context could only start at 1x.

/** Playback speeds offered, as multiples of the clip's own frame rate. */
export const PLAYBACK_RATES = [0.25, 0.5, 1, 2, 4];

/**
 * The frame rate playback aims for when a project is opened.
 *
 * Review speed, not capture speed. A 120 fps clip played at 1x runs four times
 * faster than the shot did, which is useless for judging a track — so the
 * default rate is chosen for the *result*, not for the file.
 */
export const TARGET_PLAYBACK_FPS = 30;

/**
 * The offered rate that plays `fps` closest to {@link TARGET_PLAYBACK_FPS}
 * (todo 5, tenth round).
 *
 * 120 → 0.25, 60 → 0.5, 48 → 0.5, 30 / 25 / 24 → 1. Ties go to the slower rate:
 * between two speeds equally far from the target, the one that shows more of
 * every frame is the one worth defaulting to.
 *
 * A clip whose frame rate is unknown (0, or a file the decoder could not report
 * on) gets 1x — the honest answer when there is nothing to scale against.
 */
export function defaultPlaybackRate(fps: number): number {
  if (!Number.isFinite(fps) || fps <= 0) return 1;
  let best = 1;
  let bestDistance = Infinity;
  for (const rate of PLAYBACK_RATES) {
    const distance = Math.abs(fps * rate - TARGET_PLAYBACK_FPS);
    if (distance < bestDistance) {
      best = rate;
      bestDistance = distance;
    }
  }
  return best;
}
