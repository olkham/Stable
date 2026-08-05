import { useCallback, useEffect, useRef, useState } from "react";

/** How many frames either side of the current one to fetch before they are asked for. */
const PREFETCH_AHEAD = 12;
const PREFETCH_BEHIND = 4;

/** Decoded frames held at once. A half-res JPEG is ~50 KB, so this is a few MB. */
const CACHE_LIMIT = 300;

/** Browsers cap concurrent connections per origin; going wider just queues. */
const MAX_IN_FLIGHT = 6;

export interface FrameImages {
  /** The image to draw: the current frame, or the last one that loaded. */
  image: HTMLImageElement | null;
  /** Whether `image` is actually the frame that was asked for. */
  isCurrent: boolean;
  /** Set when the current frame could not be loaded at all. */
  error: boolean;
  /** Bumped whenever the answer changes, for canvas redraw effects to depend on. */
  revision: number;
}

/**
 * Proxy frames for a scrubbing viewer, without the black flash (todo 5).
 *
 * Two separate causes were making navigation flicker, and both are handled
 * here rather than at the call site:
 *
 * 1. **Nothing to draw yet.** Asking for frame N and painting immediately gave
 *    a blank rectangle until N arrived. The previous frame is kept and returned
 *    instead (`isCurrent: false` says so), so the picture only ever changes to
 *    another picture. On a scrub that reads as a slightly stale image for a few
 *    milliseconds, which is invisible, where a black flash is not.
 * 2. **Every frame was a cold fetch.** Neighbours are prefetched in the
 *    direction of travel, so stepping or dragging along the timeline usually
 *    hits an image that is already decoded.
 *
 * The cache is keyed by URL, so the stabilized preview and the raw proxy
 * coexist without invalidating each other, and it is capped in count with plain
 * insertion-order eviction (a long scrub session must not grow without bound).
 *
 * @param urlFor Builds the URL for a frame, or returns null when the proxy is
 *   not ready yet. Must be referentially stable across renders (`useCallback`).
 * @param frame The frame currently being displayed.
 * @param frameCount Total frames in the clip, bounding the prefetch.
 */
export function useFrameImages(
  urlFor: (frame: number) => string | null,
  frame: number,
  frameCount: number,
): FrameImages {
  const cache = useRef(new Map<string, HTMLImageElement>());
  const inFlight = useRef(new Map<string, HTMLImageElement>());
  const lastDrawn = useRef<HTMLImageElement | null>(null);
  const previousFrame = useRef(frame);
  const [revision, bump] = useState(0);
  const [error, setError] = useState(false);

  const load = useCallback(
    (url: string, onDone?: () => void): HTMLImageElement | null => {
      const cached = cache.current.get(url);
      if (cached) return cached;
      if (inFlight.current.has(url)) return null;
      if (inFlight.current.size >= MAX_IN_FLIGHT && onDone === undefined) return null;

      const img = new Image();
      inFlight.current.set(url, img);
      img.onload = () => {
        inFlight.current.delete(url);
        cache.current.set(url, img);
        if (cache.current.size > CACHE_LIMIT) {
          const oldest = cache.current.keys().next().value;
          if (oldest !== undefined) cache.current.delete(oldest);
        }
        onDone?.();
      };
      img.onerror = () => {
        inFlight.current.delete(url);
        onDone?.();
      };
      img.src = url;
      return null;
    },
    [],
  );

  const url = urlFor(frame);

  // The frame being displayed: load it if needed, and redraw when it arrives.
  useEffect(() => {
    if (!url) return;
    setError(false);
    const ready = cache.current.get(url);
    if (ready) {
      lastDrawn.current = ready;
      bump((n) => n + 1);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      cache.current.set(url, img);
      if (cancelled) return;
      lastDrawn.current = img;
      bump((n) => n + 1);
    };
    img.onerror = () => {
      if (!cancelled) setError(true);
    };
    img.src = url;
    return () => {
      // Only the *subscription* is cancelled, never the request: a frame that
      // arrives after the user has scrubbed past it still belongs in the cache,
      // and will be there instantly when they scrub back.
      cancelled = true;
    };
  }, [url]);

  // Neighbours, biased in the direction of travel. Runs after the effect above
  // so the frame actually being looked at never queues behind a prefetch.
  useEffect(() => {
    if (!url || frameCount <= 1) return;
    const backwards = frame < previousFrame.current;
    previousFrame.current = frame;
    const ahead = backwards ? -PREFETCH_AHEAD : PREFETCH_AHEAD;
    const behind = backwards ? PREFETCH_BEHIND : -PREFETCH_BEHIND;

    const wanted: number[] = [];
    const step = ahead > 0 ? 1 : -1;
    for (let offset = 1; Math.abs(offset) <= Math.abs(ahead); offset += step) {
      wanted.push(frame + offset);
    }
    for (let offset = 1; Math.abs(offset) <= Math.abs(behind); offset += 1) {
      wanted.push(frame - step * offset);
    }

    const timer = setTimeout(() => {
      for (const index of wanted) {
        if (index < 0 || index >= frameCount) continue;
        const target = urlFor(index);
        if (target) load(target);
      }
    }, 0);
    return () => clearTimeout(timer);
  }, [url, frame, frameCount, urlFor, load]);

  const current = url ? (cache.current.get(url) ?? null) : null;
  return {
    image: current ?? lastDrawn.current,
    isCurrent: current !== null,
    error,
    revision,
  };
}
