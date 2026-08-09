// The frame transport: one WebSocket per project, carrying pictures and the
// geometry drawn over them (stable/server/routes/ws.py, `frame_stream`).
//
// This replaces `hooks/useFrameImages.ts`, whose two hard-won lessons are kept:
//
//  1. **Never blank the canvas.** Asking for frame N and painting immediately
//     gave a black rectangle until N arrived. The previous view stays up until
//     the next one is complete, so the picture only ever changes to another
//     picture — a few milliseconds of a slightly stale frame is invisible where
//     a black flash is not.
//  2. **Prefetch in the direction of travel**, so stepping or dragging along the
//     timeline usually hits something already decoded.
//
// and one defect is fixed. The old hook returned a *possibly stale* image while
// its caller drew markers, output rectangles and target crosshairs for the
// frame the playhead was on — so during a scrub the overlay described a frame
// the picture was not showing. Here a "view" is atomic: every requested picture
// of one frame, plus that frame's export geometry, or nothing at all. The
// server helps by putting the geometry in the same message as the pixels, which
// is a guarantee two HTTP responses cannot make.
//
// Falls back to plain GETs whenever the socket is not open, so a failed socket
// degrades to the previous behaviour rather than to an empty canvas.

import { frameUrl, stabilizedFrameUrl } from "./client";

/** The two pictures of a frame: the proxy as shot, and what the export writes. */
export type FrameVariant = "source" | "output";

const ALL_VARIANTS: FrameVariant[] = ["source", "output"];

/** The similarity the render applies to a frame — mirrors `ExportContext.Affine`. */
export interface StreamWarp {
  tx: number;
  ty: number;
  rotation: number;
  scale: number;
}

/**
 * The per-frame half of the export preview, delivered with the pixels.
 *
 * Everything here is true of *this frame only*, which is why it travels with
 * the frame rather than being fetched alongside it.
 */
export interface FrameGeometry {
  frame: number;
  frames: number;
  source: { width: number; height: number };
  output: { width: number; height: number };
  /** The output rectangle's four corners, in **source** pixels. */
  rect: [number, number][];
  /** The stabilisation warp alone, before the crop/zoom fit. */
  warp: StreamWarp;
  /**
   * Source pixels → **output** pixels: the same warp with the crop/zoom fit
   * composed in, i.e. what actually produced the picture on the output canvas.
   * `rect` is its inverse applied to the output corners.
   *
   * This is the one marks are drawn with: a mark is a fraction of the output
   * frame, so it is at `u × output.width` here and at this transform's inverse
   * of that on the source canvas.
   */
  output_warp: StreamWarp;
  uses_rotation_and_scale: boolean;
  /** One row per mark: where it was asked for, where it landed, the miss. */
  targets: {
    point_id: string | null;
    target: [number, number];
    landed: [number, number] | null;
    residual: number | null;
  }[];
}

/**
 * A decoded picture, reference-counted, because more than one thing owns it.
 *
 * An `ImageBitmap` has to be closed or its pixels are pinned for the life of the
 * page — and closing one that is still going to be drawn throws
 * `InvalidStateError: the image source is detached`, out of a `useEffect`, which
 * React answers by unmounting the tree. That is a blank application, and it is
 * the bug this counter exists to make unreachable.
 *
 * Three things own a picture and none of them knows about the others: the
 * variant cache, every part-assembled frame that has picked it up, and the view
 * currently published (i.e. on a canvas). Guarding only the last of those — by
 * scanning the published view before closing, and deferring what it found — was
 * what shipped, and it misses the middle one entirely: `show()` copies cached
 * pictures straight into an `Assembly`, and an eviction or an
 * `invalidateOutput()` underneath it would close a bitmap that frame was still
 * going to be drawn from. Counting references states the ownership instead of
 * approximating it, and there is no case left to miss.
 */
export interface Decoded {
  image: CanvasImageSource;
  width: number;
  height: number;
  /** Take a reference. Every owner takes one; none of them may skip it. */
  retain: () => void;
  /** Give one back. The pixels are freed when the last owner does. */
  release: () => void;
}

/** Wrap a decoded picture in a reference count, held once by its creator. */
function counted(
  image: CanvasImageSource,
  width: number,
  height: number,
  dispose: () => void,
): Decoded {
  let refs = 1;
  return {
    image,
    width,
    height,
    retain: () => {
      refs += 1;
    },
    release: () => {
      refs -= 1;
      if (refs === 0) dispose();
    },
  };
}

/**
 * One frame, complete: every picture that was asked for, and its geometry.
 *
 * A view is only ever published whole. That is the point — the canvas draws
 * from one of these, so the overlay it paints and the pixels underneath are
 * always the same frame.
 */
export interface FrameView {
  frame: number;
  images: Partial<Record<FrameVariant, Decoded>>;
  geometry: FrameGeometry | null;
  /** Which variants could not be produced, and why (e.g. "not solved yet"). */
  errors: Partial<Record<FrameVariant, string>>;
  /** Bumped on every publish, for canvas redraw effects to depend on. */
  revision: number;
}

/**
 * Decoded frames held per variant.
 *
 * Much smaller than the 300 the old URL cache held, and deliberately so: an
 * `ImageBitmap` pins its decoded pixels (a half-res 1080p frame is ~2 MB of
 * RGBA), where the `HTMLImageElement`s it replaces let the browser drop decoded
 * data and re-decode from the retained JPEG. Sixty is comfortably more than the
 * ±12 prefetch window, so scrubbing still lands in the cache, at ~120 MB rather
 * than ~600 MB.
 */
const CACHE_LIMIT: Record<FrameVariant, number> = { source: 60, output: 30 };

/**
 * How far ahead/behind to prefetch, per variant, in the direction of travel.
 *
 * `output` is warped and JPEG-encoded per frame on the server, so it is warmed
 * narrowly: a wide guess there spends real work on frames a scrub will pass
 * without ever drawing.
 */
const PREFETCH_AHEAD: Record<FrameVariant, number> = { source: 12, output: 3 };
const PREFETCH_BEHIND: Record<FrameVariant, number> = { source: 4, output: 1 };

const RECONNECT_MS = 1000;

interface VariantCache {
  /** The id of the inputs these pictures were produced from; see `plan_revision`. */
  rev: string;
  images: Map<number, Decoded>;
}

/** One frame being put together from the replies to one request for it. */
interface Assembly {
  frame: number;
  /** Request order. A view is only ever replaced by one asked for later. */
  seq: number;
  variants: FrameVariant[];
  images: Partial<Record<FrameVariant, Decoded>>;
  errors: Partial<Record<FrameVariant, string>>;
  geometry: FrameGeometry | null;
  /** False until the server has answered about geometry — possibly with null. */
  geometryResolved: boolean;
}

/**
 * How many part-assembled frames to keep.
 *
 * The server answers only the newest want, so a superseded one never completes
 * and would otherwise linger for the life of the socket. Small: this is a
 * window on "what might still land", not a history.
 */
const ASSEMBLY_LIMIT = 4;

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`could not load ${url}`));
    image.src = url;
  });
}

async function decode(bytes: Uint8Array): Promise<Decoded> {
  const blob = new Blob([bytes as BlobPart], { type: "image/jpeg" });
  if (typeof createImageBitmap === "function") {
    const bitmap = await createImageBitmap(blob);
    return counted(bitmap, bitmap.width, bitmap.height, () => bitmap.close());
  }
  // No `createImageBitmap` (old Safari): a blob URL costs a main-thread decode,
  // which is exactly what the bitmap path avoids at playback rate.
  const url = URL.createObjectURL(blob);
  const image = await loadImage(url);
  return counted(image, image.naturalWidth, image.naturalHeight, () =>
    URL.revokeObjectURL(url),
  );
}

/**
 * Frames for one project, over one socket, with an HTTP fallback.
 *
 * Deliberately not a hook: the socket, the caches and the in-flight bookkeeping
 * outlive any single render, and `FrameStreamContext` is the thin React layer
 * over this.
 */
export class FrameStream {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;
  private seq = 0;
  // Whether anything on screen actually draws the export geometry. Computing it
  // costs the server a plan lookup and an options read per frame, which is pure
  // waste while the export tab is closed and nothing would draw the result.
  private geometryWanted = false;
  private revision = 0;
  private previousFrame = 0;

  private caches: Record<FrameVariant, VariantCache> = {
    source: { rev: "", images: new Map() },
    output: { rev: "", images: new Map() },
  };

  // Geometry is cached per frame under the *plan's* stamp, not the proxy tier's:
  // it moves when the smoothing or the render options move and the proxy frames
  // do not. Keying per frame is what makes reusing it safe — the failure this
  // class exists to prevent is drawing one frame's geometry over another's
  // pixels, and a per-frame key cannot do that.
  private geometryCache: { rev: string; frames: Map<number, FrameGeometry> } = {
    rev: "",
    frames: new Map(),
  };

  /**
   * Frames part-way through arriving, newest last.
   *
   * More than one, and that is the point. A single slot meant a reply that
   * landed after the playhead had moved on was thrown away — which is fine when
   * replies are fast and catastrophic when they are not: during playback of a
   * big clip on a busy server *every* reply lands late, so nothing was ever
   * complete, nothing was published, and the picture froze until pause let one
   * request finally win. Now each requested frame assembles on its own and any
   * that completes is shown, so playback degrades to a lower frame rate instead
   * of to a still image.
   */
  private assemblies = new Map<number, Assembly>();

  /** The frame the playhead is on: the one worth having, if it can be had. */
  private wantedFrame: number | null = null;
  private wantedVariants: FrameVariant[] = [];

  /** The last view handed out. Its pictures are on a canvas, so it holds refs. */
  private published: FrameView | null = null;
  /** Request order of what is on screen, so a late reply cannot rewind it. */
  private publishedSeq = -1;

  constructor(
    private readonly projectId: string,
    private readonly onView: (view: FrameView) => void,
    private readonly onStreamingChange: (streaming: boolean) => void,
  ) {
    this.connect();
  }

  /** Whether the socket is carrying frames (false means the HTTP fallback). */
  get streaming(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  close() {
    this.closed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.socket?.close();
    // Every owner gives its references back, in the order they were taken: the
    // assemblies that may never complete, the view on screen, then the caches.
    for (const assembly of this.assemblies.values()) this.dropAssembly(assembly);
    this.assemblies.clear();
    this.dropPublished();
    for (const variant of ALL_VARIANTS) this.clear(variant);
  }

  // -- the socket ---------------------------------------------------------------

  private connect() {
    if (this.closed) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
      `${protocol}//${window.location.host}/ws/projects/${this.projectId}/frames`,
    );
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.onopen = () => {
      this.onStreamingChange(true);
      // Re-ask for whatever was outstanding when the socket dropped, so a
      // reconnect does not leave the canvas waiting on a request nobody heard.
      if (this.wantedFrame !== null) this.show(this.wantedFrame, this.wantedVariants, true);
    };
    socket.onmessage = (event) => void this.receive(event);
    socket.onclose = () => {
      this.onStreamingChange(false);
      if (this.closed) return;
      this.reconnectTimer = setTimeout(() => this.connect(), RECONNECT_MS);
    };
    socket.onerror = () => socket.close();
  }

  private send(frame: number, variants: FrameVariant[], seq: number): boolean {
    if (!this.streaming || variants.length === 0) return false;
    this.socket!.send(
      JSON.stringify({ type: "want", seq, frame, variants, geometry: this.geometryWanted }),
    );
    return true;
  }

  /** Say whether the export geometry is being drawn, and re-ask if that changed. */
  setGeometryWanted(wanted: boolean) {
    if (this.geometryWanted === wanted) return;
    this.geometryWanted = wanted;
    if (this.wantedFrame !== null) this.show(this.wantedFrame, this.wantedVariants, true);
  }

  // -- ownership ----------------------------------------------------------------
  //
  // Each of the three owners takes a reference when it picks a picture up and
  // gives it back when it lets go. Nothing here asks whether a picture is in
  // use elsewhere, because nothing here can know — that is the counter's job.

  /** Give back the references one part-assembled frame is holding. */
  private dropAssembly(assembly: Assembly) {
    for (const variant of ALL_VARIANTS) assembly.images[variant]?.release();
    assembly.images = {};
  }

  /** Give back the references the view on screen is holding. */
  private dropPublished() {
    const shown = this.published;
    this.published = null;
    if (!shown) return;
    for (const variant of ALL_VARIANTS) shown.images[variant]?.release();
  }

  // -- the caches ---------------------------------------------------------------

  /** Drop every decoded picture of one variant. */
  private clear(variant: FrameVariant) {
    for (const decoded of this.caches[variant].images.values()) decoded.release();
    this.caches[variant].images.clear();
  }

  /** Take a picture into the cache. The caller hands over its own reference. */
  private store(variant: FrameVariant, frame: number, rev: string, decoded: Decoded) {
    const cache = this.caches[variant];
    if (cache.rev !== rev) {
      // Everything under the old stamp was produced from settings that no
      // longer apply. This is the whole invalidation story: no explicit purge
      // protocol, no stale warp left on screen. Anything an assembly or the
      // published view is still holding survives this, by its own reference.
      this.clear(variant);
      cache.rev = rev;
    }
    cache.images.get(frame)?.release();
    cache.images.set(frame, decoded);
    if (cache.images.size > CACHE_LIMIT[variant]) {
      const oldest = cache.images.keys().next().value;
      if (oldest !== undefined) {
        cache.images.get(oldest)?.release();
        cache.images.delete(oldest);
      }
    }
  }

  /**
   * Throw away rendered output frames and the geometry that goes with them.
   *
   * The stamp on each message catches staleness anyway — but only once a fresh
   * frame arrives. Calling this when the render options change means the stale
   * picture is not what the operator looks at in the meantime.
   */
  invalidateOutput() {
    this.clear("output");
    this.caches.output.rev = "";
    this.geometryCache = { rev: "", frames: new Map() };
    if (this.wantedFrame !== null) this.show(this.wantedFrame, this.wantedVariants, true);
  }

  // -- receiving ----------------------------------------------------------------

  /**
   * Take the geometry off a reply, if it carried any.
   *
   * A want's reply always has the key - even when the value is null because the
   * project is not solved, and even when the reply is a failure - so a frame
   * being assembled always learns where it stands. A *prefetch* reply has no
   * key at all, which is what stops a guess about frame 260 from being taken
   * for an answer about the frame on screen.
   */
  private absorbGeometry(header: Record<string, unknown>, assembly: Assembly) {
    if (!("geometry" in header)) return;
    const geometry = (header.geometry as FrameGeometry | null) ?? null;
    const rev = String(header.geometry_rev ?? "");
    if (this.geometryCache.rev !== rev) this.geometryCache = { rev, frames: new Map() };
    if (geometry) this.geometryCache.frames.set(assembly.frame, geometry);
    assembly.geometry = geometry;
    assembly.geometryResolved = true;
  }

  private async receive(event: MessageEvent) {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message?.type !== "error") return;
      const failed = this.assemblies.get(message.frame);
      if (!failed) return;
      const variant = message.variant as FrameVariant;
      if (failed.variants.includes(variant)) {
        failed.errors[variant] = String(message.detail ?? "frame not available");
      }
      this.absorbGeometry(message, failed);
      this.publishIfComplete(failed);
      return;
    }

    const buffer = event.data as ArrayBuffer;
    const headerLength = new DataView(buffer).getUint32(0, false);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    const decoded = await decode(new Uint8Array(buffer, 4 + headerLength));
    const variant = header.variant as FrameVariant;
    // The cache takes the reference `decode` created; anyone else who wants this
    // picture takes one of their own below.
    this.store(variant, header.frame, String(header.rev ?? ""), decoded);

    // Looked up after the decode, because the playhead may have moved while it
    // ran. No assembly means a prefetch, or a frame requested so long ago it has
    // been pruned: cached above, drawn never.
    const assembly = this.assemblies.get(header.frame);
    if (!assembly) return;
    this.absorbGeometry(header, assembly);
    if (assembly.variants.includes(variant)) this.adopt(assembly, variant, decoded);
    this.publishIfComplete(assembly);
  }

  /** Put a picture into an assembly, taking a reference for it. */
  private adopt(assembly: Assembly, variant: FrameVariant, decoded: Decoded) {
    if (assembly.images[variant] === decoded) return;
    decoded.retain();
    assembly.images[variant]?.release();
    assembly.images[variant] = decoded;
  }

  /**
   * Show a frame once every picture of it that was asked for has settled.
   *
   * The frame need not be the one under the playhead, and under load it usually
   * is not: the server answers the newest want it can get to, and by then the
   * playhead has moved again. Showing that reply anyway is what makes playback
   * degrade to a lower frame rate rather than to a frozen picture - which is
   * what a single in-flight slot produced, because every reply arrived stale and
   * every one was discarded. It costs nothing in correctness: a view is still
   * one frame's pixels with that same frame's geometry, and the viewer says
   * which frame it is showing.
   *
   * `seq` is what keeps it honest - a view is only ever replaced by one asked
   * for later, so a straggler cannot rewind the picture.
   */
  private publishIfComplete(assembly: Assembly) {
    if (assembly.seq <= this.publishedSeq) return;
    const settled = assembly.variants.filter(
      (variant) => assembly.images[variant] || assembly.errors[variant],
    );
    if (settled.length < assembly.variants.length) return;
    // A view without its geometry is exactly the half-drawn state this replaces:
    // the picture would go up and the output rectangle would follow a frame
    // later. Wait for the whole thing.
    if (!assembly.geometryResolved) return;
    // Retained before the outgoing view lets go, not after: the two views often
    // share a picture (a scrub that only changes the `output` variant keeps the
    // same `source`), and releasing first would take the count to zero and free
    // a bitmap that is about to be drawn again.
    const images: Partial<Record<FrameVariant, Decoded>> = {};
    for (const variant of ALL_VARIANTS) {
      const picture = assembly.images[variant];
      if (!picture) continue;
      picture.retain();
      images[variant] = picture;
    }
    this.dropPublished();
    this.published = {
      frame: assembly.frame,
      images,
      geometry: assembly.geometry,
      errors: { ...assembly.errors },
      revision: ++this.revision,
    };
    this.publishedSeq = assembly.seq;
    // Anything asked for earlier can no longer reach the screen — including this
    // one, whose pictures the view above now holds its own references to.
    for (const [frame, other] of this.assemblies) {
      if (other.seq > this.publishedSeq) continue;
      this.dropAssembly(other);
      this.assemblies.delete(frame);
    }
    this.onView(this.published);
  }

  // -- requesting ---------------------------------------------------------------

  /**
   * Ask for one frame in the given variants, and publish it once it is whole.
   *
   * Cached pictures count as arrived, so scrubbing back over ground already
   * covered publishes without asking for anything.
   */
  show(frame: number, variants: FrameVariant[], force = false) {
    if (
      !force &&
      this.wantedFrame === frame &&
      this.wantedVariants.length === variants.length &&
      this.wantedVariants.every((variant) => variants.includes(variant))
    ) {
      return;
    }
    const images: Partial<Record<FrameVariant, Decoded>> = {};
    const missing: FrameVariant[] = [];
    for (const variant of variants) {
      const cached = this.caches[variant].images.get(frame);
      // A reference of this frame's own: the cache may evict or clear it before
      // the frame is whole, and that must not free pixels this view will draw.
      if (cached) {
        cached.retain();
        images[variant] = cached;
      } else missing.push(variant);
    }
    const geometry = this.geometryCache.frames.get(frame) ?? null;
    const assembly: Assembly = {
      frame,
      seq: ++this.seq,
      variants: [...variants],
      images,
      errors: {},
      geometry,
      // Nothing on the HTTP path carries geometry, so waiting for it there would
      // wait forever; `ExportContext` notices and asks `/export/preview` itself.
      // Nothing is waited for either when nothing draws it.
      geometryResolved: geometry !== null || !this.streaming || !this.geometryWanted,
    };
    this.wantedFrame = frame;
    this.wantedVariants = assembly.variants;
    // Deleted before it is set, so re-showing a frame moves it to the end of the
    // map. Map order is insertion order, and the eviction below takes the front.
    const superseded = this.assemblies.get(frame);
    if (superseded) {
      this.dropAssembly(superseded);
      this.assemblies.delete(frame);
    }
    this.assemblies.set(frame, assembly);
    while (this.assemblies.size > ASSEMBLY_LIMIT) {
      const oldest = this.assemblies.keys().next().value;
      if (oldest === undefined) break;
      const evicted = this.assemblies.get(oldest);
      if (evicted) this.dropAssembly(evicted);
      this.assemblies.delete(oldest);
    }

    // Only what is actually missing goes on the wire. Re-asking for an `output`
    // frame already in hand would spend a warp to learn nothing; asking for the
    // cheap `source` variant alone is how a frame whose pictures are all cached
    // still gets its geometry.
    let ask = missing;
    if (!ask.length && !assembly.geometryResolved) ask = ["source"];
    if (!ask.length) {
      this.publishIfComplete(assembly);
      return;
    }
    if (!this.send(frame, ask, assembly.seq)) void this.fetchOverHttp(frame, missing);
  }

  /** Warm the frames the playhead is heading towards. */
  prefetch(frame: number, variants: FrameVariant[], frameCount: number) {
    const backwards = frame < this.previousFrame;
    this.previousFrame = frame;
    if (!this.streaming || frameCount <= 1) return;
    const step = backwards ? -1 : 1;
    for (const variant of variants) {
      const wanted: number[] = [];
      for (let offset = 1; offset <= PREFETCH_AHEAD[variant]; offset += 1) {
        wanted.push(frame + step * offset);
      }
      for (let offset = 1; offset <= PREFETCH_BEHIND[variant]; offset += 1) {
        wanted.push(frame - step * offset);
      }
      const missing = wanted.filter(
        (index) => index >= 0 && index < frameCount && !this.caches[variant].images.has(index),
      );
      if (missing.length) {
        this.socket!.send(
          JSON.stringify({ type: "prefetch", frames: missing, variants: [variant] }),
        );
      }
    }
  }

  /**
   * The fallback: the same GETs the viewer used to make, when the socket is down.
   *
   * There is no geometry on this path, so `ExportContext` goes back to asking
   * `/export/preview` per frame — which is what it did before this round.
   * Degraded, not broken.
   */
  private async fetchOverHttp(frame: number, variants: FrameVariant[]) {
    const requested = this.assemblies.get(frame);
    if (!requested) return;
    await Promise.all(
      variants.map(async (variant) => {
        const url =
          variant === "source"
            ? frameUrl(this.projectId, frame)
            : stabilizedFrameUrl(this.projectId, frame);
        let decoded: Decoded | null = null;
        try {
          const image = await loadImage(url);
          // An `HTMLImageElement` has nothing to close — the browser owns its
          // decoded pixels — but it is counted like everything else so this path
          // and the socket's have one set of rules between them.
          decoded = counted(image, image.naturalWidth, image.naturalHeight, () => undefined);
        } catch {
          // fall through: the assembly is told the frame is unavailable
        }
        // Re-read: the load took time, and the assembly may have been superseded
        // and its references given back while it did.
        const assembly = this.assemblies.get(frame);
        if (assembly !== requested || !assembly) {
          decoded?.release();
          return;
        }
        if (decoded) {
          this.store(variant, frame, "http", decoded);
          this.adopt(assembly, variant, decoded);
        } else {
          assembly.errors[variant] = "frame not available";
        }
        this.publishIfComplete(assembly);
      }),
    );
  }
}
