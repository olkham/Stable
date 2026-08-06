import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { FrameStream, type FrameGeometry, type FrameVariant, type FrameView } from "../api/frameStream";
import { useProjectContext } from "./ProjectContext";

interface FrameStreamValue {
  /**
   * The frame currently on the canvas: its pictures and its export geometry,
   * always of one frame. Null only before the first one has arrived.
   */
  view: FrameView | null;
  /** The frame being *shown*, which lags the playhead while one is in flight. */
  viewFrame: number;
  /** This frame's export geometry, or null before the project is solved. */
  geometry: FrameGeometry | null;
  /** False when the socket is down and frames are coming over plain GETs. */
  streaming: boolean;
  /** What went wrong with the frame being shown, if anything. */
  error: string | null;

  /** Ask for the stabilized output picture as well — the side-by-side canvas. */
  wantOutput: boolean;
  setWantOutput: (on: boolean) => void;
  /**
   * Ask for each frame's export geometry too.
   *
   * Only the export tab draws it, and producing it costs the server a plan
   * lookup and an options read per frame — so it is asked for rather than
   * assumed.
   */
  setWantGeometry: (on: boolean) => void;
  /** Drop every rendered output frame, e.g. because the render options moved. */
  invalidateOutput: () => void;
}

const FrameStreamContext = createContext<FrameStreamValue | null>(null);

export function useFrameStream(): FrameStreamValue {
  const ctx = useContext(FrameStreamContext);
  if (!ctx) throw new Error("useFrameStream must be used inside <FrameStreamProvider>");
  return ctx;
}

/**
 * The project's frames, for everything that draws one (todo 2, fifth round).
 *
 * The viewer used to load frames itself, one `Image()` per frame per canvas,
 * while the export overlay drawn on top of them came from a separate
 * `/export/preview` request per frame. Two consequences, both of which this
 * removes:
 *
 * * **The overlay could describe a different frame than the picture.** The old
 *   hook deliberately kept the previous image on screen until the next decoded
 *   (so navigation would not flash black) and the caller drew markers, output
 *   rectangles and crosshairs for the playhead's frame regardless. Here the
 *   unit is a *view* — every requested picture of one frame plus that frame's
 *   geometry, delivered together — and the viewer draws `viewFrame`, not
 *   `currentFrame`. The two are equal whenever nothing is in flight, which is
 *   almost always.
 * * **Superseded work was still done.** A scrub asked the server for every
 *   frame it passed over, including the expensive warped ones, and the server
 *   had no way to know they no longer mattered. One socket lets it drop them.
 *
 * Sits between `ProjectProvider` (whose playhead it follows) and
 * `ExportProvider` (which reads the geometry off the view rather than asking
 * for it per frame).
 */
export function FrameStreamProvider({ children }: { children: ReactNode }) {
  const { project, currentFrame, stabilizedPreview, smoothingRevision } = useProjectContext();
  const [stream, setStream] = useState<FrameStream | null>(null);
  const [view, setView] = useState<FrameView | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [wantOutput, setWantOutput] = useState(false);
  const [wantGeometry, setWantGeometry] = useState(false);

  const proxy = project.proxy["half"];
  const proxyReady = Boolean(proxy?.ready && proxy.width && proxy.height);
  const projectId = project.id;
  const frameCount = project.video.frames;

  useEffect(() => {
    if (!proxyReady) return;
    const opened = new FrameStream(projectId, setView, setStreaming);
    setStream(opened);
    return () => {
      opened.close();
      setStream(null);
      setView(null);
      setStreaming(false);
    };
  }, [projectId, proxyReady]);

  // Which pictures of each frame are needed: the one the main canvas draws,
  // plus the stabilized one whenever the export's side-by-side view is up. Asked
  // for together so both canvases show the same frame — two independent
  // requests would let one lead the other during playback.
  const variants = useMemo<FrameVariant[]>(() => {
    const main: FrameVariant = stabilizedPreview ? "output" : "source";
    return wantOutput && main !== "output" ? [main, "output"] : [main];
  }, [stabilizedPreview, wantOutput]);

  useEffect(() => {
    stream?.setGeometryWanted(wantGeometry);
  }, [stream, wantGeometry]);

  useEffect(() => {
    if (!stream) return;
    stream.show(currentFrame, variants);
    // After the frame being looked at, never before it: a prefetch that queues
    // ahead of the current frame is the head-of-line problem the socket exists
    // to remove, re-created on the client.
    const timer = setTimeout(() => stream.prefetch(currentFrame, variants, frameCount), 0);
    return () => clearTimeout(timer);
  }, [stream, currentFrame, variants, frameCount]);

  // New smoothing is new warps, so every rendered output frame is stale.
  useEffect(() => {
    stream?.invalidateOutput();
  }, [stream, smoothingRevision]);

  const invalidateOutput = useCallback(() => stream?.invalidateOutput(), [stream]);

  const value = useMemo<FrameStreamValue>(() => {
    const errors = view?.errors ?? {};
    const failed = (Object.keys(errors) as FrameVariant[]).find((variant) => errors[variant]);
    return {
      view,
      viewFrame: view?.frame ?? currentFrame,
      geometry: view?.geometry ?? null,
      streaming,
      error: failed ? (errors[failed] ?? null) : null,
      wantOutput,
      setWantOutput,
      setWantGeometry,
      invalidateOutput,
    };
  }, [view, currentFrame, streaming, wantOutput, invalidateOutput]);

  return <FrameStreamContext.Provider value={value}>{children}</FrameStreamContext.Provider>;
}
