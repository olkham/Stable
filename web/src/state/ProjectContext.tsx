import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import * as api from "../api/client";
import { useJobsSocket } from "../hooks/useJobsSocket";
import type {
  Correction,
  HistoryStatus,
  JobEvent,
  ProjectDetail,
  TrackData,
  TrackerInfo,
} from "../types";

export type PlacingMode = "none" | "new-point" | "add-anchor";

/**
 * What a drag on the frame means.
 *
 * `offset` nudges the track and blends the nudge out over `falloff` frames
 * either side — the everyday "it's a couple of pixels off through here" fix.
 * `replace` hard-sets one frame and ignores what the tracker said — "I know
 * exactly where this is". Both are non-destructive corrections; the difference
 * is only which one gets appended.
 */
export type DragMode = "offset" | "replace";

const EMPTY_HISTORY: HistoryStatus = {
  can_undo: false,
  can_redo: false,
  undo_label: null,
  redo_label: null,
  undo_depth: 0,
  redo_depth: 0,
};

interface ProjectContextValue {
  project: ProjectDetail;
  refresh: () => Promise<ProjectDetail>;
  trackers: TrackerInfo[];

  jobs: Record<string, JobEvent>;
  latestJobFor: (kind: string, target?: string | null) => JobEvent | undefined;

  currentFrame: number;
  setCurrentFrame: (frame: number) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;

  selectedPointId: string | null;
  setSelectedPointId: (id: string | null) => void;

  placingMode: PlacingMode;
  setPlacingMode: (mode: PlacingMode) => void;

  stabilizedPreview: boolean;
  setStabilizedPreview: (on: boolean) => void;

  tracks: Record<string, TrackData>;
  loadTrack: (pointId: string) => Promise<void>;
  invalidateTrack: (pointId: string) => void;

  // -- manual editing (Phase 5) --
  /** Whether dragging a point on the frame edits it (vs. only scrubbing). */
  editing: boolean;
  setEditing: (on: boolean) => void;
  dragMode: DragMode;
  setDragMode: (mode: DragMode) => void;
  /** Frames an offset correction blends out over, either side of its keyframe. */
  falloff: number;
  setFalloff: (frames: number) => void;

  history: HistoryStatus;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  /** Apply one correction to a point and refresh everything that depends on it. */
  applyCorrection: (
    pointId: string,
    correction: Partial<Correction> & { kind: string },
  ) => Promise<void>;
  /** Re-read a point's track, the project and the history after any edit. */
  afterEdit: (pointId?: string) => Promise<void>;
  editError: string | null;
  setEditError: (message: string | null) => void;
  /** Bumped whenever an edit lands, so curve/solve views know to re-fetch. */
  editRevision: number;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function useProjectContext(): ProjectContextValue {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error("useProjectContext must be used inside <ProjectProvider>");
  return ctx;
}

export function ProjectProvider({
  projectId,
  onClose,
  children,
}: {
  projectId: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [trackers, setTrackers] = useState<TrackerInfo[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [placingMode, setPlacingMode] = useState<PlacingMode>("none");
  const [stabilizedPreview, setStabilizedPreview] = useState(false);
  const [tracks, setTracks] = useState<Record<string, TrackData>>({});
  const trackRequests = useRef(new Set<string>());

  const [editing, setEditing] = useState(false);
  const [dragMode, setDragMode] = useState<DragMode>("offset");
  const [falloff, setFalloff] = useState(8);
  const [history, setHistory] = useState<HistoryStatus>(EMPTY_HISTORY);
  const [editError, setEditError] = useState<string | null>(null);
  const [editRevision, setEditRevision] = useState(0);

  const jobs = useJobsSocket(projectId);

  const refresh = useCallback(async () => {
    const detail = await api.getProject(projectId);
    setProject(detail);
    return detail;
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    refresh().catch(() => {
      if (!cancelled) onClose();
    });
    api.getTrackers().then((rows) => {
      if (!cancelled) setTrackers(rows);
    });
    api
      .getHistory(projectId)
      .then((status) => {
        if (!cancelled) setHistory(status);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const loadTrack = useCallback(
    async (pointId: string) => {
      if (trackRequests.current.has(pointId)) return;
      trackRequests.current.add(pointId);
      try {
        const data = await api.getTrack(projectId, pointId);
        setTracks((prev) => ({ ...prev, [pointId]: data }));
      } catch {
        // point has not been tracked yet; leave it absent from the cache
      } finally {
        trackRequests.current.delete(pointId);
      }
    },
    [projectId],
  );

  const invalidateTrack = useCallback((pointId: string) => {
    setTracks((prev) => {
      if (!(pointId in prev)) return prev;
      const next = { ...prev };
      delete next[pointId];
      return next;
    });
  }, []);

  // A track fetch that always hits the network, for after an edit — loadTrack
  // deduplicates by point id, which is right for "make sure it is loaded" and
  // wrong when the answer has just changed.
  const reloadTrack = useCallback(
    async (pointId: string) => {
      const data = await api.getTrack(projectId, pointId);
      setTracks((prev) => ({ ...prev, [pointId]: data }));
    },
    [projectId],
  );

  const trackIds = useRef<string[]>([]);
  trackIds.current = Object.keys(tracks);

  const settle = useCallback(
    async (pointId?: string) => {
      // Undo can touch any point (one edit may have spanned several), so with no
      // specific target every cached track is re-read rather than guessed at.
      const targets = pointId ? [pointId] : trackIds.current;
      await Promise.all(targets.map((id) => reloadTrack(id).catch(() => undefined)));
      await refresh().catch(() => undefined);
      setHistory(await api.getHistory(projectId).catch(() => EMPTY_HISTORY));
      setEditRevision((n) => n + 1);
    },
    [projectId, reloadTrack, refresh],
  );

  const applyCorrection = useCallback(
    async (pointId: string, correction: Partial<Correction> & { kind: string }) => {
      setEditError(null);
      try {
        await api.addCorrection(projectId, pointId, correction);
        await settle(pointId);
      } catch (err) {
        setEditError(err instanceof Error ? err.message : String(err));
      }
    },
    [projectId, settle],
  );

  const step = useCallback(
    async (direction: "undo" | "redo") => {
      setEditError(null);
      try {
        if (direction === "undo") await api.undoEdit(projectId);
        else await api.redoEdit(projectId);
        await settle();
      } catch (err) {
        setEditError(err instanceof Error ? err.message : String(err));
      }
    },
    [projectId, settle],
  );

  const undo = useCallback(() => step("undo"), [step]);
  const redo = useCallback(() => step("redo"), [step]);

  // Ctrl/Cmd+Z and Ctrl+Shift+Z (or Ctrl+Y), ignored while typing in a field.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!(event.ctrlKey || event.metaKey)) return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const key = event.key.toLowerCase();
      if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        void undo();
      } else if ((key === "z" && event.shiftKey) || key === "y") {
        event.preventDefault();
        void redo();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

  // Whenever a track/retrack job for a point finishes, drop the stale cache
  // entry and refresh the project so its summary (frames, confidence) updates.
  const seenDoneJobs = useRef(new Set<string>());
  useEffect(() => {
    for (const job of Object.values(jobs)) {
      if (job.status !== "done" || seenDoneJobs.current.has(job.id)) continue;
      seenDoneJobs.current.add(job.id);
      if ((job.kind === "track" || job.kind === "retrack") && job.target) {
        invalidateTrack(job.target);
        loadTrack(job.target);
      }
      if (job.kind === "retrack") {
        api.getHistory(projectId).then(setHistory).catch(() => undefined);
        setEditRevision((n) => n + 1);
      }
      if (["track", "retrack", "proxy", "solve"].includes(job.kind)) {
        refresh().catch(() => undefined);
      }
    }
  }, [jobs, invalidateTrack, loadTrack, refresh, projectId]);

  const latestJobFor = useCallback(
    (kind: string, target?: string | null) => {
      let best: JobEvent | undefined;
      for (const job of Object.values(jobs)) {
        if (job.kind !== kind) continue;
        if (target !== undefined && job.target !== target) continue;
        if (!best || job.updated_at > best.updated_at) best = job;
      }
      return best;
    },
    [jobs],
  );

  const value = useMemo<ProjectContextValue | null>(() => {
    if (!project) return null;
    return {
      project,
      refresh,
      trackers,
      jobs,
      latestJobFor,
      currentFrame,
      setCurrentFrame,
      isPlaying,
      setIsPlaying,
      selectedPointId,
      setSelectedPointId,
      placingMode,
      setPlacingMode,
      stabilizedPreview,
      setStabilizedPreview,
      tracks,
      loadTrack,
      invalidateTrack,
      editing,
      setEditing,
      dragMode,
      setDragMode,
      falloff,
      setFalloff,
      history,
      undo,
      redo,
      applyCorrection,
      afterEdit: settle,
      editError,
      setEditError,
      editRevision,
    };
  }, [
    project,
    refresh,
    trackers,
    jobs,
    latestJobFor,
    currentFrame,
    isPlaying,
    selectedPointId,
    placingMode,
    stabilizedPreview,
    tracks,
    loadTrack,
    invalidateTrack,
    editing,
    dragMode,
    falloff,
    history,
    undo,
    redo,
    applyCorrection,
    settle,
    editError,
    editRevision,
  ]);

  if (!value) {
    return <div className="loading-screen">Loading project…</div>;
  }

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}
