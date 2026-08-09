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
import { useLiveTracking, type LiveTracking } from "../hooks/useLiveTracking";
import type {
  ClipRangeInfo,
  Correction,
  HistoryStatus,
  JobEvent,
  ProjectDetail,
  SmoothedTrack,
  SmoothingFilter,
  SmoothingResponse,
  TrackData,
  TrackerInfo,
} from "../types";

/**
 * What a left-click on the frame means (todo 4, fourth round).
 *
 * This was `PlacingMode` — `none | new-point | add-anchor` — which was already
 * a tool palette, just without a palette: the two placing modes were reachable
 * only from buttons inside the Tracks panel, and "none" silently meant three
 * different things depending on what was selected and where the pointer was.
 * Naming all five makes the canvas able to say which one is in force, in the
 * toolbar and in the cursor.
 *
 * `move`
 *     Select a track by clicking it, and drag (or click) the selected one to
 *     correct it. The everyday tool, and the default.
 * `add-point`
 *     The next click starts a new track there.
 * `add-anchor`
 *     The next click anchors the selected track at that position on this frame.
 * `delete`
 *     Click an anchor to remove it, or a track's marker to remove the whole
 *     track (confirmed, since that one is not an undoable correction).
 * `pan`
 *     Left-drag pans the view. Middle-drag does this in every tool; the tool
 *     exists for trackpads and tablets with no middle button.
 */
export type Tool = "move" | "add-point" | "add-anchor" | "delete" | "pan";

/**
 * Which overlay elements are drawn for a track.
 *
 * These used to be five global checkboxes above the canvas, which is the wrong
 * granularity once a shot has several tracks: "hide the dotted trails" is
 * usually "hide *that* track's trails", and hiding everything to see the frame
 * loses the one track being worked on too. So the checkboxes remain as the
 * defaults, and each track card can override them for itself.
 */
export interface Overlays {
  /** The trail through the whole clip. */
  paths: boolean;
  /** The same trail after the smoothing's stabilisation warp (todo 3). */
  smoothed: boolean;
  /** Where the point is on this frame — independent of the trail. */
  markers: boolean;
  /** The anchor markers: where the operator has stated the point's position. */
  anchors: boolean;
  /** The before/after ghost a correction leaves. */
  ghost: boolean;
  /** The pattern and search boxes (selected track only). */
  windows: boolean;
}

export const ALL_OVERLAYS: Overlays = {
  paths: true,
  smoothed: true,
  markers: true,
  anchors: true,
  ghost: true,
  windows: true,
};

/** Every overlay off — what "hide this track" resolves to. */
export const NO_OVERLAYS: Overlays = {
  paths: false,
  smoothed: false,
  markers: false,
  anchors: false,
  ghost: false,
  windows: false,
};

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

/**
 * What the canvas colours a track's path and current-frame marker by.
 *
 * The paths were always confidence-coloured, which is a genuinely useful signal
 * — green means trust it, red means go and look — but it means every track on
 * the frame is drawn from the same ramp and none of them can be told apart.
 * Each point already *has* a colour (`pointColor`, used for its anchors on the
 * frame and the timeline, and its panel swatch); this is the switch that lets the
 * path use it. The timeline's heat strip stays confidence-coloured either way,
 * so the signal is never entirely given up.
 */
export type PathColor = "confidence" | "track";

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
  /** Drop a job that has not started yet — the export queue's cancel (todo 8). */
  cancelJob: (jobId: string) => Promise<void>;
  /** In-flight track/retrack runs, by point id — their live cursor and partial path. */
  liveTracking: Record<string, LiveTracking>;

  currentFrame: number;
  setCurrentFrame: (frame: number) => void;
  isPlaying: boolean;
  setIsPlaying: (playing: boolean) => void;
  /** Playback speed as a multiple of the clip's own frame rate (todo 3). */
  playbackRate: number;
  setPlaybackRate: (rate: number) => void;
  /**
   * Whether Play stays inside the In/Out markers (todo 1, sixth round).
   *
   * Tracking and the export have honoured the range since it existed; playback
   * was the last thing that did not, so reviewing the section you are actually
   * working on meant watching the whole clip go past it. Lives here rather than
   * in the timeline because the range does, and because the two have to agree
   * about what "inside" means — `out` is exclusive.
   */
  playRangeOnly: boolean;
  setPlayRangeOnly: (on: boolean) => void;

  // -- the working range: In/Out markers (todo 1) --
  /** What tracking and export are limited to, or null before it has loaded. */
  clipRange: ClipRangeInfo | null;
  setRangeMarker: (marker: "in" | "out", frame: number) => Promise<void>;
  clearRange: () => Promise<void>;
  /**
   * Jump the playhead to a marker.
   *
   * "out" lands on `out - 1`, the last frame *inside* the range: `out` is
   * exclusive and names a frame that will be neither tracked nor exported,
   * which is not where "go to the out point" means to land. Does nothing when
   * no range is set — there is no marker to go to.
   */
  goToMarker: (marker: "in" | "out") => void;

  // -- watching a run (todo 15) --
  /**
   * Whether the viewer follows a running tracker's cursor.
   *
   * It releases the moment the operator scrubs: a view that fights the pointer
   * is worse than one that does not follow at all.
   */
  followTracker: boolean;
  setFollowTracker: (on: boolean) => void;

  selectedPointId: string | null;
  setSelectedPointId: (id: string | null) => void;

  /** What a left-click on the frame does; see :type:`Tool`. */
  tool: Tool;
  setTool: (tool: Tool) => void;
  /**
   * Anchor a point where it already is on the current frame (the `A` shortcut
   * and the panel's "+ anchor" both go through this).
   *
   * Returns false when the point has no position on this frame — an untracked
   * point, or a gap — which is the caller's cue to arm click-to-place instead:
   * there is nowhere to anchor it *to* without being told where.
   */
  addAnchorHere: (pointId?: string | null) => Promise<boolean>;
  /** Add an anchor at an explicit position, e.g. from a click on the frame. */
  addAnchorAt: (pointId: string, frame: number, x: number, y: number) => Promise<void>;
  /**
   * Remove the anchor on a frame, and whatever was tracked from it.
   *
   * Lives here rather than in a panel because there are two panels behind a
   * flag (`flags.ts`) and a rule this load-bearing — "deleting an anchor is one
   * request, then re-read the track it lived on" — must not have two
   * implementations that can drift.
   *
   * Returns the server's message on refusal (the last anchor, a frame with no
   * anchor on it), or null on success, leaving each panel to decide where to
   * put it.
   */
  deleteAnchorAt: (pointId: string, frame: number) => Promise<string | null>;
  /**
   * Track one direction from one frame, bounded.
   *
   * `x`/`y` are deliberately not passed: the run starts from wherever the point
   * already is on that frame, which is the anchor there when it has one. Same
   * return contract as `deleteAnchorAt`.
   */
  runFrom: (
    pointId: string,
    frame: number,
    direction: "fwd" | "bwd",
    bounds?: api.RunBounds,
  ) => Promise<string | null>;

  stabilizedPreview: boolean;
  setStabilizedPreview: (on: boolean) => void;

  // -- overlay visibility (todos 7, 10; per track, todo 9) --
  /** The defaults every track uses unless it says otherwise. */
  overlays: Overlays;
  setOverlays: (next: Overlays) => void;
  /** Master switch: no track draws anything while this is off. */
  overlaysEnabled: boolean;
  setOverlaysEnabled: (on: boolean) => void;
  /** What to draw for one track, after its own overrides and the master switch. */
  overlaysFor: (pointId: string) => Overlays;
  /** Override one element for one track (`null` restores the default). */
  setTrackOverlay: (pointId: string, element: keyof Overlays, value: boolean | null) => void;
  /** Whether a track has any override of its own, i.e. whether "reset" does anything. */
  trackOverrides: Record<string, Partial<Overlays>>;
  /**
   * What colours a track's path and marker on the canvas.
   *
   * `confidence` is the red→amber→green quality ramp, which answers "is this
   * track any good" but paints every track the same. `track` uses the point's
   * own colour, which is the only way to tell several tracks apart on one
   * frame. Both are wanted; neither can be both (todo 5, second round).
   */
  pathColor: PathColor;
  setPathColor: (mode: PathColor) => void;

  tracks: Record<string, TrackData>;
  loadTrack: (pointId: string) => Promise<void>;
  invalidateTrack: (pointId: string) => void;

  // -- smoothing, shared by every view of it (todo 2, fourth round) --
  //
  // The smoothing panel and the curve editor used to hold a copy each, fetched
  // separately and refreshed on different triggers, so changing a filter in one
  // left the other showing the previous answer until it remounted — "the curves
  // view doesn't get updates until the user closes and reopens it". One copy,
  // written through one function, makes that impossible to reintroduce.
  /** The filter list plus the raw and smoothed curves, or null before it loads. */
  smoothing: SmoothingResponse | null;
  smoothingError: string | null;
  reloadSmoothing: () => Promise<void>;
  /** Replace the whole filter list and adopt the curves the server computes. */
  commitSmoothing: (filters: SmoothingFilter[]) => Promise<void>;
  /** Bumped whenever the smoothed curves change — what re-reads derive from. */
  smoothingRevision: number;
  /**
   * Each point's track after the smoothing's stabilisation warp (todo 3).
   *
   * Where the feature ends up once the shot is stabilised, which is the only
   * per-point thing the smoothing actually produces — the filters themselves
   * act on four camera curves, not on any one track.
   */
  smoothedTracks: Record<string, SmoothedTrack>;

  // -- manual editing (Phase 5) --
  //
  // There is no `editing` flag: correcting a track used to be a mode you had to
  // enter, which meant the commonest action in the app was two actions (todo
  // 25). The selected track is now always editable, and undo is what makes that
  // safe. What remains is *what a move means*.
  dragMode: DragMode;
  setDragMode: (mode: DragMode) => void;
  /** Frames an offset correction blends out over, either side of the frame dragged. */
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
  const [currentFrame, setCurrentFrameState] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [playRangeOnly, setPlayRangeOnly] = useState(false);
  const [clipRange, setClipRange] = useState<ClipRangeInfo | null>(null);
  const [followTracker, setFollowTracker] = useState(true);
  // Set by the follow-the-tracker effect so the setter below can tell its own
  // scrubbing from the operator's (see `setCurrentFrame`).
  const followingRef = useRef(false);

  /**
   * Move the playhead, and stop following the tracker if it was the operator
   * who moved it. Everything scrubs through here, so "following releases on
   * the first manual scrub" is one rule in one place rather than a flag every
   * caller has to remember.
   */
  const setCurrentFrame = useCallback((frame: number) => {
    if (!followingRef.current) setFollowTracker(false);
    setCurrentFrameState(frame);
  }, []);
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("move");
  const [stabilizedPreview, setStabilizedPreview] = useState(false);
  const [overlays, setOverlays] = useState<Overlays>(ALL_OVERLAYS);
  const [overlaysEnabled, setOverlaysEnabled] = useState(true);
  const [trackOverrides, setTrackOverrides] = useState<Record<string, Partial<Overlays>>>({});
  const [pathColor, setPathColor] = useState<PathColor>("confidence");
  const [tracks, setTracks] = useState<Record<string, TrackData>>({});
  const trackRequests = useRef(new Set<string>());

  const [dragMode, setDragMode] = useState<DragMode>("offset");
  const [falloff, setFalloff] = useState(8);
  const [history, setHistory] = useState<HistoryStatus>(EMPTY_HISTORY);
  const [editError, setEditError] = useState<string | null>(null);
  const [editRevision, setEditRevision] = useState(0);
  const [smoothing, setSmoothing] = useState<SmoothingResponse | null>(null);
  const [smoothingError, setSmoothingError] = useState<string | null>(null);
  const [smoothingRevision, setSmoothingRevision] = useState(0);
  const [smoothedTracks, setSmoothedTracks] = useState<Record<string, SmoothedTrack>>({});

  const jobs = useJobsSocket(projectId);
  const liveTracking = useLiveTracking(jobs);

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

  // -- smoothing (todo 2, fourth round) -----------------------------------------

  const adoptSmoothing = useCallback((response: SmoothingResponse) => {
    setSmoothing(response);
    setSmoothingError(null);
    setSmoothingRevision((n) => n + 1);
  }, []);

  const reloadSmoothing = useCallback(async () => {
    try {
      adoptSmoothing(await api.getSmoothing(projectId));
    } catch (err) {
      setSmoothingError(err instanceof Error ? err.message : String(err));
    }
  }, [projectId, adoptSmoothing]);

  const commitSmoothing = useCallback(
    async (filters: SmoothingFilter[]) => {
      try {
        adoptSmoothing(await api.updateSmoothing(projectId, filters));
      } catch (err) {
        setSmoothingError(err instanceof Error ? err.message : String(err));
        // A rejected filter list leaves the server holding the previous one, so
        // re-read rather than leaving the panels showing an edit that did not land.
        await reloadSmoothing();
      }
    },
    [projectId, adoptSmoothing, reloadSmoothing],
  );

  // A track edit changes the raw solve's inputs, so the curves are stale too.
  useEffect(() => {
    void reloadSmoothing();
  }, [reloadSmoothing, editRevision]);

  // Where each point ends up once the shot is stabilised (todo 3). Re-read
  // whenever the smoothing changes, because that is the only thing that moves it.
  useEffect(() => {
    let cancelled = false;
    api
      .getSmoothedTracks(projectId)
      .then((response) => {
        if (cancelled) return;
        setSmoothedTracks(
          Object.fromEntries(response.points.map((track) => [track.point_id, track])),
        );
      })
      .catch(() => {
        // Not solved (or not smoothed) yet: there is no stabilised path to draw,
        // which is a state, not an error.
        if (!cancelled) setSmoothedTracks({});
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, smoothingRevision]);

  // -- the working range (todo 1) ---------------------------------------------

  const reloadRange = useCallback(async () => {
    try {
      setClipRange(await api.getClipRange(projectId));
    } catch {
      setClipRange(null);
    }
  }, [projectId]);

  useEffect(() => {
    void reloadRange();
  }, [reloadRange]);

  const setRangeMarker = useCallback(
    async (marker: "in" | "out", frame: number) => {
      setEditError(null);
      try {
        setClipRange(await api.updateClipRange(projectId, { [marker]: frame }));
      } catch (err) {
        setEditError(err instanceof Error ? err.message : String(err));
      }
    },
    [projectId],
  );

  const clearRange = useCallback(async () => {
    setEditError(null);
    try {
      setClipRange(await api.updateClipRange(projectId, { in: null, out: null }));
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err));
    }
  }, [projectId]);

  const goToMarker = useCallback(
    (marker: "in" | "out") => {
      if (!clipRange || clipRange.whole_clip) return;
      setCurrentFrame(marker === "in" ? clipRange.in : Math.max(clipRange.in, clipRange.out - 1));
    },
    [clipRange, setCurrentFrame],
  );

  // -- following a run (todo 15) ----------------------------------------------
  //
  // The live cursor is already published (`useLiveTracking`); this is the
  // viewer keeping up with it. Scrubbing manually turns it off — see
  // `setCurrentFrame`, which is the only place that distinction is made.
  useEffect(() => {
    if (!followTracker) return;
    const cursor = Object.values(liveTracking)
      .map((live) => live.frame)
      .filter((frame): frame is number => frame != null);
    if (cursor.length === 0) return;
    followingRef.current = true;
    setCurrentFrameState(Math.max(...cursor));
    // Cleared after the state update so a scrub in the same tick still counts
    // as the operator's.
    const id = setTimeout(() => {
      followingRef.current = false;
    }, 0);
    return () => clearTimeout(id);
  }, [followTracker, liveTracking]);

  const overlaysFor = useCallback(
    (pointId: string): Overlays => {
      if (!overlaysEnabled) return NO_OVERLAYS;
      return { ...overlays, ...(trackOverrides[pointId] ?? {}) };
    },
    [overlays, overlaysEnabled, trackOverrides],
  );

  const setTrackOverlay = useCallback(
    (pointId: string, element: keyof Overlays, value: boolean | null) => {
      setTrackOverrides((prev) => {
        const current = { ...(prev[pointId] ?? {}) };
        if (value === null) delete current[element];
        else current[element] = value;
        const next = { ...prev };
        if (Object.keys(current).length === 0) delete next[pointId];
        else next[pointId] = current;
        return next;
      });
    },
    [],
  );

  // -- anchors ----------------------------------------------------------------
  //
  // Adding an anchor goes through `settle`, not a bare `refresh`. A point's own
  // anchors live on the point, but the *list* the card and the timeline draw is
  // the server's `anchors` (the point's own plus the ones its corrections were
  // promoted to) and comes with the track — so refreshing only the project left
  // a just-placed anchor invisible in the list until the next run replaced the
  // cached track.
  const addAnchorAt = useCallback(
    async (pointId: string, frame: number, x: number, y: number) => {
      setEditError(null);
      try {
        await api.addAnchor(projectId, pointId, frame, x, y);
        await settle(pointId);
      } catch (err) {
        setEditError(err instanceof Error ? err.message : String(err));
      }
    },
    [projectId, settle],
  );

  const deleteAnchorAt = useCallback(
    async (pointId: string, frame: number) => {
      try {
        await api.deleteAnchor(projectId, pointId, frame);
        // Through `settle`, not a bare refresh: the anchor list *and* the spans
        // a run from that anchor produced both come with the track, which is
        // cached separately from the project.
        await settle(pointId);
        return null;
      } catch (err) {
        return err instanceof Error ? err.message : String(err);
      }
    },
    [projectId, settle],
  );

  const runFrom = useCallback(
    async (
      pointId: string,
      frame: number,
      direction: "fwd" | "bwd",
      bounds: api.RunBounds = {},
    ) => {
      try {
        await api.retrackPoint(projectId, pointId, { frame, direction, ...bounds });
        invalidateTrack(pointId);
        return null;
      } catch (err) {
        return err instanceof Error ? err.message : String(err);
      }
    },
    [projectId, invalidateTrack],
  );

  const addAnchorHere = useCallback(
    async (pointId?: string | null) => {
      const id = pointId ?? selectedPointId;
      if (!id || !project) return false;
      const track = tracks[id];
      const x = track?.x[currentFrame];
      const y = track?.y[currentFrame];
      if (x != null && y != null) {
        await addAnchorAt(id, currentFrame, x, y);
        return true;
      }
      // No tracked position here: an anchor already on this frame is the
      // answer, so report success rather than arming a pointless click.
      const anchored = project.points
        .find((p) => p.id === id)
        ?.queries.some((q) => q.frame === currentFrame);
      return Boolean(anchored);
    },
    [project, tracks, currentFrame, selectedPointId, addAnchorAt],
  );

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

  // Frame navigation (todo 9): left/right steps a frame, space plays/pauses,
  // up/down jumps to the selected point's next/previous anchor (its own plus
  // whatever its manual corrections have been promoted to -- the track's
  // `anchors`, which is the one list everything reads), home/end jump to the
  // first/last frame. Ignored while typing in a field, same guard as undo/redo
  // above.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!project || event.ctrlKey || event.metaKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      // A focused button also treats space/enter as "activate it" -- deferring
      // to that native behaviour there avoids a focused Play button both
      // toggling itself and being toggled again by this shortcut.
      if (target && /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(target.tagName)) return;
      const lastFrame = Math.max(project.video.frames - 1, 0);

      if (event.key === "ArrowRight") {
        event.preventDefault();
        setCurrentFrame(Math.min(lastFrame, currentFrame + 1));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setCurrentFrame(Math.max(0, currentFrame - 1));
      } else if (event.key === " " || event.code === "Space") {
        event.preventDefault();
        setIsPlaying((playing) => !playing);
      } else if (event.key === "Home") {
        event.preventDefault();
        setCurrentFrame(0);
      } else if (event.key === "End") {
        event.preventDefault();
        setCurrentFrame(lastFrame);
      } else if (event.key === "i" || event.key === "I") {
        // In/Out from the playhead (todo 1) — the pair of keys every NLE uses.
        // Shifted, they *go to* the marker instead of setting it, which is the
        // same NLE convention and the answer to "where did I put the in point
        // on this ten-minute take" (todo 3, second round).
        event.preventDefault();
        if (event.shiftKey) goToMarker("in");
        else void setRangeMarker("in", currentFrame);
      } else if (event.key === "o" || event.key === "O") {
        event.preventDefault();
        if (event.shiftKey) goToMarker("out");
        else void setRangeMarker("out", currentFrame + 1);
      } else if (event.key === "a" || event.key === "A") {
        // Anchor the selected point on this frame. Where it already has a
        // position that needs no aiming, so plain `A` just does it; with
        // Shift (or with nothing to anchor, e.g. an untracked point) it arms
        // click-to-place instead, which is the same two-step the panel button
        // uses. Pressing it again disarms, so the key is never a trap.
        if (!selectedPointId) return;
        event.preventDefault();
        if (tool === "add-anchor") {
          setTool("move");
          return;
        }
        if (event.shiftKey) {
          setTool("add-anchor");
          return;
        }
        void addAnchorHere(selectedPointId).then((done) => {
          if (!done) setTool("add-anchor");
        });
      } else if (event.key === "v" || event.key === "V") {
        // The tool palette's shortcuts (todo 4). `A` is above, because anchoring
        // where the point already is does not need the tool at all.
        event.preventDefault();
        setTool("move");
      } else if (event.key === "n" || event.key === "N") {
        event.preventDefault();
        setTool("add-point");
      } else if (event.key === "d" || event.key === "D") {
        event.preventDefault();
        setTool("delete");
      } else if (event.key === "h" || event.key === "H") {
        event.preventDefault();
        setTool("pan");
      } else if (event.key === "Escape") {
        // Always a way back to the harmless tool, from anywhere.
        setTool("move");
      } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        if (!selectedPointId) return;
        const point = project.points.find((p) => p.id === selectedPointId);
        // The track's list when it is loaded (it includes the anchors the
        // corrections were promoted to), the point's own otherwise.
        const anchors = tracks[selectedPointId]?.anchors;
        const frames = (anchors ?? point?.queries)?.map((q) => q.frame) ?? [];
        const marks = [...new Set(frames)].sort((a, b) => a - b);
        if (marks.length === 0) return;
        event.preventDefault();
        if (event.key === "ArrowUp") {
          const next = marks.find((f) => f > currentFrame);
          setCurrentFrame(next ?? marks[marks.length - 1]);
        } else {
          const prev = [...marks].reverse().find((f) => f < currentFrame);
          setCurrentFrame(prev ?? marks[0]);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    project,
    currentFrame,
    selectedPointId,
    tracks,
    tool,
    addAnchorHere,
    setRangeMarker,
    goToMarker,
    setCurrentFrame,
    setIsPlaying,
  ]);

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
      if (job.kind === "solve") {
        // A new solve is new raw curves, so every view of them is stale — and
        // the smoothed ones were recomputed server-side by the same job.
        void reloadSmoothing();
      }
      if (["track", "retrack", "proxy", "solve"].includes(job.kind)) {
        refresh().catch(() => undefined);
      }
    }
  }, [jobs, invalidateTrack, loadTrack, refresh, reloadSmoothing, projectId]);

  const cancelJob = useCallback(
    async (jobId: string) => {
      setEditError(null);
      try {
        await api.cancelJob(projectId, jobId);
      } catch (err) {
        // Most often "it already started", which is a fact about the queue
        // rather than a failure of the click.
        setEditError(err instanceof Error ? err.message : String(err));
      }
    },
    [projectId],
  );

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
      cancelJob,
      liveTracking,
      currentFrame,
      setCurrentFrame,
      isPlaying,
      setIsPlaying,
      playbackRate,
      setPlaybackRate,
      playRangeOnly,
      setPlayRangeOnly,
      clipRange,
      setRangeMarker,
      clearRange,
      goToMarker,
      followTracker,
      setFollowTracker,
      selectedPointId,
      setSelectedPointId,
      tool,
      setTool,
      addAnchorHere,
      addAnchorAt,
      deleteAnchorAt,
      runFrom,
      stabilizedPreview,
      setStabilizedPreview,
      overlays,
      setOverlays,
      overlaysEnabled,
      setOverlaysEnabled,
      overlaysFor,
      setTrackOverlay,
      trackOverrides,
      pathColor,
      setPathColor,
      tracks,
      loadTrack,
      invalidateTrack,
      smoothing,
      smoothingError,
      reloadSmoothing,
      commitSmoothing,
      smoothingRevision,
      smoothedTracks,
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
    cancelJob,
    liveTracking,
    currentFrame,
    setCurrentFrame,
    isPlaying,
    playbackRate,
    playRangeOnly,
    clipRange,
    setRangeMarker,
    clearRange,
    goToMarker,
    followTracker,
    selectedPointId,
    tool,
    addAnchorHere,
    addAnchorAt,
    deleteAnchorAt,
    runFrom,
    stabilizedPreview,
    overlays,
    overlaysEnabled,
    overlaysFor,
    setTrackOverlay,
    trackOverrides,
    pathColor,
    tracks,
    loadTrack,
    invalidateTrack,
    smoothing,
    smoothingError,
    reloadSmoothing,
    commitSmoothing,
    smoothingRevision,
    smoothedTracks,
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
