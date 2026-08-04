import { useEffect, useRef, useState } from "react";
import { listJobs } from "../api/client";
import type { JobEvent } from "../types";

/**
 * Live job status for one project.
 *
 * Opens a WebSocket to stable/server's `/ws/projects/{id}` (see
 * stable/server/routes/ws.py) and keeps a map of every job seen, keyed by id.
 * Seeded from the REST snapshot first so jobs that finished before the socket
 * connects (e.g. proxy generation that raced the page load) still show up.
 * Reconnects automatically if the socket drops.
 */
export function useJobsSocket(projectId: string | undefined): Record<string, JobEvent> {
  const [jobs, setJobs] = useState<Record<string, JobEvent>>({});
  const projectRef = useRef(projectId);
  projectRef.current = projectId;

  useEffect(() => {
    setJobs({});
    if (!projectId) return;

    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    listJobs(projectId)
      .then((initial) => {
        if (cancelled) return;
        setJobs((prev) => {
          const next = { ...prev };
          for (const job of initial) next[job.id] = job;
          return next;
        });
      })
      .catch(() => {
        /* project may not exist yet; the socket will still try */
      });

    function connect() {
      if (cancelled) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/projects/${projectId}`);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as JobEvent & { type: string };
          if (payload.type === "job") {
            setJobs((prev) => ({ ...prev, [payload.id]: payload }));
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      socket.onclose = () => {
        if (cancelled) return;
        reconnectTimer = setTimeout(connect, 1000);
      };
      socket.onerror = () => {
        socket?.close();
      };
    }
    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [projectId]);

  return jobs;
}
