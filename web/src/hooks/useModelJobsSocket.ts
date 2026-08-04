import { useEffect, useState } from "react";
import { listModelJobs } from "../api/client";
import type { JobEvent } from "../types";

/**
 * Live job status for model-checkpoint downloads (stable/server/routes/models.py).
 *
 * Mirrors `useJobsSocket`, but model downloads are process-wide rather than
 * scoped to a project, so this connects once to `/ws/models` instead of
 * `/ws/projects/{id}`. Kept as its own hook (rather than parameterizing
 * `useJobsSocket`) since the two have different lifecycles: this one is not
 * keyed by anything that changes while the model manager is mounted, and
 * mount/unmount of the manager dialog is what starts and stops the socket.
 */
export function useModelJobsSocket(): Record<string, JobEvent> {
  const [jobs, setJobs] = useState<Record<string, JobEvent>>({});

  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    listModelJobs()
      .then((initial) => {
        if (cancelled) return;
        setJobs((prev) => {
          const next = { ...prev };
          for (const job of initial) next[job.id] = job;
          return next;
        });
      })
      .catch(() => {
        /* server may be mid-startup; the socket will still try */
      });

    function connect() {
      if (cancelled) return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/models`);
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
  }, []);

  return jobs;
}
