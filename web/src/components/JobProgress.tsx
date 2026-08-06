import type { JobEvent } from "../types";

/**
 * A small inline progress bar for one background job (track/solve/proxy/export).
 *
 * When the job publishes a live cursor (`partial`, see stable/server/jobs.py)
 * the bar says which frame is being worked on rather than only a percentage —
 * on a long track that is the difference between "it is doing something" and
 * "it is on frame 812, run 2 of 4" (todo 4).
 *
 * The run counter comes from `job.message`, which the tracking job already sets
 * to "run 2/4" (stable/server/tracking.py). It used to be appended here from
 * `partial.runs_total` as well, which named the same run twice; the trackers
 * now report a fraction per *frame* with no message of their own, so the
 * message is reliably the run counter and the percentage actually moves.
 */
export function JobProgress({ job }: { job: JobEvent | undefined }) {
  // "cancelled" is a job auto-track superseded: moving two keyframes quickly
  // queues a second re-track and the server drops the first. That is the system
  // working, so it should not leave a stalled bar sitting on the card.
  if (!job || job.status === "done" || job.status === "cancelled") return null;
  if (job.status === "error") {
    return <div className="job-progress job-progress--error">Failed: {job.error}</div>;
  }
  const pct = Math.round(job.fraction * 100);
  const partial = job.partial;
  const details: string[] = [];
  if (partial?.frame != null) details.push(`frame ${partial.frame}`);

  return (
    <div className="job-progress">
      <div className="job-progress__bar">
        <div className="job-progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="job-progress__label">
        {job.message || job.status} ({pct}%)
        {details.length > 0 && <> · {details.join(" · ")}</>}
      </span>
    </div>
  );
}
