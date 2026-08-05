import type { JobEvent } from "../types";

/**
 * A small inline progress bar for one background job (track/solve/proxy/export).
 *
 * When the job publishes a live cursor (`partial`, see stable/server/jobs.py)
 * the bar says which frame is being worked on and how many planned runs are
 * done, rather than only a percentage — on a long track that is the difference
 * between "it is doing something" and "it is on frame 812, run 2 of 4" (todo 4).
 */
export function JobProgress({ job }: { job: JobEvent | undefined }) {
  if (!job || job.status === "done") return null;
  if (job.status === "error") {
    return <div className="job-progress job-progress--error">Failed: {job.error}</div>;
  }
  const pct = Math.round(job.fraction * 100);
  const partial = job.partial;
  const details: string[] = [];
  if (partial?.frame != null) details.push(`frame ${partial.frame}`);
  if (partial?.runs_total) details.push(`run ${Math.min(partial.runs_done ?? 0, partial.runs_total) + 1}/${partial.runs_total}`);

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
