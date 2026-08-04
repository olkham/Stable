import type { JobEvent } from "../types";

/** A small inline progress bar for one background job (track/solve/proxy/export). */
export function JobProgress({ job }: { job: JobEvent | undefined }) {
  if (!job || job.status === "done") return null;
  if (job.status === "error") {
    return <div className="job-progress job-progress--error">Failed: {job.error}</div>;
  }
  const pct = Math.round(job.fraction * 100);
  return (
    <div className="job-progress">
      <div className="job-progress__bar">
        <div className="job-progress__fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="job-progress__label">
        {job.message || job.status} ({pct}%)
      </span>
    </div>
  );
}
