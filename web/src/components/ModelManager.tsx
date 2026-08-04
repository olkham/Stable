import { useEffect, useRef, useState } from "react";
import * as api from "../api/client";
import { useModelJobsSocket } from "../hooks/useModelJobsSocket";
import type { ModelInfo } from "../types";
import { JobProgress } from "./JobProgress";

interface OptIn {
  unpinned: boolean;
  noncommercial: boolean;
}

/**
 * Model checkpoint manager: every checkpoint an installed tracker plugin
 * needs, its cache state, and a download (with live progress) or delete.
 *
 * Weights are never bundled (see `stable/trackers/weights.py`) and the two
 * shipped checkpoints ship **unpinned** — no verified sha256 yet — so a
 * download fails closed until pinned or explicitly allowed unverified;
 * CoTracker3's are additionally non-commercial (CC-BY-NC) and need a licence
 * opt-in. Both gates are surfaced here as checkboxes rather than a workaround
 * for either — see CONTRIBUTING.md "Pinning a checkpoint".
 */
export function ModelManager({ onClose }: { onClose: () => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [env, setEnv] = useState({ allow_unpinned: false, accept_noncommercial: false });
  const [opt, setOpt] = useState<Record<string, OptIn>>({});
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const jobs = useModelJobsSocket();

  const refresh = () =>
    api
      .listModels()
      .then((res) => {
        setModels(res.models);
        setEnv(res.env);
        setLoaded(true);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-read the catalogue once a download finishes, so downloaded/pinned state
  // updates without polling — mirrors ProjectContext's "seen done jobs" pattern.
  const seenDone = useRef(new Set<string>());
  useEffect(() => {
    for (const job of Object.values(jobs)) {
      if (job.kind !== "model_download" || seenDone.current.has(job.id)) continue;
      if (job.status !== "done" && job.status !== "error") continue;
      seenDone.current.add(job.id);
      refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs]);

  function optFor(key: string): OptIn {
    return opt[key] ?? { unpinned: false, noncommercial: false };
  }

  function setOptFor(key: string, patch: Partial<OptIn>) {
    setOpt((prev) => ({ ...prev, [key]: { ...optFor(key), ...patch } }));
  }

  async function handleDownload(model: ModelInfo) {
    setError(null);
    try {
      const flags = optFor(model.key);
      await api.downloadModel(model.key, {
        allowUnpinned: flags.unpinned,
        acceptNoncommercial: flags.noncommercial,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDelete(model: ModelInfo) {
    if (!confirm(`Delete the cached ${model.key} checkpoint? This frees disk space; ` +
      "it will be re-downloaded the next time a tracker that needs it prepares.")) {
      return;
    }
    setError(null);
    try {
      await api.deleteModel(model.key);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2>Model checkpoints</h2>
          <button className="link-button" onClick={onClose}>
            close
          </button>
        </div>
        <div className="modal__body">
          <p className="modal__hint">
            Deep trackers (CoTracker3, TAPIR) need weights too large to ship in the package; they
            download here, once, to <code>~/.stable/models</code>. The checkpoints below ship{" "}
            <strong>unpinned</strong> — no verified checksum yet — and CoTracker3&apos;s are also{" "}
            <strong>non-commercial</strong>. Both fail closed until you opt in below (or set{" "}
            <code>STABLE_ALLOW_UNPINNED_WEIGHTS</code> / <code>STABLE_ACCEPT_NONCOMMERCIAL</code>{" "}
            in the environment) — see CONTRIBUTING.md &quot;Pinning a checkpoint&quot;.
          </p>
          {error && <div className="error-text">{error}</div>}
          {loaded && models.length === 0 && (
            <p className="panel__empty">No installed tracker plugin registers a checkpoint.</p>
          )}
          <ul className="model-list">
            {models.map((model) => {
              const job = Object.values(jobs)
                .filter((j) => j.kind === "model_download" && j.target === model.key)
                .sort((a, b) => b.updated_at - a.updated_at)[0];
              const running = job && job.status !== "done" && job.status !== "error";
              const flags = optFor(model.key);
              const needsUnpinned = !model.pinned && !env.allow_unpinned;
              const needsLicense = model.non_commercial && !env.accept_noncommercial;
              const canDownload =
                (!needsUnpinned || flags.unpinned) && (!needsLicense || flags.noncommercial);
              return (
                <li key={model.key} className="model-list__item">
                  <div className="model-list__row">
                    <span className="model-list__key">{model.key}</span>
                    <span className="model-list__license">{model.license}</span>
                    {!model.pinned && (
                      <span
                        className="model-list__badge model-list__badge--warn"
                        title="No verified checksum yet; see CONTRIBUTING.md"
                      >
                        unpinned
                      </span>
                    )}
                    {model.non_commercial && (
                      <span
                        className="model-list__badge model-list__badge--warn"
                        title="Licence forbids commercial use; never bundled, opt-in only"
                      >
                        non-commercial
                      </span>
                    )}
                    {model.downloaded && (
                      <span className="model-list__badge model-list__badge--ok">cached</span>
                    )}
                  </div>
                  <div className="model-list__meta">
                    used by:{" "}
                    {model.trackers.length ? model.trackers.join(", ") : "(no installed tracker)"}
                  </div>
                  {model.notes && <p className="model-list__notes">{model.notes}</p>}
                  {!model.downloaded && (needsUnpinned || needsLicense) && (
                    <div className="model-list__opts">
                      {needsUnpinned && (
                        <label className="field-row">
                          <input
                            type="checkbox"
                            checked={flags.unpinned}
                            onChange={(e) => setOptFor(model.key, { unpinned: e.target.checked })}
                          />
                          Download anyway, unverified (no pinned sha256)
                        </label>
                      )}
                      {needsLicense && (
                        <label className="field-row">
                          <input
                            type="checkbox"
                            checked={flags.noncommercial}
                            onChange={(e) =>
                              setOptFor(model.key, { noncommercial: e.target.checked })
                            }
                          />
                          I accept the {model.license} non-commercial licence
                        </label>
                      )}
                    </div>
                  )}
                  <div className="model-list__actions">
                    {!model.downloaded && (
                      <button disabled={!canDownload || running} onClick={() => handleDownload(model)}>
                        {running ? "Downloading…" : "Download"}
                      </button>
                    )}
                    {model.downloaded && (
                      <button className="model-list__delete" onClick={() => handleDelete(model)}>
                        Delete cached file
                      </button>
                    )}
                    <JobProgress job={job} />
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
