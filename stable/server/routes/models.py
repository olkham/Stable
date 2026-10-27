"""Model checkpoint management: list, download (with live progress), and delete.

Model weights (:mod:`stable.trackers.weights`) are process-wide, not tied to
any project — unlike every other route module, this one has no ``project_id``
in its paths. A download still wants the same "background work with live
progress" shape the job queue already gives track/solve/render jobs, so it is
submitted under :data:`~stable.server.jobs.MODELS_JOB_SCOPE`, a reserved
pseudo-project id, and streams over ``/ws/models`` (see ``routes/ws.py``)
rather than inventing a second progress mechanism.

Phase 2 shipped both bundled checkpoints **unpinned** (``sha256=None``) and
made :func:`~stable.trackers.weights.ensure_weights` fail closed on that
rather than silently trust the network; CoTracker3's are additionally
non-commercial (CC-BY-NC) and gated the same way. This module is the UI
surface for the explicit opt-in that unblocks either gate — never a way
around them. See ``CONTRIBUTING.md`` "Pinning a checkpoint".
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...trackers import weights as weights_mod
from ...trackers.registry import registry
from ..jobs import MODELS_JOB_SCOPE, JobManager, get_job_manager
from ..schemas import HfModelDownloadRequest, ModelDownloadRequest

__all__ = ["router"]

router = APIRouter(prefix="/models", tags=["models"])


def _jobs() -> JobManager:
    return get_job_manager()


def _trackers_for(key: str) -> list[str]:
    """Return the names of every discoverable tracker plugin using checkpoint ``key``.

    Uses whatever plugins are already registered rather than re-discovering,
    so callers control when the (possibly import-heavy) discovery pass runs.
    """
    names: list[str] = []
    for name, cls in registry.items():
        spec = getattr(cls, "model_spec", None)
        if spec is not None and spec.key == key:
            names.append(name)
    return sorted(names)


def _describe(spec: weights_mod.ModelSpec) -> dict[str, Any]:
    """UI-ready dict for one checkpoint: its cache/licence state plus who needs it."""
    row = spec.describe()
    row["notes"] = spec.notes
    row["trackers"] = _trackers_for(spec.key)
    return row


def _spec_or_404(key: str) -> weights_mod.ModelSpec:
    """Return the registered :class:`~stable.trackers.weights.ModelSpec` for ``key``.

    Raises:
        HTTPException: 404 if no registered checkpoint uses that key.
    """
    registry.discover()  # import every installed plugin so it has registered its spec(s)
    for spec in weights_mod.registered_models():
        if spec.key == key:
            return spec
    raise HTTPException(status_code=404, detail=f"no registered model {key!r}")


def _hf_plugins() -> dict[str, Any]:
    """Return every registered tracker whose weights come from a Hugging Face repo."""
    return {name: cls for name, cls in registry.items() if getattr(cls, "hf_repo", None)}


def _describe_hf(name: str, cls: Any) -> dict[str, Any]:
    """UI-ready dict for one Hugging Face-hosted checkpoint."""
    return {
        "key": name,
        "repo": cls.hf_repo,
        "license": getattr(cls, "license", ""),
        "downloaded": bool(cls.weights_downloaded()),
        "trackers": [name],
    }


def _hf_token_configured() -> bool:
    """Whether ``HF_TOKEN`` or a ``huggingface-cli login`` token is already available."""
    try:
        from huggingface_hub import get_token  # noqa: PLC0415
    except Exception:
        return False
    return get_token() is not None


@router.get("")
def list_models() -> dict[str, Any]:
    """List every registered checkpoint plus the process-wide download opt-in flags.

    The env flags (``STABLE_ALLOW_UNPINNED_WEIGHTS`` / ``STABLE_ACCEPT_NONCOMMERCIAL``)
    are surfaced so the UI can explain *why* a download needs an explicit
    checkbox instead of just failing — see the module docstring.
    """
    registry.discover()
    return {
        "models": [_describe(spec) for spec in weights_mod.registered_models()],
        "hf_models": [_describe_hf(name, cls) for name, cls in _hf_plugins().items()],
        "hf_token_configured": _hf_token_configured(),
        "env": weights_mod.env_flags(),
    }


@router.get("/jobs")
def list_model_jobs(jobs: JobManager = Depends(_jobs)) -> list[dict[str, Any]]:
    """Return every tracked model-download job.

    REST fallback for the initial paint, before the ``/ws/models`` WebSocket
    connects — mirrors ``GET /api/projects/{id}/jobs``.
    """
    return jobs.snapshot(MODELS_JOB_SCOPE)


@router.post("/{key}/download")
def download_model(
    key: str, body: ModelDownloadRequest, jobs: JobManager = Depends(_jobs)
) -> dict[str, str]:
    """Submit a background job that downloads and verifies one checkpoint.

    ``body.accept_noncommercial`` / ``body.allow_unpinned`` are this one
    download's explicit opt-in, exactly like the same-named
    :func:`~stable.trackers.weights.ensure_weights` arguments — a checked box
    in the UI, not a process-wide environment variable.

    Raises:
        HTTPException: 404 if ``key`` names no registered checkpoint.
    """
    spec = _spec_or_404(key)

    def _run(progress: Any) -> dict[str, Any]:
        path = weights_mod.ensure_weights(
            spec,
            progress=progress,
            accept_noncommercial=body.accept_noncommercial,
            allow_unpinned=body.allow_unpinned,
        )
        return {"key": spec.key, "path": str(path)}

    job = jobs.submit(MODELS_JOB_SCOPE, "model_download", _run, target=key)
    return {"job_id": job.id}


@router.post("/hf/{name}/download")
def download_hf_model(
    name: str, body: HfModelDownloadRequest, jobs: JobManager = Depends(_jobs)
) -> dict[str, str]:
    """Submit a background job that fetches a tracker's Hugging Face repo into the HF cache.

    ``body.token`` authenticates this one download (gated repos such as
    ``facebook/sam3``) and is never stored or echoed back.

    Raises:
        HTTPException: 404 if ``name`` is no tracker with a Hugging Face repo.
    """
    registry.discover()
    cls = _hf_plugins().get(name)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"no Hugging Face model for {name!r}")
    repo = str(cls.hf_repo)
    token = body.token.get_secret_value().strip() if body.token is not None else ""

    def _run(progress: Any) -> dict[str, Any]:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is not installed: pip install 'stable-tracker[sam3]'"
            ) from exc
        progress(0.0, f"downloading {repo}")
        path = snapshot_download(repo, token=token or None)
        return {"key": name, "path": str(path)}

    job = jobs.submit(MODELS_JOB_SCOPE, "model_download", _run, target=f"hf:{name}")
    return {"job_id": job.id}


@router.delete("/{key}")
def delete_model(key: str) -> dict[str, Any]:
    """Delete one checkpoint's cached file and checksum stamp, if present.

    Raises:
        HTTPException: 404 if ``key`` names no registered checkpoint.
    """
    spec = _spec_or_404(key)
    removed = weights_mod.clear_weights(spec)
    row = _describe(spec)
    row["removed"] = removed
    return row
