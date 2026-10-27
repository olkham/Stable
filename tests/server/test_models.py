"""Tests for the model checkpoint manager: GET /api/models, download, delete.

Downloads are served from ``file://`` URLs (see ``tests/test_weights.py``),
so the whole route is exercised offline. Every test registers its own
throwaway :class:`~stable.trackers.weights.ModelSpec` under a unique key —
:mod:`stable.trackers.weights`'s model table is process-wide, so reusing a
key across tests (or colliding with the real ``cotracker3_offline`` /
``bootstapir`` specs) would make tests order-dependent.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from stable.trackers.weights import ENV_MODEL_DIR, ModelSpec, register_model

PAYLOAD = b"pretend checkpoint bytes\n" * 16
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture(autouse=True)
def _model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the model cache at a temp directory so nothing touches the real one."""
    root = tmp_path / "models"
    monkeypatch.setenv(ENV_MODEL_DIR, str(root))
    monkeypatch.delenv("STABLE_ALLOW_UNPINNED_WEIGHTS", raising=False)
    monkeypatch.delenv("STABLE_ACCEPT_NONCOMMERCIAL", raising=False)
    return root


@pytest.fixture
def served(tmp_path: Path) -> str:
    """Write a fake checkpoint to disk and return a file:// URL for it."""
    source = tmp_path / "served" / "weights.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PAYLOAD)
    return source.as_uri()


def _unique_key() -> str:
    return "srvtest_" + uuid4().hex[:10]


def wait_model_job(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict:
    """Poll GET /api/models/jobs until ``job_id`` finishes (or raise on timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/api/models/jobs")
        assert response.status_code == 200, response.text
        for job in response.json():
            if job["id"] == job_id and job["status"] in ("done", "error", "cancelled"):
                return job
        time.sleep(0.02)
    raise TimeoutError(f"model job {job_id} did not finish within {timeout}s")


# -- listing ----------------------------------------------------------------


def test_lists_the_shipped_checkpoints_with_their_trackers(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    rows = {row["key"]: row for row in body["models"]}

    assert rows["cotracker3_offline"]["non_commercial"] is True
    assert rows["cotracker3_offline"]["license"] == "CC-BY-NC-4.0"
    assert rows["cotracker3_offline"]["trackers"] == ["cotracker3"]
    # Pinned in W5 (todo 22); see CONTRIBUTING.md "Pinning a checkpoint".
    assert rows["cotracker3_offline"]["pinned"] is True
    assert rows["cotracker3_offline"]["downloaded"] is False

    assert rows["bootstapir"]["non_commercial"] is False
    assert rows["bootstapir"]["license"] == "Apache-2.0"
    assert rows["bootstapir"]["trackers"] == ["tapir"]

    assert set(body["env"]) == {"allow_unpinned", "accept_noncommercial"}


def test_env_flags_reflect_the_process_environment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STABLE_ALLOW_UNPINNED_WEIGHTS", "1")
    monkeypatch.setenv("STABLE_ACCEPT_NONCOMMERCIAL", "1")
    body = client.get("/api/models").json()
    assert body["env"] == {"allow_unpinned": True, "accept_noncommercial": True}


def test_unknown_model_operations_404(client: TestClient) -> None:
    assert client.post("/api/models/nonexistent/download", json={}).status_code == 404
    assert client.delete("/api/models/nonexistent").status_code == 404


# -- download / delete --------------------------------------------------------


def test_download_and_delete_round_trip(client: TestClient, served: str) -> None:
    key = _unique_key()
    register_model(ModelSpec(key=key, url=served, sha256=DIGEST, notes="a throwaway test model"))

    before = {row["key"]: row for row in client.get("/api/models").json()["models"]}[key]
    assert before["downloaded"] is False
    assert before["notes"] == "a throwaway test model"
    assert before["trackers"] == []  # no shipped tracker plugin references it

    response = client.post(f"/api/models/{key}/download", json={})
    assert response.status_code == 200
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["key"] == key

    after = {row["key"]: row for row in client.get("/api/models").json()["models"]}[key]
    assert after["downloaded"] is True

    delete_response = client.delete(f"/api/models/{key}")
    assert delete_response.status_code == 200
    assert delete_response.json()["removed"] is True
    assert delete_response.json()["downloaded"] is False

    # Deleting again is not an error, just a no-op.
    assert client.delete(f"/api/models/{key}").json()["removed"] is False


def test_unpinned_download_fails_closed_without_opt_in(client: TestClient, served: str) -> None:
    key = _unique_key()
    register_model(ModelSpec(key=key, url=served, sha256=None))

    response = client.post(f"/api/models/{key}/download", json={})
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "error"
    assert "sha256" in job["error"]

    # The explicit per-request opt-in (a checkbox, not an env var) unblocks it.
    response = client.post(
        f"/api/models/{key}/download", json={"allow_unpinned": True}
    )
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "done", job


def test_non_commercial_download_fails_closed_without_opt_in(
    client: TestClient, served: str
) -> None:
    key = _unique_key()
    register_model(
        ModelSpec(key=key, url=served, sha256=DIGEST, non_commercial=True, license="CC-BY-NC-4.0")
    )

    response = client.post(f"/api/models/{key}/download", json={})
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "error"
    assert "non-commercial" in job["error"]

    response = client.post(
        f"/api/models/{key}/download", json={"accept_noncommercial": True}
    )
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "done", job


# -- Hugging Face-hosted checkpoints (SAM 3) ---------------------------------


def test_lists_sam3_as_a_hugging_face_model(client: TestClient) -> None:
    body = client.get("/api/models").json()
    rows = {row["key"]: row for row in body["hf_models"]}
    assert rows["sam3"]["repo"] == "facebook/sam3"
    assert rows["sam3"]["trackers"] == ["sam3"]
    assert isinstance(body["hf_token_configured"], bool)


def test_hf_download_passes_the_token_and_never_echoes_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls: list[tuple[str, str | None]] = []

    def fake_snapshot_download(repo: str, token: str | None = None) -> str:
        calls.append((repo, token))
        return "/fake/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    secret = "hf_" + uuid4().hex
    response = client.post("/api/models/hf/sam3/download", json={"token": secret})
    assert response.status_code == 200
    job = wait_model_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert calls == [("facebook/sam3", secret)]
    assert secret not in str(job)


def test_hf_download_unknown_tracker_404(client: TestClient) -> None:
    assert client.post("/api/models/hf/lk/download", json={}).status_code == 404
