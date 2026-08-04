"""Health check, tracker catalogue and device info endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_trackers_lists_builtins(client: TestClient) -> None:
    response = client.get("/api/trackers")
    assert response.status_code == 200
    names = {row["name"] for row in response.json()}
    assert {"dummy", "lk", "template"} <= names
    # Every entry is UI-ready: greys out unavailable trackers rather than 500ing.
    for row in response.json():
        assert set(row) == {"name", "display_name", "requires_gpu", "license", "available"}


def test_devices(client: TestClient) -> None:
    response = client.get("/api/devices")
    assert response.status_code == 200
    body = response.json()
    assert "selected" in body
    assert body["selected"] in ("cpu", "cuda", "mps") or body["selected"].startswith("cuda")
