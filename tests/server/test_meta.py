"""Health check, tracker catalogue and device info endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_only_the_dev_server_may_call_across_origins(client: TestClient) -> None:
    # The UI is same-origin (static mount or Vite's proxy), so an open CORS
    # policy only ever helped *other* pages: any site the operator visited
    # could have deleted projects or started jobs here and read the answers.
    stranger = client.get("/api/health", headers={"Origin": "https://example.com"})
    assert stranger.status_code == 200
    assert "access-control-allow-origin" not in stranger.headers

    preflight = client.options(
        "/api/projects/abc",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers

    dev = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert dev.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_trackers_lists_builtins(client: TestClient) -> None:
    response = client.get("/api/trackers")
    assert response.status_code == 200
    names = {row["name"] for row in response.json()}
    assert {"dummy", "lk", "template"} <= names
    # Every entry is UI-ready: greys out unavailable trackers rather than 500ing,
    # and says what would make an unavailable one work.
    for row in response.json():
        assert set(row) == {
            "name",
            "display_name",
            "requires_gpu",
            "license",
            "tracks_boxes",
            "inputs",
            "available",
            "unavailable_reason",
        }
        assert (row["unavailable_reason"] is None) == row["available"]


def test_devices(client: TestClient) -> None:
    response = client.get("/api/devices")
    assert response.status_code == 200
    body = response.json()
    assert "selected" in body
    assert body["selected"] in ("cpu", "cuda", "mps") or body["selected"].startswith("cuda")
