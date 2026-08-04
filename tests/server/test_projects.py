"""Project CRUD, upload, and proxy-frame serving."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ._helpers import upload_clip, wait_job


def test_create_project_generates_proxy(client: TestClient, tiny_clip) -> None:
    path, source = tiny_clip
    detail = upload_clip(client, path)

    assert detail["video"]["frames"] == source.frame_count
    assert detail["video"]["w"] == source.width
    assert detail["video"]["h"] == source.height
    assert detail["points"] == []
    assert detail["proxy"]["half"] == {"ready": False}

    job = wait_job(client, detail["id"], detail["proxy_job_id"])
    assert job["status"] == "done", job
    assert job["result"]["count"] == source.frame_count

    refreshed = client.get(f"/api/projects/{detail['id']}").json()
    assert refreshed["proxy"]["half"]["ready"] is True
    assert refreshed["proxy"]["half"]["count"] == source.frame_count


def test_create_project_rejects_empty_upload(client: TestClient) -> None:
    response = client.post("/api/projects", files={"file": ("empty.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_create_project_rejects_garbage_bytes(client: TestClient) -> None:
    response = client.post(
        "/api/projects", files={"file": ("not-a-video.mp4", b"this is not a video", "video/mp4")}
    )
    assert response.status_code == 400


def test_list_and_delete_project(client: TestClient, tiny_clip) -> None:
    path, _source = tiny_clip
    detail = upload_clip(client, path, name="Alpha")

    listing = client.get("/api/projects").json()
    assert any(p["id"] == detail["id"] and p["name"] == "Alpha" for p in listing)

    response = client.delete(f"/api/projects/{detail['id']}")
    assert response.status_code == 200

    listing_after = client.get("/api/projects").json()
    assert all(p["id"] != detail["id"] for p in listing_after)

    missing = client.get(f"/api/projects/{detail['id']}")
    assert missing.status_code == 404


def test_rename_project(client: TestClient, tiny_clip) -> None:
    path, _source = tiny_clip
    detail = upload_clip(client, path, name="Original")

    response = client.patch(f"/api/projects/{detail['id']}", json={"name": "Renamed"})
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"


def test_unknown_project_returns_404(client: TestClient) -> None:
    response = client.get("/api/projects/does-not-exist")
    assert response.status_code == 404


def test_frame_requires_proxy(client: TestClient, tiny_clip) -> None:
    path, _source = tiny_clip
    detail = upload_clip(client, path)
    # Fetching before the proxy job finishes must fail cleanly, not 500.
    response = client.get(f"/api/projects/{detail['id']}/frames/0")
    assert response.status_code == 404

    wait_job(client, detail["id"], detail["proxy_job_id"])
    ready = client.get(f"/api/projects/{detail['id']}/frames/0")
    assert ready.status_code == 200
    assert ready.headers["content-type"] == "image/jpeg"

    out_of_range = client.get(f"/api/projects/{detail['id']}/frames/9999")
    assert out_of_range.status_code == 404
