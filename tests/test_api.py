from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_does_not_load_model() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "fallguard-lite"
    assert body["model"] in {"not_loaded", "ready", "error"}


def test_index_has_security_headers() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "FallGuard Lite" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_websocket_rejects_non_jpeg_frame() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_bytes(b"not-a-jpeg")
        response = websocket.receive_json()
    assert response["type"] == "error"
    assert response["code"] == "frame_format"
