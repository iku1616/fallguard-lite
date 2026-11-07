from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


def make_jpeg(width: int, height: int, *, progressive: bool = False) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
        [cv2.IMWRITE_JPEG_PROGRESSIVE, int(progressive)],
    )
    assert success
    return encoded.tobytes()


def test_websocket_rejects_oversized_image_before_decode(monkeypatch) -> None:
    payload = make_jpeg(1600, 1400)
    assert len(payload) < main.MAX_FRAME_BYTES
    decoded_sizes = []
    real_decode = cv2.imdecode

    def observed_decode(*args, **kwargs):
        frame = real_decode(*args, **kwargs)
        decoded_sizes.append(frame.nbytes if frame is not None else 0)
        return frame

    monkeypatch.setattr(main.cv2, "imdecode", observed_decode)
    with TestClient(main.app) as client, client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_bytes(payload)
        assert websocket.receive_json()["code"] == "frame_dimensions"
    assert decoded_sizes == []


@pytest.mark.parametrize("progressive", [False, True])
def test_valid_jpeg_reaches_inference_through_websocket(monkeypatch, progressive) -> None:
    payload = make_jpeg(80, 48, progressive=progressive)
    observed_shapes = []

    def predict(frame):
        observed_shapes.append(frame.shape)
        return []

    monkeypatch.setattr(main.pose_model, "predict", predict)
    with TestClient(main.app) as client, client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_bytes(payload)
        result = websocket.receive_json()
        if result["type"] == "status":
            result = websocket.receive_json()
        assert result["type"] == "result"
        assert (result["frame_width"], result["frame_height"]) == (80, 48)
    assert observed_shapes == [(48, 80, 3)]


def test_jpeg_metadata_cannot_spoof_dimensions() -> None:
    from app.jpeg import jpeg_dimensions

    payload = make_jpeg(1600, 1400)
    # Fake small SOF bytes inside an APP segment must never be parsed as a marker.
    metadata = b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    app_segment = b"\xff\xe1" + (len(metadata) + 2).to_bytes(2, "big") + metadata
    payload = payload[:2] + app_segment + payload[2:]
    assert jpeg_dimensions(payload) == (1600, 1400)
    with pytest.raises(ValueError, match="frame_dimensions"):
        main._decode_frame(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xd8\xff",  # Truncated marker.
        b"\xff\xd8\xff\xe1\x00\x01",  # Segment length includes its own two bytes.
        b"\xff\xd8\xff\xe1\xff\xff",  # Segment extends beyond the message.
        b"\xff\xd8\xff\xda\x00\x02",  # Scan before any dimensions.
        b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x01\x00\x01\x00",  # No components.
    ],
)
def test_malformed_header_is_rejected_through_websocket(payload) -> None:
    with TestClient(main.app) as client, client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_bytes(payload)
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert response["code"] in {"frame_format", "frame_decode"}


def test_jpeg_cannot_defer_height_to_after_scan() -> None:
    payload = bytearray(make_jpeg(80, 48))
    sof = payload.index(b"\xff\xc0")
    payload[sof + 5 : sof + 7] = b"\x00\x00"
    with pytest.raises(ValueError, match="frame_dimensions"):
        main._decode_frame(bytes(payload))


def test_native_decoder_rejection_is_reported_without_details() -> None:
    payload = bytearray(make_jpeg(80, 48))
    # A valid header passes preflight; an invalid quantization table is rejected
    # by the real native decoder.
    dqt = payload.index(b"\xff\xdb")
    payload[dqt + 4] = 0x0F
    with TestClient(main.app) as client, client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_bytes(bytes(payload))
        response = websocket.receive_json()
        assert response == {
            "type": "error",
            "code": "frame_decode",
            "message": "The frame is invalid or too large.",
        }
