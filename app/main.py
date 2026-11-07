"""FastAPI entry point for FallGuard Lite."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.detector import PoseModel, SessionAnalyzer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", "2500000"))
MAX_FRAME_PIXELS = int(os.getenv("MAX_FRAME_PIXELS", "2100000"))
MIN_FRAME_INTERVAL = float(os.getenv("MIN_FRAME_INTERVAL", "0.05"))

app = FastAPI(
    title="FallGuard Lite",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
pose_model = PoseModel()


@app.middleware("http")
async def security_headers(request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self' blob: data:; "
        "media-src 'self' blob:; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    if pose_model.load_error:
        model_status = "error"
    else:
        model_status = "ready" if pose_model.ready else "not_loaded"
    return {"status": "ok", "service": "fallguard-lite", "model": model_status}


def _origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    configured = {
        item.strip().rstrip("/")
        for item in os.getenv("ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    }
    host = websocket.headers.get("host", "")
    configured.update({f"http://{host}", f"https://{host}"})
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    normalized = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return parsed.scheme in {"http", "https"} and normalized in configured


def _decode_frame(payload: bytes):  # type: ignore[no-untyped-def]
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("frame_size")
    if len(payload) < 4 or payload[:3] != b"\xff\xd8\xff":
        raise ValueError("frame_format")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame_decode")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or width * height > MAX_FRAME_PIXELS:
        raise ValueError("frame_dimensions")
    return frame


@app.websocket("/ws/analyze")
async def analyze_socket(websocket: WebSocket) -> None:
    if not _origin_allowed(websocket):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await websocket.accept()
    analyzer = SessionAnalyzer()
    last_frame_at = 0.0
    try:
        while True:
            payload = await websocket.receive_bytes()
            received_at = time.monotonic()
            if received_at - last_frame_at < MIN_FRAME_INTERVAL:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "rate_limited",
                        "message": "Frames are arriving too quickly.",
                    }
                )
                continue
            last_frame_at = received_at
            try:
                frame = _decode_frame(payload)
            except ValueError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": str(exc),
                        "message": "The frame is invalid or too large.",
                    }
                )
                continue

            if not pose_model.ready:
                await websocket.send_json({"type": "status", "status": "loading"})
            try:
                detections = await asyncio.to_thread(pose_model.predict, frame)
                people = analyzer.analyze(detections, received_at)
            except Exception:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "inference_failed",
                        "message": "Pose inference failed.",
                    }
                )
                continue

            height, width = frame.shape[:2]
            await websocket.send_json(
                {
                    "type": "result",
                    "frame_width": width,
                    "frame_height": height,
                    "latency_ms": round((time.monotonic() - received_at) * 1000.0, 1),
                    "people": [person.as_dict() for person in people],
                }
            )
    except WebSocketDisconnect:
        return
    except RuntimeError:
        await websocket.close(code=1011, reason="Connection error")
