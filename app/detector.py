"""Pose inference and lightweight per-session person tracking.

The expensive YOLO model is shared by the service, while :class:`SessionAnalyzer`
owns tracking and fall state.  This keeps histories from different browser
connections isolated.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.fall_detector import FallDetector, FallThresholds, PoseObservation


@dataclass(slots=True)
class RawPerson:
    """A person returned by pose inference before temporal tracking."""

    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]]
    torso_angle_degrees: float | None
    keypoint_confidence: float | None


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_seen: float
    hits: int = 1


@dataclass(slots=True)
class PersonResult:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]]
    state: str
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [round(value, 2) for value in self.bbox],
            "confidence": round(self.confidence, 4),
            "keypoints": [
                [round(x, 2), round(y, 2), round(confidence, 3)]
                for x, y, confidence in self.keypoints
            ],
            "state": self.state,
            "risk_score": round(self.risk_score, 4),
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class PoseModel:
    """Thread-safe, lazily loaded Ultralytics pose model."""

    def __init__(self) -> None:
        self.model_path = os.getenv("MODEL_PATH", "yolo11n-pose.pt")
        self.model_sha256 = os.getenv("MODEL_SHA256", "").strip().lower()
        self.device = os.getenv("MODEL_DEVICE", "cpu")
        self.image_size = int(os.getenv("MODEL_IMAGE_SIZE", "640"))
        self.confidence = float(os.getenv("MODEL_CONFIDENCE", "0.38"))
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _verify_model_file(self) -> None:
        expected = self.model_sha256
        path = Path(self.model_path)
        checksum_path = path.with_name(f"{path.name}.sha256")
        if not expected and checksum_path.is_file():
            expected = checksum_path.read_text(encoding="utf-8").split()[0].strip().lower()
        if not expected:
            return
        if not path.is_file():
            raise RuntimeError("MODEL_SHA256 was supplied, but MODEL_PATH is not a local file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("The pose model checksum does not match MODEL_SHA256")

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            self._verify_model_file()
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._load_error = None
            return self._model
        except Exception as exc:
            self._load_error = type(exc).__name__
            raise RuntimeError("The pose model could not be loaded") from exc

    def predict(self, frame: Any) -> list[RawPerson]:
        """Run pose inference. Calls are serialized because the model is shared."""

        with self._lock:
            model = self._ensure_loaded()
            predictions = model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence,
                classes=[0],
                device=self.device,
                verbose=False,
            )
        if not predictions:
            return []
        result = predictions[0]
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        coordinates = result.keypoints.xy.detach().cpu().tolist()
        if result.keypoints.conf is None:
            point_confidences = [[1.0] * len(points) for points in coordinates]
        else:
            point_confidences = result.keypoints.conf.detach().cpu().tolist()

        people: list[RawPerson] = []
        for bbox, confidence, points, point_scores in zip(
            boxes, confidences, coordinates, point_confidences, strict=False
        ):
            keypoints = [
                (float(point[0]), float(point[1]), float(score))
                for point, score in zip(points, point_scores, strict=False)
            ]
            torso_angle, torso_confidence = _torso_features(keypoints)
            people.append(
                RawPerson(
                    bbox=tuple(float(value) for value in bbox[:4]),
                    confidence=float(confidence),
                    keypoints=keypoints,
                    torso_angle_degrees=torso_angle,
                    keypoint_confidence=torso_confidence,
                )
            )
        return people


def _torso_features(
    keypoints: list[tuple[float, float, float]],
) -> tuple[float | None, float | None]:
    """Return torso angle from vertical and mean confidence for COCO keypoints."""

    if len(keypoints) < 13:
        return None, None
    required = [keypoints[index] for index in (5, 6, 11, 12)]
    confidence = sum(point[2] for point in required) / 4.0
    if confidence < 0.25:
        return None, confidence
    shoulder_x = (required[0][0] + required[1][0]) / 2.0
    shoulder_y = (required[0][1] + required[1][1]) / 2.0
    hip_x = (required[2][0] + required[3][0]) / 2.0
    hip_y = (required[2][1] + required[3][1]) / 2.0
    dx = abs(hip_x - shoulder_x)
    dy = abs(hip_y - shoulder_y)
    if dx + dy < 1e-6:
        return None, confidence
    return math.degrees(math.atan2(dx, dy)), confidence


class PersonTracker:
    """Small dependency-free tracker using IoU and center proximity."""

    def __init__(self, ttl_seconds: float = 1.25) -> None:
        self.ttl_seconds = ttl_seconds
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[RawPerson], timestamp: float) -> list[tuple[int, RawPerson]]:
        self._expire(timestamp)
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = _iou(track.bbox, detection.bbox)
                proximity = _center_proximity(track.bbox, detection.bbox)
                score = overlap * 0.75 + proximity * 0.25
                if overlap >= 0.12 or proximity >= 0.72:
                    candidates.append((score, track_id, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        matches: dict[int, int] = {}
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            matches[detection_index] = track_id

        output: list[tuple[int, RawPerson]] = []
        for index, detection in enumerate(detections):
            track_id = matches.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(track_id, detection.bbox, timestamp)
            else:
                track = self._tracks[track_id]
                track.bbox = detection.bbox
                track.last_seen = timestamp
                track.hits += 1
            output.append((track_id, detection))
        return output

    def _expire(self, timestamp: float) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.ttl_seconds
        ]
        for track_id in stale:
            del self._tracks[track_id]


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_proximity(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
    scale = max(
        1.0,
        (first[2] - first[0]),
        (first[3] - first[1]),
        (second[2] - second[0]),
        (second[3] - second[1]),
    )
    return max(0.0, 1.0 - distance / scale)


class SessionAnalyzer:
    """Turn independent model detections into tracked, temporal results."""

    def __init__(self, fall_detector: FallDetector | None = None) -> None:
        self.tracker = PersonTracker()
        self.fall_detector = fall_detector or FallDetector(
            FallThresholds(
                min_standing_seconds=float(os.getenv("STANDING_SECONDS", "1.0")),
                fallen_hold_seconds=float(os.getenv("FALL_HOLD_SECONDS", "0.75")),
                recovery_hold_seconds=float(os.getenv("RECOVERY_SECONDS", "1.0")),
                track_ttl_seconds=float(os.getenv("TRACK_TTL_SECONDS", "2.0")),
                drop_speed_heights_per_second=float(os.getenv("DROP_SPEED", "0.65")),
            )
        )

    def analyze(
        self, detections: list[RawPerson], timestamp: float | None = None
    ) -> list[PersonResult]:
        observed_at = time.monotonic() if timestamp is None else timestamp
        people: list[PersonResult] = []
        for track_id, detection in self.tracker.update(detections, observed_at):
            fall = self.fall_detector.update(
                PoseObservation(
                    track_id=track_id,
                    timestamp=observed_at,
                    bbox=detection.bbox,
                    torso_angle_degrees=detection.torso_angle_degrees,
                    keypoint_confidence=detection.keypoint_confidence,
                )
            )
            payload = fall.as_dict()
            people.append(
                PersonResult(
                    track_id=track_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=detection.keypoints,
                    state=str(payload["state"]),
                    risk_score=float(payload["score"]),
                    reasons=list(payload["reasons"]),
                    metrics=dict(payload["metrics"]),
                )
            )
        self.fall_detector.cleanup(observed_at)
        return people
