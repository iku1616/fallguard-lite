"""Temporal fall-state estimation independent from the vision model.

The detector consumes only small, already-extracted pose features. Keeping this
module free of OpenCV/ML dependencies makes its state transitions deterministic
and easy to test.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import median

TrackId = Hashable


class FallState(StrEnum):
    NORMAL = "normal"
    SUSPECTED = "suspected"
    FALLEN = "fallen"
    RECOVERING = "recovering"


@dataclass(frozen=True, slots=True)
class FallThresholds:
    """Time and geometry thresholds used by :class:`FallDetector`."""

    min_standing_seconds: float = 1.0
    fallen_hold_seconds: float = 0.75
    recovery_hold_seconds: float = 1.0
    drop_memory_seconds: float = 1.25
    cooldown_seconds: float = 2.5
    track_ttl_seconds: float = 2.0
    drop_speed_heights_per_second: float = 0.65
    upright_aspect_ratio: float = 0.85
    upright_torso_angle_degrees: float = 35.0
    horizontal_aspect_ratio: float = 1.05
    strong_horizontal_aspect_ratio: float = 1.35
    horizontal_torso_angle_degrees: float = 55.0
    collapsed_height_ratio: float = 0.78
    min_keypoint_confidence: float = 0.25
    history_size: int = 30

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "history_size":
                if not isinstance(value, int) or value < 3:
                    raise ValueError("history_size must be an integer >= 3")
            elif not _finite_number(value) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """Pose features for one person.

    ``bbox`` is ``(x1, y1, x2, y2)``. ``torso_angle_degrees`` is the
    absolute angle from vertical (zero=upright, 90=horizontal).
    """

    track_id: TrackId
    timestamp: float
    bbox: Sequence[float] | None
    torso_angle_degrees: float | None = None
    keypoint_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class FallResult:
    track_id: TrackId
    state: FallState
    previous_state: FallState
    changed: bool
    fall_detected: bool
    score: float
    reasons: tuple[str, ...]
    metrics: Mapping[str, float | bool | None]

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "changed": self.changed,
            "fall_detected": self.fall_detected,
            "score": round(self.score, 3),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class _Track:
    state: FallState
    first_seen: float
    last_seen: float
    previous_center_y: float | None = None
    previous_center_timestamp: float | None = None
    upright_since: float | None = None
    horizontal_since: float | None = None
    recovery_since: float | None = None
    drop_until: float = -math.inf
    cooldown_until: float = -math.inf
    standing_ready: bool = False
    standing_heights: deque[float] = field(default_factory=deque)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_bbox(bbox: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if bbox is None or isinstance(bbox, (str, bytes)):
        return None
    try:
        values = tuple(bbox)
    except TypeError:
        return None
    if len(values) != 4 or not all(_finite_number(value) for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class FallDetector:
    """Estimate temporal fall states for multiple person tracks."""

    def __init__(self, thresholds: FallThresholds | None = None) -> None:
        self.thresholds = thresholds or FallThresholds()
        self._tracks: dict[TrackId, _Track] = {}

    @property
    def active_track_ids(self) -> tuple[TrackId, ...]:
        return tuple(self._tracks)

    def cleanup(self, now: float) -> tuple[TrackId, ...]:
        if not _finite_number(now):
            raise ValueError("now must be a finite timestamp")
        timestamp = float(now)
        expired = tuple(
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.thresholds.track_ttl_seconds
        )
        for track_id in expired:
            del self._tracks[track_id]
        return expired

    def update(self, observation: PoseObservation) -> FallResult:
        if observation.track_id is None:
            raise ValueError("track_id must not be None")
        if not _finite_number(observation.timestamp):
            raise ValueError("timestamp must be finite")
        timestamp = float(observation.timestamp)
        self.cleanup(timestamp)

        track = self._tracks.get(observation.track_id)
        if track is None:
            track = _Track(
                state=FallState.NORMAL,
                first_seen=timestamp,
                last_seen=timestamp,
                standing_heights=deque(maxlen=self.thresholds.history_size),
            )
            self._tracks[observation.track_id] = track
        elif timestamp < track.last_seen:
            raise ValueError("timestamps for a track must be monotonic")

        previous_state = track.state
        track.last_seen = timestamp
        bbox = _valid_bbox(observation.bbox)
        if bbox is None:
            return self._result(
                observation.track_id,
                track,
                previous_state,
                0.0,
                ("invalid_bbox",),
                {
                    "standing_ready": track.standing_ready,
                    "horizontal": None,
                    "upright": None,
                    "drop_recent": timestamp <= track.drop_until,
                },
            )

        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        center_y = (y1 + y2) / 2.0
        aspect_ratio = width / height
        torso_angle = self._usable_torso_angle(observation)
        upright = self._is_upright(aspect_ratio, torso_angle)
        standing_height = median(track.standing_heights) if track.standing_heights else None
        vertical_speed = self._vertical_speed(track, center_y, timestamp, standing_height or height)
        if (
            vertical_speed is not None
            and vertical_speed >= self.thresholds.drop_speed_heights_per_second
        ):
            track.drop_until = max(
                track.drop_until, timestamp + self.thresholds.drop_memory_seconds
            )

        if upright:
            if track.upright_since is None:
                track.upright_since = timestamp
            track.standing_heights.append(height)
            standing_height = median(track.standing_heights)
            if timestamp - track.upright_since >= self.thresholds.min_standing_seconds:
                track.standing_ready = True
        else:
            track.upright_since = None

        horizontal_score = self._horizontal_score(
            aspect_ratio=aspect_ratio,
            torso_angle=torso_angle,
            height=height,
            standing_height=standing_height,
        )
        horizontal = horizontal_score >= 0.65
        if horizontal:
            if track.horizontal_since is None:
                track.horizontal_since = timestamp
        else:
            track.horizontal_since = None

        drop_recent = timestamp <= track.drop_until
        horizontal_seconds = (
            timestamp - track.horizontal_since if track.horizontal_since is not None else 0.0
        )
        reasons: list[str] = []
        if not track.standing_ready:
            reasons.append("awaiting_standing_history")
        if drop_recent:
            reasons.append("rapid_downward_motion")
        if horizontal:
            reasons.append("horizontal_posture")
        if standing_height and height / standing_height <= self.thresholds.collapsed_height_ratio:
            reasons.append("reduced_person_height")

        eligible = track.standing_ready and timestamp >= track.cooldown_until
        if track.state is FallState.NORMAL:
            if eligible and drop_recent and horizontal:
                track.state = FallState.SUSPECTED
                reasons.append("fall_sequence_started")
        elif track.state is FallState.SUSPECTED:
            if not horizontal:
                track.state = FallState.NORMAL
                track.horizontal_since = None
                reasons.append("suspicion_cleared")
            elif drop_recent and horizontal_seconds >= self.thresholds.fallen_hold_seconds:
                track.state = FallState.FALLEN
                reasons.append("fall_confirmed")
        elif track.state is FallState.FALLEN:
            if upright:
                track.state = FallState.RECOVERING
                track.recovery_since = timestamp
                reasons.append("upright_recovery_started")
        elif track.state is FallState.RECOVERING:
            if horizontal:
                track.state = FallState.FALLEN
                track.recovery_since = None
                reasons.append("returned_to_fallen_posture")
            elif not upright:
                track.recovery_since = None
            else:
                if track.recovery_since is None:
                    track.recovery_since = timestamp
                if timestamp - track.recovery_since >= self.thresholds.recovery_hold_seconds:
                    track.state = FallState.NORMAL
                    track.recovery_since = None
                    track.cooldown_until = timestamp + self.thresholds.cooldown_seconds
                    track.drop_until = -math.inf
                    reasons.append("recovery_confirmed")

        track.previous_center_y = center_y
        track.previous_center_timestamp = timestamp
        state_weight = {
            FallState.NORMAL: 0.0,
            FallState.SUSPECTED: 0.18,
            FallState.FALLEN: 0.35,
            FallState.RECOVERING: 0.12,
        }[track.state]
        score = min(
            1.0,
            horizontal_score * 0.55 + (0.25 if drop_recent else 0.0) + state_weight,
        )
        metrics: dict[str, float | bool | None] = {
            "aspect_ratio": round(aspect_ratio, 4),
            "torso_angle_degrees": round(torso_angle, 3) if torso_angle is not None else None,
            "vertical_speed_heights_per_second": (
                round(vertical_speed, 4) if vertical_speed is not None else None
            ),
            "horizontal_seconds": round(horizontal_seconds, 3),
            "standing_ready": track.standing_ready,
            "horizontal": horizontal,
            "upright": upright,
            "drop_recent": drop_recent,
        }
        return self._result(
            observation.track_id,
            track,
            previous_state,
            score,
            tuple(dict.fromkeys(reasons)),
            metrics,
        )

    def update_many(self, observations: Iterable[PoseObservation]) -> tuple[FallResult, ...]:
        return tuple(self.update(observation) for observation in observations)

    def _usable_torso_angle(self, observation: PoseObservation) -> float | None:
        angle = observation.torso_angle_degrees
        confidence = observation.keypoint_confidence
        if not _finite_number(angle):
            return None
        if confidence is not None and (
            not _finite_number(confidence)
            or float(confidence) < self.thresholds.min_keypoint_confidence
        ):
            return None
        return max(0.0, min(90.0, abs(float(angle))))

    def _is_upright(self, aspect_ratio: float, torso_angle: float | None) -> bool:
        if aspect_ratio > self.thresholds.upright_aspect_ratio:
            return False
        return torso_angle is None or torso_angle <= self.thresholds.upright_torso_angle_degrees

    def _horizontal_score(
        self,
        *,
        aspect_ratio: float,
        torso_angle: float | None,
        height: float,
        standing_height: float | None,
    ) -> float:
        score = 0.0
        if aspect_ratio >= self.thresholds.strong_horizontal_aspect_ratio:
            score += 0.55
        elif aspect_ratio >= self.thresholds.horizontal_aspect_ratio:
            score += 0.4
        if torso_angle is not None:
            if torso_angle >= self.thresholds.horizontal_torso_angle_degrees:
                score += 0.45
            elif torso_angle >= self.thresholds.upright_torso_angle_degrees:
                score += 0.2
        if standing_height and height / standing_height <= self.thresholds.collapsed_height_ratio:
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def _vertical_speed(
        track: _Track,
        center_y: float,
        timestamp: float,
        reference_height: float,
    ) -> float | None:
        if track.previous_center_y is None or track.previous_center_timestamp is None:
            return None
        elapsed = timestamp - track.previous_center_timestamp
        if elapsed <= 0 or reference_height <= 0:
            return None
        return (center_y - track.previous_center_y) / reference_height / elapsed

    @staticmethod
    def _result(
        track_id: TrackId,
        track: _Track,
        previous_state: FallState,
        score: float,
        reasons: tuple[str, ...],
        metrics: Mapping[str, float | bool | None],
    ) -> FallResult:
        return FallResult(
            track_id=track_id,
            state=track.state,
            previous_state=previous_state,
            changed=track.state is not previous_state,
            fall_detected=track.state is FallState.FALLEN,
            score=max(0.0, min(1.0, score)),
            reasons=reasons,
            metrics=metrics,
        )
