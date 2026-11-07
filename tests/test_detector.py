from __future__ import annotations

import math

from app.detector import PersonTracker, RawPerson, SessionAnalyzer, _torso_features


def person(
    bbox: tuple[float, float, float, float],
    *,
    angle: float = 5.0,
) -> RawPerson:
    return RawPerson(
        bbox=bbox,
        confidence=0.9,
        keypoints=[],
        torso_angle_degrees=angle,
        keypoint_confidence=0.9,
    )


def test_tracker_keeps_identity_for_nearby_box() -> None:
    tracker = PersonTracker()
    first = tracker.update([person((10, 10, 50, 110))], 0.0)
    second = tracker.update([person((14, 12, 54, 112))], 0.1)
    assert first[0][0] == second[0][0]


def test_tracker_expires_identity() -> None:
    tracker = PersonTracker(ttl_seconds=1.0)
    first_id = tracker.update([person((10, 10, 50, 110))], 0.0)[0][0]
    second_id = tracker.update([person((10, 10, 50, 110))], 2.0)[0][0]
    assert second_id != first_id


def test_tracker_does_not_mix_two_people() -> None:
    tracker = PersonTracker()
    first = tracker.update([person((10, 10, 50, 110)), person((200, 10, 240, 110))], 0.0)
    second = tracker.update([person((204, 12, 244, 112)), person((14, 12, 54, 112))], 0.1)
    assert first[0][0] == second[1][0]
    assert first[1][0] == second[0][0]


def test_torso_angle_is_measured_from_vertical() -> None:
    points = [(0.0, 0.0, 0.0)] * 17
    points[5] = (40.0, 20.0, 0.9)
    points[6] = (60.0, 20.0, 0.9)
    points[11] = (40.0, 80.0, 0.9)
    points[12] = (60.0, 80.0, 0.9)
    angle, confidence = _torso_features(points)
    assert angle == 0.0
    assert math.isclose(confidence or 0.0, 0.9)


def test_low_confidence_torso_is_ignored() -> None:
    points = [(0.0, 0.0, 0.0)] * 17
    for index in (5, 6, 11, 12):
        points[index] = (10.0, float(index), 0.1)
    angle, confidence = _torso_features(points)
    assert angle is None
    assert math.isclose(confidence or 0.0, 0.1)


def test_session_analyzer_returns_serializable_person() -> None:
    analyzer = SessionAnalyzer()
    result = analyzer.analyze([person((10, 10, 50, 110))], timestamp=0.0)[0].as_dict()
    assert result["track_id"] == 1
    assert result["state"] == "normal"
    assert result["bbox"] == [10, 10, 50, 110]
